"""DGCIS (Hindistan): Rusya'dan ham petrol ithalatı, liman bazlı aylık."""
import re
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from ..config import END_DATE, FILES
from ..logger import get_logger

log = get_logger(__name__)

DGCIS_URL = "https://ftddp.dgciskol.gov.in/dgcis/principalcommditysearch.html#!/freeuser"
DGCIS_START = "2021-04-01"   # Portalın serbest erişimde sunduğu ilk ay
COLUMNS = ["Period", "No", "PC_Code", "Commodity", "Country", "Port", "Unit", "Qty", "Value_INR", "Value_USD"]


def target_periods(now: Optional[pd.Timestamp] = None) -> List[str]:
    """DGCIS_START'tan END_DATE'e (en geç tamamlanmış son ay) kadar 'Mon-YYYY' listesi."""
    ref = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    last_completed = ref.to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
    end_cap = pd.Timestamp(END_DATE).to_period("M").to_timestamp()
    last = min(end_cap, last_completed)
    if last < pd.Timestamp(DGCIS_START):
        return []
    return [d.strftime("%b-%Y") for d in pd.date_range(DGCIS_START, last, freq="MS")]


def _append_rows(rows: List[dict]) -> None:
    """Her ay sonrası diske ekler: tarayıcı çökse bile çekilen aylar kaybolmaz."""
    path = FILES["dgcis"]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, mode="a", header=write_header, index=False)


def _scroll_center(driver, el) -> None:
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", el)
    except Exception:  # noqa: BLE001
        pass


def _wait_select_value(driver, wait, sel_id: str, value: str) -> None:
    import sys

    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    intercept_exc = getattr(sys.modules["selenium.common.exceptions"], "ElementClickInterceptedException", RuntimeError)

    def _ready(d) -> bool:
        try:
            opts = Select(d.find_element(By.ID, sel_id)).options
            return any(o.get_attribute("value") == value and "{{" not in o.text for o in opts)
        except StaleElementReferenceException:
            return False

    wait.until(_ready)
    el = driver.find_element(By.ID, sel_id)
    _scroll_center(driver, el)
    try:
        Select(el).select_by_value(value)
    except intercept_exc:
        # Sayfa altındaki marquee bandı option.click()'i engellerse doğrudan JS + Angular ile seç.
        driver.execute_script(
            """
            const s = arguments[0], val = arguments[1];
            for (let i = 0; i < s.options.length; i++) {
                if (s.options[i].value === val) {
                    s.selectedIndex = i;
                    s.options[i].selected = true;
                    break;
                }
            }
            s.dispatchEvent(new Event('change', {bubbles: true}));
            if (window.angular) {
                try { window.angular.element(s).triggerHandler('change'); } catch (e) {}
            }
            """,
            el,
            value,
        )


def _wait_select_index(driver, wait, sel_id: str, index: int = 1) -> None:
    import sys

    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    intercept_exc = getattr(sys.modules["selenium.common.exceptions"], "ElementClickInterceptedException", RuntimeError)

    def _ready(d) -> bool:
        try:
            opts = Select(d.find_element(By.ID, sel_id)).options
            return len(opts) > index and "{{" not in opts[index].text
        except StaleElementReferenceException:
            return False

    wait.until(_ready)
    el = driver.find_element(By.ID, sel_id)
    _scroll_center(driver, el)
    try:
        Select(el).select_by_index(index)
    except intercept_exc:
        # Sayfa altındaki marquee bandı option.click()'i engellerse doğrudan JS + Angular ile seç.
        driver.execute_script(
            """
            const s = arguments[0], idx = arguments[1];
            if (idx >= 0 && idx < s.options.length) {
                s.selectedIndex = idx;
                s.options[idx].selected = true;
            }
            s.dispatchEvent(new Event('change', {bubbles: true}));
            if (window.angular) {
                try { window.angular.element(s).triggerHandler('change'); } catch (e) {}
            }
            """,
            el,
            index,
        )


def _consume_alert(driver) -> Optional[str]:
    """Açık JS uyarısı (alert) varsa metnini alır ve kabul ederek kapatır."""
    import sys

    no_alert_exc = getattr(sys.modules["selenium.common.exceptions"], "NoAlertPresentException", Exception)
    try:
        alert = driver.switch_to.alert
        txt = (getattr(alert, "text", None) or "").strip()
        alert.accept()
        return txt
    except no_alert_exc:
        return None
    except Exception:  # noqa: BLE001
        return None


def _read_filter_options(driver, list_id: str) -> Optional[List[str]]:
    vals = driver.execute_script(
        "const l = document.getElementById(arguments[0]); return l ? Array.from(l.options).map(o => o.value) : null;",
        list_id,
    )
    if vals is None:
        return None
    return [str(v) for v in vals]


def _clear_filter_list(driver, click, list_id: str) -> None:
    """Hedef çoklu seçim kutusundaki mevcut ('ALL' vb.) öğeleri temizler."""
    from selenium.webdriver.common.by import By

    driver.execute_script(
        """
        const l = document.getElementById(arguments[0]);
        if (!l) return;
        for (let i = 0; i < l.options.length; i++) { l.options[i].selected = true; }
        l.dispatchEvent(new Event('change', {bubbles: true}));
        """,
        list_id,
    )
    del_btns = driver.find_elements(
        By.XPATH,
        f"//*[@onclick and contains(@onclick, 'adOption') and contains(@onclick, \"'{list_id}'\") and not(contains(@onclick, 'addOption'))]",
    )
    for btn in del_btns:
        try:
            click(btn)
            _consume_alert(driver)
        except Exception:  # noqa: BLE001
            _consume_alert(driver)

    driver.execute_script(
        """
        const l = document.getElementById(arguments[0]);
        if (l && l.options.length > 0) {
            l.options.length = 0;
            l.dispatchEvent(new Event('change', {bubbles: true}));
            if (window.angular) {
                try { window.angular.element(l).triggerHandler('change'); } catch (e) {}
            }
        }
        """,
        list_id,
    )


def _ensure_filter_added(
    driver,
    wait,
    click,
    sel_id: str,
    value: str,
    list_id: str,
    attempts: int = 3,
) -> None:
    """Seçim kutusunu hedef listeye ekler; önceden dolu 'ALL' / alert hatalarını temizleyip tekrarlar."""
    import sys

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC

    alert_exc = getattr(sys.modules["selenium.common.exceptions"], "UnexpectedAlertPresentException", RuntimeError)
    add_xpath = f"//input[@onclick=\"adOption.addOption('{list_id}','{sel_id}');\"]"
    last_err: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        _consume_alert(driver)
        try:
            existing = _read_filter_options(driver, list_id)
            if existing == [value]:
                return
            if existing:
                log.info("DGCIS %s içinde önceki değerler temizleniyor: %s", list_id, existing)
                _clear_filter_list(driver, click, list_id)

            _wait_select_value(driver, wait, sel_id, value)
            click(wait.until(EC.element_to_be_clickable((By.XPATH, add_xpath))))

            after = _read_filter_options(driver, list_id)
            if after is None or value in after:
                return
            log.warning("DGCIS %s=%s eklendikten sonra %s boş/farklı (%s); yeniden deneniyor.", sel_id, value, list_id, after)
            _clear_filter_list(driver, click, list_id)
        except alert_exc as exc:
            last_err = exc
            alert_txt = _consume_alert(driver) or getattr(exc, "alert_text", None) or str(exc)
            cur = None
            try:
                cur = _read_filter_options(driver, list_id)
            except Exception:  # noqa: BLE001
                pass
            if cur == [value]:
                log.info("DGCIS %s (%s=%s) uyarısı kapatıldı, değer listede mevcut: %s", list_id, sel_id, value, alert_txt)
                return
            log.warning(
                "DGCIS %s=%s -> %s uyarısı yakalandı (deneme %d/%d, mevcut=%s): %s",
                sel_id,
                value,
                list_id,
                attempt,
                attempts,
                cur,
                alert_txt,
            )
            try:
                _clear_filter_list(driver, click, list_id)
            except Exception:  # noqa: BLE001
                _consume_alert(driver)

    if last_err is not None:
        raise last_err


_RESULT_SNAPSHOT_JS = """
const res = document.getElementById('result');
if (!res) return {state: 'missing', rows: [], sig: '', pending: 0, html: ''};
const html = res.innerHTML || '';
const txt = (res.innerText || res.textContent || '').replace(/\\s+/g, ' ').trim();
const low = txt.toLowerCase();
const hasTpl = html.includes('{{') || txt.includes('{{');

let pending = 0;
try {
    if (window.angular) {
        const el = window.angular.element(res).injector() ? window.angular.element(res) : window.angular.element(document.body);
        const inj = el.injector && el.injector();
        if (inj) pending += inj.get('$http').pendingRequests.length;
    }
} catch (e) {}
try {
    if (window.jQuery && typeof window.jQuery.active === 'number') {
        pending += window.jQuery.active;
    }
} catch (e) {}

const isVisible = (el) => {
    let cur = el;
    while (cur && cur.nodeType === 1) {
        const st = window.getComputedStyle(cur);
        if (st.display === 'none' || st.visibility === 'hidden') return false;
        cur = cur.parentElement;
    }
    return true;
};

const trs = Array.from(res.querySelectorAll('table.table-bordered tbody tr'));
const rows = [];
for (const tr of trs) {
    if (!isVisible(tr)) continue;
    const tds = Array.from(tr.querySelectorAll('td'));
    if (tds.length < 7) continue;
    const cells = tds.map(td => (td.innerText || td.textContent || '').trim());
    if (cells.some(c => c.includes('{{'))) continue;
    if (!cells.some(c => c.length > 0)) continue;
    rows.push(cells.slice(0, 9));
}

if (rows.length > 0) {
    const sig = rows.map(r => r.join('|')).join('\\n');
    return {state: 'rows', rows: rows, sig: sig, pending: pending, html: html};
}
const isZero = !hasTpl && (
    /no\\s+data|no\\s+record|no\\s+matching|not\\s+found/.test(low) ||
    /\\btotal\\s+0\\s+records\\b/.test(low)
);
if (isZero) {
    return {state: 'empty', rows: [], sig: 'EMPTY:' + low, pending: pending, html: html};
}
return {state: 'pending', rows: [], sig: 'PENDING:' + txt.slice(0, 120), pending: pending, html: html};
"""


def _snapshot_result(driver) -> Optional[dict]:
    if not hasattr(driver, "execute_script"):
        return None
    snap = driver.execute_script(_RESULT_SNAPSHOT_JS)
    return snap if isinstance(snap, dict) else None


def _wait_result_refresh(
    driver,
    wait,
    old_tbody,
    old_html: str,
    old_sig: Optional[str] = None,
    out_rows: Optional[List[List[str]]] = None,
) -> bool:
    """Önceki ayın DOM tablosunun yenilenmesini bekler; veri satırı varsa True (ve out_rows dolu), boş sonuç onaylandıysa False döner."""
    import time

    from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    # 1) Atomik JS destekli yol (gerçek tarayıcı + tam sahte sürücüler): TOCTOU yarışını ve başlık/loading <tr> tuzaklarını önler.
    if hasattr(driver, "execute_script"):
        tracker = {
            "saw_transition": not bool(old_sig),
            "started_at": time.monotonic(),
            "has_rows": False,
            "rows": [],
            "stable_sig": None,
            "stable_hits": 0,
        }

        def _is_old_stale(d) -> bool:
            if old_tbody is None:
                return False
            try:
                return bool(EC.staleness_of(old_tbody)(d))
            except StaleElementReferenceException:
                return True

        def _ready_atomic(d) -> bool:
            if _is_old_stale(d):
                tracker["saw_transition"] = True

            snap = _snapshot_result(d)
            if snap is None:
                return False

            state = snap.get("state", "pending")
            sig = str(snap.get("sig") or "")
            pending = int(snap.get("pending") or 0)
            elapsed = time.monotonic() - tracker["started_at"]

            if pending > 0 or state in ("pending", "missing"):
                tracker["saw_transition"] = True
                tracker["stable_sig"] = None
                tracker["stable_hits"] = 0
                return False

            if sig and old_sig and sig != old_sig:
                tracker["saw_transition"] = True

            if state == "rows":
                # Önceki ayın satırları henüz ekrandan temizlenmemişse bekle (12 sn sonrası aynı veri tekrarı kabul edilir).
                if not tracker["saw_transition"] and old_sig and sig == old_sig and elapsed < 12.0:
                    tracker["stable_sig"] = None
                    tracker["stable_hits"] = 0
                    return False

                if tracker["stable_sig"] == sig:
                    tracker["stable_hits"] += 1
                else:
                    tracker["stable_sig"] = sig
                    tracker["stable_hits"] = 1
                if tracker["stable_hits"] < 2:
                    return False

                tracker["has_rows"] = True
                tracker["rows"] = [list(r) for r in (snap.get("rows") or [])]
                return True

            if state == "empty":
                # Ardışık iki ay da boşsa HTML değişmeyebilir; 2.5 sn bekleme + ardışık 2 kararlı kontrol ile geçici 0 flash'ını ele.
                if not tracker["saw_transition"] and elapsed < 2.5:
                    tracker["stable_sig"] = None
                    tracker["stable_hits"] = 0
                    return False

                if tracker["stable_sig"] == sig:
                    tracker["stable_hits"] += 1
                else:
                    tracker["stable_sig"] = sig
                    tracker["stable_hits"] = 1
                if tracker["stable_hits"] < 3:
                    return False

                tracker["has_rows"] = False
                tracker["rows"] = []
                return True

            return False

        wait.until(_ready_atomic)
        if out_rows is not None:
            out_rows.clear()
            out_rows.extend(tracker["rows"])
        return tracker["has_rows"]

    # 2) execute_script olmayan sade mock sürücüler için geriye dönük yedek yol
    def _changed(d) -> bool:
        if old_tbody is not None:
            try:
                if EC.staleness_of(old_tbody)(d):
                    return True
            except StaleElementReferenceException:
                return True
        try:
            res = d.find_element(By.ID, "result")
            return res.get_attribute("innerHTML") != old_html
        except StaleElementReferenceException:
            return True

    try:
        WebDriverWait(driver, 15).until(_changed)
    except TimeoutException:
        pass

    state = {"has_rows": False}

    def _ready(d) -> bool:
        try:
            res_el = d.find_element(By.ID, "result")
            rows = d.find_elements(By.XPATH, "//div[@id='result']//table[contains(@class, 'table-bordered')]//tbody//tr")
            if any(r.is_displayed() and r.text.strip() and "{{" not in r.text for r in rows):
                state["has_rows"] = True
                return True
            txt = res_el.text.lower()
            if "{{" not in txt and (
                any(k in txt for k in ("no data", "no record", "no matching", "not found"))
                or re.search(r"\btotal\s+0\s+records\b", txt) is not None
            ):
                state["has_rows"] = False
                return True
            return False
        except StaleElementReferenceException:
            return False

    wait.until(_ready)
    return state["has_rows"]


def _set_period_field(driver, wait, field_id: str, period: str) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC

    el = wait.until(EC.element_to_be_clickable((By.ID, field_id)))
    _scroll_center(driver, el)
    el.clear()
    el.send_keys(period)
    if (el.get_attribute("value") or "").strip() != period:
        driver.execute_script(
            """
            const f = arguments[0], val = arguments[1];
            f.value = val;
            f.dispatchEvent(new Event('input', {bubbles: true}));
            f.dispatchEvent(new Event('change', {bubbles: true}));
            if (window.angular) {
                try { window.angular.element(f).triggerHandler('change'); } catch (e) {}
            }
            """,
            el,
            period,
        )


def update(headless: bool = True) -> None:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, UnexpectedAlertPresentException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    log.info("DGCIS (Hindistan) veri çekme başlıyor...")
    path = FILES["dgcis"]
    legacy = path.parent / "russia_oil_ports_2018_2026.csv"
    if not path.exists() and legacy.exists():
        log.info("Eski DGCIS dosyası bulundu, yeni ada taşınıyor: %s", legacy.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(legacy).reindex(columns=COLUMNS).to_csv(path, index=False)

    done = (
        set(pd.read_csv(path, usecols=["Period"])["Period"].dropna())
        if path.exists() and path.stat().st_size > 0
        else set()
    )
    todo = [p for p in target_periods() if p not in done]
    if not todo:
        log.info("DGCIS: tüm periyotlar mevcut.")
        return
    log.info("DGCIS: %d eksik ay çekilecek.", len(todo))

    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    click = lambda el: driver.execute_script("arguments[0].click();", el)  # noqa: E731

    try:
        driver.get(DGCIS_URL)
        wait = WebDriverWait(driver, 30)
        for field in ("datepicker1", "datepicker"):
            _set_period_field(driver, wait, field, todo[0])

        click(wait.until(EC.element_to_be_clickable((By.ID, "eximp2"))))
        click(wait.until(EC.element_to_be_clickable((By.ID, "currency3"))))
        _wait_select_value(driver, wait, "type", "10")
        _wait_select_index(driver, wait, "sorted", 1)
        for sel_id, value, list_id in (
            ("comm", "S5", "sel_list"),
            ("con", "15344", "sel_list1"),
            ("port", "A", "sel_list4"),
        ):
            _ensure_filter_added(driver, wait, click, sel_id, value, list_id)
        _consume_alert(driver)

        for period in todo:
            try:
                prev_snap = _snapshot_result(driver)
                if prev_snap is not None:
                    old_html = prev_snap.get("html", "")
                    old_sig = prev_snap.get("sig", "")
                else:
                    old_html = driver.find_element(By.ID, "result").get_attribute("innerHTML")
                    old_sig = ""
                old_tbodies = driver.find_elements(
                    By.XPATH, "//div[@id='result']//table[contains(@class, 'table-bordered')]//tbody"
                )
                old_tbody = old_tbodies[0] if old_tbodies else None

                for field in ("datepicker1", "datepicker"):
                    _set_period_field(driver, wait, field, period)
                click(wait.until(EC.element_to_be_clickable((By.ID, "select_all"))))

                extracted_cells: List[List[str]] = []
                has_rows = _wait_result_refresh(driver, wait, old_tbody, old_html, old_sig=old_sig, out_rows=extracted_cells)

                rows: List[dict] = []
                if has_rows and extracted_cells:
                    for cells in extracted_cells:
                        cols = (list(cells) + ["", ""])[:9]
                        rows.append(dict(zip(COLUMNS, [period] + cols)))
                elif has_rows:
                    soup = BeautifulSoup(driver.find_element(By.ID, "result").get_attribute("outerHTML"), "html.parser")
                    for tr in soup.select("tbody tr"):
                        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                        if len(cols) >= 7 and not any("{{" in c for c in cols):
                            cols = (cols + ["", ""])[:9]
                            rows.append(dict(zip(COLUMNS, [period] + cols)))

                if rows:
                    _append_rows(rows)
                elif not has_rows:
                    # Onaylı "0 kayıt / veri yok" ayı: 0 ton yazılır ki transform eksik sanıp ayı düşürmesin.
                    _append_rows(
                        [
                            {
                                "Period": period,
                                "No": "0",
                                "PC_Code": "S5",
                                "Commodity": "NO DATA / ZERO",
                                "Country": "RUSSIA",
                                "Port": "ALL",
                                "Unit": "TON",
                                "Qty": "0",
                                "Value_INR": "0",
                                "Value_USD": "0",
                            }
                        ]
                    )
                else:
                    log.warning("DGCIS %s: tablo göründü ama geçerli satır okunamadı; eksik bırakıldı.", period)
                log.info("DGCIS %s: %d kayıt", period, len(rows))
            except UnexpectedAlertPresentException as exc:
                alert_txt = _consume_alert(driver) or getattr(exc, "alert_text", None) or str(exc)
                log.warning("DGCIS %s uyarısı nedeniyle atlandı: %s", period, alert_txt)
            except TimeoutException:
                log.warning("DGCIS %s zaman aşımı, atlandı.", period)
    finally:
        driver.quit()
