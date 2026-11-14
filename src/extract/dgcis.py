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


def _wait_result_refresh(driver, wait, old_tbody, old_html: str) -> bool:
    """Önceki ayın DOM tablosunun yenilenmesini bekler; veri satırı varsa True, boş sonuç onaylandıysa False döner."""
    from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

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
        # Ardışık iki ay tamamen aynı "No data" metni üretirse içerik değişmeyebilir; alt kontrol devreye girer.
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
            el = wait.until(EC.element_to_be_clickable((By.ID, field)))
            el.clear()
            el.send_keys(todo[0])

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
                result_el = driver.find_element(By.ID, "result")
                old_html = result_el.get_attribute("innerHTML")
                old_tbodies = driver.find_elements(
                    By.XPATH, "//div[@id='result']//table[contains(@class, 'table-bordered')]//tbody"
                )
                old_tbody = old_tbodies[0] if old_tbodies else None

                for field in ("datepicker1", "datepicker"):
                    el = wait.until(EC.element_to_be_clickable((By.ID, field)))
                    _scroll_center(driver, el)
                    el.clear()
                    el.send_keys(period)
                click(wait.until(EC.element_to_be_clickable((By.ID, "select_all"))))
                has_rows = _wait_result_refresh(driver, wait, old_tbody, old_html)

                soup = BeautifulSoup(driver.find_element(By.ID, "result").get_attribute("outerHTML"), "html.parser")
                rows = []
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
                log.info("DGCIS %s: %d kayıt", period, len(rows))
            except UnexpectedAlertPresentException as exc:
                alert_txt = _consume_alert(driver) or getattr(exc, "alert_text", None) or str(exc)
                log.warning("DGCIS %s uyarısı nedeniyle atlandı: %s", period, alert_txt)
            except TimeoutException:
                log.warning("DGCIS %s zaman aşımı, atlandı.", period)
    finally:
        driver.quit()
