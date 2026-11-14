"""GACC (Çin Gümrüğü): HS 27090000 ham petrol ithalatı, ülke kırılımlı aylık.

ÖNEMLİ: GACC sorgu sonucunu bir doğrulama (slider captcha) arkasında sunar. Bu modül captcha'yı
otomatik çözmez; görünür bir Firefox penceresi açar, formu doldurur ve doğrulamayı sizin
tamamlamanızı bekler. Doğrulama geçildiği anda indirme ve dosya adlandırma otomatik devam eder.
Geçmiş yıllar bir kez indirildikten sonra tekrar istenmez, yani pratikte yılda/ayda bir
kaç saniyelik bir etkileşim gerekir.

Alternatif (tamamen manuel): GACC'den indirdiğiniz CSV'leri data/raw/china/{YIL}_china.csv
adıyla koyun; transform katmanı bu klasörü doğrudan okur.
"""
import time
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from ..config import CHINA_RAW_DIR, END_DATE, START_DATE
from ..logger import get_logger

log = get_logger(__name__)

GACC_URL = "http://stats.customs.gov.cn/indexEn"
HS_CODE = "27090000"
HUMAN_TIMEOUT_S = 300
DOWNLOAD_TIMEOUT_S = 120
_PARTIAL_SUFFIXES = (".part", ".tmp", ".crdownload")

X_DOWNLOAD = "//*[contains(text(), 'Download')] | //a[contains(@class, 'download') or contains(., 'Download')]"
X_CONFIRM = (
    "//button[contains(.,'Confirm')] | //span[contains(text(),'Confirm')] | "
    "//a[contains(text(),'Confirm')] | //div[contains(@class, 'button') and contains(., 'Confirm')]"
)
X_ENQUIRY = (
    "//input[@type='button' and contains(@value,'Enquiry')] | //button[contains(.,'Enquiry')] | "
    "//a[contains(text(),'Enquiry')]"
)
X_BACK = "//*[contains(text(), 'Back')] | //a[contains(., 'Back')]"


# ---------------------------------------------------------
# Dosya kontrolleri
# ---------------------------------------------------------
def months_in_file(path: Path) -> Set[int]:
    """CSV'deki farklı ayları döndürür (satır sayısı değil: ülke kırılımı yüzünden yanıltıcıydı)."""
    for enc in ("utf-8", "utf-8-sig", "gbk", "utf-16", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            break
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    else:
        return set()

    df.columns = df.columns.astype(str).str.strip()
    if "Date of data" not in df.columns:
        return set()
    dates = pd.to_datetime(
        df["Date of data"].astype(str).str.replace(r"[-/]", "", regex=True).str[:6],
        format="%Y%m",
        errors="coerce",
    )
    return set(dates.dropna().dt.month)


def missing_years(
    years: Optional[List[int]] = None,
    now: Optional[pd.Timestamp] = None,
    end_date: Optional[str] = None,
) -> List[int]:
    ref_now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    end_ts = pd.Timestamp(END_DATE if end_date is None else end_date)
    start_year = pd.Timestamp(START_DATE).year
    cap_year = min(ref_now.year, end_ts.year)

    if years is None:
        if cap_year < start_year:
            return []
        years = list(range(start_year, cap_year + 1))

    out = []
    for y in years:
        if y > cap_year:
            continue
        f = CHINA_RAW_DIR / f"{y}_china.csv"
        end_cap_m = end_ts.month if y == end_ts.year else 12
        # GACC yayın gecikmesi (~1-2 ay) yalnızca içinde bulunulan takvim yılında uygulanır
        avail_m = max(0, ref_now.month - 2) if y == ref_now.year else 12
        expected = min(end_cap_m, avail_m)
        if expected == 0:
            continue
        if not f.exists() or len(months_in_file(f)) < expected:
            out.append(y)
    return out


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _wait_new_download(before: Set[Path], timeout: int) -> Optional[Path]:
    end = time.time() + timeout
    while time.time() < end:
        entries = set(CHINA_RAW_DIR.iterdir())
        new = [p for p in entries - before if p.suffix not in _PARTIAL_SUFFIXES and _safe_size(p) > 0]
        partials = [p for p in entries if p.suffix in _PARTIAL_SUFFIXES]
        if new and not partials:
            return max(new, key=lambda p: p.stat().st_mtime)
        time.sleep(1)
    return None


# ---------------------------------------------------------
# Selenium yardımcıları
# ---------------------------------------------------------
def _first_displayed(driver, xpath: str):
    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.by import By

    for e in driver.find_elements(By.XPATH, xpath):
        try:
            if e.is_displayed():
                return e
        except StaleElementReferenceException:
            continue
    return None


def _find_in_frames(driver, xpath: str):
    """Elemanı ana DOM'da, 1. ve 2. seviye iframe'lerde arar; bulunca o frame'de kalır."""
    from selenium.common.exceptions import NoSuchFrameException, StaleElementReferenceException
    from selenium.webdriver.common.by import By

    driver.switch_to.default_content()
    el = _first_displayed(driver, xpath)
    if el is not None:
        return el

    for i in range(len(driver.find_elements(By.TAG_NAME, "iframe"))):
        try:
            driver.switch_to.default_content()
            frames = driver.find_elements(By.TAG_NAME, "iframe")
            if i >= len(frames):
                break
            driver.switch_to.frame(frames[i])
            el = _first_displayed(driver, xpath)
            if el is not None:
                return el
            nested = driver.find_elements(By.TAG_NAME, "iframe")
            for j in range(len(nested)):
                try:
                    driver.switch_to.frame(nested[j])
                    el = _first_displayed(driver, xpath)
                    if el is not None:
                        return el
                except (NoSuchFrameException, StaleElementReferenceException):
                    pass
                driver.switch_to.parent_frame()
        except (NoSuchFrameException, StaleElementReferenceException):
            continue

    driver.switch_to.default_content()
    return None


def _wait_for(driver, xpath: str, timeout: float, poll: float = 1.0):
    end = time.time() + timeout
    while time.time() < end:
        el = _find_in_frames(driver, xpath)
        if el is not None:
            return el
        time.sleep(poll)
    return None


def wait_for_human_verification(driver, timeout: int = HUMAN_TIMEOUT_S) -> bool:
    """Kullanıcı tarayıcıda doğrulamayı tamamlayana (sonuç sayfası açılana) kadar bekler."""
    log.warning(">>> Tarayıcı penceresinde GACC doğrulamasını tamamlayın (en fazla %d sn bekleniyor) <<<", timeout)
    print("\a", end="", flush=True)  # terminal zili
    return _wait_for(driver, X_DOWNLOAD, timeout, poll=2.0) is not None


def _setup_driver():
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    options = Options()  # Görünür pencere: doğrulama kullanıcı tarafından yapılır
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.dir", str(CHINA_RAW_DIR))
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference(
        "browser.helperApps.neverAsk.saveToDisk",
        "application/csv,text/csv,application/vnd.ms-excel,"
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    return webdriver.Firefox(options=options)


def _js_select(driver, select_el, target: str) -> None:
    driver.execute_script(
        """
        const s = arguments[0], t = arguments[1];
        for (const o of s.options) {
            const v = o.value.replace(/^0+/, ''), x = o.text.trim().replace(/^0+/, '');
            if (o.value === t || v === t || o.text.trim() === t || x === t) {
                s.value = o.value;
                s.dispatchEvent(new Event('change', {bubbles: true}));
                break;
            }
        }""",
        select_el,
        target,
    )


def _fill_static_form(driver) -> None:
    """Para birimi, aylık kırılım, HS kodu ve partner seçimlerini yapar (yıl döngüsünden bağımsız)."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    fire = lambda el, ev: driver.execute_script(  # noqa: E731
        f"arguments[0].dispatchEvent(new Event('{ev}', {{bubbles: true}}));",
        el,
    )

    usd = _wait_for(driver, "//input[@type='radio' and @value='usd']", 40)
    if usd is None:
        raise RuntimeError("GACC formu yüklenemedi (USD seçeneği bulunamadı).")
    driver.execute_script("arguments[0].click();", usd)

    try:
        chk = driver.find_element(
            By.XPATH,
            "//input[@type='checkbox' and (contains(@id,'month') or contains(@name,'month'))] | "
            "//span[contains(text(),'By month')]/following-sibling::input",
        )
        if not chk.is_selected():
            driver.execute_script("arguments[0].click();", chk)
    except Exception as e:  # noqa: BLE001
        log.warning("GACC 'By month' seçilemedi: %s", e)

    for idx, (value, keyword) in enumerate((("1", "commodity"), ("2", "partner"))):
        selects = driver.find_elements(By.XPATH, "//*[contains(text(),'Select:')]/following::select")
        if len(selects) <= idx:
            continue
        sel = Select(selects[idx])
        try:
            sel.select_by_value(value)
        except Exception:  # noqa: BLE001
            for opt in sel.options:
                if keyword in opt.text.lower():
                    sel.select_by_visible_text(opt.text)
                    break
        fire(selects[idx], "change")
        time.sleep(0.8)

        if idx == 0:
            # GACC HS kodu input'u başlangıçta disabled/readonly olabilir: JS ile etkinleştirip yaz.
            inputs = driver.find_elements(By.XPATH, "//*[contains(text(),'Select:')]/following::input[@type='text']")
            if inputs:
                driver.execute_script(
                    """
                    arguments[0].removeAttribute('disabled');
                    arguments[0].removeAttribute('readonly');
                    arguments[0].value = arguments[1];
                    """,
                    inputs[0],
                    HS_CODE,
                )
                fire(inputs[0], "input")
                fire(inputs[0], "change")
        time.sleep(0.5)


def _set_period(driver, year: int, start_month: int, end_month: int) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    xp = "//*[contains(text(),'Period')]/following::select"
    sels = driver.find_elements(By.XPATH, xp)
    if len(sels) < 3:
        raise RuntimeError("GACC dönem seçicileri bulunamadı.")
    try:
        Select(sels[0]).select_by_visible_text(str(year))
    except Exception:  # noqa: BLE001
        Select(sels[0]).select_by_value(str(year))
    driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}));", sels[0])
    time.sleep(2)
    for i, m in ((1, start_month), (2, end_month)):
        sels = driver.find_elements(By.XPATH, xp)
        if len(sels) > i:
            _js_select(driver, sels[i], str(m))
        time.sleep(1)


# ---------------------------------------------------------
# Ana akış
# ---------------------------------------------------------
def update(years: Optional[List[int]] = None) -> None:
    CHINA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = missing_years(years)
    if not todo:
        log.info("GACC: tüm yıllar eksiksiz, tarayıcı açılmayacak.")
        return
    log.info("GACC: eksik yıllar %s. Görünür tarayıcı açılıyor...", todo)

    now = pd.Timestamp.now()
    end_ts = pd.Timestamp(END_DATE)
    driver = _setup_driver()
    try:
        driver.get(GACC_URL)
        _fill_static_form(driver)

        for i, year in enumerate(todo):
            end_cap_m = end_ts.month if year == end_ts.year else 12
            avail_m = max(1, now.month - 1) if year == now.year else 12
            end_month = min(end_cap_m, avail_m)
            log.info("GACC %d (1-%d) sorgulanıyor (%d/%d)", year, end_month, i + 1, len(todo))
            _wait_for(driver, "//*[contains(text(),'Period')]", 20)
            if i > 0:
                from selenium.webdriver.common.by import By

                hs_inputs = driver.find_elements(By.XPATH, "//*[contains(text(),'Select:')]/following::input[@type='text']")
                if not hs_inputs or hs_inputs[0].get_attribute("value") != HS_CODE:
                    _fill_static_form(driver)
            _set_period(driver, year, 1, end_month)

            enquiry = _wait_for(driver, X_ENQUIRY, 30)
            if enquiry is None:
                log.error("GACC Enquiry butonu bulunamadı, %d atlandı.", year)
                continue
            driver.execute_script("arguments[0].click();", enquiry)
            confirm = _wait_for(driver, X_CONFIRM, 10)
            if confirm is not None:
                driver.execute_script("arguments[0].click();", confirm)

            if not wait_for_human_verification(driver):
                log.error("GACC %d: doğrulama süresi doldu, atlandı.", year)
                continue

            dl_btn = _find_in_frames(driver, X_DOWNLOAD)
            if dl_btn is None:
                log.error("GACC %d: Download butonu bulunamadı, atlandı.", year)
                continue

            before = set(CHINA_RAW_DIR.iterdir())
            driver.execute_script("arguments[0].click();", dl_btn)
            confirm = _wait_for(driver, X_CONFIRM, 15)
            if confirm is not None:
                driver.execute_script("arguments[0].click();", confirm)

            downloaded = _wait_new_download(before, DOWNLOAD_TIMEOUT_S)
            if downloaded is None:
                log.error("GACC %d: indirme tamamlanmadı.", year)
            else:
                parsed_months = months_in_file(downloaded)
                target = CHINA_RAW_DIR / f"{year}_china.csv"
                if not parsed_months:
                    log.error(
                        "GACC %d: indirilen dosya (%s) geçerli 'Date of data' satırı içermiyor; "
                        "mevcut hedef korunuyor.",
                        year,
                        downloaded.name,
                    )
                else:
                    downloaded.replace(target)
                    log.info("GACC %d indirildi -> %s (%d ay)", year, target.name, len(parsed_months))

            if i < len(todo) - 1:
                back = _wait_for(driver, X_BACK, 20)
                if back is None:
                    log.error("GACC 'Back' bulunamadı, kalan yıllar bir sonraki çalıştırmaya kaldı.")
                    break
                driver.execute_script("arguments[0].click();", back)
                time.sleep(3)
    finally:
        driver.quit()
