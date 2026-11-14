"""DGCIS (Hindistan): Rusya'dan ham petrol ithalatı, liman bazlı aylık."""
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


def _wait_select_value(driver, wait, sel_id: str, value: str) -> None:
    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    def _ready(d) -> bool:
        try:
            opts = Select(d.find_element(By.ID, sel_id)).options
            return any(o.get_attribute("value") == value and "{{" not in o.text for o in opts)
        except StaleElementReferenceException:
            return False

    wait.until(_ready)
    el = driver.find_element(By.ID, sel_id)
    Select(el).select_by_value(value)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}));", el)


def _wait_select_index(driver, wait, sel_id: str, index: int = 1) -> None:
    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    def _ready(d) -> bool:
        try:
            opts = Select(d.find_element(By.ID, sel_id)).options
            return len(opts) > index and "{{" not in opts[index].text
        except StaleElementReferenceException:
            return False

    wait.until(_ready)
    el = driver.find_element(By.ID, sel_id)
    Select(el).select_by_index(index)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}));", el)


def _wait_result_refresh(driver, wait, old_tbody, old_html: str) -> None:
    """Önceki ayın DOM tablosunun bayatlamasını (yenilenmesini) bekler."""
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

    def _ready(d) -> bool:
        try:
            res_el = d.find_element(By.ID, "result")
            if d.find_elements(By.XPATH, "//div[@id='result']//table[contains(@class, 'table-bordered')]//tbody//tr"):
                return True
            txt = res_el.text.lower()
            return any(k in txt for k in ("no data", "no record", "no matching", "not found"))
        except StaleElementReferenceException:
            return False

    wait.until(_ready)


def update(headless: bool = True) -> None:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
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
        wait.until(EC.element_to_be_clickable((By.ID, "datepicker1"))).send_keys(todo[0])
        wait.until(EC.element_to_be_clickable((By.ID, "datepicker"))).send_keys(todo[0])

        click(wait.until(EC.element_to_be_clickable((By.ID, "eximp2"))))
        click(wait.until(EC.element_to_be_clickable((By.ID, "currency3"))))
        _wait_select_value(driver, wait, "type", "10")
        _wait_select_index(driver, wait, "sorted", 1)
        for sel_id, value, list_id in (
            ("comm", "S5", "sel_list"),
            ("con", "15344", "sel_list1"),
            ("port", "A", "sel_list4"),
        ):
            _wait_select_value(driver, wait, sel_id, value)
            click(wait.until(EC.element_to_be_clickable((By.XPATH, f"//input[@onclick=\"adOption.addOption('{list_id}','{sel_id}');\"]"))))

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
                    el.clear()
                    el.send_keys(period)
                click(wait.until(EC.element_to_be_clickable((By.ID, "select_all"))))
                _wait_result_refresh(driver, wait, old_tbody, old_html)

                soup = BeautifulSoup(driver.find_element(By.ID, "result").get_attribute("outerHTML"), "html.parser")
                rows = []
                for tr in soup.select("tbody tr"):
                    cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cols) >= 7:
                        cols = (cols + ["", ""])[:9]
                        rows.append(dict(zip(COLUMNS, [period] + cols)))
                if rows:
                    _append_rows(rows)
                log.info("DGCIS %s: %d kayıt", period, len(rows))
            except TimeoutException:
                log.warning("DGCIS %s zaman aşımı, atlandı.", period)
    finally:
        driver.quit()
