"""JODI-Oil World Database: ülke bazlı aylık ham petrol ithalatı (KBD)."""
import io

import pandas as pd
from bs4 import BeautifulSoup

from ..config import FILES, START_DATE
from ..logger import get_logger
from .common import atomic_write_csv, create_retry_session

log = get_logger(__name__)

JODI_PAGE = "https://www.jodidata.org/oil/database/data-downloads.aspx"

# CN sadece GACC toplamlarını doğrulamak için tutulur, global toplamda kullanılmaz.
TARGET_ISO = {
    "JP", "KR", "TW", "CA", "AU", "NZ", "MX", "GB", "DE", "FR", "IT", "ES",
    "GR", "NO", "IN", "AT", "BE", "BG", "CH", "CZ", "HU", "IE", "NL", "PL", "RO", "SE", "CN",
}

# Önceki sürümde contains('CRUDE') + contains('KB') filtresi CRUDEOIL/TOTCRUDE/OTHERCRUDE
# ürünlerini ve KBD/KBBL birimlerini birlikte topluyordu (çoklu sayım). Artık tam eşleşme kullanılıyor.
PRODUCT = "CRUDEOIL"
FLOW = "TOTIMPSB"
UNIT = "KBD"

# JODI son aylarda revizyon yapar: son N yıl her çalıştırmada yeniden indirilir.
REFRESH_YEARS = 2

COLUMN_MAP = {
    "TIME": "TIME_PERIOD",
    "VAL": "OBS_VALUE",
    "VALUE": "OBS_VALUE",
    "PRODUCT": "ENERGY_PRODUCT",
    "FLOW": "FLOW_BREAKDOWN",
    "UNIT": "UNIT_MEASURE",
}
REQUIRED_COLUMNS = {"TIME_PERIOD", "REF_AREA", "ENERGY_PRODUCT", "FLOW_BREAKDOWN", "UNIT_MEASURE", "OBS_VALUE"}


def filter_crude_imports(df: pd.DataFrame) -> pd.DataFrame:
    """Ham JODI CSV'sinden hedef ülkelerin ham petrol ithalatını (KBD) uzun formatta döndürür."""
    df = df.rename(columns=lambda c: str(c).upper().strip()).rename(columns=COLUMN_MAP)
    if missing := REQUIRED_COLUMNS - set(df.columns):
        raise ValueError(f"JODI CSV eksik sütunlar içeriyor: {sorted(missing)}")

    for col in ("ENERGY_PRODUCT", "FLOW_BREAKDOWN", "UNIT_MEASURE", "REF_AREA"):
        df[col] = df[col].astype(str).str.upper().str.strip()

    mask = (
        (df["ENERGY_PRODUCT"] == PRODUCT)
        & (df["FLOW_BREAKDOWN"] == FLOW)
        & (df["UNIT_MEASURE"] == UNIT)
        & df["REF_AREA"].isin(TARGET_ISO)
    )
    out = df.loc[mask, ["TIME_PERIOD", "REF_AREA", "OBS_VALUE"]].copy()
    if out.empty:
        log.warning(
            "JODI filtresi boş döndü. Mevcut değerler -> ürün: %s | akış: %s | birim: %s",
            sorted(df["ENERGY_PRODUCT"].unique())[:10],
            sorted(df["FLOW_BREAKDOWN"].unique())[:10],
            sorted(df["UNIT_MEASURE"].unique()),
        )
    out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")  # boş/gizli -> NaN (0 değil!)
    return out.dropna(subset=["TIME_PERIOD"])


def _find_year_links(html: str, start_year: int, end_year: int) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    links = {}
    for a in soup.find_all("a", href=True):
        text, href = a.text.strip(), a["href"]
        if not (text.isdigit() and start_year <= int(text) <= end_year and ".csv" in href.lower()):
            continue
        parent = a.find_parent()
        parent_text = parent.get_text().lower() if parent else ""
        if "primary" in parent_text or "primary" in href.lower() or "world" in href.lower():
            links[int(text)] = "https://www.jodidata.org" + href if href.startswith("/") else href
    return links


def update() -> None:
    master = FILES["jodi"]
    log.info("JODI veri çekme başlıyor...")

    df_old = pd.read_csv(master, index_col=0, parse_dates=True) if master.exists() else pd.DataFrame()
    existing_years = set(df_old.index.year) if not df_old.empty else set()

    current_year = pd.Timestamp.now().year
    session = create_retry_session()
    res = session.get(JODI_PAGE, timeout=30)
    res.raise_for_status()
    links = _find_year_links(res.text, pd.Timestamp(START_DATE).year, current_year)
    if not links:
        raise RuntimeError("JODI sayfasında yıl bazlı CSV linki bulunamadı (site yapısı değişmiş olabilir).")

    frames = []
    refreshed_years = set()
    for year in sorted(links):
        if year in existing_years and year < current_year - REFRESH_YEARS + 1:
            continue
        log.info("JODI %s indiriliyor...", year)
        r = session.get(links[year], timeout=120)
        r.raise_for_status()
        parsed = filter_crude_imports(pd.read_csv(io.BytesIO(r.content), low_memory=False))
        frames.append(parsed)
        refreshed_years.add(year)

    if not frames:
        log.info("JODI: yeni veri yok, master güncel.")
        return

    all_long = pd.concat(frames, ignore_index=True)
    if all_long.empty:
        log.warning("JODI: indirilen dosyalarda eşleşen kayıt bulunamadı.")
        return

    df_new = all_long.pivot_table(
        index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="mean", dropna=False
    )
    # Yenilenen yılları temiz biçimde değiştir: revizyonda gizlenen hücre eski değerden kalmasın (ffill YOK).
    if not df_old.empty:
        df_kept = df_old.loc[~df_old.index.year.isin(refreshed_years)]
        df_final = pd.concat([df_kept, df_new], axis=0)
        df_final = df_final[~df_final.index.duplicated(keep="last")]
    else:
        df_final = df_new

    df_final.index = pd.to_datetime(df_final.index)
    df_final = df_final.sort_index()
    df_final.index.name = "TIME_PERIOD"
    atomic_write_csv(df_final, master)
    log.info("JODI master güncellendi: %s (%d ay, %d ülke)", master.name, len(df_final), df_final.shape[1])
