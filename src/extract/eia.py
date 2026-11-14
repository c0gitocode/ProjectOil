"""EIA: ABD aylık ham petrol ithalatı (KBD)."""
import pandas as pd

from ..config import FILES, START_DATE, get_eia_api_key
from ..logger import get_logger
from .common import atomic_write_csv

log = get_logger(__name__)

# Aylık resmi seri (PSM). Haftalık seriden aylık ortalama almak hafta-sonu tarihleri yüzünden
# ay sınırlarında kayma yaratıyordu; aylık seri yoksa haftalığa düşülür.
MONTHLY_SERIES = "PET.MCRIMUS2.M"
WEEKLY_SERIES = "PET.WCRIMUS2.W"


def _extract_series_values(df: pd.DataFrame) -> pd.Series:
    """myeia API'sinin döndürdüğü DataFrame'den sayısal değer serisini güvenle çıkarır."""
    if df is None or df.empty:
        raise ValueError("EIA API boş veri döndürdü.")
    col = "value" if "value" in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"EIA '{col}' sütununda geçerli sayısal değer bulunamadı.")
    return s


def update() -> None:
    from myeia.api import API  # ağır bağımlılık: sadece gerektiğinde

    log.info("EIA (ABD) veri çekme başlıyor...")
    eia = API(get_eia_api_key())

    try:
        s = _extract_series_values(eia.get_series(MONTHLY_SERIES, start_date=START_DATE))
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    except Exception as e:  # noqa: BLE001 - fallback'e geçmek için
        log.warning("Aylık EIA serisi alınamadı (%s), haftalık seriden türetiliyor.", e)
        w = _extract_series_values(eia.get_series(WEEKLY_SERIES, start_date=START_DATE))
        w.index = pd.to_datetime(w.index)
        s = w.resample("MS").mean().dropna()

    df = s.sort_index().loc[START_DATE:].to_frame("ABD_Ithalat_KBD")
    df.index.name = "Date"
    atomic_write_csv(df, FILES["eia"])
    log.info("EIA güncellendi: %s (%d ay)", FILES["eia"].name, len(df))
