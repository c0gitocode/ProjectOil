"""Merkezi proje ayarları.

Tüm tarih aralıkları, birim sabitleri ve dizin yolları buradan okunur.
Tarih aralıkları .env üzerinden (START_DATE / END_DATE) override edilebilir.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------
# Dizinler
# ---------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHINA_RAW_DIR = RAW_DIR / "china"
PROCESSED_DIR = DATA_DIR / "processed"
ASSUMPTIONS_DIR = DATA_DIR / "assumptions"
OUTPUTS_DIR = BASE_DIR / "outputs"
DB_PATH = PROCESSED_DIR / "oil_database.db"

for _d in (RAW_DIR, CHINA_RAW_DIR, PROCESSED_DIR, ASSUMPTIONS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Analiz dönemi (tek kaynak: hiçbir modülde tarih hardcode edilmez)
# ---------------------------------------------------------
START_DATE = os.getenv("START_DATE", "2018-01-01")
END_DATE = os.getenv("END_DATE", "2026-02-28")
EVENT_DATE = "2022-02-24"          # Ukrayna işgali (grafik işareti)
BREAK_DATE = "2022-02-01"          # Yapısal kırılma ayı (ekonometri)
SENSITIVITY = 0.05                 # Ödeme payı varsayımları için ±%5 bant
BAND_START = BREAK_DATE            # Duyarlılık bandı bu tarihten itibaren uygulanır

# JODI: Bir ülkenin "zorunlu" sayılması için örneklemde veri bulunma oranı
JODI_REQUIRED_COVERAGE = 0.90

# ---------------------------------------------------------
# Birim sabitleri
# ---------------------------------------------------------
BBL_PER_TONNE = 7.33               # Ham petrol ortalama dönüşüm katsayısı

# ---------------------------------------------------------
# İşlenmiş dosya adları (modüller arası sözleşme)
# ---------------------------------------------------------
FILES = {
    "jodi": PROCESSED_DIR / "jodi_crude_imports_kbd.csv",
    "eia": PROCESSED_DIR / "us_crude_imports_kbd.csv",
    "epdk": PROCESSED_DIR / "tr_epdk_master.csv",
    "dgcis": PROCESSED_DIR / "india_russia_oil_dgcis.csv",
    "tr_master": PROCESSED_DIR / "MASTER_ANALYSIS.csv",
    "china_imports": PROCESSED_DIR / "china_oil_imports_processed.csv",
    "china_payment": PROCESSED_DIR / "china_petrodolar_petroyuan_kbd.csv",
    "india": PROCESSED_DIR / "hindistan_nihai_hacim_analizi_kbd.csv",
    "global": PROCESSED_DIR / "GLOBAL_PETRODOLAR_VS_PETROYUAN.csv",
    "zero_global": PROCESSED_DIR / "GLOBAL_ZERO_ASSUMPTION.csv",
    "panel": PROCESSED_DIR / "panel_long.csv",
}

ASSUMPTION_FILES = {
    "china": ASSUMPTIONS_DIR / "china_alt_currency_shares.csv",
    "india": ASSUMPTIONS_DIR / "india_alt_currency_shares.csv",
}


def get_eia_api_key() -> str:
    """EIA anahtarını sadece gerektiğinde ister (import anında patlamaz)."""
    key = os.getenv("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY .env dosyasında bulunamadı.")
    return key