import os
from dotenv import load_dotenv

load_dotenv()

EIA_API_KEY = os.getenv("EIA_API_KEY")
if not EIA_API_KEY:
    raise ValueError("[!] EIA_API_KEY .env dosyasında bulunamadı!")

# Temel Dizin Yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Hindistan Hacim Analizi Sabitleri
INDIA_V_VOL = 1000.0
INDIA_O_VOL = 1000.0
INDIA_P_VOL = 1000.0
INDIA_TOTAL_VOL = INDIA_V_VOL + INDIA_O_VOL + INDIA_P_VOL