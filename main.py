import time
import sys
import os

# src klasörünü Python arama yoluna ekle
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from extract import jodi_guncelle, eia_guncelle, epdk_guncelle, dgcis_guncelle
from transform import master_analiz_olustur, process_china_oil_data, process_india_oil_data, process_china_payment_breakdown, global_ikili_hegemonya_birlestir
from load import load_data_to_sqlite


def run_pipeline():
    start_time = time.time()

    print("=== 1. VERİ ÇEKME (EXTRACT) KATMANI BAŞLIYOR ===")
    try:
        jodi_guncelle()
        eia_guncelle()
        epdk_guncelle()
        dgcis_guncelle()
    except Exception as e:
        print(f"[!] Extract aşamasında hata: {e}")

    print("\n=== 2. VERİ İŞLEME (TRANSFORM) KATMANI BAŞLIYOR ===")
    try:
        master_analiz_olustur()
        process_china_oil_data()
        process_china_payment_breakdown()  # Yeni eklenen Çin ödeme analizi
        process_india_oil_data()
        global_ikili_hegemonya_birlestir()  # Küresel birleştirme
    except Exception as e:
        print(f"[!] Transform aşamasında hata: {e}")

    print("\n=== 3. VERİ YÜKLEME (LOAD) KATMANI BAŞLIYOR ===")
    try:
        load_data_to_sqlite()
    except Exception as e:
        print(f"[!] Load aşamasında hata: {e}")

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[+] Tüm ETL boru hattı başarıyla tamamlandı. Geçen süre: {elapsed} saniye.")


if __name__ == "__main__":
    run_pipeline()

from src.utils import global_grafik_ciz

# Pipeline bittikten sonra grafiği çizdir
global_grafik_ciz()