import os
import sqlite3
import pandas as pd
from config import PROCESSED_DIR


def load_data_to_sqlite():
    db_path = os.path.join(PROCESSED_DIR, "oil_database.db")
    print("[*] Load katmanı: Veriler SQLite veritabanına yükleniyor...")

    conn = sqlite3.connect(db_path)

    # Yüklenecek dosya ve tablo eşleşmeleri
    files_to_load = {
        "master_analysis": "MASTER_ANALYSIS.csv",
        "china_imports": "china_oil_imports_processed.csv",
        "india_analysis": "hindistan_nihai_hacim_analizi_kbd.csv",
        "tr_epdk": "tr_epdk_master.csv",
        "jodi_europe_asia": "jodi_avrupa_asya_petrodolar.csv",
        "us_petrodollar": "saf_abd_petrodolar.csv"
    }

    loaded_count = 0
    for table_name, file_name in files_to_load.items():
        file_path = os.path.join(PROCESSED_DIR, file_name)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Tabloyu veritabanına yaz (varsa üzerine yaz)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"[+] Tablo güncellendi/oluşturuldu: {table_name} ({len(df)} satır)")
                loaded_count += 1
            except Exception as e:
                print(f"[!] Hata ({table_name}): {e}")
        else:
            print(f"[-] Dosya bulunamadı, atlanıyor: {file_name}")

    conn.close()
    print(f"[+] Load katmanı tamamlandı. Toplam {loaded_count} tablo veritabanına işlendi: {db_path}")


if __name__ == "__main__":
    load_data_to_sqlite()