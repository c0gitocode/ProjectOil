import os
import io
import time
import warnings
import gc
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from curl_cffi import requests as c_requests
import pdfplumber
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from config import RAW_DIR, PROCESSED_DIR, EIA_API_KEY
from myeia.api import API

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. JODI (AVRUPA & ASYA) - Retry & Timeout Korumalı Versiyon
# ---------------------------------------------------------
def jodi_guncelle():
    master_file = os.path.join(PROCESSED_DIR, "jodi_avrupa_asya_petrodolar.csv")
    hedef_iso = [
        'JP', 'KR', 'TW', 'CA', 'AU', 'NZ', 'MX', 'GB',
        'DE', 'FR', 'IT', 'ES', 'GR', 'NO', 'IN', 'AT',
        'BE', 'BG', 'CH', 'CZ', 'HU', 'IE', 'NL', 'PL', 'RO', 'SE'
    ]
    print("[*] JODI veri çekme işlemi başlıyor...")

    existing_dates = []
    df_old = pd.DataFrame()
    existing_years = set()

    if os.path.exists(master_file):
        df_old = pd.read_csv(master_file, index_col=0, parse_dates=True)
        existing_dates = pd.to_datetime(df_old.index).tolist()
        existing_years = set(pd.to_datetime(df_old.index).year)
        max_date = max(existing_dates) if existing_dates else pd.to_datetime('1900-01-01')
    else:
        max_date = pd.to_datetime('1900-01-01')

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    soup = None
    for attempt in range(3):
        try:
            res = requests.get("https://www.jodidata.org/oil/database/data-downloads.aspx", headers=headers, timeout=60)
            soup = BeautifulSoup(res.text, 'html.parser')
            break
        except Exception as e:
            print(f"[!] JODI Bağlantı denemesi {attempt + 1}/3 başarısız: {e}")
            time.sleep(3)

    if not soup:
        print("[!] JODI sunucusuna ulaşılamadı, işlem iptal ediliyor.")
        return

    yil_linkleri = {}
    for a in soup.find_all('a', href=True):
        text = a.text.strip()
        href = a['href'].lower()
        if text.isdigit() and 2018 <= int(text) <= 2026 and '.csv' in href:
            parent_text = a.find_parent().get_text().lower()
            if 'primary' in parent_text or 'primary' in href or 'world' in href:
                yil_linkleri[text] = "https://www.jodidata.org" + a['href'] if a['href'].startswith('/') else a['href']

    new_dfs = []
    current_year = pd.Timestamp.now().year

    for yil in sorted(yil_linkleri.keys()):
        yil_int = int(yil)

        # AKILLI ATLAMA (SKIP): Eğer geçmiş bir yılsa ve bu yıla ait veriler zaten master dosyada varsa, indirme!
        if yil_int < current_year and yil_int in existing_years and not df_old.empty:
            print(f"[*] JODI {yil} verisi zaten mevcut, indirme atlanıyor.")
            continue

        success = False
        for attempt in range(3):
            try:
                print(f"[*] JODI {yil} verisi indiriliyor (Deneme {attempt + 1})...")
                r = requests.get(yil_linkleri[yil], headers=headers, timeout=120)
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.upper().strip() for c in df.columns]

                mapping = {
                    'TIME': 'TIME_PERIOD', 'VAL': 'OBS_VALUE', 'VALUE': 'OBS_VALUE',
                    'PRODUCT': 'ENERGY_PRODUCT', 'FLOW': 'FLOW_BREAKDOWN', 'UNIT': 'UNIT_MEASURE'
                }
                df = df.rename(columns=mapping)

                for col in ['REF_AREA', 'ENERGY_PRODUCT', 'FLOW_BREAKDOWN', 'UNIT_MEASURE']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.upper().str.replace(' ', '')

                mask = ((df['ENERGY_PRODUCT'].str.contains('CRUDE')) &
                        (df['FLOW_BREAKDOWN'].str.contains('IMP')) &
                        (df['UNIT_MEASURE'].str.contains('KB')) &
                        (df['REF_AREA'].isin(hedef_iso)))

                df_filtered = df[mask].copy()
                if not df_filtered.empty:
                    df_filtered['TIME_PERIOD'] = pd.to_datetime(df_filtered['TIME_PERIOD'])
                    new_dfs.append(df_filtered[['TIME_PERIOD', 'REF_AREA', 'OBS_VALUE']])

                success = True
                time.sleep(1)
                break
            except Exception as e:
                print(f"[!] JODI Hata ({yil} - Deneme {attempt + 1}): {e}")
                time.sleep(3)

        if not success:
            print(f"[!] JODI {yil} verisi indirilemedi, atlanıyor.")

    if not new_dfs and df_old.empty:
        print("[*] JODI: İndirilen yeni veri bulunamadı.")
        return

    # Eski veri ile yeni gelen verileri birleştir
    if new_dfs:
        df_new = pd.concat(new_dfs, ignore_index=True)
        df_new['OBS_VALUE'] = pd.to_numeric(df_new['OBS_VALUE'], errors='coerce').fillna(0)
        df_pivot = df_new.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE',
                                      aggfunc='sum').fillna(0)

        if not df_old.empty:
            df_final = pd.concat([df_old, df_pivot]).groupby(level=0).last()
        else:
            df_final = df_pivot
    else:
        df_final = df_old

    df_final.index = pd.to_datetime(df_final.index)
    df_final.sort_index(inplace=True)

    country_cols = [c for c in df_final.columns if c != 'Toplam_Avrupa_Asya_Petrodolar']
    df_final[country_cols] = df_final[country_cols].replace(0, np.nan).ffill().fillna(0)
    df_final['Toplam_Avrupa_Asya_Petrodolar'] = df_final[country_cols].sum(axis=1)

    df_final.to_csv(master_file)
    print("[+] JODI: İşlem tamamlandı. Master dosya güncellendi.")

# ---------------------------------------------------------
# 2. EIA (ABD)
# ---------------------------------------------------------
def eia_guncelle():
    master_file = os.path.join(PROCESSED_DIR, "saf_abd_petrodolar.csv")
    print("[*] EIA (ABD) veri çekme işlemi başlıyor...")

    eia = API(EIA_API_KEY)
    df_old = pd.DataFrame()
    if os.path.exists(master_file):
        df_old = pd.read_csv(master_file, index_col=0)
        df_old.index = pd.to_datetime(df_old.index)

    try:
        df_imp_w = pd.DataFrame(eia.get_series("PET.WCRIMUS2.W"))
        df_imp_w.index = pd.to_datetime(df_imp_w.index)
        df_imp_m = df_imp_w.resample('MS').mean()
        df_new = pd.DataFrame(index=df_imp_m.index)
        df_new['ABD_Ithalat_KBD'] = df_imp_m.iloc[:, 0]
        df_new = df_new.ffill().dropna().loc["2018-01-01":]

        if not df_old.empty:
            df_combined = pd.concat([df_old, df_new[['ABD_Ithalat_KBD']]])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
        else:
            df_combined = df_new[['ABD_Ithalat_KBD']]

        df_combined.to_csv(master_file)
        print("[+] EIA (ABD) verisi güncellendi.")
    except Exception as e:
        print(f"[!] EIA Hata: {e}")


# ---------------------------------------------------------
# 3. EPDK (TÜRKİYE)
# ---------------------------------------------------------
def sutun_adi_temizle(c):
    return str(c).upper().replace('\n', ' ').replace('\r', ' ').strip()


def metrik_ton_to_kbd(mt, yil, ay):
    if pd.isna(mt) or mt <= 0: return 0
    try:
        gun_sayisi = monthrange(yil, ay)[1]
    except:
        gun_sayisi = 30
    return round(((mt * 7.33) / gun_sayisi / 1000), 2)


def temizle_sayi(val):
    if pd.isna(val) or val == '' or val == '-': return 0.0
    if isinstance(val, (int, float)): return float(val)
    val = str(val).strip()
    if ',' in val: val = val.replace('.', '').replace(',', '.')
    try:
        return float(val)
    except:
        return 0.0


def pdf_tablo_ayikla_ve_oku(dosya_yolu):
    with pdfplumber.open(dosya_yolu) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables: continue
            for table in tables:
                df = pd.DataFrame(table).fillna('')
                df_str = df.astype(str).apply(lambda x: x.str.upper())
                if df_str.apply(lambda x: x.str.contains('HAM PETROL', na=False)).any().any():
                    header_row = -1
                    for i, row in df.iterrows():
                        r_vals = [sutun_adi_temizle(v) for v in row.values]
                        if any('LKE' in v for v in r_vals) and any('HAM' in v and 'PETROL' in v for v in r_vals):
                            header_row = i
                            break
                    if header_row != -1:
                        headers = [sutun_adi_temizle(v) for v in df.iloc[header_row].values]
                        ulke_col = next((idx for idx, h in enumerate(headers) if 'LKE' in h), 0)
                        hp_col = next((idx for idx, h in enumerate(headers) if 'HAM' in h and 'PETROL' in h), 1)
                        toplam_hp = 0.0
                        rusya_hp = 0.0
                        hesaplanan_toplam = 0.0
                        for r in range(header_row + 1, len(df)):
                            val_ulke = str(df.iloc[r, ulke_col]).strip().upper()
                            if not val_ulke: continue
                            val_hp = temizle_sayi(df.iloc[r, hp_col])
                            if 'RUSYA' in val_ulke:
                                rusya_hp = val_hp
                            if 'TOPLAM' in val_ulke:
                                toplam_hp = val_hp
                            else:
                                hesaplanan_toplam += val_hp
                        if toplam_hp <= 0:
                            toplam_hp = hesaplanan_toplam
                        if toplam_hp > 0:
                            return toplam_hp, rusya_hp
    return None, None


def rapor_isle(yil, ay, link):
    session = c_requests.Session(impersonate="chrome")
    yil_dir = os.path.join(RAW_DIR, str(yil))
    os.makedirs(yil_dir, exist_ok=True)
    dosya_tmp = os.path.join(yil_dir, f"epdk_{yil}_{ay}.pdf")
    try:
        res = session.get(link, timeout=60)
        with open(dosya_tmp, 'wb') as f:
            f.write(res.content)
        toplam, rusya = pdf_tablo_ayikla_ve_oku(dosya_tmp)
        if not toplam or toplam <= 0:
            if os.path.exists(dosya_tmp): os.remove(dosya_tmp)
            return None
        t_kbd = metrik_ton_to_kbd(toplam, yil, ay)
        r_kbd = metrik_ton_to_kbd(rusya, yil, ay)
        gc.collect()
        if t_kbd <= 0:
            return None
        return {
            'Tarih': f"{yil}-{str(ay).zfill(2)}-01",
            'TR_Toplam_Ithalat_KBD': t_kbd,
            'TR_Rusya_Iskontosu_KBD': r_kbd,
            'TR_Saf_Petrodolar_KBD': round(t_kbd - r_kbd, 2),
            'Rusya_Payi_Yuzde': round((r_kbd / t_kbd * 100), 2) if t_kbd > 0 else 0
        }
    except Exception as e:
        if os.path.exists(dosya_tmp): os.remove(dosya_tmp)
        return None
    finally:
        session.close()


def epdk_guncelle():
    print("[*] EPDK veri çekme işlemi başlıyor...")
    master_file = os.path.join(PROCESSED_DIR, "tr_epdk_master.csv")
    processed_dates = []
    if os.path.exists(master_file):
        existing_df = pd.read_csv(master_file)
        processed_dates = existing_df['Tarih'].tolist()

    base_session = c_requests.Session(impersonate="chrome")
    soup = BeautifulSoup(
        base_session.get("https://www.epdk.gov.tr/Detay/Icerik/3-0-104/petrolaylik-sektor-raporu").text, 'html.parser')
    harita = {}

    for a in soup.find_all('a', href=True):
        if 'DownloadDocument' in a['href']:
            href = a['href']
            parent_txt = (a.text.strip() or a.find_parent().text.strip()).lower()
            if any(x in parent_txt for x in ['il bazında', 'bayi']): continue

            is_pdf = False
            img = a.find('img')
            if (img and img.get('src') and ('pdf' in img['src'].lower())) or (
                    'pdf' in parent_txt or href.lower().endswith('.pdf')):
                is_pdf = True
            if not is_pdf: continue

            for y in range(2018, 2028):
                for m_n, m_s in {1: "ocak", 2: "şubat", 3: "mart", 4: "nisan", 5: "mayıs", 6: "haziran", 7: "temmuz",
                                 8: "ağustos", 9: "eylül", 10: "ekim", 11: "kasım", 12: "aralık"}.items():
                    if str(y) in parent_txt and m_s in parent_txt:
                        tarih_str = f"{y}-{str(m_n).zfill(2)}-01"
                        if tarih_str in processed_dates: continue
                        link = "https://www.epdk.gov.tr" + href if href.startswith('/') else href
                        if (y, m_n) not in harita:
                            harita[(y, m_n)] = link

    gorevler = [(y, m, l) for (y, m), l in harita.items()]
    if not gorevler:
        print("[*] EPDK: Yeni PDF bulunamadı. Veri seti güncel.")
    else:
        sonuclar = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(rapor_isle, y, m, l) for y, m, l in gorevler]
            for f in as_completed(futures):
                r = f.result()
                if r: sonuclar.append(r)

        if sonuclar:
            new_df = pd.DataFrame(sonuclar)
            combined_df = pd.concat([pd.read_csv(master_file), new_df]) if os.path.exists(master_file) else new_df
            combined_df = combined_df.drop_duplicates(subset=['Tarih']).sort_values('Tarih')
            combined_df.to_csv(master_file, index=False)
            print(f"[+] EPDK verisi güncellendi.")


# ---------------------------------------------------------
# 4. DGCIS (HİNDİSTAN) - GÜNCELLENMİŞ VERSİYON
# ---------------------------------------------------------
def dgcis_guncelle():
    print("[*] DGCIS (Hindistan) veri çekme işlemi başlıyor...")
    csv_filename = os.path.join(PROCESSED_DIR, "russia_oil_ports_2018_2026.csv")
    months_map = [("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
                  ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12)]

    date_range = []
    for year in range(2018, 2027):
        for m_name, m_num in months_map:
            # 2021 Nisan öncesini tamamen atla (Vakit kaybını önler)
            if year < 2021 or (year == 2021 and m_num < 4):
                continue
            if year == 2026 and m_num > 4:
                break
            date_range.append(f"{m_name}-{year}")

    existing_data = []
    completed_periods = set()
    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        if "Period" in df_existing.columns:
            completed_periods = set(df_existing["Period"].unique())
            existing_data = df_existing.to_dict("records")
            print(f"[*] Mevcut dosya bulundu. Tamamlanmış {len(completed_periods)} ay atlanacak.")

    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)
    new_data = []

    try:
        driver.get("https://ftddp.dgciskol.gov.in/dgcis/principalcommditysearch.html#!/freeuser")
        wait = WebDriverWait(driver, 25)

        print("[*] Form şablonu ilk kez kuruluyor...")
        first_period = date_range[0]

        to_date = wait.until(EC.element_to_be_clickable((By.ID, "datepicker1")))
        to_date.clear()
        to_date.send_keys(first_period)

        from_date = wait.until(EC.element_to_be_clickable((By.ID, "datepicker")))
        from_date.clear()
        from_date.send_keys(first_period)

        import_radio = driver.find_element(By.ID, "eximp2")
        driver.execute_script("arguments[0].click();", import_radio)

        currency_radio = driver.find_element(By.ID, "currency3")
        driver.execute_script("arguments[0].click();", currency_radio)

        # Rapor Tipi: 10
        type_elem = driver.find_element(By.ID, "type")
        driver.execute_script("arguments[0].scrollIntoView(true);", type_elem)
        Select(type_elem).select_by_value("10")
        time.sleep(2)

        # Sorted By
        sorted_elem = driver.find_element(By.ID, "sorted")
        driver.execute_script("arguments[0].scrollIntoView(true);", sorted_elem)
        Select(sorted_elem).select_by_index(1)
        time.sleep(1)

        # Emtia Seçimi: S5
        comm_elem = driver.find_element(By.ID, "comm")
        driver.execute_script("arguments[0].scrollIntoView(true);", comm_elem)
        Select(comm_elem).select_by_value("S5")
        add_comm_btn = driver.find_element(By.XPATH, "//input[@onclick=\"adOption.addOption('sel_list','comm');\"]")
        driver.execute_script("arguments[0].click();", add_comm_btn)
        time.sleep(1)

        # Ülke Seçimi: 15344
        country_elem = driver.find_element(By.ID, "con")
        driver.execute_script("arguments[0].scrollIntoView(true);", country_elem)
        Select(country_elem).select_by_value("15344")
        add_country_btn = driver.find_element(By.XPATH, "//input[@onclick=\"adOption.addOption('sel_list1','con');\"]")
        driver.execute_script("arguments[0].click();", add_country_btn)
        time.sleep(1)

        # Port Seçimi: A
        port_elem = driver.find_element(By.ID, "port")
        driver.execute_script("arguments[0].scrollIntoView(true);", port_elem)
        Select(port_elem).select_by_value("A")
        add_port_btn = driver.find_element(By.XPATH, "//input[@onclick=\"adOption.addOption('sel_list4','port');\"]")
        driver.execute_script("arguments[0].click();", add_port_btn)
        time.sleep(1)

        print("[*] Form hazır. Eksik aylar taranıyor...")
        for period in date_range:
            if period in completed_periods:
                continue
            try:
                to_date = wait.until(EC.element_to_be_clickable((By.ID, "datepicker1")))
                to_date.clear()
                to_date.send_keys(period)
                from_date = wait.until(EC.element_to_be_clickable((By.ID, "datepicker")))
                from_date.clear()
                from_date.send_keys(period)

                execute_btn = wait.until(EC.element_to_be_clickable((By.ID, "select_all")))
                driver.execute_script("arguments[0].scrollIntoView(true);", execute_btn)
                driver.execute_script("arguments[0].click();", execute_btn)

                # Zaman aşımı koruması için explicit wait süresi
                wait.until(
                    EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'table-bordered')]//tbody")))
                time.sleep(2)

                table_html = driver.find_element(By.ID, "result").get_attribute("outerHTML")
                soup = BeautifulSoup(table_html, "html.parser")
                rows = soup.select("tbody tr")

                found_count = 0
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) >= 7:
                        new_data.append({
                            "Period": period,
                            "No": cols[0].text.strip(),
                            "PC_Code": cols[1].text.strip(),
                            "Commodity": cols[2].text.strip(),
                            "Country": cols[3].text.strip(),
                            "Port": cols[4].text.strip(),
                            "Unit": cols[5].text.strip(),
                            "Qty": cols[6].text.strip(),
                            "Value_INR": cols[7].text.strip() if len(cols) > 7 else "",
                            "Value_USD": cols[8].text.strip() if len(cols) > 8 else "",
                        })
                        found_count += 1
                print(f"[+] {period} tarandı. Eklenen kayıt: {found_count}")

            except Exception as e:
                print(f"[!] DGCIS {period} ayında zaman aşımı/hata oluştu, pas geçiliyor: {e}")
                time.sleep(5)
                continue

        all_combined = existing_data + new_data
        if all_combined:
            final_df = pd.DataFrame(all_combined)
            final_df.to_csv(csv_filename, index=False)
            print(f"[+] DGCIS (Hindistan) verisi güncellendi: {csv_filename}")
        else:
            print("[*] DGCIS (Hindistan): Yeni veri bulunamadı.")

    finally:
        driver.quit()