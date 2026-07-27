import os
import io
import time
import warnings
import base64
import random
import glob
import traceback
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List

import cv2
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from curl_cffi import requests as c_requests
import pdfplumber
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options

from config import RAW_DIR, PROCESSED_DIR, EIA_API_KEY
from myeia.api import API

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def create_retry_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def sutun_adi_temizle(c: Any) -> str:
    return str(c).upper().replace('\n', ' ').replace('\r', ' ').strip()


def metrik_ton_to_kbd(mt: float, yil: int, ay: int) -> float:
    if pd.isna(mt) or mt <= 0:
        return 0.0
    try:
        gun_sayisi = monthrange(yil, ay)[1]
    except ValueError:
        gun_sayisi = 30
    return round(((mt * 7.33) / gun_sayisi / 1000), 2)


def temizle_sayi(val: Any) -> float:
    if pd.isna(val) or val in ('', '-'):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().replace('.', '').replace(',', '.')
    try:
        return float(val_str)
    except ValueError:
        return 0.0


# ---------------------------------------------------------
# 1. JODI (AVRUPA & ASYA)
# ---------------------------------------------------------
def jodi_guncelle() -> None:
    master_file = os.path.join(PROCESSED_DIR, "jodi_avrupa_asya_petrodolar.csv")
    hedef_iso = {
        'JP', 'KR', 'TW', 'CA', 'AU', 'NZ', 'MX', 'GB', 'DE', 'FR', 'IT', 'ES',
        'GR', 'NO', 'IN', 'AT', 'BE', 'BG', 'CH', 'CZ', 'HU', 'IE', 'NL', 'PL', 'RO', 'SE'
    }
    print("[*] JODI veri çekme işlemi başlıyor...")

    df_old = pd.DataFrame()
    existing_years = set()

    if os.path.exists(master_file):
        df_old = pd.read_csv(master_file, index_col=0, parse_dates=True)
        existing_years = set(pd.to_datetime(df_old.index).year)

    session = create_retry_session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    try:
        res = session.get("https://www.jodidata.org/oil/database/data-downloads.aspx", headers=headers, timeout=30)
        soup = BeautifulSoup(res.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"[!] JODI sunucusuna ulaşılamadı: {e}")
        return

    yil_linkleri = {}
    for a in soup.find_all('a', href=True):
        text = a.text.strip()
        href = a['href'].lower()
        if text.isdigit() and 2018 <= int(text) <= 2026 and '.csv' in href:
            parent_text = a.find_parent().get_text().lower()
            if 'primary' in parent_text or 'primary' in href or 'world' in href:
                yil_linkleri[int(text)] = "https://www.jodidata.org" + a['href'] if a['href'].startswith('/') else a[
                    'href']

    new_dfs = []
    current_year = pd.Timestamp.now().year

    for yil in sorted(yil_linkleri.keys()):
        if yil < current_year and yil in existing_years and not df_old.empty:
            continue

        try:
            print(f"[*] JODI {yil} verisi indiriliyor...")
            r = session.get(yil_linkleri[yil], headers=headers, timeout=60)
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.upper().strip() for c in df.columns]

            mapping = {
                'TIME': 'TIME_PERIOD', 'VAL': 'OBS_VALUE', 'VALUE': 'OBS_VALUE',
                'PRODUCT': 'ENERGY_PRODUCT', 'FLOW': 'FLOW_BREAKDOWN', 'UNIT': 'UNIT_MEASURE'
            }
            df = df.rename(columns=mapping)

            mask = (
                    (df['ENERGY_PRODUCT'].astype(str).str.upper().str.contains('CRUDE')) &
                    (df['FLOW_BREAKDOWN'].astype(str).str.upper().str.contains('IMP')) &
                    (df['UNIT_MEASURE'].astype(str).str.upper().str.contains('KB')) &
                    (df['REF_AREA'].astype(str).str.upper().str.strip().isin(hedef_iso))
            )

            df_filtered = df[mask].copy()
            if not df_filtered.empty:
                df_filtered['TIME_PERIOD'] = pd.to_datetime(df_filtered['TIME_PERIOD'])
                new_dfs.append(df_filtered[['TIME_PERIOD', 'REF_AREA', 'OBS_VALUE']])
        except (requests.exceptions.RequestException, pd.errors.EmptyDataError) as e:
            print(f"[!] JODI Hata ({yil}): {e}")

    if not new_dfs and df_old.empty:
        print("[*] JODI: İndirilen yeni veri bulunamadı.")
        return

    if new_dfs:
        df_new = pd.concat(new_dfs, ignore_index=True)
        df_new['OBS_VALUE'] = pd.to_numeric(df_new['OBS_VALUE'], errors='coerce').fillna(0)
        df_pivot = df_new.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE',
                                      aggfunc='sum').fillna(0)

        df_final = pd.concat([df_old, df_pivot]).groupby(level=0).last() if not df_old.empty else df_pivot
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
def eia_guncelle() -> None:
    master_file = os.path.join(PROCESSED_DIR, "saf_abd_petrodolar.csv")
    print("[*] EIA (ABD) veri çekme işlemi başlıyor...")

    eia = API(EIA_API_KEY)
    df_old = pd.DataFrame()
    if os.path.exists(master_file):
        df_old = pd.read_csv(master_file, index_col=0, parse_dates=True)

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
        print(f"[!] EIA API Hatası: {e}")


# ---------------------------------------------------------
# 3. EPDK (TÜRKİYE)
# ---------------------------------------------------------
def pdf_tablo_ayikla_ve_oku(dosya_yolu: str) -> Tuple[Optional[float], Optional[float]]:
    with pdfplumber.open(dosya_yolu) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                df = pd.DataFrame(table).fillna('')
                df_str = df.astype(str).map(sutun_adi_temizle)

                if not df_str.apply(lambda x: x.str.contains('HAM PETROL')).any().any():
                    continue

                header_mask = df_str.apply(
                    lambda row: row.str.contains('LKE').any() and row.str.contains('HAM PETROL').any(), axis=1)
                if not header_mask.any():
                    continue

                header_row_idx = header_mask.idxmax()
                headers = df_str.iloc[header_row_idx].tolist()

                try:
                    ulke_col = next(i for i, h in enumerate(headers) if 'LKE' in h)
                    hp_col = next(i for i, h in enumerate(headers) if 'HAM PETROL' in h)
                except StopIteration:
                    continue

                data_df = df_str.iloc[header_row_idx + 1:].copy()
                data_df[hp_col] = data_df[hp_col].apply(temizle_sayi)

                rusya_mask = data_df[ulke_col].str.contains('RUSYA')
                toplam_mask = data_df[ulke_col].str.contains('TOPLAM')

                rusya_hp = data_df.loc[rusya_mask, hp_col].sum()
                toplam_hp = data_df.loc[toplam_mask, hp_col].sum()

                if toplam_hp <= 0:
                    toplam_hp = data_df.loc[~toplam_mask, hp_col].sum()

                if toplam_hp > 0:
                    return toplam_hp, rusya_hp

    return None, None


def rapor_isle(yil: int, ay: int, link: str) -> Optional[Dict[str, Any]]:
    session = c_requests.Session(impersonate="chrome")
    yil_dir = os.path.join(RAW_DIR, str(yil))
    os.makedirs(yil_dir, exist_ok=True)
    dosya_tmp = os.path.join(yil_dir, f"epdk_{yil}_{ay}.pdf")

    try:
        res = session.get(link, timeout=30)
        with open(dosya_tmp, 'wb') as f:
            f.write(res.content)

        toplam, rusya = pdf_tablo_ayikla_ve_oku(dosya_tmp)
        if not toplam or toplam <= 0:
            return None

        t_kbd = metrik_ton_to_kbd(toplam, yil, ay)
        r_kbd = metrik_ton_to_kbd(rusya, yil, ay)

        if t_kbd <= 0:
            return None

        return {
            'Tarih': f"{yil}-{str(ay).zfill(2)}-01",
            'TR_Toplam_Ithalat_KBD': t_kbd,
            'TR_Rusya_Iskontosu_KBD': r_kbd,
            'TR_Saf_Petrodolar_KBD': round(t_kbd - r_kbd, 2),
            'Rusya_Payi_Yuzde': round((r_kbd / t_kbd * 100), 2) if t_kbd > 0 else 0.0
        }
    except Exception:
        return None
    finally:
        session.close()
        if os.path.exists(dosya_tmp) and os.path.getsize(dosya_tmp) < 1024:
            os.remove(dosya_tmp)


def epdk_guncelle() -> None:
    print("[*] EPDK veri çekme işlemi başlıyor...")
    master_file = os.path.join(PROCESSED_DIR, "tr_epdk_master.csv")
    processed_dates = set()

    if os.path.exists(master_file):
        processed_dates = set(pd.read_csv(master_file)['Tarih'].tolist())

    base_session = c_requests.Session(impersonate="chrome")
    try:
        res = base_session.get("https://www.epdk.gov.tr/Detay/Icerik/3-0-104/petrolaylik-sektor-raporu", timeout=30)
        soup = BeautifulSoup(res.text, 'html.parser')
    except Exception as e:
        print(f"[!] EPDK anasayfasına ulaşılamadı: {e}")
        return
    finally:
        base_session.close()

    harita = {}
    aylar_map = {1: "ocak", 2: "şubat", 3: "mart", 4: "nisan", 5: "mayıs", 6: "haziran",
                 7: "temmuz", 8: "ağustos", 9: "eylül", 10: "ekim", 11: "kasım", 12: "aralık"}

    for a in soup.find_all('a', href=True):
        if 'DownloadDocument' not in a['href']:
            continue

        href = a['href']
        parent_txt = (a.text.strip() or a.find_parent().text.strip()).lower()
        if 'il bazında' in parent_txt or 'bayi' in parent_txt:
            continue

        img = a.find('img')
        is_pdf = (img and img.get('src') and 'pdf' in img['src'].lower()) or (
                'pdf' in parent_txt or href.lower().endswith('.pdf'))
        if not is_pdf:
            continue

        for y in range(2018, 2028):
            for m_n, m_s in aylar_map.items():
                if str(y) in parent_txt and m_s in parent_txt:
                    tarih_str = f"{y}-{str(m_n).zfill(2)}-01"
                    if tarih_str not in processed_dates:
                        link = "https://www.epdk.gov.tr" + href if href.startswith('/') else href
                        harita[(y, m_n)] = link

    if not harita:
        print("[*] EPDK: Yeni PDF bulunamadı. Veri seti güncel.")
        return

    sonuclar = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(rapor_isle, y, m, l) for (y, m), l in harita.items()]
        for f in as_completed(futures):
            res = f.result()
            if res:
                sonuclar.append(res)

    if sonuclar:
        new_df = pd.DataFrame(sonuclar)
        combined_df = pd.concat([pd.read_csv(master_file), new_df]) if os.path.exists(master_file) else new_df
        combined_df.drop_duplicates(subset=['Tarih']).sort_values('Tarih').to_csv(master_file, index=False)
        print("[+] EPDK verisi güncellendi.")


# ---------------------------------------------------------
# 4. DGCIS (HİNDİSTAN)
# ---------------------------------------------------------
def dgcis_guncelle() -> None:
    print("[*] DGCIS (Hindistan) veri çekme işlemi başlıyor...")
    csv_filename = os.path.join(PROCESSED_DIR, "russia_oil_ports_2018_2026.csv")
    months_map = [("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
                  ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12)]

    date_range = [
        f"{m_name}-{year}"
        for year in range(2021, 2027)
        for m_name, m_num in months_map
        if not (year == 2021 and m_num < 4) and not (year == 2026 and m_num > 4)
    ]

    existing_data = []
    completed_periods = set()

    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        if "Period" in df_existing.columns:
            completed_periods = set(df_existing["Period"].unique())
            existing_data = df_existing.to_dict("records")
            print(f"[*] Tamamlanmış {len(completed_periods)} ay atlanacak.")

    if all(period in completed_periods for period in date_range):
        print("[*] DGCIS: Tüm periyotlar zaten çekilmiş, işlem atlanıyor.")
        return

    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)
    new_data = []

    try:
        driver.get("https://ftddp.dgciskol.gov.in/dgcis/principalcommditysearch.html#!/freeuser")
        wait = WebDriverWait(driver, 15)

        first_period = date_range[0]
        wait.until(EC.element_to_be_clickable((By.ID, "datepicker1"))).send_keys(first_period)
        wait.until(EC.element_to_be_clickable((By.ID, "datepicker"))).send_keys(first_period)

        driver.execute_script("arguments[0].click();", driver.find_element(By.ID, "eximp2"))
        driver.execute_script("arguments[0].click();", driver.find_element(By.ID, "currency3"))

        Select(driver.find_element(By.ID, "type")).select_by_value("10")
        Select(driver.find_element(By.ID, "sorted")).select_by_index(1)

        Select(driver.find_element(By.ID, "comm")).select_by_value("S5")
        driver.execute_script("arguments[0].click();", driver.find_element(By.XPATH,
                                                                           "//input[@onclick=\"adOption.addOption('sel_list','comm');\"]"))

        Select(driver.find_element(By.ID, "con")).select_by_value("15344")
        driver.execute_script("arguments[0].click();", driver.find_element(By.XPATH,
                                                                           "//input[@onclick=\"adOption.addOption('sel_list1','con');\"]"))

        Select(driver.find_element(By.ID, "port")).select_by_value("A")
        driver.execute_script("arguments[0].click();", driver.find_element(By.XPATH,
                                                                           "//input[@onclick=\"adOption.addOption('sel_list4','port');\"]"))

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
                driver.execute_script("arguments[0].click();", execute_btn)

                wait.until(
                    EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'table-bordered')]//tbody")))

                soup = BeautifulSoup(driver.find_element(By.ID, "result").get_attribute("outerHTML"), "html.parser")
                rows = soup.select("tbody tr")

                found = 0
                for row in rows:
                    cols = [td.text.strip() for td in row.find_all("td")]
                    if len(cols) >= 7:
                        new_data.append({
                            "Period": period, "No": cols[0], "PC_Code": cols[1],
                            "Commodity": cols[2], "Country": cols[3], "Port": cols[4],
                            "Unit": cols[5], "Qty": cols[6],
                            "Value_INR": cols[7] if len(cols) > 7 else "",
                            "Value_USD": cols[8] if len(cols) > 8 else "",
                        })
                        found += 1
                print(f"[+] {period} tarandı. Eklenen kayıt: {found}")

            except TimeoutException:
                print(f"[!] DGCIS {period} ayında zaman aşımı oluştu, atlanıyor.")

    finally:
        driver.quit()

    if new_data:
        pd.DataFrame(existing_data + new_data).to_csv(csv_filename, index=False)
        print(f"[+] DGCIS güncellendi: {csv_filename}")


# ---------------------------------------------------------
# 5. GACC (ÇİN)
# ---------------------------------------------------------
def solve_gacc_captcha(driver) -> bool:
    print("[*] Captcha çözüm döngüsü başlatılıyor...")
    max_attempts = 5
    KALIBRASYON_PAYI = 15

    for attempt in range(max_attempts):
        print(f"\n[*] Deneme {attempt + 1}/{max_attempts} başlatılıyor...")
        slider_button = None
        timeout = 25 if attempt == 0 else 5
        end_time = time.time() + timeout

        while time.time() < end_time:
            driver.switch_to.default_content()
            try:
                el = driver.find_element(By.CSS_SELECTOR, "div.slider")
                if el.is_displayed():
                    slider_button = el
                    break
            except NoSuchElementException:
                pass

            driver.switch_to.default_content()
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            found = False
            for i in range(len(iframes)):
                try:
                    driver.switch_to.default_content()
                    iframe_list = driver.find_elements(By.TAG_NAME, "iframe")
                    if i >= len(iframe_list): break
                    driver.switch_to.frame(iframe_list[i])
                    try:
                        el = driver.find_element(By.CSS_SELECTOR, "div.slider")
                        if el.is_displayed():
                            slider_button = el
                            found = True
                            break
                    except NoSuchElementException:
                        nested_iframes = driver.find_elements(By.TAG_NAME, "iframe")
                        for j in range(len(nested_iframes)):
                            try:
                                driver.switch_to.frame(nested_iframes[j])
                                el = driver.find_element(By.CSS_SELECTOR, "div.slider")
                                if el.is_displayed():
                                    slider_button = el
                                    found = True
                                    break
                            except NoSuchElementException:
                                driver.switch_to.parent_frame()
                        if found: break
                except Exception:
                    pass
                if found: break

            if slider_button: break
            time.sleep(1)

        if not slider_button:
            if attempt == 0:
                print("[!] Captcha elementi bulunamadı (Zaman aşımı).")
                return False
            else:
                print("[+] Slider ekranda yok. Captcha başarıyla geçildi!")
                driver.switch_to.default_content()
                return True

        time.sleep(2.0)
        offset_x = 180
        try:
            canvas_data = driver.execute_script("""
                try {
                    let bgNode = document.querySelector('canvas:not([width="62"])') || document.querySelectorAll('img')[0];
                    let sliceNode = document.querySelector('canvas[width="62"], canvas.block') || document.querySelectorAll('img')[1];
                    if (!bgNode || !sliceNode) return null;
                    function getBase64(node) {
                        if (node.tagName.toLowerCase() === 'canvas') return node.toDataURL('image/png');
                        let cvs = document.createElement('canvas');
                        cvs.width = node.naturalWidth || node.width;
                        cvs.height = node.naturalHeight || node.height;
                        cvs.getContext('2d').drawImage(node, 0, 0);
                        return cvs.toDataURL('image/png');
                    }
                    let domWidth = bgNode.getBoundingClientRect().width;
                    let natWidth = bgNode.naturalWidth || bgNode.width || 300;
                    return [getBase64(bgNode), getBase64(sliceNode), natWidth, domWidth];
                } catch(e) { return null; }
            """)

            if canvas_data and canvas_data[0] and canvas_data[1]:
                bg_bytes = base64.b64decode(canvas_data[0].split(',')[1])
                slice_bytes = base64.b64decode(canvas_data[1].split(',')[1])
                bg_img = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_COLOR)
                slice_img_full = cv2.imdecode(np.frombuffer(slice_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

                slice_internal_offset = 0
                if slice_img_full is not None and slice_img_full.shape[2] == 4:
                    alpha_channel = slice_img_full[:, :, 3]
                    _, thresh = cv2.threshold(alpha_channel, 200, 255, cv2.THRESH_BINARY)
                    x, y, w, h = cv2.boundingRect(thresh)
                    if w > 0 and h > 0:
                        slice_img = slice_img_full[y:y + h, x:x + w]
                        slice_internal_offset = x
                    else:
                        slice_img = slice_img_full
                else:
                    slice_img = slice_img_full

                if bg_img is not None and slice_img is not None:
                    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
                    if slice_img.shape[2] == 4:
                        slice_gray = cv2.cvtColor(cv2.cvtColor(slice_img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
                    else:
                        slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)

                    bg_blur = cv2.GaussianBlur(bg_gray, (3, 3), 0)
                    slice_blur = cv2.GaussianBlur(slice_gray, (3, 3), 0)
                    bg_edge = cv2.Canny(bg_blur, 100, 200)
                    slice_edge = cv2.Canny(slice_blur, 100, 200)

                    result = cv2.matchTemplate(bg_edge, slice_edge, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(result)

                    nat_w = float(canvas_data[2])
                    dom_w = float(canvas_data[3])
                    scale = dom_w / nat_w if nat_w > 0 else 1.0

                    raw_calculated = int(max_loc[0] * scale)
                    adjusted_internal = int(slice_internal_offset * scale)
                    calculated_offset = raw_calculated - adjusted_internal + KALIBRASYON_PAYI

                    if calculated_offset > 30:
                        offset_x = calculated_offset
                        print(
                            f"[+] OpenCV Mesafe: {offset_x}px (Ham: {raw_calculated}, Kalibrasyon: +{KALIBRASYON_PAYI})")
                    else:
                        print(f"[!] Hatalı hesaplama ({calculated_offset}px). Varsayılan ({offset_x}px) kullanılacak.")
        except Exception as e:
            print(f"[!] OpenCV hatası ({e}), varsayılan mesafe kullanılacak.")

        try:
            print(f"[*] Sürükleme başlatılıyor ({offset_x}px)...")
            track = []
            moved = 0
            while moved < offset_x:
                step = random.randint(15, 30)
                if moved + step > offset_x: step = offset_x - moved
                track.append(step)
                moved += step

            action = ActionChains(driver)
            action.click_and_hold(slider_button)
            action.pause(random.uniform(0.2, 0.4))

            for step in track:
                action.move_by_offset(step, 0)
                action.pause(random.uniform(0.02, 0.05))

            action.move_by_offset(2, 0).pause(0.1).move_by_offset(-2, 0).pause(0.1)
            action.release().perform()
            print("[+] Sürükleme tamamlandı.")
            time.sleep(1.5)
        except Exception as e:
            print(f"[!] Sürükleme sırasında hata: {e}")

        try:
            clicked = driver.execute_script("""
                const elements = Array.from(document.querySelectorAll('button, div.btn, a, input'));
                const target = elements.find(el => {
                    const text = (el.innerText || el.value || "").trim().toLowerCase();
                    return text.includes('enquiry') && el.offsetParent !== null;
                });
                if (target) { target.click(); return true; } return false;
            """)
            if not clicked:
                enq_btn = driver.find_element(By.XPATH,
                                              "//[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'enquiry') or contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'enquiry')]")
                driver.execute_script("arguments[0].click();", enq_btn)
        except Exception:
            driver.switch_to.default_content()
            try:
                clicked_main = driver.execute_script("""
                    const elements = Array.from(document.querySelectorAll('button, div.btn, a, input'));
                    const target = elements.find(el => {
                        const text = (el.innerText || el.value || "").trim().toLowerCase();
                        return text.includes('enquiry') && el.offsetParent !== null;
                    });
                    if (target) { target.click(); return true; } return false;
                """)
                if not clicked_main:
                    enq_btn = driver.find_element(By.XPATH,
                                                  "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'enquiry') or contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'enquiry')]")
                    driver.execute_script("arguments[0].click();", enq_btn)
            except Exception:
                pass
        time.sleep(3.0)

    print("[!] Maksimum deneme sayısına ulaşıldı.")
    driver.switch_to.default_content()
    return False


def setup_gacc_driver():
    options = Options()
    options.add_argument("--headless")
    options.set_preference("dom.webdriver.enabled", False)
    options.set_preference('useAutomationExtension', False)
    options.set_preference("devtools.debugger.remote-enabled", False)
    options.set_preference("devtools.debugger.enabled", False)

    download_dir = os.path.join(RAW_DIR, "china")
    os.makedirs(download_dir, exist_ok=True)

    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.dir", download_dir)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk",
                           "application/csv, text/csv, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    driver = webdriver.Firefox(options=options)
    driver.execute_script("""
        const originalEval = window.eval;
        window.eval = function(text) {
            if (typeof text === 'string') text = text.replace(/debugger/g, '');
            return originalEval.apply(this, arguments);
        };
        const originalFunction = window.Function;
        window.Function = function(...args) {
            if (args.length > 0 && typeof args[args.length - 1] === 'string') {
                args[args.length - 1] = args[args.length - 1].replace(/debugger/g, '');
            }
            return originalFunction.apply(this, args);
        };
    """)
    return driver


def gacc_guncelle(years: list = None, start_month: str = "1", end_month: str = None):
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month
    safe_current_month = max(1, current_month - 1)

    if years is None:
        years = [str(y) for y in range(2018, current_year + 1)]

    download_dir = os.path.join(RAW_DIR, "china")
    driver = setup_gacc_driver()
    wait = WebDriverWait(driver, 40)

    try:
        print("[*] GACC sayfasına bağlanılıyor...")
        driver.get("http://stats.customs.gov.cn/indexEn")

        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(3)

        try:
            wait.until(EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME, "iframe")))
        except Exception:
            print("[i] Iframe doğrudan bulunamadı, ana DOM üzerinden aranıyor.")

        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio' and @value='usd']")))
        time.sleep(2)

        try:
            usd_radio = driver.find_element(By.XPATH, "//input[@type='radio' and @value='usd']")
            driver.execute_script("arguments[0].click();", usd_radio)
            time.sleep(1)
        except Exception as e:
            print(f"[!] USD seçimi başarısız: {e}")

        try:
            by_month_chk = driver.find_element(By.XPATH,
                                               "//input[@type='checkbox' and (contains(@id,'month') or contains(@name,'month'))] | //span[contains(text(),'By month')]/following-sibling::input")
            if not by_month_chk.is_selected():
                driver.execute_script("arguments[0].click();", by_month_chk)
        except Exception as e:
            print(f"[!] By month uyarısı: {e}")

        time.sleep(1)
        row_selects = driver.find_elements(By.XPATH, "//*[contains(text(),'Select:')]/following::select")
        if len(row_selects) > 0:
            comm_sel = Select(row_selects[0])
            try:
                comm_sel.select_by_value("1")
            except Exception:
                for opt in comm_sel.options:
                    if "commodity" in opt.text.lower():
                        comm_sel.select_by_visible_text(opt.text)
                        break
            driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", row_selects[0])

        time.sleep(1)
        text_inputs = driver.find_elements(By.XPATH, "//*[contains(text(),'Select:')]/following::input[@type='text']")
        if len(text_inputs) > 0:
            target_input = text_inputs[0]
            driver.execute_script("""
                arguments[0].removeAttribute('disabled');
                arguments[0].removeAttribute('readonly');
                arguments[0].value = '27090000';
            """, target_input)
            driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", target_input)
            driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", target_input)

        if len(row_selects) > 1:
            part_sel = Select(row_selects[1])
            try:
                part_sel.select_by_value("2")
            except Exception:
                for opt in part_sel.options:
                    if "partner" in opt.text.lower():
                        part_sel.select_by_visible_text(opt.text)
                        break
            driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", row_selects[1])
        time.sleep(1)

        for idx, year in enumerate(years):
            target_end_month = str(safe_current_month) if int(year) == current_year else (end_month or "12")
            target_start_month = str(start_month)

            print(
                f"\n[*] === {year} YILI İŞLENİYOR (Aylar: {target_start_month}-{target_end_month}) ({idx + 1}/{len(years)}) ===")

            target_file = os.path.join(download_dir, f"{year}_china.csv")
            if os.path.exists(target_file) and int(year) < current_year:
                try:
                    df_check = pd.read_csv(target_file, on_bad_lines='skip', encoding_errors='ignore')
                    if len(df_check) >= 12:
                        print(f"[+] {year}_china.csv tam, atlanıyor.")
                        continue
                except:
                    pass

            driver.switch_to.default_content()
            try:
                wait.until(EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME, "iframe")))
            except Exception:
                pass

            period_selects = driver.find_elements(By.XPATH, "//*[contains(text(),'Period')]/following::select")
            if len(period_selects) >= 3:
                y_select = Select(period_selects[0])
                try:
                    y_select.select_by_visible_text(year)
                except Exception:
                    y_select.select_by_value(year)
                driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));",
                                      period_selects[0])
                time.sleep(3)

                period_selects = driver.find_elements(By.XPATH, "//*[contains(text(),'Period')]/following::select")
                driver.execute_script(f"""
                    let select = arguments[0];
                    let target = '{target_start_month}';
                    for (let i = 0; i < select.options.length; i++) {{
                        let optVal = select.options[i].value; let optText = select.options[i].text.trim();
                        if (optVal === target || optVal.replace(/^0+/, '') === target || optText === target || optText.replace(/^0+/, '') === target) {{
                            select.value = optVal; select.dispatchEvent(new Event('change', {{ bubbles: true }})); break;
                        }}
                    }}
                """, period_selects[1])
                time.sleep(2)

                period_selects = driver.find_elements(By.XPATH, "//*[contains(text(),'Period')]/following::select")
                driver.execute_script(f"""
                    let select = arguments[0];
                    let target = '{target_end_month}';
                    for (let i = 0; i < select.options.length; i++) {{
                        let optVal = select.options[i].value; let optText = select.options[i].text.trim();
                        if (optVal === target || optVal.replace(/^0+/, '') === target || optText === target || optText.replace(/^0+/, '') === target) {{
                            select.value = optVal; select.dispatchEvent(new Event('change', {{ bubbles: true }})); break;
                        }}
                    }}
                """, period_selects[2])
                time.sleep(2)

            enquiry_btn = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                                 "//input[@type='button' and contains(@value,'Enquiry')] | //button[contains(.,'Enquiry')] | //a[contains(text(),'Enquiry')]")))
            driver.execute_script("arguments[0].click();", enquiry_btn)

            try:
                confirm_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,
                                                                                          "//button[contains(.,'Confirm')] | //span[contains(text(),'Confirm')] | //a[contains(text(),'Confirm')]")))
                driver.execute_script("arguments[0].click();", confirm_btn)
            except Exception:
                pass

            time.sleep(3)
            is_solved = solve_gacc_captcha(driver)
            if not is_solved:
                print(f"[!] Çin {year} yılı için Captcha aşılamadı, atlanıyor.")
                continue

            time.sleep(18)
            driver.switch_to.default_content()
            try:
                wait.until(EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME, "iframe")))
            except Exception:
                pass

            try:
                download_btn = wait.until(EC.presence_of_element_located((By.XPATH,
                                                                          "//*[contains(text(), 'Download')] | //a[contains(@class, 'download') or contains(., 'Download')]")))
                driver.execute_script("arguments[0].click();", download_btn)
                try:
                    download_confirm_btn = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH,
                                                                                                       "//button[contains(.,'Confirm')] | //span[contains(text(),'Confirm')] | //a[contains(text(),'Confirm')] | //div[contains(@class, 'button') and contains(., 'Confirm')]")))
                    driver.execute_script("arguments[0].click();", download_confirm_btn)
                except Exception:
                    driver.switch_to.default_content()
                    download_confirm_btn = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH,
                                                                                                       "//button[contains(.,'Confirm')] | //span[contains(text(),'Confirm')] | //a[contains(text(),'Confirm')] | //div[contains(@class, 'button') and contains(., 'Confirm')]")))
                    driver.execute_script("arguments[0].click();", download_confirm_btn)

                time.sleep(18)
                list_of_files = glob.glob(os.path.join(download_dir, '*'))
                if list_of_files:
                    valid_files = [f for f in list_of_files if not f.endswith('.part')]
                    if valid_files:
                        latest_file = max(valid_files, key=os.path.getctime)
                        new_file_name = os.path.join(download_dir, f"{year}_china.csv")
                        if os.path.exists(new_file_name): os.remove(new_file_name)
                        os.rename(latest_file, new_file_name)
                        print(f"[+] İndirme tamamlandı. Dosya adı: {new_file_name}")
            except Exception as e:
                print(f"[!] Download işleminde hata: {e}")

            if idx < len(years) - 1:
                driver.switch_to.default_content()
                try:
                    wait.until(EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME, "iframe")))
                except:
                    pass
                try:
                    back_btn = wait.until(EC.presence_of_element_located(
                        (By.XPATH, "//*[contains(text(), 'Back')] | //a[contains(., 'Back')]")))
                    driver.execute_script("arguments[0].click();", back_btn)
                    time.sleep(5)
                except Exception as e:
                    print(f"[!] 'Back' butonuna basılamadı, işlem sonlandırılıyor: {e}")
                    break

    except Exception as e:
        print(f"[!] Hata oluştu: {e}")
        traceback.print_exc()
    finally:
        driver.quit()


if __name__ == "__main__":
    jodi_guncelle()
    eia_guncelle()
    epdk_guncelle()
    dgcis_guncelle()
    gacc_guncelle()