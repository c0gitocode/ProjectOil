import os
import glob
import numpy as np
import pandas as pd
import warnings
from config import RAW_DIR, PROCESSED_DIR

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. TÜRKİYE / TÜİK & EPDK BİRLEŞTİRME
# ---------------------------------------------------------
def tuik_excel_temizle(dosya_yolu, veri_kolon_adi):
    df_raw = pd.read_excel(dosya_yolu, header=None)
    header_idx = 0
    for i, row in df_raw.iterrows():
        row_str = " ".join(str(val).upper() for val in row.values)
        if 'YIL' in row_str and 'AY' in row_str:
            header_idx = i
            break
    df = pd.read_excel(dosya_yolu, skiprows=header_idx)
    df = df.rename(columns={df.columns[0]: 'Yil', df.columns[1]: 'Ay', df.columns[-1]: 'Deger'})
    df['Yil'] = pd.to_numeric(df['Yil'].ffill(), errors='coerce')
    df['Ay'] = pd.to_numeric(df['Ay'], errors='coerce')
    df['Deger'] = pd.to_numeric(df['Deger'], errors='coerce')
    df = df.dropna(subset=['Ay', 'Deger'])
    df['Tarih'] = pd.to_datetime(df['Yil'].astype(int).astype(str) + "-" + df['Ay'].astype(int).astype(str) + "-01")
    indicator_col = df.columns[2]

    if any(keyword in str(indicator_col).lower() for keyword in ['ödeme', 'odeme', 'ruble', 'yuan']):
        df = df.pivot_table(index='Tarih', columns=indicator_col, values='Deger', aggfunc='sum').reset_index()
        new_cols = ['Tarih']
        for col in df.columns[1:]:
            if 'Ruble' in str(col):
                new_cols.append('RUB_USD')
            elif 'Yuan' in str(col):
                new_cols.append('CNY_USD')
            else:
                new_cols.append(f"Doviz_{col}")
        df.columns = new_cols
    else:
        df = df.groupby('Tarih')['Deger'].sum().reset_index()
        df.columns = ['Tarih', veri_kolon_adi]
    return df


def master_analiz_olustur():
    print("[*] Türkiye Master Analiz (TÜİK & EPDK) oluşturuluyor...")
    doviz_dosyalari = glob.glob(os.path.join(RAW_DIR, "*ruble-yuan*.xlsx"))
    if doviz_dosyalari:
        df_doviz = pd.concat([tuik_excel_temizle(f, "Doviz") for f in doviz_dosyalari]).groupby(
            'Tarih').sum().reset_index()
    else:
        df_doviz = pd.DataFrame(columns=['Tarih'])

    rusya_dosyalari = [f for f in glob.glob(os.path.join(RAW_DIR, "*rusya*.xlsx")) if "petrol" not in f.lower()]
    if rusya_dosyalari:
        df_rusya = pd.concat([tuik_excel_temizle(f, "Rusya_Toplam_USD") for f in rusya_dosyalari]).groupby(
            'Tarih').sum().reset_index()
    else:
        df_rusya = pd.DataFrame(columns=['Tarih'])

    enerji_dosyalari = glob.glob(os.path.join(RAW_DIR, "*fasil27*.xlsx"))
    if enerji_dosyalari:
        df_enerji = pd.concat([tuik_excel_temizle(f, "Enerji_Fasil27_USD") for f in enerji_dosyalari]).groupby(
            'Tarih').sum().reset_index()
    else:
        df_enerji = pd.DataFrame(columns=['Tarih'])

    epdk_path = os.path.join(PROCESSED_DIR, "tr_epdk_master.csv")
    if os.path.exists(epdk_path):
        df_epdk = pd.read_csv(epdk_path)
        df_epdk['Tarih'] = pd.to_datetime(df_epdk['Tarih'])
        miktar_col = next((c for c in df_epdk.columns if 'RUSYA' in c.upper() and 'MIKTAR' in c.upper()),
                          next((c for c in df_epdk.columns if 'RUSYA' in c.upper()), None))
        if miktar_col:
            df_epdk = df_epdk[['Tarih', miktar_col]].rename(columns={miktar_col: 'Rusya_Petrol_Miktar_EPDK'})
        else:
            df_epdk = pd.DataFrame(columns=['Tarih'])
    else:
        df_epdk = pd.DataFrame(columns=['Tarih'])

    df_master = df_doviz.merge(df_rusya, on='Tarih', how='outer') \
        .merge(df_enerji, on='Tarih', how='outer') \
        .merge(df_epdk, on='Tarih', how='outer')
    df_master = df_master.sort_values('Tarih').fillna(0)
    master_out_path = os.path.join(PROCESSED_DIR, "MASTER_ANALYSIS.csv")
    df_master.to_csv(master_out_path, index=False)
    print(f"[+] Master analiz verisi oluşturuldu: {master_out_path}")


# ---------------------------------------------------------
# 2. ÇİN / GÜMRÜK VERİSİ İŞLEME
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2. ÇİN / GÜMRÜK VERİSİ İŞLEME (TÜM ÜLKELER DAHİL)
# ---------------------------------------------------------
def process_china_oil_data():
    print("[*] Çin ham petrol ithalat verileri işleniyor...")
    input_dir = os.path.join(RAW_DIR, "china")
    file_pattern = os.path.join(input_dir, "*_china.csv")
    file_list = glob.glob(file_pattern)
    if not file_list:
        print("[!] Çin için ham veri (CSV) bulunamadı.")
        return False

    df_list = []
    for file in file_list:
        for enc in ['utf-8', 'utf-8-sig', 'gbk', 'latin1', 'utf-16']:
            try:
                df_list.append(pd.read_csv(file, encoding=enc))
                break
            except UnicodeDecodeError:
                continue

    if not df_list:
        print("[!] Dosyalar okunamadı (Encoding hatası).")
        return False

    df_raw = pd.concat(df_list, ignore_index=True)
    df_raw.columns = df_raw.columns.str.strip()
    numeric_cols = ['Quantity', 'Supplementary Quantity', 'US dollar']

    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(',', '', regex=False).str.strip(),
                                        errors='coerce').fillna(0.0)

    df_raw['Supplementary Quantity'] = df_raw['Supplementary Quantity'].astype(float)

    # 0 olan değerleri tonaj (Quantity) üzerinden varil karşılığına çevir
    mask = (df_raw['Supplementary Quantity'] == 0) | (df_raw['Supplementary Quantity'].isna())
    if 'Quantity' in df_raw.columns:
        df_raw.loc[mask, 'Supplementary Quantity'] = df_raw.loc[mask, 'Quantity'] * 0.00733

    # Eğer GACC ham verilerinde "World" adında zaten bir toplam satırı varsa mükerrer sayımı engelle
    df_raw = df_raw[df_raw['Trading partner'] != 'World']

    # Genel toplamı hesapla
    df_world = df_raw.groupby('Date of data')[numeric_cols].sum().reset_index()
    df_world['Trading partner'] = 'World'
    df_world['Commodity code'] = 27090000

    # Ham verideki tüm ülkeleri tut (df_raw) ve altına kendi hesapladığımız World satırını ekle
    df_final = pd.concat([df_raw, df_world], ignore_index=True).sort_values(
        by=['Date of data', 'Trading partner']).reset_index(drop=True)

    output_file = os.path.join(PROCESSED_DIR, "china_oil_imports_processed.csv")
    df_final.to_csv(output_file, index=False)
    print(f"[+] Çin verisi işlendi ve kaydedildi (Tüm ülkeler korundu): {output_file}")
    return True

# ---------------------------------------------------------
# 3. HİNDİSTAN / DGCIS & JODI BİRLEŞTİRME VE DUYARLILIK
# ---------------------------------------------------------
def get_india_ratios(date_obj, ratio_dict):
    if date_obj < pd.to_datetime("2021-04-01"):
        return 1.0, 0.0
    yr = str(date_obj.year)[-2:]
    half = "H1" if date_obj.month <= 6 else "H2"
    key = f"{yr}-{half}"
    if key in ratio_dict:
        return ratio_dict[key]["USD_Pct"], ratio_dict[key]["Alt_Pct"]
    else:
        if date_obj.year >= 2026:
            return ratio_dict["25-H2"]["USD_Pct"], ratio_dict["25-H2"]["Alt_Pct"]
        return 1.0, 0.0


def process_india_oil_data():
    print("[*] Hindistan ham petrol verileri (±%5 Duyarlılık Bandı ile) işleniyor...")
    jodi_file = os.path.join(PROCESSED_DIR, "jodi_avrupa_asya_petrodolar.csv")
    dgcis_file = os.path.join(PROCESSED_DIR, "russia_oil_ports_2018_2026.csv")
    output_csv = os.path.join(PROCESSED_DIR, "hindistan_nihai_hacim_analizi_kbd.csv")

    if not os.path.exists(jodi_file) or not os.path.exists(dgcis_file):
        print("[!] JODI veya DGCIS csv dosyası eksik.")
        return False

    india_quarterly_alt_rates = {
        (2022, 1): 0.024, (2022, 2): 0.080, (2022, 3): 0.108, (2022, 4): 0.150,
        (2023, 1): 0.182, (2023, 2): 0.242, (2023, 3): 0.182, (2023, 4): 0.142,
        (2024, 1): 0.221, (2024, 2): 0.261, (2024, 3): 0.281, (2024, 4): 0.301,
        (2025, 1): 0.293, (2025, 2): 0.313, (2025, 3): 0.322, (2025, 4): 0.342,
    }

    def get_india_ratios(date_obj):
        if date_obj < pd.to_datetime("2022-01-01"):
            return 1.0, 0.0
        year = date_obj.year
        quarter = (date_obj.month - 1) // 3 + 1
        q_key = (year, quarter)
        if q_key in india_quarterly_alt_rates:
            alt_pct = india_quarterly_alt_rates[q_key]
        elif year >= 2026:
            alt_pct = india_quarterly_alt_rates[(2025, 4)]
        else:
            alt_pct = 0.0
        return 1.0 - alt_pct, alt_pct

    df_jodi = pd.read_csv(jodi_file)
    df_jodi['TIME_PERIOD'] = pd.to_datetime(df_jodi['TIME_PERIOD'])
    ind_cols = [col for col in ['IN', 'IND'] if col in df_jodi.columns]

    df_jodi['Days_in_Month'] = df_jodi['TIME_PERIOD'].dt.daysinmonth
    df_jodi['JODI_Total_KBD'] = df_jodi[ind_cols].sum(axis=1) / df_jodi['Days_in_Month']
    df_jodi_in = df_jodi[['TIME_PERIOD', 'JODI_Total_KBD']].rename(columns={'TIME_PERIOD': 'Date'})

    df_dgcis = pd.read_csv(dgcis_file)
    df_dgcis['Date'] = pd.to_datetime(df_dgcis['Period'], format='%b-%Y', errors='coerce')
    df_dgcis['Qty_MT'] = pd.to_numeric(df_dgcis['Qty'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_dgcis['Days_in_Month'] = df_dgcis['Date'].dt.daysinmonth
    df_dgcis['Qty_KBD'] = (df_dgcis['Qty_MT'] * 7.33) / (1000 * df_dgcis['Days_in_Month'])
    df_dgcis_rus = df_dgcis.groupby('Date')['Qty_KBD'].sum().reset_index()
    df_dgcis_rus.columns = ['Date', 'DGCIS_Rus_KBD']

    df_merged = pd.merge(df_jodi_in, df_dgcis_rus, on='Date', how='left').fillna(0)
    df_merged.sort_values('Date', inplace=True)

    res_data = []
    sensitivity_margin = 0.05  # ±%5 Hata Payı

    for _, row in df_merged.iterrows():
        t_date = row['Date']
        j_tot = row['JODI_Total_KBD']
        d_rus = row['DGCIS_Rus_KBD']

        if t_date < pd.to_datetime("2022-01-01"):
            petrodolar, petroyuan = j_tot, 0.0
            rus_disi, rus = j_tot, 0.0
            py_high = py_low = 0.0
            pd_high = pd_low = j_tot
        else:
            usd_rate, alt_rate = get_india_ratios(t_date)
            rus = d_rus
            rus_disi = max(0, j_tot - rus)

            petrodolar = rus_disi + (rus * usd_rate)
            petroyuan = rus * alt_rate

            # ±%5 Duyarlılık Sınırları (0 ile 1 arasına sabitlenir)
            alt_rate_high = min(1.0, alt_rate + sensitivity_margin)
            alt_rate_low = max(0.0, alt_rate - sensitivity_margin)

            py_high = rus * alt_rate_high
            py_low = rus * alt_rate_low
            pd_high = rus_disi + (rus * (1.0 - alt_rate_low))
            pd_low = rus_disi + (rus * (1.0 - alt_rate_high))

        res_data.append({
            'Date': t_date,
            'Toplam_Ithalat_KBD': j_tot,
            'Rusya_Ithalat_KBD': rus,
            'Rusya_Disi_KBD': rus_disi,
            'Petrodolar_KBD': petrodolar,
            'Petrodolar_High_KBD': pd_high,
            'Petrodolar_Low_KBD': pd_low,
            'Petroyuan_KBD': petroyuan,
            'Petroyuan_High_KBD': py_high,
            'Petroyuan_Low_KBD': py_low
        })

    df_final = pd.DataFrame(res_data)
    df_final.to_csv(output_csv, index=False)
    print(f"[+] Hindistan duyarlılık analizi tamamlandı: {output_csv}")
    return True

# ---------------------------------------------------------
# 4. ÇİN ÖDEME DAĞILIMI (ENDONEZYA EKLENTİSİ & DİNAMİK ORANLAR)
# ---------------------------------------------------------
import os
import pandas as pd
import numpy as np


def process_china_payment_breakdown():
    print("[*] Çin ödeme birimi analizi (±%5 Duyarlılık Bandı ile) yapılıyor...")
    input_file = os.path.join(PROCESSED_DIR, "china_oil_imports_processed.csv")
    output_file = os.path.join(PROCESSED_DIR, "china_petrodolar_petroyuan_kbd.csv")

    if not os.path.exists(input_file):
        print("[!] Çin işlenmiş veri dosyası bulunamadı.")
        return False

    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date of data'].astype(str).str.replace('-', '').str[:6], format='%Y%m',
                                errors='coerce')
    df = df.dropna(subset=['Date'])

    quarterly_rates = {
        (2022, 1): {'russia': 0.150, 'saudi': 0.000, 'malaysia': 0.400, 'indonesia': 0.020, 'uae': 0.005,
                    'brazil': 0.005, 'oman': 0.000, 'iraq': 0.000},
        (2022, 2): {'russia': 0.300, 'saudi': 0.000, 'malaysia': 0.450, 'indonesia': 0.030, 'uae': 0.005,
                    'brazil': 0.010, 'oman': 0.000, 'iraq': 0.000},
        (2022, 3): {'russia': 0.450, 'saudi': 0.000, 'malaysia': 0.500, 'indonesia': 0.040, 'uae': 0.005,
                    'brazil': 0.015, 'oman': 0.000, 'iraq': 0.000},
        (2022, 4): {'russia': 0.600, 'saudi': 0.000, 'malaysia': 0.550, 'indonesia': 0.050, 'uae': 0.010,
                    'brazil': 0.020, 'oman': 0.005, 'iraq': 0.000},
        (2023, 1): {'russia': 0.700, 'saudi': 0.000, 'malaysia': 0.600, 'indonesia': 0.060, 'uae': 0.010,
                    'brazil': 0.030, 'oman': 0.005, 'iraq': 0.000},
        (2023, 2): {'russia': 0.780, 'saudi': 0.000, 'malaysia': 0.650, 'indonesia': 0.070, 'uae': 0.015,
                    'brazil': 0.100, 'oman': 0.010, 'iraq': 0.000},
        (2023, 3): {'russia': 0.850, 'saudi': 0.005, 'malaysia': 0.700, 'indonesia': 0.080, 'uae': 0.020,
                    'brazil': 0.150, 'oman': 0.015, 'iraq': 0.000},
        (2023, 4): {'russia': 0.900, 'saudi': 0.015, 'malaysia': 0.725, 'indonesia': 0.100, 'uae': 0.035,
                    'brazil': 0.200, 'oman': 0.015, 'iraq': 0.000},
        (2024, 1): {'russia': 0.930, 'saudi': 0.020, 'malaysia': 0.750, 'indonesia': 0.110, 'uae': 0.045,
                    'brazil': 0.250, 'oman': 0.020, 'iraq': 0.000},
        (2024, 2): {'russia': 0.950, 'saudi': 0.030, 'malaysia': 0.775, 'indonesia': 0.120, 'uae': 0.055,
                    'brazil': 0.300, 'oman': 0.020, 'iraq': 0.000},
        (2024, 3): {'russia': 0.970, 'saudi': 0.040, 'malaysia': 0.800, 'indonesia': 0.130, 'uae': 0.065,
                    'brazil': 0.350, 'oman': 0.025, 'iraq': 0.000},
        (2024, 4): {'russia': 0.980, 'saudi': 0.050, 'malaysia': 0.825, 'indonesia': 0.150, 'uae': 0.075,
                    'brazil': 0.400, 'oman': 0.030, 'iraq': 0.000},
        (2025, 1): {'russia': 0.990, 'saudi': 0.060, 'malaysia': 0.850, 'indonesia': 0.180, 'uae': 0.080,
                    'brazil': 0.450, 'oman': 0.035, 'iraq': 0.000},
        (2025, 2): {'russia': 0.990, 'saudi': 0.080, 'malaysia': 0.875, 'indonesia': 0.220, 'uae': 0.085,
                    'brazil': 0.480, 'oman': 0.040, 'iraq': 0.000},
        (2025, 3): {'russia': 0.991, 'saudi': 0.100, 'malaysia': 0.900, 'indonesia': 0.260, 'uae': 0.090,
                    'brazil': 0.500, 'oman': 0.045, 'iraq': 0.000},
        (2025, 4): {'russia': 0.991, 'saudi': 0.125, 'malaysia': 0.920, 'indonesia': 0.300, 'uae': 0.100,
                    'brazil': 0.520, 'oman': 0.050, 'iraq': 0.000},
    }

    other_country_rates = {
        'iran_venezuela': {2022: 0.80, 2023: 0.85, 2024: 0.90, 2025: 0.95, 2026: 1.00}
    }

    def get_rate(date_obj, country_group):
        if date_obj < pd.to_datetime("2022-02-01"):
            if country_group == 'iran_venezuela':
                return 0.15
            return 0.00
        year = date_obj.year
        quarter = (date_obj.month - 1) // 3 + 1
        q_key = (year, quarter)
        if q_key in quarterly_rates and country_group in quarterly_rates[q_key]:
            return quarterly_rates[q_key][country_group]
        elif year >= 2026 and country_group in quarterly_rates[(2025, 4)]:
            return quarterly_rates[(2025, 4)][country_group]
        y = min(2026, max(2022, year))
        if country_group in other_country_rates:
            return other_country_rates[country_group][y]
        return 0.00

    res_data = []
    sensitivity_margin = 0.05  # ±%5 Hata Payı

    for date, group in df.groupby('Date'):
        world_row = group[group['Trading partner'] == 'World']
        if world_row.empty:
            continue
        days = date.daysinmonth

        def get_country_kbd(partner_name):
            p_row = group[group['Trading partner'] == partner_name]
            if p_row.empty:
                return 0.0
            val_mt = p_row['Quantity'].values[0]
            if val_mt == 0 and 'Supplementary Quantity' in p_row.columns:
                val_mt = p_row['Supplementary Quantity'].values[0]
            return (val_mt * 7.33) / (1000000 * days)

        toplam_ithalat_kbd = get_country_kbd('World')
        if toplam_ithalat_kbd <= 0:
            toplam_ithalat_kbd = sum([get_country_kbd(p) for p in group['Trading partner'].unique()])

        countries = {
            'russia': get_country_kbd('Russia'),
            'iran_venezuela': get_country_kbd('Iran') + get_country_kbd('Venezuela'),
            'malaysia': get_country_kbd('Malaysia'),
            'indonesia': get_country_kbd('Indonesia'),
            'saudi': get_country_kbd('Saudi Arabia'),
            'uae': get_country_kbd('United Arab Emirates'),
            'iraq': get_country_kbd('Iraq'),
            'oman': get_country_kbd('Oman'),
            'brazil': get_country_kbd('Brazil')
        }

        # Baseline Petroyuan
        global_china_petroyuan = sum(countries[cg] * get_rate(date, cg) for cg in countries)

        # High & Low Senaryoları (±%5 oran sapması ve [0, 1] sınırlandırması)
        py_high = sum(countries[cg] * min(1.0, get_rate(date, cg) + sensitivity_margin) for cg in countries)
        py_low = sum(countries[cg] * max(0.0, get_rate(date, cg) - sensitivity_margin) for cg in countries)

        global_china_petrodolar = max(0, toplam_ithalat_kbd - global_china_petroyuan)
        pd_high = max(0, toplam_ithalat_kbd - py_low)
        pd_low = max(0, toplam_ithalat_kbd - py_high)

        res_data.append({
            'Date': date,
            'China_Total_KBD': round(toplam_ithalat_kbd, 2),
            'China_Petrodolar_KBD': round(global_china_petrodolar, 2),
            'China_Petrodolar_High_KBD': round(pd_high, 2),
            'China_Petrodolar_Low_KBD': round(pd_low, 2),
            'China_Petroyuan_KBD': round(global_china_petroyuan, 2),
            'China_Petroyuan_High_KBD': round(py_high, 2),
            'China_Petroyuan_Low_KBD': round(py_low, 2)
        })

    df_out = pd.DataFrame(res_data)
    df_out.to_csv(output_file, index=False)
    print(f"[+] Çin duyarlılık analizi tamamlandı: {output_file}")
    return True


def global_ikili_hegemonya_birlestir():
    print("[*] Küresel Hegemonya Analizi (Hata Payı Bantları Dahil) yapılandırılıyor...")
    path_jodi = os.path.join(PROCESSED_DIR, "jodi_avrupa_asya_petrodolar.csv")
    path_abd = os.path.join(PROCESSED_DIR, "saf_abd_petrodolar.csv")
    path_epdk = os.path.join(PROCESSED_DIR, "tr_epdk_master.csv")
    path_hindistan = os.path.join(PROCESSED_DIR, "hindistan_nihai_hacim_analizi_kbd.csv")
    path_china = os.path.join(PROCESSED_DIR, "china_petrodolar_petroyuan_kbd.csv")

    try:
        df_jodi = pd.read_csv(path_jodi, index_col=0, parse_dates=True)
        df_abd = pd.read_csv(path_abd, index_col=0, parse_dates=True)
        df_epdk = pd.read_csv(path_epdk)
        df_in = pd.read_csv(path_hindistan)
        df_cn = pd.read_csv(path_china)

        jodi_cols = [c for c in df_jodi.columns if c != 'Toplam_Avrupa_Asya_Petrodolar']
        df_jodi[jodi_cols] = df_jodi[jodi_cols].replace(0, np.nan).ffill().fillna(0)
        days_in_month = df_jodi.index.daysinmonth
        df_jodi[jodi_cols] = df_jodi[jodi_cols].div(days_in_month, axis=0)
        jodi_saf_ulkeler = [c for c in jodi_cols if c not in ['IN', 'IND', 'CN', 'CHN']]
        df_jodi['JODI_DIGER_KBD'] = df_jodi[jodi_saf_ulkeler].sum(axis=1)

        abd_col = 'ABD_Ithalat_KBD' if 'ABD_Ithalat_KBD' in df_abd.columns else df_abd.columns[0]
        t_col = 'Tarih' if 'Tarih' in df_epdk.columns else 'tarih'
        df_epdk[t_col] = pd.to_datetime(df_epdk[t_col])
        df_epdk.set_index(t_col, inplace=True)
        tr_col = 'TR_Toplam_Ithalat_KBD' if 'TR_Toplam_Ithalat_KBD' in df_epdk.columns else 'TR_BRUT_ITHALAT_KBD'
        if tr_col not in df_epdk.columns:
            df_epdk[tr_col] = df_epdk['TR_Saf_Petrodolar_KBD']

        df_in['Date'] = pd.to_datetime(df_in['Date'])
        df_in.set_index('Date', inplace=True)

        df_cn['Date'] = pd.to_datetime(df_cn['Date'])
        df_cn.set_index('Date', inplace=True)

        final_df = df_jodi[['JODI_DIGER_KBD']].join([
            df_abd[[abd_col]],
            df_epdk[[tr_col]],
            df_in[['Petrodolar_KBD', 'Petrodolar_High_KBD', 'Petrodolar_Low_KBD',
                   'Petroyuan_KBD', 'Petroyuan_High_KBD', 'Petroyuan_Low_KBD']],
            df_cn[['China_Petrodolar_KBD', 'China_Petrodolar_High_KBD', 'China_Petrodolar_Low_KBD',
                   'China_Petroyuan_KBD', 'China_Petroyuan_High_KBD', 'China_Petroyuan_Low_KBD']]
        ], how='inner')

        final_df.sort_index(inplace=True)

        petrodolar_col, petroyuan_col = [], []
        pd_high_col, pd_low_col, py_high_col, py_low_col = [], [], [], []

        for date, row in final_df.iterrows():
            jodi_diger = row['JODI_DIGER_KBD']
            abd = row[abd_col]
            tr = row[tr_col]

            cn_dolar = row['China_Petrodolar_KBD']
            cn_yuan = row['China_Petroyuan_KBD']
            cn_dolar_high = row['China_Petrodolar_High_KBD']
            cn_dolar_low = row['China_Petrodolar_Low_KBD']
            cn_yuan_high = row['China_Petroyuan_High_KBD']
            cn_yuan_low = row['China_Petroyuan_High_KBD']  # veya Low

            if date < pd.to_datetime("2021-04-01"):
                in_dolar, in_yuan = 0.0, 0.0
                in_dolar_high, in_dolar_low, in_yuan_high, in_yuan_low = 0.0, 0.0, 0.0, 0.0
            else:
                in_dolar = row['Petrodolar_KBD']
                in_yuan = row['Petroyuan_KBD']
                in_dolar_high = row['Petrodolar_High_KBD']
                in_dolar_low = row['Petrodolar_Low_KBD']
                in_yuan_high = row['Petroyuan_High_KBD']
                in_yuan_low = row['Petroyuan_Low_KBD']

            petrodolar_col.append(jodi_diger + abd + tr + cn_dolar + in_dolar)
            petroyuan_col.append(cn_yuan + in_yuan)

            py_high_col.append(cn_yuan_high + in_yuan_high)
            py_low_col.append(cn_yuan_low + in_yuan_low)
            pd_high_col.append(jodi_diger + abd + tr + cn_dolar_high + in_dolar_high)
            pd_low_col.append(jodi_diger + abd + tr + cn_dolar_low + in_dolar_low)

        final_df['GLOBAL_PETRODOLAR'] = petrodolar_col
        final_df['GLOBAL_PETROYUAN'] = petroyuan_col
        final_df['GLOBAL_PETRODOLAR_High_KBD'] = pd_high_col
        final_df['GLOBAL_PETRODOLAR_Low_KBD'] = pd_low_col
        final_df['GLOBAL_PETROYUAN_High_KBD'] = py_high_col
        final_df['GLOBAL_PETROYUAN_Low_KBD'] = py_low_col

        final_df = final_df[(final_df['GLOBAL_PETRODOLAR'] > 1000) & (final_df.index >= "2018-01-01")]

        output_master = os.path.join(PROCESSED_DIR, "GLOBAL_PETRODOLAR_VS_PETROYUAN.csv")
        final_df.to_csv(output_master)
        print(f"[+] Küresel Hegemonya Verisi (Hata Payı Bantlı) Kaydedildi: {output_master}")
        return True
    except Exception as e:
        print(f"[!] Hata: {e}")
        return False