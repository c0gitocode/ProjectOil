import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DIR, INDIA_V_VOL, INDIA_O_VOL, INDIA_P_VOL, INDIA_TOTAL_VOL


def global_grafik_ciz():
    master_path = os.path.join(PROCESSED_DIR, "GLOBAL_PETRODOLAR_VS_PETROYUAN.csv")
    output_img = os.path.join(PROCESSED_DIR, "petrodolar_vs_petroyuan_global.png")

    if not os.path.exists(master_path):
        print("[!] Küresel analiz dosyası bulunamadı.")
        return

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)

    # Şubat 2026 sonu ile kesin sınırlandırma
    df = df[(df.index >= "2018-01-01") & (df.index <= pd.to_datetime("2026-02-28"))]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    post_war_mask = (df.index >= pd.to_datetime("2022-02-01")) & (df.index <= pd.to_datetime("2026-02-28"))

    if 'GLOBAL_PETROYUAN_High_KBD' in df.columns and 'GLOBAL_PETROYUAN_Low_KBD' in df.columns:
        df_post = df[post_war_mask]
        plt.fill_between(df_post.index, df_post['GLOBAL_PETROYUAN_Low_KBD'], df_post['GLOBAL_PETROYUAN_High_KBD'],
                         color='#d95f02', alpha=0.2, label='Petroyuan ±%5 Hata Payı Bandı (2022-2026)')

    plt.plot(df.index, df['GLOBAL_PETROYUAN'], label='Global Petroyuan Bloğu (Baseline)',
             color='#d95f02', linewidth=2.5)
    plt.plot(df.index, df['GLOBAL_PETRODOLAR'],
             label='Global Petrodolar Bloğu (Baseline)', color='#2b5c8f',
             linewidth=2.5)

    plt.axvline(pd.to_datetime("2022-02-24"), color='red', linestyle=':', alpha=0.7,
                label='Ukrayna İşgali (Şubat 2022)')

    plt.title("Küresel Ham Petrol Ödeme Sistemleri: Petrodolar vs Petroyuan Hegemonyası (2018-2026 Şubat)", fontsize=14,
              fontweight='bold', pad=15)
    plt.ylabel("Günlük İthalat Hacmi (KBD)", fontsize=12)
    plt.xlabel("Tarih", fontsize=12)
    plt.xlim(pd.to_datetime("2018-01-01"), pd.to_datetime("2026-02-28"))

    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] Küresel Hegemonya Grafiği Kaydedildi: {output_img}")

    china_pay_orani_grafik_ciz()
    petroyuan_grafik_ciz()


def china_pay_orani_grafik_ciz():
    china_path = os.path.join(PROCESSED_DIR, "china_petrodolar_petroyuan_kbd.csv")
    output_img = os.path.join(PROCESSED_DIR, "china_yuan_dolar_oran_trend.png")

    if not os.path.exists(china_path):
        print("[!] Çin ödeme analizi dosyası bulunamadı.")
        return

    df = pd.read_csv(china_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Şubat 2026 sonu filtresi
    df = df[df['Date'] <= pd.to_datetime("2026-02-28")]
    df = df.sort_values('Date')

    total = df['China_Total_KBD']
    df['Yuan_Pct'] = (df['China_Petroyuan_KBD'] / total) * 100
    df['Dolar_Pct'] = (df['China_Petrodolar_KBD'] / total) * 100

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))

    plt.plot(df['Date'], df['Yuan_Pct'], label='Çin Petroyuan Payı (%)', color='#d95f02', linewidth=2.5)
    plt.plot(df['Date'], df['Dolar_Pct'], label='Çin Petrodolar Payı (%)', color='#2b5c8f', linewidth=2.5)

    plt.axvline(pd.to_datetime("2022-02-24"), color='red', linestyle=':', alpha=0.7, label='Ukrayna İşgali')

    plt.title("Çin'in Toplam Ham Petrol İthalatında Para Birimi Dağılımı (%) (2026 Şubat Sonu)", fontsize=14,
              fontweight='bold', pad=15)
    plt.ylabel("Pazar Payı Oranı (%)", fontsize=12)
    plt.xlabel("Tarih", fontsize=12)
    plt.xlim(df['Date'].min(), pd.to_datetime("2026-02-28"))
    plt.ylim(-5, 105)

    plt.legend(loc='center right', frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] Çin Yuan/Dolar Oran Grafiği Kaydedildi: {output_img}")


def petroyuan_grafik_ciz():
    india_path = os.path.join(PROCESSED_DIR, "hindistan_nihai_hacim_analizi_kbd.csv")
    output_img = os.path.join(PROCESSED_DIR, "hindistan_genel_hacim_odeme_trend_guncel.png")

    if not os.path.exists(india_path):
        print("[!] Hindistan analiz dosyası bulunamadı.")
        return

    df = pd.read_csv(india_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Şubat 2026 sonu sınırı
    df = df[(df['Date'] >= "2018-01-01") & (df['Date'] <= pd.to_datetime("2026-02-28"))]
    df = df.sort_values('Date')

    # Hindistan'ın Rusya ithalatı üzerinden alternatif (Yuan/Ruble) oranını doğrudan hesapla
    rusya_tot = df['Rusya_Ithalat_KBD']
    # Petroyuan doğrudan Rusya * alt_oran olduğu için, payı bulmak için Petroyuan / Rusya_Ithalat
    df['Yuan_Ruble_Pct'] = 0.0
    mask = rusya_tot > 0
    df.loc[mask, 'Yuan_Ruble_Pct'] = (df.loc[mask, 'Petroyuan_KBD'] / rusya_tot[mask]) * 100
    df['USD_Dirham_Pct'] = 100 - df['Yuan_Ruble_Pct']

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))

    plt.plot(df['Date'], df['USD_Dirham_Pct'], marker='o', linewidth=2.5, color="#2b5c8f",
             label="USD + Dirham (Rusya İthalatı İçinde)")
    plt.plot(df['Date'], df['Yuan_Ruble_Pct'], marker='s', linewidth=2.5, color="#d95f02",
             label="Yuan + Ruble (Alternatif)")

    plt.axvline(pd.to_datetime("2022-02-24"), color='red', linestyle=':', alpha=0.7, label='Ukrayna İşgali')

    plt.title("Hindistan'ın Rus Petrolü İthalatında Ödeme Yöntemlerinin Trendi (2026 Şubat Sonu)", fontsize=13,
              fontweight="bold", pad=20)
    plt.ylabel("Rusya İthalatı İçindeki Pay (%)", fontsize=12)
    plt.xlabel("Tarih", fontsize=12)
    plt.xlim(df['Date'].min(), pd.to_datetime("2026-02-28"))
    plt.ylim(-5, 110)

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=11, frameon=True)
    plt.tight_layout()

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] Hindistan Düzeltilmiş Trend Grafiği Kaydedildi: {output_img}")