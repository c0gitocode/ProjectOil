import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------
# 1. VERİ YÜKLEME VE 2026 ŞUBAT FİLTRESİ
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
DATA_PATH = BASE_DIR / "data" / "processed" / "GLOBAL_PETRODOLAR_VS_PETROYUAN.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Veri dosyası bulunamadı: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

# 2026 Şubat sonuna kadar filtreleme (Eksik verileri arındırma)
df = df[df[date_col] <= '2026-02-28'].copy()

# Milat: Şubat 2022
CUTOFF_DATE = pd.to_datetime('2022-02-01')
df['post_2022'] = (df[date_col] >= CUTOFF_DATE).astype(int)
df['trend'] = np.arange(len(df))

pre_df = df[df[date_col] < CUTOFF_DATE]
post_df = df[df[date_col] >= CUTOFF_DATE]

n_pre = len(pre_df)
n_post = len(post_df)


# ---------------------------------------------------------
# HİPOTEZ TESTLERİ FONKSİYONU
# ---------------------------------------------------------
def run_econometric_tests(target_col, series_name):
    y_all = df[target_col]
    y_pre = pre_df[target_col]
    y_post = post_df[target_col]

    # 1. Welch's t-test
    t_stat, p_val_t = stats.ttest_ind(y_pre, y_post, equal_var=False)

    # 2. Mann-Whitney U test
    u_stat, p_val_u = stats.mannwhitneyu(y_pre, y_post, alternative='two-sided')

    # 3. OLS with HAC (Newey-West) standard errors
    X = sm.add_constant(df[['post_2022', 'trend']])
    model_hac = sm.OLS(y_all, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

    # 4. Chow Test
    def chow_test(y, X_vars, split_idx):
        res_pooled = sm.OLS(y, X_vars).fit()
        rss_pooled = res_pooled.ssr

        res_1 = sm.OLS(y.iloc[:split_idx], X_vars.iloc[:split_idx]).fit()
        rss_1 = res_1.ssr

        res_2 = sm.OLS(y.iloc[split_idx:], X_vars.iloc[split_idx:]).fit()
        rss_2 = res_2.ssr

        k = X_vars.shape[1]
        N1 = split_idx
        N2 = len(y) - split_idx

        f_stat = ((rss_pooled - (rss_1 + rss_2)) / k) / ((rss_1 + rss_2) / (N1 + N2 - 2 * k))
        p_val = 1 - stats.f.cdf(f_stat, k, N1 + N2 - 2 * k)
        return f_stat, p_val

    split_point = len(pre_df)
    chow_f, chow_p = chow_test(y_all, X[['const', 'trend']], split_point)

    # RAPOR
    print("=" * 75)
    print(f" AMPİRİK DOĞRULAMA TEST RAPORU: {series_name}")
    print("=" * 75)
    print(f"Örneklem -> Öncesi: {n_pre} ay | Sonrası: {n_post} ay (Toplam: {len(df)} ay)")
    print(f"Ortalama -> Öncesi: {y_pre.mean():.2f} KBD | Sonrası: {y_post.mean():.2f} KBD\n")

    print("1. WELCH'S T-TESTİ (Ortalama Farkı)")
    print(f"   T-İstatistiği : {t_stat:.4f}")
    print(f"   P-Değeri      : {p_val_t:.4e} ({'Anlamlı Fark Var' if p_val_t < 0.05 else 'Anlamlı Fark Yok'})\n")

    print("2. MANN-WHITNEY U TESTİ (Dağılım Kayması)")
    print(f"   U-İstatistiği : {u_stat:.4f}")
    print(f"   P-Değeri      : {p_val_u:.4e} ({'Anlamlı Kayma Var' if p_val_u < 0.05 else 'Anlamlı Kayma Yok'}\n)")

    print("3. OLS REGRESYONU + HAC (Newey-West Düzeltmesi)")
    print(f"   Post-2022 Katsayısı (Etki) : {model_hac.params['post_2022']:.2f} KBD")
    print(f"   Katsayı P-Değeri (HAC)    : {model_hac.pvalues['post_2022']:.4e}")
    print(f"   R-Squared                 : {model_hac.rsquared:.4f}\n")

    print("4. CHOW TESTİ (Yapısal Kırılma - Şubat 2022)")
    print(f"   F-İstatistiği : {chow_f:.4f}")
    print(f"   P-Değeri      : {chow_p:.4e} ({'Kesin Yapısal Kırılma Var' if chow_p < 0.05 else 'Kırılma Yok'})")
    print("=" * 75 + "\n")


# İki seri için de testleri koştur
run_econometric_tests('GLOBAL_PETRODOLAR', 'GLOBAL PETRODOLAR')
run_econometric_tests('GLOBAL_PETROYUAN', 'GLOBAL PETROYUAN')