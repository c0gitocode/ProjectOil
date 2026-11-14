"""Ekonometrik test ve modelleme paketi.

Yöntem çerçevesi:
1. Birim kök & durağanlık:
   - Seviye serileri: ADF (H0: birim kök, sabit+trend 'ct') ve KPSS (H0: trend-durağan 'ct').
   - 1. fark serileri: ADF ve KPSS sabit terimle ('c'), çünkü doğrusal trendin farkı sabit sürüklenmeye dönüşür.
2. Kesintili zaman serisi (ITS) modelleri, Newey-West HAC standart hatalarıyla:
   - Seviye ITS:
       y_t = a + b*trend + c*post + d*post*(trend - t0) + ay kuklaları + e_t
   - 1. Fark ITS (seviye I(1) çıktığında sahte regresyonu önlemek için):
       dy_t = mu + c_d*d_post + d_d*post_lag + ay kuklaları + u_t
     Burada d_post = Delta(post_t) kırılma ayındaki anlık seviye sıçramasını (c),
     post_lag = post_{t-1} ise kırılma sonrası kalıcı aylık eğim/büyüme değişimini (d) ölçer.
   HAC gecikmesi Newey-West kuralıyla seçilir: floor(4 * (T / 100)^(2/9)).
3. Artık tanı testleri (OLS artıkları üzerinde):
   - Durbin-Watson, Breusch-Godfrey LM otokorelasyon testi, Breusch-Pagan değişen varyans testi.
4. Yapısal kırılma testleri:
   - Bilinen tarihli Chow testi (temel [sabit, trend] + yeterli gözlem varsa mevsimsel kontrollü Chow).
   - Quandt-Andrews sup-F ve sup-Wald (bilinmeyen kırılma tarihi, %15 trimming):
     Andrews (1993) Tablo 1 (p=2, pi_0=0.15) kritik değerleri: sup-Wald %10=10.01, %5=11.79, %1=15.51.
5. Duyarlılık bandı sağlamlık kontrolü:
   - ±%5 ödeme payı varsayım bandı (Low / Baseline / High) altında ITS katsayı kararlılığı.
6. İki yönlü sabit etkili (TWFE) Panel Farkların Farkı (DiD):
   - panel_long.csv üzerinde ülke (alpha_i) ve ay-yıl (lambda_t) sabit etkileri +
     bölge (ülke) bazında kümelenmiş standart hatalarla Treated x Post tahmini.
"""
from dataclasses import asdict, dataclass
import inspect
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .config import ASSUMPTION_FILES, BREAK_DATE, END_DATE, FILES, OUTPUTS_DIR, START_DATE
from .logger import get_logger, setup_logging
from .transform import load_shares, shares_for_dates

log = get_logger(__name__)

TARGETS = {
    "GLOBAL_PETRODOLAR": "Global Petrodolar (KBD)",
    "GLOBAL_PETROYUAN": "Global Petroyuan (KBD)",
    "PETROYUAN_SHARE_PCT": "Petroyuan Payı (%)",
}

ZERO_TARGETS = {
    "CN_Russia_Share_Pct": "Çin Rusya Ham Petrol Payı — GACC (%)",
    "CN_Strict_Share_Pct": "Çin Yaptırımlı Payı — Strict: Rusya+İran+Ven (%)",
    "CN_Broad_Share_Pct": "Çin Yaptırımlı Payı — Broad: +Malezya (%)",
    "TR_Russia_Share_Pct": "Türkiye Rusya Ham Petrol Payı — EPDK (%)",
    "CN_TR_Sanctioned_Strict_Share_Pct": "Küresel Kapsanan Pay — Çin+TR Strict (%)",
    "CN_TR_Sanctioned_Broad_Share_Pct": "Küresel Kapsanan Pay — Çin+TR Broad (%)",
    "IN_Russia_Share_Pct": "Hindistan Rusya Payı — DGCIS/JODI, 2021-04+ (%)",
    "Zero_Strict_Share_Pct": "Küresel Sıfır Varsayım Payı — Strict, 2021-04+ (%)",
    "Zero_Broad_Share_Pct": "Küresel Sıfır Varsayım Payı — Broad, 2021-04+ (%)",
}

# Andrews (1993), Econometrica 61(4), Table 1: p = 2 (const, trend), trimming pi_0 = 0.15
ANDREWS_1993_SUP_WALD_CV = {0.10: 10.01, 0.05: 11.79, 0.01: 15.51}


def _legacy_result_kwarg(fn) -> Dict[str, bool]:
    return {"result_object": False} if "result_object" in inspect.signature(fn).parameters else {}


def newey_west_lags(n: int) -> int:
    return max(1, int(np.floor(4 * (max(n, 1) / 100) ** (2 / 9))))


def _calendar_trend(index: pd.DatetimeIndex) -> np.ndarray:
    """Boşluk bırakılmış aylarda bile gerçek takvim mesafesini (ay cinsinden) korur."""
    idx = pd.DatetimeIndex(index)
    return ((idx.year - idx[0].year) * 12 + (idx.month - idx[0].month)).astype(float)


def _month_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    cats = pd.Categorical(idx.month, categories=list(range(1, 13)))
    months = pd.get_dummies(cats, prefix="m", drop_first=True, dtype=float)
    months.index = idx
    return months.loc[:, (months != 0).any(axis=0)]


def andrews_significance(sup_wald: float) -> str:
    if np.isnan(sup_wald):
        return "hesaplanamadı"
    if sup_wald >= ANDREWS_1993_SUP_WALD_CV[0.01]:
        return "p < 0.01 (***)"
    if sup_wald >= ANDREWS_1993_SUP_WALD_CV[0.05]:
        return "p < 0.05 (**)"
    if sup_wald >= ANDREWS_1993_SUP_WALD_CV[0.10]:
        return "p < 0.10 (*)"
    return "p >= 0.10 (anlamsız)"


# ---------------------------------------------------------
# 1. Birim kök & Durağanlık
# ---------------------------------------------------------
def unit_root_tests(y: pd.Series) -> pd.DataFrame:
    rows = []
    specs = (("seviye", y.dropna(), "ct"), ("1. fark", y.diff().dropna(), "c"))
    for label, s, reg in specs:
        if len(s) < 10 or s.nunique(dropna=True) <= 1 or np.isclose(s.std(ddof=0), 0.0):
            rows.append(
                {
                    "seri": label,
                    "regresyon": reg,
                    "ADF_stat": np.nan,
                    "ADF_p": np.nan,
                    "KPSS_stat": np.nan,
                    "KPSS_p": np.nan,
                }
            )
            continue
        adf_stat, adf_p, *_ = adfuller(s, regression=reg, autolag="AIC", **_legacy_result_kwarg(adfuller))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)  # p-değeri tablo sınırında kırpılır
            kpss_stat, kpss_p, *_ = kpss(s, regression=reg, nlags="auto", **_legacy_result_kwarg(kpss))
        rows.append(
            {
                "seri": label,
                "regresyon": reg,
                "ADF_stat": float(adf_stat),
                "ADF_p": float(adf_p),
                "KPSS_stat": float(kpss_stat),
                "KPSS_p": float(kpss_p),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. ITS Modelleri (Seviye + 1. Fark) ve Artık Tanıları
# ---------------------------------------------------------
def design_matrix(index: pd.DatetimeIndex, break_date: str = BREAK_DATE, seasonal: bool = True) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    t = _calendar_trend(idx)
    bd = pd.Timestamp(break_date)
    post = (idx >= bd).astype(float)
    t0 = float((bd.year - idx[0].year) * 12 + (bd.month - idx[0].month))
    X = pd.DataFrame({"const": 1.0, "trend": t, "post": post, "post_trend": post * np.maximum(0.0, t - t0)}, index=idx)
    if seasonal:
        X = X.join(_month_dummies(idx))
    return X


def diff_design_matrix(
    level_index: pd.DatetimeIndex,
    diff_index: pd.DatetimeIndex,
    break_date: str = BREAK_DATE,
    seasonal: bool = True,
) -> pd.DataFrame:
    """Seviye ITS denkleminin tam 1. fark tasarımı:

    Delta(c*post_t + d*post_t*(t-t0)) = c*d_post_t + d*post_{t-1}.
    Böylece `d_post` kırılma ayındaki anlık seviye sıçramasını, `post_lag` kalıcı eğim değişimini verir.
    """
    idx = pd.DatetimeIndex(level_index)
    bd = pd.Timestamp(break_date)
    post = pd.Series((idx >= bd).astype(float), index=idx)
    d_post = post.diff().reindex(diff_index).fillna(0.0)
    post_lag = post.shift(1).reindex(diff_index).fillna(0.0)

    X = pd.DataFrame({"const": 1.0, "d_post": d_post.to_numpy(), "post_lag": post_lag.to_numpy()}, index=diff_index)
    if seasonal:
        X = X.join(_month_dummies(pd.DatetimeIndex(diff_index)))
    return X


def its_regression(y: pd.Series, break_date: str = BREAK_DATE, seasonal: bool = True):
    y = y.dropna()
    X = design_matrix(y.index, break_date, seasonal)
    lags = newey_west_lags(len(y))
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags}), lags


def its_diff_regression(y: pd.Series, break_date: str = BREAK_DATE, seasonal: bool = True):
    y_clean = y.dropna()
    dy = y_clean.diff().dropna()
    X = diff_design_matrix(y_clean.index, dy.index, break_date, seasonal)
    lags = newey_west_lags(len(dy))
    return sm.OLS(dy, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags}), lags


def residual_diagnostics(y: pd.Series, X: pd.DataFrame, bg_lags: Optional[int] = None) -> Dict[str, float]:
    ols = sm.OLS(y, X).fit()
    dw = float(durbin_watson(ols.resid)) if len(y) > 1 else float("nan")

    max_bg = max(1, len(y) - X.shape[1] - 2)
    lags = min(bg_lags or newey_west_lags(len(y)), max_bg)
    if lags >= 1 and len(y) > X.shape[1] + lags + 1:
        bg_stat, bg_p, _, _ = acorr_breusch_godfrey(ols, nlags=lags, **_legacy_result_kwarg(acorr_breusch_godfrey))
    else:
        bg_stat, bg_p = float("nan"), float("nan")

    if len(y) > X.shape[1] + 1 and ols.model.exog.shape[1] >= 2:
        bp_stat, bp_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
    else:
        bp_stat, bp_p = float("nan"), float("nan")

    return {
        "dw_stat": float(dw),
        "bg_lags": int(lags),
        "bg_stat": float(bg_stat),
        "bg_p": float(bg_p),
        "bp_stat": float(bp_stat),
        "bp_p": float(bp_p),
    }


# ---------------------------------------------------------
# 3. Yapısal Kırılma Testleri (Chow & Quandt-Andrews)
# ---------------------------------------------------------
def chow_test(y: pd.Series, X: pd.DataFrame, split: int) -> Tuple[float, float]:
    k, n = X.shape[1], len(y)
    if split <= k or (n - split) <= k:
        return float("nan"), float("nan")
    X1, X2 = X.iloc[:split], X.iloc[split:]
    if np.linalg.matrix_rank(X1) < k or np.linalg.matrix_rank(X2) < k:
        return float("nan"), float("nan")
    rss_p = sm.OLS(y, X).fit().ssr
    rss_u = sm.OLS(y.iloc[:split], X1).fit().ssr + sm.OLS(y.iloc[split:], X2).fit().ssr
    if rss_u <= 0 or n - 2 * k <= 0:
        return float("nan"), float("nan")
    f = ((rss_p - rss_u) / k) / (rss_u / (n - 2 * k))
    return float(f), float(stats.f.sf(f, k, n - 2 * k))


def quandt_andrews(y: pd.Series, trim: float = 0.15) -> Tuple[pd.Timestamp, float, float, pd.Series]:
    trend = pd.Series(_calendar_trend(y.index), index=y.index, name="trend")
    X = sm.add_constant(trend)
    k = X.shape[1]
    lo = max(k + 1, int(np.floor(len(y) * trim)))
    hi = min(len(y) - k - 1, int(np.ceil(len(y) * (1 - trim))))
    f_map = {}
    for i in range(lo, hi):
        f_val, _ = chow_test(y, X, i)
        if np.isfinite(f_val):
            f_map[y.index[i]] = f_val
    if not f_map:
        return y.index[len(y) // 2], float("nan"), float("nan"), pd.Series(dtype=float)
    f_stats = pd.Series(f_map)
    sup_f = float(f_stats.max())
    sup_wald = float(k * sup_f)  # p=2 için sup-Wald = 2 * sup-F
    return f_stats.idxmax(), sup_f, sup_wald, f_stats


# ---------------------------------------------------------
# 4. Tek Seri Analiz & Rapor
# ---------------------------------------------------------
@dataclass
class Summary:
    seri: str
    n_pre: int
    n_post: int
    mean_pre: float
    mean_post: float
    its_level_shift: float
    its_level_p: float
    its_slope_change: float
    its_slope_p: float
    its_r2: float
    diff_level_shift: float
    diff_level_p: float
    diff_slope_change: float
    diff_slope_p: float
    hac_lags: int
    dw_stat: float
    bg_p: float
    bp_p: float
    chow_F: float
    chow_p: float
    chow_seasonal_F: float
    chow_seasonal_p: float
    qa_break: str
    qa_supF: float
    qa_supWald: float
    qa_sig: str
    adf_level_p: float
    kpss_level_p: float
    adf_diff_p: float
    kpss_diff_p: float


def _can_fit_its(X: pd.DataFrame, key_cols: Tuple[str, ...]) -> bool:
    if len(X) <= X.shape[1] + 1:
        return False
    return all(c in X.columns and not np.isclose(X[c].std(ddof=0), 0.0) for c in key_cols)


def analyze(y: pd.Series, name: str, break_date: str = BREAK_DATE) -> Tuple[Summary, List[str]]:
    y = y.dropna().sort_index()
    bd = pd.Timestamp(break_date)
    pre, post = y[y.index < bd], y[y.index >= bd]

    ur = unit_root_tests(y)
    X_level = design_matrix(y.index, break_date, seasonal=True)
    if len(pre) >= 2 and len(post) >= 2 and _can_fit_its(X_level, ("post", "post_trend")):
        model, lags = its_regression(y, break_date, seasonal=True)
        diag = residual_diagnostics(y, X_level, bg_lags=lags)
        its_level_shift = float(model.params["post"])
        its_level_p = float(model.pvalues["post"])
        its_slope_change = float(model.params["post_trend"])
        its_slope_p = float(model.pvalues["post_trend"])
        its_r2 = float(model.rsquared)
    else:
        log.warning("%s: Seviye ITS için yetersiz örneklem (öncesi=%d, sonrası=%d, X_k=%d).", name, len(pre), len(post), X_level.shape[1])
        lags = 0
        diag = {"dw_stat": float("nan"), "bg_lags": 0, "bg_stat": float("nan"), "bg_p": float("nan"), "bp_stat": float("nan"), "bp_p": float("nan")}
        its_level_shift = its_level_p = its_slope_change = its_slope_p = its_r2 = float("nan")

    dy = y.diff().dropna()
    X_diff = diff_design_matrix(y.index, dy.index, break_date, seasonal=True) if not dy.empty else pd.DataFrame()
    if len(pre) >= 2 and len(post) >= 2 and not X_diff.empty and _can_fit_its(X_diff, ("d_post", "post_lag")):
        diff_model, diff_lags = its_diff_regression(y, break_date, seasonal=True)
        diff_level_shift = float(diff_model.params["d_post"])
        diff_level_p = float(diff_model.pvalues["d_post"])
        diff_slope_change = float(diff_model.params["post_lag"])
        diff_slope_p = float(diff_model.pvalues["post_lag"])
    else:
        diff_lags = 0
        diff_level_shift = diff_level_p = diff_slope_change = diff_slope_p = float("nan")

    X_base = sm.add_constant(pd.Series(_calendar_trend(y.index), index=y.index, name="trend"))
    chow_f, chow_p = chow_test(y, X_base, len(pre))
    chow_s_f, chow_s_p = chow_test(y, X_base.join(_month_dummies(y.index)), len(pre))

    qa_date, qa_f, qa_w, _ = quandt_andrews(y)
    qa_sig = andrews_significance(qa_w)

    if len(pre) >= 2 and len(post) >= 2:
        t_stat, t_p = stats.ttest_ind(pre, post, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(pre, post, alternative="two-sided")
    else:
        t_stat = t_p = u_stat = u_p = float("nan")

    mean_pre = float(pre.mean()) if not pre.empty else float("nan")
    mean_post = float(post.mean()) if not post.empty else float("nan")

    s = Summary(
        seri=name,
        n_pre=len(pre),
        n_post=len(post),
        mean_pre=mean_pre,
        mean_post=mean_post,
        its_level_shift=its_level_shift,
        its_level_p=its_level_p,
        its_slope_change=its_slope_change,
        its_slope_p=its_slope_p,
        its_r2=its_r2,
        diff_level_shift=diff_level_shift,
        diff_level_p=diff_level_p,
        diff_slope_change=diff_slope_change,
        diff_slope_p=diff_slope_p,
        hac_lags=lags,
        dw_stat=diag["dw_stat"],
        bg_p=diag["bg_p"],
        bp_p=diag["bp_p"],
        chow_F=chow_f,
        chow_p=chow_p,
        chow_seasonal_F=chow_s_f,
        chow_seasonal_p=chow_s_p,
        qa_break=qa_date.strftime("%Y-%m"),
        qa_supF=qa_f,
        qa_supWald=qa_w,
        qa_sig=qa_sig,
        adf_level_p=float(ur.loc[0, "ADF_p"]),
        kpss_level_p=float(ur.loc[0, "KPSS_p"]),
        adf_diff_p=float(ur.loc[1, "ADF_p"]),
        kpss_diff_p=float(ur.loc[1, "KPSS_p"]),
    )

    sig = lambda p: "anlamlı" if np.isfinite(p) and p < 0.05 else "anlamsız"  # noqa: E731
    lines = [
        "=" * 82,
        f" {name}",
        "=" * 82,
        f"Örneklem: {y.index.min():%Y-%m} – {y.index.max():%Y-%m} | Öncesi {len(pre)} ay, sonrası {len(post)} ay",
        f"Ortalama: öncesi {mean_pre:.2f} | sonrası {mean_post:.2f}",
        "",
        "1) Birim kök & durağanlık testleri (seviye='ct', 1. fark='c')",
        ur.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        f"2) Seviye ITS regresyonu + HAC (Newey-West, {lags} gecikme, ay kuklaları dahil)",
        f"   Seviye kayması (post)      : {s.its_level_shift:10.3f}  p={s.its_level_p:.4f} ({sig(s.its_level_p)})",
        f"   Eğim değişimi (post_trend) : {s.its_slope_change:10.3f}  p={s.its_slope_p:.4f} ({sig(s.its_slope_p)})",
        f"   R²                         : {s.its_r2:.4f}",
        f"   Artık tanıları: DW={s.dw_stat:.3f} | BG({diag['bg_lags']}) p={s.bg_p:.4g} | BP p={s.bp_p:.4g}",
        "",
        f"3) 1. Fark ITS regresyonu + HAC (Newey-West, {diff_lags} gecikme, I(1) kontrolü)",
        f"   Anlık kırılma sıçraması (d_post) : {s.diff_level_shift:10.3f}  p={s.diff_level_p:.4f} ({sig(s.diff_level_p)})",
        f"   Kalıcı büyüme farkı (post_lag)   : {s.diff_slope_change:10.3f}  p={s.diff_slope_p:.4f} ({sig(s.diff_slope_p)})",
        "",
        f"4) Chow testi ({bd:%Y-%m}): temel F={chow_f:.3f} p={chow_p:.4g} ({sig(chow_p)}) | "
        f"mevsimsel F={chow_s_f:.3f} p={chow_s_p:.4g} ({sig(chow_s_p)})",
        f"5) Quandt-Andrews (%15 trimming, p=2): en olası kırılma {s.qa_break}, "
        f"sup-F={qa_f:.3f}, sup-Wald={qa_w:.3f} -> {qa_sig}",
        "   [Andrews (1993) Tablo 1 sup-Wald kritik değerleri: %10=10.01, %5=11.79, %1=15.51]",
        "",
        "6) Betimsel (otokorelasyonu yok sayar)",
        f"   Welch t={t_stat:.3f} p={t_p:.4g} | Mann-Whitney U={u_stat:.1f} p={u_p:.4g}",
        "",
    ]
    if np.isfinite(s.adf_level_p) and np.isfinite(s.kpss_level_p) and s.adf_level_p > 0.05 and s.kpss_level_p < 0.05:
        lines.insert(
            -1,
            "   [Not] Seviye serisi I(1) görünüyor; çıkarımda Bölüm 3 (1. Fark ITS) ve Bölüm 7 (Panel FE DiD) esas alınmalıdır.",
        )
    return s, lines


# ---------------------------------------------------------
# 5. Duyarlılık Bandı Sağlamlık Kontrolü (±%5)
# ---------------------------------------------------------
def sensitivity_robustness(df: pd.DataFrame, break_date: str = BREAK_DATE) -> pd.DataFrame:
    df = df.dropna(subset=["GLOBAL_PETROYUAN", "GLOBAL_PETRODOLAR", "GLOBAL_TOTAL_KBD"]).sort_index()
    if df.empty:
        return pd.DataFrame()

    scenarios: List[Tuple[str, str, pd.Series]] = []
    if "GLOBAL_PETROYUAN_Low_KBD" in df.columns and "GLOBAL_PETROYUAN_High_KBD" in df.columns:
        scenarios.extend(
            [
                ("Global Petroyuan (KBD)", "Low (-%5)", df["GLOBAL_PETROYUAN_Low_KBD"]),
                ("Global Petroyuan (KBD)", "Baseline", df["GLOBAL_PETROYUAN"]),
                ("Global Petroyuan (KBD)", "High (+%5)", df["GLOBAL_PETROYUAN_High_KBD"]),
            ]
        )
        total = df["GLOBAL_TOTAL_KBD"].replace(0, np.nan)
        scenarios.extend(
            [
                ("Petroyuan Payı (%)", "Low (-%5)", df["GLOBAL_PETROYUAN_Low_KBD"] / total * 100.0),
                ("Petroyuan Payı (%)", "Baseline", df["PETROYUAN_SHARE_PCT"]),
                ("Petroyuan Payı (%)", "High (+%5)", df["GLOBAL_PETROYUAN_High_KBD"] / total * 100.0),
            ]
        )
    if "GLOBAL_PETRODOLAR_Low_KBD" in df.columns and "GLOBAL_PETRODOLAR_High_KBD" in df.columns:
        scenarios.extend(
            [
                ("Global Petrodolar (KBD)", "Low (+%5 py)", df["GLOBAL_PETRODOLAR_Low_KBD"]),
                ("Global Petrodolar (KBD)", "Baseline", df["GLOBAL_PETRODOLAR"]),
                ("Global Petrodolar (KBD)", "High (-%5 py)", df["GLOBAL_PETRODOLAR_High_KBD"]),
            ]
        )

    bd = pd.Timestamp(break_date)
    rows = []
    for target_name, scen_name, series in scenarios:
        s = series.dropna()
        if len(s) < 18 or (s.index < bd).sum() < 2 or (s.index >= bd).sum() < 2:
            continue
        lvl, _ = its_regression(s, break_date, seasonal=True)
        diff, _ = its_diff_regression(s, break_date, seasonal=True)
        rows.append(
            {
                "seri": target_name,
                "senaryo": scen_name,
                "its_level_shift": float(lvl.params["post"]),
                "its_level_p": float(lvl.pvalues["post"]),
                "its_slope_change": float(lvl.params["post_trend"]),
                "its_slope_p": float(lvl.pvalues["post_trend"]),
                "diff_level_shift": float(diff.params["d_post"]),
                "diff_level_p": float(diff.pvalues["d_post"]),
                "diff_slope_change": float(diff.params["post_lag"]),
                "diff_slope_p": float(diff.pvalues["post_lag"]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 6. İki Yönlü Sabit Etkili Panel DiD (TWFE + Kümelenmiş SE)
# ---------------------------------------------------------
def panel_fe_did(
    panel_df: pd.DataFrame,
    break_date: str = BREAK_DATE,
    treated_regions: Iterable[str] = ("CN", "IN"),
) -> pd.DataFrame:
    df_p = panel_df.copy()
    df_p["Date"] = pd.to_datetime(df_p["Date"])
    if "total_kbd" not in df_p.columns:
        df_p["total_kbd"] = df_p["petrodollar_kbd"] + df_p["petroyuan_kbd"]
    if "petroyuan_share_pct" not in df_p.columns:
        df_p["petroyuan_share_pct"] = np.where(
            df_p["total_kbd"] > 0, df_p["petroyuan_kbd"] / df_p["total_kbd"] * 100.0, np.nan
        )

    bd = pd.Timestamp(break_date)
    treated_set = set(treated_regions)
    df_p["treated"] = df_p["region"].isin(treated_set).astype(float)
    df_p["post"] = (df_p["Date"] >= bd).astype(float)
    df_p["treated_post"] = df_p["treated"] * df_p["post"]

    rows = []
    for col, label in (
        ("petroyuan_kbd", "Ülke Petroyuan Hacmi (KBD)"),
        ("petroyuan_share_pct", "Ülke Petroyuan Payı (%)"),
    ):
        sub = df_p.dropna(subset=[col, "region", "Date"]).copy()
        n_regions = sub["region"].nunique()
        n_months = sub["Date"].nunique()
        if n_regions < 2 or n_months < 2 or sub["treated_post"].nunique() < 2:
            continue

        reg_d = pd.get_dummies(sub["region"], prefix="reg", drop_first=True, dtype=float)
        time_d = pd.get_dummies(sub["Date"].dt.strftime("%Y-%m"), prefix="t", drop_first=True, dtype=float)
        X_fe = pd.concat([pd.Series(1.0, index=sub.index, name="const"), reg_d, time_d], axis=1)

        y_vec = sub[col].astype(float)
        x_vec = sub["treated_post"].astype(float)
        # Frisch-Waugh-Lovell (FWL): Sabit etkileri (ulke + ay-yil) ortogonalize ederek tam
        # katsayiyi korur, G << K kukla kaynakli nuans matris sayisal hatalarini onler.
        y_res = sm.OLS(y_vec, X_fe).fit().resid
        x_res = sm.OLS(x_vec, X_fe).fit().resid
        if np.isclose(x_res.std(ddof=0), 0.0):
            continue

        fwl = sm.OLS(y_res, x_res.to_frame("treated_post")).fit(
            cov_type="cluster", cov_kwds={"groups": sub["region"]}
        )
        n_treated_post = int(sub.loc[sub["treated_post"] > 0, "region"].nunique())
        n_obs = len(sub)
        k_full = int(np.linalg.matrix_rank(X_fe)) + 1
        dof_scale = float(np.sqrt((n_obs - 1) / max(1, n_obs - k_full)))
        coef = float(fwl.params["treated_post"])
        if n_treated_post < 2:
            log.warning(
                "Panel DiD (%s): kırılma sonrası verisi olan yalnızca %d treated bölge var; "
                "tek treated kümede bölge-kümeli SE güvenilmez olduğu için NaN bırakıldı.",
                label,
                n_treated_post,
            )
            se_cluster = t_stat = p_val = float("nan")
        else:
            se_cluster = float(fwl.bse["treated_post"] * dof_scale)
            t_stat = float(coef / se_cluster) if se_cluster > 0 else float("nan")
            p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n_regions - 1)) if np.isfinite(t_stat) else float("nan")
        overall_r2 = float(sm.OLS(y_vec, pd.concat([sub[["treated_post"]], X_fe], axis=1)).fit().rsquared)

        rows.append(
            {
                "bagimli_degisken": label,
                "did_katsayisi": coef,
                "se_kumelenmis": se_cluster,
                "t_ist": t_stat,
                "p_degeri": p_val,
                "r2_toplam": overall_r2,
                "r2_within": float(fwl.rsquared),
                "n_gozlem": int(n_obs),
                "n_ulke": int(n_regions),
                "n_ay": int(n_months),
            }
        )
    return pd.DataFrame(rows)


def _monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp()
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def constant_share_robustness(
    global_df: pd.DataFrame,
    china_pay: Optional[pd.DataFrame] = None,
    india_df: Optional[pd.DataFrame] = None,
    china_shares: Optional[pd.DataFrame] = None,
    break_date: str = BREAK_DATE,
) -> pd.DataFrame:
    """Merdivensiz / sabit-oran ve Malezya'sız Strict sağlamlık testi.

    Varsayım merdiveninin (2022-2025 arası yükselen alt_share) sonuçları tek başına yaratıp
    yaratmadığını sınamak için:
      1) Strict (Malezya/Suudi/Brezilya hariç, yalnızca Rusya + İran + Venezuela)
      2) Sabit Oran 2022+ (Merdivensiz %50): Savaş sonrası tüm aylarda alt_share=%50 sabitlenir;
         böylece kırılma sonrası büyüme tamamen gerçek fiziksel ithalat hacminden gelir.
    """
    if china_pay is None or india_df is None or global_df.empty or "GLOBAL_TOTAL_KBD" not in global_df.columns:
        return pd.DataFrame()
    req_cn = {"CN_russia_KBD", "CN_iran_venezuela_KBD"}
    req_in = {"Rusya_Ithalat_KBD", "Toplam_Ithalat_KBD", "Petroyuan_KBD"}
    if not req_cn <= set(china_pay.columns) or not req_in <= set(india_df.columns):
        return pd.DataFrame()

    if china_shares is None:
        if not ASSUMPTION_FILES["china"].exists():
            return pd.DataFrame()
        china_shares = load_shares(ASSUMPTION_FILES["china"])

    g_m = _monthly_index(global_df).dropna(subset=["GLOBAL_TOTAL_KBD"])
    cn_m = _monthly_index(china_pay).reindex(g_m.index)
    in_m = _monthly_index(india_df).reindex(g_m.index)
    if g_m.empty:
        return pd.DataFrame()

    sh_base = shares_for_dates(china_shares, g_m.index)
    bd = pd.Timestamp(break_date)
    is_post = pd.Series(g_m.index >= bd, index=g_m.index)
    total = g_m["GLOBAL_TOTAL_KBD"].replace(0, np.nan)

    # 1) Strict: Yalnızca doğrudan yaptırımlı ülkeler (CN_russia + CN_iran_venezuela + IN_Russia baseline pay)
    cn_strict = (
        cn_m["CN_russia_KBD"] * sh_base.get("russia", 0.0)
        + cn_m["CN_iran_venezuela_KBD"] * sh_base.get("iran_venezuela", 0.0)
    )
    py_strict = cn_strict + in_m["Petroyuan_KBD"]

    # 2) Sabit Oran 2022+ (Merdivensiz %50): Post dönemde 0.50 sabit oran (hiç adım artışı yok)
    sh_rus_const = sh_base.get("russia", pd.Series(0.0, index=g_m.index)).where(~is_post, 0.50)
    sh_iv_const = sh_base.get("iran_venezuela", pd.Series(0.0, index=g_m.index)).where(~is_post, 0.50)
    cn_const = cn_m["CN_russia_KBD"] * sh_rus_const + cn_m["CN_iran_venezuela_KBD"] * sh_iv_const
    in_const = pd.Series(0.0, index=g_m.index).where(
        ~is_post,
        np.minimum(0.50 * in_m["Rusya_Ithalat_KBD"], in_m["Toplam_Ithalat_KBD"]),
    )
    py_const = cn_const + in_const

    scenarios = [
        ("Global Petroyuan (KBD)", "Strict (Yalnızca Yaptırımlı)", py_strict),
        ("Petroyuan Payı (%)", "Strict (Yalnızca Yaptırımlı)", py_strict / total * 100.0),
        ("Global Petroyuan (KBD)", "Sabit Oran (Merdivensiz %50)", py_const),
        ("Petroyuan Payı (%)", "Sabit Oran (Merdivensiz %50)", py_const / total * 100.0),
    ]

    rows = []
    for target_name, scen_name, series in scenarios:
        s = series.dropna()
        if len(s) < 18 or (s.index < bd).sum() < 2 or (s.index >= bd).sum() < 2:
            continue
        lvl, _ = its_regression(s, break_date, seasonal=True)
        diff, _ = its_diff_regression(s, break_date, seasonal=True)
        rows.append(
            {
                "seri": target_name,
                "senaryo": scen_name,
                "its_level_shift": float(lvl.params["post"]),
                "its_level_p": float(lvl.pvalues["post"]),
                "its_slope_change": float(lvl.params["post_trend"]),
                "its_slope_p": float(lvl.pvalues["post_trend"]),
                "diff_level_shift": float(diff.params["d_post"]),
                "diff_level_p": float(diff.pvalues["d_post"]),
                "diff_slope_change": float(diff.params["post_lag"]),
                "diff_slope_p": float(diff.pvalues["post_lag"]),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_gacc_jodi(zero_df: pd.DataFrame) -> Dict[str, float]:
    """Çin gümrük (GACC) ile JODI Çin ham petrol toplamlarını çapraz doğrular."""
    if zero_df.empty or not {"CN_Total_KBD", "JODI_CN_KBD"} <= set(zero_df.columns):
        return {"n": 0, "pearson_r": float("nan"), "pearson_p": float("nan"), "mape_pct": float("nan"), "mean_diff_pct": float("nan"), "median_abs_diff_pct": float("nan")}

    sub = zero_df.dropna(subset=["CN_Total_KBD", "JODI_CN_KBD"]).copy()
    sub = sub[sub["JODI_CN_KBD"] > 0]
    n = len(sub)
    if n < 2:
        return {"n": int(n), "pearson_r": float("nan"), "pearson_p": float("nan"), "mape_pct": float("nan"), "mean_diff_pct": float("nan"), "median_abs_diff_pct": float("nan")}

    diff_pct = (sub["CN_Total_KBD"] - sub["JODI_CN_KBD"]) / sub["JODI_CN_KBD"] * 100.0
    abs_diff = diff_pct.abs()
    if np.isclose(sub["CN_Total_KBD"].std(ddof=0), 0.0) or np.isclose(sub["JODI_CN_KBD"].std(ddof=0), 0.0):
        r_val, p_val = float("nan"), float("nan")
    else:
        r_val, p_val = stats.pearsonr(sub["CN_Total_KBD"], sub["JODI_CN_KBD"])

    return {
        "n": int(n),
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "mape_pct": float(abs_diff.mean()),
        "mean_diff_pct": float(diff_pct.mean()),
        "median_abs_diff_pct": float(abs_diff.median()),
    }


def price_discount_regression(
    zero_df: pd.DataFrame,
    break_date: str = BREAK_DATE,
) -> pd.DataFrame:
    """Ekonomik mekanizma testi: Rusya varil iskontosu ($/bbl) Çin yaptırımlı petrol payını sürüklüyor mu?

    Seviye modeli (HAC): y_t ~ const + iskonto_t + trend + post + post_trend + ay_kukla
    1. fark modeli (HAC): dy_t ~ const + d_iskonto_t + d_post + post_lag + ay_kukla
    """
    if zero_df.empty or "CN_Russia_Discount_USD_bbl" not in zero_df.columns:
        return pd.DataFrame()

    df = _monthly_index(zero_df).sort_index()
    bd = pd.Timestamp(break_date)
    targets = (
        ("CN_Russia_Share_Pct", "Çin Rusya Payı (%)"),
        ("CN_Strict_Share_Pct", "Çin Strict Payı (%)"),
        ("CN_Broad_Share_Pct", "Çin Broad Payı (%)"),
    )

    rows = []
    for col, label in targets:
        if col not in df.columns:
            continue
        sub = df[[col, "CN_Russia_Discount_USD_bbl"]].dropna().sort_index()
        if len(sub) < 18 or (sub.index < bd).sum() < 2 or (sub.index >= bd).sum() < 2:
            continue
        if np.isclose(sub["CN_Russia_Discount_USD_bbl"].std(ddof=0), 0.0):
            continue

        X_base = design_matrix(sub.index, break_date=break_date, seasonal=True)
        X_lvl = pd.concat([X_base[["const"]], sub[["CN_Russia_Discount_USD_bbl"]].rename(columns={"CN_Russia_Discount_USD_bbl": "spread"}), X_base.drop(columns=["const"])], axis=1)
        if len(X_lvl) <= X_lvl.shape[1] + 1 or np.linalg.matrix_rank(X_lvl) < X_lvl.shape[1]:
            continue
        lags_lvl = newey_west_lags(len(sub))
        fit_lvl = sm.OLS(sub[col], X_lvl).fit(cov_type="HAC", cov_kwds={"maxlags": lags_lvl})

        dy = sub[col].diff().dropna()
        d_spread = sub["CN_Russia_Discount_USD_bbl"].diff().reindex(dy.index).rename("d_spread")
        X_diff_base = diff_design_matrix(sub.index, dy.index, break_date=break_date, seasonal=True)
        X_diff = pd.concat([X_diff_base[["const"]], d_spread.to_frame(), X_diff_base.drop(columns=["const"])], axis=1).dropna()
        dy = dy.reindex(X_diff.index)
        if len(X_diff) <= X_diff.shape[1] + 1 or np.linalg.matrix_rank(X_diff) < X_diff.shape[1]:
            d_coef = d_se = d_p = d_r2 = float("nan")
        else:
            lags_diff = newey_west_lags(len(dy))
            fit_diff = sm.OLS(dy, X_diff).fit(cov_type="HAC", cov_kwds={"maxlags": lags_diff})
            d_coef = float(fit_diff.params["d_spread"])
            d_se = float(fit_diff.bse["d_spread"])
            d_p = float(fit_diff.pvalues["d_spread"])
            d_r2 = float(fit_diff.rsquared)

        pre_sp = sub.loc[sub.index < bd, "CN_Russia_Discount_USD_bbl"]
        post_sp = sub.loc[sub.index >= bd, "CN_Russia_Discount_USD_bbl"]
        rows.append(
            {
                "seri": label,
                "n_gozlem": int(len(sub)),
                "ort_iskonto_pre": float(pre_sp.mean()) if not pre_sp.empty else float("nan"),
                "ort_iskonto_post": float(post_sp.mean()) if not post_sp.empty else float("nan"),
                "seviye_beta": float(fit_lvl.params["spread"]),
                "seviye_se": float(fit_lvl.bse["spread"]),
                "seviye_p": float(fit_lvl.pvalues["spread"]),
                "seviye_r2": float(fit_lvl.rsquared),
                "fark_beta": d_coef,
                "fark_se": d_se,
                "fark_p": d_p,
                "fark_r2": d_r2,
            }
        )
    return pd.DataFrame(rows)


def panel_event_study(
    panel_df: pd.DataFrame,
    break_date: str = BREAK_DATE,
    treated_regions: Iterable[str] = ("CN", "IN"),
    pre_window: int = 12,
    post_window: int = 24,
) -> pd.DataFrame:
    """TWFE Panel Olay Çalışması (Event Study, ref = -1) + Paralel Eğilimler ortak Wald/F testi."""
    df_p = panel_df.copy()
    df_p["Date"] = pd.to_datetime(df_p["Date"])
    if "total_kbd" not in df_p.columns:
        df_p["total_kbd"] = df_p["petrodollar_kbd"] + df_p["petroyuan_kbd"]
    if "petroyuan_share_pct" not in df_p.columns:
        df_p["petroyuan_share_pct"] = np.where(
            df_p["total_kbd"] > 0, df_p["petroyuan_kbd"] / df_p["total_kbd"] * 100.0, np.nan
        )

    bd = pd.Timestamp(break_date)
    treated_set = set(treated_regions)
    df_p["treated"] = df_p["region"].isin(treated_set).astype(float)
    df_p["rel_m"] = (df_p["Date"].dt.year - bd.year) * 12 + (df_p["Date"].dt.month - bd.month)

    rows = []
    for col, label in (
        ("petroyuan_kbd", "Ülke Petroyuan Hacmi (KBD)"),
        ("petroyuan_share_pct", "Ülke Petroyuan Payı (%)"),
    ):
        sub = df_p.dropna(subset=[col, "region", "Date"]).copy()
        n_regions = sub["region"].nunique()
        n_months = sub["Date"].nunique()
        if n_regions < 2 or n_months < 3 or sub["treated"].nunique() < 2:
            continue

        rel_binned = sub["rel_m"].clip(lower=-pre_window, upper=post_window).astype(int)
        avail_k = sorted(int(k) for k in rel_binned.unique())
        ref_k = -1 if -1 in avail_k else max([k for k in avail_k if k < 0], default=avail_k[0])

        event_cols: Dict[str, pd.Series] = {}
        for k in avail_k:
            if k == ref_k:
                continue
            ind = (sub["treated"] * (rel_binned == k)).astype(float)
            if not np.isclose(ind.std(ddof=0), 0.0):
                event_cols[str(k)] = ind
        if not event_cols:
            continue

        D_df = pd.DataFrame(event_cols, index=sub.index)
        reg_d = pd.get_dummies(sub["region"], prefix="reg", drop_first=True, dtype=float)
        time_d = pd.get_dummies(sub["Date"].dt.strftime("%Y-%m"), prefix="t", drop_first=True, dtype=float)
        X_fe = pd.concat([pd.Series(1.0, index=sub.index, name="const"), reg_d, time_d], axis=1)

        X_np = X_fe.to_numpy(dtype=float)
        y_vec = sub[col].astype(float)
        M = np.column_stack([y_vec.to_numpy(dtype=float), D_df.to_numpy(dtype=float)])
        coef_fe, _, rank_fe, _ = np.linalg.lstsq(X_np, M, rcond=None)
        res_mat = M - X_np @ coef_fe

        y_res = pd.Series(res_mat[:, 0], index=sub.index, name=col)
        D_res = pd.DataFrame(res_mat[:, 1:], index=sub.index, columns=D_df.columns)
        keep_cols = [c for c in D_res.columns if not np.isclose(D_res[c].std(ddof=0), 0.0)]
        if not keep_cols:
            continue
        D_res = D_res[keep_cols]

        fwl = sm.OLS(y_res, D_res).fit(cov_type="cluster", cov_kwds={"groups": sub["region"]})
        n_obs = len(sub)
        k_fe = int(rank_fe)
        k_d = D_res.shape[1]
        dof_adj = (n_obs - k_d) / max(1, n_obs - k_fe - k_d)
        cov_adj = fwl.cov_params() * dof_adj

        n_treated_post = int(sub.loc[(sub["treated"] > 0) & (sub["rel_m"] >= 0), "region"].nunique())
        df_clust = max(1, n_regions - 1)
        crit = float(stats.t.ppf(0.975, df=df_clust))

        pre_cols = [c for c in keep_cols if int(c) < ref_k]
        if n_treated_post < 2 or not pre_cols:
            pre_f = pre_p = float("nan")
        else:
            V_pre = cov_adj.loc[pre_cols, pre_cols].to_numpy(dtype=float)
            V_sym = 0.5 * (V_pre + V_pre.T)
            b_pre = fwl.params.loc[pre_cols].to_numpy(dtype=float)
            eigvals, eigvecs = np.linalg.eigh(V_sym)
            max_ev = float(eigvals.max()) if eigvals.size else 0.0
            pos = eigvals > max(1e-8, max_ev * 1e-3)
            rank_v = int(pos.sum())
            if rank_v > 0:
                proj = eigvecs[:, pos].T @ b_pre
                wald = float(np.sum((proj ** 2) / eigvals[pos]))
                pre_f = wald / rank_v
                pre_p = float(stats.f.sf(pre_f, rank_v, df_clust))
            else:
                pre_f = pre_p = float("nan")

        if n_treated_post < 2:
            log.warning(
                "Panel Event-Study (%s): kırılma sonrası verisi olan yalnızca %d treated bölge var; "
                "kümeli SE güvenilmez olduğu için NaN bırakıldı.",
                label,
                n_treated_post,
            )

        sub_rows = [
            {
                "bagimli_degisken": label,
                "rel_month": int(ref_k),
                "event_date": (bd + pd.DateOffset(months=int(ref_k))).strftime("%Y-%m"),
                "coef": 0.0,
                "se": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "t_ist": float("nan"),
                "p_degeri": float("nan"),
                "is_ref": True,
                "pretrend_f": pre_f,
                "pretrend_p": pre_p,
                "n_gozlem": int(n_obs),
                "n_ulke": int(n_regions),
            }
        ]
        for c in keep_cols:
            k = int(c)
            coef = float(fwl.params[c])
            if n_treated_post < 2:
                se = ci_l = ci_h = t_s = p_v = float("nan")
            else:
                se = float(np.sqrt(max(0.0, float(cov_adj.loc[c, c]))))
                ci_l = coef - crit * se
                ci_h = coef + crit * se
                t_s = coef / se if se > 0 else float("nan")
                p_v = float(2.0 * stats.t.sf(abs(t_s), df=df_clust)) if np.isfinite(t_s) else float("nan")
            sub_rows.append(
                {
                    "bagimli_degisken": label,
                    "rel_month": k,
                    "event_date": (bd + pd.DateOffset(months=k)).strftime("%Y-%m"),
                    "coef": coef,
                    "se": se,
                    "ci_low": ci_l,
                    "ci_high": ci_h,
                    "t_ist": t_s,
                    "p_degeri": p_v,
                    "is_ref": False,
                    "pretrend_f": pre_f,
                    "pretrend_p": pre_p,
                    "n_gozlem": int(n_obs),
                    "n_ulke": int(n_regions),
                }
            )
        rows.extend(sorted(sub_rows, key=lambda r: r["rel_month"]))
    return pd.DataFrame(rows)


def run_zero_tests(
    zero_df: Optional[pd.DataFrame] = None,
    break_date: str = BREAK_DATE,
    targets: Dict[str, str] = ZERO_TARGETS,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, List[str]]:
    """Sıfır varsayımlı (pure customs/EPDK/JODI) seriler, GACC-JODI doğrulaması ve fiyat iskontosu analizini çalıştırır."""
    if zero_df is None:
        z_path = FILES["zero_global"]
        if not z_path.exists():
            return pd.DataFrame(), {}, pd.DataFrame(), []
        df = pd.read_csv(z_path, index_col=0, parse_dates=True).sort_index().loc[START_DATE:END_DATE]
    else:
        df = zero_df.copy().sort_index().loc[START_DATE:END_DATE]

    if df.empty:
        return pd.DataFrame(), {}, pd.DataFrame(), []
    df = df.asfreq("MS")

    summaries, report = [], []
    for col, name in targets.items():
        if col not in df.columns or df[col].dropna().empty:
            continue
        s, lines = analyze(df[col], name, break_date=break_date)
        summaries.append(asdict(s))
        report.extend(lines)

    xval = cross_validate_gacc_jodi(df)
    if xval.get("n", 0) >= 2:
        report.extend(
            [
                "=" * 82,
                " GACC vs JODI Çin Toplam İthalatı Çapraz Doğrulaması",
                "=" * 82,
                f"Örtüşen ay: {xval['n']} | Pearson r = {xval['pearson_r']:.4f} (p={xval['pearson_p']:.4g}) | "
                f"MAPE = %{xval['mape_pct']:.2f} | Ort. fark = %{xval['mean_diff_pct']:+.2f} | "
                f"Medyan |fark| = %{xval['median_abs_diff_pct']:.2f}",
                "",
            ]
        )

    disc_df = price_discount_regression(df, break_date=break_date)
    if not disc_df.empty:
        report.extend(
            [
                "=" * 82,
                " Fiyat Mekanizması: Rusya Gümrük İskontosu ($/varil) -> Çin Yaptırımlı Payı (%)",
                "=" * 82,
                disc_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
                "",
            ]
        )

    summary_df = pd.DataFrame(summaries)
    if zero_df is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        if not summary_df.empty:
            summary_df.to_csv(OUTPUTS_DIR / "econometrics_zero_assumption_summary.csv", index=False)
        if not disc_df.empty:
            disc_df.to_csv(OUTPUTS_DIR / "econometrics_discount_mechanism.csv", index=False)
        if report:
            (OUTPUTS_DIR / "econometrics_zero_assumption_report.txt").write_text("\n".join(report), encoding="utf-8")
            log.info("Sıfır varsayım ekonometri raporu kaydedildi: outputs/econometrics_zero_assumption_report.txt")
    return summary_df, xval, disc_df, report


def run_tests(targets: Dict[str, str] = TARGETS) -> bool:
    path = FILES["global"]
    if not path.exists():
        log.error("Ekonometri: %s bulunamadı.", path.name)
        return False
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().loc[START_DATE:END_DATE]
    if df.empty:
        log.error("Ekonometri: %s içinde %s–%s aralığında veri yok.", path.name, START_DATE, END_DATE)
        return False
    df = df.asfreq("MS")
    if df.isna().all(axis=1).any():
        log.warning("Global seride ay boşlukları var (complete-case). Testler boşluksuz alt örneklem üzerinde.")

    summaries, report = [], []
    for col, name in targets.items():
        if col not in df.columns:
            continue
        s, lines = analyze(df[col], name)
        summaries.append(asdict(s))
        report.extend(lines)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sens_df = sensitivity_robustness(df)
    if FILES["china_payment"].exists() and FILES["india"].exists():
        cn_pay = pd.read_csv(FILES["china_payment"], index_col=0, parse_dates=True)
        in_df = pd.read_csv(FILES["india"], index_col=0, parse_dates=True)
        const_df = constant_share_robustness(df, cn_pay, in_df)
        if not const_df.empty:
            sens_df = pd.concat([sens_df, const_df], ignore_index=True)

    if not sens_df.empty:
        sens_df.to_csv(OUTPUTS_DIR / "econometrics_sensitivity.csv", index=False)
        report.extend(
            [
                "=" * 82,
                " 7) Varsayım Duyarlılık Bandı (±%5) & Sabit-Oran / Strict Sağlamlık Kontrolü",
                "=" * 82,
                sens_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
                "",
            ]
        )

    if FILES["panel"].exists():
        panel_raw = pd.read_csv(FILES["panel"])
        panel_df = panel_fe_did(panel_raw)
        if not panel_df.empty:
            panel_df.to_csv(OUTPUTS_DIR / "econometrics_panel_summary.csv", index=False)
            report.extend(
                [
                    "=" * 82,
                    " 8) İki Yönlü Sabit Etkili Panel DiD (Ülke + Ay-Yıl FE, Bölge Kümelenmiş SE)",
                    "=" * 82,
                    panel_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
                    "",
                ]
            )
        es_df = panel_event_study(panel_raw)
        if not es_df.empty:
            es_df.to_csv(OUTPUTS_DIR / "econometrics_panel_event_study.csv", index=False)
            es_lines = [
                "=" * 82,
                " 9) Panel Olay Çalışması (Event Study, ref=-1) & Paralel Eğilimler Testi",
                "=" * 82,
            ]
            for dep, grp in es_df.groupby("bagimli_degisken", sort=False):
                f_pre = grp["pretrend_f"].iloc[0]
                p_pre = grp["pretrend_p"].iloc[0]
                key_t = grp[grp["rel_month"].isin([-12, -6, 0, 6, 12, 24])]
                snap = ", ".join(f"t={int(r.rel_month):+d}:{r.coef:.2f}" for r in key_t.itertuples())
                es_lines.append(f"{dep}: Pre-trend F={f_pre:.3f} (p={p_pre:.4g}) | Özet: {snap}")
            es_lines.append("")
            report.extend(es_lines)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "econometrics_report.txt").write_text("\n".join(report), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(OUTPUTS_DIR / "econometrics_summary.csv", index=False)

    _, _, _, zero_lines = run_zero_tests()
    full_print = report + ([""] + zero_lines if zero_lines else [])
    print("\n".join(full_print))
    log.info("Ekonometri raporu kaydedildi: outputs/econometrics_report.txt")
    return True


if __name__ == "__main__":
    setup_logging()
    run_tests()
