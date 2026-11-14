import numpy as np
import pandas as pd
import pytest

from src import econometrics


def _synthetic_break_series(n: int = 96, break_idx: int = 49, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    t = np.arange(n, dtype=float)
    post = (t >= break_idx).astype(float)
    seasonal = 40.0 * np.sin(2 * np.pi * dates.month / 12.0)
    y = 1000.0 + 3.0 * t + 300.0 * post + 25.0 * post * (t - break_idx) + seasonal + rng.normal(0, 12.0, n)
    return pd.Series(y, index=dates, name="y")


def test_unit_root_specs_and_constant_guard():
    y = _synthetic_break_series()
    ur = econometrics.unit_root_tests(y)
    assert list(ur["regresyon"]) == ["ct", "c"]
    assert ur.loc[1, "ADF_p"] < 0.05  # 1. fark durağan

    const_s = pd.Series(100.0, index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    ur_const = econometrics.unit_root_tests(const_s)
    assert ur_const["ADF_p"].isna().all()


def test_its_level_and_first_diff_recovery_and_quandt_andrews():
    y = _synthetic_break_series()
    break_date = y.index[49].strftime("%Y-%m-%d")  # 2022-02-01

    lvl_model, _ = econometrics.its_regression(y, break_date=break_date)
    assert lvl_model.params["post"] == pytest.approx(300.0, abs=30.0)
    assert lvl_model.params["post_trend"] == pytest.approx(25.0, abs=2.5)
    assert lvl_model.pvalues["post"] < 0.01
    assert lvl_model.pvalues["post_trend"] < 0.01

    diff_model, _ = econometrics.its_diff_regression(y, break_date=break_date)
    # 1. fark ITS'teki d_post = seviye sıçraması (c=300), post_lag = aylık eğim değişimi (d=25)
    assert diff_model.params["d_post"] == pytest.approx(300.0, abs=40.0)
    assert diff_model.params["post_lag"] == pytest.approx(25.0, abs=4.0)
    assert diff_model.pvalues["d_post"] < 0.01
    assert diff_model.pvalues["post_lag"] < 0.01

    qa_date, sup_f, sup_wald, _ = econometrics.quandt_andrews(y)
    assert sup_wald == pytest.approx(2.0 * sup_f)
    assert sup_wald >= econometrics.ANDREWS_1993_SUP_WALD_CV[0.01]
    assert abs((qa_date - pd.Timestamp(break_date)).days) <= 65


def test_residual_diagnostics_detects_ar1_error():
    rng = np.random.default_rng(12)
    n = 96
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal(0, 10.0)
    y = pd.Series(500.0 + 2.0 * np.arange(n) + e, index=dates)
    X = econometrics.design_matrix(y.index, "2022-02-01", seasonal=False)
    diag = econometrics.residual_diagnostics(y, X, bg_lags=4)
    assert diag["dw_stat"] < 1.0
    assert diag["bg_p"] < 0.01


def test_panel_fe_did_cluster_recovery():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")
    regions = ["CN", "IN", "US", "TR"] + [f"C{i:02d}" for i in range(16)]
    rows = []
    for r in regions:
        base_tot = 3000.0 if r in ("CN", "IN") else 1000.0
        for d in dates:
            post = int(d >= pd.Timestamp("2022-02-01"))
            py = (400.0 + rng.normal(0, 10.0)) if (r in ("CN", "IN") and post) else 0.0
            tot = base_tot + 50.0 * np.sin(d.month) + rng.normal(0, 20.0)
            pdol = max(0.0, tot - py)
            rows.append({"Date": d, "region": r, "petrodollar_kbd": pdol, "petroyuan_kbd": py})

    panel_df = pd.DataFrame(rows)
    res = econometrics.panel_fe_did(panel_df, break_date="2022-02-01", treated_regions=("CN", "IN"))
    vol_row = res.loc[res["bagimli_degisken"].str.contains("Hacmi")].iloc[0]
    assert vol_row["did_katsayisi"] == pytest.approx(400.0, abs=10.0)
    assert vol_row["p_degeri"] < 0.001
    assert vol_row["n_ulke"] == 20


def test_analyze_underdetermined_sample_safe():
    import warnings

    dates = pd.date_range("2018-01-01", periods=5, freq="MS")
    y = pd.Series([40000.0, 40500.0, 41000.0, 40200.0, 40800.0], index=dates, name="GLOBAL_PETRODOLAR")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary, lines = econometrics.analyze(y, "Test 5 ay", break_date="2022-02-01")
    assert not any("SingularMatrixWarning" in str(w.category) for w in caught)
    assert np.isnan(summary.its_level_p)
    assert np.isnan(summary.its_r2)
    assert summary.n_post == 0
    assert "p=nan" in "\n".join(lines)

    # Tek treated kümede bölge-kümeli SE güvenilmez -> NaN bırakılmalı
    p_dates = pd.date_range("2021-01-01", periods=24, freq="MS")
    p_rows = []
    for r in ("CN", "US", "TR", "JP"):
        for d in p_dates:
            post = int(d >= pd.Timestamp("2022-02-01"))
            p_rows.append({"Date": d, "region": r, "petrodollar_kbd": 1000.0, "petroyuan_kbd": 500.0 * post if r == "CN" else 0.0})
    single_res = econometrics.panel_fe_did(pd.DataFrame(p_rows), break_date="2022-02-01", treated_regions=("CN", "IN"))
    assert np.isnan(single_res.loc[0, "se_kumelenmis"])
    assert np.isnan(single_res.loc[0, "p_degeri"])
