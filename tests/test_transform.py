import numpy as np
import pandas as pd
import pytest

from src import config, transform


def test_assumptions_csv_validity_and_band_start():
    cn_shares = transform.load_shares(config.ASSUMPTION_FILES["china"])
    in_shares = transform.load_shares(config.ASSUMPTION_FILES["india"])
    assert not cn_shares.empty
    assert not in_shares.empty

    dates = pd.to_datetime(["2020-01-01", "2022-01-01", "2022-02-01", "2024-06-01"])
    wide = transform.shares_for_dates(cn_shares, dates)
    assert wide.loc["2020-01-01", "russia"] == pytest.approx(0.0)
    assert wide.loc["2022-02-01", "russia"] == pytest.approx(0.15)
    assert wide.loc["2024-06-01", "russia"] == pytest.approx(0.95)

    sc = transform.share_scenarios(wide)
    # 2022-02 öncesi belirsizlik bandı açılmamalı (High == Low == Base)
    assert (sc["_High"].loc["2020-01-01"] == sc[""].loc["2020-01-01"]).all()
    assert (sc["_Low"].loc["2022-01-01"] == sc[""].loc["2022-01-01"]).all()
    assert sc["_High"].loc["2022-02-01", "russia"] == pytest.approx(0.20)
    assert sc["_Low"].loc["2022-02-01", "russia"] == pytest.approx(0.10)


def test_compute_china_imports_and_payment():
    raw = pd.DataFrame(
        [
            {"Date of data": "2021-01", "Trading partner": "World", "Quantity": "999999999", "US dollar": "1"},
            {"Date of data": "2021-01", "Trading partner": "Russia", "Quantity": "1,000,000", "US dollar": "400000"},
            {"Date of data": "2021-01", "Trading partner": "Iran", "Quantity": "500000", "US dollar": "150000"},
            {"Date of data": "2023-01", "Trading partner": "Russia", "Quantity": "2000000", "US dollar": "1000000"},
            {"Date of data": "2023-01", "Trading partner": "Saudi Arabia", "Quantity": "2000000", "US dollar": "1200000"},
        ]
    )
    imp = transform.compute_china_imports(raw)
    world_2021 = imp[(imp["Date"] == "2021-01-01") & (imp["Partner"] == "World")].iloc[0]
    assert world_2021["Quantity_kg"] == pytest.approx(1_500_000.0)
    assert world_2021["KBD"] == pytest.approx(1500.0 * config.BBL_PER_TONNE / 1000.0 / 31.0)

    pay = transform.compute_china_payment(imp, transform.load_shares(config.ASSUMPTION_FILES["china"]))
    # Muhasebe özdeşliği: Petroyuan + Petrodolar == Total ve düşük/yüksek çapraz simetrisi
    assert (pay["China_Petroyuan_KBD"] + pay["China_Petrodolar_KBD"]).tolist() == pytest.approx(
        pay["China_Total_KBD"].tolist(), abs=0.01
    )
    assert (pay["China_Petroyuan_Low_KBD"] + pay["China_Petrodolar_High_KBD"]).tolist() == pytest.approx(
        pay["China_Total_KBD"].tolist(), abs=0.01
    )
    assert pay.loc["2021-01-01", "China_Petroyuan_High_KBD"] == pytest.approx(
        pay.loc["2021-01-01", "China_Petroyuan_Low_KBD"]
    )


def test_compute_dgcis_and_india_bounds():
    dgcis = pd.DataFrame(
        [
            {"Period": "May-2023", "Unit": "TON", "Qty": "1,00,000"},  # Hint virgül gruplaması = 100k ton
            {"Period": "May-2023", "Unit": "KGS", "Qty": "5,000,000"},  # 5k ton
            {"Period": "Jun-2023", "Unit": "MT", "Qty": "99999999.0"},  # Aşırı büyük -> toplam ile kırpılmalı
        ]
    )
    rus = transform.compute_dgcis_russia_kbd(dgcis)
    expected_may = 105_000.0 * config.BBL_PER_TONNE / 1000.0 / 31.0
    assert rus.loc["2023-05-01"] == pytest.approx(expected_may)

    jodi = pd.DataFrame(
        {"IN": [4000.0, 4200.0, 4300.0, 4100.0]},
        index=pd.to_datetime(["2020-01-01", "2022-05-01", "2023-05-01", "2023-06-01"]),
    )
    ind = transform.compute_india(jodi, rus, transform.load_shares(config.ASSUMPTION_FILES["india"]))
    # 2020'de DGCIS yok ama Alt_Share=0 olduğundan Petroyuan=0, Petrodolar=Toplam
    assert ind.loc["2020-01-01", "Petroyuan_KBD"] == pytest.approx(0.0)
    assert ind.loc["2020-01-01", "Petrodolar_KBD"] == pytest.approx(4000.0)
    # 2022-05'te Alt_Share > 0 ama DGCIS yok -> NaN kalmalı (ffill YOK)
    assert pd.isna(ind.loc["2022-05-01", "Petroyuan_KBD"])
    # 2023-06'da Rusya toplamı aşsa bile Petroyuan <= Toplam ve Petrodolar >= 0
    assert ind.loc["2023-06-01", "Petroyuan_KBD"] <= ind.loc["2023-06-01", "Toplam_Ithalat_KBD"]
    assert ind.loc["2023-06-01", "Petrodolar_KBD"] >= 0.0


def test_compute_jodi_other_balanced_basket_and_panel_unsorted_index():
    dates = pd.date_range("2022-01-01", periods=10, freq="MS")
    # JP %100 dolu, DE 1 ay eksik (9/10=%90 -> zorunlu, o ay NaN olmalı),
    # IE %40 (düzensiz -> sepet dışı kalmalı, başka ayları zıplatmamalı),
    # IN/CN/US/TR çift sayımı önlemek için dışlanmalı
    jodi = pd.DataFrame(
        {
            "JP": [100.0] * 10,
            "DE": [50.0, 50.0, np.nan, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            "IE": [20.0, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan, np.nan, 20.0, 20.0],
            "IN": [999.0] * 10,
            "US": [888.0] * 10,
            "TR": [777.0] * 10,
        },
        index=dates,
    )
    other = transform.compute_jodi_other(jodi, required_coverage=0.90)
    assert other.loc[dates[0], "JODI_DIGER_KBD"] == pytest.approx(150.0)  # IE hariç!
    assert other.loc[dates[4], "JODI_DIGER_KBD"] == pytest.approx(150.0)  # IE bildirmiş olsa da sabit sepet!
    assert pd.isna(other.loc[dates[2], "JODI_DIGER_KBD"])  # DE eksik -> NaN

    # Panel regresyon testi (sırasız birleşen tarihlerde pandas KeyError atmamalı)
    us = pd.DataFrame({"ABD_Ithalat_KBD": [500.0] * 10}, index=dates)
    tr = pd.DataFrame({"TR_Toplam_Ithalat_KBD": [200.0] * 10}, index=dates)
    india = pd.DataFrame(
        {
            "Petrodolar_KBD": [300.0] * 10,
            "Petroyuan_KBD": [50.0] * 10,
            "Petroyuan_High_KBD": [60.0] * 10,
            "Petroyuan_Low_KBD": [40.0] * 10,
        },
        index=dates,
    )
    china = pd.DataFrame(
        {
            "China_Petrodolar_KBD": [800.0] * 10,
            "China_Petroyuan_KBD": [200.0] * 10,
            "China_Petroyuan_High_KBD": [220.0] * 10,
            "China_Petroyuan_Low_KBD": [180.0] * 10,
        },
        index=dates,
    )
    panel = transform.compute_panel(jodi, us, tr, india, china, start="2022-01-01", end="2022-10-01")
    regions = set(panel["region"].unique())
    assert {"JP", "DE", "IE", "US", "TR", "IN", "CN"} == regions
    # US/TR/IN JODI kaynaklı çift kayıt içermemeli (bölge x tarih tekil)
    assert panel.duplicated(subset=["Date", "region"]).sum() == 0
    assert panel.loc[panel["region"] == "CN", "petroyuan_share_pct"].iloc[0] == pytest.approx(20.0)


def test_compute_zero_assumption_global_strict_broad_discount_and_no_backfill():
    dates = pd.to_datetime(["2021-01-01", "2021-06-01", "2022-06-01"])
    jodi = pd.DataFrame(
        {
            "JP": [2000.0, 2000.0, 2000.0],
            "IN": [4000.0, 4200.0, 4500.0],
            "CN": [10000.0, 10000.0, 10000.0],
        },
        index=dates,
    )
    us = pd.DataFrame({"ABD_Ithalat_KBD": [6000.0, 6000.0, 6000.0]}, index=dates)
    tr = pd.DataFrame(
        {"TR_Toplam_Ithalat_KBD": [600.0, 600.0, 700.0], "TR_Rusya_Ithalat_KBD": [100.0, 120.0, 350.0]},
        index=dates,
    )
    # 2021-01 için DGCIS yok -> NaN kalmalı; 2022-06'da taşma varsa IN_Total ile kırpılmalı
    dgcis_rus = pd.Series([800.0, 9999.0], index=pd.to_datetime(["2021-06-01", "2022-06-01"]))

    # 1 varil = 1/7.33 ton -> kolay doğrulama için kg = n_bbl / BBL_PER_TONNE * 1000
    kg_per_bbl = 1000.0 / config.BBL_PER_TONNE
    gacc_rows = []
    for d in dates:
        # 10,000 bbl Russia @ $50/bbl, 5,000 bbl Malaysia @ $60/bbl, 10,000 bbl Saudi @ $80/bbl
        gacc_rows.append({"Date": d, "Partner": "Russia", "Quantity_kg": 1000.0 * kg_per_bbl, "USD": 50_000.0, "KBD": 1000.0})
        gacc_rows.append({"Date": d, "Partner": "Malaysia", "Quantity_kg": 500.0 * kg_per_bbl, "USD": 30_000.0, "KBD": 500.0})
        gacc_rows.append({"Date": d, "Partner": "Saudi Arabia", "Quantity_kg": 1000.0 * kg_per_bbl, "USD": 80_000.0, "KBD": 1000.0})
        gacc_rows.append({"Date": d, "Partner": "World", "Quantity_kg": 2500.0 * kg_per_bbl, "USD": 160_000.0, "KBD": 10_000.0})
    china_imp = pd.DataFrame(gacc_rows)

    z = transform.compute_zero_assumption_global(
        jodi, us, tr, dgcis_rus, china_imp, start="2021-01-01", end="2022-06-01"
    )
    # 2021-01: DGCIS eksik -> IN_Russia_KBD ve Zero_Strict/Broad NaN kalmalı, CN+TR yine dolu olmalı
    assert pd.isna(z.loc["2021-01-01", "IN_Russia_KBD"])
    assert pd.isna(z.loc["2021-01-01", "Zero_Broad_Share_Pct"])
    assert z.loc["2021-01-01", "CN_TR_Sanctioned_Strict_KBD"] == pytest.approx(1100.0)
    assert z.loc["2021-01-01", "CN_TR_Sanctioned_Broad_KBD"] == pytest.approx(1600.0)

    # Fiyat farkı kontrolü: Benchmark ($80) - Russia ($50) = $30/bbl
    assert z.loc["2021-06-01", "CN_Russia_Discount_USD_bbl"] == pytest.approx(30.0)
    assert z.loc["2021-06-01", "GACC_vs_JODI_CN_Diff_Pct"] == pytest.approx(0.0)

    # 2021-06: Strict = 1000 (CN) + 800 (IN) + 120 (TR) = 1920; Broad = +500 (MY) = 2420
    assert z.loc["2021-06-01", "Zero_Strict_KBD"] == pytest.approx(1920.0)
    assert z.loc["2021-06-01", "Zero_Broad_KBD"] == pytest.approx(2420.0)

    # 2022-06: IN_Russia_KBD (9999) JODI toplamıyla (4500) kırpılmalı
    assert z.loc["2022-06-01", "IN_Russia_KBD"] == pytest.approx(4500.0)
    assert z.loc["2022-06-01", "Zero_Strict_KBD"] == pytest.approx(1000.0 + 4500.0 + 350.0)
    assert z.loc["2022-06-01", "Zero_Broad_KBD"] == pytest.approx(1500.0 + 4500.0 + 350.0)
