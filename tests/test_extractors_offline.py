import io
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import config
from src.extract import dgcis, eia, epdk, gacc, jodi


# ---------------------------------------------------------
# JODI testleri
# ---------------------------------------------------------
def test_jodi_filter_crude_imports_no_double_counting():
    raw = pd.DataFrame(
        [
            {"REF_AREA": "JP", "TIME_PERIOD": "2023-05", "ENERGY_PRODUCT": "CRUDEOIL", "FLOW_BREAKDOWN": "TOTIMPSB", "UNIT_MEASURE": "KBD", "OBS_VALUE": "2500"},
            {"REF_AREA": "JP", "TIME_PERIOD": "2023-05", "ENERGY_PRODUCT": "CRUDEOIL", "FLOW_BREAKDOWN": "TOTIMPSB", "UNIT_MEASURE": "KBBL", "OBS_VALUE": "77500"},
            {"REF_AREA": "JP", "TIME_PERIOD": "2023-05", "ENERGY_PRODUCT": "TOTCRUDE", "FLOW_BREAKDOWN": "TOTIMPSB", "UNIT_MEASURE": "KBD", "OBS_VALUE": "2600"},
            {"REF_AREA": "DE", "TIME_PERIOD": "2023-05", "ENERGY_PRODUCT": "CRUDEOIL", "FLOW_BREAKDOWN": "TOTIMPSB", "UNIT_MEASURE": "KBD", "OBS_VALUE": "c"},
            {"REF_AREA": "US", "TIME_PERIOD": "2023-05", "ENERGY_PRODUCT": "CRUDEOIL", "FLOW_BREAKDOWN": "TOTIMPSB", "UNIT_MEASURE": "KBD", "OBS_VALUE": "6000"},
        ]
    )
    out = jodi.filter_crude_imports(raw)
    assert len(out) == 2
    jp_val = out.loc[out["REF_AREA"] == "JP", "OBS_VALUE"].iloc[0]
    de_val = out.loc[out["REF_AREA"] == "DE", "OBS_VALUE"].iloc[0]
    assert jp_val == pytest.approx(2500.0)
    assert math.isnan(de_val)  # Gizli veri 0 değil NaN olmalı


def test_jodi_filter_missing_columns_raises():
    with pytest.raises(ValueError, match="eksik sütunlar"):
        jodi.filter_crude_imports(pd.DataFrame({"REF_AREA": ["JP"], "OBS_VALUE": [10]}))


def test_jodi_find_year_links_and_revision_update(tmp_path, monkeypatch):
    html = """
    <html><body>
      <div>Primary oil world CSV:
        <a href="/downloads/world_primary_2024.csv">2024</a>
        <a href="/downloads/world_primary_2025.csv">2025</a>
      </div>
      <div>Secondary: <a href="/sec_2025.csv">2025</a></div>
    </body></html>
    """
    links = jodi._find_year_links(html, 2024, 2025)
    assert links == {
        2024: "https://www.jodidata.org/downloads/world_primary_2024.csv",
        2025: "https://www.jodidata.org/downloads/world_primary_2025.csv",
    }

    master = tmp_path / "jodi.csv"
    old = pd.DataFrame(
        {"JP": [100.0, 200.0], "DE": [50.0, 60.0]},
        index=pd.to_datetime(["2023-01-01", "2025-01-01"]),
    )
    old.to_csv(master)
    monkeypatch.setitem(config.FILES, "jodi", master)
    monkeypatch.setitem(jodi.FILES, "jodi", master)

    csv_2025 = (
        "REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,FLOW_BREAKDOWN,UNIT_MEASURE,OBS_VALUE\n"
        "JP,2025-01,CRUDEOIL,TOTIMPSB,KBD,215\n"
        "DE,2025-01,CRUDEOIL,TOTIMPSB,KBD,-\n"
    ).encode()

    class FakeSession:
        def get(self, url, timeout=30):
            r = MagicMock()
            r.raise_for_status = lambda: None
            r.text = '<div>Primary world: <a href="/world_primary_2025.csv">2025</a></div>'
            r.content = csv_2025
            return r

    monkeypatch.setattr(jodi, "create_retry_session", lambda: FakeSession())
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: pd.Timestamp("2025-06-01")))
    jodi.update()

    updated = pd.read_csv(master, index_col=0, parse_dates=True)
    assert updated.loc["2023-01-01", "DE"] == pytest.approx(50.0)
    assert updated.loc["2025-01-01", "JP"] == pytest.approx(215.0)
    # Revizyon sonrası DE NaN olmuşsa eski değer (60.0) taşınmamalı!
    assert pd.isna(updated.loc["2025-01-01", "DE"])


# ---------------------------------------------------------
# EIA testleri
# ---------------------------------------------------------
def test_eia_extract_and_weekly_fallback(tmp_path, monkeypatch):
    import sys

    out_csv = tmp_path / "us.csv"
    monkeypatch.setitem(config.FILES, "eia", out_csv)
    monkeypatch.setattr(eia, "get_eia_api_key", lambda: "TEST_KEY")

    class FakeAPI:
        def __init__(self, key):
            self.key = key

        def get_series(self, sid, **kwargs):
            if sid == eia.MONTHLY_SERIES:
                return pd.DataFrame()  # Boş dönüş -> haftalık fallback tetiklenmeli
            return pd.DataFrame(
                {"value": [6000.0, 6200.0, 6100.0]},
                index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-02-02"]),
            )

    monkeypatch.setitem(sys.modules, "myeia.api", SimpleNamespace(API=FakeAPI))
    eia.update()

    df = pd.read_csv(out_csv, index_col=0, parse_dates=True)
    assert df.loc["2024-01-01", "ABD_Ithalat_KBD"] == pytest.approx(6100.0)
    assert df.loc["2024-02-01", "ABD_Ithalat_KBD"] == pytest.approx(6100.0)


# ---------------------------------------------------------
# EPDK testleri
# ---------------------------------------------------------
def test_epdk_month_detection_table_and_links(tmp_path, monkeypatch):
    assert epdk.detect_report_month("Kasım 2024 Petrol Piyasası Sektör Raporu") == (2024, 11)
    assert epdk.detect_report_month("NİSAN 2023 Aylık Rapor") == (2023, 4)
    assert epdk.detect_report_month("Ocak - Şubat 2024 Raporu") is None

    table = [
        ["Ülke", "Ham Petrol (Ton)", "Motorin"],
        ["Rusya Federasyonu", "1.234.567,5", "0"],
        ["Irak", "500.000", "0"],
        ["ARA TOPLAM", "1.734.567,5", "0"],
        ["GENEL TOPLAM", "2.000.000", "100"],
    ]
    total_t, rus_t = epdk.parse_import_table(table)
    assert total_t == pytest.approx(2_000_000.0)
    assert rus_t == pytest.approx(1_234_567.5)

    html = """
    <table>
      <tr>
        <td>Kasım 2024 Petrol Piyasası Aylık Sektör Raporu</td>
        <td><a href="/DownloadDocument?id=101"><img src="pdf.png"/>İndir</a></td>
      </tr>
      <tr>
        <td>Kasım 2024 İl Bazında Bayi Satış Raporu</td>
        <td><a href="/DownloadDocument?id=102">PDF İndir</a></td>
      </tr>
    </table>
    """
    links = epdk.find_report_links(html, 2020)
    assert links == {(2024, 11): "https://www.epdk.gov.tr/DownloadDocument?id=101"}

    # %PDF başlığı olmayan HTML hata yanıtını reddetmeli
    monkeypatch.setattr(epdk, "RAW_DIR", tmp_path)

    class FakeCurlSession:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, timeout=60):
            r = MagicMock()
            r.raise_for_status = lambda: None
            r.content = b"<html><body>Error 403</body></html>" * 100
            return r

    import sys

    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(requests=SimpleNamespace(Session=FakeCurlSession)))
    with pytest.raises(ValueError, match="geçerli PDF değil"):
        epdk._process_report(2024, 11, "https://example.com/bad")


# ---------------------------------------------------------
# DGCIS ve GACC testleri
# ---------------------------------------------------------
def test_dgcis_target_periods_completed_month(monkeypatch):
    monkeypatch.setattr(dgcis, "END_DATE", "2026-12-31")
    periods = dgcis.target_periods(now=pd.Timestamp("2026-11-14"))
    assert periods[0] == "Apr-2021"
    assert periods[-1] == "Oct-2026"


def test_gacc_months_and_missing_years(tmp_path, monkeypatch):
    monkeypatch.setattr(gacc, "CHINA_RAW_DIR", tmp_path)
    monkeypatch.setattr(gacc, "START_DATE", "2024-01-01")

    # 2024: 12 benzersiz ay -> tam kabul edilmeli
    rows_2024 = [
        {"Date of data": f"2024-{m:02d}", "Trading partner": p, "Quantity": "1000"}
        for m in range(1, 13)
        for p in ("Russia", "Brazil")
    ]
    pd.DataFrame(rows_2024).to_csv(tmp_path / "2024_china.csv", index=False)

    # 2025: 13 satır ama sadece 2 benzersiz ay! (satır sayısı tuzağı testi)
    rows_2025 = [
        {"Date of data": f"20250{1 if i < 7 else 2}", "Trading partner": f"Partner{i}", "Quantity": "100"}
        for i in range(13)
    ]
    pd.DataFrame(rows_2025).to_csv(tmp_path / "2025_china.csv", index=False)

    assert gacc.months_in_file(tmp_path / "2024_china.csv") == set(range(1, 13))
    assert gacc.months_in_file(tmp_path / "2025_china.csv") == {1, 2}

    missing = gacc.missing_years(now=pd.Timestamp("2026-04-10"), end_date="2026-02-28")
    assert 2024 not in missing
    assert 2025 in missing
    assert 2026 in missing
