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


def test_epdk_find_report_links_html_variants():
    assert epdk.detect_report_month("2024 Yılı Kasım Ayı Raporu (Revize: Ocak 2025)") == (2024, 11)
    assert epdk.detect_report_month("2024 Yılı Ocak - Şubat Ayı Raporu") is None

    html = """
    <div class="accordion">
      <p>2023 Yılı Haziran Ayı Petrol Piyasası Sektör Raporu
         <strong><a href="/Detay/DownloadDocument?id=201"><i class="fa fa-file-pdf"></i></a></strong>
      </p>
      <p>2023 Yılı Temmuz Ayı Petrol Piyasası Sektör Raporu
         <a href="/Detay/DownloadDocument?id=202"><img alt="PDF" src="/img/doc.png"/></a>
         <a href="/Detay/DownloadDocument?id=203"><img alt="Excel" src="/img/xls.png"/></a>
      </p>
      <div>
        2023 Yılı Ağustos Ayı Petrol Piyasası Sektör Raporu <a href="/Detay/DownloadDocument?id=204">PDF</a><br/>
        2023 Yılı Eylül Ayı Petrol Piyasası Sektör Raporu <a href="/Detay/DownloadDocument?id=205">PDF</a>
      </div>
      <table>
        <tr>
          <td>2023 Yılı Ekim Ayı İl Bazında Bayi Satış Raporu</td>
          <td><a href="/Detay/DownloadDocument?id=206">2023 Ekim</a></td>
        </tr>
      </table>
    </div>
    """
    links = epdk.find_report_links(html, 2020)
    assert links == {
        (2023, 6): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=201",
        (2023, 7): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=202",
        (2023, 8): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=204",
        (2023, 9): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=205",
    }


def test_epdk_real_2025_titles_and_candidate_fallback(tmp_path, monkeypatch):
    import sys

    # Gerçek EPDK çıktısında görülen başlık biçimleri (jenerik img ikonu ve "pdf" kelimesi olmadan)
    html = """
    <div>
      <p><a href="/Detay/DownloadDocument?id=jKekVOLPSRk="><img src="/img/file.png"/>2025 Yılı Petrol Piyasası Ocak Ayı Sektör Raporu</a></p>
      <p><a href="/Detay/DownloadDocument?id=ayK0wM7uGi0="><img src="/img/file.png"/>2025 Yılı Petrol Piyasası Şubat Ayı Sektör Raporu2</a></p>
      <p><a href="/Detay/DownloadDocument?id=9MUrmuaYSXY="><img src="/img/file.png"/>2025_Petrol Piyasası Mart Ayı Sektör Raporu</a></p>
      <p><a href="/Detay/DownloadDocument?id=OrqvVTwVfqI="><img src="/img/file.png"/>2025 Yılı Petrol Piyasası Nisan_Ayı Sektör Raporu_Rev</a></p>
      <p><a href="/Detay/DownloadDocument?id=YVOtDmShkzg="><img src="/img/file.png"/>2025 Yılı_Petrol Piyasası Mayıs Ayı Sektör Raporu</a></p>
      <p>
        2025 Yılı Haziran Ayı Petrol Piyasası Sektör Raporu
        <a href="/Detay/DownloadDocument?id=haz_ek"><img src="/img/generic.png"/>Ek Tablo</a>
        <a href="/Detay/DownloadDocument?id=haz_pdf"><img src="/img/pdf.png"/>Rapor</a>
      </p>
    </div>
    """
    links, stats = epdk._collect_report_links(html, 2020)
    assert set(links.keys()) == {(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5), (2025, 6)}
    # Haziran için açık PDF etiketi olan aday öncelikli seçilmeli
    assert links[(2025, 6)] == "https://www.epdk.gov.tr/Detay/DownloadDocument?id=haz_pdf"
    assert stats["candidates_by_ym"][(2025, 6)] == [
        "https://www.epdk.gov.tr/Detay/DownloadDocument?id=haz_pdf",
        "https://www.epdk.gov.tr/Detay/DownloadDocument?id=haz_ek",
    ]

    # Çoklu aday fallback: ilk URL xlsx (PK...), ikinci URL geçerli %PDF ise ikinciye düşüp başarmalı
    monkeypatch.setattr(epdk, "RAW_DIR", tmp_path)
    monkeypatch.setattr(epdk, "parse_pdf", lambda p: (1_000_000.0, 400_000.0))

    class FakeResp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    class FakeCurlSession:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, timeout=60):
            if "ek" in url:
                return FakeResp(b"PK\x03\x04" + b"\x00" * 2048)
            return FakeResp(b"%PDF-1.4\n" + b"0" * 2048)

    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(requests=SimpleNamespace(Session=FakeCurlSession)))
    res = epdk._process_report(2025, 7, ["https://example.com/ek", "https://example.com/rapor"])
    assert res is not None
    assert res["Tarih"] == "2025-07-01"
    assert res["TR_Toplam_Ithalat_Ton"] == pytest.approx(1_000_000.0)


# ---------------------------------------------------------
# DGCIS ve GACC testleri
# ---------------------------------------------------------
def test_dgcis_target_periods_completed_month(monkeypatch):
    monkeypatch.setattr(dgcis, "END_DATE", "2026-12-31")
    periods = dgcis.target_periods(now=pd.Timestamp("2026-11-14"))
    assert periods[0] == "Apr-2021"
    assert periods[-1] == "Oct-2026"


def test_dgcis_wait_select_helpers(monkeypatch):
    import sys

    class FakeOpt:
        def __init__(self, value, text):
            self._v = value
            self.text = text

        def get_attribute(self, name):
            return self._v if name == "value" else None

    class FakeSelect:
        def __init__(self, el):
            self.el = el
            self.options = el.current_options()

        def select_by_index(self, idx):
            if idx >= len(self.options):
                raise RuntimeError("NoSuchElement")
            self.el.selected = self.options[idx].get_attribute("value")

        def select_by_value(self, val):
            if not any(o.get_attribute("value") == val for o in self.options):
                raise RuntimeError("NoSuchElement")
            self.el.selected = val

    class FakeEl:
        def __init__(self, snapshots):
            self.snapshots = snapshots
            self.calls = 0
            self.selected = None

        def current_options(self):
            idx = min(self.calls, len(self.snapshots) - 1)
            self.calls += 1
            return self.snapshots[idx]

    sorted_el = FakeEl(
        [
            [FakeOpt("", "SELECT"), FakeOpt("{{item.COMBO_CODE}}", "{{item.COMBO_DESC}}")],
            [FakeOpt("", "SELECT"), FakeOpt("1", "QTY WISE")],
        ]
    )
    comm_el = FakeEl([[FakeOpt("", "SELECT")], [FakeOpt("S5", "CRUDE OIL")]])

    class FakeDriver:
        def find_element(self, by, sid):
            return sorted_el if sid == "sorted" else comm_el

        def execute_script(self, script, el):
            pass

    class FakeWait:
        def __init__(self, drv):
            self.drv = drv

        def until(self, cond):
            for _ in range(5):
                if cond(self.drv):
                    return True
            raise AssertionError("condition never satisfied")

    monkeypatch.setitem(sys.modules, "selenium.webdriver.common.by", SimpleNamespace(By=SimpleNamespace(ID="id")))
    monkeypatch.setitem(
        sys.modules,
        "selenium.common.exceptions",
        SimpleNamespace(StaleElementReferenceException=RuntimeError),
    )
    monkeypatch.setitem(sys.modules, "selenium.webdriver.support.ui", SimpleNamespace(Select=FakeSelect))

    drv = FakeDriver()
    wait = FakeWait(drv)
    dgcis._wait_select_index(drv, wait, "sorted", 1)
    dgcis._wait_select_value(drv, wait, "comm", "S5")
    assert sorted_el.selected == "1"
    assert comm_el.selected == "S5"


def test_dgcis_marquee_intercept_and_refresh(monkeypatch):
    import sys

    class FakeInterceptError(Exception):
        pass

    class FakeOpt:
        def __init__(self, value, text):
            self._v = value
            self.text = text

        def get_attribute(self, name):
            return self._v if name == "value" else None

    class FakeEl:
        def __init__(self, opts):
            self._opts = opts
            self.selected = None

        def current_options(self):
            return self._opts

    class BlockingSelect:
        def __init__(self, el):
            self.options = el.current_options()

        def select_by_index(self, idx):
            raise FakeInterceptError("marquee obscures element")

        def select_by_value(self, val):
            raise FakeInterceptError("marquee obscures element")

    sorted_el = FakeEl([FakeOpt("", "SELECT"), FakeOpt("1", "QTY WISE")])
    comm_el = FakeEl([FakeOpt("", "SELECT"), FakeOpt("S5", "CRUDE OIL")])

    class FakeDriver:
        def find_element(self, by, sid):
            return sorted_el if sid == "sorted" else comm_el

        def execute_script(self, script, *args):
            if "selectedIndex" in script and len(args) >= 2:
                el, val_or_idx = args[0], args[1]
                if isinstance(val_or_idx, int):
                    el.selected = el._opts[val_or_idx].get_attribute("value")
                else:
                    el.selected = val_or_idx

    class FakeWait:
        def __init__(self, drv):
            self.drv = drv

        def until(self, cond):
            for _ in range(5):
                if cond(self.drv):
                    return True
            raise AssertionError("condition never satisfied")

    monkeypatch.setitem(sys.modules, "selenium.webdriver.common.by", SimpleNamespace(By=SimpleNamespace(ID="id", XPATH="xpath")))
    monkeypatch.setitem(
        sys.modules,
        "selenium.common.exceptions",
        SimpleNamespace(
            StaleElementReferenceException=RuntimeError,
            ElementClickInterceptedException=FakeInterceptError,
            TimeoutException=TimeoutError,
        ),
    )
    monkeypatch.setitem(sys.modules, "selenium.webdriver.support.ui", SimpleNamespace(Select=BlockingSelect, WebDriverWait=lambda d, t: FakeWait(d)))
    monkeypatch.setitem(
        sys.modules,
        "selenium.webdriver.support.expected_conditions",
        SimpleNamespace(staleness_of=lambda el: (lambda d: True)),
    )

    drv = FakeDriver()
    wait = FakeWait(drv)
    dgcis._wait_select_index(drv, wait, "sorted", 1)
    dgcis._wait_select_value(drv, wait, "comm", "S5")
    assert sorted_el.selected == "1"
    assert comm_el.selected == "S5"

    # _wait_result_refresh: önce {{user.RNUM}} şablonunu yok saymalı, sonra "Total 0 Records" için False dönmeli
    class FakeNode:
        def __init__(self, text, displayed=True, html=""):
            self.text = text
            self._displayed = displayed
            self._html = html

        def is_displayed(self):
            return self._displayed

        def get_attribute(self, name):
            return self._html

    class RefreshDriver:
        def __init__(self, steps):
            self.steps = steps
            self.idx = 0

        def find_element(self, by, sid):
            cur = self.steps[min(self.idx, len(self.steps) - 1)]
            return FakeNode(cur["res_text"], html=cur["html"])

        def find_elements(self, by, xp):
            cur = self.steps[min(self.idx, len(self.steps) - 1)]
            self.idx += 1
            return cur["rows"]

    rdrv = RefreshDriver(
        [
            {"res_text": "{{user.RNUM}} Total {{total_count}} Records", "html": "changed", "rows": [FakeNode("{{user.RNUM}}")]},
            {"res_text": "Total 0 Records", "html": "changed2", "rows": []},
        ]
    )
    assert dgcis._wait_result_refresh(rdrv, FakeWait(rdrv), None, "init") is False

    rdrv_ok = RefreshDriver(
        [{"res_text": "Total 1 Records", "html": "changed", "rows": [FakeNode("1 S5 CRUDE RUSSIA 1000")]}]
    )
    assert dgcis._wait_result_refresh(rdrv_ok, FakeWait(rdrv_ok), None, "init") is True


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
