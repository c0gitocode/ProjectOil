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


def test_epdk_sibling_il_bazinda_annex_does_not_drop_main_report():
    # Canlı EPDK HTML'inde 2024 Kasım ve Aralık ana rapor linkleri aynı <p>/<tr> içinde
    # "Eki - İl Bazında Teslimler" linkiyle kardeş olarak durduğunda ana rapor korunmalı, ek atlanmalıdır.
    html = """
    <div>
      <p>
        <a href="/Detay/DownloadDocument?id=LFSK+oClrtg=">2024_Kasım_Petrol Piyasası Sektör Raporu</a>
        <a href="/Detay/DownloadDocument?id=rEzZvUfJKNU=">Petrol Piyasası Sektör Raporu Eki - İl Bazında Teslimler Kasım 2024</a>
      </p>
      <table>
        <tr>
          <td>
            <a href="/Detay/DownloadDocument?id=4R1jVrGAsKc=">2024 Yılı Petrol Piyasası Aralık Ayı Sektör Raporu</a><br/>
            <a href="/Detay/DownloadDocument?id=annex_dec">Petrol Piyasası Sektör Raporu Eki - İl Bazında Teslimler Aralık 2024</a>
          </td>
        </tr>
        <tr>
          <td>2024 Yılı Ekim Ayı İl Bazında Bayi Satış Raporu</td>
          <td><a href="/Detay/DownloadDocument?id=bayi_oct">PDF İndir</a></td>
        </tr>
      </table>
    </div>
    """
    links, stats = epdk._collect_report_links(html, 2020)
    assert links == {
        (2024, 11): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=LFSK+oClrtg=",
        (2024, 12): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=4R1jVrGAsKc=",
    }
    assert stats["skipped_il_bazinda"] == 3


def _make_minimal_docx_bytes() -> bytes:
    import zipfile

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        "</Types>"
    )
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:tbl>"
        # Row 0: 'Ülke' (vMerge=restart) + 'İthalat (Ton)' (gridSpan=2)
        "<w:tr>"
        '<w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>Ülke</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>İthalat (Ton)</w:t></w:r></w:p></w:tc>'
        "</w:tr>"
        # Row 1: vMerge continue -> 'Ülke' miras alınmalı, 'Ham Petrol', 'Motorin'
        "<w:tr>"
        '<w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>'
        "<w:tc><w:p><w:r><w:t>Ham </w:t></w:r><w:r><w:t>Petrol</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Motorin</w:t></w:r></w:p></w:tc>"
        "</w:tr>"
        "<w:tr>"
        "<w:tc><w:p><w:r><w:t>Rusya Federasyonu</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>1.250.000,5</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>100</w:t></w:r></w:p></w:tc>"
        "</w:tr>"
        "<w:tr>"
        "<w:tc><w:p><w:r><w:t>Genel Toplam</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>2.500.000</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>200</w:t></w:r></w:p></w:tc>"
        "</w:tr>"
        "</w:tbl></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", doc_xml)
    # _process_report <1024 byte koruması için ZIP yorumu veya ek dolgu dosyası ekle
    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/styles.xml", "<w:styles/>" + " " * 1200)
    return buf2.getvalue()


def test_epdk_docx_xlsx_zip_and_word_links(tmp_path, monkeypatch):
    import openpyxl
    import sys
    import zipfile

    docx_bytes = _make_minimal_docx_bytes()
    assert epdk._detect_doc_type(docx_bytes) == "docx"
    assert epdk._detect_doc_type(b"%PDF-1.7\n...") == "pdf"
    assert epdk._detect_doc_type(b"<html>Error</html>") == "html"
    assert epdk._detect_doc_type(epdk._OLE2_MAGIC + b"W\x00o\x00r\x00d\x00D\x00o\x00c\x00u\x00m\x00e\x00n\x00t") == "doc"

    # stdlib XML fallback (python-docx olmadan da çalışmalı)
    docx_file = tmp_path / "sample.docx"
    docx_file.write_bytes(docx_bytes)
    monkeypatch.setitem(sys.modules, "docx", None)
    parsed = epdk.parse_docx(docx_file)
    assert parsed is not None
    assert parsed[0] == pytest.approx(2_500_000.0)
    assert parsed[1] == pytest.approx(1_250_000.5)

    # python-docx kütüphane yolu da doğru çalışmalı
    FakeCell = lambda t: SimpleNamespace(text=t, tables=[])  # noqa: E731
    FakeRow = lambda vals: SimpleNamespace(cells=[FakeCell(v) for v in vals])  # noqa: E731
    FakeTable = SimpleNamespace(
        rows=[
            FakeRow(["Ülke", "Ham Petrol", "Motorin"]),
            FakeRow(["Rusya", "600.000", "10"]),
            FakeRow(["Toplam", "1.500.000", "50"]),
        ]
    )
    monkeypatch.setitem(sys.modules, "docx", SimpleNamespace(Document=lambda p: SimpleNamespace(tables=[FakeTable])))
    assert epdk.parse_docx(docx_file) == pytest.approx((1_500_000.0, 600_000.0))
    monkeypatch.setitem(sys.modules, "docx", None)

    # XLSX doğrulaması + parse_xlsx
    xlsx_file = tmp_path / "sample.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Ülke", "Ham Petrol (Ton)"])
    ws.append(["Rusya", 300000])
    ws.append(["Toplam", 900000])
    wb.save(xlsx_file)
    assert epdk._detect_doc_type(xlsx_file.read_bytes()) == "xlsx"
    assert epdk.parse_xlsx(xlsx_file) == pytest.approx((900_000.0, 300_000.0))

    # Word ikonu / .docx linki elenmemeli, PDF öncelikli kalmalı, Excel atlanmalı
    html = """
    <div>
      <p>2020 Yılı Kasım Ayı Petrol Piyasası Sektör Raporu
        <a href="/Detay/DownloadDocument?id=word_nov"><img src="/img/word.png" alt="Word"/>Rapor (.docx)</a>
        <a href="/Detay/DownloadDocument?id=xls_nov"><img src="/img/xls.png" alt="Excel"/>Tablo</a>
      </p>
      <p>2021 Yılı Mart Ayı Petrol Piyasası Sektör Raporu
        <a href="/files/2021_mart.docx">Rapor İndir</a>
      </p>
    </div>
    """
    links = epdk.find_report_links(html, 2018)
    assert links == {
        (2020, 11): "https://www.epdk.gov.tr/Detay/DownloadDocument?id=word_nov",
        (2021, 3): "https://www.epdk.gov.tr/files/2021_mart.docx",
    }

    # ZIP içindeki DOCX yanıtı + cookie/Referer aktarımı + cache okuma doğrulaması
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("rapor_2025_02.docx", docx_bytes)
    zip_bytes = zip_buf.getvalue()

    monkeypatch.setattr(epdk, "RAW_DIR", tmp_path)
    seen_sessions = []

    class FakeResp:
        def __init__(self, content):
            self.content = content
            self.status_code = 200
            self.headers = {"Content-Type": "application/zip", "Content-Disposition": 'attachment; filename="rapor.zip"'}

        def raise_for_status(self):
            return None

    class FakeCurlSession:
        def __init__(self, **kwargs):
            self.headers = {}
            self.cookies = {}
            seen_sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, timeout=60):
            return FakeResp(zip_bytes)

    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(requests=SimpleNamespace(Session=FakeCurlSession)))
    row = epdk._process_report(2025, 2, "https://example.com/doc202502", cookies={"ASP.NET_SessionId": "abc"})
    assert row is not None
    assert row["TR_Toplam_Ithalat_Ton"] == pytest.approx(2_500_000.0)
    assert row["TR_Rusya_Ithalat_Ton"] == pytest.approx(1_250_000.5)
    assert seen_sessions[-1].headers.get("Referer") == epdk.EPDK_PAGE
    assert seen_sessions[-1].cookies.get("ASP.NET_SessionId") == "abc"
    assert (tmp_path / "epdk" / "2025" / "epdk_2025_02.docx").exists()

    # Önbellekte geçerli DOCX varken ağa gitmeden doğrudan okunmalı
    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(requests=SimpleNamespace(Session=lambda **kw: (_ for _ in ()).throw(AssertionError("ağ çağrılmamalı")))))
    cached_row = epdk._process_report(2025, 2, "https://example.com/doc202502")
    assert cached_row["TR_Toplam_Ithalat_Ton"] == pytest.approx(2_500_000.0)


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


def test_dgcis_alert_first_delete_all_recovery(monkeypatch):
    import sys

    class FakeNoAlertError(Exception):
        pass

    class FakeUnexpectedAlertError(Exception):
        def __init__(self, msg="", alert_text=None):
            super().__init__(msg)
            self.alert_text = alert_text

    class FakeOpt:
        def __init__(self, value, text):
            self._v = value
            self.text = text

        def get_attribute(self, name):
            return self._v if name == "value" else None

    class FakeSelectEl:
        def __init__(self, opts):
            self._opts = opts
            self.selected = None

        def current_options(self):
            return self._opts

    class FakeSelect:
        def __init__(self, el):
            self.el = el
            self.options = el.current_options()

        def select_by_value(self, val):
            self.el.selected = val

    class FakeAlert:
        def __init__(self, parent, text):
            self.parent = parent
            self.text = text

        def accept(self):
            self.parent.accepted_alerts.append(self.text)
            self.parent.open_alert = None

    class FakeSwitchTo:
        def __init__(self, parent):
            self.parent = parent

        @property
        def alert(self):
            if self.parent.open_alert is None:
                raise FakeNoAlertError("no alert")
            return FakeAlert(self.parent, self.parent.open_alert)

    class AlertDriver:
        def __init__(self):
            # sel_list1 başlangıçta "ALL" dolu olsun; ilk add'te açık uyarı üretsin, temizlendikten sonra başarılsın
            self.lists = {"sel_list1": ["ALL"], "sel_list4": ["ALL"]}
            self.selects = {
                "con": FakeSelectEl([FakeOpt("", "SELECT"), FakeOpt("15344", "RUSSIA")]),
                "port": FakeSelectEl([FakeOpt("", "SELECT"), FakeOpt("A", "ALL")]),
            }
            self.open_alert = None
            self.accepted_alerts = []
            self.switch_to = FakeSwitchTo(self)
            self.add_calls = 0

        def find_element(self, by, loc):
            if self.open_alert:
                raise FakeUnexpectedAlertError(self.open_alert, alert_text=self.open_alert)
            if by == "id":
                return self.selects[loc]
            return ("btn", loc)

        def find_elements(self, by, loc):
            if "not(contains(@onclick, 'addOption'))" in loc:
                return [("del", "sel_list1")] if "sel_list1" in loc else [("del", "sel_list4")]
            return []

        def execute_script(self, script, *args):
            if self.open_alert:
                raise FakeUnexpectedAlertError(self.open_alert, alert_text=self.open_alert)
            if "Array.from(l.options)" in script:
                return list(self.lists.get(args[0], []))
            if "l.options.length = 0" in script:
                self.lists[args[0]] = []
            return None

        def fake_click(self, el):
            if isinstance(el, tuple) and el[0] == "del":
                self.lists[el[1]] = []
            elif isinstance(el, tuple) and "sel_list1" in el[1]:
                self.add_calls += 1
                if self.add_calls == 1:
                    # İlk denemede DGCIS "FIRST DELETE ALL VALUE " fırlatsın
                    self.open_alert = "FIRST DELETE ALL VALUE "
                else:
                    self.lists["sel_list1"] = ["15344"]
            elif isinstance(el, tuple) and "sel_list4" in el[1]:
                self.lists["sel_list4"] = ["A"]

    class FakeWait:
        def __init__(self, drv):
            self.drv = drv

        def until(self, cond):
            return cond(self.drv)

    monkeypatch.setitem(sys.modules, "selenium.webdriver.common.by", SimpleNamespace(By=SimpleNamespace(ID="id", XPATH="xpath")))
    monkeypatch.setitem(
        sys.modules,
        "selenium.common.exceptions",
        SimpleNamespace(
            StaleElementReferenceException=RuntimeError,
            ElementClickInterceptedException=RuntimeError,
            NoAlertPresentException=FakeNoAlertError,
            UnexpectedAlertPresentException=FakeUnexpectedAlertError,
        ),
    )
    monkeypatch.setitem(sys.modules, "selenium.webdriver.support.ui", SimpleNamespace(Select=FakeSelect))
    monkeypatch.setitem(
        sys.modules,
        "selenium.webdriver.support.expected_conditions",
        SimpleNamespace(element_to_be_clickable=lambda loc: (lambda d: d.find_element(*loc))),
    )

    drv = AlertDriver()
    wait = FakeWait(drv)
    dgcis._ensure_filter_added(drv, wait, drv.fake_click, "con", "15344", "sel_list1")
    assert drv.lists["sel_list1"] == ["15344"]
    assert drv.accepted_alerts == ["FIRST DELETE ALL VALUE "]

    # sel_list4 "A" zaten seçili değil; "ALL" temizlenip "A" eklenmeli
    dgcis._ensure_filter_added(drv, wait, drv.fake_click, "port", "A", "sel_list4")
    assert drv.lists["sel_list4"] == ["A"]


def test_dgcis_atomic_refresh_ignores_stale_and_loading_rows(monkeypatch):
    import sys

    class AtomicFakeDriver:
        def __init__(self, snapshots):
            self.snapshots = snapshots
            self.idx = 0

        def execute_script(self, script, *args):
            cur = self.snapshots[min(self.idx, len(self.snapshots) - 1)]
            self.idx += 1
            return cur

    class FakeWait:
        def __init__(self, drv):
            self.drv = drv

        def until(self, cond):
            for _ in range(30):
                if cond(self.drv):
                    return True
            raise AssertionError("atomic condition never satisfied")

    monkeypatch.setitem(
        sys.modules,
        "selenium.common.exceptions",
        SimpleNamespace(StaleElementReferenceException=RuntimeError, TimeoutException=TimeoutError),
    )
    monkeypatch.setitem(sys.modules, "selenium.webdriver.common.by", SimpleNamespace(By=SimpleNamespace(ID="id")))
    monkeypatch.setitem(
        sys.modules,
        "selenium.webdriver.support.ui",
        SimpleNamespace(WebDriverWait=lambda d, t: FakeWait(d)),
    )
    monkeypatch.setitem(
        sys.modules,
        "selenium.webdriver.support.expected_conditions",
        SimpleNamespace(staleness_of=lambda el: (lambda d: False)),
    )

    old_sig = "1|S5|CRUDE|RUSSIA|MUNDRA|TON|1000|1|1"
    new_sig = "1|S5|CRUDE|RUSSIA|SIKKA|TON|2500|2|2"
    new_row = ["1", "S5", "CRUDE", "RUSSIA", "SIKKA", "TON", "2500", "2", "2"]

    # Senaryo 1: Önce eski ay satırları + pending HTTP, sonra geçici "Total 0" flash'ı, ardından gerçek yeni ay satırları (2 kararlı okuma)
    drv1 = AtomicFakeDriver(
        [
            {"state": "rows", "rows": [["1", "S5", "CRUDE", "RUSSIA", "MUNDRA", "TON", "1000", "1", "1"]], "sig": old_sig, "pending": 1, "html": "old"},
            {"state": "pending", "rows": [], "sig": "PENDING:Loading", "pending": 0, "html": "load"},
            {"state": "empty", "rows": [], "sig": "EMPTY:total 0 records", "pending": 0, "html": "empty_flash"},
            {"state": "rows", "rows": [new_row], "sig": new_sig, "pending": 0, "html": "new1"},
            {"state": "rows", "rows": [new_row], "sig": new_sig, "pending": 0, "html": "new1"},
        ]
    )
    extracted = []
    has_rows = dgcis._wait_result_refresh(drv1, FakeWait(drv1), None, "old", old_sig=old_sig, out_rows=extracted)
    assert has_rows is True
    assert extracted == [new_row]

    # Senaryo 2: Onaylı boş ay (geçiş + 3 kararlı boş kontrol) -> False ve boş satır listesi
    drv2 = AtomicFakeDriver(
        [
            {"state": "pending", "rows": [], "sig": "PENDING:", "pending": 1, "html": "wait"},
            {"state": "empty", "rows": [], "sig": "EMPTY:total 0 records", "pending": 0, "html": "z"},
            {"state": "empty", "rows": [], "sig": "EMPTY:total 0 records", "pending": 0, "html": "z"},
            {"state": "empty", "rows": [], "sig": "EMPTY:total 0 records", "pending": 0, "html": "z"},
        ]
    )
    extracted2 = []
    has_rows2 = dgcis._wait_result_refresh(drv2, FakeWait(drv2), None, "old", old_sig=old_sig, out_rows=extracted2)
    assert has_rows2 is False
    assert extracted2 == []


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
