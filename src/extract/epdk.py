"""EPDK Petrol Piyasası Aylık Sektör Raporları: Türkiye ham petrol ithalatı (Rusya kırılımlı)."""
import io
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from bs4 import BeautifulSoup, Comment, NavigableString

from ..config import FILES, RAW_DIR, START_DATE
from ..logger import get_logger
from ..units import tonnes_to_kbd_scalar
from .common import atomic_write_csv, clean_header, normalize_epdk_columns, parse_number, tr_lower

log = get_logger(__name__)

EPDK_PAGE = "https://www.epdk.gov.tr/Detay/Icerik/3-0-104/petrolaylik-sektor-raporu"
MONTHS_TR = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}
_MONTH_PATTERN = "|".join(MONTHS_TR)
_MONTH_RE = re.compile(rf"(?<![a-zçğıöşü])({_MONTH_PATTERN})(?![a-zçğıöşü])")
_MONTH_RANGE_RE = re.compile(rf"(?<![a-zçğıöşü])(?:{_MONTH_PATTERN})\s*[-–/]\s*(?:{_MONTH_PATTERN})(?![a-zçğıöşü])")
_PRIMARY_YM_RE = re.compile(rf"(?<!\d)(20[1-3]\d)\s*(?:yılı\s*)?({_MONTH_PATTERN})\s*ayı(?![a-zçğıöşü])")
_YEAR_RE = re.compile(r"(?<!\d)(20[1-3]\d)(?!\d)")
_OOXML_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_MIN_DOC_BYTES = 128
_MAX_ZIP_MEMBER_BYTES = 50 * 1024 * 1024


def detect_report_month(text: str) -> Optional[Tuple[int, int]]:
    """Link metninden (yıl, ay) çıkarır. Aralık/çoklu ay metinlerinde None döner."""
    t = tr_lower(text)
    if _MONTH_RANGE_RE.search(t):
        return None
    months = set(_MONTH_RE.findall(t))
    years = set(_YEAR_RE.findall(t))
    if len(months) == 1 and len(years) == 1:
        return int(years.pop()), MONTHS_TR[months.pop()]
    # Başlık "2024 Yılı Kasım Ayı Raporu (Revize: Ocak 2025)" gibiyse ve çoklu dönem değilse ana tarihi al.
    if len(t) < 250 and len(months) <= 2 and len(years) <= 2:
        m = _PRIMARY_YM_RE.search(t)
        if m:
            return int(m.group(1)), MONTHS_TR[m.group(2)]
        head = re.split(r"\b(?:revize|rev|düzeltme)\b", t, maxsplit=1)[0]
        h_months = set(_MONTH_RE.findall(head))
        h_years = set(_YEAR_RE.findall(head))
        if len(h_months) == 1 and len(h_years) == 1:
            return int(h_years.pop()), MONTHS_TR[h_months.pop()]
    return None


def parse_import_table(table: List[List[Optional[str]]]) -> Optional[Tuple[float, float]]:
    """'Ülke / Ham Petrol' tablosundan (toplam_ton, rusya_ton) döndürür."""
    df = pd.DataFrame(table).fillna("").astype(str).map(clean_header)
    header_mask = df.apply(lambda r: r.str.contains("LKE").any() and r.str.contains("HAM PETROL").any(), axis=1)
    if not header_mask.any():
        return None
    h = int(header_mask.idxmax())
    headers = df.iloc[h].tolist()
    try:
        c_country = next(i for i, x in enumerate(headers) if "LKE" in x)
        c_crude = next(i for i, x in enumerate(headers) if "HAM PETROL" in x)
    except StopIteration:
        return None

    body = df.iloc[h + 1:]
    country = body[c_country]
    values = body[c_crude].map(lambda v: parse_number(v, locale="tr"))
    is_total = country.str.contains("TOPLAM")

    russia = values[country.str.contains("RUSYA")].sum(min_count=1)
    # Birden fazla TOPLAM satırı olabilir (ara toplam + genel toplam): toplamak çift sayar, en büyüğü alınır.
    total = values[is_total].max() if is_total.any() else values[~is_total].sum(min_count=1)
    if pd.isna(total) or total <= 0:
        return None
    return float(total), float(0.0 if pd.isna(russia) else russia)


def parse_pdf(path: Path) -> Optional[Tuple[float, float]]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                flat = " ".join(clean_header(c) for row in table for c in row if c)
                if "HAM PETROL" not in flat:
                    continue
                parsed = parse_import_table(table)
                if parsed:
                    return parsed
    return None


def _extract_docx_tables_xml(path: Path) -> List[List[List[str]]]:
    """python-docx olmadan standart kütüphane (zipfile + ElementTree) ile DOCX tablolarını okur."""
    try:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return []

    def _cell_text(tc: ET.Element) -> str:
        parts = []
        for p in tc.findall(f".//{{{_OOXML_W}}}p"):
            txt = "".join(t.text or "" for t in p.iter(f"{{{_OOXML_W}}}t"))
            if txt:
                parts.append(txt)
        if not parts:
            return "".join(t.text or "" for t in tc.iter(f"{{{_OOXML_W}}}t"))
        return " ".join(parts)

    tables: List[List[List[str]]] = []
    for tbl in root.iter(f"{{{_OOXML_W}}}tbl"):
        grid_rows: List[List[str]] = []
        prev_row: List[str] = []
        for tr in tbl.findall(f"{{{_OOXML_W}}}tr"):
            row: List[str] = []
            col_idx = 0
            for tc in tr.findall(f"{{{_OOXML_W}}}tc"):
                tc_pr = tc.find(f"{{{_OOXML_W}}}tcPr")
                span = 1
                vmerge = None
                if tc_pr is not None:
                    gs = tc_pr.find(f"{{{_OOXML_W}}}gridSpan")
                    if gs is not None:
                        try:
                            span = max(1, int(gs.attrib.get(f"{{{_OOXML_W}}}val", "1")))
                        except ValueError:
                            span = 1
                    vm = tc_pr.find(f"{{{_OOXML_W}}}vMerge")
                    if vm is not None:
                        vmerge = vm.attrib.get(f"{{{_OOXML_W}}}val", "continue")
                txt = _cell_text(tc)
                # Çok satırlı başlık bloklarında (ilk 5 satır) dikey birleşmiş hücre başlığını alt satıra taşı.
                if (
                    len(grid_rows) < 5
                    and vmerge is not None
                    and vmerge != "restart"
                    and not txt
                    and col_idx < len(prev_row)
                ):
                    txt = prev_row[col_idx]
                for _ in range(span):
                    row.append(txt)
                    col_idx += 1
            if row:
                grid_rows.append(row)
                prev_row = row
        if grid_rows:
            tables.append(grid_rows)
    return tables


def _iter_docx_tables_lib(container) -> List[List[List[str]]]:
    extracted: List[List[List[str]]] = []
    for tbl in getattr(container, "tables", []):
        rows = [[cell.text for cell in row.cells] for row in tbl.rows]
        if rows:
            extracted.append(rows)
        for row in tbl.rows:
            for cell in row.cells:
                if getattr(cell, "tables", None):
                    extracted.extend(_iter_docx_tables_lib(cell))
    return extracted


def parse_docx(path: Path) -> Optional[Tuple[float, float]]:
    """EPDK Word (.docx) raporundan 'Ülke / Ham Petrol' tablosunu okur (std-lib XML fallback dahil)."""
    candidate_tables: List[List[List[str]]] = []
    try:
        import docx  # type: ignore[import-not-found]

        doc = docx.Document(str(path))
        candidate_tables = _iter_docx_tables_lib(doc)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("python-docx %s okuyamadı, XML fallback deneniyor: %s", path.name, exc)

    for table in candidate_tables:
        flat = " ".join(clean_header(c) for row in table for c in row if c)
        if "HAM PETROL" not in flat:
            continue
        parsed = parse_import_table(table)
        if parsed:
            return parsed

    # python-docx yüklü değilse veya SDT/revizyon etiketli tabloysa ham OOXML'e düş.
    for table in _extract_docx_tables_xml(path):
        flat = " ".join(clean_header(c) for row in table for c in row if c)
        if "HAM PETROL" not in flat:
            continue
        parsed = parse_import_table(table)
        if parsed:
            return parsed
    return None


def parse_xlsx(path: Path) -> Optional[Tuple[float, float]]:
    try:
        import openpyxl
    except ImportError:
        return None

    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:  # noqa: BLE001
        return None
    try:
        for name in wb.sheetnames:
            ws = wb[name]
            rows = [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
            if not rows:
                continue
            flat = " ".join(clean_header(c) for row in rows[:50] for c in row if c)
            if "HAM PETROL" not in flat:
                continue
            parsed = parse_import_table(rows)
            if parsed:
                return parsed
    finally:
        wb.close()
    return None


def _is_valid_pdf_bytes(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def _safe_header(headers: Optional[Mapping], name: str) -> str:
    if not isinstance(headers, Mapping):
        return ""
    for k, v in headers.items():
        if str(k).lower() == name.lower():
            return str(v) if isinstance(v, (str, bytes)) else ""
    return ""


def _detect_doc_type(data: bytes, headers: Optional[Mapping] = None) -> str:
    if not data:
        return "empty"
    head = data.lstrip()[:64]
    head_low = head.lower()
    if head.startswith(b"%PDF"):
        return "pdf"
    if head_low.startswith((b"<!doctype", b"<html", b"<head", b"<body", b"<script", b"<!--")) or (
        head_low.startswith(b"<?xml") and b"<html" in data[:512].lower()
    ):
        return "html"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = {n.lower() for n in zf.namelist()}
            if "word/document.xml" in names:
                return "docx"
            if "xl/workbook.xml" in names:
                return "xlsx"
            if any(
                n.endswith((".pdf", ".docx", ".doc", ".xlsx")) and not n.startswith("__macosx/")
                for n in names
            ):
                return "zip"
        except zipfile.BadZipFile:
            return "unknown"
    if data[:8] == _OLE2_MAGIC:
        ct = _safe_header(headers, "Content-Type").lower()
        cd = _safe_header(headers, "Content-Disposition").lower()
        if "ms-excel" in ct or ".xls" in cd or b"W\x00o\x00r\x00k\x00b\x00o\x00o\x00k" in data or b"B\x00o\x00o\x00k" in data:
            return "xls"
        return "doc"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    return "unknown"


def _detect_doc_type_path(path: Path) -> str:
    try:
        return _detect_doc_type(path.read_bytes())
    except OSError:
        return "unknown"


def _extract_report_from_zip(zip_bytes: bytes) -> Optional[Tuple[str, bytes]]:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [
                info
                for info in zf.infolist()
                if not info.is_dir()
                and not info.filename.lower().startswith("__macosx/")
                and ".." not in Path(info.filename).parts
                and _MIN_DOC_BYTES <= info.file_size <= _MAX_ZIP_MEMBER_BYTES
            ]
            priority = {".pdf": 0, ".docx": 1, ".xlsx": 2, ".doc": 3}
            members.sort(key=lambda i: priority.get(Path(i.filename).suffix.lower(), 99))
            for info in members:
                inner = zf.read(info.filename)
                inner_type = _detect_doc_type(inner)
                if inner_type in ("pdf", "docx", "xlsx", "doc"):
                    return inner_type, inner
    except zipfile.BadZipFile:
        return None
    return None


def _convert_doc_to_docx(doc_bytes: bytes, out_docx: Path, suffix: str = ".doc") -> bool:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src = tmp_path / f"input{suffix}"
        src.write_bytes(doc_bytes)
        proc = subprocess.run(
            [
                soffice,
                "--headless",
                "--norestore",
                f"-env:UserInstallation=file://{tmp_path}/profile",
                "--convert-to",
                "docx",
                "--outdir",
                str(tmp_path),
                str(src),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
        converted = tmp_path / "input.docx"
        if proc.returncode != 0 or not converted.exists():
            return False
        converted_bytes = converted.read_bytes()
        if _detect_doc_type(converted_bytes) != "docx":
            return False
        out_docx.write_bytes(converted_bytes)
        return True


def _parse_report_path(path: Path, dtype: str) -> Optional[Tuple[float, float]]:
    if dtype == "pdf":
        return parse_pdf(path)
    if dtype == "docx":
        return parse_docx(path)
    if dtype == "xlsx":
        return parse_xlsx(path)
    return None


def _response_diagnostic(res, data: bytes, dtype: str) -> str:
    status = getattr(res, "status_code", "?")
    if not isinstance(status, int):
        status = "?"
    hdrs = getattr(res, "headers", None)
    ct = _safe_header(hdrs, "Content-Type") or "?"
    cd = _safe_header(hdrs, "Content-Disposition") or "-"
    head_hex = data[:12].hex() if data else "-"
    return f"tür={dtype}, status={status}, ct={ct}, cd={cd}, len={len(data)}, head={head_hex}"


def _process_report(
    year: int,
    month: int,
    link: Union[str, Sequence[str]],
    cookies: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    from curl_cffi import requests as c_requests

    urls = [link] if isinstance(link, str) else list(dict.fromkeys(link))
    if not urls:
        return None

    out_dir = RAW_DIR / "epdk" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_candidates = (
        ("pdf", out_dir / f"epdk_{year}_{month:02d}.pdf"),
        ("docx", out_dir / f"epdk_{year}_{month:02d}.docx"),
        ("xlsx", out_dir / f"epdk_{year}_{month:02d}.xlsx"),
    )

    parsed: Optional[Tuple[float, float]] = None
    saw_valid_doc = False
    last_name = cache_candidates[0][1].name
    for expected_dtype, cand_path in cache_candidates:
        if cand_path.exists() and cand_path.stat().st_size >= _MIN_DOC_BYTES and _detect_doc_type_path(cand_path) == expected_dtype:
            saw_valid_doc = True
            last_name = cand_path.name
            parsed = _parse_report_path(cand_path, expected_dtype)
            if parsed is not None:
                break

    if parsed is None:
        last_err: Optional[Exception] = None
        with c_requests.Session(impersonate="chrome") as s:
            if hasattr(s, "headers") and isinstance(s.headers, Mapping):
                s.headers["Referer"] = EPDK_PAGE
            if cookies and hasattr(s, "cookies"):
                s.cookies.update(cookies)

            for idx, url in enumerate(urls):
                try:
                    res = s.get(url, timeout=60)
                    res.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    continue

                data = res.content or b""
                hdrs = getattr(res, "headers", None)
                dtype = _detect_doc_type(data, headers=hdrs) if len(data) >= _MIN_DOC_BYTES else "too_short"
                if dtype == "zip":
                    unpacked = _extract_report_from_zip(data)
                    if unpacked is not None:
                        dtype, data = unpacked
                    else:
                        dtype = "unknown_zip"

                target_path: Optional[Path] = None
                if dtype in ("doc", "rtf"):
                    docx_target = out_dir / f"epdk_{year}_{month:02d}.docx"
                    if _convert_doc_to_docx(data, docx_target, suffix=f".{dtype}"):
                        dtype = "docx"
                        target_path = docx_target
                    else:
                        diag = _response_diagnostic(res, data, dtype)
                        last_err = ValueError(
                            f"EPDK {year}-{month:02d} indirilen yanıt geçerli PDF değil "
                            f"(eski .{dtype} dosyası için LibreOffice 'soffice' gerekli; {diag})."
                        )
                        continue
                elif dtype in ("pdf", "docx", "xlsx"):
                    target_path = out_dir / f"epdk_{year}_{month:02d}.{dtype}"
                    target_path.write_bytes(data)
                else:
                    diag = _response_diagnostic(res, data, dtype)
                    last_err = ValueError(
                        f"EPDK {year}-{month:02d} indirilen yanıt geçerli PDF değil ({diag})."
                    )
                    continue

                saw_valid_doc = True
                last_name = target_path.name
                parsed = _parse_report_path(target_path, dtype)
                if parsed is not None:
                    break
                if idx < len(urls) - 1:
                    target_path.unlink(missing_ok=True)

        if parsed is None:
            if not saw_valid_doc and last_err is not None:
                raise last_err
            log.warning("EPDK %d-%02d: rapor içinde ithalat tablosu bulunamadı (%s)", year, month, last_name)
            return None

    total_t, russia_t = parsed
    total_kbd = tonnes_to_kbd_scalar(total_t, year, month)
    russia_kbd = tonnes_to_kbd_scalar(russia_t, year, month)
    return {
        "Tarih": f"{year}-{month:02d}-01",
        "TR_Toplam_Ithalat_Ton": total_t,
        "TR_Rusya_Ithalat_Ton": russia_t,
        "TR_Toplam_Ithalat_KBD": round(total_kbd, 3),
        "TR_Rusya_Ithalat_KBD": round(russia_kbd, 3),
        "TR_Rusya_Disi_KBD": round(max(0.0, total_kbd - russia_kbd), 3),
        "Rusya_Payi_Yuzde": round(russia_kbd / total_kbd * 100, 2),
    }


_EXCEL_HINTS = ("excel", ".xlsx", ".xls", "xlsx", "xls", ".csv")


def _link_meta_text(a) -> str:
    parts = [a.get("href") or "", a.get_text(" ", strip=True), a.get("title") or "", a.get("aria-label") or ""]
    parts.extend(a.get("class") or [])
    for tag in a.find_all(["img", "i", "span", "svg"]):
        parts.extend(
            [
                tag.get("src") or "",
                tag.get("alt") or "",
                tag.get("title") or "",
                " ".join(tag.get("class") or []),
            ]
        )
    return tr_lower(" ".join(str(p) for p in parts if p))


def _row_non_sibling_anchor_text(container, anchor) -> str:
    """Kapsayıcıdaki diğer kardeş <a> linklerinin metnini dışlayarak metin toplar."""
    parts: List[str] = []
    for node in container.descendants:
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue
        if getattr(node.parent, "name", None) in ("script", "style"):
            continue
        parent_a = node.find_parent("a")
        if parent_a is None or parent_a is anchor:
            txt = " ".join(str(node).split())
            if txt:
                parts.append(txt)
    return " ".join(parts)


def _link_candidates(a) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def _add(txt: Optional[str]) -> None:
        clean = " ".join((txt or "").split())
        if clean and clean not in seen:
            seen.add(clean)
            candidates.append(clean)

    _add(a.get_text(" ", strip=True))
    _add(a.get("title"))

    # Aynı satır/paragraf içinde <br> ile ayrılmış linklerde önceki metni yakala (başka <a> görünce dur).
    sib_chunks: List[str] = []
    for sib in a.previous_siblings:
        name = getattr(sib, "name", None)
        if name in ("a", "br", "hr", "p", "tr", "li", "div"):
            break
        txt = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib).strip()
        if txt:
            sib_chunks.append(txt)
        if len(sib_chunks) >= 3:
            break
    if sib_chunks:
        _add(" ".join(reversed(sib_chunks)) + " " + (a.get_text(" ", strip=True) or ""))

    for parent in a.parents:
        if parent.name in ("[document]", "html", "body", "table", "tbody"):
            break
        if parent.name == "div" and len(parent.find_all("a", href=True)) > 2:
            break
        if parent.name in ("span", "strong", "b", "td", "p", "li", "tr", "div"):
            if len(parent.find_all("a", href=True)) > 1:
                _add(_row_non_sibling_anchor_text(parent, a))
            txt = parent.get_text(" ", strip=True)
            if len(txt) <= 1000:
                _add(txt)
            if parent.name in ("tr", "li", "p"):
                break
    return candidates


def _collect_report_links(html: str, start_year: int) -> Tuple[Dict[Tuple[int, int], str], Dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    ranked: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
    stats: Dict[str, object] = {
        "doc_links": 0,
        "skipped_il_bazinda": 0,
        "skipped_non_pdf": 0,
        "skipped_no_date": 0,
        "skipped_old": 0,
        "samples": [],
        "candidates_by_ym": {},
    }

    for idx, a in enumerate(soup.find_all("a", href=True)):
        href = a["href"]
        href_low = href.lower()
        if "downloaddocument" not in href_low and not href_low.endswith((".pdf", ".docx", ".doc")):
            continue
        stats["doc_links"] = int(stats["doc_links"]) + 1

        candidates = _link_candidates(a)
        matched_text = None
        ym = None
        for cand in candidates:
            cand_ym = detect_report_month(cand)
            if cand_ym is not None:
                ym = cand_ym
                matched_text = cand
                break

        context_text = matched_text or (candidates[-1] if candidates else "")
        low = tr_lower(context_text)
        own_low = tr_lower(" ".join(p for p in (a.get_text(" ", strip=True), a.get("title") or "") if p))
        # Sadece tek satırlı kapsayıcılarda ("tr", "li", "p") kardeş <a> linkleri dışındaki metne bak.
        row_only = a.find_parent(["tr", "li", "p"])
        excl_text = tr_lower(_row_non_sibling_anchor_text(row_only, a)) if row_only is not None else low
        if (
            "il bazında" in own_low
            or "bayi" in own_low
            or "il bazında" in excl_text
            or "bayi" in excl_text
        ):
            stats["skipped_il_bazinda"] = int(stats["skipped_il_bazinda"]) + 1
            continue

        meta = _link_meta_text(a)
        has_word_hint = (
            any(h in meta for h in ("word", ".docx", "docx", ".doc"))
            or href_low.endswith((".docx", ".doc"))
        )
        is_excel = any(h in meta for h in _EXCEL_HINTS) and "pdf" not in meta and not has_word_hint
        if is_excel:
            stats["skipped_non_pdf"] = int(stats["skipped_non_pdf"]) + 1
            continue

        if ym is None:
            stats["skipped_no_date"] = int(stats["skipped_no_date"]) + 1
            log.debug("EPDK link atlandı (ay/yıl belirsiz): %s", context_text[:80])
            if len(stats["samples"]) < 5:  # type: ignore[arg-type]
                stats["samples"].append(f"no_date: {context_text[:70]} [{href[:40]}]")  # type: ignore[attr-defined]
            continue

        if ym[0] < start_year:
            stats["skipped_old"] = int(stats["skipped_old"]) + 1
            continue

        has_pdf_hint = "pdf" in meta or href_low.endswith(".pdf") or ("pdf" in tr_lower(" ".join(candidates)))
        # Öncelik: 0=PDF, 1=Word/Rapor, 2=diğer tarihli DownloadDocument linkleri.
        # Gerçek biçim (PDF / DOCX / XLSX / ZIP) _process_report içinde dosya imzasıyla doğrulanır.
        priority = 0 if has_pdf_hint else (1 if (has_word_hint or "rapor" in low) else 2)
        url = "https://www.epdk.gov.tr" + href if href.startswith("/") else href
        ranked.setdefault(ym, []).append((priority, idx, url))

    links: Dict[Tuple[int, int], str] = {}
    candidates_by_ym: Dict[Tuple[int, int], List[str]] = {}
    for ym, items in ranked.items():
        ordered = [u for _, _, u in sorted(items, key=lambda t: (t[0], t[1]))]
        deduped = list(dict.fromkeys(ordered))
        links[ym] = deduped[0]
        candidates_by_ym[ym] = deduped
    stats["candidates_by_ym"] = candidates_by_ym
    return links, stats


def find_report_links(html: str, start_year: int) -> Dict[Tuple[int, int], str]:
    links, _ = _collect_report_links(html, start_year)
    return links


def update(max_workers: int = 2) -> None:
    from curl_cffi import requests as c_requests

    log.info("EPDK veri çekme başlıyor...")
    master = FILES["epdk"]
    if master.exists():
        df_old = normalize_epdk_columns(pd.read_csv(master))
    else:
        df_old = pd.DataFrame(columns=["Tarih", "TR_Toplam_Ithalat_KBD"])

    valid_done = (
        df_old.dropna(subset=["Tarih", "TR_Toplam_Ithalat_KBD"])["Tarih"].astype(str)
        if "TR_Toplam_Ithalat_KBD" in df_old.columns
        else df_old["Tarih"].dropna().astype(str)
    )
    done = set(valid_done)

    with c_requests.Session(impersonate="chrome") as s:
        res = s.get(EPDK_PAGE, timeout=30)
        res.raise_for_status()
        cookies = dict(s.cookies)
    start_year = pd.Timestamp(START_DATE).year
    links, link_stats = _collect_report_links(res.text, start_year)
    candidates_by_ym: Dict[Tuple[int, int], List[str]] = link_stats["candidates_by_ym"]  # type: ignore[assignment]
    todo = {ym: candidates_by_ym.get(ym, [l]) for ym, l in links.items() if f"{ym[0]}-{ym[1]:02d}-01" not in done}
    log.info("EPDK: %d rapor linki bulundu, %d yeni.", len(links), len(todo))
    if len(links) < 24 and pd.Timestamp.now().year - start_year >= 2:
        log.warning(
            "EPDK beklenenden az rapor linki (%d); istatistikler -> toplam_doc=%s, il_bazinda=%s, "
            "excel=%s, tarih_yok=%s, eski=%s | örnek=%s",
            len(links),
            link_stats["doc_links"],
            link_stats["skipped_il_bazinda"],
            link_stats["skipped_non_pdf"],
            link_stats["skipped_no_date"],
            link_stats["skipped_old"],
            link_stats["samples"],
        )
    if not todo:
        return

    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_report, y, m, urls, cookies): (y, m) for (y, m), urls in todo.items()}
        for f in as_completed(futures):
            y, m = futures[f]
            try:
                r = f.result()
                rows.append(r) if r else failed.append(f"{y}-{m:02d}")
            except Exception as e:  # noqa: BLE001 - tek rapor hatası tüm akışı durdurmamalı
                failed.append(f"{y}-{m:02d}")
                log.error("EPDK %d-%02d işlenemedi: %s", y, m, e)

    if failed:
        log.warning("EPDK başarısız aylar: %s", ", ".join(sorted(failed)))
    if rows:
        combined = pd.concat([df_old, pd.DataFrame(rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Tarih"], keep="last").sort_values("Tarih")
        atomic_write_csv(combined, master, index=False)
        log.info("EPDK güncellendi: %d yeni ay.", len(rows))
