"""EPDK Petrol Piyasası Aylık Sektör Raporları: Türkiye ham petrol ithalatı (Rusya kırılımlı)."""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from bs4 import BeautifulSoup

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


def _is_valid_pdf_bytes(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def _process_report(year: int, month: int, link: Union[str, Sequence[str]]) -> Optional[Dict]:
    from curl_cffi import requests as c_requests

    urls = [link] if isinstance(link, str) else list(dict.fromkeys(link))
    if not urls:
        return None

    out_dir = RAW_DIR / "epdk" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"epdk_{year}_{month:02d}.pdf"

    parsed: Optional[Tuple[float, float]] = None
    valid_cached = (
        pdf_path.exists()
        and pdf_path.stat().st_size >= 1024
        and _is_valid_pdf_bytes(pdf_path.read_bytes()[:16])
    )
    if valid_cached:
        parsed = parse_pdf(pdf_path)

    if parsed is None:
        saw_valid_pdf = valid_cached
        last_err: Optional[Exception] = None
        with c_requests.Session(impersonate="chrome") as s:
            for idx, url in enumerate(urls):
                try:
                    res = s.get(url, timeout=60)
                    res.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    continue
                if len(res.content) < 1024 or not _is_valid_pdf_bytes(res.content[:16]):
                    last_err = ValueError(
                        f"EPDK {year}-{month:02d} indirilen yanıt geçerli PDF değil (HTML hata sayfası olabilir)."
                    )
                    continue
                saw_valid_pdf = True
                pdf_path.write_bytes(res.content)
                parsed = parse_pdf(pdf_path)
                if parsed is not None:
                    break
                if idx < len(urls) - 1:
                    pdf_path.unlink(missing_ok=True)

        if parsed is None:
            if not saw_valid_pdf and last_err is not None:
                raise last_err
            log.warning("EPDK %d-%02d: PDF içinde ithalat tablosu bulunamadı (%s)", year, month, pdf_path.name)
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


_NON_PDF_HINTS = ("excel", ".xlsx", ".xls", "xlsx", "xls", ".csv", ".zip", ".docx", ".doc")


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
        if "downloaddocument" not in href_low and not href_low.endswith(".pdf"):
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
        # Sadece tek satırlı kapsayıcılarda ("tr", "li", "p") tüm adaylara bak; div satırları paylaşabilir.
        row_only = a.find_parent(["tr", "li", "p"])
        excl_text = tr_lower(row_only.get_text(" ", strip=True)) if row_only is not None else low
        if "il bazında" in excl_text or "bayi" in excl_text:
            stats["skipped_il_bazinda"] = int(stats["skipped_il_bazinda"]) + 1
            continue

        meta = _link_meta_text(a)
        is_non_pdf = any(h in meta for h in _NON_PDF_HINTS) and "pdf" not in meta
        if is_non_pdf:
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
        # Öncelik: 0=açık PDF etiketi/ikonu, 1=başlıkta "rapor" geçen DownloadDocument, 2=diğer tarihli linkler.
        # Gerçek PDF olup olmadığı _process_report içinde b"%PDF" ile kesin doğrulanır.
        priority = 0 if has_pdf_hint else (1 if "rapor" in low else 2)
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


def update(max_workers: int = 3) -> None:
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
    start_year = pd.Timestamp(START_DATE).year
    links, link_stats = _collect_report_links(res.text, start_year)
    candidates_by_ym: Dict[Tuple[int, int], List[str]] = link_stats["candidates_by_ym"]  # type: ignore[assignment]
    todo = {ym: candidates_by_ym.get(ym, [l]) for ym, l in links.items() if f"{ym[0]}-{ym[1]:02d}-01" not in done}
    log.info("EPDK: %d rapor linki bulundu, %d yeni.", len(links), len(todo))
    if len(links) < 24 and pd.Timestamp.now().year - start_year >= 2:
        log.warning(
            "EPDK beklenenden az rapor linki (%d); istatistikler -> toplam_doc=%s, il_bazinda=%s, "
            "pdf_degil=%s, tarih_yok=%s, eski=%s | örnek=%s",
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
        futures = {ex.submit(_process_report, y, m, urls): (y, m) for (y, m), urls in todo.items()}
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
