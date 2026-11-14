"""EPDK Petrol Piyasası Aylık Sektör Raporları: Türkiye ham petrol ithalatı (Rusya kırılımlı)."""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
_MONTH_RE = re.compile(r"(?<![a-zçğıöşü])(" + "|".join(MONTHS_TR) + r")(?![a-zçğıöşü])")
_YEAR_RE = re.compile(r"(?<!\d)(20[1-3]\d)(?!\d)")


def detect_report_month(text: str) -> Optional[Tuple[int, int]]:
    """Link metninden (yıl, ay) çıkarır. Belirsizse (birden fazla ay/yıl) None döner."""
    t = tr_lower(text)
    months = set(_MONTH_RE.findall(t))
    years = set(_YEAR_RE.findall(t))
    if len(months) != 1 or len(years) != 1:
        return None
    return int(years.pop()), MONTHS_TR[months.pop()]


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


def _process_report(year: int, month: int, link: str) -> Optional[Dict]:
    from curl_cffi import requests as c_requests

    out_dir = RAW_DIR / "epdk" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"epdk_{year}_{month:02d}.pdf"

    valid_cached = (
        pdf_path.exists()
        and pdf_path.stat().st_size >= 1024
        and _is_valid_pdf_bytes(pdf_path.read_bytes()[:16])
    )
    if not valid_cached:
        with c_requests.Session(impersonate="chrome") as s:
            res = s.get(link, timeout=60)
            res.raise_for_status()
            if len(res.content) < 1024 or not _is_valid_pdf_bytes(res.content[:16]):
                raise ValueError(
                    f"EPDK {year}-{month:02d} indirilen yanıt geçerli PDF değil (HTML hata sayfası olabilir)."
                )
            pdf_path.write_bytes(res.content)

    parsed = parse_pdf(pdf_path)
    if not parsed:
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


def find_report_links(html: str, start_year: int) -> Dict[Tuple[int, int], str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Dict[Tuple[int, int], str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "DownloadDocument" not in href:
            continue

        candidates = [a.get_text(" ", strip=True)]
        parent = a.find_parent()
        if parent:
            candidates.append(parent.get_text(" ", strip=True))
        row_container = a.find_parent(["tr", "li", "div"])
        if row_container and row_container is not parent:
            candidates.append(row_container.get_text(" ", strip=True))

        matched_text = None
        ym = None
        for cand in candidates:
            if not cand:
                continue
            cand_ym = detect_report_month(cand)
            if cand_ym is not None:
                ym = cand_ym
                matched_text = cand
                break

        context_text = matched_text or (candidates[1] if len(candidates) > 1 else candidates[0])
        low = tr_lower(context_text)
        if "il bazında" in low or "bayi" in low:
            continue

        img = a.find("img")
        is_pdf = (
            (img is not None and "pdf" in (img.get("src") or "").lower())
            or "pdf" in tr_lower(" ".join(candidates[:2]))
            or href.lower().endswith(".pdf")
        )
        if not is_pdf:
            continue

        if ym is None:
            log.debug("EPDK link atlandı (ay/yıl belirsiz): %s", context_text[:80])
            continue
        if ym[0] >= start_year:
            links.setdefault(ym, "https://www.epdk.gov.tr" + href if href.startswith("/") else href)
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
    links = find_report_links(res.text, pd.Timestamp(START_DATE).year)
    todo = {ym: l for ym, l in links.items() if f"{ym[0]}-{ym[1]:02d}-01" not in done}
    log.info("EPDK: %d rapor linki bulundu, %d yeni.", len(links), len(todo))
    if not todo:
        return

    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_report, y, m, l): (y, m) for (y, m), l in todo.items()}
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
