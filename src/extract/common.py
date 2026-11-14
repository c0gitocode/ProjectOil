"""Extract katmanı ortak yardımcıları."""
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
# Hem standart 3'lü binlik ('1,234,567') hem Hint gruplaması ('12,34,567')
_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{2,3})*,\d{3}$")
_THOUSANDS_DOT = re.compile(r"^-?\d{1,3}(\.\d{3})+$")
NumberLocale = Literal["auto", "tr", "en"]


def create_retry_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def tr_lower(text: Any) -> str:
    """Türkçe uyumlu küçük harf: 'KASIM' -> 'kasım', 'NİSAN' -> 'nisan'.

    Python'un str.lower() fonksiyonu 'I' -> 'i' ve 'İ' -> 'i̇' (birleşik nokta) üretir,
    bu da 'kasım', 'aralık', 'nisan' gibi ay isimlerinin eşleşmemesine yol açıyordu.
    """
    return str(text).replace("I", "ı").replace("İ", "i").lower()


def tr_upper(text: Any) -> str:
    return str(text).replace("i", "İ").replace("ı", "I").upper()


def clean_header(c: Any) -> str:
    return re.sub(r"\s+", " ", tr_upper(c)).strip()


def normalize_epdk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Eski EPDK sürüm sütunlarını tek bir şemada birleştirir (çift sütun üretmeden)."""
    df = df.copy()
    legacy_map = {
        "TR_Rusya_Iskontosu_KBD": "TR_Rusya_Ithalat_KBD",
        "TR_Saf_Petrodolar_KBD": "TR_Rusya_Disi_KBD",
    }
    for old, new in legacy_map.items():
        if old not in df.columns:
            continue
        if new in df.columns:
            df[new] = pd.to_numeric(df[new], errors="coerce").combine_first(
                pd.to_numeric(df[old], errors="coerce")
            )
            df = df.drop(columns=[old])
        else:
            df = df.rename(columns={old: new})
    return df


def parse_number(val: Any, locale: NumberLocale = "auto") -> float:
    """Sayıları locale'e uygun float'a çevirir; okunamazsa NaN döner.

    * locale='tr': '1.234' -> 1234.0, '1.234.567,89' -> 1234567.89, '12,5' -> 12.5
    * locale='en': '1.234' -> 1.234, '1,234,567.89' -> 1234567.89, '12,34,567' -> 1234567.0
    * locale='auto': İki ayraç varsa son ayraca bakar; çoklu nokta/virgülü binlik sayar.
    """
    if val is None or isinstance(val, bool):
        return float("nan")
    if not isinstance(val, (str, bytes)):
        if pd.isna(val):
            return float("nan")
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(val)
        except (TypeError, ValueError):
            pass

    s = str(val).strip().replace(" ", "").replace("\u00a0", "")
    if s in ("", "-", "--", "x", "X", "nan", "NaN", "NA", "N/A", "null"):
        return float("nan")

    has_comma, has_dot = "," in s, "." in s
    if has_comma and has_dot:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        if locale == "tr":
            s = s.replace(",", ".")
        elif locale == "en":
            if _THOUSANDS_COMMA.match(s) or s.count(",") > 1:
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
        else:
            if _THOUSANDS_COMMA.match(s) or s.count(",") > 1:
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
    elif has_dot:
        if locale == "tr" and _THOUSANDS_DOT.match(s):
            s = s.replace(".", "")
        elif locale == "auto" and s.count(".") > 1 and _THOUSANDS_DOT.match(s):
            s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return float("nan")


def atomic_write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Önce geçici dosyaya yazar, sonra yer değiştirir: yarıda kesilen yazma master dosyayı bozmaz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
