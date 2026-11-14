"""Transform katmanı.

Tasarım: her hesap saf (pure) bir `compute_*` fonksiyonudur (DataFrame alır, DataFrame döndürür,
dosyaya dokunmaz) ve test edilebilir. `run_*` fonksiyonları sadece dosya okuma/yazma yapar.

Ekonometrik ilkeler:
* Eksik veri 0 ile doldurulmaz ve ileriye taşınmaz (ffill yok). Eksik ay NaN kalır, global
  seride ise tam-gözlem (complete-case) örneklemi kullanılır ve düşen aylar loglanır.
* JODI 'Diğer' toplamında örneklem boyunca sabit ülke sepeti (balanced basket) korunur;
  düzensiz raporlayan ülkelerin giriş-çıkışından doğan sahte sıçramalar önlenir.
* Tüm akımlar KBD (bin varil/gün) birimindedir; dönüşümler sadece `units.py`'dedir.
* Ödeme para birimi payları kodda değil, `data/assumptions/*.csv` dosyalarındadır.
"""
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import (
    ASSUMPTION_FILES,
    BAND_START,
    BBL_PER_TONNE,
    CHINA_RAW_DIR,
    END_DATE,
    FILES,
    JODI_REQUIRED_COVERAGE,
    RAW_DIR,
    SENSITIVITY,
    START_DATE,
)
from .extract.common import atomic_write_csv, normalize_epdk_columns, parse_number
from .logger import get_logger
from .units import kg_to_kbd, tonnes_to_kbd

log = get_logger(__name__)

# Ayrı ulusal kaynaklardan (DGCIS/GACC/EIA/EPDK) modellenen ülkeler JODI toplamına ve paneline
# tekrar dahil edilmez (çift sayım önlemi).
JODI_EXCLUDED_FROM_OTHER = {"IN", "IND", "CN", "CHN", "US", "USA", "TR", "TUR"}

# GACC partner adı (küçük harf) -> varsayım grubu
CHINA_PARTNER_GROUPS = {
    "russia": "russia",
    "russian federation": "russia",
    "iran": "iran_venezuela",
    "iran, islamic rep.": "iran_venezuela",
    "venezuela": "iran_venezuela",
    "malaysia": "malaysia",
    "indonesia": "indonesia",
    "saudi arabia": "saudi",
    "united arab emirates": "uae",
    "uae": "uae",
    "iraq": "iraq",
    "oman": "oman",
    "brazil": "brazil",
}

SCENARIOS = ("", "_High", "_Low")


# =========================================================
# Ortak: varsayım (pay) tabloları
# =========================================================
def load_shares(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["effective_from"])
    bad = df[(df["alt_share"] < 0) | (df["alt_share"] > 1)]
    if not bad.empty:
        raise ValueError(f"{path.name}: alt_share [0,1] dışında değerler içeriyor:\n{bad}")
    return df


def shares_for_dates(shares: pd.DataFrame, dates: Iterable) -> pd.DataFrame:
    """Parça-sabit (piecewise constant) payları verilen aylara yayar. Son değer ileri taşınır."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique().sort_values()
    if len(dates) == 0:
        cols = sorted(shares["country_group"].unique()) if not shares.empty else []
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    wide = shares.pivot_table(index="effective_from", columns="country_group", values="alt_share", aggfunc="last")
    out = wide.reindex(wide.index.union(dates)).sort_index().ffill().reindex(dates).fillna(0.0)
    out.index.name = "Date"
    return out


def share_scenarios(
    base: pd.DataFrame,
    sensitivity: float = SENSITIVITY,
    band_start: str = BAND_START,
) -> Dict[str, pd.DataFrame]:
    """Baz, yüksek ve düşük senaryo payları. Bant sadece band_start sonrası uygulanır
    (öncesinde varsayım belirsizliği yok; eski kod 2018'den itibaren tüm ülkelere +%5 ekliyordu)."""
    active = pd.DataFrame(
        np.broadcast_to((base.index >= pd.Timestamp(band_start))[:, None], base.shape),
        index=base.index,
        columns=base.columns,
    )
    high = base.where(~active, (base + sensitivity).clip(0, 1))
    low = base.where(~active, (base - sensitivity).clip(0, 1))
    return {"": base, "_High": high, "_Low": low}


# =========================================================
# 1. TÜRKİYE (TÜİK Excel + EPDK)
# =========================================================
def _clean_tuik_excel(path: Path, value_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    upper = raw.astype(str).apply(lambda col: col.str.upper().str.strip())
    mask = upper.apply(lambda r: "YIL" in r.values and "AY" in r.values, axis=1)
    header_idx = int(mask.idxmax()) if mask.any() else 0

    df = pd.read_excel(path, skiprows=header_idx)
    df = df.rename(columns={df.columns[0]: "Yil", df.columns[1]: "Ay", df.columns[-1]: "Deger"})
    df["Yil"] = pd.to_numeric(df["Yil"].ffill(), errors="coerce")
    df["Ay"] = pd.to_numeric(df["Ay"], errors="coerce")
    df["Deger"] = pd.to_numeric(df["Deger"], errors="coerce")
    df = df.dropna(subset=["Yil", "Ay", "Deger"])
    df["Tarih"] = pd.to_datetime(dict(year=df["Yil"].astype(int), month=df["Ay"].astype(int), day=1))

    indicator = df.columns[2]
    if any(k in str(indicator).lower() for k in ("ödeme", "odeme", "ruble", "yuan")):
        df = df.pivot_table(index="Tarih", columns=indicator, values="Deger", aggfunc="sum")
        df.columns = [
            "RUB_USD" if "Ruble" in str(c) else "CNY_USD" if "Yuan" in str(c) else f"Doviz_{c}"
            for c in df.columns
        ]
        return df.reset_index()
    return df.groupby("Tarih", as_index=False)["Deger"].sum().rename(columns={"Deger": value_name})


def _read_epdk() -> pd.DataFrame:
    df = pd.read_csv(FILES["epdk"], parse_dates=["Tarih"])
    return normalize_epdk_columns(df)


def run_turkey_master() -> None:
    def collect(pattern: str, name: str, exclude: Optional[str] = None) -> pd.DataFrame:
        files = [f for f in RAW_DIR.glob(pattern) if not (exclude and exclude in f.name.lower())]
        if not files:
            return pd.DataFrame(columns=["Tarih"])
        return (
            pd.concat([_clean_tuik_excel(f, name) for f in files], ignore_index=True)
            .groupby("Tarih", as_index=False)
            .sum(numeric_only=True)
        )

    frames = [
        collect("*ruble-yuan*.xlsx", "Doviz"),
        collect("*rusya*.xlsx", "Rusya_Toplam_USD", exclude="petrol"),
        collect("*fasil27*.xlsx", "Enerji_Fasil27_USD"),
    ]
    if FILES["epdk"].exists():
        epdk_df = _read_epdk()
        keep = [
            c
            for c in ("Tarih", "TR_Toplam_Ithalat_KBD", "TR_Rusya_Ithalat_KBD", "TR_Rusya_Disi_KBD")
            if c in epdk_df.columns
        ]
        if len(keep) > 1:
            frames.append(epdk_df[keep])

    frames = [f for f in frames if len(f.columns) > 1]
    if not frames:
        log.warning("Türkiye master: girdi dosyası yok, atlandı.")
        return
    master = reduce(lambda l, r: pd.merge(l, r, on="Tarih", how="outer"), frames).sort_values("Tarih")
    atomic_write_csv(master, FILES["tr_master"], index=False)  # NaN korunur (fillna(0) yok)
    log.info("Türkiye master yazıldı: %d ay", len(master))


# =========================================================
# 2. ÇİN (GACC)
# =========================================================
def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "utf-16", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"{path.name} hiçbir kodlamayla okunamadı")


def compute_china_imports(raw: pd.DataFrame) -> pd.DataFrame:
    """GACC ham tablo(lar)ı -> uzun format: Date, Partner, Quantity_kg, USD, KBD, USD_per_bbl."""
    df = raw.rename(columns=lambda c: str(c).strip())
    required = {"Date of data", "Trading partner", "Quantity"}
    if missing := required - set(df.columns):
        raise KeyError(f"GACC dosyasında eksik sütunlar: {missing}")

    date_str = df["Date of data"].astype(str).str.replace(r"[-/]", "", regex=True).str.strip().str[:6]
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(date_str, format="%Y%m", errors="coerce"),
            "Partner": df["Trading partner"].astype(str).str.strip(),
            "Quantity_kg": df["Quantity"].map(lambda v: parse_number(v, locale="en")),
            "USD": df["US dollar"].map(lambda v: parse_number(v, locale="en")) if "US dollar" in df.columns else np.nan,
        }
    ).dropna(subset=["Date"])

    p_low = out["Partner"].str.lower()
    out = out[~p_low.isin(["world", "", "nan", "none"])]
    # Yıllık dosyalar çakışırsa (ör. güncel yıl yeniden indirildi) son okunan geçerli
    out = out.drop_duplicates(subset=["Date", "Partner"], keep="last")

    world = out.groupby("Date", as_index=False)[["Quantity_kg", "USD"]].sum(min_count=1)
    world["Partner"] = "World"
    out = pd.concat([out, world], ignore_index=True)

    out["KBD"] = kg_to_kbd(out["Quantity_kg"], out["Date"])
    barrels = out["Quantity_kg"] / 1000.0 * BBL_PER_TONNE
    out["USD_per_bbl"] = (out["USD"] / barrels).where(barrels > 0)
    return out.sort_values(["Date", "Partner"]).reset_index(drop=True)


def compute_china_payment(imports: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Ülke-grubu bazında petroyuan/petrodolar ayrıştırması (vektörel, ±bant senaryolu)."""
    partners = imports[imports["Partner"] != "World"].copy()
    partners["group"] = partners["Partner"].str.lower().map(CHINA_PARTNER_GROUPS)

    total = imports[imports["Partner"] == "World"].set_index("Date")["KBD"].sort_index()
    by_group = (
        partners.dropna(subset=["group"])
        .pivot_table(index="Date", columns="group", values="KBD", aggfunc="sum")
        .reindex(total.index)
        .fillna(0.0)
    )

    base = shares_for_dates(shares, total.index)
    unknown = set(base.columns) - set(by_group.columns)
    if unknown:
        log.debug("Çin: verisi olmayan varsayım grupları: %s", unknown)
    by_group = by_group.reindex(columns=base.columns, fill_value=0.0)

    out = pd.DataFrame(index=total.index)
    out["China_Total_KBD"] = total
    py = {s: (by_group * sh).sum(axis=1) for s, sh in share_scenarios(base).items()}
    out["China_Petroyuan_KBD"] = py[""]
    out["China_Petroyuan_High_KBD"] = py["_High"]
    out["China_Petroyuan_Low_KBD"] = py["_Low"]
    # Bant semantiği: düşük petroyuan senaryosu <-> yüksek petrodolar senaryosu
    out["China_Petrodolar_KBD"] = (total - py[""]).clip(lower=0)
    out["China_Petrodolar_High_KBD"] = (total - py["_Low"]).clip(lower=0)
    out["China_Petrodolar_Low_KBD"] = (total - py["_High"]).clip(lower=0)
    for g in by_group.columns:
        out[f"CN_{g}_KBD"] = by_group[g]
    out.index.name = "Date"
    return out.round(3)


def run_china() -> bool:
    files = sorted(CHINA_RAW_DIR.glob("*_china.csv"))
    if not files:
        log.warning(
            "Çin ham verisi yok (%s). GACC adımını çalıştırın veya CSV'leri manuel koyun.",
            CHINA_RAW_DIR,
        )
        return False
    imports = compute_china_imports(pd.concat([_read_csv_any_encoding(f) for f in files], ignore_index=True))
    atomic_write_csv(imports, FILES["china_imports"], index=False)

    payment = compute_china_payment(imports, load_shares(ASSUMPTION_FILES["china"]))
    atomic_write_csv(payment, FILES["china_payment"])
    log.info("Çin: %d ay işlendi (%s – %s)", len(payment), payment.index.min().date(), payment.index.max().date())
    return True


# =========================================================
# 3. HİNDİSTAN (JODI toplam + DGCIS Rusya)
# =========================================================
def compute_dgcis_russia_kbd(dgcis: pd.DataFrame) -> pd.Series:
    if dgcis.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="Date"), name="DGCIS_Rus_KBD")

    df = dgcis.copy()
    df["Date"] = pd.to_datetime(df["Period"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="Date"), name="DGCIS_Rus_KBD")

    qty = df["Qty"].map(lambda v: parse_number(v, locale="en"))
    unit = df["Unit"].astype(str).str.upper().str.strip() if "Unit" in df.columns else pd.Series("TON", df.index)

    is_kg = unit.str.contains("KGS|KG")
    is_ton = unit.str.contains("TON|MTS|^MT$", regex=True)
    if (~(is_kg | is_ton)).any():
        log.warning("DGCIS: tanınmayan birimler ton varsayıldı: %s", sorted(unit[~(is_kg | is_ton)].unique()))
    tonnes = np.where(is_kg, qty / 1000.0, qty)
    df["KBD"] = tonnes_to_kbd(tonnes, df["Date"])
    return df.groupby("Date")["KBD"].sum(min_count=1).rename("DGCIS_Rus_KBD")


def compute_india(jodi_kbd: pd.DataFrame, dgcis_rus: pd.Series, shares: pd.DataFrame) -> pd.DataFrame:
    in_cols = [c for c in ("IN", "IND") if c in jodi_kbd.columns]
    total = jodi_kbd[in_cols].sum(axis=1, min_count=1) if in_cols else pd.Series(np.nan, index=jodi_kbd.index)
    df = pd.DataFrame({"Toplam_Ithalat_KBD": total}).join(dgcis_rus.rename("Rusya_Ithalat_KBD"), how="left")
    df.index.name = "Date"

    base = shares_for_dates(shares, df.index)["russia"].to_frame()
    sc = share_scenarios(base)
    rus = df["Rusya_Ithalat_KBD"]
    # Pay 0 olan aylarda (2022 öncesi) DGCIS eksikliği sonucu etkilemez; pay > 0 iken eksikse NaN kalır.
    no_effect = sc["_High"]["russia"].eq(0)
    rus_eff = rus.where(rus.notna() | ~no_effect, 0.0)

    over = rus > df["Toplam_Ithalat_KBD"]
    if over.any():
        log.warning(
            "Hindistan: %d ayda DGCIS Rusya ithalatı JODI toplamını aştı (%s); toplam ile sınırlandırılıyor.",
            int(over.sum()),
            ", ".join(pd.DatetimeIndex(df.index[over]).strftime("%Y-%m")[:6]),
        )

    df["Rusya_Disi_KBD"] = (df["Toplam_Ithalat_KBD"] - rus).clip(lower=0)
    cap = df["Toplam_Ithalat_KBD"]
    for suffix, sh in sc.items():
        raw_py = rus_eff * sh["russia"]
        df[f"Petroyuan{suffix}_KBD"] = np.where(cap.notna(), np.minimum(raw_py, cap), raw_py)
    df["Petrodolar_KBD"] = (df["Toplam_Ithalat_KBD"] - df["Petroyuan_KBD"]).clip(lower=0)
    df["Petrodolar_High_KBD"] = (df["Toplam_Ithalat_KBD"] - df["Petroyuan_Low_KBD"]).clip(lower=0)
    df["Petrodolar_Low_KBD"] = (df["Toplam_Ithalat_KBD"] - df["Petroyuan_High_KBD"]).clip(lower=0)
    df["Alt_Share"] = base["russia"]
    return df.round(3)


def run_india() -> bool:
    if not FILES["jodi"].exists():
        log.warning("Hindistan: JODI dosyası yok.")
        return False
    jodi = pd.read_csv(FILES["jodi"], index_col=0, parse_dates=True)
    dgcis = pd.read_csv(FILES["dgcis"]) if FILES["dgcis"].exists() else pd.DataFrame(columns=["Period", "Qty"])
    if dgcis.empty:
        log.warning("Hindistan: DGCIS verisi yok; 2022 sonrası petroyuan NaN kalacak.")
    df = compute_india(jodi, compute_dgcis_russia_kbd(dgcis), load_shares(ASSUMPTION_FILES["india"]))
    atomic_write_csv(df, FILES["india"])
    log.info("Hindistan: %d ay işlendi.", len(df))
    return True


# =========================================================
# 4. KÜRESEL BİRLEŞTİRME + PANEL
# =========================================================
def compute_jodi_other(
    jodi_kbd: pd.DataFrame,
    required_coverage: float = JODI_REQUIRED_COVERAGE,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """IN/CN/US/TR dışı JODI ülkelerinin dengeli (constant-basket) toplamı + kapsama kontrolü.

    'Zorunlu' ülkeler: analiz penceresinde >= required_coverage oranında veri bildirenler.
    Toplama yalnızca zorunlu ülkeler girer (düzensiz ülkelerin giriş-çıkışı sahte sıçrama
    yaratmasın). Zorunlu ülkelerden biri bir ayda eksikse o ayın toplamı NaN olur.
    """
    cols = [c for c in jodi_kbd.columns if c not in JODI_EXCLUDED_FROM_OTHER]
    sub = jodi_kbd[cols].sort_index()
    window = sub.loc[start:end] if (start or end) else sub

    coverage = window.notna().mean() if not window.empty else sub.notna().mean()
    required = coverage[coverage >= required_coverage].index.tolist()

    out = pd.DataFrame(index=sub.index)
    if required:
        complete = sub[required].notna().all(axis=1)
        out["JODI_DIGER_KBD"] = sub[required].sum(axis=1, min_count=1).where(complete)
        out["JODI_Reporting"] = sub[required].notna().sum(axis=1)
    else:
        out["JODI_DIGER_KBD"] = np.nan
        out["JODI_Reporting"] = 0

    dropped = sorted(set(cols) - set(required))
    if dropped:
        log.info("JODI: kapsama altında kalan düzensiz ülkeler hariç tutuldu: %s", dropped)
    return out


def compute_global(
    jodi_kbd: pd.DataFrame,
    us: pd.DataFrame,
    tr: pd.DataFrame,
    india: pd.DataFrame,
    china: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    def monthly(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
        return df.sort_index()[~df.sort_index().index.duplicated(keep="last")]

    india_cols = [f"Petro{b}{s}_KBD" for b in ("dolar", "yuan") for s in SCENARIOS]
    china_cols = [f"China_Petro{b}{s}_KBD" for b in ("dolar", "yuan") for s in SCENARIOS]

    parts = [
        monthly(compute_jodi_other(jodi_kbd, start=start, end=end)),
        monthly(us[["ABD_Ithalat_KBD"]]),
        monthly(tr[["TR_Toplam_Ithalat_KBD"]]),
        monthly(india[india_cols]),
        monthly(china[china_cols]),
    ]
    df = pd.concat(parts, axis=1, join="outer").sort_index().loc[start:end]
    df.index.name = "Date"

    needed = ["JODI_DIGER_KBD", "ABD_Ithalat_KBD", "TR_Toplam_Ithalat_KBD"] + india_cols + china_cols
    incomplete = df[needed].isna().any(axis=1)
    if incomplete.any():
        missing = df.loc[incomplete, needed].isna()
        summary = {d.strftime("%Y-%m"): list(missing.columns[missing.loc[d]]) for d in missing.index}
        log.warning(
            "Global: %d ay eksik bileşen nedeniyle örneklemden çıkarıldı: %s",
            incomplete.sum(),
            "; ".join(f"{k}: {','.join(v)}" for k, v in list(summary.items())[:12])
            + (" ..." if len(summary) > 12 else ""),
        )
    df = df[~incomplete].copy()

    base_pd = df["JODI_DIGER_KBD"] + df["ABD_Ithalat_KBD"] + df["TR_Toplam_Ithalat_KBD"]
    for s in SCENARIOS:
        df[f"GLOBAL_PETRODOLAR{s and s + '_KBD'}"] = (
            base_pd + df[f"China_Petrodolar{s}_KBD"] + df[f"Petrodolar{s}_KBD"]
        )
        df[f"GLOBAL_PETROYUAN{s and s + '_KBD'}"] = df[f"China_Petroyuan{s}_KBD"] + df[f"Petroyuan{s}_KBD"]

    df["GLOBAL_TOTAL_KBD"] = df["GLOBAL_PETRODOLAR"] + df["GLOBAL_PETROYUAN"]
    df["PETROYUAN_SHARE_PCT"] = (df["GLOBAL_PETROYUAN"] / df["GLOBAL_TOTAL_KBD"] * 100).where(
        df["GLOBAL_TOTAL_KBD"] > 0
    )
    return df.round(3)


def compute_panel(
    jodi_kbd: pd.DataFrame,
    us: pd.DataFrame,
    tr: pd.DataFrame,
    india: pd.DataFrame,
    china: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Panel ekonometri için uzun format: Date, region, petrodollar/petroyuan (baz/yüksek/düşük) + pay."""
    frames: List[pd.DataFrame] = []

    def add(
        region: str,
        pdol: pd.Series,
        pyuan: Optional[pd.Series] = None,
        extra: Optional[Dict[str, pd.Series]] = None,
    ) -> None:
        idx = pd.to_datetime(pdol.index).to_period("M").to_timestamp()
        f = pd.DataFrame({"petrodollar_kbd": pdol.to_numpy()}, index=idx)
        f = f[~f.index.duplicated(keep="last")]
        if pyuan is None:
            f["petroyuan_kbd"] = 0.0
        else:
            py_aligned = pd.Series(
                pyuan.to_numpy(),
                index=pd.to_datetime(pyuan.index).to_period("M").to_timestamp(),
            )
            py_aligned = py_aligned[~py_aligned.index.duplicated(keep="last")]
            f["petroyuan_kbd"] = py_aligned.reindex(f.index)
        for k, v in (extra or {}).items():
            v_aligned = pd.Series(
                v.to_numpy(),
                index=pd.to_datetime(v.index).to_period("M").to_timestamp(),
            )
            v_aligned = v_aligned[~v_aligned.index.duplicated(keep="last")]
            f[k] = v_aligned.reindex(f.index)
        f["region"] = region
        frames.append(f)

    for c in jodi_kbd.columns:
        if c not in JODI_EXCLUDED_FROM_OTHER:
            add(c, jodi_kbd[c])
    add("US", us["ABD_Ithalat_KBD"])
    add("TR", tr["TR_Toplam_Ithalat_KBD"])
    add(
        "IN",
        india["Petrodolar_KBD"],
        india["Petroyuan_KBD"],
        {"petroyuan_high_kbd": india["Petroyuan_High_KBD"], "petroyuan_low_kbd": india["Petroyuan_Low_KBD"]},
    )
    add(
        "CN",
        china["China_Petrodolar_KBD"],
        china["China_Petroyuan_KBD"],
        {
            "petroyuan_high_kbd": china["China_Petroyuan_High_KBD"],
            "petroyuan_low_kbd": china["China_Petroyuan_Low_KBD"],
        },
    )

    panel = pd.concat(frames).sort_index()
    panel.index.name = "Date"
    panel = panel.loc[start:end].reset_index()
    panel["total_kbd"] = panel["petrodollar_kbd"] + panel["petroyuan_kbd"]
    panel["petroyuan_share_pct"] = np.where(
        panel["total_kbd"] > 0, panel["petroyuan_kbd"] / panel["total_kbd"] * 100.0, np.nan
    )
    for col in ("petroyuan_high_kbd", "petroyuan_low_kbd"):
        panel[col] = panel[col].fillna(panel["petroyuan_kbd"])
    out = (
        panel.dropna(subset=["petrodollar_kbd", "petroyuan_kbd"])
        .sort_values(["region", "Date"])
        .reset_index(drop=True)
    )
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(3)
    return out


def run_global() -> bool:
    req = ["jodi", "eia", "epdk", "india", "china_payment"]
    if missing := [k for k in req if not FILES[k].exists()]:
        log.error("Global birleştirme için eksik dosyalar: %s", [FILES[k].name for k in missing])
        return False

    jodi = pd.read_csv(FILES["jodi"], index_col=0, parse_dates=True)
    us = pd.read_csv(FILES["eia"], index_col=0, parse_dates=True)
    tr = _read_epdk().set_index("Tarih")
    india = pd.read_csv(FILES["india"], index_col=0, parse_dates=True)
    china = pd.read_csv(FILES["china_payment"], index_col=0, parse_dates=True)

    g = compute_global(jodi, us, tr, india, china)
    atomic_write_csv(g, FILES["global"])
    atomic_write_csv(compute_panel(jodi, us, tr, india, china), FILES["panel"], index=False)
    if g.empty:
        log.error("Global seri boş: bileşenlerin ortak dönemi yok.")
        return False
    log.info("Global seri: %d ay (%s – %s)", len(g), g.index.min().date(), g.index.max().date())
    return True


def compute_zero_assumption_global(
    jodi_kbd: pd.DataFrame,
    us: pd.DataFrame,
    tr: pd.DataFrame,
    dgcis_rus: pd.Series,
    china_imports: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Sıfır varsayımlı (pure customs / EPDK / JODI / EIA) küresel yaptırımlı akım ve fiyat farkı tablosu.

    Hiçbir tahmini ödeme para birimi katsayısı (alt_share) kullanılmaz; doğrudan resmi gümrük ve
    düzenleyici kurum miktarları (KBD) ile Çin gümrük birim fiyatları (USD/varil) işlenir.
    """
    def monthly_idx(obj):
        obj = obj.copy()
        obj.index = pd.to_datetime(obj.index).to_period("M").to_timestamp()
        obj = obj.sort_index()
        return obj[~obj.index.duplicated(keep="last")]

    imp = china_imports.copy()
    imp["Date"] = pd.to_datetime(imp["Date"]).dt.to_period("M").dt.to_timestamp()
    world_df = imp[imp["Partner"] == "World"]
    if world_df.empty and not imp.empty:
        cn_world = monthly_idx(imp.groupby("Date")["KBD"].sum(min_count=1))
    else:
        cn_world = monthly_idx(world_df.set_index("Date")["KBD"])
    non_world = imp[imp["Partner"] != "World"].copy()
    non_world["group"] = non_world["Partner"].str.lower().str.strip().map(CHINA_PARTNER_GROUPS)

    by_group = (
        non_world.dropna(subset=["group"])
        .pivot_table(index="Date", columns="group", values="KBD", aggfunc="sum")
        .reindex(cn_world.index)
        .fillna(0.0)
    )
    cn_rus = by_group["russia"] if "russia" in by_group.columns else pd.Series(0.0, index=cn_world.index)
    cn_iv = by_group["iran_venezuela"] if "iran_venezuela" in by_group.columns else pd.Series(0.0, index=cn_world.index)
    cn_mal = by_group["malaysia"] if "malaysia" in by_group.columns else pd.Series(0.0, index=cn_world.index)

    # Çin gümrük birim fiyat iskontosu (Rusya vs yaptırımsız diğer tedarikçiler, $/varil)
    valid_px = non_world.dropna(subset=["USD", "Quantity_kg"]).copy()
    valid_px = valid_px[(valid_px["USD"] > 0) & (valid_px["Quantity_kg"] > 0)]
    valid_px["bbl"] = valid_px["Quantity_kg"] / 1000.0 * BBL_PER_TONNE

    rus_px_grp = valid_px[valid_px["group"] == "russia"].groupby("Date")[["USD", "bbl"]].sum()
    cn_rus_usd_bbl = (rus_px_grp["USD"] / rus_px_grp["bbl"]).where(rus_px_grp["bbl"] > 0).reindex(cn_world.index)

    bench_mask = ~valid_px["group"].isin(["russia", "iran_venezuela", "malaysia"])
    bench_grp = valid_px[bench_mask].groupby("Date")[["USD", "bbl"]].sum()
    cn_bench_usd_bbl = (bench_grp["USD"] / bench_grp["bbl"]).where(bench_grp["bbl"] > 0).reindex(cn_world.index)

    jodi_m = monthly_idx(jodi_kbd)
    jodi_other = monthly_idx(compute_jodi_other(jodi_m, start=start, end=end)["JODI_DIGER_KBD"])
    in_cols = [c for c in ("IN", "IND") if c in jodi_m.columns]
    in_tot = jodi_m[in_cols].sum(axis=1, min_count=1) if in_cols else pd.Series(np.nan, index=jodi_m.index)

    cn_jodi_cols = [c for c in ("CN", "CHN") if c in jodi_m.columns]
    jodi_cn = jodi_m[cn_jodi_cols].sum(axis=1, min_count=1) if cn_jodi_cols else pd.Series(np.nan, index=jodi_m.index)

    us_m = monthly_idx(us["ABD_Ithalat_KBD"])
    tr_df = monthly_idx(tr)
    tr_tot = tr_df["TR_Toplam_Ithalat_KBD"] if "TR_Toplam_Ithalat_KBD" in tr_df.columns else pd.Series(np.nan, index=tr_df.index)
    tr_rus = tr_df["TR_Rusya_Ithalat_KBD"] if "TR_Rusya_Ithalat_KBD" in tr_df.columns else pd.Series(np.nan, index=tr_df.index)
    dgcis_m = monthly_idx(dgcis_rus) if not dgcis_rus.empty else pd.Series(dtype=float)

    df = pd.DataFrame(
        {
            "JODI_DIGER_KBD": jodi_other,
            "US_Total_KBD": us_m,
            "TR_Total_KBD": tr_tot,
            "TR_Russia_KBD": tr_rus,
            "CN_Total_KBD": cn_world,
            "CN_Russia_KBD": cn_rus,
            "CN_Iran_Ven_KBD": cn_iv,
            "CN_Malaysia_KBD": cn_mal,
            "CN_Russia_USD_bbl": cn_rus_usd_bbl,
            "CN_Benchmark_USD_bbl": cn_bench_usd_bbl,
            "IN_Total_KBD": in_tot,
            "IN_Russia_KBD": dgcis_m,
            "JODI_CN_KBD": jodi_cn,
        }
    ).sort_index().loc[start:end]
    df.index.name = "Date"

    # Fiziki sınır koruması (DGCIS Rusya ithalatı JODI Hindistan toplamını aşamaz; öncesi NaN korunur)
    both_in = df["IN_Russia_KBD"].notna() & df["IN_Total_KBD"].notna()
    df.loc[both_in, "IN_Russia_KBD"] = np.minimum(df.loc[both_in, "IN_Russia_KBD"], df.loc[both_in, "IN_Total_KBD"])
    both_tr = df["TR_Russia_KBD"].notna() & df["TR_Total_KBD"].notna()
    df.loc[both_tr, "TR_Russia_KBD"] = np.clip(df.loc[both_tr, "TR_Russia_KBD"], 0.0, df.loc[both_tr, "TR_Total_KBD"])

    df["CN_Strict_KBD"] = df["CN_Russia_KBD"] + df["CN_Iran_Ven_KBD"]
    df["CN_Broad_KBD"] = df["CN_Strict_KBD"] + df["CN_Malaysia_KBD"]
    df["CN_Russia_Share_Pct"] = (df["CN_Russia_KBD"] / df["CN_Total_KBD"] * 100.0).where(df["CN_Total_KBD"] > 0)
    df["CN_Strict_Share_Pct"] = (df["CN_Strict_KBD"] / df["CN_Total_KBD"] * 100.0).where(df["CN_Total_KBD"] > 0)
    df["CN_Broad_Share_Pct"] = (df["CN_Broad_KBD"] / df["CN_Total_KBD"] * 100.0).where(df["CN_Total_KBD"] > 0)
    df["CN_Russia_Discount_USD_bbl"] = df["CN_Benchmark_USD_bbl"] - df["CN_Russia_USD_bbl"]

    df["TR_Russia_Share_Pct"] = (df["TR_Russia_KBD"] / df["TR_Total_KBD"] * 100.0).where(df["TR_Total_KBD"] > 0)
    df["IN_Russia_Share_Pct"] = (df["IN_Russia_KBD"] / df["IN_Total_KBD"] * 100.0).where(df["IN_Total_KBD"] > 0)

    tot_cols = ["JODI_DIGER_KBD", "US_Total_KBD", "TR_Total_KBD", "CN_Total_KBD", "IN_Total_KBD"]
    complete_tot = df[tot_cols].notna().all(axis=1)
    df["Global_Total_KBD"] = df[tot_cols].sum(axis=1).where(complete_tot)

    # Tam örneklem (2018-01+) Çin+Türkiye yaptırımlı akımları
    df["CN_TR_Sanctioned_Strict_KBD"] = (df["CN_Strict_KBD"] + df["TR_Russia_KBD"]).where(
        df["CN_Strict_KBD"].notna() & df["TR_Russia_KBD"].notna()
    )
    df["CN_TR_Sanctioned_Broad_KBD"] = (df["CN_Broad_KBD"] + df["TR_Russia_KBD"]).where(
        df["CN_Broad_KBD"].notna() & df["TR_Russia_KBD"].notna()
    )
    df["CN_TR_Sanctioned_Strict_Share_Pct"] = (
        df["CN_TR_Sanctioned_Strict_KBD"] / df["Global_Total_KBD"] * 100.0
    ).where(df["Global_Total_KBD"] > 0)
    df["CN_TR_Sanctioned_Broad_Share_Pct"] = (
        df["CN_TR_Sanctioned_Broad_KBD"] / df["Global_Total_KBD"] * 100.0
    ).where(df["Global_Total_KBD"] > 0)

    # DGCIS dahil tam küresel sıfır varsayım (2021-04+)
    has_strict_all = (
        df["CN_Strict_KBD"].notna()
        & df["IN_Russia_KBD"].notna()
        & df["TR_Russia_KBD"].notna()
        & (df["Global_Total_KBD"] > 0)
    )
    df["Zero_Strict_KBD"] = (df["CN_Strict_KBD"] + df["IN_Russia_KBD"] + df["TR_Russia_KBD"]).where(has_strict_all)
    df["Zero_Strict_Share_Pct"] = (df["Zero_Strict_KBD"] / df["Global_Total_KBD"] * 100.0).where(has_strict_all)

    has_all_broad = has_strict_all & df["CN_Broad_KBD"].notna()
    df["Zero_Broad_KBD"] = (df["CN_Broad_KBD"] + df["IN_Russia_KBD"] + df["TR_Russia_KBD"]).where(has_all_broad)
    df["Zero_Broad_Share_Pct"] = (df["Zero_Broad_KBD"] / df["Global_Total_KBD"] * 100.0).where(has_all_broad)
    df["Zero_Traditional_KBD"] = (df["Global_Total_KBD"] - df["Zero_Broad_KBD"]).clip(lower=0.0).where(has_all_broad)

    df["GACC_vs_JODI_CN_Diff_Pct"] = (
        (df["CN_Total_KBD"] - df["JODI_CN_KBD"]) / df["JODI_CN_KBD"] * 100.0
    ).where(df["JODI_CN_KBD"] > 0)
    return df.round(3)


def run_zero_assumption() -> bool:
    req = ["jodi", "eia", "epdk", "china_imports"]
    if missing := [k for k in req if not FILES[k].exists()]:
        log.warning("Sıfır varsayım tablosu için eksik dosyalar: %s", [FILES[k].name for k in missing])
        return False

    jodi = pd.read_csv(FILES["jodi"], index_col=0, parse_dates=True)
    us = pd.read_csv(FILES["eia"], index_col=0, parse_dates=True)
    tr = _read_epdk().set_index("Tarih")
    dgcis = pd.read_csv(FILES["dgcis"]) if FILES["dgcis"].exists() else pd.DataFrame(columns=["Period", "Qty"])
    china_imp = pd.read_csv(FILES["china_imports"], parse_dates=["Date"])

    z = compute_zero_assumption_global(jodi, us, tr, compute_dgcis_russia_kbd(dgcis), china_imp)
    atomic_write_csv(z, FILES["zero_global"])
    valid_global = z["Zero_Broad_Share_Pct"].dropna()
    log.info(
        "Sıfır varsayım küresel seri yazıldı: %d ay (tam DGCIS dahil: %d ay)",
        int(z["Global_Total_KBD"].notna().sum()),
        len(valid_global),
    )
    return not z.empty


def run_all() -> Dict[str, bool]:
    results = {}
    for name, fn in (
        ("turkey", run_turkey_master),
        ("china", run_china),
        ("india", run_india),
        ("global", run_global),
        ("zero_assumption", run_zero_assumption),
    ):
        try:
            results[name] = fn() is not False
        except Exception:  # noqa: BLE001 - bir alt adımın hatası diğerlerini durdurmamalı
            log.exception("Transform/%s başarısız", name)
            results[name] = False
    return results