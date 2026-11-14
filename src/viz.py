"""Görselleştirme katmanı."""
import matplotlib

matplotlib.use("Agg")  # GUI gerektirmez; sunucu/CI'da da çalışır
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import BAND_START, END_DATE, EVENT_DATE, FILES, OUTPUTS_DIR, START_DATE  # noqa: E402
from .logger import get_logger  # noqa: E402

log = get_logger(__name__)
BLUE, ORANGE = "#2b5c8f", "#d95f02"


def _save(fig, name: str) -> None:
    path = OUTPUTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # açık figürler bellekte birikmesin
    log.info("Grafik kaydedildi: %s", path.name)


def _event_line(ax, label: str = "Ukrayna İşgali (Şubat 2022)") -> None:
    ax.axvline(pd.Timestamp(EVENT_DATE), color="red", linestyle=":", alpha=0.7, label=label)


def plot_global() -> None:
    df = pd.read_csv(FILES["global"], index_col=0, parse_dates=True).sort_index().loc[START_DATE:END_DATE]
    if df.empty:
        log.warning("plot_global atlandı: filtrelenen aralıkta veri yok.")
        return
    end_label = df.index.max().strftime("%Y-%m")

    fig, ax = plt.subplots(figsize=(14, 8))
    post = df.loc[BAND_START:]
    if not post.empty:
        ax.fill_between(
            post.index,
            post["GLOBAL_PETROYUAN_Low_KBD"],
            post["GLOBAL_PETROYUAN_High_KBD"],
            color=ORANGE,
            alpha=0.2,
            label="Petroyuan ±%5 varsayım bandı",
        )
        ax.fill_between(
            post.index,
            post["GLOBAL_PETRODOLAR_Low_KBD"],
            post["GLOBAL_PETRODOLAR_High_KBD"],
            color=BLUE,
            alpha=0.15,
            label="Petrodolar ±%5 varsayım bandı",
        )
    ax.plot(df.index, df["GLOBAL_PETROYUAN"], label="Global Petroyuan Bloğu", color=ORANGE, lw=2.5)
    ax.plot(df.index, df["GLOBAL_PETRODOLAR"], label="Global Petrodolar Bloğu", color=BLUE, lw=2.5)
    _event_line(ax)
    ax.set(
        title=f"Küresel Ham Petrol Ödeme Sistemleri: Petrodolar vs Petroyuan ({START_DATE[:4]}–{end_label})",
        ylabel="Günlük İthalat Hacmi (KBD)",
        xlabel="Tarih",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    _save(fig, "petrodolar_vs_petroyuan_global.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["PETROYUAN_SHARE_PCT"], color=ORANGE, lw=2.5, label="Petroyuan payı (%)")
    _event_line(ax)
    ax.set(title="Kapsanan Ham Petrol İthalatında Petroyuan Payı (%)", ylabel="%", xlabel="Tarih")
    ax.legend(loc="upper left")
    _save(fig, "petroyuan_share_global.png")


def plot_china_share() -> None:
    df = pd.read_csv(FILES["china_payment"], index_col=0, parse_dates=True).sort_index().loc[START_DATE:END_DATE]
    if df.empty:
        log.warning("plot_china_share atlandı: filtrelenen aralıkta veri yok.")
        return
    end_label = df.index.max().strftime("%Y-%m")
    total = df["China_Total_KBD"].replace(0, float("nan"))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df["China_Petroyuan_KBD"] / total * 100, label="Çin Petroyuan Payı (%)", color=ORANGE, lw=2.5)
    ax.plot(df.index, df["China_Petrodolar_KBD"] / total * 100, label="Çin Petrodolar Payı (%)", color=BLUE, lw=2.5)
    _event_line(ax, "Ukrayna İşgali")
    ax.set(
        title=f"Çin Ham Petrol İthalatında Para Birimi Dağılımı (%) (–{end_label})",
        ylabel="Pay (%)",
        xlabel="Tarih",
        ylim=(-5, 105),
    )
    ax.legend(loc="center right", frameon=True, fontsize=10)
    _save(fig, "china_yuan_dolar_oran_trend.png")


def plot_india_share() -> None:
    df = pd.read_csv(FILES["india"], index_col=0, parse_dates=True).sort_index().loc[START_DATE:END_DATE]
    if df.empty:
        log.warning("plot_india_share atlandı: filtrelenen aralıkta veri yok.")
        return
    end_label = df.index.max().strftime("%Y-%m")
    rus = df["Rusya_Ithalat_KBD"]
    yuan_pct = (df["Petroyuan_KBD"] / rus * 100).where(rus > 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, 100 - yuan_pct, marker="o", lw=2.5, color=BLUE, label="USD + Dirham (Rusya ithalatı içinde)")
    ax.plot(df.index, yuan_pct, marker="s", lw=2.5, color=ORANGE, label="Yuan + Ruble (Alternatif)")
    _event_line(ax, "Ukrayna İşgali")
    ax.set(
        title=f"Hindistan'ın Rus Petrolü İthalatında Ödeme Yöntemleri (–{end_label})",
        ylabel="Rusya ithalatı içindeki pay (%)",
        xlabel="Tarih",
        ylim=(-5, 110),
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=11, frameon=True)
    _save(fig, "hindistan_genel_hacim_odeme_trend_guncel.png")


def plot_zero_assumption() -> None:
    df = pd.read_csv(FILES["zero_global"], index_col=0, parse_dates=True).sort_index().loc[START_DATE:END_DATE]
    if df.empty:
        log.warning("plot_zero_assumption atlandı: filtrelenen aralıkta veri yok.")
        return
    end_label = df.index.max().strftime("%Y-%m")

    # 1) Bölgesel & Küresel Sıfır Varsayım Payları (%)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for col, lbl, color, ls in (
        ("CN_Strict_Share_Pct", "Çin Strict Payı (Rusya+İran+Ven %)", ORANGE, "-"),
        ("CN_Broad_Share_Pct", "Çin Broad Payı (+Malezya %)", "#e7298a", "--"),
        ("TR_Russia_Share_Pct", "Türkiye Rusya Payı — EPDK (%)", BLUE, "-"),
        ("IN_Russia_Share_Pct", "Hindistan Rusya Payı — DGCIS/JODI, 2021-04+ (%)", "#1b9e77", "-"),
    ):
        if col in df.columns and df[col].notna().any():
            ax1.plot(df.index, df[col], label=lbl, color=color, lw=2.2, ls=ls)
    _event_line(ax1)
    ax1.set(
        title=f"Sıfır Varsayım: Büyük Alıcıların Yaptırımlı Ham Petrol Payları ({START_DATE[:4]}–{end_label})",
        ylabel="Pay (%)",
    )
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    for col, lbl, color, ls in (
        ("CN_TR_Sanctioned_Broad_Share_Pct", "Küresel Kapsanan Pay — Çin+TR Broad (2018+ %)", "#7570b3", "--"),
        ("Zero_Strict_Share_Pct", "Küresel Sıfır Varsayım — Strict: CN+IN+TR (2021-04+ %)", BLUE, "-"),
        ("Zero_Broad_Share_Pct", "Küresel Sıfır Varsayım — Broad: +Malezya (2021-04+ %)", ORANGE, "-"),
    ):
        if col in df.columns and df[col].notna().any():
            ax2.plot(df.index, df[col], label=lbl, color=color, lw=2.3, ls=ls)
    _event_line(ax2)
    ax2.set(
        title="Küresel Ham Petrol İthalatında Doğrudan Ölçülen Yaptırımlı Akım Payı (%)",
        ylabel="Küresel Pay (%)",
        xlabel="Tarih",
    )
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    _save(fig, "zero_assumption_shares.png")

    # 2) Fiyat Mekanizması: Rusya İskontosu ($/bbl) vs Çin Rusya Payı (%)
    if "CN_Russia_Discount_USD_bbl" in df.columns and df["CN_Russia_Discount_USD_bbl"].notna().sum() >= 2:
        fig, ax_l = plt.subplots(figsize=(14, 6.5))
        ax_r = ax_l.twinx()

        disc = df["CN_Russia_Discount_USD_bbl"].dropna()
        l1 = ax_l.plot(disc.index, disc, color=BLUE, lw=2.2, label="Rusya İskontosu: Benchmark - Rusya ($/varil)")
        ax_l.axhline(0.0, color="gray", lw=0.9, ls="--", alpha=0.6)
        ax_l.axvline(pd.Timestamp(EVENT_DATE), color="red", ls=":", alpha=0.7)

        lines = list(l1)
        if "CN_Russia_Share_Pct" in df.columns and df["CN_Russia_Share_Pct"].notna().any():
            r_share = df["CN_Russia_Share_Pct"].dropna()
            lines += ax_r.plot(r_share.index, r_share, color=ORANGE, lw=2.3, label="Çin İthalatında Rusya Payı (%, sağ eksen)")
        elif "CN_Strict_Share_Pct" in df.columns:
            s_share = df["CN_Strict_Share_Pct"].dropna()
            lines += ax_r.plot(s_share.index, s_share, color=ORANGE, lw=2.3, label="Çin Strict Payı (%, sağ eksen)")

        ax_l.set(
            title=f"Ekonomik Mekanizma: Çin Gümrük Rusya Varil İskontosu ve İthalat Payı ({START_DATE[:4]}–{end_label})",
            ylabel="Fiyat İskontosu ($/varil)",
            xlabel="Tarih",
        )
        ax_r.set_ylabel("İthalat Payı (%)")
        ax_l.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True, fontsize=9.5)
        _save(fig, "china_russia_discount_vs_share.png")


def plot_panel_event_study() -> None:
    path = OUTPUTS_DIR / "econometrics_panel_event_study.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    deps = list(df["bagimli_degisken"].drop_duplicates())
    fig, axes = plt.subplots(len(deps), 1, figsize=(13, 5 * len(deps)), sharex=True)
    if len(deps) == 1:
        axes = [axes]

    for ax, dep in zip(axes, deps):
        sub = df[df["bagimli_degisken"] == dep].sort_values("rel_month")
        ax.axhline(0.0, color="black", lw=0.9)
        ax.axvline(-1.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="Referans ay (t=-1, Ocak 2022)")
        ax.axvline(0.0, color="red", ls=":", lw=1.3, alpha=0.8, label="Ukrayna İşgali (t=0, Şubat 2022)")

        valid_ci = sub.dropna(subset=["ci_low", "ci_high"])
        if not valid_ci.empty:
            ax.fill_between(valid_ci["rel_month"], valid_ci["ci_low"], valid_ci["ci_high"], color=BLUE, alpha=0.18, label="%95 Kümelenmiş GA")
        ax.plot(sub["rel_month"], sub["coef"], marker="o", ms=4, color=BLUE, lw=2.0, label="TWFE Olay Katsayısı")

        pre_p = sub["pretrend_p"].dropna().iloc[0] if sub["pretrend_p"].notna().any() else float("nan")
        suffix = f" | Pre-trend F-testi p={pre_p:.3g}" if pd.notna(pre_p) else ""
        ax.set(title=f"Panel DiD Olay Çalışması: {dep}{suffix}", ylabel="Etki (ref: t=-1)")
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    axes[-1].set_xlabel("Kırılmaya Göreli Ay (t = 0: Şubat 2022)")
    _save(fig, "panel_did_event_study.png")


def plot_all() -> bool:
    sns.set_theme(style="whitegrid")
    ok = True
    for key, fn in (("global", plot_global), ("china_payment", plot_china_share), ("india", plot_india_share)):
        if not FILES[key].exists():
            log.warning("Grafik atlandı, dosya yok: %s", FILES[key].name)
            ok = False
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            log.exception("Grafik hatası: %s", fn.__name__)
            ok = False

    if FILES["zero_global"].exists():
        try:
            plot_zero_assumption()
        except Exception:  # noqa: BLE001
            log.exception("Grafik hatası: plot_zero_assumption")
            ok = False

    if (OUTPUTS_DIR / "econometrics_panel_event_study.csv").exists():
        try:
            plot_panel_event_study()
        except Exception:  # noqa: BLE001
            log.exception("Grafik hatası: plot_panel_event_study")
            ok = False

    return ok
