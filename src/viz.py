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
    return ok
