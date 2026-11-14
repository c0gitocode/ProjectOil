import sqlite3

import numpy as np
import pandas as pd

import main as pipeline_main
from src import config, econometrics, load, logger, transform, viz


def test_full_pipeline_skip_extract_e2e(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    china_dir = raw_dir / "china"
    proc_dir = tmp_path / "processed"
    out_dir = tmp_path / "outputs"
    for d in (raw_dir, china_dir, proc_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Yolları tmp_path'e yönlendir (FILES dict'i tüm modüller tarafından referansla paylaşılır)
    for k, orig in list(config.FILES.items()):
        monkeypatch.setitem(config.FILES, k, proc_dir / orig.name)
    for mod in (config, transform, load, viz, econometrics, logger):
        if hasattr(mod, "OUTPUTS_DIR"):
            monkeypatch.setattr(mod, "OUTPUTS_DIR", out_dir)
    monkeypatch.setattr(logger, "_CONFIGURED", False)
    monkeypatch.setattr(transform, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transform, "CHINA_RAW_DIR", china_dir)
    monkeypatch.setattr(load, "DB_PATH", proc_dir / "oil_database.db")

    dates = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
    monkeypatch.setattr(config, "START_DATE", "2020-01-01")
    monkeypatch.setattr(config, "END_DATE", "2024-12-01")
    monkeypatch.setattr(transform, "START_DATE", "2020-01-01")
    monkeypatch.setattr(transform, "END_DATE", "2024-12-01")
    monkeypatch.setattr(econometrics, "START_DATE", "2020-01-01")
    monkeypatch.setattr(econometrics, "END_DATE", "2024-12-01")
    monkeypatch.setattr(viz, "START_DATE", "2020-01-01")
    monkeypatch.setattr(viz, "END_DATE", "2024-12-01")

    rng = np.random.default_rng(123)
    # 1) JODI (JP, DE, KR + IN)
    jodi_df = pd.DataFrame(
        {
            "JP": 2500.0 + rng.normal(0, 40, len(dates)),
            "DE": 1700.0 + rng.normal(0, 30, len(dates)),
            "KR": 2800.0 + rng.normal(0, 45, len(dates)),
            "IN": 4500.0 + rng.normal(0, 60, len(dates)),
        },
        index=dates,
    )
    jodi_df.to_csv(config.FILES["jodi"])

    # 2) EIA (US)
    pd.DataFrame({"ABD_Ithalat_KBD": 6000.0 + rng.normal(0, 80, len(dates))}, index=dates).to_csv(
        config.FILES["eia"]
    )

    # 3) EPDK (TR - legacy/yeni karma sütun testi dahil)
    epdk_df = pd.DataFrame(
        {
            "Tarih": dates.strftime("%Y-%m-%d"),
            "TR_Toplam_Ithalat_KBD": 650.0 + rng.normal(0, 15, len(dates)),
            "TR_Rusya_Iskontosu_KBD": [100.0 if i < 25 else np.nan for i in range(len(dates))],
            "TR_Rusya_Ithalat_KBD": [np.nan if i < 25 else 300.0 for i in range(len(dates))],
            "TR_Rusya_Disi_KBD": 350.0,
        }
    )
    epdk_df.to_csv(config.FILES["epdk"], index=False)

    # 4) DGCIS (Hindistan)
    dgcis_rows = []
    for d in dates:
        if d >= pd.Timestamp("2021-04-01"):
            ton = "200,000" if d < pd.Timestamp("2022-02-01") else "5,000,000"
            dgcis_rows.append({"Period": d.strftime("%b-%Y"), "Unit": "TON", "Qty": ton})
    pd.DataFrame(dgcis_rows).to_csv(config.FILES["dgcis"], index=False)

    # 5) GACC ham yıllık Çin CSV'leri
    for y in range(2020, 2025):
        gacc_rows = []
        for m in range(1, 13):
            gacc_rows.append(
                {
                    "Date of data": f"{y}-{m:02d}",
                    "Trading partner": "Russia",
                    "Quantity": str(int(6_500_000_000 + rng.normal(0, 1e8))),
                    "US dollar": "3500000000",
                }
            )
            gacc_rows.append(
                {
                    "Date of data": f"{y}-{m:02d}",
                    "Trading partner": "Saudi Arabia",
                    "Quantity": str(int(7_000_000_000 + rng.normal(0, 1e8))),
                    "US dollar": "4000000000",
                }
            )
        pd.DataFrame(gacc_rows).to_csv(china_dir / f"{y}_china.csv", index=False)

    rc = pipeline_main.main(["--skip-extract"])
    assert rc == 0

    # Çıktı CSV'leri
    for key in ("tr_master", "china_imports", "china_payment", "india", "global", "panel"):
        assert config.FILES[key].exists(), f"Eksik çıktı: {key}"

    # SQLite tabloları ve _metadata
    with sqlite3.connect(proc_dir / "oil_database.db") as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"global_petrodollar_petroyuan", "panel_long", "china_payment", "india_analysis", "_metadata"} <= tables
        global_rows = conn.execute("SELECT COUNT(*) FROM global_petrodollar_petroyuan").fetchone()[0]
        assert global_rows == 60

    # Ekonometri & Grafikler
    assert (out_dir / "econometrics_report.txt").exists()
    assert (out_dir / "econometrics_summary.csv").exists()
    assert (out_dir / "econometrics_sensitivity.csv").exists()
    assert (out_dir / "econometrics_panel_summary.csv").exists()

    for img in (
        "petrodolar_vs_petroyuan_global.png",
        "petroyuan_share_global.png",
        "china_yuan_dolar_oran_trend.png",
        "hindistan_genel_hacim_odeme_trend_guncel.png",
    ):
        p = out_dir / img
        assert p.exists() and p.stat().st_size > 10_000

    # --only-analysis modu mevcut processed verilerle tek başına çalışabilmeli
    rc_only = pipeline_main.main(["--only-analysis"])
    assert rc_only == 0


def test_extract_source_isolation_and_tuik_excel(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    out_dir = tmp_path / "outputs"
    for d in (raw_dir, proc_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    for k, orig in list(config.FILES.items()):
        monkeypatch.setitem(config.FILES, k, proc_dir / orig.name)
    monkeypatch.setattr(transform, "RAW_DIR", raw_dir)
    monkeypatch.setattr(logger, "OUTPUTS_DIR", out_dir)
    monkeypatch.setattr(logger, "_CONFIGURED", False)

    # Sentetik TÜİK fasil27 Excel dosyası
    tuik_df = pd.DataFrame(
        [
            ["TÜİK Başlık", None, None, None],
            ["Yıl", "Ay", "Gösterge", "Değer"],
            [2023, 1, "Enerji İthalatı", 5000000],
            [None, 2, "Enerji İthalatı", 4800000],
        ]
    )
    tuik_df.to_excel(raw_dir / "tuik_fasil27.xlsx", index=False, header=False)
    transform.run_turkey_master()
    assert config.FILES["tr_master"].exists()
    tr_m = pd.read_csv(config.FILES["tr_master"])
    assert len(tr_m) == 2
    assert tr_m["Enerji_Fasil27_USD"].tolist() == [5000000, 4800000]

    # Bir kaynak hata verse bile diğer seçili kaynak çalışmaya devam etmeli
    calls = []

    def fail_src():
        calls.append("fail")
        raise RuntimeError("Ağ hatası")

    def ok_src():
        calls.append("ok")

    monkeypatch.setitem(pipeline_main.SOURCES, "jodi", fail_src)
    monkeypatch.setitem(pipeline_main.SOURCES, "eia", ok_src)
    ext_res = pipeline_main._extract_step(["jodi", "eia"])
    assert calls == ["fail", "ok"]
    assert ext_res == {"extract/jodi": False, "extract/eia": True}

