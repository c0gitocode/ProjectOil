"""ProjectOil ana ETL ve ekonometrik analiz boru hattı."""
import argparse
import time
from typing import Dict, List, Optional

from src import econometrics, load, transform, viz
from src.extract import SOURCES
from src.logger import get_logger, setup_logging

log = get_logger(__name__)


def _extract_step(sources: Optional[List[str]] = None) -> Dict[str, bool]:
    selected = sources if sources else list(SOURCES.keys())
    results: Dict[str, bool] = {}
    for name in selected:
        try:
            SOURCES[name]()
            results[f"extract/{name}"] = True
        except Exception:  # noqa: BLE001 - tek kaynak hatası diğer kaynakları durdurmamalı
            log.exception("Extract/%s başarısız", name)
            results[f"extract/{name}"] = False
    return results


def run_pipeline(
    skip_extract: bool = False,
    sources: Optional[List[str]] = None,
    only_analysis: bool = False,
) -> Dict[str, bool]:
    setup_logging()
    t0 = time.perf_counter()
    results: Dict[str, bool] = {}

    if not only_analysis:
        if not skip_extract:
            log.info("=== 1. VERİ ÇEKME (EXTRACT) ===")
            results.update(_extract_step(sources))
        else:
            log.info("=== 1. VERİ ÇEKME ATLANDI (--skip-extract) ===")

        log.info("=== 2. DÖNÜŞTÜRME (TRANSFORM) ===")
        for k, ok in transform.run_all().items():
            results[f"transform/{k}"] = ok

        log.info("=== 3. VERİTABANI YÜKLEME (LOAD) ===")
        try:
            results["load"] = load.load_data_to_sqlite()
        except Exception:  # noqa: BLE001
            log.exception("Load başarısız")
            results["load"] = False
    else:
        log.info("=== EXTRACT/TRANSFORM/LOAD ATLANDI (--only-analysis) ===")

    log.info("=== 4. EKONOMETRİK ANALİZ ===")
    try:
        results["econometrics"] = econometrics.run_tests()
    except Exception:  # noqa: BLE001
        log.exception("Ekonometrik analiz başarısız")
        results["econometrics"] = False

    log.info("=== 5. GÖRSELLEŞTİRME ===")
    try:
        results["viz"] = viz.plot_all()
    except Exception:  # noqa: BLE001
        log.exception("Görselleştirme başarısız")
        results["viz"] = False

    elapsed = time.perf_counter() - t0
    ok_n = sum(results.values())
    log.info("=== ÖZET (%d/%d adım başarılı, %.2f sn) ===", ok_n, len(results), elapsed)
    for step, status in results.items():
        log.info("  %-24s %s", step, "OK" if status else "BAŞARISIZ / ATLANDI")
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProjectOil ETL & Ekonometri Boru Hattı")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Dış kaynaklardan veri çekme adımını atla (yerel data/raw ve master CSV'lerle çalıştır)",
    )
    parser.add_argument(
        "--only-analysis",
        action="store_true",
        help="Sadece ekonometrik testleri ve grafikleri çalıştır (data/processed mevcut olmalı)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCES.keys()),
        help="Yalnızca belirtilen kaynakları çek (ör. --sources jodi eia epdk)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    results = run_pipeline(
        skip_extract=args.skip_extract,
        sources=args.sources,
        only_analysis=args.only_analysis,
    )
    return 0 if results.get("transform/global", True) and results.get("econometrics", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())