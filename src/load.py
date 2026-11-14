"""Load katmanı: işlenmiş CSV'leri SQLite'a yükler."""
import sqlite3
from datetime import datetime

import pandas as pd

from .config import DB_PATH, FILES
from .logger import get_logger

log = get_logger(__name__)

# tablo adı -> (dosya anahtarı, tarih index'li mi)
TABLES = {
    "global_petrodollar_petroyuan": ("global", True),
    "global_zero_assumption": ("zero_global", True),
    "panel_long": ("panel", False),
    "china_payment": ("china_payment", True),
    "china_imports": ("china_imports", False),
    "india_analysis": ("india", True),
    "tr_epdk": ("epdk", False),
    "tr_master_analysis": ("tr_master", False),
    "jodi_crude_imports": ("jodi", True),
    "us_crude_imports": ("eia", True),
}


def load_data_to_sqlite() -> bool:
    log.info("SQLite yükleme: %s", DB_PATH)
    loaded = []
    with sqlite3.connect(DB_PATH) as conn:
        for table, (key, indexed) in TABLES.items():
            path = FILES[key]
            if not path.exists():
                continue
            df = pd.read_csv(path, index_col=0 if indexed else None, parse_dates=indexed or False)
            df.to_sql(table, conn, if_exists="replace", index=indexed)
            loaded.append((table, len(df)))
        pd.DataFrame([{"table": t, "rows": n, "loaded_at": datetime.now().isoformat(timespec="seconds")}
                      for t, n in loaded]).to_sql("_metadata", conn, if_exists="replace", index=False)
    for t, n in loaded:
        log.info("  %-32s %6d satır", t, n)
    return bool(loaded)