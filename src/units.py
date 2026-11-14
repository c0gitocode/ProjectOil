"""Birim dönüşümleri. 7.33 katsayısı sadece burada kullanılır."""
from typing import Iterable, Union

import numpy as np
import pandas as pd

from .config import BBL_PER_TONNE

ArrayLike = Union[pd.Series, np.ndarray, Iterable[float], float]


def days_in_month(dates) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(pd.to_datetime(dates)).days_in_month, dtype=float)


def _as_float(values: ArrayLike) -> np.ndarray:
    return np.asarray(pd.to_numeric(pd.Series(np.atleast_1d(values)), errors="coerce"), dtype=float)


def tonnes_to_kbd(tonnes: ArrayLike, dates) -> np.ndarray:
    """Aylık metrik ton -> bin varil/gün (KBD)."""
    return _as_float(tonnes) * BBL_PER_TONNE / 1000.0 / days_in_month(dates)


def kg_to_kbd(kg: ArrayLike, dates) -> np.ndarray:
    """Aylık kilogram -> KBD (GACC miktar birimi kg'dır)."""
    return tonnes_to_kbd(_as_float(kg) / 1000.0, dates)


def kbbl_to_kbd(kbbl: ArrayLike, dates) -> np.ndarray:
    """Aylık bin varil -> KBD."""
    return _as_float(kbbl) / days_in_month(dates)


def tonnes_to_kbd_scalar(tonnes: float, year: int, month: int) -> float:
    return float(tonnes_to_kbd([tonnes], [pd.Timestamp(year=year, month=month, day=1)])[0])
