import math
import pandas as pd
import pytest

from src.extract.common import (
    atomic_write_csv,
    clean_header,
    normalize_epdk_columns,
    parse_number,
    tr_lower,
    tr_upper,
)
from src.units import kbbl_to_kbd, kg_to_kbd, tonnes_to_kbd, tonnes_to_kbd_scalar


def test_turkish_case_helpers():
    assert tr_lower("KASIM NİSAN AĞUSTOS IĞDIR") == "kasım nisan ağustos ığdır"
    assert tr_upper("nisan ağustos") == "NİSAN AĞUSTOS"
    assert clean_header("Ülke\nAdı\t ") == "ÜLKE ADI"


def test_parse_number_locales():
    # Türkçe biçim
    assert parse_number("1.234", locale="tr") == pytest.approx(1234.0)
    assert parse_number("1.234.567,89", locale="tr") == pytest.approx(1234567.89)
    assert parse_number("12,5", locale="tr") == pytest.approx(12.5)

    # İngilizce / Hindistan biçimi
    assert parse_number("1.234", locale="en") == pytest.approx(1.234)
    assert parse_number("1,234,567.89", locale="en") == pytest.approx(1234567.89)
    assert parse_number("12,34,567", locale="en") == pytest.approx(1234567.0)
    assert parse_number("12,34,567.25", locale="en") == pytest.approx(1234567.25)

    # Auto modu
    assert parse_number("1.234.567", locale="auto") == pytest.approx(1234567.0)
    assert parse_number("1.234", locale="auto") == pytest.approx(1.234)
    assert parse_number("12,34,567", locale="auto") == pytest.approx(1234567.0)

    # Boş / geçersiz / pd.NA / bool
    for bad in (None, pd.NA, float("nan"), True, False, "", "-", "--", "x", "N/A", "abc"):
        assert math.isnan(parse_number(bad))


def test_unit_conversions_calendar_days():
    # 2024 Şubat artık yıl (29 gün), 2023 Şubat 28 gün
    kbd_leap = tonnes_to_kbd_scalar(1000.0, 2024, 2)
    kbd_norm = tonnes_to_kbd_scalar(1000.0, 2023, 2)
    assert kbd_leap == pytest.approx(1000.0 * 7.33 / 1000.0 / 29.0)
    assert kbd_norm == pytest.approx(1000.0 * 7.33 / 1000.0 / 28.0)

    dates = ["2024-02-01", "2024-01-01"]
    res_kg = kg_to_kbd([1_000_000.0, 1_000_000.0], dates)
    res_t = tonnes_to_kbd([1000.0, 1000.0], dates)
    assert res_kg == pytest.approx(res_t)
    assert kbbl_to_kbd([310.0], ["2024-01-01"]) == pytest.approx([10.0])


def test_normalize_epdk_columns_mixed_schema():
    df = pd.DataFrame(
        {
            "Tarih": ["2022-01-01", "2025-01-01"],
            "TR_Rusya_Iskontosu_KBD": [100.0, float("nan")],
            "TR_Rusya_Ithalat_KBD": [float("nan"), 250.0],
            "TR_Saf_Petrodolar_KBD": [400.0, float("nan")],
            "TR_Rusya_Disi_KBD": [float("nan"), 300.0],
        }
    )
    out = normalize_epdk_columns(df)
    assert list(out.columns) == ["Tarih", "TR_Rusya_Ithalat_KBD", "TR_Rusya_Disi_KBD"]
    assert out["TR_Rusya_Ithalat_KBD"].tolist() == [100.0, 250.0]
    assert out["TR_Rusya_Disi_KBD"].tolist() == [400.0, 300.0]


def test_atomic_write_csv_creates_parents(tmp_path):
    target = tmp_path / "nested" / "dir" / "out.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    atomic_write_csv(df, target, index=False)
    assert target.exists()
    loaded = pd.read_csv(target)
    assert loaded.equals(df)
