"""Artificial paired prices only; never decode a saved scientific input."""
from fractions import Fraction
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
pilot = importlib.import_module("run_o1_m1_r8_price_pilot")


def test_paired_weighting_keeps_axis_covariance():
    rows = [dict(cosine={"cx_count": i}, sine={"cx_count": 7-i}) for i in range(8)]
    result = pilot.paired_statistics(rows, {"cosine": 11, "sine": 11})
    assert result["shot_weighted_mean"] == 3.5
    assert result["shot_weighted_sample_SE"] == 0
    assert result["shot_weighted_mean_fraction"] == {"numerator": 7, "denominator": 2}
    unequal = pilot.paired_statistics(rows, {"cosine": 1, "sine": 3})
    assert unequal["shot_weighted_mean"] == float(Fraction(7, 2))
    assert unequal["shot_weighted_sample_SE"] > 0


def test_repeated_prices_are_retained_and_count_is_fixed():
    row = dict(cosine={"cx_count": 5}, sine={"cx_count": 7})
    result = pilot.paired_statistics([row]*8, {"cosine": 2, "sine": 3})
    assert result["shot_weighted_mean"] == 6.2
    assert result["shot_weighted_sample_SE"] == 0
    with pytest.raises(ValueError, match="eight"):
        pilot.paired_statistics([row]*7, {"cosine": 2, "sine": 3})
