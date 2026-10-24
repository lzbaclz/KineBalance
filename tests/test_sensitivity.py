"""Tests for the E-value confounding-sensitivity module."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from src.model.sensitivity import smd_to_rr, e_value, indirect_effect_evalue


def test_smd_to_rr_null_and_monotone():
    assert smd_to_rr(0.0) == pytest.approx(1.0)
    assert smd_to_rr(0.5) > smd_to_rr(0.2) > 1.0
    assert smd_to_rr(-0.5) == pytest.approx(smd_to_rr(0.5))  # uses |d|


def test_evalue_null_and_symmetry():
    assert e_value(1.0) == pytest.approx(1.0)
    assert e_value(2.0) > 1.0
    assert e_value(3.0) > e_value(2.0)
    assert e_value(2.0) == pytest.approx(e_value(0.5))  # RR and 1/RR same E-value


def test_evalue_known_value():
    # RR = 2 -> E = 2 + sqrt(2*1) = 3.414...
    assert e_value(2.0) == pytest.approx(2.0 + np.sqrt(2.0), rel=1e-6)


def test_indirect_effect_evalue_ci_crossing_zero():
    out = indirect_effect_evalue(0.10, -0.05, 0.25)  # CI includes 0
    assert out["evalue_ci"] == 1.0
    assert out["evalue_point"] > 1.0


def test_indirect_effect_evalue_significant():
    # an interval that excludes 0.
    out = indirect_effect_evalue(-0.32, -0.65, -0.04)
    assert out["evalue_point"] > out["evalue_ci"] > 1.0
    # CI bound nearest null is -0.04.
    assert out["ci_bound_used"] == pytest.approx(-0.04)
    # Point E-value around 2 for this modest standardized effect.
    assert 1.7 < out["evalue_point"] < 2.5


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
