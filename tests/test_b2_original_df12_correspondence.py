"""Artificial operators only; no molecular files, DF decomposition or solve."""
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
package = types.ModuleType("b2_correspondence_test_core")
package.__path__ = [str(ROOT/"src/trotterlib")]
sys.modules[package.__name__] = package
m = importlib.import_module(package.__name__+".b2_original_df12_correspondence")
ref = m._reference_module()


@pytest.fixture(autouse=True)
def forbid_scientific_io_and_state_solves(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Artificial operator tests cannot load science or solve states")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    import scipy.linalg
    monkeypatch.setattr(scipy.linalg, "eigh", forbidden)


def toy():
    one = np.array([[.4, .2+.1j], [.2-.1j, -.3]])
    two = np.zeros((2,)*4)
    two[0, 1, 1, 0] = .17
    two[1, 0, 0, 1] = .17
    return .73, one, two


def test_scalar_and_vector_diagnostics_accept_vacuum_check_shape():
    assert m.basic_metrics(np.zeros(3))["max_abs_entry"] == 0
    assert m.basic_metrics(np.array(0.))["induced_infinity_norm"] == 0


def test_CAR_sparse_JW_and_occupation_agree_with_complex_signs_and_scalar():
    c, one, two = toy()
    a = ref.annihilators(2)
    car = c*np.eye(4)+ref.one_body(one, a).toarray()+ref.quartic(two, a).toarray()
    reversal = [0, 2, 1, 3]
    big = m.exact_jw(c, one, two)
    oracle = np.asarray(m.occupation_original(c, one, two, range(4)))
    np.testing.assert_allclose(car, big[np.ix_(reversal, reversal)], atol=1e-14, rtol=0)
    np.testing.assert_allclose(big, oracle, atol=1e-14, rtol=0)
    assert big[0, 0] == c


def test_sparse_reference_preserves_sub_EQ_terms_without_global_cutoff_change():
    from openfermion import InteractionOperator, get_sparse_operator
    from openfermion.config import EQ_TOLERANCE
    one = np.array([[0, 1e-10], [1e-10, 0]])
    two = np.zeros((2,)*4)
    exact = m.exact_jw(0, one, two)
    symbolic = get_sparse_operator(InteractionOperator(0, one, two)).toarray()
    assert np.max(abs(exact)) == 1e-10
    assert np.max(abs(symbolic)) == 0
    assert EQ_TOLERANCE == 1e-8


def test_sector_reference_rejects_leakage_and_quartic_order_is_preserved():
    c, one, two = toy()
    full = np.asarray(m.occupation_original(c, one, two, range(4)))
    sector = np.asarray(m.occupation_original(c, one, two, [1, 2]))
    np.testing.assert_allclose(sector, full[np.ix_([1, 2], [1, 2])], atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match="LEAVES_SECTOR"):
        m.occupation_original(c, one, two, [1])


def test_signed_saved_squares_equal_normal_order_and_independent_reference():
    c, one, _ = toy()
    factors = np.array([one, [[.2, -.3j], [.3j, .8]]])
    weights = np.array([-.6, .3])
    a = ref.annihilators(2)
    direct = c*np.eye(4)+ref.one_body(one, a).toarray()
    for w, g in zip(weights, factors):
        lifted = ref.one_body(g, a)
        direct += w*(lifted@lifted).toarray()
    quadratic = np.einsum("l,lpq,lqr->pr", weights, factors, factors)
    quartic = -np.einsum("l,lpq,lrs->prqs", weights, factors, factors)
    normal = c*np.eye(4)+ref.one_body(one+quadratic, a).toarray()+ref.quartic(quartic, a).toarray()
    independent = np.asarray(m.occupation.occupation_df_matrix(c, one, weights, factors, range(4)))
    reversal = [0, 2, 1, 3]
    np.testing.assert_allclose(direct, normal, atol=1e-14, rtol=0)
    np.testing.assert_allclose(direct, independent[np.ix_(reversal, reversal)], atol=1e-14, rtol=0)


def test_precision_path_measures_nonzero_model_difference_without_state_solve():
    c, one, two = toy()
    pairs = {"H_phys": (one, two), "H_src": (one*.9, two)}
    payload = (c, pairs, one, np.array([-.2]), np.array([one]), [1, 2])
    low, high = [m.precision_task((dps, *payload)) for dps in (80, 120)]
    binary = {name: np.asarray(m.occupation_original(c, h, w, [1, 2]))
              for name, (h, w) in pairs.items()}
    binary["H_DF12"] = np.asarray(m.occupation.occupation_df_matrix(c, one, [-.2], [one], [1, 2]))
    r = m.precision_comparisons(low, high, binary)
    assert all(v["passed"] for v in r["stability_80_vs_120"].values())
    assert float(r["model_differences_120"]["H_phys_minus_H_src"]["max_abs_entry"]) > .01
    assert r["certified_interval"] is False
