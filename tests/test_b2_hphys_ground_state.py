"""Artificial operators only; no saved molecular inputs or extra science."""
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
package = types.ModuleType("b2_state_test_core")
package.__path__ = [str(ROOT/"src/trotterlib")]
sys.modules[package.__name__] = package
m = importlib.import_module(package.__name__+".b2_hphys_ground_state")


@pytest.fixture(autouse=True)
def no_saved_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No saved scientific inputs in artificial tests")
    monkeypatch.setattr(np, "load", forbidden)


def toy():
    # The unselected state at -7 is lower than the selected sector ground.
    h = np.diag([-7., .2, .2, 5.]).astype(complex)
    h[1, 2], h[2, 1] = .1j, -.1j
    h[1, 2] += 2e-15j
    return h, np.array([1, 2])


def test_existing_solver_with_mock_eigh_selects_sector_and_checks_residuals(monkeypatch):
    solver = importlib.import_module(package.__name__+".h2_non_df_pilot")
    calls = []
    def counted(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        # Known analytical two-mode fixture, no additional numerical state solve.
        off_diagonal = abs(matrix[0, 1])
        return (np.array([.2-off_diagonal, .2+off_diagonal]),
                np.array([[1., 1.], [1j, -1j]])/np.sqrt(2))
    monkeypatch.setattr(solver, "eigh", counted)
    raw, indices = toy()
    state, sector, result = m.solve_checked(raw, indices, raw[np.ix_(indices, indices)])
    assert calls == [(2, 2)]
    assert result["status"] == "HPHYS_SECTOR_GROUND_STATE_CORRECTNESS_PASS"
    assert result["energy_Ha"] > -1
    assert state[0] == state[3] == 0
    np.testing.assert_array_equal(sector, state[indices])
    assert result["symmetry"]["raw_full"]["spectral_norm"] > 0
    assert all(c["passed"] for c in result["checks"])


def test_nonhermitian_input_stops_before_solve(monkeypatch):
    solver = importlib.import_module(package.__name__+".h2_non_df_pilot")
    def forbidden(*args, **kwargs):
        raise AssertionError("Ineligible operator must not reach state solve")
    monkeypatch.setattr(solver, "canonical_ground", forbidden)
    raw, indices = toy()
    raw[1, 2] += .01j
    with pytest.raises(ValueError, match="PRE_SOLVE_OPERATOR"):
        m.solve_checked(raw, indices, raw[np.ix_(indices, indices)])


def test_independent_sector_mismatch_is_rejected_before_solve():
    raw, indices = toy()
    with pytest.raises(ValueError, match="PRE_SOLVE_OPERATOR"):
        m.solve_checked(raw, indices, np.eye(2))


def test_phase_recovery_and_OF_Qiskit_mapping_use_same_pivot():
    reverse_bits = importlib.import_module(package.__name__+".h2_non_df_pilot").reverse_bits
    state = np.array([0, .4j, -.8, .2])
    pivot = int(np.argmax(abs(state)))
    canonical = m.phase_only(state, pivot)
    np.testing.assert_allclose(m.phase_only(np.exp(.371j)*state, pivot), canonical, atol=1e-14, rtol=0)
    permutation = reverse_bits(2)
    np.testing.assert_array_equal(state[permutation][permutation], state)
    assert canonical[pivot].real > 0 and abs(canonical[pivot].imag) < 1e-14
