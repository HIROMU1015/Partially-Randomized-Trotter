"""Artificial algebra and mocked I/O only: never load the H4 snapshot.

The existing package-alias pattern skips trotterlib's eager research imports.
Core Pauli, DF and normal-ordering helpers below are real implementations.
"""
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "m1_synthetic_core"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/trotterlib")]
sys.modules[PACKAGE] = package
adapter = importlib.import_module(f"{PACKAGE}.o1_m1_matched_split")


@pytest.fixture(autouse=True)
def no_saved_inputs_or_circuits(monkeypatch):
    import qiskit

    def forbidden(*args, **kwargs):
        raise AssertionError("No saved scientific input load or circuit construction in synthetic tests")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(qiskit, "QuantumCircuit", forbidden)


def annihilators(n):
    """Independent CAR reference, qubit 0 = least significant state bit."""
    result = []
    for q in range(n):
        matrix = np.zeros((1 << n, 1 << n), complex)
        for column in range(1 << n):
            if column & (1 << q):
                sign = (-1) ** ((column & ((1 << q) - 1)).bit_count())
                matrix[column ^ (1 << q), column] = sign
        result.append(matrix)
    return result


def lifted_one(matrix, annihilation):
    n = len(annihilation)
    return sum((matrix[p, q] * annihilation[p].conj().T @ annihilation[q]
                for p in range(n) for q in range(n)),
               np.zeros((1 << n, 1 << n), complex))


def lifted_two(tensor, annihilation):
    n = len(annihilation)
    return sum((tensor[p, q, r, s] * annihilation[p].conj().T
                @ annihilation[q].conj().T @ annihilation[r] @ annihilation[s]
                for p, q, r, s in np.argwhere(tensor != 0)),
               np.zeros((1 << n, 1 << n), complex))


def toy_inputs():
    bare = np.array([[.23, .09 + .11j], [.09 - .11j, -.41]])
    effective = bare + np.array([[.17, -.03j], [.03j, .08]])
    factors = np.array([[[.4, .2 + .3j], [.2 - .3j, -.6]],
                        [[-.1, -.21j], [.21j, .5]]])
    two = np.zeros((2,) * 4)
    two[0, 1, 1, 0] = .31
    two[1, 0, 0, 1] = .31
    return adapter.MatchedAInputs(
        .73, bare, two, effective, np.array([-.7, .25]), factors,
        np.array([1 + .2j, .3 - .4j, -.7j, .8]) * 1.7, np.array([1, 2]),
    )


def test_complex_factors_signed_squares_and_original_target_are_separate():
    inputs = toy_inputs()
    split = adapter.prepare_matched_a(inputs)
    a = annihilators(2)
    retained = lifted_one(inputs.df_one_body, a)
    residual = sum((w * (lifted_one(g, a) @ lifted_one(g, a))
                    for w, g in zip(inputs.df_lambdas, inputs.df_g_matrices)),
                   np.zeros((4, 4), complex))
    original = inputs.nuclear_repulsion * np.eye(4) + lifted_one(inputs.spin_one, a) + lifted_two(inputs.spin_two, a)
    np.testing.assert_allclose(split.pauli_retained.sparse_operator().toarray(), retained, rtol=0, atol=1e-14)
    np.testing.assert_allclose(split.pauli_residual.sparse_operator().toarray(), residual, rtol=0, atol=1e-14)
    np.testing.assert_allclose(split.pauli_full().sparse_operator().toarray(),
                               inputs.nuclear_repulsion * np.eye(4) + retained + residual, rtol=0, atol=1e-14)
    np.testing.assert_allclose(split.original_pauli.sparse_operator().toarray(), original, rtol=0, atol=1e-14)
    # A deliberate original-vs-DF mismatch must not be hidden in only P's tail.
    assert np.linalg.norm(original - split.pauli_full().sparse_operator().toarray()) > .1
    np.testing.assert_array_equal(split.df_hamiltonian.one_body, inputs.df_one_body)
    np.testing.assert_array_equal(split.df_hamiltonian.lambdas, [-.7, .25])
    np.testing.assert_array_equal(split.df_hamiltonian.g_matrices, inputs.df_g_matrices)
    np.testing.assert_array_equal(split.inputs.state, inputs.state)
    assert np.vdot(split.inputs.state, split.inputs.state).real > 1
    assert not split.inputs.state.flags.writeable
    assert split.df_hamiltonian.metadata["state_field"] == "state"


def test_normal_ordering_correction_sign_prefactor_and_missing_correction():
    # An exact artificial spin-symmetric square with an independent CAR tensor.
    spatial_g = np.array([[.7, .2], [.2, -.4]])
    g = np.kron(spatial_g, np.eye(2))
    weight = -.6
    bare = np.kron(np.array([[.1, .03j], [-.03j, -.2]]), np.eye(2))
    two = weight * np.einsum("ps,qr->pqrs", g, g)
    correction = -weight * (g @ g)
    values = dict(nuclear_repulsion=.19, spin_one=bare, spin_two=two,
                  df_one_body=bare + correction, df_lambdas=np.array([weight]),
                  df_g_matrices=np.array([g]), state=np.arange(16) + 1j,
                  physical_qiskit_indices=np.array([3, 6, 9, 12]))
    split = adapter.prepare_matched_a(adapter.MatchedAInputs(**values))
    np.testing.assert_allclose(split.one_body_correction, correction, rtol=0, atol=1e-15)
    np.testing.assert_allclose(split.source_one_body_correction, correction, rtol=0, atol=1e-15)
    a = annihilators(4)
    np.testing.assert_allclose(
        lifted_two(two, a), lifted_one(correction, a) + weight * (lifted_one(g, a) @ lifted_one(g, a)),
        rtol=0, atol=2e-15,
    )
    np.testing.assert_allclose(split.pauli_full().sparse_operator().toarray(),
                               split.original_pauli.sparse_operator().toarray(), rtol=0, atol=2e-15)
    # This synthetic regression would detect the tempting bare-for-effective swap.
    values["df_one_body"] = bare
    missing = adapter.prepare_matched_a(adapter.MatchedAInputs(**values))
    assert np.linalg.norm(missing.one_body_correction - missing.source_one_body_correction) > .1
    assert np.linalg.norm(missing.pauli_full().sparse_operator().toarray()
                          - missing.original_pauli.sparse_operator().toarray()) > .1


def test_nuclear_retained_and_tail_identity_are_accounted_once():
    split = adapter.prepare_matched_a(toy_inputs())
    Pauli = type(split.pauli_residual)
    expected_tail_identity = sum(float(w) * ((np.trace(g) ** 2 + np.trace(g @ g)).real) / 4
                                 for w, g in zip(split.inputs.df_lambdas, split.inputs.df_g_matrices))
    assert split.pauli_residual.constant == pytest.approx(expected_tail_identity)
    assert split.pauli_retained.constant == pytest.approx(np.trace(split.inputs.df_one_body).real / 2)
    for policy in ("faithful_identity_in_tail", "extract_identity_phase"):
        terms, shift = adapter.pauli_tail_coefficients(split, identity_policy=policy)
        residual = Pauli.from_terms(terms, num_qubits=2).sparse_operator().toarray() + shift * np.eye(4)
        np.testing.assert_allclose(residual, split.pauli_residual.sparse_operator().toarray(), rtol=0, atol=1e-15)
        full = (split.inputs.nuclear_repulsion * np.eye(4)
                + split.pauli_retained.sparse_operator().toarray() + residual)
        np.testing.assert_allclose(full, split.pauli_full().sparse_operator().toarray(), rtol=0, atol=1e-15)
        assert any(label == "II" for label, _ in terms) == (policy == "faithful_identity_in_tail")
    with pytest.raises(ValueError, match="identity policy"):
        adapter.pauli_tail_coefficients(split, identity_policy="implicit")


def test_explicit_loader_reads_original_state_only_and_closes_mock(monkeypatch):
    state = np.zeros(256, complex)
    state[17] = 2j
    payload = dict(nuclear_repulsion=.1, spin_one=np.zeros((8, 8)), spin_two=np.zeros((8,) * 4),
                   df_one_body=np.zeros((8, 8)), df_lambdas=np.arange(10.),
                   df_g_matrices=np.zeros((10, 8, 8)), state=state,
                   physical_qiskit_indices=np.arange(36))
    payload["DF_state"] = np.zeros(256, complex)
    payload["DF_state"][5] = -3  # deliberately a different stored field and ray
    read = []

    class MockArchive:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __getitem__(self, key):
            read.append(key)
            if key == "DF_state":
                raise AssertionError("DF_state is a different scientific state")
            return payload[key]

    archive = MockArchive()

    def mock_load(path, *, allow_pickle):
        assert path == "explicit-mocked-input.npz" and allow_pickle is False
        return archive

    monkeypatch.setattr(np, "load", mock_load)
    inputs = adapter.load_h4_a_snapshot("explicit-mocked-input.npz")
    assert archive.closed
    assert set(read) == set(payload) - {"DF_state"}
    np.testing.assert_array_equal(inputs.state, state)
    state[17] = 99
    assert inputs.state[17] == 2j  # independent raw copy, no normalization
    # Wrong saved rank is rejected; it cannot trigger a replacement DF run.
    payload["df_lambdas"] = np.arange(12.)
    payload["df_g_matrices"] = np.zeros((12, 8, 8))
    with pytest.raises(ValueError, match="rank10"):
        adapter.load_h4_a_snapshot("explicit-mocked-input.npz")
    del payload["state"]
    with pytest.raises(KeyError, match="state"):
        adapter.load_h4_a_snapshot("explicit-mocked-input.npz")
    assert "DF_state" not in read


def test_saved_arrays_reject_lossy_weights_and_nonhermitian_factors():
    from dataclasses import replace

    inputs = toy_inputs()
    with pytest.raises(ValueError, match="weights must be real"):
        replace(inputs, df_lambdas=np.array([-.7 + .1j, .25]))
    bad = inputs.df_g_matrices.copy()
    bad[0, 0, 1] += .1j
    with pytest.raises(ValueError, match="Hermitian"):
        adapter.prepare_matched_a(replace(inputs, df_g_matrices=bad))


def test_df_tail_dispatch_uses_all_saved_factors_without_building_circuits(monkeypatch):
    split = adapter.prepare_matched_a(toy_inputs())
    # Load the real extractor module, then replace only its circuit-producing entry.
    subpackage = types.ModuleType(f"{PACKAGE}.df_trotter")
    subpackage.__path__ = [str(ROOT / "src/trotterlib/df_trotter")]
    monkeypatch.setitem(sys.modules, subpackage.__name__, subpackage)
    tail_module = importlib.import_module(f"{PACKAGE}.df_rte_tail")
    sentinel = object()

    def extraction(tail_id, hamiltonian, indices, **kwargs):
        assert tail_id == "M1_A_V_square"
        assert hamiltonian is split.df_hamiltonian
        assert indices == (0, 1)
        assert kwargs == dict(identity_policy="extract_identity_phase", coefficient_atol=0.)
        return sentinel

    monkeypatch.setattr(tail_module, "extract_df_tail_from_hamiltonian", extraction)
    assert adapter.extract_matched_df_tail(split, identity_policy="extract_identity_phase") is sentinel
