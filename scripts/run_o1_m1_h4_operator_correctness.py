"""Authorized M1 H4-A operator-only check; no circuit or evolution entry.

Acceptance rules fixed before the first H4 numerical load:
eps=binary64 epsilon; m=4*n**4+8*2**n+16*rank;
gamma=m*eps/(1-m*eps); tol_entry=gamma*max(1,S), where
S=|c|+sum|h_bare|+sum|W|+sum|h_eff|+sum_l |lambda_l|*(sum|G_l|)**2.
This conservative numerical check is NOT a rigorous forward-error certificate.
Original-vs-DF Frobenius error uses the saved input diagnostic's separate 1e-9
operator criterion, not a new energy budget. No threshold adapts to discrepancies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import numpy as np
from scipy.sparse import csr_matrix

ADAPTER_COMMIT = "5e4f00487720dced86d53637464c48245a35a557"
INPUT_COMMIT = "4591b8d75a2a6a55c74a51225f0c61648023c8ee"
INPUT_PATH = "artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz"
INPUT_SHA256 = "8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50"
ADAPTER_PATH = "src/trotterlib/o1_m1_matched_split.py"
ORIGINAL_OPERATOR_TOLERANCE = 1e-9
ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def annihilators(n):
    result = []
    for q in range(n):
        columns = [j for j in range(1 << n) if j & (1 << q)]
        rows = [j ^ (1 << q) for j in columns]
        signs = [(-1) ** ((j & ((1 << q) - 1)).bit_count()) for j in columns]
        result.append(csr_matrix((signs, (rows, columns)), shape=(1 << n, 1 << n), dtype=complex))
    return result


def one_body(matrix, a):
    result = csr_matrix(a[0].shape, dtype=complex)
    for p, q in np.argwhere(matrix != 0):
        result += matrix[p, q] * (a[p].conj().T @ a[q])
    return result


def quartic(tensor, a):
    pairs = {(p, q): a[p] @ a[q] for p in range(len(a)) for q in range(len(a))}
    result = csr_matrix(a[0].shape, dtype=complex)
    for p, q, r, s in np.argwhere(tensor != 0):
        result += tensor[p, q, r, s] * (pairs[q, p].conj().T @ pairs[r, s])
    return result


def fock_lift(u):
    """Exterior-power/minor representation of a supplied orbital U, no gates."""
    n = len(u)
    result = np.zeros((1 << n, 1 << n), complex)
    for count in range(n + 1):
        occupations = list(itertools.combinations(range(n), count))
        indices = [sum(1 << q for q in occupied) for occupied in occupations]
        if count == 0:
            result[0, 0] = 1
            continue
        out = np.asarray(occupations)
        minors = u[out[:, None, :, None], out[None, :, None, :]]
        result[np.ix_(indices, indices)] = np.linalg.det(minors)
    return result


def pauli_coefficients(matrix):
    """All coefficients by XOR diagonals + Walsh transform, with no cutoff."""
    d = len(matrix)
    j = np.arange(d)
    work = np.array([matrix[j ^ x, j] for x in range(d)], dtype=complex)
    width = 1
    while width < d:
        blocks = work.reshape(d, -1, 2 * width)
        left, right = blocks[:, :, :width].copy(), blocks[:, :, width:].copy()
        blocks[:, :, :width] = left + right
        blocks[:, :, width:] = left - right
        width *= 2
    phase = np.array([[(-1j) ** ((x & z).bit_count()) for z in range(d)] for x in range(d)])
    return work * phase / d


def encoded_pauli(model):
    result = np.zeros((1 << model.num_qubits,) * 2, complex)
    result[0, 0] = model.constant
    for label, coefficient in model.terms:
        x = sum(1 << q for q, axis in enumerate(reversed(label)) if axis in "XY")
        z = sum(1 << q for q, axis in enumerate(reversed(label)) if axis in "ZY")
        result[x, z] = coefficient
    return result


def coefficient_comparison(model, reference, eq_tolerance, arithmetic_tolerance):
    direct = pauli_coefficients(reference)
    encoded = encoded_pauli(model)
    delta = direct - encoded
    absent = (encoded == 0) & (direct != 0)
    magnitudes = np.abs(direct[absent])
    return dict(max_abs_coefficient_difference=float(np.max(np.abs(delta))),
                coefficient_difference_l1=float(np.sum(np.abs(delta))),
                absent_nonzero_reference_count=int(np.count_nonzero(absent)),
                absent_max_abs=float(np.max(magnitudes, initial=0)),
                absent_l1=float(np.sum(magnitudes)),
                absent_above_openfermion_eq_tolerance=int(np.count_nonzero(magnitudes > eq_tolerance)),
                absent_above_arithmetic_tolerance=int(np.count_nonzero(magnitudes > arithmetic_tolerance)),
                reference_max_imaginary=float(np.max(np.abs(direct.imag))),
                note="Absent nonzero values include binary64 CAR/Walsh roundoff; not all are exact physical terms.")


def numerical_comparison(left, right, state, tolerance):
    difference = np.asarray(left) - np.asarray(right)
    action_norm = expectation_value = None
    if difference.shape == (len(state), len(state)):
        norm = float(np.vdot(state, state).real)
        action = difference @ state
        expectation = complex(np.vdot(state, action) / norm)
        action_norm = float(np.linalg.norm(action))
        expectation_value = dict(real=expectation.real, imag=expectation.imag)
    largest = float(np.max(np.abs(difference)))
    return dict(max_abs_entry=largest, frobenius=float(np.linalg.norm(difference)),
                induced_infinity_norm=float(np.max(np.sum(np.abs(difference), axis=1))),
                state_action_l2=action_norm,
                normalized_state_expectation=expectation_value,
                max_entry_tolerance=tolerance, passed_max_entry=largest <= tolerance)


def self_check():
    a = annihilators(2)
    g = np.array([[.4, .2 + .3j], [.2 - .3j, -.6]])
    eta, u = np.linalg.eigh(g)
    f = fock_lift(u)
    occupancy = np.array([[((j >> q) & 1) for q in range(2)] for j in range(4)])
    np.testing.assert_allclose(f @ np.diag(occupancy @ eta) @ f.conj().T,
                               one_body(g, a).toarray(), rtol=0, atol=1e-14)
    y = np.kron(np.eye(2), np.array([[0, -1j], [1j, 0]]))
    coefficients = pauli_coefficients(y)
    assert abs(coefficients[1, 1] - 1) < 1e-14
    assert np.count_nonzero(np.abs(coefficients) > 1e-14) == 1


def run(args):
    started = time.perf_counter()
    fixed_source = subprocess.check_output(["git", "show", f"{ADAPTER_COMMIT}:{ADAPTER_PATH}"], cwd=ROOT)
    assert fixed_source == (ROOT / ADAPTER_PATH).read_bytes(), "Fixed adapter changed"
    snapshot = Path(args.snapshot)
    if not snapshot.exists():
        blob = subprocess.check_output(["git", "show", f"{INPUT_COMMIT}:{INPUT_PATH}"], cwd=ROOT)
        assert sha(blob) == INPUT_SHA256
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write(blob)
    assert sha(snapshot.read_bytes()) == INPUT_SHA256
    self_check()

    # Existing light-package pattern avoids eager GPU/runner imports.
    package = types.ModuleType("m1_operator_core")
    package.__path__ = [str(ROOT / "src/trotterlib")]
    sys.modules[package.__name__] = package
    subpackage = types.ModuleType(f"{package.__name__}.df_trotter")
    subpackage.__path__ = [str(ROOT / "src/trotterlib/df_trotter")]
    sys.modules[subpackage.__name__] = subpackage
    adapter = importlib.import_module(f"{package.__name__}.o1_m1_matched_split")
    df_core = importlib.import_module(f"{package.__name__}.df_hamiltonian")
    tail_core = importlib.import_module(f"{package.__name__}.df_rte_tail")
    import openfermion
    import qiskit

    def no_circuit(*unused, **kwargs):
        raise RuntimeError("Circuit construction is outside this operator-only scope")

    qiskit.QuantumCircuit = no_circuit
    inputs = adapter.load_h4_a_snapshot(snapshot)
    split = adapter.prepare_matched_a(inputs)
    n, rank, d = inputs.num_qubits, len(inputs.df_lambdas), 1 << inputs.num_qubits
    scale = (abs(inputs.nuclear_repulsion) + np.sum(np.abs(inputs.spin_one))
             + np.sum(np.abs(inputs.spin_two)) + np.sum(np.abs(inputs.df_one_body))
             + sum(abs(w) * np.sum(np.abs(g)) ** 2
                   for w, g in zip(inputs.df_lambdas, inputs.df_g_matrices, strict=True)))
    eps = np.finfo(float).eps
    m = 4 * n**4 + 8 * d + 16 * rank
    gamma = m * eps / (1 - m * eps)
    tolerance = float(gamma * max(1, scale))
    state, identity = inputs.state, np.eye(d, dtype=complex)
    a = annihilators(n)
    retained = one_body(inputs.df_one_body, a).toarray()
    bare = one_body(inputs.spin_one, a).toarray()
    normal = quartic(inputs.spin_two, a).toarray()
    square = np.zeros((d, d), complex)
    diagonal_dictionary = np.zeros_like(square)
    tail_identity = 0.
    fragments = []
    occupations = np.array([[((j >> q) & 1) for q in range(n)] for j in range(d)])
    for index, (weight, g) in enumerate(zip(inputs.df_lambdas, inputs.df_g_matrices, strict=True)):
        lifted = one_body(g, a)
        direct = float(weight) * (lifted @ lifted).toarray()
        square += direct
        eta, u = np.linalg.eigh(g)
        order = np.argsort(np.abs(eta))[::-1]
        eta, u = eta[order], u[:, order]
        f = fock_lift(u)
        coefficients = tail_core.exact_df_diagonal_coefficients(eta, weight)
        diagonal = np.zeros(d)
        for support, coefficient in coefficients:
            diagonal += coefficient * np.prod(1 - 2 * occupations[:, support], axis=1)
            if not support:
                tail_identity += coefficient
        reconstructed = (f * diagonal[None, :]) @ f.conj().T
        diagonal_dictionary += reconstructed
        fragments.append(dict(index=index,
                              operator=numerical_comparison(reconstructed, direct, state, tolerance),
                              basis_unitarity_max_abs=float(np.max(np.abs(f.conj().T @ f - identity))),
                              orbital_eigenpair_max_abs=float(np.max(np.abs(g @ u - u * eta))),
                              nonzero_diagonal_components=sum(c != 0 for _, c in coefficients)))

    original = inputs.nuclear_repulsion * identity + bare + normal
    native = inputs.nuclear_repulsion * identity + retained + square
    p_retained = split.pauli_retained.sparse_operator().toarray()
    p_square = split.pauli_residual.sparse_operator().toarray()
    p_full = split.pauli_full().sparse_operator().toarray()
    p_original = split.original_pauli.sparse_operator().toarray()
    with np.load(snapshot, allow_pickle=False) as saved:
        saved_original, saved_df = saved["non_DF_matrix"], saved["DF_matrix"]
    correction_operator = one_body(split.one_body_correction, a).toarray()
    source_correction_operator = one_body(split.source_one_body_correction, a).toarray()
    # Independent existing matrix-free DF implementation uses OF big-endian indices.
    sector = df_core.PhysicalSector(n, np.arange(d))
    operator, counter = df_core.df_linear_operator(split.df_hamiltonian, sector, backend="python")
    production_big = df_core._linear_operator_to_dense(operator)
    reverse = np.array([int(f"{j:0{n}b}"[::-1], 2) for j in range(d)])
    production = production_big[np.ix_(reverse, reverse)]
    comparisons = {}

    def compare(name, left, right):
        comparisons[name] = numerical_comparison(left, right, state, tolerance)

    compare("retained_P_vs_CAR", p_retained, retained)
    compare("residual_P_vs_CAR", p_square, square)
    compare("full_P_vs_CAR_DF", p_full, native)
    compare("original_P_vs_CAR", p_original, original)
    compare("full_existing_DF_vs_CAR", production, native)
    compare("residual_numeric_S_dictionary_vs_CAR", diagonal_dictionary, square)
    compare("full_P_vs_numeric_S_dictionary", p_full, inputs.nuclear_repulsion * identity + retained + diagonal_dictionary)
    compare("saved_original_matrix_vs_CAR", saved_original, original)
    compare("saved_DF_matrix_vs_CAR", saved_df, native)
    compare("one_body_correction_matrix", split.one_body_correction, split.source_one_body_correction)
    compare("normal_ordering_V_normal_vs_D_plus_V_square", normal, source_correction_operator + square)
    compare("retained_T_eff_vs_T_bare_plus_D", retained, bare + source_correction_operator)
    compare("one_body_correction_lift", correction_operator, source_correction_operator)
    policies = {}
    Pauli = type(split.pauli_residual)
    for policy in ("faithful_identity_in_tail", "extract_identity_phase"):
        terms, transfer = adapter.pauli_tail_coefficients(split, identity_policy=policy)
        reconstructed = Pauli.from_terms(terms, num_qubits=n).sparse_operator().toarray() + transfer * identity
        policies[policy] = dict(residual_transfer=transfer,
                                residual=numerical_comparison(reconstructed, square, state, tolerance))
    scalar = dict(nuclear=inputs.nuclear_repulsion, retained_P_identity=split.pauli_retained.constant,
                  retained_trace_identity=float(np.trace(inputs.df_one_body).real / 2),
                  residual_P_identity=split.pauli_residual.constant,
                  residual_numeric_S_identity=float(tail_identity),
                  residual_full_Fock_trace_identity=float(np.trace(square).real / d),
                  full_P_identity=split.pauli_full().constant,
                  full_CAR_trace_identity=float(np.trace(native).real / d))
    scalar_error = max(abs(scalar["retained_P_identity"] - scalar["retained_trace_identity"]),
                       abs(scalar["residual_P_identity"] - tail_identity),
                       abs(scalar["residual_P_identity"] - scalar["residual_full_Fock_trace_identity"]),
                       abs(scalar["full_P_identity"] - scalar["full_CAR_trace_identity"]))
    target_difference = numerical_comparison(original, native, state, tolerance)
    target_difference["saved_input_operator_tolerance"] = ORIGINAL_OPERATOR_TOLERANCE
    target_difference["passed_saved_input_operator_criterion_via_Frobenius_bound"] = target_difference["frobenius"] <= ORIGINAL_OPERATOR_TOLERANCE
    target_difference["note"] = "A separate numerical difference, not strict equality or an adopted energy-error allowance."
    coefficient_results = {name: coefficient_comparison(model, reference, openfermion.config.EQ_TOLERANCE, tolerance)
                           for name, model, reference in (("retained", split.pauli_retained, retained),
                                                          ("residual", split.pauli_residual, square),
                                                          ("original", split.original_pauli, original),
                                                          ("full", split.pauli_full(), native))}
    status = (all(x["passed_max_entry"] for x in comparisons.values())
              and all(x["operator"]["passed_max_entry"] for x in fragments)
              and all(x["residual"]["passed_max_entry"] for x in policies.values())
              and scalar_error <= tolerance
              and target_difference["passed_saved_input_operator_criterion_via_Frobenius_bound"])
    payload = dict(status="PASS" if status else "FAIL", scope="H4_A_operator_only",
                   source=dict(adapter_commit=ADAPTER_COMMIT, adapter_path=ADAPTER_PATH,
                               adapter_sha256=sha(fixed_source), driver_path=str(Path(__file__).relative_to(ROOT)),
                               driver_sha256=sha(Path(__file__).read_bytes())),
                   input=dict(commit=INPUT_COMMIT, path=INPUT_PATH, sha256=INPUT_SHA256,
                              local_copy=str(snapshot), rank=rank, qubits=n, sector_dimension=len(inputs.physical_qiskit_indices),
                              state_field="state", state_hash=importlib.import_module(f"{package.__name__}.pauli_signal_adapter").array_hash(state),
                              state_squared_norm=float(np.vdot(state, state).real),
                              DF_state_read=False),
                   precision=dict(array_dtype="complex128 / float64", epsilon=eps,
                                  m=m, gamma=float(gamma), absolute_input_scale=float(scale),
                                  max_entry_tolerance=tolerance, original_operator_tolerance=ORIGINAL_OPERATOR_TOLERANCE,
                                  rationale=__doc__, forward_error_certificate=False),
                   runtime=dict(python=platform.python_version(), executable=sys.executable,
                                packages={x:importlib.metadata.version(x) for x in ["numpy","scipy","openfermion","qiskit","pyscf"]},
                                elapsed_seconds=time.perf_counter()-started, utc=datetime.now(timezone.utc).isoformat(),
                                argv=sys.argv, df_linear_operator_actions=counter["count"]),
                   comparisons=comparisons, fragment_numeric_S_dictionary=fragments,
                   scalar=dict(values=scalar,max_abs_difference=float(scalar_error),passed=scalar_error <= tolerance),
                   identity_policies=policies, original_minus_DF=target_difference,
                   openfermion_microterms=dict(EQ_TOLERANCE=openfermion.config.EQ_TOLERANCE,
                                               coefficient_checks=coefficient_results),
                   limitations=["No Gaussian basis circuits / extract_matched_df_tail / build_df_blocks invoked.",
                                "Numeric S dictionary uses orbital eigenvectors and exterior powers only; circuit encoding is untested.",
                                "No time evolution, simulation, RNG, RTE law, shots, cost, fit, new DF, or ground-vector regeneration.",
                                "Original saved state used only for operator differences; no ground-state accuracy certification."])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(status=payload["status"], tolerance=tolerance,
                          maximum_comparison_entry=max(x["max_abs_entry"] for x in comparisons.values()),
                          original_minus_DF=target_difference, output=str(output)), ensure_ascii=False))
    return 0 if status else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
