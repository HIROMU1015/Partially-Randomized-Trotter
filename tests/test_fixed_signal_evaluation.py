"""Artificial/mocked inputs only; no real molecule is constructed or evolved."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from numpy.testing import assert_allclose
from openfermion import QubitOperator, get_sparse_operator
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from trotterlib.fixed_signal_accounting import AXES, fixed_signal_shots
from trotterlib.fixed_signal_evaluation import (
    FixedSignalConfig, evaluate_fixed_signal, hadamard_wrapper, prepare_signal_adapter,
    validate_fixed_signal_result,
)
from trotterlib.fixed_signal_io import load_artificial_input
from trotterlib.pauli_signal_adapter import (
    PauliHamiltonian, append_pauli_event, sparse_mean_action,
)
from trotterlib.rte import enumerate_rte_events, event_unitary, sample_rte_events


ROOT = Path(__file__).resolve().parents[1]
NUMERICS = dict(numerical_allowance={a: 1e-10 for a in AXES},
                evidence_kind="EMPIRICAL", evidence_ref="artificial declared tolerance; not certified",
                log_B_margin=1e-12)


def pauli():
    return PauliHamiltonian.from_terms([("I", .1), ("Z", .7), ("X", -.2)])


def df():
    from trotterlib.df_hamiltonian import DFHamiltonian
    return DFHamiltonian(.1, np.array([[.2, .03], [.03, -.1]]), np.array([.4, -.2]),
        (np.array([[.5, .1], [.1, .2]]), np.diag([.1, .6])), {"model_id": "artificial_df"})


def test_pauli_consolidation_order_identity_hash_and_no_cutoff():
    h = PauliHamiltonian.from_terms([("XI", .2), ("II", .1), ("IZ", -.7), ("XI", .1), ("YY", 1e-17)])
    reordered = PauliHamiltonian.from_terms([("YY", 1e-17), ("XI", .1), ("IZ", -.7), ("II", .1), ("XI", .2)])
    assert h.constant == .1 and h.terms[0] == ("IZ", -.7)
    assert h.terms[-1] == ("YY", 1e-17)
    assert h.hamiltonian_hash == reordered.hamiltonian_hash
    assert h.metadata["df_rank"] is None


@pytest.mark.parametrize("term", [("A", 1), ("X", 1j), ("X", float("nan")), ("X", float("inf"))])
def test_pauli_rejects_invalid_coefficient_or_label(term):
    with pytest.raises(ValueError):
        PauliHamiltonian.from_terms([term])


def test_openfermion_to_qiskit_bit_order_with_y():
    op = QubitOperator("X0 Y1", -.3) + QubitOperator("Z0", .7) + QubitOperator((), .1)
    h = PauliHamiltonian.from_openfermion(op, num_qubits=2)
    permutation = [0, 2, 1, 3]
    dense = get_sparse_operator(op, n_qubits=2).toarray()
    assert_allclose(h.sparse_operator().toarray(), dense[np.ix_(permutation, permutation)])
    assert ("YX", -.3) in h.terms


def test_supplied_spatial_integrals_give_manual_number_operator_hamiltonian():
    # One spatial orbital: E_nuc + h(n_alpha+n_beta) + eri*n_alpha*n_beta.
    h = PauliHamiltonian.from_spatial_integrals([[.4]], [[[[.6]]]], nuclear_repulsion=.13)
    assert_allclose(h.sparse_operator().toarray(), np.diag([.13, .53, .53, 1.53]))
    assert h.metadata["spin_order"] == "2*p+alpha(0)/beta(1)"
    assert h.metadata["two_body_prefactor"] == .5


def test_supplied_integral_offdiagonal_convention_against_openfermion():
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial
    one = np.array([[.4, .03], [.03, -.2]])
    a = np.array([[.7, .2], [.2, -.3]])
    eri = np.einsum("pq,rs->pqrs", a, a)
    ours = PauliHamiltonian.from_spatial_integrals(one, eri, nuclear_repulsion=.1)
    spin_one, spin_two = spinorb_from_spatial(one, eri.transpose(0, 2, 3, 1))
    reference = PauliHamiltonian.from_openfermion(jordan_wigner(get_fermion_operator(
        InteractionOperator(.1, spin_one, spin_two/2))), num_qubits=4)
    assert_allclose(ours.sparse_operator().toarray(), reference.sparse_operator().toarray(), atol=1e-15)


def test_spatial_integrals_reject_unsupported_complex_or_invalid_shape():
    with pytest.raises(NotImplementedError):
        PauliHamiltonian.from_spatial_integrals(np.array([[.4+0j]]), [[[[.1]]]], nuclear_repulsion=0)
    with pytest.raises(ValueError):
        PauliHamiltonian.from_spatial_integrals([[.4]], np.zeros((2,)*4), nuclear_repulsion=0)


@pytest.mark.parametrize("T", [.3, -.3, 0.])
def test_s2_sign_order_constant_and_time_reversal(T):
    h = pauli()
    a = prepare_signal_adapter(h, FixedSignalConfig("pauli_jw", T=T))
    circuit = a.trajectory(q=1, seed=0)
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1, -1])
    expected = np.exp(-1j*.1*T) * (expm(-1j*.7*Z*T/2) @ expm(1j*.2*X*T) @ expm(-1j*.7*Z*T/2))
    assert_allclose(Operator(circuit).data, expected, atol=1e-14)
    inverse = prepare_signal_adapter(h, FixedSignalConfig("pauli_jw", T=-T)).trajectory(q=1, seed=0)
    assert_allclose(Operator(inverse).data @ Operator(circuit).data, np.eye(2), atol=1e-14)


def test_s2_has_cubic_local_error():
    h = pauli()
    errors = []
    for T in [.2, .1]:
        a = prepare_signal_adapter(h, FixedSignalConfig("pauli_jw", T=T))
        errors.append(np.linalg.norm(Operator(a.trajectory(q=1, seed=0)).data - expm(-1j*T*h.sparse_operator().toarray())))
    assert 7.9 < errors[0]/errors[1] < 8.1


@pytest.mark.parametrize("axis,component", [("cosine", "real"), ("sine", "imag")])
def test_hadamard_wrapper_measures_complex_signal_including_identity(axis, component):
    h = pauli()
    a = prepare_signal_adapter(h, FixedSignalConfig("pauli_jw", T=.3, q=2))
    evo = a.trajectory(q=2, seed=0)
    psi = np.array([1., 1j])/np.sqrt(2)
    signal = np.vdot(psi, Operator(evo).data @ psi)
    wrapped = hadamard_wrapper(evo, axis, measure=False)
    probabilities = Statevector(np.r_[psi, [0., 0.]]).evolve(wrapped).probabilities([1])
    assert probabilities[0]-probabilities[1] == pytest.approx(getattr(signal, component), abs=1e-14)


def test_ordinary_control_preserves_global_identity_phase():
    evo = QuantumCircuit(2)
    evo.global_phase = -.37
    control = QuantumCircuit(3)
    control.append(evo.to_gate().control(), [2, 0, 1])
    assert_allclose(Operator(control).data, np.diag([1.]*4 + [np.exp(-.37j)]*4), atol=1e-14)


@pytest.mark.parametrize("T,K", [(.8, 0), (.8, 2), (-.8, 2), (.8, 4)])
def test_all_pauli_events_match_generic_involution_definition_and_mean(T, K):
    h = PauliHamiltonian.from_terms([("I", .13), ("X", .4), ("Y", -.3), ("Z", .2)])
    config = FixedSignalConfig("pauli_jw", method="PR", T=T, L_D=1, K=K)
    adapter = prepare_signal_adapter(h, config)
    matrices = {str(j): np.sign(c)*PauliHamiltonian.from_terms([(s, 1.)]).sparse_operator().toarray()
                for j, (s,c) in enumerate(adapter.tail)}
    events = enumerate_rte_events(adapter.components, adapter.distribution)
    assert math.fsum(e.event_probability for e in events) == pytest.approx(1., abs=1e-14)
    mean = np.zeros((2, 2), dtype=complex)
    for event in events:
        circuit = QuantumCircuit(1)
        append_pauli_event(circuit, event, adapter.labels)
        actual = Operator(circuit).data
        assert_allclose(actual, event_unitary(event, matrices), atol=2e-14)
        mean += event.event_probability * actual
    tail = adapter.operators()[2].toarray()
    poly = np.eye(2, dtype=complex)
    term = poly.copy()
    for degree in range(1, K+2):
        term = term @ (-1j*T*tail/degree)
        poly += term
    assert_allclose(mean, poly/adapter.distribution.exact_finite_distribution, atol=2e-14)


def test_pr_reconstruction_sparse_mean_repetition_seed_and_empty_tail():
    h = pauli()
    c = FixedSignalConfig("pauli_jw", method="PR", L_D=1, T=.2, q=3, r=2)
    a = prepare_signal_adapter(h, c)
    full, blocks, tail, scalar = a.operators()
    assert_allclose(full.toarray(), blocks[0].toarray()+tail.toarray()+scalar*np.eye(2))
    x = Operator(a.trajectory(q=3, seed=123)).data
    assert_allclose(x, Operator(a.trajectory(q=3, seed=123)).data, atol=0)
    psi = np.array([1., 1j])/np.sqrt(2)
    expected = []
    for state in np.eye(2):
        expected.append(sparse_mean_action(a, state, q=1))
    one_step = np.stack(expected, axis=1)
    assert_allclose(sparse_mean_action(a, psi, q=3), np.linalg.matrix_power(one_step, 3)@psi, atol=1e-14)
    empty = prepare_signal_adapter(h, replace(c, L_D=2))
    assert not empty.randomized and empty.log_B_step == 0


def test_pauli_b0_and_b0i_are_identical_without_duplicate_phase():
    h = pauli()
    circuits = [prepare_signal_adapter(h, FixedSignalConfig("pauli_jw", method=m, L_D=1)).trajectory(q=1, seed=1)
                for m in ["B0", "B0I"]]
    assert_allclose(Operator(circuits[0]).data, Operator(circuits[1]).data, atol=0)


@pytest.mark.parametrize("representation", ["pauli_jw", "df_native"])
@pytest.mark.parametrize("method", ["PF", "PR", "B0", "B0I"])
def test_artificial_end_to_end_C_N_G_and_compiled_wrapper(representation, method):
    h = pauli() if representation == "pauli_jw" else df()
    state = np.array([1., 1j])/np.sqrt(2) if representation == "pauli_jw" else np.array([0., 1., 1j, 0.])/np.sqrt(2)
    config = FixedSignalConfig(representation, method=method, L_D=None if method == "PF" else 1,
                               T=.2, q=2, r=2, cost_samples=3)
    result = evaluate_fixed_signal(h, state, config, epsilon_sig=.2, verify_compiled=True, **NUMERICS)
    json.dumps(result, allow_nan=False)
    assert result["shots"]["usable"]
    total = sum(result["shots"]["axes"][a]["shots"]*result["circuit_cost"]["axes"][a]["mean"] for a in AXES)
    assert result["total_RZ"]["mean"] == pytest.approx(total)
    assert result["circuit_cost"]["compiled_signal_max_error"] < 1e-9
    assert not result["validation"]["molecular_validation"]
    if representation == "pauli_jw":
        assert result["model"]["DF_rank"] is None


@pytest.mark.parametrize("representation", ["pauli_jw", "df_native"])
def test_deterministic_adapter_sparse_mean_matches_native_circuit(representation):
    h = pauli() if representation == "pauli_jw" else df()
    a = prepare_signal_adapter(h, FixedSignalConfig(representation, T=-.17, q=3))
    psi = np.ones(1 << a.num_qubits)/math.sqrt(1 << a.num_qubits)
    assert_allclose(sparse_mean_action(a, psi, q=3), Statevector(psi).evolve(a.trajectory(q=3, seed=0)).data, atol=1e-13)


def test_df_b0i_phase_matches_known_tail_identity():
    h = df()
    b0 = prepare_signal_adapter(h, FixedSignalConfig("df_native", method="B0", L_D=1))
    b0i = prepare_signal_adapter(h, FixedSignalConfig("df_native", method="B0I", L_D=1))
    phase = np.exp(-1j*.2*b0i.original_preparation.extracted_identity_coefficient)
    assert_allclose(Operator(b0i.trajectory(q=1, seed=0)).data,
                    phase*Operator(b0.trajectory(q=1, seed=0)).data, atol=1e-13)
    assert not b0i.metadata()["B0I_equivalent_to_B0"]


@pytest.mark.parametrize("K", [0, 2])
def test_artificial_df_exact_circuit_law_mean_matches_sparse_mean(K):
    from trotterlib.df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
    from trotterlib.df_partial_s2_repeated_cost import make_exact_df_partial_s2_repeated_trajectory_stream
    a = prepare_signal_adapter(df(), FixedSignalConfig("df_native", method="PR", L_D=1, K=K))
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(a.preparation, a.step_time, 1,
        a.config, a.distribution, construction_policy="raw_concatenation")
    psi = np.array([0., 1., 1j, 0.])/np.sqrt(2)
    mean = np.zeros_like(psi)
    mass = 0.
    for request, probability in stream.records:
        evo = QiskitDFPartialS2RepeatedCircuitBuilder().build(request).circuit
        # Qiskit 1.3 Statevector may apply a circuit's global phase in place.
        mean += probability * Statevector(psi.copy()).evolve(evo).data
        mass += probability
    assert mass == pytest.approx(1., abs=1e-14)
    assert_allclose(mean, sparse_mean_action(a, psi, q=1), atol=1e-13)


def test_paired_axis_MC_total_includes_covariance():
    class FakeCache:
        hit_count = miss_count = 0

        def __init__(self):
            self.values = iter([0, 4, 4, 0])

        def get_or_transpile(self, *args, **kwargs):
            return types.SimpleNamespace(rz_count=next(self.values)), "mock", False

    result = evaluate_fixed_signal(pauli(), [1., 0.],
        FixedSignalConfig("pauli_jw", method="PR", L_D=1, cost_samples=2, T=0.),
        epsilon_sig=.1, cost_cache=FakeCache(), **NUMERICS)
    assert result["circuit_cost"]["axes"]["cosine"]["standard_error"] > 0
    assert result["total_RZ"]["standard_error"] == 0


def test_24_qubit_pauli_circuit_path_never_constructs_a_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("matrix construction requested")
    monkeypatch.setattr(PauliHamiltonian, "sparse_operator", forbidden)
    h = PauliHamiltonian.from_terms([("Z"*24, .4), ("X"*24, -.2)])
    a = prepare_signal_adapter(h, FixedSignalConfig("pauli_jw", method="PR", L_D=1))
    circuit = a.trajectory(q=2, seed=1)
    assert circuit.num_qubits == 24


def test_two_qubit_noncommuting_pauli_end_to_end_and_input_state_preserved():
    h = PauliHamiltonian.from_terms([("II", .1), ("ZI", .7), ("XZ", -.2), ("YY", .11), ("IX", .09)])
    psi = np.array([1., 1j, -.4, .2j])
    psi /= np.linalg.norm(psi)
    original = psi.copy()
    for method in ["PF", "PR"]:
        c = FixedSignalConfig("pauli_jw", method=method, L_D=1 if method == "PR" else None,
                              T=.15, q=2, r=2, cost_samples=4)
        result = evaluate_fixed_signal(h, psi, c, epsilon_sig=.1, verify_compiled=True, **NUMERICS)
        reference = np.vdot(original, expm(-.15j*h.sparse_operator().toarray())@original)
        assert_allclose(complex(**result["reference"]["signal"]), reference, atol=1e-14)
        assert result["total_RZ"] is not None
        assert_allclose(psi, original, atol=0)


def test_result_validator_rejects_invented_G_and_DF_rank():
    import copy
    result = evaluate_fixed_signal(pauli(), [1., 0.], FixedSignalConfig("pauli_jw"), epsilon_sig=.1)
    bad = copy.deepcopy(result)
    bad["total_RZ"] = dict(mean=0.)
    with pytest.raises(ValueError, match="NULL"):
        validate_fixed_signal_result(bad)
    bad = copy.deepcopy(result)
    bad["model"]["DF_rank"] = 0
    with pytest.raises(ValueError, match="DF_RANK"):
        validate_fixed_signal_result(bad)


def test_result_validator_checks_exact_CNG_identity():
    result = evaluate_fixed_signal(pauli(), [1., 0.], FixedSignalConfig("pauli_jw"), epsilon_sig=.1, **NUMERICS)
    result["total_RZ"]["mean_exact"] = "1"
    with pytest.raises(ValueError, match="G_RZ_IDENTITY"):
        validate_fixed_signal_result(result)


@pytest.mark.parametrize("allocation", [dict(epsilon_axes={}), dict(alpha_axes={}),
    dict(epsilon_axes=dict(cosine=.1, sine=.1)), dict(alpha_axes=dict(cosine=.5, sine=.5))])
def test_invalid_axis_allocations_are_rejected(allocation):
    with pytest.raises(ValueError):
        fixed_signal_shots(epsilon_sig=.1, bias={a:0. for a in AXES}, allowance={a:0. for a in AXES},
                           log_B_upper=0., **allocation)


def test_molecular_guard_is_active(request):
    if not request.config.pluginmanager.hasplugin("no_molecular_plugin"):
        pytest.skip("Guard probe requires -p no_molecular_plugin; no molecular call attempted")
    import pyscf.gto
    with pytest.raises(RuntimeError, match="NO_MOLECULAR_WORK"):
        pyscf.gto.Mole().build()


@pytest.mark.parametrize("numerics", [{}, dict(NUMERICS, log_B_margin=None),
                                   dict(NUMERICS, numerical_allowance={a:None for a in AXES})])
def test_missing_numerical_or_normalization_evidence_never_invents_N_G(numerics):
    result = evaluate_fixed_signal(pauli(), [1., 0.], FixedSignalConfig("pauli_jw"), epsilon_sig=.1, **numerics)
    assert result["shots"]["N_total"] is None and result["total_RZ"] is None


def test_ineligible_precision_has_no_successful_total():
    result = evaluate_fixed_signal(pauli(), [1., 0.], FixedSignalConfig("pauli_jw", method="B0", L_D=0),
                                   epsilon_sig=1e-5, **NUMERICS)
    assert not result["shots"]["usable"] and result["total_RZ"] is None


@pytest.mark.parametrize("kwargs", [dict(q=0), dict(q=True), dict(K=1), dict(r=0), dict(T=float("nan")),
                                    dict(seed=-1), dict(cost_samples=0), dict(hamiltonian_type="other"),
                                    dict(product_formula="S4"), dict(randomization_law="qdrift")])
def test_invalid_or_unsupported_configuration(kwargs):
    options = dict(hamiltonian_type="pauli_jw")
    options.update(kwargs)
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        FixedSignalConfig(**options)


@pytest.mark.parametrize("state", [[2., 0.], [float("nan"), 0.], [1., 0., 0., 0.]])
def test_invalid_state_rejected(state):
    with pytest.raises(ValueError):
        evaluate_fixed_signal(pauli(), state, FixedSignalConfig("pauli_jw"), epsilon_sig=.1)


def test_pf_truncation_requires_explicit_B0():
    with pytest.raises(ValueError, match="B0"):
        prepare_signal_adapter(pauli(), FixedSignalConfig("pauli_jw", L_D=0))


def legacy_shots():
    folder = ROOT/"artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/fixed_provider/src/trottertracks/resource_applicability"
    package = "_fixed_signal_legacy_accounting"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        sys.modules[package] = module
    spec = importlib.util.spec_from_file_location(package+".ax2b_numerical_accounting", folder/"ax2b_numerical_accounting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.u_aware_shots


@pytest.mark.parametrize("bias,u,epsilon,logB", [(0., 0., .05, 0.), (.01, 1e-8, .1, .3),
    (.1, .001, .01, 0.), (.01, .1, .1, 0.), (0., 0., .1, 50.), (None, None, .1, 0.)])
def test_shared_hoeffding_reproduces_frozen_DF_shot_contract(bias,u,epsilon,logB):
    old = legacy_shots()({a:bias for a in ["real", "imag"]}, {a:u for a in ["real", "imag"]},
                         epsilon=epsilon, log_B_upper=logB, evidence_kind="EMPIRICAL", evidence_ref="synthetic")
    new = fixed_signal_shots(epsilon_sig=epsilon, bias={a:bias for a in AXES}, allowance={a:u for a in AXES},
                             log_B_upper=logB, evidence_kind="EMPIRICAL", evidence_ref="synthetic")
    assert new["N_total"] == old["N_total"]
    for a, b in zip(AXES, ["real", "imag"]):
        for key in ["status", "headroom", "shots"]:
            assert new["axes"][a][key] == old["axes"][b][key]


def test_unequal_axis_precision_and_failure_allocations():
    result = fixed_signal_shots(epsilon_sig=.5, bias={a:0. for a in AXES}, allowance={a:0. for a in AXES},
        epsilon_axes=dict(cosine=.3, sine=.4), alpha_axes=dict(cosine=.01, sine=.04),
        log_B_upper=0., evidence_kind="EMPIRICAL", evidence_ref="artificial")
    assert result["alpha_union"] == .05
    assert result["axes"]["cosine"]["shots"] == math.ceil(2/.3**2*math.log(200))


def test_input_schema_and_cli(tmp_path):
    data = dict(input_kind="artificial", model_id="one_qubit", state_order="qiskit_little_endian",
                state=[[1., 0.], [0., 0.]], hamiltonian=dict(format="pauli_terms", num_qubits=1,
                terms=[["I", .1], ["Z", .7], ["X", .2]]))
    inp = tmp_path/"input.json"
    out = tmp_path/"output.json"
    inp.write_text(json.dumps(data))
    h, state, provenance = load_artificial_input(inp, "pauli_jw")
    assert h.constant == .1 and len(provenance["input_sha256"]) == 64
    command = [sys.executable, str(ROOT/"scripts/run_fixed_signal_evaluation.py"), "--input", str(inp),
               "--output", str(out), "--hamiltonian-type", "pauli_jw", "--method", "PR", "--L-D", "1",
               "--epsilon-sig", ".2", "--cost-samples", "2", "--numerical-allowance", "1e-10",
               "--log-B-margin", "1e-12", "--evidence-ref", "artificial declared tolerance", "--verify-compiled"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert result["shots"]["usable"] and result["total_RZ"]["mean"] > 0
    assert result["provenance"]["source"]["input"]["input_sha256"] == provenance["input_sha256"]
    assert subprocess.run(command, capture_output=True).returncode != 0
    data["input_kind"] = "molecule"
    inp.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="artificial"):
        load_artificial_input(inp, "pauli_jw")
