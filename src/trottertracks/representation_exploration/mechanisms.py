"""Bounded, exact-data development diagnostics for representation candidates A/B/C.

No molecular data, ground-state oracle, stochastic optimizer, or production API
is changed. Dense exponentials here are references, never scalable oracles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

from trotterlib.df_rte_tail import dense_df_block_hamiltonian, exact_df_diagonal_coefficients
from trotterlib.df_trotter.model import Block, DFModel
from trotterlib.df_trotter.ops import build_df_blocks_givens
from trotterlib.rte import (
    InvolutoryTailTerm, enumerate_rte_events, event_unitary,
    exact_enumerated_event_mean_operator, finite_rte_distribution,
    finite_taylor_operator, normalize_involutory_tail,
)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
TIMES = (0.05, 0.2, 0.4)
ATOL = 2e-10
BASIS = ("rz", "sx", "x", "cx")
SEED = 20261010


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, 2))


def matrix(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=complex)
    return {"real": a.real.tolist(), "imag": a.imag.tolist()}


def rotate_factors(factors: list[np.ndarray], rotation: np.ndarray) -> list[np.ndarray]:
    """Real orthogonal label mixing for *unweighted*, Hermitian square factors."""
    o = np.asarray(rotation)
    if np.iscomplexobj(o) and np.any(o.imag != 0):
        raise ValueError("Factor mixing must be real to preserve Hermiticity.")
    o = np.asarray(o.real, dtype=float)
    if o.shape != (len(factors), len(factors)) or not np.all(np.isfinite(o)):
        raise ValueError("Invalid rotation shape or entries.")
    if not np.allclose(o.T @ o, np.eye(len(factors)), atol=1e-12, rtol=0):
        raise ValueError("Factor rotation must be orthogonal.")
    if any(not np.allclose(f, f.conj().T, atol=1e-12, rtol=0) for f in factors):
        raise ValueError("Factors must be Hermitian.")
    return [sum(o[a, j] * f for j, f in enumerate(factors)) for a in range(len(factors))]


def rotation(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])


def second_quantize_one_body(g: np.ndarray) -> np.ndarray:
    """Independent Jordan-Wigner reference; mode 0 is the least significant bit."""
    n = len(g)
    if not 1 <= n <= 3:
        raise ValueError("This exploration admits at most three fermionic modes.")
    annihilation = []
    for mode in range(n):
        a = np.zeros((2**n, 2**n), dtype=complex)
        for state in range(2**n):
            if (state >> mode) & 1:
                sign = (-1) ** ((state & ((1 << mode) - 1)).bit_count())
                a[state ^ (1 << mode), state] = sign
        annihilation.append(a)
    return sum(g[p, q] * annihilation[p].conj().T @ annihilation[q]
               for p in range(n) for q in range(n))


def controlled(u: np.ndarray) -> np.ndarray:
    """ordinary control with control qubit 0, hence interleaved matrix indices."""
    out = np.eye(2 * len(u), dtype=complex)
    out[1::2, 1::2] = u
    return out


@dataclass
class CircuitAudit:
    count: int = 0
    cap: int = 64

    def compile(self, circuit: QuantumCircuit, reference: np.ndarray) -> dict[str, Any]:
        if self.count >= self.cap or circuit.num_qubits > 5:
            raise ValueError("Circuit resource guard exceeded.")
        self.count += 1
        built_residual = norm(Operator(circuit).data - reference)
        compiled = transpile(circuit, basis_gates=list(BASIS), optimization_level=1,
                             seed_transpiler=SEED, num_processes=1)
        compiled_residual = norm(Operator(compiled).data - reference)
        if max(built_residual, compiled_residual) > ATOL:
            raise AssertionError(f"Circuit equivalence failed: {built_residual}, {compiled_residual}")
        return {"qubits": circuit.num_qubits, "rz": int(compiled.count_ops().get("rz", 0)),
                "cx": int(compiled.count_ops().get("cx", 0)),
                "sx": int(compiled.count_ops().get("sx", 0)),
                "x": int(compiled.count_ops().get("x", 0)),
                "size": compiled.size(), "depth": compiled.depth(),
                "build_operator_residual": built_residual,
                "compiled_operator_residual": compiled_residual,
                "qasm": compiled.qasm() if hasattr(compiled, "qasm") else None}


def wrapper(qc: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits + 1)
    out.h(0)
    out.append(qc.to_gate().control(1), range(out.num_qubits))
    out.h(0)
    return out


def wrapper_reference(u: np.ndarray) -> np.ndarray:
    h = np.kron(np.eye(len(u)), np.array([[1, 1], [1, -1]]) / math.sqrt(2))
    return h @ controlled(u) @ h


def coefficient_l1(g: np.ndarray, *, identity: bool) -> float:
    eta = np.linalg.eigvalsh(g)
    return float(sum(abs(c) for support, c in exact_df_diagonal_coefficients(eta, 1.0)
                     if identity or support))


def factor_cases() -> dict[str, list[np.ndarray]]:
    diagonal = np.diag([1., 1., -1.])
    other = np.diag([.1, -.8, -.7])
    perturbed = other.copy()
    perturbed[0, 2] = perturbed[2, 0] = .04
    return {"review_squares": [I + X, I + Z],
            "spectral_proxy_mismatch": [diagonal, perturbed],
            "commuting_control": [diagonal, other],
            "isotropic_no_gain": [X / math.sqrt(2), Z / math.sqrt(2)]}


def candidate_a(audit: CircuitAudit) -> dict[str, Any]:
    records = []
    angles = (0., math.atan(.1), -math.pi / 8, math.pi / 8, math.pi / 4)
    inputs = {}
    for name, factors in factor_cases().items():
        n = len(factors[0])
        original = [second_quantize_one_body(g) for g in factors]
        h = sum(a @ a for a in original)
        gram = np.array([[np.vdot(a, b).real for b in factors] for a in factors])
        proxy_optimum = float(np.linalg.eigvalsh(gram)[-1])
        inputs[name] = {"factors": [matrix(g) for g in factors], "hamiltonian": matrix(h),
                        "gram": gram.tolist(), "rank": 2, "modes": n,
                        "geometry": None, "basis": "synthetic real orbital matrices / JW full Fock space",
                        "ld": 1, "one_body": "zero", "constant": 0.,
                        "gram_rank": int(np.linalg.matrix_rank(gram))}
        inputs[name]["all_original_ld1_choices"] = [
            {"d_index":idx,"tail_l1_identity_extracted":coefficient_l1(factors[1-idx],identity=False),
             "tail_l1_faithful":coefficient_l1(factors[1-idx],identity=True)} for idx in (0,1)]
        for theta in angles:
            gs = rotate_factors(factors, rotation(theta))
            lifted = [second_quantize_one_body(g) for g in gs]
            squares = [a @ a for a in lifted]
            weights = [float(np.vdot(g, g).real) for g in gs]
            d_index = int(np.argmax(weights))
            r_index = 1 - d_index
            d, r = squares[d_index], squares[r_index]
            delta_errors = []
            for t in TIMES:
                s2 = expm(-.5j*t*d) @ expm(-1j*t*r) @ expm(-.5j*t*d)
                delta_errors.append({"delta": t, "s2_operator_error": norm(s2-expm(-1j*t*h)),
                                     "local_error_over_delta_cubed": norm(s2-expm(-1j*t*h))/t**3})
            tail_l1 = coefficient_l1(gs[r_index], identity=False)
            faithful = coefficient_l1(gs[r_index], identity=True)
            record = {"case": name, "theta": theta, "d_index": d_index, "r_index": r_index,
                      "hamiltonian_residual": norm(sum(squares)-h),
                      "factors": [matrix(g) for g in gs], "weights": weights,
                      "total_frobenius_weight": sum(weights),
                      "retained_proxy_regret": proxy_optimum-weights[d_index],
                      "rank": int(np.linalg.matrix_rank(np.stack([g.ravel() for g in gs]))),
                      "tail_l1_identity_extracted": tail_l1,
                      "tail_l1_faithful": faithful,
                      "commutator_norm": norm(d @ r-r @ d), "delta_rows": delta_errors,
                      "b_squared_diagnostic_at_delta_0p2_k2":
                      finite_rte_distribution(.2*tail_l1, 2).exact_finite_distribution**2}
            if theta == 0. or theta == (math.pi/4 if name == "review_squares" else math.atan(.1)):
                model = DFModel(np.ones(2), gs, np.zeros((n,n)), 0., n)
                blocks = build_df_blocks_givens(model)
                record["df_reconstruction_residual"] = max(
                    norm(dense_df_block_hamiltonian(b)-square) for b, square in zip(blocks,squares))
                qc = QuantumCircuit(n)
                for idx, time in ((d_index,.1),(r_index,.2),(d_index,.1)):
                    Block.from_df(blocks[idx]).apply(qc,time)
                u = expm(-.1j*d) @ expm(-.2j*r) @ expm(-.1j*d)
                record["controlled_s2_wrapper_cost"] = audit.compile(wrapper(qc), wrapper_reference(u))
            records.append(record)

    # Orthogonality does not preserve unequal signed/weighted squares if the
    # coefficients are left attached to the transformed labels.
    f = [I+X, I+Z]
    mixed = rotate_factors(f, rotation(math.pi/4))
    wrong_positive = norm(2*mixed[0]@mixed[0]+mixed[1]@mixed[1]-(2*f[0]@f[0]+f[1]@f[1]))
    wrong_signed = norm(mixed[0]@mixed[0]-mixed[1]@mixed[1]-(f[0]@f[0]-f[1]@f[1]))
    weighted = rotate_factors([math.sqrt(2)*f[0],f[1]],rotation(math.pi/4))
    weighted_ok = norm(sum(g@g for g in weighted)-(2*f[0]@f[0]+f[1]@f[1]))
    epsilon = .01
    fs = factor_cases()["spectral_proxy_mismatch"]
    h_full = sum(second_quantize_one_body(g)@second_quantize_one_body(g) for g in fs)
    core = sum(second_quantize_one_body(np.diag(np.diag(g)))@
               second_quantize_one_body(np.diag(np.diag(g))) for g in fs)
    residual = h_full-core
    paulis = SparsePauliOp.from_operator(Operator(residual),atol=0.,rtol=0.)
    return {"inputs": inputs, "rows": records,
            "shared_diagonal_frame_exact_residual":{
                "case":"spectral_proxy_mismatch","core":matrix(core),
                "residual":matrix(residual),"reconstruction_residual":norm(core+residual-h_full),
                "residual_operator_norm":norm(residual),
                "residual_pauli_l1":float(sum(abs(c) for c in paulis.coeffs)),
                "pauli_coefficients":[{"label":label,"real":c.real,"imag":c.imag}
                                      for label,c in zip(paulis.paulis.to_labels(),paulis.coeffs)],
                "comment":"Exact-data shared computational frame diagnostic; no residual dropped or cost winner inferred."},
            "weighted_controls": {"unabsorbed_unequal_positive_residual": wrong_positive,
                                  "signed_orthogonal_residual": wrong_signed,
                                  "absorbed_positive_residual": weighted_ok},
            "residual_dictionary_counterexample": {
                "epsilon": epsilon, "residual_operator_norm": epsilon,
                "unmerged_coefficients": [1+epsilon,-1.],
                "unmerged_l1": 2+epsilon, "merged_l1": epsilon,
                "interpretation": "epsilon Z=(1+epsilon)Z-Z; no small-l1 inference from operator norm"}}


def reflected_dictionary(terms: list[InvolutoryTailTerm], s: np.ndarray) -> list[InvolutoryTailTerm]:
    return [InvolutoryTailTerm(f"{term.component_id}:{bit}", term.coefficient/2,
                              term.operator if bit == 0 else s @ term.operator @ s)
            for term in terms for bit in (0,1)]


def candidate_b() -> dict[str, Any]:
    # Tensor order is aux (most significant) then system (least significant).
    p = np.diag([1.,1.,0.,0.]).astype(complex)
    q = np.eye(4)-p
    s = 2*p-np.eye(4)
    terms = [InvolutoryTailTerm("physical_z", .7, np.kron(I,Z)),
             InvolutoryTailTerm("conditional_x", .2, np.kron(Z,X)),
             InvolutoryTailTerm("leaky_xx", .4, np.kron(X,X))]
    raw = normalize_involutory_tail("leaky", terms)
    expanded = normalize_involutory_tail("generator_reflected",reflected_dictionary(terms,s))
    h = raw.dense_hamiltonian
    hbar = (h+s@h@s)/2
    psi = np.array([1.,0.,0.,0.],dtype=complex)
    rho = np.outer(psi,psi.conj())
    rows, event_rows = [], []
    for t in (.05,.2,-.2):
        for k in (0,2,4):
            dist = finite_rte_distribution(expanded.lambda_r*t,k)
            events = enumerate_rte_events(expanded.components,dist,max_events=10000)
            ops = {c.component_id:o for c,o in zip(expanded.components,expanded.operators)}
            mean = exact_enumerated_event_mean_operator(events,ops)
            target_poly = finite_taylor_operator(hbar/expanded.lambda_r,expanded.lambda_r*t,k)
            # One bit shared across the whole word gives the twirl of a different polynomial.
            wrong = finite_taylor_operator(h/raw.lambda_r,raw.lambda_r*t,k)
            wrong = (wrong+s@wrong@s)/2
            channel = np.zeros((4,4),dtype=complex)
            second_moment = 0.
            reflection_expectation = 0.
            for idx,event in enumerate(events):
                u = event_unitary(event,ops)
                state = u@psi
                leakage = float(np.vdot(state,q@state).real)
                channel += event.event_probability * u@rho@u.conj().T
                second_moment += event.event_probability * np.linalg.norm(q@u@p,"fro")**2
                reflections = 2*sum(cid.endswith(":1") for cid in event.selected_component_ids)
                reflection_expectation += event.event_probability*reflections
                event_rows.append({"t":t,"k":k,"index":idx,
                                   "order":event.taylor_order,
                                   "selected_ids":list(event.selected_component_ids),
                                   "rotation_angle":event.rotation_angle,
                                   "phase_real":event.phase.real,"phase_imag":event.phase.imag,
                                   "probability":event.event_probability,
                                   "leakage_probability_psi0":leakage,
                                   "unfused_reflection_count":reflections})
            corrected = dist.exact_finite_distribution*mean
            physical_exact = expm(-1j*t*hbar)
            whole_twirl = (expm(-1j*t*h)+s@expm(-1j*t*h)@s)/2
            rows.append({"t":t,"k":k,"event_count":len(events),
                         "probability_sum":sum(e.event_probability for e in events),
                         "lambda_original":raw.lambda_r,"lambda_expanded":expanded.lambda_r,
                         "lambda_merged_pauli":.9,
                         "normalization":dist.exact_finite_distribution,
                         "b_squared_variance_envelope":dist.exact_finite_distribution**2,
                         "hoeffding_axis_shots_diagnostic_epsilon0p05_alpha_total0p05":
                         math.ceil(2*dist.exact_finite_distribution**2/.05**2*math.log(4/.05)),
                         "corrected_polynomial_residual":norm(corrected-target_poly),
                         "corrected_mean_leakage_norm":norm(q@corrected@p),
                         "channel_leakage_probability":float(np.trace(q@channel).real),
                         "individual_max_leakage_probability":max(r["leakage_probability_psi0"]
                                                                  for r in event_rows[-len(events):]),
                         "finite_taylor_physical_bias":norm(p@(corrected-physical_exact)@p),
                         "wrong_whole_word_physical_bias":norm(p@(wrong-target_poly)@p),
                         "whole_exact_twirl_physical_bias":norm(p@(whole_twirl-physical_exact)@p),
                         "original_exact_physical_leakage_norm":norm(q@expm(-1j*t*h)@p),
                         "expected_unfused_reflections":reflection_expectation,
                         "corrected_leakage_frobenius_rms_at_n1024":
                         dist.exact_finite_distribution*math.sqrt(second_moment/1024),
                         "mean_operator":matrix(mean),"corrected_operator":matrix(corrected)})
    # A sum may preserve P while splitting its canceling leakage across blocks does not.
    a = .7*np.kron(I,Z)+.4*np.kron(X,X)
    b = .2*np.kron(Z,X)-.4*np.kron(X,X)
    split = expm(-.1j*a) @ expm(-.2j*b) @ expm(-.1j*a)
    return {"inputs":{"p":matrix(p),"s":matrix(s),"h_tilde":matrix(h),"h_bar":matrix(hbar),
                      "terms":[{"id":term.component_id,"coefficient":term.coefficient,
                                "operator":matrix(term.operator)} for term in terms],
                      "psi":matrix(psi),"geometry":None,"basis":"one aux qubit and one system qubit",
                      "df_rank":None,"ld":0,"rte_steps":1},
            "rows":rows,"events":event_rows,
            "symmetric_generator_leakage":norm(q@hbar@p),
            "all_blocks_condition_counterexample":{
                "total_generator_leakage":norm(q@(a+b)@p),
                "s2_leakage_norm_at_0p2":norm(q@split@p)},
            "known_symmetry_echo_reference":[{
                "r":r,"t":.2,"exact_h_tilde_oracle_calls":2*r,
                "physical_bias":norm(p@(np.linalg.matrix_power(
                    expm(-.1j*h/r)@s@expm(-.1j*h/r)@s,r)-expm(-.2j*hbar))@p),
                "leakage_norm":norm(q@np.linalg.matrix_power(
                    expm(-.1j*h/r)@s@expm(-.1j*h/r)@s,r)@p),
                "cost_scope":"abstract exact H_tilde oracle count; not compiled cost"
                } for r in (1,2,4,8)],
            "analytic_whole_twirl_counterexample":{
                "h_tilde":"X","p":"|0><0|","s":"Z","t":.2,
                "correct_exp":"I","wrong_mean":"cos(t) I",
                "physical_bias":1-math.cos(.2)}}


def vacuum_reflection(n_aux: int) -> QuantumCircuit:
    if n_aux < 1 or n_aux > 3:
        raise ValueError("Auxiliary reflection capped at three qubits.")
    qc = QuantumCircuit(n_aux)
    for i in range(n_aux):
        qc.x(i)
    if n_aux == 1:
        qc.z(0)
    else:
        qc.h(n_aux-1)
        qc.mcx(list(range(n_aux-1)),n_aux-1)
        qc.h(n_aux-1)
    for i in range(n_aux):
        qc.x(i)
    qc.global_phase = math.pi  # -(I-2|vac><vac|) = 2P-I, control-sensitive.
    return qc


def reflection_costs(audit: CircuitAudit) -> list[dict[str, Any]]:
    rows = []
    for n_aux in (1,2,3):
        reflection = vacuum_reflection(n_aux)
        s = -np.eye(2**n_aux,dtype=complex)
        s[0,0] = 1
        row = {"aux_qubits":n_aux,"reflection":audit.compile(reflection,s)}
        base = QuantumCircuit(n_aux+1)
        base.h(0)
        base.crx(.36,0,1)
        base.h(0)
        conjugate = QuantumCircuit(n_aux+1)
        conjugate.compose(reflection,qubits=list(range(1,n_aux+1)),inplace=True)
        conjugate.compose(base,inplace=True)
        conjugate.compose(reflection,qubits=list(range(1,n_aux+1)),inplace=True)
        sf = np.kron(s,I)
        row["controlled_base"] = audit.compile(base,Operator(base).data)
        row["uncontrolled_reflection_sandwich"] = audit.compile(conjugate,sf@Operator(base).data@sf)
        # The control=0 branch really is identity before the Hadamard wrapper.
        rows.append(row)
    return rows


def lie_dimension(generators: list[np.ndarray], atol: float = 1e-9) -> int:
    """Numerical traceless Hermitian Lie closure; small matrices only."""
    d = len(generators[0])
    if d > 8:
        raise ValueError("Lie closure dimension guard exceeded.")
    basis: list[np.ndarray] = []
    def add(h: np.ndarray) -> bool:
        h = (h+h.conj().T)/2
        h = h-np.trace(h)*np.eye(d)/d
        for v in basis:
            h = h-np.vdot(v,h).real*v
        for v in basis:
            h = h-np.vdot(v,h).real*v
        length = np.linalg.norm(h)
        if length <= atol:
            return False
        basis.append(h/length)
        return True
    for g in generators:
        add(g)
    cursor = 0
    while cursor < len(basis):
        for other in list(basis[:cursor]):
            add(1j*(basis[cursor]@other-other@basis[cursor]))
        cursor += 1
        if len(basis) >= d*d-1:
            break
    return len(basis)


def diagonal_a(t: float) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(2*t,0)
    qc.rz(1.4*t,1)
    qc.rzz(.6*t,0,1)
    return qc


def conditional_mixed(t: float, alpha: float, target: int) -> QuantumCircuit:
    """Exact mixed propagator for A=Z0+.7Z1+.3Z0Z1 and B=X_target.

    This is a known conditional SU(2) construction, with two spectator branches.
    It is a primitive witness, not a new algorithm.
    """
    if target not in (0,1):
        raise ValueError("Target must be 0 or 1.")
    spectator = 1-target
    field = (1.,.7)[target]
    spectator_field = (.7,1.)[target]
    qc = QuantumCircuit(2)
    qc.rz(2*t*spectator_field,spectator)
    for bit in (0,1):
        detuning = field+.3*(-1)**bit
        theta = math.atan2(alpha,detuning)
        radius = math.hypot(detuning,alpha)
        if bit == 0:
            qc.x(spectator)
        qc.cry(-theta,spectator,target)
        qc.crz(2*t*radius,spectator,target)
        qc.cry(theta,spectator,target)
        if bit == 0:
            qc.x(spectator)
    return qc


def candidate_c(audit: CircuitAudit) -> dict[str, Any]:
    a = np.kron(I,Z)+.7*np.kron(Z,I)+.3*np.kron(Z,Z)
    bs = [np.kron(I,X),np.kron(X,I)]
    rows = []
    for alpha in (.05,.1,.2):
        t = .2
        exact = expm(-1j*t*(a+alpha*sum(bs)))
        thrift = expm(-1j*t*(a+alpha*bs[0])) @ expm(1j*t*a) @ expm(-1j*t*(a+alpha*bs[1]))
        ordinary = expm(-1j*t*a) @ expm(-1j*t*alpha*bs[0]) @ expm(-1j*t*alpha*bs[1])
        # Replacing each mixed oracle by ordinary splitting removes the alpha^2 benefit.
        surrogate = (expm(-1j*t*a)@expm(-1j*t*alpha*bs[0])) @ expm(1j*t*a) @ (
                     expm(-1j*t*a)@expm(-1j*t*alpha*bs[1]))
        record = {"alpha":alpha,"t":t,"thrift_error":norm(thrift-exact),
                  "ordinary_first_order_error":norm(ordinary-exact),
                  "cheap_split_surrogate_error":norm(surrogate-exact),
                  "mixed_primitive_errors":[]}
        for target in (0,1):
            qc = conditional_mixed(t,alpha,target)
            u = expm(-1j*t*(a+alpha*bs[target]))
            record["mixed_primitive_errors"].append(norm(Operator(qc).data-u))
        if alpha == .1:
            qc = QuantumCircuit(2)
            qc.compose(conditional_mixed(t,alpha,1),inplace=True)
            qc.compose(diagonal_a(-t),inplace=True)
            qc.compose(conditional_mixed(t,alpha,0),inplace=True)
            record["thrift_wrapper_cost"] = audit.compile(wrapper(qc),wrapper_reference(thrift))
            qc2 = QuantumCircuit(2)
            qc2.rx(2*t*alpha,1)
            qc2.rx(2*t*alpha,0)
            qc2.compose(diagonal_a(t),inplace=True)
            record["ordinary_wrapper_cost"] = audit.compile(wrapper(qc2),wrapper_reference(ordinary))
        rows.append(record)
    return {"inputs":{"a":matrix(a),"bs":[matrix(b) for b in bs],
                      "geometry":None,"basis":"two-qubit computational Pauli basis",
                      "df_rank":None,"ld":None,"t":.2},"rows":rows,
            "lie_closures":{"commuting":lie_dimension([np.kron(I,Z),np.kron(Z,I)]),
                            "single_su2":lie_dimension([Z,X]),
                            "one_conditional_mixed":lie_dimension([a,bs[0]]),
                            "all_access_generators":lie_dimension([a,*bs])},
            "interpretation":"Mixed oracles exist here with bounded two-branch SU(2) cost; the union need not have a small Lie closure. No general chemistry construction established."}


def run_all() -> dict[str, Any]:
    audit = CircuitAudit()
    a = candidate_a(audit)
    b = candidate_b()
    b["reflection_cost_rows"] = reflection_costs(audit)
    c = candidate_c(audit)
    return {"schema_version":1,"status":"INITIAL_MECHANISMS_COMPLETE_AWAITING_GPT_REVIEW",
            "scope":"synthetic exact-data development diagnostics, no independent scientific replication",
            "candidate_a":a,"candidate_b":b,"candidate_c":c,
            "resource_counts":{"compiled_circuits":audit.count,
                               "enumerated_b_events":len(b["events"]),
                               "molecular_loads":0,"ground_state_solves":0,
                               "quantum_shots":0,"gpu_calls":0},
            "next_stage_authorized":False,"central_hypothesis_adopted":None}
