"""Explicit matched T_eff/V_square circuit/mean diagnostics; no input I/O.

H4 calls require separate execution permission. Only deterministic selected
events are built, never sampled. Mean evaluation uses existing finite-RTE
moments and an independent paired-polynomial action, not event Monte Carlo.
"""
from dataclasses import asdict
import math
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from .o1_m1_matched_split import extract_matched_df_tail, pauli_tail_coefficients
from .df_rte_tail import basis_change_unitary, prepare_df_rte_event_inputs
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_trotter.decompose import diag_hermitian
from .df_trotter.ops import (build_one_body_gaussian_block, one_body_diagonal_primitives,
                             append_diagonal_primitives)
from .pauli_signal_adapter import (PauliHamiltonian, append_pauli_event,
                                  paired_polynomial_action)
from .rte import (RTEComponent, _make_event, make_rte_config, event_unitary,
                  finite_rte_operator_moments)


def difference(left, right):
    delta = np.asarray(left)-np.asarray(right)
    return dict(max_entry=float(np.max(abs(delta))), frobenius=float(np.linalg.norm(delta)))


def controlled_reference(unitary):
    d = len(unitary)
    out = np.eye(2*d, dtype=complex)
    out[d:, d:] = unitary
    return out


def one_body_circuit(block, time, *, controlled=False):
    n = len(block.eps)
    circuit = QuantumCircuit(n+int(controlled))
    for gate, qubits in reversed(block.U_ops):
        circuit.append(gate.inverse(), list(qubits))
    append_diagonal_primitives(circuit, one_body_diagonal_primitives(block.eps, time),
                               controlled=controlled, ancilla_qubit=n if controlled else None)
    for gate, qubits in block.U_ops:
        circuit.append(gate, list(qubits))
    return circuit


def representative_events(components, distribution):
    """Deterministic coverage of fragment, support kind, sign and order phase.

    This is a circuit semantic check, not a probabilistic sample or complete
    enumeration. Full enumeration is tested only on a small artificial input.
    """
    chosen = {}
    for i, c in enumerate(components):
        key = (c.df_fragment_id, c.is_identity, len(c.diagonal_pauli_support or ()), c.coefficient_sign)
        chosen.setdefault(key, i)
    indices = tuple(chosen.values())
    events = [_make_event((i,), components, distribution, 0) for i in indices]
    if 2 in distribution.orders:
        oi = distribution.orders.index(2)
        for j, i in enumerate(indices):
            events.append(_make_event((i, indices[(j+1)%len(indices)],
                                       indices[(j+2)%len(indices)]), components, distribution, oi))
    return events


def validate_matched_circuit_signals(split, reference, *, step_times, r, K_values, q,
                                     identity_policy="faithful_identity_in_tail"):
    """Run only on explicitly supplied arrays. No state solve, fit or cost.

    reference is the preceding fixed driver's CAR/exterior-power helper module.
    Both dictionaries share the actual same Gaussian T_eff half-step program.
    Only the faithful policy is fixed for this minimal H4 comparison.
    """
    if identity_policy != "faithful_identity_in_tail":
        raise ValueError("This comparison fixes faithful identity in both tails")
    from .rte import require_integer_count
    r = require_integer_count(r, name="r", minimum=1)
    q = require_integer_count(q, name="q", minimum=1)
    K_values = tuple(require_integer_count(k, name="K") for k in K_values)
    step_times = tuple(float(t) for t in step_times)
    if not K_values or any(k % 2 for k in K_values) or not step_times or any(not math.isfinite(t) or t <= 0 for t in step_times):
        raise ValueError("Positive finite times and nonnegative even cutoffs required")
    # Require the actual decomposition backend, never silently choose a dense
    # UnitaryGate fallback when claiming a Gaussian circuit check.
    from qiskit_nature.second_q.circuit.library import BogoliubovTransform
    inputs = split.inputs
    n, d = inputs.num_qubits, 1 << inputs.num_qubits
    eye = np.eye(d, dtype=complex)
    a = reference.annihilators(n)
    T = reference.one_body(inputs.df_one_body, a).toarray()
    R = sum((w*(reference.one_body(g,a) @ reference.one_body(g,a)).toarray()
             for w,g in zip(inputs.df_lambdas, inputs.df_g_matrices, strict=True)), np.zeros_like(T))
    original = (inputs.nuclear_repulsion*eye + reference.one_body(inputs.spin_one,a).toarray()
                + reference.quartic(inputs.spin_two,a).toarray())
    df_full = inputs.nuclear_repulsion*eye+T+R
    extraction = extract_matched_df_tail(split, identity_policy=identity_policy)
    preparation = prepare_df_rte_event_inputs(extraction)
    retained = build_one_body_gaussian_block(inputs.df_one_body)
    coefficients, transfer = pauli_tail_coefficients(split, identity_policy=identity_policy)
    assert transfer == extraction.deterministic_identity_coefficient == 0.
    pR = PauliHamiltonian.from_terms(coefficients, num_qubits=n).sparse_operator().toarray()
    p_lambda = math.fsum(abs(c) for _,c in coefficients)
    p_components = tuple(RTEComponent(str(i), abs(c)/p_lambda, abs(c), 1 if c>0 else -1,
                                      is_identity=set(label)=={"I"})
                         for i,(label,c) in enumerate(coefficients))
    p_tail = SimpleNamespace(tail_id="matched_P", tail_hash="matched_P_numerical_check",
                             lambda_r=p_lambda, components=p_components)
    p_labels = {str(i):label for i,(label,c) in enumerate(coefficients)}
    s_tail = preparation.symbolic_tail
    bases = {key:basis_change_unitary(extraction.basis_definition(key)) for key in extraction.referenced_basis_ids}
    occupations = np.array([[(j>>bit)&1 for bit in range(n)] for j in range(d)])
    # Derive the test thresholds from binary64, operands and program lengths,
    # before computing any comparison delta. No synthetic atol is transferred.
    eps = np.finfo(float).eps
    scale = (abs(inputs.nuclear_repulsion)+np.sum(abs(inputs.spin_one))+np.sum(abs(inputs.spin_two))
             +np.sum(abs(inputs.df_one_body))
             +sum(abs(w)*np.sum(abs(g))**2 for w,g in zip(inputs.df_lambdas,inputs.df_g_matrices)))
    m = 4*n**4+8*d+16*len(inputs.df_lambdas)
    op_tol = float(m*eps/(1-m*eps)*max(1,scale))
    max_basis_length = max([len(retained.U_ops), *[len(extraction.basis_operations(k)) for k in bases]])
    basis_tol = float(32*eps*d*max(1,max_basis_length))
    implementation_tol = max(op_tol, basis_tol)*(1+q*r*(max(K_values)+2))*(1+max(step_times))
    checks = []

    def check(name, actual, expected, tolerance):
        record = dict(name=name, **difference(actual,expected), tolerance=tolerance)
        record["passed"] = record["max_entry"] <= tolerance
        checks.append(record)

    sR = np.zeros_like(T)
    component_ops = {}
    fragment_diagnostics = []
    for index,g in enumerate(inputs.df_g_matrices):
        U, eta = diag_hermitian(g, assume_hermitian=True)
        components = [c for c in extraction.components if c.df_fragment_id==f"df-fragment-{index}"]
        basis_id = components[0].basis_id
        b = bases[basis_id]
        # A basis gate may carry a global vacuum phase. It cancels in the
        # unconditional U^dagger / controlled-diagonal / U sandwich. Record
        # this convention explicitly; never phase-align an evolution/event.
        single = b[np.ix_([1<<j for j in range(n)], [1<<j for j in range(n)])]
        vacuum_phase = complex(b[0,0])
        if abs(vacuum_phase) < .5:
            raise ValueError("Gaussian basis does not preserve the vacuum")
        orbital = single/vacuum_phase
        check(f"Gaussian_basis_{index}_Fock_action_recorded_vacuum_phase", b,
              vacuum_phase*reference.fock_lift(orbital), basis_tol)
        check(f"Gaussian_basis_{index}_diagonalizes_saved_G",orbital.conj().T@g@orbital,np.diag(eta),op_tol)
        check(f"Gaussian_basis_{index}_vacuum_modulus",np.array([abs(vacuum_phase)]),np.array([1.]),basis_tol)
        diagonal = np.zeros(d)
        for c in components:
            z = np.prod(1-2*occupations[:,c.diagonal_pauli_support],axis=1)
            diagonal += c.coefficient*z
        reconstructed = (b*diagonal) @ b.conj().T
        raw = reference.one_body(g,a)
        exact = inputs.df_lambdas[index]*(raw@raw).toarray()
        check(f"Gaussian_fragment_{index}",reconstructed,exact,op_tol)
        sR += reconstructed
        fragment_diagnostics.append(dict(index=index, basis_id=basis_id,
            gate_names=sorted({gate.name for gate,qubits in extraction.basis_operations(basis_id)}),
            vacuum_phase=dict(real=vacuum_phase.real,imag=vacuum_phase.imag),
            orbital_column_gauge_max_entry=float(np.max(abs(U.conj().T@orbital-np.eye(n))))))
    check("DF_dictionary_sum_vs_CAR_R",sR,R,op_tol)
    check("Pauli_dictionary_sum_vs_CAR_R",pR,R,op_tol)
    builder = QiskitDFRTEEventCircuitBuilder(basis_registry=preparation.basis_registry)

    def s_operator(component_id):
        if component_id not in component_ops:
            c = next(c for c in extraction.components if c.component_id==component_id)
            z = np.prod(1-2*occupations[:,c.diagonal_pauli_support],axis=1)
            b = bases[c.basis_id]
            # Absorb the coefficient sign exactly as the RTE law requires.
            component_ops[component_id] = c.coefficient_sign*(b*z)@b.conj().T
        return component_ops[component_id]

    norm = float(np.vdot(inputs.state,inputs.state).real)
    if norm <= 0:
        raise ValueError("Nonzero original diagnostic state required")
    def signal(matrix):
        value = complex(np.vdot(inputs.state,matrix@inputs.state)/norm)
        return dict(real=value.real, imag=value.imag, radius=abs(value), phase=float(np.angle(value)))

    def signal_difference(actual, expected):
        left, right = signal(actual), signal(expected)
        z_left, z_right = complex(left["real"],left["imag"]), complex(right["real"],right["imag"])
        return dict(complex_absolute=abs(z_left-z_right), radius_absolute=abs(abs(z_left)-abs(z_right)),
                    wrapped_phase_absolute=float(abs(np.angle(z_left/z_right))) if abs(z_left)*abs(z_right)>0 else None)

    records = []
    event_records = []
    for delta in step_times:
        half_circuit = one_body_circuit(retained,delta/2)
        half = np.asarray(Operator(half_circuit).data)
        exact_half = expm(-.5j*delta*T)
        check(f"T_half_{delta}",half,exact_half,implementation_tol)
        controlled_half_circuit = one_body_circuit(retained,delta/2,controlled=True)
        controlled_half = np.asarray(Operator(controlled_half_circuit).data)
        check(f"controlled_T_half_relative_phase_{delta}",controlled_half,controlled_reference(exact_half),implementation_tol)
        pf_step = np.exp(-1j*delta*inputs.nuclear_repulsion)*exact_half@expm(-1j*delta*R)@exact_half
        pf = np.linalg.matrix_power(pf_step,q)
        original_exact = expm(-1j*q*delta*original)
        df_exact = expm(-1j*q*delta*df_full)
        for K in K_values:
            dictionary_records = {}
            corrected = {}
            independent_tail = paired_polynomial_action(R,eye,delta,r,K,normalization=1.)
            nominal_step = np.exp(-1j*delta*inputs.nuclear_repulsion)*exact_half@independent_tail@exact_half
            nominal_corrected = np.linalg.matrix_power(nominal_step,q)
            for name,tail,physical_tail in (("P",p_tail,pR),("DF",s_tail,sR)):
                config, distribution = make_rte_config(tail,evolution_time=delta,rte_steps=r,
                    truncation_tolerance=1.,finite_taylor_order=K)
                moments = finite_rte_operator_moments(physical_tail/tail.lambda_r,config)
                check(f"{name}_corrected_tail_{delta}_{K}",moments.corrected_operator,independent_tail,implementation_tol)
                step = np.exp(-1j*delta*inputs.nuclear_repulsion)*half@moments.corrected_operator@half
                corrected[name] = np.linalg.matrix_power(step,q)
                B = moments.normalization_product**q
                raw = corrected[name]/B
                check(f"{name}_normalization_identity_{delta}_{K}",B*raw,corrected[name],implementation_tol)
                check(f"{name}_implemented_vs_nominal_corrected_mean_{delta}_{K}",corrected[name],nominal_corrected,implementation_tol)
                dictionary_records[name] = dict(lambda_r=tail.lambda_r, config=asdict(config),
                    normalization=B, attenuation=1/B, corrected_signal=signal(corrected[name]), raw_signal=signal(raw),
                    implementation_error_vs_nominal=difference(corrected[name],nominal_corrected),
                    implementation_signal_error=signal_difference(corrected[name],nominal_corrected),
                    finite_RTE_error_vs_PF=difference(nominal_corrected,pf),
                    finite_RTE_signal_error_vs_PF=signal_difference(nominal_corrected,pf))
                # Selected native circuit semantics at the largest registered
                # cutoff. They do not estimate an expectation statistically.
                if K == max(K_values):
                    for event_number,event in enumerate(representative_events(tail.components,distribution)):
                        if name=="DF":
                            ops={cid:s_operator(cid) for cid in event.selected_component_ids}
                            expected=event_unitary(event,ops)
                            for control in (False,True):
                                built=builder.build_event(preparation.request_for_event(event,
                                    controlled=control,ancilla_qubit=n if control else None))
                                target=controlled_reference(expected) if control else expected
                                check(f"DF_event_{delta}_{len(event_records)}_control{control}",
                                      Operator(built.circuit).data,target,implementation_tol)
                            central_controlled = built.circuit
                        else:
                            ops={cid:tail.components[int(cid)].coefficient_sign*
                                 PauliHamiltonian.from_terms([(p_labels[cid],1.)],num_qubits=n).sparse_operator().toarray()
                                 for cid in event.selected_component_ids}
                            expected=event_unitary(event,ops)
                            circuit=QuantumCircuit(n)
                            append_pauli_event(circuit,event,p_labels)
                            check(f"P_event_{delta}_{len(event_records)}",Operator(circuit).data,expected,implementation_tol)
                            controlled=QuantumCircuit(n+1)
                            controlled.append(circuit.to_gate().control(1),[n,*range(n)])
                            check(f"P_controlled_event_{delta}_{len(event_records)}",Operator(controlled).data,
                                  controlled_reference(expected),implementation_tol)
                            central_controlled = controlled
                        if event_number == 0:
                            # One complete controlled step per dictionary/time
                            # checks the nuclear phase and both retained halves
                            # together, in addition to all selected event checks.
                            complete = QuantumCircuit(n+1)
                            complete.compose(controlled_half_circuit,inplace=True)
                            complete.compose(central_controlled,inplace=True)
                            complete.compose(controlled_half_circuit,inplace=True)
                            complete.p(-delta*inputs.nuclear_repulsion,n)
                            expected_step=np.exp(-1j*delta*inputs.nuclear_repulsion)*exact_half@expected@exact_half
                            check(f"{name}_complete_controlled_step_relative_phase_{delta}",
                                  Operator(complete).data,controlled_reference(expected_step),implementation_tol)
                        event_records.append(dict(dictionary=name,time=delta,order=event.taylor_order,
                            selected=event.selected_component_ids,phase_real=event.phase.real))
            check(f"P_DF_corrected_mean_{delta}_{K}",corrected["P"],corrected["DF"],implementation_tol)
            records.append(dict(delta=delta,q=q,r=r,K=K,T=q*delta,dictionaries=dictionary_records,
                PF_error_vs_saved_DF_exact=difference(pf,df_exact),
                PF_signal_error_vs_saved_DF_exact=signal_difference(pf,df_exact),
                original_vs_DF_exact_difference=difference(original_exact,df_exact),
                original_vs_DF_exact_signal_difference=signal_difference(original_exact,df_exact),
                nominal_corrected_signal=signal(nominal_corrected),
                original_exact_signal=signal(original_exact),DF_exact_signal=signal(df_exact),PF_signal=signal(pf)))
    return dict(status="PASS" if all(c["passed"] for c in checks) else "FAIL", checks=checks,
        tolerance=dict(operator=op_tol,basis=basis_tol,implementation=implementation_tol,epsilon=eps,
                       operand_scale=float(scale),m=m,max_basis_length=max_basis_length,
                       rationale="Predeclared binary64/program-scale absolute tests, no phase alignment, not rigorous enclosures"),
        conditions=dict(step_times=step_times,q=q,r=r,K_values=K_values,identity_policy=identity_policy,
                        common_retained_program="same Gaussian T_eff half circuit",full_dimension=d,
                        original_state_field="state",state_squared_norm=norm,sector_projection=False),
        backend=BogoliubovTransform.__module__,fragments=fragment_diagnostics,
        records=records,selected_events=event_records,
        limitation="Selected circuit events plus analytic exact finite mean; no full H4 event enumeration or sampling, no energy or cost qualification")
