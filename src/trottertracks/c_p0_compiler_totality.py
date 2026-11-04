"""Technical domain checks around unchanged P0; no sampler or H4 runner.

These checks reject unsupported inputs/outputs. Passing them is neither a
termination proof nor a synthesis-error certificate. Caller supplies the
existing shared Qiskit call context and owns scientific execution permission.
"""
import math

BASE = 'bd560a45410409573ae849ca5aaf4050343910bf'
SCOPE = 'C_P0_COMPILER_TOTALITY_READINESS_V1'
CONTRACT = 'docs/research/research_c_p0_compiler_totality_contract_unapproved_20261104.json'
FAMILY = 'artifacts/research_c_p0_compiler_totality/2026-11-04'


def check_word(word, atom_ids, grid_counts, *, length=32):
    """Supplied IDs/counts only: no probabilities regenerated, no draws.

    The positional table must already have the scientifically bound ordering.
    This schema cannot establish that provenance on its own.
    """
    ids, counts, word = tuple(atom_ids), tuple(grid_counts), tuple(word)
    if (type(length) is not int or length < 1 or len(word) != length
            or not ids or len(ids) != len(counts)
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
            or any(type(x) is not int or x < 0 for x in counts)
            or sum(counts) != 1 << 53):
        raise ValueError('WORD_OR_53_BIT_SUPPORT_SCHEMA')
    support = {key for key, count in zip(ids, counts) if count > 0}
    if any(key not in support for key in word):
        raise ValueError('WORD_OUTSIDE_POSITIVE_GRID_SUPPORT')
    return dict(length=length, support_size=len(support),
                all_q_words_compile_success_qualified=False)


def _finite(value):
    try:
        if not math.isfinite(float(value)):
            raise ValueError('NONFINITE_PARAMETER_OR_PHASE')
    except (TypeError, OverflowError) as error:
        raise ValueError('FINITE_BOUND_REAL_REQUIRED') from error


def check_circuit(circuit, *, qubits, native=False):
    """Bound, finite, unitary fixed-emitter grammar; no operator evaluation.

    Standard classes are checked as well as names; opaque look-alike gates
    cannot enter the covered definition path. No extra numeric-angle cutoff.
    """
    from qiskit.circuit import ControlledGate
    from qiskit.circuit.library import (RZGate, SXGate, XGate, CXGate,
        PhaseGate, XXPlusYYGate, CRZGate, RZZGate)
    classes = {RZGate, SXGate, XGate, CXGate}
    if not native:
        classes |= {PhaseGate, XXPlusYYGate, CRZGate}
    if (type(qubits) is not int or qubits < 1 or circuit.num_qubits != qubits
            or circuit.num_clbits or circuit.parameters or circuit.calibrations):
        raise ValueError('BOUND_UNITARY_REGISTER_GRAMMAR')
    _finite(circuit.global_phase)
    if circuit.layout is not None:
        if circuit.layout.final_index_layout(filter_ancillas=True) != list(range(qubits)):
            raise ValueError('IDENTITY_LAYOUT_REQUIRED')
    for item in circuit.data:
        op = item.operation
        controlled_rzz = (not native and op.base_class is ControlledGate
            and op.name == 'crzz' and op.num_ctrl_qubits == 1 and op.ctrl_state == 1
            and op.base_gate.base_class is RZZGate)
        if op.base_class not in classes and not controlled_rzz:
            raise ValueError('UNSUPPORTED_P0_GRAMMAR_GATE:' + op.name)
        if isinstance(op, ControlledGate) and op.ctrl_state != (1 << op.num_ctrl_qubits)-1:
            raise ValueError('CLOSED_CONTROL_REQUIRED')
        if item.clbits or op.condition is not None:
            raise ValueError('NO_CLASSICAL_CONDITION')
        for value in op.params:
            _finite(value)
    return dict(qubits=qubits, gates=len(circuit.data), native=native,
                finite_parameters=True, synthesis_error_bound=None)


def compile_once(circuit, profile, *, call):
    """No exception suppression, retry, fallback, filtering, or re-sampling.

    `call` is the existing SharedQiskitCalls instance shared with this job's
    builders/adjoints. This adapter is opt-in and does not patch old runners.
    """
    from .c1_h4_native_fair import compile_checked
    profile.validate()
    n = circuit.num_qubits
    check_circuit(circuit, qubits=n)
    compiled = call('compile', 'P0 domain-checked single attempt',
                    lambda: compile_checked(circuit, profile))
    check_circuit(compiled, qubits=n, native=True)
    return compiled


def require_complete_sample(outcomes, expected_N):
    """Readiness guard for the existing all-success calibration lane.

    A future failure-aware estimator needs its own adopted scope. Passing a
    shortened list after a failed attempt cannot satisfy this fixed-N gate.
    Repeated word IDs are allowed: IID sampling can draw duplicates.
    """
    rows = tuple(outcomes)
    if type(expected_N) is not int or expected_N < 1 or len(rows) != expected_N:
        raise ValueError('ALL_FIXED_N_ATTEMPTS_REQUIRED')
    if any(row != 'SUCCESS' for row in rows):
        raise ValueError('COMPILER_FAILURE_OR_UNKNOWN_STOP_NO_DISCARD')
    return dict(samples=expected_N, all_q_words_compile_success_qualified=False,
                population_failure_probability=None)


def readiness():
    """Reuse B1 syntax counts; never promote any finite diagnostic to proof."""
    from .c_p0_uniform_method import rewrite_budget
    b = rewrite_budget(100914, 6608)
    return dict(scope=SCOPE, status='ALL_WORD_SUCCESS_NOT_ESTABLISHED',
        all_q_words_compile_success_qualified=False,
        structural_budget=b, monotone_iterations_upper=b['initial_size']+1,
        fixed_controller_iterations=1000,
        missing=['prove fixed-point convergence within unchanged 1000 iterations',
                 'all reachable finite numeric/phase and definition paths',
                 'bound saved input/frame/support identity under current SOURCE',
                 'all native output/register/layout invariants without exceptions'],
        synthesis_error_bound=None, calibration_ready=False,
        maximum_logical_units=None, scientific_calls=0, mandatory_stop=True)


def run_science(*, execute_science=False):
    if not execute_science:
        raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_PERMISSION_REQUIRED')
    raise NotImplementedError('NONEXECUTABLE_UNAPPROVED_METHOD_N_AND_FAILURE_POLICY')


def preflight(root, contract):
    """Technical SOURCE/contract bytes only; no scientific input decoding."""
    import hashlib
    import json
    from pathlib import Path
    root = Path(root).resolve()
    c = json.loads(Path(contract).read_text())
    expected = dict(scope=SCOPE, base_commit=BASE, executable=False,
        maximum_logical_units=None, samples=None, method=None,
        delta_compile=None, delta_cal=None, adopted_failure_policy=None,
        new_H4_builds=0, new_H4_compiles=0, new_H4_samples=0,
        Stage_P_calls=0, retry=0, resume=0, mandatory_stop=True)
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('FIXED_NONEXECUTABLE_TOTALITY_CONTRACT')
    for name, want in c['technical_source_files'].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise ValueError('TECHNICAL_SOURCE:' + name)
    if (root/c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_TECHNICAL_ONLY', scientific_calls=0,
                Stage_P_calls=0, execution_ready=False, arrays_decoded=0,
                all_q_words_compile_success_qualified=False, mandatory_stop=True)
