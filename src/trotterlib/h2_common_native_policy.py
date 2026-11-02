"""H2 fixed-four technical interfaces; stdlib only, no scientific dispatch.

Load this file directly, not trotterlib.__init__. The only executable numerical
path in this preparation SOURCE requires explicitly synthetic diagnostics.
Scientific kernels are described by pinned bindings and exercised by mock ports;
there is no production grant issuer, molecular loader, RNG, or science CLI.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import sys


class Stop(ValueError):
    """Fail closed; callers must not retry science automatically."""


def require(condition, reason):
    if not condition:
        raise Stop(reason)


def number(value, name, *, nonnegative=True):
    require(type(value) in (int, float), name + ':NUMBER_REQUIRED')
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise Stop(name + ':UNREPRESENTABLE') from exc
    require(finite, name + ':NONFINITE')
    require(not nonnegative or value >= 0, name + ':NEGATIVE')
    return value


def integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, name + ':INTEGER_REQUIRED')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def sha(value, name):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), name + ':SHA256_REQUIRED')
    return value


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


@dataclass(frozen=True)
class Candidate:
    id: str
    method: str
    formula: str
    prefix: int
    q: int
    K: int | None
    r_policy: str | None
    r_value: int | None


CANDIDATES = (
    Candidate('B1_p3_q4_2nd', 'FULL', '2nd', 3, 4, None, None, None),
    Candidate('B1_p3_q2_4th', 'FULL', '4th', 3, 2, None, None, None),
    Candidate('H2O32_c7ec590a32a859bc', 'PR', '2nd', 1, 4,
              4, 'ABS_TIME_PER_OUTER', 32),
    Candidate('H2O32_8a2ecdd0294ffb74', 'PR', '4th', 1, 2,
              4, 'ABS_TIME_PER_OUTER', 32),
)
PHYSICAL = dict(system='linear_H2_1Angstrom', basis='STO-3G', rank=3,
                rank_policy='tol_only_1e-8', T=0.8, epsilon=0.0005,
                alpha_axis=0.025, P=0, n_qubits=4, ancilla=4,
                control='symmetric_directional', bias_mode='USUAL_SIGNED')
COMPILER = dict(qiskit='1.3.0', basis_gates=['rz', 'sx', 'x', 'cx'],
                optimization_level=1, seed_transpiler=17, backend=None,
                coupling_map=None, layout=None, routing=None, num_processes=1,
                phase_policy='H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1')
NATIVE_POLICY = 'H2_COMMON_ADJACENT_QR_SIGNED_PIECES_V1_PROPOSED'
ALLOWANCE_POLICY = 'H2_COMMON_ADAPTIVE_A_RULE_V1_PROPOSED'
ACQUISITION_SOURCE = '4ec26921c6872ed6e8667f167074e1e405cf1a3f'
PROVIDER_SOURCE = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
BINDINGS = {
    'prepare': (ACQUISITION_SOURCE, 'scripts/h46_new2_native.py', 'prepare'),
    'gaussian': (ACQUISITION_SOURCE, 'scripts/h46_new2_native.py', 'gaussian_operations'),
    'schedule': (ACQUISITION_SOURCE, 'src/trotterlib/h46_m5_plan.py', 'schedule'),
    'compose': (ACQUISITION_SOURCE, 'scripts/h46_m5_native.py', 'compose'),
    'requests': (ACQUISITION_SOURCE, 'scripts/h46_m5_native.py', 'make_requests'),
    'wrapper': (PROVIDER_SOURCE, 'src/trottertracks/resource_applicability/ax2b_native_df_v5.py', 'build_native_hadamard_wrapper'),
    'compiler_guard': (ACQUISITION_SOURCE, 'scripts/h2_m5_compiler_phase_guard.py', 'transpile'),
    'cost': (PROVIDER_SOURCE, 'src/trotterlib/rte_compiled_cost.py', 'transpile_and_measure_cost'),
}


def fixed_candidate(record):
    require(type(record) is dict, 'CANDIDATE_SCHEMA')
    match = next((c for c in CANDIDATES if c.id == record.get('id')), None)
    require(match is not None and strict_equal(record, asdict(match)), 'FIXED_CANDIDATE_ONLY')
    return match


def validate_draft(plan):
    """Validate a non-executable proposal; this never authorizes a science call."""
    require(type(plan) is dict and plan.get('schema') == 'h2_common_preflight_v1',
            'DRAFT_SCHEMA')
    require(set(plan) == {'schema', 'status', 'science_authorized', 'native_adopted',
        'allowance_adopted', 'physical', 'compiler', 'candidates', 'native_policy',
        'allowance_policy', 'production_dispatch_bound', 'grant', 'ledger', 'marker',
        'retry', 'mandatory_STOP', 'new_call_counts', 'missing_inputs'}, 'DRAFT_FIELDS')
    require(plan.get('status') == 'DRAFT_NOT_EXECUTABLE', 'DRAFT_STATUS')
    require(plan.get('science_authorized') is False, 'SCIENCE_NOT_AUTHORIZED')
    require(plan.get('native_adopted') is False and
            plan.get('allowance_adopted') is False, 'POLICIES_NOT_ADOPTED')
    require(strict_equal(plan.get('physical'), PHYSICAL), 'FIXED_PHYSICAL_CONDITIONS')
    require(strict_equal(plan.get('compiler'), COMPILER), 'FIXED_COMPILER_PROPOSAL')
    require(strict_equal(plan.get('candidates'), [asdict(c) for c in CANDIDATES]),
            'FOUR_FIXED_CANDIDATES_IN_ORDER')
    require(plan.get('native_policy') == NATIVE_POLICY and
            plan.get('allowance_policy') == ALLOWANCE_POLICY, 'PROPOSED_POLICIES')
    require(plan.get('production_dispatch_bound') is False, 'NO_LIVE_DISPATCH')
    require(plan.get('grant') is None and plan.get('ledger') is None and
            plan.get('marker') is None, 'NO_LIVE_GRANT_OR_RESERVATION')
    require(plan.get('retry') is False and plan.get('mandatory_STOP') is True,
            'ONE_SHOT_STOP_NO_RETRY')
    require(plan.get('new_call_counts') is None, 'DO_NOT_GUESS_SCIENCE_COUNTS')
    missing = plan.get('missing_inputs')
    require(type(missing) is list and bool(missing) and
            all(type(x) is str and bool(x) for x in missing) and
            len(set(missing)) == len(missing), 'DRAFT_MISSING_INPUTS_EXPLICIT')
    return dict(status='VALID_TECHNICAL_DRAFT', science_GO=False,
                execution_ready=False, plan_sha256=digest(plan))


def refuse_science(*args, **kwargs):
    raise Stop('SCIENCE_DISPATCH_UNBOUND: explicit policy adoption, exact calls, '
               'sealed runtime/input and a separately approved runner are required')


def deferred_kernel_recipe(candidate_record):
    """Describe exact pinned call boundaries, without importing or calling them.

    Symbols beginning with '$' are unresolved inputs, never executable strings.
    This recipe is not an executable scientific call budget or runtime context.
    """
    candidate = fixed_candidate(candidate_record)
    pr = candidate.method == 'PR'
    group = dict(method=candidate.method, formula=candidate.formula,
                 prefix=candidate.prefix, q=candidate.q,
                 policy=candidate.r_policy, r_value=candidate.r_value)
    return dict(candidate=asdict(candidate), group=group, native_policy=NATIVE_POLICY,
        executable=False, scientific_calls_performed=0,
        context='NEW_SEPARATELY_SEALED_CONTEXT_NOT_H2_OUTER32_PREFIX1_CONTEXT',
        prepare=dict(binding=BINDINGS['prepare'], ham='$saved_hamiltonian',
                     prefix=candidate.prefix, gaussian_binding=BINDINGS['gaussian']),
        schedule=dict(binding=BINDINGS['schedule'], formula=candidate.formula,
                      q=candidate.q, policy=candidate.r_policy or 'PER_PIECE',
                      r_value=candidate.r_value or 1,
                      preserve_SOURCE_float_operation_order=True,
                      FULL_rte_field='UNUSED_SOURCE_PLACEHOLDER' if not pr else None),
        requests=dict(binding=BINDINGS['requests'] if pr else None,
                      value='$accepted_saved_events_or_separately_approved_requests' if pr else [],
                      sampling_authorized=False, conditional_sampling=False,
                      K=4 if pr else None, ancilla=4),
        compose=dict(binding=BINDINGS['compose'], group=group,
                     preparation='$common_preparation',
                     requests='$validated_requests' if pr else [],
                     scalar='$ham.constant + tail_identity' if pr else '$ham.constant',
                     scalar_per_piece=0.0, scalar_application='ONCE_AFTER_ORDERED_PIECES',
                     api='$pinned_native_provider', make_circuit='$QuantumCircuit',
                     fingerprint='$pinned_fingerprint'),
        wrapper=dict(binding=BINDINGS['wrapper'], axes=['cosine', 'sine'],
                     include_measurement=True, state_preparation_cost=0),
        compile=dict(binding=BINDINGS['cost'], guard=BINDINGS['compiler_guard'],
                     settings=copy.deepcopy(COMPILER), apply_guard_to_method=candidate.method,
                     private_process_installation_required=True))


@dataclass(frozen=True)
class SyntheticAuthority:
    domain: str = 'SYNTHETIC_ONLY'

    def check(self, evidence):
        require(self.domain == 'SYNTHETIC_ONLY' and
                evidence.get('domain') == 'SYNTHETIC_ONLY',
                'SAVED_SCIENCE_VALUES_MUST_NOT_BE_EVALUATED')


def synthetic_gate(authority, evidence):
    require(type(authority) is SyntheticAuthority, 'NO_EXECUTION_AUTHORITY')
    authority.check(evidence)


def check_schedule(candidate, pieces):
    """Validate supplied schedule metadata; do not evaluate PF coefficients."""
    count = 1 if candidate.formula == '2nd' else 3
    require(type(pieces) in (list, tuple) and len(pieces) == candidate.q * count,
            'SIGNED_SCHEDULE_SIZE')
    for outer in range(candidate.q):
        segment = pieces[outer*count:(outer+1)*count]
        for index, piece in enumerate(segment):
            require(set(piece) == {'outer', 'piece', 'signed_time', 'rte_steps'},
                    'SIGNED_PIECE_SCHEMA')
            require(type(piece['outer']) is int and piece['outer'] == outer and
                    type(piece['piece']) is int and piece['piece'] == index,
                    'SIGNED_PIECE_ORDER')
            t = number(piece['signed_time'], 'signed_time', nonnegative=False)
            require(t < 0 if count == 3 and index == 1 else t > 0,
                    'SIGNED_PIECE_SIGN')
            integer(piece['rte_steps'], 'rte_steps')
            if candidate.method == 'FULL':
                require(piece['rte_steps'] == 0, 'FULL_HAS_NO_RTE')
            else:
                require(piece['rte_steps'] > 0, 'PR_PIECE_HAS_RTE')
        if candidate.method == 'PR':
            require(sum(p['rte_steps'] for p in segment) == 32, 'OUTER32_TOTAL')
    return True


def validate_preparation(candidate, receipt):
    require(receipt.get('native_policy') == NATIVE_POLICY, 'COMMON_GAUSSIAN_POLICY')
    require(type(receipt.get('prefix')) is int and receipt['prefix'] == candidate.prefix and
            strict_equal(receipt.get('fragment_order'), list(range(candidate.prefix))),
            'SAVED_GENERATION_FRAGMENT_ORDER')
    require(type(receipt.get('n_qubits')) is int and receipt['n_qubits'] == 4 and
            type(receipt.get('ancilla')) is int and receipt['ancilla'] == 4,
            'SYSTEM_AND_ANCILLA_WIRES')
    for key in ['preparation_sha256', 'basis_sha256', 'operations_sha256']:
        sha(receipt.get(key), key)
    require(receipt.get('basis_order') == 'SAVED_JW_PROVIDER_ORDER', 'BASIS_ORDER')
    require(receipt.get('scalar_policy') == 'ORIGINAL_PLUS_PR_TAIL_ONCE',
            'SCALAR_POLICY')
    return True


class MockNativePort:
    """Pure operation log standing in for the pinned scientific functions.

    No Hamiltonian, matrix, quantum gate, RNG or callable is accepted. Production
    bindings have the same conceptual boundaries but need a separately sealed
    science runner; passing a mocked receipt is never a production authorization.
    """
    def __init__(self):
        self.events = []

    def record(self, operation, **metadata):
        self.events.append(dict(operation=operation, **copy.deepcopy(metadata)))


class CommonNativeAdapter:
    """Implementation of common preparation/composition ordering on mock ports."""
    def __init__(self, port, authority=None):
        require(type(port) is MockNativePort, 'MOCK_PORT_ONLY_IN_PREPARATION_SOURCE')
        self.port, self.authority = port, authority

    def build(self, candidate_record, evidence, preparation, pieces, requests):
        synthetic_gate(self.authority, evidence)  # before touching a kernel/input
        candidate = fixed_candidate(candidate_record)
        validate_preparation(candidate, preparation)
        check_schedule(candidate, pieces)
        scalar = number(evidence.get('scalar'), 'scalar', nonnegative=False)
        if candidate.method == 'FULL':
            require(requests == [], 'FULL_NO_RANDOM_REQUESTS')
        else:
            require(len(requests) == len(pieces), 'ONE_REQUEST_PER_PIECE')
            for request, piece in zip(requests, pieces):
                require(strict_equal(request, dict(piece=piece, K=4, controlled=True,
                        ancilla=4, preparation_sha256=preparation['preparation_sha256'],
                        conditioning='UNCONDITIONAL_SAVED_LAW')), 'REQUEST_IDENTITY')
        # All metadata is checked before any partial output is emitted.
        self.port.record('prepare', prefix=candidate.prefix, policy=NATIVE_POLICY,
                         binding=BINDINGS['prepare'])
        for piece in pieces:
            self.port.record('signed_s2_piece', piece=piece, scalar=0.0,
                             prefix=candidate.prefix, control='symmetric_directional')
        self.port.record('scalar_once', scalar=scalar, T=0.8, ancilla=4)
        for axis in ('cosine', 'sine'):
            self.port.record('full_hadamard_wrapper', axis=axis, ancilla=4,
                             include_measurement=True, state_preparation=False)
            self.port.record('guarded_compile', axis=axis, settings=COMPILER,
                             phase_alignment=False, state_fit=False)
        return dict(candidate=candidate_record, native_policy=NATIVE_POLICY,
                    schedule_sha256=digest(pieces), preparation=preparation,
                    compiled_identity_attested=False, synthetic_only=True)


@dataclass(frozen=True)
class Observation:
    value: int | float | None
    unit: str
    scope: str
    evidence: str

    def validate(self, unit, scope):
        require(self.value is not None, 'MISSING:' + scope)
        require(self.unit == unit and self.scope == scope, 'DIAGNOSTIC_DEFINITION:' + scope)
        require(type(self.evidence) is str and bool(self.evidence), 'MISSING_PROVENANCE')
        return number(self.value, scope)


@dataclass(frozen=True)
class AdaptiveInputs:
    candidate_id: str
    domain: str
    reference_id: str | None
    reference_error: Observation
    path_signal_errors: tuple[Observation, ...]
    path_vector_errors: tuple[Observation, ...]
    corrected_signal_closure: Observation
    log_B_discrepancy: Observation
    work_primary: dict | None
    work_independent: dict | None
    maximum_norm: Observation
    norm_coverage: str | None
    trace_evidence: str | None
    native_policy: str = 'UNJOINED_SAVED_NATIVE'
    input_binding_sha256: str | None = None

    def missing(self):
        names = []
        for key in ['reference_error', 'corrected_signal_closure',
                    'log_B_discrepancy', 'maximum_norm']:
            if getattr(self, key).value is None:
                names.append(key)
        for key in ['reference_id', 'path_signal_errors', 'path_vector_errors',
                    'work_primary', 'work_independent', 'norm_coverage', 'trace_evidence']:
            if not getattr(self, key):
                names.append(key)
        if self.native_policy != NATIVE_POLICY:
            names.append('common_native_diagnostic_binding')
        if self.input_binding_sha256 is None:
            names.append('input_binding_sha256')
        return names

    def validate(self, common_reference_id):
        require(self.candidate_id in {c.id for c in CANDIDATES}, 'FIXED_CANDIDATE_ONLY')
        require(not self.missing(), 'MISSING_COMMON_DIAGNOSTICS:' + ','.join(self.missing()))
        require(self.reference_id == common_reference_id and bool(common_reference_id),
                'SHARED_REFERENCE_NOT_BOUND')
        sha(self.input_binding_sha256, 'input_binding_sha256')
        self.reference_error.validate('CORRECTED_SIGNAL', 'H2_REFERENCE_COMBINED_DIAGNOSTIC')
        for x in self.path_signal_errors:
            x.validate('CORRECTED_SIGNAL', 'PRIMARY_VS_INDEPENDENT_SIGNAL')
        for x in self.path_vector_errors:
            x.validate('CORRECTED_STATE_VECTOR', 'PRIMARY_VS_INDEPENDENT_VECTOR')
        self.corrected_signal_closure.validate('CORRECTED_SIGNAL', 'B_RAW_MINUS_CORRECTED_SIGNAL')
        self.log_B_discrepancy.validate('LOG_NORMALIZATION', 'TWO_PATH_LOG_B_DIFFERENCE')
        self.maximum_norm.validate('CORRECTED_STATE_VECTOR', 'BOTH_PATHS_ALL_REQUIRED_STAGES')
        require(self.norm_coverage == 'BOTH_PATHS_ALL_REQUIRED_STAGES', 'INCOMPLETE_STAGE_NORMS')
        require(type(self.trace_evidence) is str and bool(self.trace_evidence), 'NORM_TRACE_REQUIRED')
        for counts in (self.work_primary, self.work_independent):
            require(type(counts) is dict and set(counts) == {'deterministic', 'tail'}, 'WORK_SCOPE')
            for key, value in counts.items():
                integer(value, 'work.' + key)
        return True


def observation(value, unit, scope, evidence):
    return Observation(value, unit, scope, evidence)


def map_old_diagnostics(record, reference, evidence):
    """Copy original diagnostics only. No new u, closure, N, G or bias arithmetic."""
    cid = record['cell']['id']
    require(cid in {c.id for c in CANDIDATES if c.method == 'FULL'}, 'OLD_FULL_CANDIDATE')
    make = lambda value, unit, scope: observation(value, unit, scope, evidence)
    return AdaptiveInputs(cid, 'SAVED_SCIENCE', 'OLD_FP_REFERENCE',
        make(reference.get('error'), 'CORRECTED_SIGNAL', 'H2_REFERENCE_COMBINED_DIAGNOSTIC'),
        tuple(make(v, 'CORRECTED_SIGNAL', 'PRIMARY_VS_INDEPENDENT_SIGNAL')
              for v in record.get('signal_discrepancies', [])),
        tuple(make(v, 'CORRECTED_STATE_VECTOR', 'PRIMARY_VS_INDEPENDENT_VECTOR')
              for v in record.get('state_discrepancies', [])),
        make(record.get('B_raw_corrected_difference'), 'CORRECTED_SIGNAL', 'B_RAW_MINUS_CORRECTED_SIGNAL'),
        make(record.get('normalization_log_discrepancy'), 'LOG_NORMALIZATION', 'TWO_PATH_LOG_B_DIFFERENCE'),
        copy.deepcopy(record.get('action_counts')), copy.deepcopy(record.get('oracle_action_counts')),
        make(record.get('intermediate_norm_max'), 'CORRECTED_STATE_VECTOR', 'BOTH_PATHS_ALL_REQUIRED_STAGES'),
        'BOTH_PATHS_ALL_REQUIRED_STAGES', evidence + ':frozen_correctness_source_scope')


def map_pr_diagnostics(projection, evidence):
    """Copy PR fields; explicitly leave A-rule quantities absent when unsaved."""
    record = projection['saved']
    cid = record['candidate']['id']
    require(cid in {c.id for c in CANDIDATES if c.method == 'PR'}, 'PR_FIXED_CANDIDATE')
    make = lambda value, unit, scope: observation(value, unit, scope, evidence)
    return AdaptiveInputs(cid, 'SAVED_SCIENCE', 'OUTER32_REFERENCE_UNJOINED',
        make(None, 'CORRECTED_SIGNAL', 'H2_REFERENCE_COMBINED_DIAGNOSTIC'),
        tuple(make(v, 'CORRECTED_SIGNAL', 'PRIMARY_VS_INDEPENDENT_SIGNAL')
              for v in record.get('independent_forward_abs_errors', [])), (),
        make(None, 'CORRECTED_SIGNAL', 'B_RAW_MINUS_CORRECTED_SIGNAL'),
        make(record.get('normalization_error'), 'LOG_NORMALIZATION', 'TWO_PATH_LOG_B_DIFFERENCE'),
        None, None,
        make(None, 'CORRECTED_STATE_VECTOR', 'BOTH_PATHS_ALL_REQUIRED_STAGES'), None, None)


def evaluate_synthetic_allowance(inputs, authority=None, *, common_reference_id):
    """Executable proposal on artificial values only; saved science fails first.

    Arithmetic order follows the old empirical_allowance SOURCE. Parameters are
    proposed, not adopted. This is never an upper-bound certificate or shot rule.
    """
    synthetic_gate(authority, {'domain': inputs.domain})
    inputs.validate(common_reference_id)
    try:
        work = sum(inputs.work_primary.values()) + sum(inputs.work_independent.values())
        maximum = inputs.maximum_norm.value
        rounding = 64 * sys.float_info.epsilon * (1 + work) * max(1., maximum)
        paths = [x.value for x in inputs.path_signal_errors + inputs.path_vector_errors]
        agreement = inputs.reference_error.value + max(paths) + inputs.corrected_signal_closure.value
        u = 10 * max(1e-12, rounding, agreement)
        margin = 10 * max(1e-12, rounding, inputs.log_B_discrepancy.value)
    except OverflowError as exc:
        raise Stop('ALLOWANCE_ARITHMETIC_OVERFLOW') from exc
    number(u, 'u'); number(margin, 'log_B_margin')
    return dict(policy=ALLOWANCE_POLICY, u_real=u, u_imag=u, log_B_margin=margin,
                rounding_proxy=rounding, agreement=agreement, certified=False,
                formally_adopted=False, synthetic_only=True)


def check_cost_reuse(saved, proposed):
    """Identity-only decision; never infer C from old counts or recompute cost."""
    keys = ('candidate_id', 'hamiltonian_sha256', 'state_sha256', 'native_policy',
            'preparation_sha256', 'schedule_sha256', 'wrapper_sha256',
            'compiler_source_sha256', 'compiler_settings_sha256',
            'runtime_closure_sha256', 'event_law_sha256', 'cohort_id')
    mismatches = [k for k in keys if saved.get(k) is None or
                  proposed.get(k) is None or saved.get(k) != proposed.get(k)]
    for key in keys:
        if key.endswith('_sha256'):
            try:
                sha(saved.get(key), key); sha(proposed.get(key), key)
            except Stop:
                if key not in mismatches:
                    mismatches.append(key)
        elif not saved.get(key) or not proposed.get(key):
            if key not in mismatches:
                mismatches.append(key)
    accepted = saved.get('compiled_correctness_accepted') is True
    return dict(reusable=not mismatches and accepted,
                missing_or_different=mismatches,
                compiled_correctness_accepted=accepted,
                action='COPY_SAVED_C_ONLY' if not mismatches and accepted else
                       'RETRIEVE_EVIDENCE_OR_PLAN_SEPARATELY_APPROVED_VALIDATION',
                automatic_recompile=False)


IDENTITY_FIELDS = ('implementation_sha256', 'hamiltonian_sha256', 'state_sha256',
    'preparation_sha256', 'basis_sha256', 'operations_sha256', 'schedule_sha256',
    'wrapper_sha256', 'compiler_source_sha256', 'compiler_settings_sha256',
    'runtime_closure_sha256', 'event_law_sha256')
GATE_LIMITS = dict(native_low=1e-9, native_leakage=1e-9,
    independent_occupation_vector=1e-9, hadamard_cosine=1e-9,
    hadamard_sine=1e-9, compiled_vector_cosine=1e-7, compiled_vector_sine=1e-7)


def validate_synthetic_result(record, authority=None):
    """Exercise proposed provenance/correctness schema on mock receipts only.

    A well-formed receipt does not prove a measured gate or authorize execution.
    Real acceptance must be implemented by the future approved, sealed runner.
    """
    synthetic_gate(authority, record)
    require(set(record) == {'schema', 'domain', 'candidate', 'native_policy',
        'identity', 'checks', 'phase_alignment', 'state_fit', 'retry',
        'mandatory_STOP'}, 'RESULT_FIELDS')
    require(record['schema'] == 'h2_common_result_proposal_v1', 'RESULT_SCHEMA')
    fixed_candidate(record['candidate'])
    require(record['native_policy'] == NATIVE_POLICY, 'RESULT_NATIVE_POLICY')
    require(type(record['identity']) is dict and
            set(record['identity']) == set(IDENTITY_FIELDS), 'RESULT_IDENTITY_FIELDS')
    for key in IDENTITY_FIELDS:
        sha(record['identity'][key], key)
    require(record['phase_alignment'] is False and record['state_fit'] is False,
            'NO_PHASE_ALIGNMENT_OR_STATE_FIT')
    require(record['retry'] is False and record['mandatory_STOP'] is True,
            'RESULT_STOP_NO_RETRY')
    require(type(record['checks']) is dict and set(record['checks']) == set(GATE_LIMITS),
            'ALL_RESULT_GATES_REQUIRED')
    for name, limit in GATE_LIMITS.items():
        check = record['checks'][name]
        require(type(check) is dict and set(check) == {'error', 'evidence_sha256'},
                'RESULT_GATE_FIELDS')
        sha(check['evidence_sha256'], 'gate_evidence')
        require(number(check['error'], name) < limit, 'CORRECTNESS_GATE:' + name)
    return dict(status='SYNTHETIC_SCHEMA_PASS', science_GO=False,
                real_correctness_accepted=False, all_state_certificate=False,
                receipt_sha256=digest(record))
