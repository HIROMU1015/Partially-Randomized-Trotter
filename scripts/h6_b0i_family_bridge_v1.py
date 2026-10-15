"""Independent native B0+I bridge. No chemistry, sampling or compile entry.

An explicit validated future grant is required before any real input loader
or native API. Current unit tests use tiny matrices and a mock native API.
"""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import pickle


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

g = _load('b0i_execution_guard', 'h6_b0i_family_execution_guard_v1.py')
p = _load('b0i_frozen_prototype', 'pre_review_native_b0i_prototype_v1.py')
SCOPE = 'H6_NATIVE_B0I_FULL_FAMILY12_V1'
MATCHED_SCOPE = 'H6_NATIVE_MATCHED_B0_PILOT6_V1'
PILOT = tuple((p,q) for p in (5,10,15) for q in (1,2,4,8) if (p,q) not in ((5,1),(10,2),(15,8)))
NATIVE_MODULE = 'trottertracks.resource_applicability.ax2b_native_df_v5'


def binding_for(grant, prefix, q):
    value = deepcopy(grant.contract['binding'])
    fingerprints = value.pop('b0_fingerprints')
    value['b0_fingerprint'] = fingerprints[f'{prefix}/{q}']
    return value


def original_digest(prepared):
    raw = pickle.dumps(prepared, protocol=5)  # never unpickled, persisted, or used as a scientific cache key.
    if len(raw) > 4*2**20: raise g.Stop('PREPARED_MUTATION_GUARD_SIZE')
    return g.sha(raw)


def request_digest(request):
    return g.d0.digest({k: v for k, v in request.items()
        if k not in ('prepared', 'built', 'request_digest')})


def upper_float(value):
    if value < 0:
        raise g.Stop('NEGATIVE_ERROR_BOUND')
    result = p.finite(float(value))
    if Fraction(result) < value:
        result = p.finite(math.nextafter(result, math.inf))
    return result


def exact_identity(matrix, lam):
    """Exact rational identity of stored binary64 coefficients, not chemistry error."""
    p.fragment_identity(matrix, lam)  # finite, Hermitian, dimension validation.
    diag = [Fraction(float(matrix[i][i].real if hasattr(matrix[i][i], 'real') else matrix[i][i]))
        for i in range(len(matrix))]
    square = sum((x*x for x in diag), Fraction(0))
    for i in range(len(matrix)):
        for j in range(i):
            z = complex(matrix[i][j])
            square += 2*(Fraction(z.real)**2+Fraction(z.imag)**2)
    return Fraction(float(lam))/4*(sum(diag, Fraction(0))**2+square)


def blueprint(prepared, binding, *, prefix, q):
    """Prepared B0 identity, not a compiled/numeric Qiskit fingerprint."""
    return dict(preparation_hash=prepared.preparation_hash, constant=p.finite(prepared.constant_coefficient),
        extracted_tail=p.finite(prepared.extracted_identity_coefficient),
        fragment_indices=list(prepared.deterministic_fragment_indices),
        randomized_fragment_indices=list(prepared.randomized_block_indices),
        block_order=list(prepared.deterministic_block_order), modes=prepared.num_system_qubits,
        prefix=prefix, q=q, T=.8, formula='2nd', binding={k: v for k, v in binding.items()
            if k not in ('b0_fingerprint', 'fragment_order')})


def saved_signal_overlay(record, *, C, C_error, rotation_roundoff, phase_roundoff):
    """Rotate B0/exact-truncated; leave full-H reference unchanged.

    Separate signal/reference empirical error budgets are mandatory. There is
    no implicit reuse of an opaque combined old u as a certified axis bound.
    """
    errors = record['error_components']
    if errors.get('kind') != 'EMPIRICAL' or errors.get('certified') is not False:
        raise g.Stop('EXPLICIT_EMPIRICAL_ERROR_COMPONENTS_REQUIRED')
    def z(value): return p.finite_complex(complex(value['real'], value['imag']))
    corrected = p.rotate_saved_signal(z(record['signals']['corrected']), total_time=.8, tail_identity=C)
    truncated = p.rotate_saved_signal(z(record['signals']['exact_truncated']), total_time=.8, tail_identity=C)
    reference = z(record['reference'])
    magnitude = p.finite(errors['signal_abs_upper'])
    if magnitude < abs(z(record['signals']['corrected'])):
        raise g.Stop('SIGNAL_MAGNITUDE_UPPER_REQUIRED')
    phase_error = upper_float(Fraction(.8)*Fraction(C_error)+Fraction(phase_roundoff))
    ur, ui = (p.finite(errors['B0_signal'][a]) for a in ('real', 'imag'))
    allowance = p.rotated_allowance(ur, ui, angle=-.8*C, signal_abs_upper=magnitude,
        phase_error_upper=phase_error, rotation_roundoff_upper=rotation_roundoff)
    c, s = abs(math.cos(-.8*C)), abs(math.sin(-.8*C))
    extra_error = Fraction(magnitude)*Fraction(min(2.,phase_error))+Fraction(rotation_roundoff)
    # Outward rounding of the algebra using the supplied binary64 trigonometric
    # values. Transcendental/library error remains in the explicit roundoff term.
    allowance['real'] = upper_float(Fraction(c)*Fraction(ur)+Fraction(s)*Fraction(ui)+extra_error)
    allowance['imag'] = upper_float(Fraction(s)*Fraction(ur)+Fraction(c)*Fraction(ui)+extra_error)
    for axis in ('real', 'imag'):
        extra = p.finite(errors['full_H_reference'][axis])
        if extra < 0: raise g.Stop('NEGATIVE_REFERENCE_ERROR')
        allowance[axis] = upper_float(Fraction(allowance[axis])+Fraction(extra))
    def pack(value): return dict(real=value.real, imag=value.imag)
    allowance.update(kind='EMPIRICAL', certified=False, log_B_margin=p.finite(record['allowance']['log_B_margin']),
        policy='ROTATE_B0_SIGNAL_ADD_UNROTATED_FULL_H_REFERENCE_V1',
        scalar_error_scope='STORED_COEFFICIENT_ARITHMETIC_PLUS_DECLARED_INPUT_ERROR',
        phase_error_upper=phase_error, rotation_roundoff_upper=rotation_roundoff,
        error_component_evidence=deepcopy(errors['evidence']))
    if allowance['log_B_margin'] < 0:
        raise g.Stop('NEGATIVE_LOG_ALLOWANCE')
    return dict(signals=dict(corrected=pack(corrected), exact_truncated=pack(truncated)),
        reference=deepcopy(record['reference']),
        error_decomposition=dict(total_signed=pack(corrected-reference)), allowance=allowance,
        log_B=0., normalized_deterministic=True, scientific_adoption=False)


def conservative_saved_error_components(record):
    """Separate unchanged H6 empirical u conservatively, without certification.

    Frozen H6 u = safety * max(floor, rounding, reference_error + max(path)
    + raw_closure). This u dominates each declared empirical contribution.
    Use u for EACH contribution and add them after rotation; this intentionally
    redundant diagnostic does not bound the unknown true numerical error.
    """
    u = record['allowance']
    if u.get('kind') != 'EMPIRICAL' or u.get('certified') is not False:
        raise g.Stop('SAVED_EMPIRICAL_H6_ALLOWANCE_REQUIRED')
    values = {axis:p.finite(u[axis]) for axis in ('real','imag')}
    if min(values.values()) < 0: raise g.Stop('NEGATIVE_ALLOWANCE')
    z = complex(record['signals']['corrected']['real'], record['signals']['corrected']['imag'])
    magnitude = upper_float(Fraction(abs(p.finite_complex(z)))+Fraction(values['real'])+Fraction(values['imag']))
    return dict(kind='EMPIRICAL', certified=False, B0_signal=deepcopy(values),
        full_H_reference=deepcopy(values), signal_abs_upper=magnitude,
        evidence=dict(policy='SAVED_COMBINED_EMPIRICAL_U_UPPER_FOR_EACH_CONTRIBUTION',
            source_rule='h6_matched_accounting_v1.empirical_allowance',
            safety_factor=10, strict_error_certificate=False,
            separate_contributions_are_not_asserted_equal=True))


def prepare(grant, *, prefix, q, binding, load_fragments, load_prepared,
        saved_record, scalar_input_error_upper, rotation_roundoff_upper,
        input_mapping_loader=lambda: {}, guard=lambda: None):
    """Validate grant before callbacks; prepare a tail-only overlay once."""
    if grant is None or not hasattr(grant, 'require'): raise g.Stop('VALIDATED_GRANT_REQUIRED')
    grant.require(SCOPE)
    if (prefix, q) not in PILOT or binding_for(grant, prefix, q) != binding:
        raise g.Stop('PILOT_SOURCE_INPUT_BINDING')
    p.validate_binding(binding)
    g.runtime_gate(grant, grant.contract['runtime_root'])
    seen = getattr(grant, '_h6_case_reservations', set())
    if (prefix, q) in seen: raise g.Stop('TAIL_EXTRACTION_ALREADY_RESERVED')
    seen.add((prefix, q)); grant._h6_case_reservations = seen
    guard()
    paths = input_mapping_loader()
    verified = g.load_inputs(grant, paths)
    if not grant.synthetic:
        expected_id = f'B0_p{prefix}_q{q}_2nd'
        original_records = g.d0.strict_json(verified['signal_summary.json'])['signals']
        matches = [r for r in original_records if r['cell']['id'] == expected_id]
        original_view = {k:v for k,v in saved_record.items() if k != 'error_components'}
        if len(matches) != 1 or matches[0] != original_view:
            raise g.Stop('SAVED_B0_RECORD_INPUT_BINDING')
    effective_record = deepcopy(saved_record)
    if 'error_components' not in effective_record:
        if grant.contract.get('error_component_policy') != 'SAVED_COMBINED_EMPIRICAL_U_UPPER_FOR_EACH_CONTRIBUTION':
            raise g.Stop('EXPLICIT_ERROR_COMPONENT_POLICY_REQUIRED')
        effective_record['error_components'] = conservative_saved_error_components(effective_record)
    fragments = load_fragments(verified); prepared = load_prepared(verified)
    if (prepared.extracted_identity_coefficient != 0 or prepared.randomized_block_indices or
            tuple(prepared.deterministic_fragment_indices) != tuple(range(prefix)) or
            prepared.ld != prefix or
            (not grant.synthetic and prepared.num_system_qubits != 12)):
        raise g.Stop('BASE_IS_NOT_UNOVERLAID_NATIVE_B0')
    before = blueprint(prepared, binding, prefix=prefix, q=q)
    before_bytes = original_digest(prepared)
    if g.d0.digest(before) != binding['b0_fingerprint']:
        raise g.Stop('B0_BLUEPRINT_FINGERPRINT_MISMATCH')
    base = dict(method='B0', binding=binding, retained_fragment_ids=binding['fragment_order'][:prefix],
        scalar=p.finite(prepared.constant_coefficient), scalar_semantics='ORIGINAL_H_CONSTANT_ONLY',
        extracted_tail_identity=0., tail_identity_added=False,
        control=grant.contract['control'], deterministic_blocks=list(prepared.deterministic_block_order))
    if len(fragments) != 19 or any(len(f['G']) != prepared.num_system_qubits for f in fragments):
        raise g.Stop('H6_DF19_MODE_BINDING')
    overlay = p.overlay_b0(base, fragments, prefix=prefix, total_time=.8, binding=binding)
    exact = sum((exact_identity(f['G'], f['lambda']) for f in fragments[prefix:]), Fraction(0))
    C = overlay['extracted_tail_identity']
    arithmetic_error = upper_float(abs(Fraction(C)-exact))
    input_error = p.finite(scalar_input_error_upper)
    if rotation_roundoff_upper is None:
        if grant.contract.get('rotation_roundoff_policy') != 'EMPIRICAL_64_EPS_PHASE_SCALE_FLOOR_1E12':
            raise g.Stop('EXPLICIT_ROTATION_ROUNDOFF_POLICY_REQUIRED')
        import sys
        z = complex(effective_record['signals']['corrected']['real'],effective_record['signals']['corrected']['imag'])
        rounding = max(1e-12,64*sys.float_info.epsilon*(1+abs(.8*C))*max(1.,abs(z)))
    else:
        rounding = p.finite(rotation_roundoff_upper)
    if min(input_error, rounding) < 0: raise g.Stop('NEGATIVE_ERROR_BOUND')
    C_error = upper_float(Fraction(arithmetic_error)+Fraction(input_error))
    # Rounding of original constant+C and -T*the combined scalar affects the
    # physical branch; include it in the scalar/phase uncertainty receipt.
    combined_error = upper_float(abs(Fraction(overlay['scalar'])-
        (Fraction(base['scalar'])+Fraction(C))))
    C_error = upper_float(Fraction(C_error)+Fraction(combined_error))
    native_angle = p.finite(-.8*overlay['scalar'])
    phase_rounding = upper_float(abs(Fraction(native_angle)+Fraction(.8)*Fraction(overlay['scalar'])))
    analytic_angle = p.finite(-.8*C)
    analytic_rounding = upper_float(abs(Fraction(analytic_angle)+Fraction(.8)*Fraction(C)))
    phase_rounding = upper_float(Fraction(phase_rounding)+Fraction(analytic_rounding))
    record_digest = g.d0.digest(saved_record)
    signal = saved_signal_overlay(effective_record, C=C, C_error=C_error,
        rotation_roundoff=rounding, phase_roundoff=phase_rounding)
    if (blueprint(prepared, binding, prefix=prefix, q=q) != before or
            original_digest(prepared) != before_bytes or g.d0.digest(saved_record) != record_digest):
        raise g.Stop('B0_OR_SAVED_RECORD_MUTATED')
    guard()
    g.load_inputs(grant, paths)
    request = dict(namespace='H6_NATIVE_B0I_FAMILY_BRIDGE_V1', binding=deepcopy(binding), prefix=prefix, q=q,
        blueprint=before, prepared=prepared, overlay=overlay, signal=signal,
        prepared_original_bytes_digest=before_bytes,
        scalar_receipt=dict(C=C, exact_stored_scalar_numerator=str(exact.numerator),
            exact_stored_scalar_denominator=str(exact.denominator), arithmetic_error_upper=arithmetic_error,
            declared_input_error_upper=input_error, combined_scalar_roundoff_upper=combined_error,
            native_phase_roundoff_upper=phase_rounding, scalar_error_upper=C_error,
            rotation_roundoff_upper=rounding, rotation_error_is_empirical=True,
            full_Fock_identity=True, new_scientific_certificate=False),
        built=False, matched_B0=dict(status='NOT_ACQUIRED_BY_THIS_BRIDGE', automatic_build=False,
            cost_delta=None, fingerprint_kind='PREPARED_BLUEPRINT_NOT_COMPILED_CIRCUIT'))
    request['request_digest'] = request_digest(request)
    return request


def build(grant, request, native_api, *, guard=lambda: None):
    """Call the unchanged native builder with constant+C exactly once.

    native_api is either the pinned API loaded below or an artificial mock.
    Building or compiling matched B0 is a different authorization scope.
    """
    grant.require(SCOPE)
    if (request['namespace'] != 'H6_NATIVE_B0I_FAMILY_BRIDGE_V1' or request['built'] or
            request['binding'] != binding_for(grant, request['prefix'], request['q']) or
            request_digest(request) != request['request_digest']):
        raise g.Stop('OVERLAY_OR_REQUEST_REUSE')
    prepared = request['prepared']
    if (blueprint(prepared, request['binding'], prefix=request['prefix'], q=request['q']) != request['blueprint'] or
            original_digest(prepared) != request['prepared_original_bytes_digest']):
        raise g.Stop('PREPARED_B0_CHANGED')
    if not grant.synthetic and (getattr(native_api, '__name__', None) != NATIVE_MODULE or
            native_api is not getattr(grant, '_verified_native_provider', None)):
        raise g.Stop('PINNED_NATIVE_PROVIDER_REQUIRED')
    build_key = (request['prefix'], request['q'])
    seen = getattr(grant, '_h6_build_reservations', set())
    if build_key in seen:
        raise g.Stop('OVERLAY_OR_REQUEST_REUSE')
    seen.add(build_key); grant._h6_build_reservations = seen
    request['built'] = True  # reservation remains consumed on failure.
    guard()
    evolution = native_api.build_deterministic_native(deepcopy(prepared.deterministic_blocks),
        num_system_qubits=prepared.num_system_qubits, T=.8, q=request['q'], formula='2nd',
        scalar=request['overlay']['scalar'], control_policy=grant.contract['control'],
        max_instructions=grant.contract['max_instructions'])
    wrappers = {}
    for axis in ('cosine', 'sine'):
        guard()
        wrappers[axis] = native_api.build_native_hadamard_wrapper(evolution, axis=axis,
            include_measurement=True, max_instructions=grant.contract['max_instructions'])
    if (blueprint(prepared, request['binding'], prefix=request['prefix'], q=request['q']) != request['blueprint'] or
            original_digest(prepared) != request['prepared_original_bytes_digest']):
        raise g.Stop('PREPARED_B0_MUTATED_BY_PROVIDER')
    guard()
    return dict(evolution=evolution, wrappers=wrappers, source_binding=deepcopy(request['binding']),
        matched_B0=deepcopy(request['matched_B0']), new_compile_calls=0,
        axes=dict(cosine='real', sine='imag'), scalar_application='NATIVE_FINAL_ANCILLA_P_ONCE')


def native_provider(grant, checkout):
    """Verify the separately fixed scientific checkout before lazy native import."""
    grant.require(SCOPE)
    if grant.synthetic: raise g.Stop('ARTIFICIAL_TEST_MUST_USE_MOCK_PROVIDER')
    import subprocess
    import sys
    checkout = g.home_path(checkout)
    provider = grant.contract['native_provider']
    for name, expected in provider['source_hashes'].items():
        raw = g.read_bytes(checkout/name)
        fixed = subprocess.check_output(['git', '-C', str(checkout), 'show', provider['SOURCE']+':'+name])
        if g.sha(raw) != expected or raw != fixed:
            raise g.Stop('NATIVE_PROVIDER_SOURCE_BINDING')
    for key, existing in tuple(sys.modules.items()):
        if (key.startswith('trottertracks') or key.startswith('trotterlib')) and getattr(existing, '__file__', None):
            if not Path(existing.__file__).resolve().is_relative_to(checkout/'src'):
                raise g.Stop('SCIENTIFIC_PROVIDER_IMPORTED_FROM_DIFFERENT_CHECKOUT')
    # No import at module load or under a draft/unapproved grant.
    sys.path.insert(0, str(checkout/'src'))
    try:
        import importlib
        module = importlib.import_module(NATIVE_MODULE)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != checkout/'src/trottertracks/resource_applicability/ax2b_native_df_v5.py':
        raise g.Stop('NATIVE_PROVIDER_ORIGIN')
    grant._verified_native_provider = module
    return module


def prepare_from_snapshot(grant, *, prefix, q, input_mapping_loader, guard=lambda: None):
    """Ready connection to the fixed H6 snapshot loader and _prepare_discard.

    Called only by a future authorized bounded pilot. No imports or arrays are
    touched until the grant, source/profile and all input bytes have passed.
    """
    if grant is None or not hasattr(grant, 'require'): raise g.Stop('VALIDATED_GRANT_REQUIRED')
    grant.require(SCOPE)
    if grant.synthetic: raise g.Stop('ARTIFICIAL_FIXTURE_USE_MOCK_PREPARE')
    g.runtime_gate(grant, grant.contract['runtime_root'])
    paths = input_mapping_loader(); verified = g.load_inputs(grant, paths)
    required = {'signal_summary.json','h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json'}
    if not required <= verified.keys(): raise g.Stop('H6_SNAPSHOT_SCALAR_RECEIPTS_REQUIRED')
    records = g.d0.strict_json(verified['signal_summary.json'])['signals']
    cid = f'B0_p{prefix}_q{q}_2nd'
    record, = [s for s in records if s['cell']['id']==cid]
    snapshot_receipt = g.d0.strict_json(verified['snapshot_receipt.json'])
    for name in ('hamiltonian','state'):
        if snapshot_receipt['metadata'][name+'_hash'] != grant.contract['binding'][name]:
            raise g.Stop('H6_SNAPSHOT_INTERNAL_IDENTITY')
    provider = native_provider(grant, grant.contract['native_provider']['checkout'])
    import importlib
    loader = importlib.import_module('trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1')
    prep_api = importlib.import_module('trotterlib.pr2_matched_accuracy_m1_execution')
    cache = {}
    def fragments(bound):
        guard()
        ham, _, _, _, _, _ = loader.load_h6_snapshot(g.home_path(paths['h6_input_snapshot.npz'],existing=True),
            snapshot_receipt, g.d0.strict_json(bound['df_receipt.json']))
        cache['ham'] = ham
        return [dict(id=grant.contract['binding']['fragment_order'][i],
            G=tuple(tuple(complex(z) for z in row) for row in matrix), **{'lambda':float(ham.lambdas[i])})
            for i,matrix in enumerate(ham.g_matrices)]
    def prepared(bound):
        guard(); return prep_api._prepare_discard(cache['ham'],prefix)
    error_policy = grant.contract['scalar_error_policy'][f'{prefix}/{q}']
    request = prepare(grant,prefix=prefix,q=q,binding=binding_for(grant,prefix,q),
        load_fragments=fragments,load_prepared=prepared,saved_record=record,
        scalar_input_error_upper=error_policy['stored_model_coefficient_error_upper'],
        rotation_roundoff_upper=error_policy['rotation_roundoff_upper'],
        input_mapping_loader=lambda:paths,guard=guard)
    return request,provider


def matched_delta(new, old):
    """Pure metadata comparison; never create/compile a missing baseline."""
    if old is None:
        return dict(status='MATCHED_B0_NOT_ACQUIRED', deltas=None)
    for field in ('input', 'compiler', 'axis', 'control', 'T', 'prefix', 'q'):
        if new[field] != old[field]: raise g.Stop('MATCHED_B0_BINDING:'+field)
    metrics = ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size')
    if any(type(r[m]) is not int or r[m] < 0 for r in (new, old) for m in metrics):
        raise g.Stop('MATCHED_INTEGER_METRICS')
    return dict(status='MATCHED_METADATA_POINT_DELTA', deltas={m: new[m]-old[m] for m in metrics},
        scientific_generalization=False)
