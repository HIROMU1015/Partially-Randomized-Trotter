"""Arb192 fixed-input round signals; no sampler, compiler or molecular runner.

All numerical entry points are for an independently authorized caller. This
readiness study calls them only with invented inputs. Bindings detect mismatches,
not dishonest producers. Existing scientific contracts are never changed.
"""
from copy import deepcopy
from fractions import Fraction as F

from . import c_fresh_iid_calibration as fresh
from . import c_iid_calibration_integration as integration
from . import c_iid_grid_expected_bias as expected
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z

VERSION = 'Arb192_fixed_input_paired_round_signal_v1'
QUALIFICATION = 'fixed_input_round_signal_enclosure_not_full_operator'
BASE = '325f07ae02d8872d4fe322d6ad6e08c256e0bef7'
FAMILY = 'artifacts/research_c_direct_round_signal/2026-11-05'
CONTRACT = 'docs/research/research_c_direct_round_signal_contract_unapproved_20261105.json'


class CompilerWordError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__('FAILED_OR_UNFINISHED_COMPILER_WORD_CANNOT_BE_DROPPED')


def bind_state(amplitudes, *, source_id, target_id, uncertainty='0', provenance):
    """Exact rational centre; uncertainty is vector2 distance to a unit state.

    Nonzero uncertainty requires an independently qualified premise. A saved
    approximate eigenvector/residual is not itself such a premise. No state is
    silently normalized and preparation cost has no bearing on this error.
    """
    fresh._hex_id(source_id); fresh._hex_id(target_id)
    q = [[expected.rational(x) for x in pair] for pair in amplitudes]
    if not q or len(q) & (len(q)-1) or any(len(x) != 2 for x in q) or not provenance:
        raise ValueError('EXPLICIT_POWER_OF_TWO_STATE_AND_PROVENANCE')
    eta = expected.nonnegative(uncertainty)
    norm2 = sum((a*a+b*b for a,b in q), F())
    if not eta and norm2 != 1:
        raise ValueError('EXACT_UNIT_STATE_OR_QUALIFIED_STATE_ERROR_REQUIRED')
    if norm2 < max(F(), 1-eta)**2 or norm2 > (1+eta)**2:
        raise ValueError('STATE_DISTANCE_PREMISE_INCONSISTENT_WITH_NORM')
    body = dict(amplitudes=[[str(x) for x in pair] for pair in q],
        source_id=source_id, target_id=target_id, uncertainty_exact=str(eta),
        centre_norm_squared_exact=str(norm2), provenance=provenance,
        qualification='exact_normalized_state' if not eta else 'conditional_on_supplied_vector2_state_error')
    return dict(body, state_id=fresh.digest(body))


def _state(record, source_id, target_id):
    want = bind_state(record['amplitudes'], source_id=source_id, target_id=target_id,
        uncertainty=record['uncertainty_exact'], provenance=record['provenance'])
    if record != want:
        raise ValueError('INITIAL_STATE_SOURCE_TARGET_IDENTITY')
    from flint import acb, acb_mat
    # Ancilla is the highest bit, initially zero; system q0 is least significant.
    psi = [acb(z._real(F(a)), z._real(F(b))) for a,b in record['amplitudes']]
    v, rounding = fixed._midpoint_vector(acb_mat([[x] for x in psi]+[[0] for _ in psi]))
    return v, F(record['uncertainty_exact'])+rounding


def native_action(record, state, radius=F(), *, cache=None):
    """Extend existing local-gate/recenter arithmetic to an arbitrary input.

    Each declared gate is exactly unitary at its stored binary64 angle.
    ||v-c|| <= E implies E' <= E + delta after outward local action/recenter.
    Context/lock are owned by the caller, as in the existing column primitive.
    """
    from flint import acb, acb_mat
    from .c_arb_column_batch import layout
    fixed._validate(record)
    if state.nrows() != 1 << record['qubits'] or state.ncols() != 1:
        raise ValueError('STATE_REGISTER_DIMENSION')
    radius = expected.nonnegative(radius); cache = {} if cache is None else cache
    v = state
    for gate in record['gates']:
        key = (192, gate['name'], tuple(gate['params_hex']))
        if key not in cache:
            cache[key] = fixed._local(key[1], [float.fromhex(x) for x in key[2]])
        ix = layout(record['qubits'], tuple(gate['qubits']))
        packed = acb_mat([[v[j,0] for j in row] for row in ix])
        product = cache[key]*packed
        out = acb_mat(v.nrows(),1)
        for i,row in enumerate(ix):
            for j,k in enumerate(row): out[k,0] = product[i,j]
        v, delta = fixed._midpoint_vector(out); radius += delta
    phase = (acb(0,1)*z._real(float.fromhex(record['phase_hex']))).exp()
    v, delta = fixed._midpoint_vector(z._scale(v,phase))
    return v, radius+delta


def _literal(reference, dimension):
    """Unpack existing CAR block enclosures without assuming unitarity.

    min(||B||F, sqrt(1+||B*B-I||F)) is an upper bound for each
    block's norm; a below is the maximum of these bounds and 1. The second
    expression avoids dimension-dependent amplification for near-unitaries.
    No polar factor, normalization, cap2, or unproved norm1 is used.
    """
    from flint import acb_mat
    from .c_pr_aware_df_fixed_car import unpack
    if reference['system_dimension'] != dimension:
        raise ValueError('LITERAL_SYSTEM_REGISTER')
    seen = []; blocks = []; a = F(1)
    for row in reference['blocks']:
        ix = row['indices']
        if not ix or any(type(i) is not int or not 0 <= i < dimension for i in ix):
            raise ValueError('LITERAL_BLOCK_INDICES')
        seen.extend(ix); B = unpack(row['entries'],len(ix))
        eye = acb_mat([[int(i == j) for j in range(len(ix))] for i in range(len(ix))])
        defect = F(z._norm(B.conjugate().transpose()*B-eye)['upper_exact'])
        norm = min(F(z._norm(B)['upper_exact']), expected.sqrt_upper(1+defect))
        a = max(a,norm); blocks.append((ix,B))
    if sorted(seen) != list(range(dimension)):
        raise ValueError('COMPLETE_DISJOINT_LITERAL_PARTITION')
    return blocks,a


def literal_action(prepared, state, radius=F()):
    from flint import acb_mat
    blocks,a = prepared; d = state.nrows()//2
    out = acb_mat([[state[i,0]] for i in range(2*d)])
    for ix,B in blocks:
        product = B*acb_mat([[state[d+i,0]] for i in ix])
        for j,i in enumerate(ix): out[d+i,0] = product[j,0]
    v,delta = fixed._midpoint_vector(out)
    # Exact rational a**n has an unnecessarily huge denominator for n=4096.
    # Re-enclose the radius outward at the SAME Arb192 precision each step;
    # this bounds representation size without weakening precision or norm.
    error = z._fraction(z._real(a*expected.nonnegative(radius)+delta).upper())
    return v,error


def interval(value, radius=F()):
    lo,hi = z._fraction(value.lower()), z._fraction(value.upper())
    return [str(lo-radius),str(hi+radius)]


def inner_product(left, right):
    from flint import acb
    if left.nrows() != right.nrows(): raise ValueError('INNER_PRODUCT_DIMENSION')
    return sum((left[i,0].conjugate()*right[i,0] for i in range(left.nrows())),acb(0))


def observable_z(state, radius):
    """Z expectation and norm, enclosing arithmetic and vector2 uncertainty."""
    from flint import arb
    d = state.nrows()//2
    value = sum(((state[i,0].real**2+state[i,0].imag**2)*(1 if i<d else -1)
                 for i in range(2*d)),arb(0))
    norm2 = inner_product(state,state).real
    norm = F(z._norm(state)['upper_exact'])
    pad = 2*norm*radius+radius*radius
    return dict(signal=interval(value,pad), norm_squared=interval(norm2,pad),
        state_vector2_radius_exact=str(radius), centre_norm_upper_exact=str(norm))


def bind_inverse(record, compiled, sample, context, occurrence):
    fixed._validate(record)
    return dict(record=deepcopy(record), record_id=fresh.digest(record),
        forward_id=compiled['record_id'],
        calibration_binding=integration.calibration_binding(sample,context,occurrence))


def _binding(sample, occurrence):
    """After ONE full fresh.validate_sample; avoid quadratic round replay."""
    return dict(sample_id=sample['sample_id'],source_id=sample['source_id'],
        candidate=sample['candidate'],law_id=sample['law_id'],occurrence=occurrence,
        word_id=sample['kernels'][occurrence]['word_id'])


def _operator(value, sample, context, occurrence, role, raw_id=None):
    """Same integration.bind_operator schema, with round validation hoisted.

    The existing per-word adapter replays the entire sample at each binding.
    This small assembly is the only duplicated metadata operation, necessary
    to keep a length-n direct round O(n) in sample validation rather than O(n²).
    """
    record=value['record'];fixed._validate(record)
    want=dict(record=record,record_id=fresh.digest(record),role=role,raw_id=raw_id,
        phase_hex=record['phase_hex'],phase_policy=context['identity']['phase_policy'],
        identity=context['identity'],context_id=context['context_id'],unit=sample['unit'],
        m=sample['m'],axis=sample['axis'],word=sample['kernels'][occurrence]['word'],
        calibration_binding=_binding(sample,occurrence))
    if value != want: raise ValueError('OPERATOR_SOURCE_WORD_ANGLE_PHASE_OR_COMPILER_IDENTITY')
    if role=='compiled' and any(g['name'] not in ('rz','sx','x','cx') for g in record['gates']):
        raise ValueError('P0_NATIVE_GRAMMAR')
    return record


def ideal_shells(qubits):
    out = {}
    for label in ('head','X','Y'):
        record = fixed.identity(qubits)
        if label == 'Y': record['gates'].append(dict(name='sdg',qubits=[qubits-1],params_hex=[]))
        record['gates'].append(dict(name='h',qubits=[qubits-1],params_hex=[]))
        out[label] = record
    return out


def evaluate_round(*, sample, context, target, state, occurrences, shells, partial=None):
    """Same realised word list for both axes; full outputs of one bound state.

    The forward Hadamard wrapper contains no extra inverse pass. Actual
    inverses are independent fixed-state diagnostics against the adjoint of
    each SAME forward record, not an inverse of nonunitary literal D.
    """
    from flint import ctx,acb_mat
    fresh.validate_sample(sample,context)
    if sample['unit'] != 'round' or sample['axis'] != 'paired_XY':
        raise ValueError('COMPLETE_PAIRED_ROUND_NOT_KERNEL')
    if len(occurrences) != len(sample['kernels']): raise ValueError('ALL_COHERENT_OCCURRENCES')
    if set(shells) != {'head','X','Y'}: raise ValueError('HEAD_X_Y_SHELL_COVERAGE')
    z.runtime_model()
    partial = {} if partial is None else partial
    partial.update(occurrences=[], axes={}, input_occurrences=deepcopy(occurrences))
    with z._BALL_LOCK,ctx.workprec(192):
        v,e = _state(state,context['source_id'],fresh.digest(target))
        n = v.nrows().bit_length()-1; d = v.nrows()//2
        for record in shells.values():
            fixed._validate(record)
            if record['qubits'] != n or any(g['qubits'] != [n-1] for g in record['gates']):
                raise ValueError('ANCILLA_LAST_ONLY_SHELL')
        ideal = ideal_shells(n); cache = {}; ref_cache = {}
        c,ec = native_action(shells['head'],v,e,cache=cache)
        l,el = native_action(ideal['head'],v,e,cache=cache)
        for i,part in enumerate(occurrences):
            if part.get('compile_status') != 'SUCCESS':
                raise CompilerWordError(part.get('compile_status'))
            R = _operator(part['raw'],sample,context,i,'raw')
            C = _operator(part['compiled'],sample,context,i,'compiled',part['raw']['record_id'])
            integration._reference(part['reference'],sample,context,i,target)
            if R['qubits'] != n or C['qubits'] != n: raise ValueError('CONTROLLED_REGISTER')
            inv = part['actual_inverse']
            fixed._validate(inv['record'])
            want_inverse=dict(record=inv['record'],record_id=fresh.digest(inv['record']),
                forward_id=part['compiled']['record_id'],calibration_binding=_binding(sample,i))
            if inv != want_inverse or inv['record']['qubits'] != n:
                raise ValueError('ACTUAL_INVERSE_SAME_FORWARD_TRAJECTORY')
            c,ec = native_action(C,c,ec,cache=cache)
            # A state witness, with explicit scope: never promoted to an operator certificate.
            u,eu = native_action(inv['record'],c,ec,cache=cache)
            w,ew = native_action(fixed.adjoint(C),c,ec,cache=cache)
            inverse_upper = F(z._norm(u-w)['upper_exact'])+eu+ew
            key = fresh.digest(part['reference'])
            if key not in ref_cache: ref_cache[key] = _literal(part['reference'],d)
            l,el = literal_action(ref_cache[key],l,el)
            partial['occurrences'].append(dict(occurrence=i,
                binding=_binding(sample,i),
                raw_id=part['raw']['record_id'],compiled_id=part['compiled']['record_id'],
                reference_id=key,actual_inverse_id=inv['record_id'],
                literal_norm_upper_exact=str(ref_cache[key][1]),
                actual_inverse_vs_adjoint_state_upper_exact=str(inverse_upper),
                inverse_qualification='propagated_input_state_only_not_operator_or_literal_inverse'))
        axes = {}; raw = {k:{} for k in ('compiled','literal_atom')}
        for axis,coordinate in (('X','real'),('Y','imag')):
            cv,ce = native_action(shells[axis],c,ec,cache=cache)
            lv,le = native_action(ideal[axis],l,el,cache=cache)
            a,b = observable_z(cv,ce),observable_z(lv,le)
            partial['axes'][axis] = dict(compiled=a,literal_atom=b)
            raw['compiled'][coordinate] = a['signal']; raw['literal_atom'][coordinate] = b['signal']
            axes[axis] = [str(F(a['signal'][0])-F(b['signal'][1])),
                          str(F(a['signal'][1])-F(b['signal'][0]))]
    return dict(status='VALIDATED_FIXED_INPUT_ROUND_SIGNAL',kind='rigorous_enclosure',
        qualification=QUALIFICATION,scope='paired_round_raw_complex_signal',
        m=sample['m'],occurrences=len(occurrences),round_word_id=sample['round_word_id'],
        system_dimension=d,controlled_dimension=2*d,input_states_evaluated=1,
        full_output_amplitudes_evaluated=True,
        sample_id=sample['sample_id'],source_id=sample['source_id'],law_id=sample['law_id'],
        target_id=fresh.digest(target),state_id=state['state_id'],state_qualification=state['qualification'],
        X=axes['X'],Y=axes['Y'],raw_signals=raw,raw_signal_schema='complex_rectangle_exact_rational_endpoints_v1',
        precision_bits=192,rounding_enclosed=True,phase_fit=False,shells_included=True,
        shell_ids={k:fresh.digest(r) for k,r in shells.items()},
        included_error_terms=[expected.TERMS[0]],
        provenance='Arb192 local action/recenter;literal CAR blocks;both axes same bound initial state and occurrence list',
        full_operator_error_certified=False,statistical_confidence=None,
        all_q_words_compile_success_qualified=False,PCG64_joint_stream_qualified=False,
        Stage_C_ready=False,actual_CA_achieved=None)


def process_round(path, *, sample, context, **kwargs):
    """Use existing exclusive fresh storage; preserve partial output and STOP.

    Inputs have already been built/compiled by a separately authorized caller.
    No callback can sample/compile here. The scientific readiness entry below
    is deliberately not executable.
    """
    fresh.save_sample(path,sample,context); part = {}; stage = 'numerical_evaluation'
    try:
        result = evaluate_round(sample=sample,context=context,partial=part,**kwargs)
        row = fresh.outcome(sample,status='SUCCESS',evidence=dict(round_enclosure=result,partial=part))
    except BaseException as exc:
        status = 'NUMERICAL_UNEVALUABLE' if isinstance(exc,Exception) else 'INCOMPLETE'
        if isinstance(exc,CompilerWordError):
            status = 'P0_COMPILE_FAILED' if exc.status == 'FAILED' else 'INCOMPLETE'
            stage = 'P0_compile' if status == 'P0_COMPILE_FAILED' else None
        row = fresh.outcome(sample,status=status,stage=stage if status != 'INCOMPLETE' else None,
            error=exc,evidence=dict(round_enclosure=None,partial=part))
        fresh.save_outcome(path,row); fresh.finish_batch(path)
        if not isinstance(exc,Exception): raise
        return row
    fresh.save_outcome(path,row)
    return row


def calibration_payload(rows, *, context, planned_ids, state_id, target_id,
                        axis_support=None, delta_cal=None, failure_policy=None,
                        iid_assumed=False, compiler_domain=None):
    """New interval schema; delegate all statistics to the unchanged engine.

    The old fresh direct path requires Full512 certificates and scalar raw
    signals. This adapter preserves rectangles without manufacturing those
    certificates or discarding numerical uncertainty.
    """
    rows = list(rows); planned = list(planned_ids); fresh._check_context(context)
    ids = [r['sample']['sample_id'] for r in rows]; missing = []
    if len(set(ids)) != len(ids) or len(set(planned)) != len(planned): raise ValueError('DUPLICATE_SAMPLE')
    if not planned or ids != planned: missing.append('all_fixed_N_attempts_in_order_no_filtering')
    if delta_cal is None: missing.append('adopted_delta_cal')
    else: expected.probability(delta_cal)
    if failure_policy not in ('stop_on_failure','record_all_fixed_N'): missing.append('adopted_failure_policy')
    if iid_assumed is not True: missing.append('explicit_mathematical_IID_assumption')
    if (not compiler_domain or compiler_domain.get('scope') != 'all_legal_q_words'
            or compiler_domain.get('kind') != 'structural_totality_argument'
            or compiler_domain.get('gates') != 'exact_ideal_P0_unitaries'
            or not compiler_domain.get('provenance')):
        missing.append('qualified_total_compiler_domain_or_separately_adopted_failure_aware_estimator')
    values = []; domains = set()
    for row in rows:
        s = row['sample']; fresh.validate_sample(s,context)
        if s['unit'] != 'round': raise ValueError('ROUND_QUALIFICATION_NOT_KERNEL')
        domains.add((s['dataset_id'],s['m'],s['axis']))
        e = row['evidence'].get('round_enclosure')
        if row['status'] != 'SUCCESS' or not e:
            missing.append('failed_or_unfinished_samples_no_filtering'); continue
        want = dict(status='VALIDATED_FIXED_INPUT_ROUND_SIGNAL',qualification=QUALIFICATION,
            kind='rigorous_enclosure',scope='paired_round_raw_complex_signal',
            m=s['m'],occurrences=len(s['kernels']),round_word_id=s['round_word_id'],
            sample_id=s['sample_id'],source_id=s['source_id'],law_id=s['law_id'],
            state_id=state_id,target_id=target_id,precision_bits=192,phase_fit=False,
            shells_included=True,rounding_enclosed=True,full_operator_error_certified=False,
            included_error_terms=[expected.TERMS[0]],statistical_confidence=None,
            raw_signal_schema='complex_rectangle_exact_rational_endpoints_v1')
        if any(e.get(k) != v for k,v in want.items()): raise ValueError('ROUND_STATE_TARGET_SOURCE_NUMERICAL_QUALIFICATION')
        if not e.get('provenance') or any(e.get(k) is None for k in ('X','Y','raw_signals')):
            missing.append('missing_enclosure_not_zero'); continue
        for axis,coordinate in (('X','real'),('Y','imag')):
            for kind in ('compiled','literal_atom'):
                lo,hi = map(expected.rational,e['raw_signals'][kind][coordinate])
                if lo > hi: raise ValueError('COMPLEX_RECTANGLE_ORDER')
            a,b = e['raw_signals']['compiled'][coordinate],e['raw_signals']['literal_atom'][coordinate]
            if e[axis] != [str(F(a[0])-F(b[1])),str(F(a[1])-F(b[0]))]:
                raise ValueError('PAIRED_SIGNED_INTERVAL_IDENTITY')
        values.append(e)
    if len(domains) != 1: missing.append('one_fixed_calibration_stratum')
    m = rows[0]['sample']['m'] if rows else None
    if (not axis_support or axis_support.get('qualification') != 'deterministic_population_bound'
            or axis_support.get('m') != m or axis_support.get('target_id') != target_id
            or axis_support.get('state_id') != state_id or not axis_support.get('provenance')):
        missing.append('qualified_round_axis_population_support')
    policy = dict(model=context['identity']['model'],origin=fresh.ORIGIN,pool_old_samples=False,
        adaptive=False,fixed_samples=len(planned),sample_ids=planned,
        reuse='calibration_only_not_measurement_shots',iid_assumed=iid_assumed,
        PCG64_joint_stream_qualified=False)
    return dict(status='NOT_READY' if missing else 'CONDITIONAL_MATHEMATICAL_IID_INPUTS',
        missing=missing,method='direct_round',samples=values,policy=policy,
        support=deepcopy(axis_support),delta_cal=delta_cal,
        numerical_qualification=QUALIFICATION,calibration_confidence=None,
        Stage_C_ready=False,actual_CA_achieved=None)


def evaluate_calibration(payload):
    if payload['status'] != 'CONDITIONAL_MATHEMATICAL_IID_INPUTS':
        return dict(status='NOT_READY',bound=None,missing=payload['missing'],Stage_C_ready=False)
    result = expected.paired_round_upper(payload['samples'],axis_support=payload['support'],
        delta_round=payload['delta_cal'],policy=payload['policy'])
    return dict(result,confidence=expected.failure_accounting(payload['delta_cal']),
        numerical_qualification=QUALIFICATION,Stage_C_ready=False,actual_CA_achieved=None)


def run_science(**kwargs):
    raise PermissionError('UNAPPROVED_NONEXECUTABLE_SCIENCE_DRAFT_MANDATORY_STOP')


def preflight(root):
    """Bytes/JSON only; never load states, circuits or scientific arrays."""
    import hashlib
    import json
    from pathlib import Path
    from .c_b1_b2_scope_integration import identity
    root = Path(root).resolve(); c = json.loads((root/CONTRACT).read_text())
    want = dict(base_commit=BASE,executable=False,precision_bits=192,identity=identity(),
        maximum_logical_units=None,N=None,delta_cal=None,delta_compile=None,
        failure_policy=None,retry=0,resume=0,mandatory_stop=True,
        Stage_C_executable=False,actual_CA_achieved=None)
    if any(c.get(k) != v for k,v in want.items()):
        raise ValueError('FIXED_UNAPPROVED_NONEXECUTABLE_DRAFT')
    source = json.loads((root/c['technical_source']).read_text())
    for name,digest in source['files'].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('TECHNICAL_SOURCE_BYTES:'+name)
    return dict(status='TECHNICAL_READINESS_ONLY_SCIENCE_NOT_READY',execution_ready=False,
        scientific_calls=0,arrays_decoded=0,fresh_samples=0,build_compile_calls=0,
        important_scientific_reviews=0,Stage_C_ready=False,actual_CA_achieved=None,
        numerical_qualification=QUALIFICATION,mandatory_stop=True)
