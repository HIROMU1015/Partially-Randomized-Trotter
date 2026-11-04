"""Codex-B fresh calibration records; passive import, no molecular runner.

Draw functions are for an authorized caller supplying its frozen law. This
module never loads H4, builds/compiles a circuit, retries, or calls Stage C.
Recorded PCG64 bits provide replay, not a proof of mathematical independence.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

BASE = 'c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e'
TOTALITY = '60dcaa94fd44675bc3184e409aa4ebdef7a07dbf'
SCOPE = 'C_FRESH_IID_GRID_CALIBRATION_INFRASTRUCTURE_V1'
FAMILY = 'artifacts/research_c_fresh_iid_calibration/2026-11-04'
CONTRACT = 'docs/research/research_c_fresh_iid_calibration_contract_unapproved_20261104.json'
ORIGIN = 'fresh_IID_grid_calibration'
STREAM = 'existing_trajectory_seed_per_occurrence;K0_order_then_atom;raw64_high53_v1'
STATUSES = ('GENERATED', 'RAW_BUILD_FAILED', 'P0_COMPILE_FAILED',
            'NATIVE_INELIGIBLE', 'PER_WORD_CERTIFICATE_FAILED',
            'NUMERICAL_UNEVALUABLE', 'INCOMPLETE', 'SUCCESS')
FAILURE_STAGES = dict(RAW_BUILD_FAILED='raw_build', P0_COMPILE_FAILED='P0_compile',
    NATIVE_INELIGIBLE='native_qualification', PER_WORD_CERTIFICATE_FAILED='per_word_certificate',
    NUMERICAL_UNEVALUABLE='numerical_evaluation')
METHODS = ('threshold_tail', 'variance_sensitive', 'direct_round')


def digest(value):
    # Same canonical encoding as the existing per-word literal binding.
    from .c_per_word_population_range import _digest
    return _digest(value)


def _hex_id(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('SHA256_ID_REQUIRED')
    return value


def make_context(*, source_id, identity, atom_ids, table, event_angle_bits):
    """Bind supplied law, candidate and new SOURCE; no saved inputs are read.

    Source ID is a scientific reproducibility identity, never approval proof.
    Caller must bind this context's law_id into its future fixed contract.
    """
    from .c_b1_b2_scope_integration import identity as fixed_identity
    from .c_iid_grid_expected_bias import grid_index
    from .c_per_word_population_range import grid_support
    if identity != fixed_identity():
        raise ValueError('FIXED_CANDIDATE_SOURCE_MODEL_COMPILER_IDENTITY')
    _hex_id(source_id)
    ids = list(atom_ids)
    cdf = table.get('cdf', [])
    if len(cdf) != len(ids) or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('ORDERED_ATOM_CDF_TABLE')
    grid_index(0, cdf)  # reuse exact binary64 CDF validation, including zero bins
    ends = [-(-(F.from_float(x)*(1 << 53)).numerator //
               (F.from_float(x)*(1 << 53)).denominator) for x in cdf]
    counts = [b-a for a, b in zip([0]+ends[:-1], ends)]
    if table.get('counts') != counts:
        raise ValueError('CDF_GRID_COUNTS_MISMATCH')
    grid_support(dict.fromkeys(ids), table, expected_ids=ids)
    if set(event_angle_bits) != set(ids):
        raise ValueError('COMPLETE_SAVED_EVENT_ANGLE_BITS')
    for angle in event_angle_bits.values():
        if not isinstance(angle, str) or not math.isfinite(float.fromhex(angle)) or float.fromhex(angle).hex() != angle:
            raise ValueError('CANONICAL_FINITE_BINARY64_ANGLE')
    law = dict(atom_ids=ids, table=deepcopy(table), event_angle_bits=dict(event_angle_bits),
               order_cdf=[1.0], order_probabilities=[1.0], orders=[0])
    body = dict(schema=SCOPE, source_id=source_id, identity=deepcopy(identity), law=law,
                law_id=digest(law), stream_policy=STREAM)
    return dict(body, context_id=digest(body))


def _check_context(context):
    c = make_context(source_id=context['source_id'], identity=context['identity'],
        atom_ids=context['law']['atom_ids'], table=context['law']['table'],
        event_angle_bits=context['law']['event_angle_bits'])
    if context != c:
        raise ValueError('CONTEXT_SOURCE_OR_LAW_IDENTITY')


def _key(context, dataset_id, sample_index, unit, m, axis):
    if not isinstance(dataset_id, str) or not dataset_id or type(sample_index) is not int or sample_index < 0:
        raise ValueError('DATASET_AND_NONNEGATIVE_SAMPLE_INDEX')
    if ((unit == 'kernel' and (m is not None or axis != 'kernel')) or
            (unit == 'round' and (type(m) is not int or not 0 <= m <= 12 or axis != 'paired_XY')) or
            unit not in ('kernel', 'round')):
        raise ValueError('KERNEL_OR_PAIRED_CLASSICAL_ROUND_SCOPE')
    return dict(context_id=context['context_id'], dataset_id=dataset_id,
                sample_index=sample_index, unit=unit, m=m, axis=axis)


def sample_from_integers(context, *, dataset_id, sample_index, integers,
                         unit='kernel', m=None, axis='kernel'):
    """Replay explicit 53-bit integers. Each occurrence consumes exactly 64.

    Even K0 consumes an order-choice word before its atom-choice word. Direct
    round records contain 2**m independent-model occurrences and paired X/Y
    classical evaluations of that list, not shared quantum measurement shots.
    """
    from .c_iid_grid_expected_bias import grid_index
    from .c_p0_compiler_totality import check_word
    _check_context(context)
    key = _key(context, dataset_id, sample_index, unit, m, axis)
    sample_id = digest(key)
    draws = [list(xs) for xs in integers]
    n = 1 if unit == 'kernel' else 1 << m
    if len(draws) != n or any(len(xs) != 64 for xs in draws):
        raise ValueError('EXACT_K0_ORDER_AND_ATOM_DRAWS_FOR_EVERY_OCCURRENCE')
    law = context['law']; kernels = []
    for occurrence, xs in enumerate(draws):
        events = []
        for e in range(32):
            order = grid_index(xs[2*e], law['order_cdf'])
            index = grid_index(xs[2*e+1], law['table']['cdf'])
            atom = law['atom_ids'][index]
            events.append(dict(event_id=f'{sample_id}:{occurrence}:{e}', event_index=e,
                coherent_occurrence=occurrence, order_index=order, taylor_order=0,
                order_draw=dict(offset=2*e, uint53=xs[2*e]),
                atom_draw=dict(offset=2*e+1, uint53=xs[2*e+1]),
                atom_index=index, atom_id=atom, rotation_angle_bits=law['event_angle_bits'][atom]))
        word = [[e['atom_id'], e['rotation_angle_bits']] for e in events]
        check_word([e['atom_id'] for e in events], law['atom_ids'], law['table']['counts'])
        kernels.append(dict(coherent_occurrence=occurrence, word=word,
                            word_id=digest(word), events=events))
    return dict(schema=SCOPE, origin=ORIGIN, **key, sample_id=sample_id,
        source_id=context['source_id'], candidate=context['identity']['candidate'], law_id=context['law_id'],
        model=deepcopy(context['identity']['model']), kernels=kernels,
        round_word_id=digest([k['word_id'] for k in kernels]),
        rng=dict(backend='explicit_uint53', stream_policy=STREAM,
                 IID_guarantee=False, PCG64_joint_stream_qualified=False))


def sample_from_pcg64(context, *, dataset_id, sample_index, master,
                      unit='kernel', m=None, axis='kernel'):
    """Future caller only: exact legacy K0 draw consumption, with replay state.

    No N/default master is selected. NumPy PCG64 supplies one uint64 per
    Generator.random double; its upper 53 bits feed the existing grid map.
    Seed domain separation reuses trajectory_seed, not a new RNG/seed scheme.
    """
    import numpy as np
    from .c1_stage3_connection import trajectory_seed
    if np.__version__ != '1.26.4' or type(master) is not int or master < 0:
        raise ValueError('FIXED_NUMPY_1_26_4_AND_EXPLICIT_MASTER')
    _check_context(context)
    _key(context, dataset_id, sample_index, unit, m, axis)
    domain_master = (ORIGIN, context['source_id'], context['law_id'], dataset_id, master)
    draws = []; streams = []
    for occurrence in range(1 if unit == 'kernel' else 1 << m):
        seed = trajectory_seed(domain_master, 'proposed', context['identity']['candidate'],
                               -1 if m is None else m, axis, sample_index, occurrence)
        bitgen = np.random.PCG64(seed)
        before = deepcopy(bitgen.state)
        draws.append([int(x) >> 11 for x in bitgen.random_raw(64)])
        streams.append(dict(occurrence=occurrence, seed=seed, before=before,
                            after=deepcopy(bitgen.state), raw64_consumed=64))
    result = sample_from_integers(context, dataset_id=dataset_id, sample_index=sample_index,
        integers=draws, unit=unit, m=m, axis=axis)
    result['rng'] = dict(backend='numpy.PCG64', numpy=np.__version__, master=master,
        domain_master=list(domain_master), streams=streams, stream_policy=STREAM,
        IID_guarantee=False, PCG64_joint_stream_qualified=False)
    return result


def validate_sample(sample, context):
    """Replay recorded bits without consuming a PRNG or reading molecular data."""
    if sample.get('origin') != ORIGIN:
        raise ValueError('NO_FIXED_FOUR_WORD_OR_NATIVE32_REUSE')
    integers = [[x for e in k['events'] for x in (e['order_draw']['uint53'], e['atom_draw']['uint53'])]
                for k in sample['kernels']]
    replay = sample_from_integers(context, dataset_id=sample['dataset_id'], sample_index=sample['sample_index'],
        integers=integers, unit=sample['unit'], m=sample['m'], axis=sample['axis'])
    rng = sample['rng']
    if (rng.get('backend') not in ('explicit_uint53', 'numpy.PCG64') or
            rng.get('stream_policy') != STREAM or rng.get('IID_guarantee') is not False or
            rng.get('PCG64_joint_stream_qualified') is not False):
        raise ValueError('REPLAY_IS_NOT_IID_PROOF')
    if rng['backend'] == 'numpy.PCG64':
        from .c1_stage3_connection import trajectory_seed
        domain = [ORIGIN, context['source_id'], context['law_id'], sample['dataset_id'], rng.get('master')]
        streams = rng.get('streams', [])
        if (rng.get('numpy') != '1.26.4' or type(rng.get('master')) is not int or rng['master'] < 0 or
                rng.get('domain_master') != domain or len(streams) != len(sample['kernels'])):
            raise ValueError('PRNG_REPLAY_METADATA')
        for i, stream in enumerate(streams):
            seed = trajectory_seed(tuple(domain), 'proposed', context['identity']['candidate'],
                -1 if sample['m'] is None else sample['m'], sample['axis'], sample['sample_index'], i)
            if (stream.get('occurrence') != i or stream.get('seed') != seed or stream.get('raw64_consumed') != 64 or
                    any(stream.get(k, {}).get('bit_generator') != 'PCG64' for k in ('before', 'after'))):
                raise ValueError('PRNG_OCCURRENCE_STREAM')
    if {k: v for k, v in sample.items() if k != 'rng'} != {k: v for k, v in replay.items() if k != 'rng'}:
        raise ValueError('SAMPLE_SOURCE_CANDIDATE_LAW_OR_EVENT_ORDER')
    return True


def rte_events(sample, context, *, components, distribution):
    """Convert recorded choices through the original event constructor only.

    Caller supplies its already prepared immutable tail/distribution. No raw
    circuit is built. K0 event phases, signs and angle calculation stay in rte.
    """
    from .c_pr_aware_df_stage_c2 import choice_grid
    from trotterlib.rte import _make_event
    validate_sample(sample, context)
    if (list(distribution.orders) != [0] or list(distribution.order_probabilities) != [1.0] or
            [c.component_id for c in components] != context['law']['atom_ids']):
        raise ValueError('ORIGINAL_K0_TAIL_DISTRIBUTION')
    mass = math.fsum(c.probability for c in components)
    table = choice_grid([c.probability/mass for c in components])
    if any(table[k] != context['law']['table'][k] for k in ('cdf', 'counts', 'bits', 'mapping')):
        raise ValueError('ORIGINAL_K0_LAW_TABLE')
    out = []
    for kernel in sample['kernels']:
        events = tuple(_make_event((e['atom_index'],), components, distribution, 0) for e in kernel['events'])
        if any(float(e.rotation_angle).hex() != row[1] for e, row in zip(events, kernel['word'])):
            raise ValueError('ORIGINAL_K0_EVENT_ANGLE')
        out.append(events)
    return tuple(out)


def outcome(sample, *, status='GENERATED', stage=None, error=None, evidence=None):
    """Keep the whole draw and all partial evidence, including failed words."""
    if status not in STATUSES or (status in FAILURE_STAGES and stage != FAILURE_STAGES[status]):
        raise ValueError('EXPLICIT_FAILURE_STAGE')
    if status in FAILURE_STAGES and error is None:
        raise ValueError('FAILURE_EXCEPTION_REQUIRED')
    exception = None if error is None else dict(type=type(error).__name__, message=str(error))
    return dict(sample=deepcopy(sample), status=status, failure_stage=stage,
                exception=exception, evidence=deepcopy(evidence or {}))


def _write_new(path, value):
    # Exclusive create prevents overwrite/re-extraction; interrupted bytes stay.
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()


def open_batch(path):
    """Fresh directory only. There is no execute/resume/retry entry point."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path/'samples').mkdir(); (path/'outcomes').mkdir()
    return path


def _active(path):
    path = Path(path)
    if (path/'STOP.json').exists():
        raise ValueError('MANDATORY_STOP_NO_RETRY_RESUME')
    return path


def save_sample(path, sample, context):
    validate_sample(sample, context)
    _write_new(_active(path)/'samples'/(_hex_id(sample['sample_id'])+'.json'), sample)


def save_outcome(path, row):
    path = _active(path); s = row['sample']; key = _hex_id(s['sample_id'])
    if json.loads((path/'samples'/(key+'.json')).read_text()) != s or row['status'] not in STATUSES:
        raise ValueError('OUTCOME_MUST_RETAIN_ORIGINAL_DRAW')
    _write_new(path/'outcomes'/(key+'.json'), row)


def read_batch(path):
    """Read-only recovery: a draw without an outcome remains INCOMPLETE."""
    path = Path(path); rows = []
    for p in sorted((path/'samples').glob('*.json')):
        sample = json.loads(p.read_text()); result = path/'outcomes'/p.name
        row = json.loads(result.read_text()) if result.exists() else outcome(sample, status='INCOMPLETE')
        if row['sample'] != sample:
            raise ValueError('SAVED_OUTCOME_DRAW_MISMATCH')
        rows.append(row)
    return rows


def finish_batch(path):
    path = _active(path); rows = read_batch(path)
    value = dict(mandatory_stop=True, retry=0, resume=0, samples=len(rows),
        outcomes={s: sum(r['status'] == s for r in rows) for s in STATUSES},
        statistical_qualification=None, failure_policy_adopted=None)
    _write_new(path/'STOP.json', value)
    return value


def calibration_payload(rows, *, context, planned_ids, method=None, delta_cal=None,
                        failure_policy=None, iid_assumed=False, population=None,
                        compiler_domain=None, axis_support=None):
    """Connect complete records to old statistics, conditional on supplied premises.

    No failure policy is adopted here. Both proposed policies retain failures;
    neither turns a successful-only subset into the original population.
    This all-success interface has no failure-aware estimator implementation.
    """
    from . import c_iid_grid_expected_bias as expected
    from . import c_per_word_population_range as population_adapter
    _check_context(context)
    rows = list(rows); planned = list(planned_ids); missing = []
    ids = [r['sample']['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(planned) != len(set(planned)):
        raise ValueError('DUPLICATE_SAMPLE_ID_NOT_DUPLICATE_WORD')
    for r in rows:
        validate_sample(r['sample'], context)
        if r.get('status') not in STATUSES:
            raise ValueError('UNKNOWN_SAMPLE_STATUS')
    if not planned or set(ids) != set(planned): missing.append('all_fixed_N_attempts')
    if method not in METHODS: missing.append('adopted_method')
    if delta_cal is None: missing.append('delta_cal')
    else: expected.probability(delta_cal)
    if failure_policy not in ('stop_on_failure', 'record_all_fixed_N'): missing.append('adopted_failure_policy')
    if iid_assumed is not True: missing.append('explicit_mathematical_IID_assumption')
    if any(r['status'] != 'SUCCESS' for r in rows): missing.append('failed_or_unfinished_samples_no_filtering')
    units = {r['sample']['unit'] for r in rows}
    if rows and method in METHODS and units != ({'round'} if method == 'direct_round' else {'kernel'}):
        raise ValueError('CALIBRATION_UNIT')
    domains = {(r['sample']['dataset_id'], r['sample']['m'], r['sample']['axis']) for r in rows}
    if len(domains) > 1:
        raise ValueError('ONE_FIXED_CALIBRATION_STRATUM')
    if method == 'variance_sensitive' and len(planned) < 2:
        missing.append('variance_requires_N_at_least_two')
    policy = dict(model=deepcopy(context['identity']['model']), origin=ORIGIN,
        pool_old_samples=False, adaptive=False, fixed_samples=len(planned), sample_ids=planned,
        reuse='calibration_only_not_measurement_shots', iid_assumed=iid_assumed,
        PCG64_joint_stream_qualified=False)
    payload = dict(status='NOT_READY', missing=missing, method=method, delta_cal=delta_cal,
        source_id=context['source_id'], context_id=context['context_id'], law_id=context['law_id'],
        identity=deepcopy(context['identity']), records=deepcopy(rows), samples=None, support=None,
        policy=policy, calibration_confidence=None, PCG64_joint_stream_qualified=False,
        all_word_uniform=False, actual_CA_achieved=None, Stage_C_ready=False)
    payload['calibration_event'] = None if method not in METHODS or delta_cal is None else dict(
        id=digest(dict(context=context['context_id'], method=method, samples=planned, delta=str(delta_cal))),
        delta=str(delta_cal), m=rows[0]['sample']['m'] if rows else None)
    if not population:
        missing.append('qualified_population_norm_range')
        return payload
    intervals = []
    for r in rows:
        s = r['sample']; certs = r['evidence'].get('per_word_enclosures', [])
        if len(certs) != len(s['kernels']):
            missing.append('per_word_enclosure:'+s['sample_id']); continue
        for cert, kernel in zip(certs, s['kernels']):
            if (cert.get('scope') != 'per_word_full512' or
                    cert.get('status') != 'VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR' or
                    cert.get('target') != population_adapter.TARGET or not cert.get('provenance')):
                missing.append('qualified_full512_per_word_enclosure:'+s['sample_id'])
                continue
            expected_binding = dict(sample_id=s['sample_id'], source_id=s['source_id'],
                candidate=s['candidate'], law_id=s['law_id'],
                occurrence=kernel['coherent_occurrence'], word_id=kernel['word_id'])
            if cert.get('calibration_binding') != expected_binding:
                raise ValueError('FRESH_CERTIFICATE_SAMPLE_SOURCE_CANDIDATE_LAW')
            binding = cert.get('binding', {})
            if binding.get('word') != kernel['word'] or binding.get('word_id') != kernel['word_id']:
                raise ValueError('CERTIFICATE_REALIZED_WORD_BINDING')
            intervals.append(cert)
    checked = population_adapter.calibration_inputs(intervals, population, compiler_domain=compiler_domain)
    if checked['status'] != 'CONDITIONAL_CALIBRATION_INPUTS': missing.append(checked['status'])
    if missing: return payload
    if method == 'direct_round':
        m = rows[0]['sample']['m']; values = []
        for r in rows:
            s = r['sample']; e = r['evidence'].get('round_enclosure')
            if not e or any(e.get(k) != v for k, v in dict(kind='rigorous_enclosure',
                    scope='paired_round_raw_complex_signal', m=m, round_word_id=s['round_word_id'],
                    sample_id=s['sample_id'], source_id=s['source_id'], law_id=s['law_id'],
                    target_id=population['target_id'], phase_fit=False, rounding_enclosed=True,
                    shells_included=True, included_error_terms=[expected.TERMS[0]]).items()):
                missing.append('qualified_round_enclosure:'+s['sample_id']); continue
            if not e.get('provenance') or any(e.get(k) is None for k in ('X', 'Y', 'raw_signals')):
                missing.append('paired_XY_enclosure_and_raw_signals:'+s['sample_id']); continue
            for signal in ('compiled', 'literal_atom'):
                raw = e['raw_signals'].get(signal, {})
                if set(raw) != {'real', 'imag'}:
                    raise ValueError('RAW_COMPLEX_SIGNAL_SCHEMA')
                for x in raw.values(): expected.rational(x)
            values.append(e)
        if (not axis_support or axis_support.get('m') != m or
                axis_support.get('target_id') != population['target_id'] or
                axis_support.get('qualification') != 'deterministic_population_bound' or
                not axis_support.get('provenance')):
            missing.append('qualified_round_axis_population_support')
        payload.update(samples=values, support=deepcopy(axis_support))
    else:
        payload.update(samples=checked['samples'], support=checked['support'])
    if not missing:
        payload.update(status='CONDITIONAL_MATHEMATICAL_IID_INPUTS',
            calibration_confidence=expected.failure_accounting(delta_cal))
    return payload


def evaluate_calibration(payload, *, threshold=None):
    """Authorized caller only; delegate scalar arithmetic without Stage C."""
    from . import c_iid_grid_expected_bias as expected
    from . import c_per_word_population_range as population
    if payload['status'] != 'CONDITIONAL_MATHEMATICAL_IID_INPUTS':
        return dict(status='NOT_READY', bound=None, missing=payload['missing'])
    args = dict(support=payload['support'], delta=payload['delta_cal'], policy=payload['policy'])
    if payload['method'] == 'threshold_tail':
        result = population.threshold_tail(payload['samples'], threshold=threshold, **args)
    elif payload['method'] == 'variance_sensitive':
        result = population.variance_sensitive(payload['samples'], **args)
    elif payload['method'] == 'direct_round':
        result = expected.paired_round_upper(payload['samples'], axis_support=args['support'],
            delta_round=args['delta'], policy=args['policy'])
    else:
        raise ValueError('CALIBRATION_METHOD')
    return dict(result, confidence=payload['calibration_confidence'],
        calibration_event=deepcopy(payload['calibration_event']),
        source_id=payload['source_id'], law_id=payload['law_id'],
        PCG64_joint_stream_qualified=False, Stage_C_ready=False)


def run_science(*, execute_science=False):
    if not execute_science:
        raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_EXECUTION_PERMISSION')
    raise NotImplementedError('NONEXECUTABLE_METHOD_N_DELTA_FAILURE_POLICY_UNADOPTED')


def preflight(root, contract):
    """Only JSON and byte identities; no scientific imports/draws/decoding."""
    root = Path(root).resolve(); c = json.loads(Path(contract).read_text())
    required = dict(scope=SCOPE, base_commit=BASE, executable=False,
        maximum_logical_units=None, N=None, N_by_round=None, delta_cal=None,
        delta_compile=None, method=None, compiler_failure_policy=None,
        retry=0, resume=0, mandatory_stop=True)
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('FIXED_NONEXECUTABLE_FRESH_CALIBRATION_CONTRACT')
    source = (root/c['source_manifest']).resolve()
    if (not source.is_relative_to(root) or
            hashlib.sha256(source.read_bytes()).hexdigest() != c['source_manifest_sha256']):
        raise ValueError('TECHNICAL_SOURCE_MANIFEST')
    files = json.loads(source.read_text())['files']
    identity = json.loads(source.read_text())['identity']
    if (c.get('identity') != identity or c.get('sampling_model') != identity['model'] or
            c.get('stream_policy') != STREAM):
        raise ValueError('FIXED_SOURCE_CANDIDATE_SAMPLING_LAW')
    for name, want in {**files, **c['saved_input_bytes']}.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise ValueError('SOURCE_INPUT_BYTES:'+name)
    fresh = (root/c['fresh_output']).resolve()
    if not fresh.is_relative_to(root) or fresh.exists():
        raise ValueError('FRESH_OUTPUT_NO_RETRY_RESUME')
    return dict(status='PASSIVE_TECHNICAL_ONLY_NOT_READY', execution_ready=False,
        scientific_calls=0, real_H4_samples=0, arrays_decoded=0, native_build_compile=0,
        calibration_calls=0, Stage_C_calls=0, important_reviews=0,
        maximum_logical_units=None, mandatory_stop=True)
