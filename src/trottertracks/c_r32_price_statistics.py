"""Prospective fixed-r32 price calibration; passive until explicit execution.

Old raw trajectories never enter the estimator. Native32 batches reuse the
unchanged complete-kernel/adjoint/X-Y pipeline, with fresh seed namespaces.
The t interval is an approximate IID-model interval, not a validated bound.
"""
import importlib.metadata
import math
from pathlib import Path
import resource
import time

from . import c_rte_r_retuning as ret

io = ret.io
BASE = 'f200e45afecc0a80b9f8a895b68c75ba4dda5dd5'
SCOPE = 'C_UPDATE005_R32_PROSPECTIVE_PRICE_V1'
VERSION = 'native32_batches_independent_holdout_approximate_t_v1'
FAMILY = 'artifacts/research_c_r32_price_statistics/2026-11-03'
FRESH = FAMILY + '/run01'
N = 640
BATCHES = N // 32
STATISTICS = dict(samples=N, batch_size=32, confidence=.95,
    precision_halfwidth_fraction_of_D=.005, primary_metric='rz_count',
    primary_lane='ideal_exact_native', method='fixed_N_two_sided_approximate_Student_t',
    old_samples_pooled=False, adaptive=False, interim_inference=False,
    equivalence_margin=None, noninferiority_margin=None,
    parameter_status='PROPOSAL_REQUIRES_USER_RESEARCH_C_ADOPTION')
LIMITS = {**ret.A_LIMITS, 'saved_r32_gate': 1, 'seed_disjoint_gate': 1,
    'law_prepare': 1, 'law_gate': 1, 'normalization_gate': 1,
    **{'native.'+k: v for k, v in ret.COMMON_NATIVE.items()},
    **{f'batch{i:02d}.'+k: v for i in range(BATCHES) for k, v in ret.SAMPLE_NATIVE.items()},
    'accounting_binding': 1, 'native_rows_validation': 1,
    'primary_statistics': 1, 'secondary_summary': 1, 'result_summary': 1}
MAX_CALLS = sum(LIMITS.values())


def sampling(batch):
    if type(batch) is not int or not 0 <= batch < BATCHES:
        raise ValueError('REGISTERED_BATCH_ONLY')
    return dict(master=2026110302, cohort=f'update005_r32_prospective_v1_batch{batch:02d}',
        axis='R32_PRICE_HOLDOUT_ONLY', samples=32, events_per_sample=32,
        PRNG='numpy.random.PCG64', paired=False, adaptive=False)


def seed_manifest(source_digest, old_contract):
    """Hash seed keys only; no PRNG draws, trajectory sampling or array decode."""
    def seeds(policy, digest):
        master = (policy['master'], policy['cohort'], digest)
        return [ret.perf.b.connection.trajectory_seed(master, 'proposed', ret.CANDIDATE,
            -1, policy['axis'], i, 0) for i in range(32)]
    old = []
    for r in ret.NEW_R:
        old.extend(seeds(ret.sampling(r), old_contract['source_manifest_sha256']))
    old.extend(seeds(ret.perf.SAMPLING, old_contract['integrated_source_manifest_sha256']))
    rows = [dict(id=f'batch{b:02d}/sample{i:02d}', global_id=32*b+i, seed=seed)
        for b in range(BATCHES) for i, seed in enumerate(seeds(sampling(b), source_digest))]
    new = [row['seed'] for row in rows]
    if len(set(new)) != N or set(new) & set(old):
        raise ValueError('SEED_COLLISION_OR_OLD_COHORT_OVERLAP')
    return dict(status='PASS', rows=rows, old_keys_checked=len(old),
        interpretation='distinct hash-derived PCG64 seeds; no mathematical independence or ideal-law distance certificate',
        seed_bits=64, old_trajectories_decoded=0, PRNG_draws=0)


def saved_gate(root, c):
    """Reuse declared scalar records. Never read old samples/metrics or QPY."""
    root = Path(root)
    saved = ret.evidence(root, c)
    v = io.read(root/c['r32_result']); stop = io.read(root/c['retuning_stop'])
    if (stop['failed'] or stop['scientific_units'] != 1265 or stop['retry'] or stop['resume']
            or v['r'] != 32 or v['status'] != 'NATIVE_AND_CONDITIONAL_COST_COMPLETE'
            or not v['native_admitted'] or v['calibration']['samples'] != 32
            or v['feasibility']['status'] != 'POTENTIALLY_FEASIBLE'
            or [x['m'] for x in v['rounds']] != list(range(13))):
        raise ValueError('SAVED_R32_QUALIFICATION')
    for lane in ('ideal_exact_native', 'assumed_raw_signal_budget'):
        value = v['lanes'][lane]
        if (not value['energy']['same_accuracy'] or value['total_shots'] is None
                or any(x['target_id'] != f'retuning:{ret.CANDIDATE}:r32' for x in value['rounds'])):
            raise ValueError('SAVED_RPE_TARGET')
    ideal = v['lanes']['ideal_exact_native']
    if (ideal['total_shots'] != 15202 or ideal['weighted_SE']['kernel_reuse_weight'] != 13979232
            or ideal['conditional_totals']['totals']['rz_count'] != 106752458375
            or v['calibration']['sample_mean']['rz_count'] != 7636.5
            or saved['D']['ideal_exact_native']['cost']['totals']['rz_count'] != 105532782860
            or v['lanes']['evidence']['conditional_totals']['totals'] is not None):
        raise ValueError('SAVED_SCALAR_OR_LANE_IDENTITY')
    return v, saved


def preflight(root, path):
    root = Path(root).resolve(); c = io.read(path)
    fixed = dict(base_commit=BASE, scope=SCOPE, implementation=VERSION, policy=ret.POLICY,
        r=32, target=ret.m4.TARGET, compiler=ret.COMPILER, allowances=ret.m3.ALLOWANCES,
        statistics=STATISTICS, call_limits=LIMITS, maximum_logical_units=MAX_CALLS,
        sampling=[sampling(i) for i in range(BATCHES)], fresh_output=FRESH,
        retry=0, resume=0, new_fit=0, optimizer=0, new_PF_RTE=0, new_shots=0)
    for key, value in fixed.items():
        if c[key] != value: raise ValueError('FIXED_PRICE_SCOPE:'+key)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    needed = {c[k] for k in ('input', 'selected', 'physical_reference', 'sector_reference',
        'basis', 'ground', 'r32_result', 'retuning_stop', 'retuning_contract')}
    needed.update(c['records'].values())
    needed.update(p for row in c['shells'].values() for p in row.values())
    if (not needed <= c['saved_inputs'].keys() or c['saved_inputs'][c['selected']] != ret.CANDIDATE
            or c['saved_inputs'][c['input']] != ret.INPUT):
        raise ValueError('PINNED_PRICE_INPUT')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != digest:
                raise ValueError('SOURCE_INPUT:'+name)
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('ENVIRONMENT:'+name)
    old = io.read(root/c['retuning_contract'])
    parent = io.read(root/c['records']['parent_contract'])
    if c['old_seed_policy'] != dict(source_manifest_sha256=old['source_manifest_sha256'],
            integrated_source_manifest_sha256=parent['source_manifest_sha256']):
        raise ValueError('OLD_SEED_PROVENANCE')
    from .c1_h4_native_fair import GAUSSIAN, PHASE
    if c['gaussian'] != GAUSSIAN or c['compiler_phase'] != PHASE:
        raise ValueError('NATIVE_PRECISION')
    saved_gate(root, c)
    seed_manifest(c['source_manifest_sha256'], c['old_seed_policy'])
    if (root/FRESH).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def prepare_native(root, c, out, call):
    a = ret.prepare(root, c, out, call)
    b = ret.perf.b
    tail, config, dist = call('law_prepare', 'fixed r32/K0',
        lambda: b.configure(a['arm'], ret.Fraction(1, 10), 32, 0))
    def gate():
        from .c_rc2_pr_stage_b import distribution_gate
        row = b.tail_gate(a['ctx'], a['arm'], a['sector']['generators'], tail)
        distribution_gate(tail, config, dist, expected_steps=32)
        if any(row[k] != a['saved']['law'][k] for k in ('lambda_R', 'identity', 'atoms')):
            raise ValueError('LAW_CANDIDATE_IDENTITY')
        return row
    io.save(out/'law.json', call('law_gate', 'same signed tail', gate))
    io.save(out/'normalization.json', call('normalization_gate', 'unchanged K0 convention',
        lambda: ret.normalization(tail, config, 32)))
    return dict(arm=a['arm'], p=a['p'], tail=tail, config=config, dist=dist,
        sector=a['sector'], ground=a['ground'])


def affine_bindings(v, saved):
    """Reuse serial composition with a unit-price kernel to identify W and S.

    Called only during permitted scientific execution or with synthetic input.
    No RPE shots formula or saved raw trajectory is evaluated.
    """
    metrics = ret.m4.METRICS; result = {}
    for lane in ('ideal_exact_native', 'assumed_raw_signal_budget'):
        plans = v['lanes'][lane]['rounds']; target = plans[0]['target_id']
        kernel = dict(target_id=target, mean={k: 1 for k in metrics},
            policy='native_kernel_repeat_with_fences', simultaneous_interval_qualified=False)
        packet = dict(target_id=target, shells=saved['shell_reports'])
        costs = [ret.m3.serial_cost(packet, kernel, m, axis)
            for m in range(13) for axis in ('X', 'Y')]
        totals = ret.m4.accuracy.aggregate_totals(v['lanes'][lane]['energy'], plans,
            costs, cost_scope=ret.accounting.COST_SCOPE)['totals']
        W = sum(2*p['shots']*(1 << p['m']) for p in plans)
        if W != v['lanes'][lane]['weighted_SE']['kernel_reuse_weight']:
            raise ValueError('SAVED_W_BINDING')
        S = {k: totals[k]-W for k in metrics}
        if any(not math.isfinite(x) or x < 0 for x in S.values()):
            raise ValueError('SHELL_ACCOUNTING')
        D_lane = 'ideal_exact_native' if lane == 'ideal_exact_native' else 'numerical'
        result[lane] = dict(W=W, shell=S, D=saved['D'][D_lane]['cost']['totals'],
            shots=v['lanes'][lane]['total_shots'])
    return result


def validate_rows(rows, count=N):
    if len(rows) != count: raise ValueError('INCOMPLETE_HOLDOUT_NO_DROPPING')
    for row in rows:
        for k in ret.m4.METRICS:
            x = row[k]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or x != int(x):
                raise ValueError('INVALID_NATIVE_METRIC')
    return dict(status='PASS', samples=count, dropped=0, winsorized=0, old_pooled=0)


def distribution_diagnostics(rows,mean,se,half,*,weight=1.,policy=None):
    """Shared existing moment/delete-one diagnostics, without a D denominator."""
    policy=STATISTICS if policy is None else policy
    expected_n=len(rows)
    if expected_n<2 or se<=0 or half<=0:raise ValueError('POSITIVE_SAMPLING_UNCERTAINTY_REQUIRED')
    sd=se*math.sqrt(expected_n)
    skew=math.fsum(((x['rz_count']-mean)/sd)**3 for x in rows)/expected_n
    influence=max(abs(x['rz_count']-mean)/(expected_n-1) for x in rows)
    fraction=policy.get('maximum_delete_one_shift_fraction_of_halfwidth',1.)
    return dict(moment_skewness=skew,maximum_delete_one_mean_shift=influence,
        delete_one_shift_exceeds_halfwidth=influence*weight>half,
        delete_one_shift_fraction_of_halfwidth=influence*weight/half,
        influence_fraction_threshold=fraction,influence_warning=influence*weight>fraction*half,
        skewness_warning=abs(skew)>policy.get('skewness_abs_threshold',2.),observations_removed=0,
        interpretation='diagnostics cannot verify independence, unseen tails or interval coverage')


def price_statistics(rows, affine, *, expected_n=N, statistics=None):
    """Approximate fixed-N t inference for one preselected price, not CA."""
    from scipy.stats import t
    policy = STATISTICS if statistics is None else statistics
    if expected_n < 2: raise ValueError('AT_LEAST_TWO_SAMPLES')
    validate_rows(rows, expected_n)
    cal = ret.perf.b2.calibration(rows)
    mean = cal['sample_mean']['rz_count']; se = cal['sample_mean_standard_error']['rz_count']
    W, S, D = affine['W'], affine['shell']['rz_count'], affine['D']['rz_count']
    if (expected_n < 2 or not W > 0 or not D > 0
            or not all(math.isfinite(x) for x in (W, S, D)) or S < 0):
        raise ValueError('INVALID_AFFINE_COMPARISON')
    estimate = W*mean+S
    out = dict(samples=expected_n, confidence=policy['confidence'], kernel_mean=mean,
        kernel_SE=se, total_point=estimate, total_SE=W*se, difference_point=estimate-D,
        ratio_point=estimate/D, total_interval=None, difference_interval=None, ratio_interval=None,
        interval_kind='APPROXIMATE_FIXED_N_IID_STUDENT_T_NOT_VALIDATED',
        actual_CA_achieved=None, guaranteed_resource_superiority=None,
        ideal_law_cost_transfer_qualified=False, equivalence=None, noninferiority=None,
        old_pooled=False, selection_scope='preselected r32 only; independent prospective holdout',
        calibration=cal)
    if se == 0:
        return dict(out, decision='UNRESOLVED_ZERO_OBSERVED_VARIANCE', precision_met=False)
    half = float(t.ppf((1+policy['confidence'])/2, expected_n-1))*W*se
    if not math.isfinite(half) or half <= 0: raise ValueError('INVALID_T_INTERVAL')
    total = [estimate-half, estimate+half]
    # Preserve the unconstrained t interval; do not clip it at observed extrema.
    out.update(total_interval=total, difference_interval=[x-D for x in total],
        ratio_interval=[x/D for x in total], halfwidth=half,
        precision_met=half/D <= policy['precision_halfwidth_fraction_of_D'])
    diagnostics=distribution_diagnostics(rows,mean,se,half,weight=W,policy=policy)
    out['diagnostics']=diagnostics
    influence_warning=diagnostics['influence_warning']
    skew_threshold=policy.get('skewness_abs_threshold',2.)
    skew=diagnostics['moment_skewness']
    if influence_warning or abs(skew)>skew_threshold:
        decision = 'UNRESOLVED_DISTRIBUTION_DIAGNOSTIC'
    elif not out['precision_met']:
        decision = 'UNRESOLVED_PRECISION_TARGET'
    elif total[1] < D: decision = 'NOMINAL_MODEL_PR_LOWER'
    elif total[0] > D: decision = 'NOMINAL_MODEL_PR_HIGHER'
    else: decision = 'UNRESOLVED_SIGN_NOT_EQUIVALENCE'
    return dict(out, decision=decision)


def secondary_summary(cal, bindings):
    return {lane: dict(qualification='DESCRIPTIVE_ONLY_NO_SIMULTANEOUS_CI_OR_RANKING',
        totals={k: a['W']*cal['sample_mean'][k]+a['shell'][k] for k in ret.m4.METRICS},
        total_SE={k: a['W']*cal['sample_mean_standard_error'][k] for k in ret.m4.METRICS})
        for lane, a in bindings.items()}


def execute(root, c, out, call, *, prepare_function=None, native_function=None, study=None):
    # Candidate adapters supply only metadata/seed hooks and a fixed design.
    # Default update005 behavior is preserved; native/statistical engines shared.
    design = dict(samples=N, statistics=STATISTICS, saved_gate=saved_gate, affine_bindings=affine_bindings,
        seed_manifest=seed_manifest, sampling=sampling, candidate=ret.CANDIDATE)
    if study is not None: design.update(study)
    count = design['samples']; batches = count // 32
    if type(count) is not int or count < 32 or count % 32: raise ValueError('FIXED_NATIVE32_BATCHES')
    v, saved = call('saved_r32_gate', 'fixed scalar qualifications, no raw32 decode', lambda: design['saved_gate'](root, c))
    seeds = call('seed_disjoint_gate', 'fresh prospective keys vs registered prior cohorts',
        lambda: design['seed_manifest'](c['source_manifest_sha256'], c['old_seed_policy']))
    io.save(out/'seed_manifest.json', seeds)
    bindings = call('accounting_binding', 'saved shots/W/shell/D; no new shots', lambda: design['affine_bindings'](v, saved))
    io.save(out/'fixed_accounting.json', bindings)
    a = (prepare_function or prepare_native)(root, c, out, call)
    prepared = {}; rows = []
    for batch in range(batches):
        folder = out/f'batch{batch:02d}'; folder.mkdir()
        def dispatch(kind, label, f):
            key = 'native.'+kind if kind in ret.COMMON_NATIVE else f'batch{batch:02d}.'+kind
            return call(key, label, f)
        dispatch.stop = call.stop
        batch_rows, _ = (native_function or ret.perf.native)(root, c, folder, dispatch, a,
            sampling=design['sampling'](batch), prepared=prepared)
        rows.extend(batch_rows)
        # No interval, ranking, stopping-by-sign or precision peeking per batch.
        io.save(folder/'completion.json', dict(completed_batches=batch+1,
            completed_samples=len(rows), interim_inference=False))
    io.save(out/'validation.json', call('native_rows_validation', f'all{count};no exclusions', lambda: validate_rows(rows, count)))
    primary = call('primary_statistics', 'one prespecified fixed-N nominal interval',
        lambda: price_statistics(rows, bindings['ideal_exact_native'], expected_n=count, statistics=design['statistics']))
    io.save(out/'primary.json', primary)
    secondary = call('secondary_summary', 'six metrics and assumed lane;descriptive',
        lambda: secondary_summary(primary['calibration'], bindings))
    return call('result_summary', 'price-only;mandatory STOP', lambda: dict(
        status='PROSPECTIVE_R32_PRICE_COMPLETE', candidate=design['candidate'], primary=primary, secondary=secondary,
        old_samples_pooled=False, quantum_shots_drawn=0, new_fit=0, retry=0, resume=0,
        actual_CA_achieved=None, guaranteed_resource_superiority=None, mandatory_stop=True))


def run(root, path):
    root = Path(root).resolve(); c = preflight(root, path); out = root/FRESH
    out.mkdir(parents=True); tick = time.monotonic()
    call = ret.m4.integrated.Calls(out, LIMITS, MAX_CALLS)
    io.save(out/'contract.json', c); io.save(out/'SOURCE.json', io.read(root/c['source_manifest']))
    try:
        value = execute(root, c, out, call); io.save(out/'result.json', value); return value
    except BaseException as error:
        call.stop.set(); io.save(out/'failure.json', dict(error=repr(error), first_failure=call.first_failure)); raise
    finally:
        io.save(out/'STOP.json', dict(failed=call.stop.is_set(), scientific_units=sum(call.counts.values()),
            maximum_logical_units=MAX_CALLS, call_counts=dict(call.counts), first_failure=call.first_failure,
            retry=0, resume=0, mandatory_stop=True))
        io.save(out/'resources.json', dict(wall_seconds=time.monotonic()-tick, CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
