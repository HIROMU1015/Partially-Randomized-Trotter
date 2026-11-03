"""Saved native32 versus fixed deterministic prices; passive on import.

Only primary_statistics/secondary_statistics do new scientific inference.
Audit/preflight read JSON and hashes, never arrays, circuits or price moments.
"""
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import time

from . import c1_stage1_execution as io

BASE = 'beb8de5cd8b71f80bb69556f92299435ff431c3e'
PR_SOURCE = '4df8eca29c59ffd55e2dd898f05da16bb246ddbe'
IDEAL_SOURCE = '2d20ad196a406752de8e4c0ce9516b101e49d927'
IDEAL_RESULT = '39ee9705b7a459c40c205c542d660b5d2ab90d33'
CANDIDATE = 'a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
OLD_SOURCE_SHA = '103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739'
PR_ROOT = 'artifacts/research_c_near_practical_r32/2026-11-03/run01'
FAMILY = 'artifacts/research_c_pr_xdf_statistics/2026-11-03'
FRESH = FAMILY + '/run01'
ENTRY = 'scripts/run_research_c_pr_xdf_statistics.py'
CONTRACT = 'docs/research/research_c_pr_xdf_cdf_statistical_comparison_contract_unapproved_20261103.json'
IDS = dict(XDF='B3_XDF_spin_relative', CDF='B2_CDF_relative')
TOTALS = dict(XDF=100267421030, CDF=102056444569)
W, SHELL = 12539808, 51583
POLICY = dict(samples=32, confidence=.95, one_sided_alpha=.025,
    precision_halfwidth_fraction_of_D=.05, primary_metric='rz_count',
    primary_lane='ideal_exact_native', skewness_abs_threshold=2.,
    maximum_delete_one_shift_fraction_of_halfwidth=.25,
    method='one_sample_approximate_Student_t_less_against_fixed_XDF',
    secondary='same_univariate_PR_interval_affine_images_no_second_test',
    old_samples_pooled=False, adaptive=False, interim_inference=False,
    selection_scope='post-hoc exploratory; existing cohort, not prospective holdout')
LIMITS = {k: 1 for k in ('saved_metadata_load', 'fixed_scope_gate',
    'saved_native32_load', 'native_row_gate', 'primary_one_sample_comparison',
    'secondary_fixed_threshold', 'result_summary')}
MAX_CALLS = sum(LIMITS.values())
SUFFIXES = ('metrics', 'events', 'raw', 'compiled', 'adjoint', 'XY')


def roles():
    paths = {f'PR_{k}': f'{PR_ROOT}/{k}.json' for k in ('result', 'contract', 'SOURCE', 'STOP', 'seed_manifest')}
    paths.update({f'ideal_{k}': f'{FAMILY}/inputs/ideal_{k}.json' for k in ('result', 'contract', 'SOURCE', 'STOP')})
    paths.update({f'PR_A_{k}': f'{PR_ROOT}/PR/A/{k}.json'
        for k in ('identity', 'normalization', 'candidate_ground', 'law')})
    paths.update({f'PR_shell_{k}': f'{PR_ROOT}/PR/C/{k}_shell.json' for k in ('head', 'X', 'Y')})
    paths['PR_journal'] = PR_ROOT + '/science_calls.jsonl'
    paths.update({label+'_packet': 'artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/candidates/'+cid+'.packet.json'
        for label,cid in IDS.items()})
    return paths


def native_paths():
    return [{k: f'{PR_ROOT}/PR/C/samples/sample{i:02d}_{k}.json' for k in SUFFIXES}
        for i in range(32)]


def fixed():
    return dict(scope='C_SAVED_PR32_FIXED_XDF_CDF_EXPLORATORY_V1', base_commit=BASE,
        status='UNAPPROVED_EXECUTABLE_TECHNICAL_PROPOSAL', entry=ENTRY,
        candidate_sha256=CANDIDATE, PR_scientific_SOURCE=PR_SOURCE,
        ideal_scientific_SOURCE=IDEAL_SOURCE, ideal_result_commit=IDEAL_RESULT,
        candidate_ids=IDS, fixed_total_RZ=TOTALS, W=W, shell_RZ=SHELL,
        statistics=POLICY, records=roles(), native_records=native_paths(),
        call_limits=LIMITS, maximum_logical_units=MAX_CALLS, fresh_output=FRESH,
        retry=0, resume=0, mandatory_stop=True,
        new_sampling=0, new_PF_RTE=0, new_shots=0, ground_eigensolve=0,
        build_compile=0, array_decode=0, optimizer=0)


def metadata(root, c):
    return {k: io.read(Path(root)/p) for k, p in c['records'].items() if k != 'PR_journal'}


def scope_gate(d):
    """Bind saved scalar records; do not derive W, costs, or statistics."""
    pr, pc, ideal, ic = (d[k] for k in ('PR_result', 'PR_contract', 'ideal_result', 'ideal_contract'))
    for prefix, units in (('PR', 1231), ('ideal', 61)):
        stop = d[prefix+'_STOP']
        if (stop['failed'] or stop['scientific_units'] != units or stop['retry'] != 0
                or stop['resume'] != 0 or not stop['mandatory_stop']):
            raise ValueError('SAVED_STOP:'+prefix)
    if (pr['status'] != 'NEAR_PRACTICAL_R32_DOWNSTREAM_COMPLETE'
            or ideal['status'] != 'COMMON_IDEAL_LANE_COMPLETE'
            or pc['candidate_sha256']['PR'] != CANDIDATE
            or pc['source_manifest_sha256'] != OLD_SOURCE_SHA
            or d['ideal_SOURCE']['execution_commit'] != IDEAL_SOURCE):
        raise ValueError('FIXED_ORIGIN')
    p, q = pc['policy'], ic['conditions']
    for a, b in (('atoms','atoms'), ('distance_A','distance_angstrom'), ('basis','basis'),
            ('Nalpha','Nalpha'), ('Nbeta','Nbeta'), ('sector_dimension','sector_dimension'),
            ('oracle','oracle'), ('state_preparation','state_preparation'), ('rounds','rounds')):
        if p[a] != q[b]: raise ValueError('COMMON_SCOPE:'+a)
    if (p['rank'] != 8 or q['CDF_rank'] != 8 or q['XDF_rank'] != 8
            or p['ld'] != 2 or p['r'] != 32 or p['K'] != 0 or p['s'] != 1
            or p['h'] != '1/10' or q['base_time'] != '1/10' or q['substeps'] != [1]
            or pc['compiler'] != ic['compiler'] or pc['target'] != ic['target']
            or pc['gaussian'] != ic['gaussian'] or pc['compiler_phase'] != ic['compiler_phase']):
        raise ValueError('LANE_COMPILER_REPRESENTATION')
    a = pr['arms']['PR']; v = a['PR']; binding = a['bindings']['ideal_exact_native']
    lane = v['lanes']['ideal_exact_native']; cal = v['calibration']
    if (a['candidate'] != CANDIDATE or not v['native_admitted'] or v['r'] != 32
            or binding['W'] != W or binding['shell']['rz_count'] != SHELL
            or binding['shots'] != 14738 or lane['total_shots'] != 14738
            or lane['weighted_SE']['kernel_reuse_weight'] != W
            or lane['conditional_totals']['totals']['rz_count'] != 94288043542
            or cal['samples'] != 32 or cal['sample_mean']['rz_count'] != 7519.09375
            or cal['sample_mean_standard_error']['rz_count'] != 107.32188319090913
            or v['lanes']['evidence']['conditional_totals']['totals'] is not None
            or v['actual_CA_achieved'] is not None or v['implemented_law_error'] is not None):
        raise ValueError('PR_FIXED_ACCOUNTING')
    for label, cid in IDS.items():
        v = ideal['candidates'][cid]; packet = d[label+'_packet']
        if (ic['candidate_identities'][label]['id'] != cid or v['id'] != cid
                or v['arm'] != label or v['lane'] != 'ideal_exact_native'
                or v['status'] != 'CONDITIONAL_IDEAL_CA_PASS'
                or v['cost']['totals']['rz_count'] != TOTALS[label]
                or not v['energy']['same_accuracy'] or v['actual_CA_achieved'] is not None
                or [x['m'] for x in v['rounds']] != list(range(13))):
            raise ValueError('FIXED_DETERMINISTIC:'+label)
        if (packet['id'] != cid or packet['s'] != 1 or not packet['alias_verified']
                or packet['state_model'] != 'own_DF_ground_state_oracle'
                or packet['kernel_metrics'] != v['kernel_metrics']
                or packet['shell_metrics'] != v['shell_metrics']
                or packet['df_error_with_guard'] != v['saved_DF_error_with_guard']
                or [p['pf_operator_error'] for p in packet['PF_rounds']]
                    != [p['raw_signal_error'] for p in v['rounds']]):
            raise ValueError('RETAINED_PF_DF_NATIVE:'+label)
    if d['PR_A_identity']['candidate'] != CANDIDATE or d['PR_A_identity']['status'] != 'PASS':
        raise ValueError('CANDIDATE_IDENTITY')
    if any(d['PR_A_'+k]['status'] != 'PASS' for k in ('normalization','candidate_ground','law')):
        raise ValueError('SAVED_PREPARATION_GATE')
    return dict(status='PASS', primary=IDS['XDF'], secondary=IDS['CDF'],
        saved_scalars_quoted_only=True, native_statistics_recomputed=False,
        distinct_DF_representations=True, actual_CA_achieved=None,
        ideal_law_cost_transfer_qualified=False)


def load_native(root, c):
    return [{k: io.read(Path(root)/p) for k, p in paths.items()} for paths in c['native_records']]


def native_gate(d, packets, journal):
    """Structural/native-correctness audit; no moment, histogram or inference."""
    from . import c_near_practical_r32 as old
    from .c_r32_price_statistics import validate_rows
    expected = old.seed_manifest(OLD_SOURCE_SHA)
    if d['PR_seed_manifest'] != expected: raise ValueError('SEED_COHORT_MANIFEST')
    if len(packets) != 32: raise ValueError('EXACTLY_32_NO_EXCLUSIONS')
    fingerprints = []
    def gate_tree(value):
        if isinstance(value, dict):
            if 'max_entry' in value:
                if (value.get('rtol') != 0 or value.get('tolerance') != 1e-10
                        or not math.isfinite(value['max_entry']) or not 0 <= value['max_entry'] <= 1e-10):
                    raise ValueError('SAVED_NATIVE_CORRECTNESS')
            for child in value.values(): gate_tree(child)
        elif isinstance(value, list):
            for child in value: gate_tree(child)
    for i, row in enumerate(packets):
        ev = row['events']; xy = row['XY']
        if (ev['sample'] != i or ev['sampling'] != old.sampling('PR') or ev['JU'] != CANDIDATE
                or ev['SOURCE'] != OLD_SOURCE_SHA or len(ev['events']) != 32
                or any(x['taylor_order'] != 0 for x in ev['events'])
                or not ev['coherent_occurrences_independent'] or ev['quantum_shots'] != 0
                or ev['old_cohort_pooled']):
            raise ValueError('SAMPLE_IDENTITY_COHORT_OR_COMPLETE_KERNEL')
        # Same realized event sequence may be legitimate: retain and report it.
        fingerprints.append(json.dumps(ev['events'], sort_keys=True, separators=(',',':')))
        if (set(row['raw']) != {'reference','unitarity'}
                or set(row['compiled']) != {'reference','raw','unitarity'}
                or set(row['adjoint']) != {'raw','compiled'}
                or any(set(row['adjoint'][k]) != {'reference','returned'} for k in ('raw','compiled'))):
            raise ValueError('MISSING_NATIVE_GATE_RECORD')
        for k in ('raw','compiled','adjoint'): gate_tree(row[k])
        if (xy['status'] != 'PASS' or xy['attenuation_applied']
                or xy['state'] != 'saved own-DF ground oracle; preparation excluded'):
            raise ValueError('XY_SAVED_GATE')
        for kind in ('raw','compiled'):
            signal = xy['signals'][kind]
            if (signal['tolerance'] != 2e-10 or not math.isfinite(signal['error'])
                    or not 0 <= signal['error'] <= 2e-10
                    or any(not math.isfinite(x) or not 0 <= x <= 1e-10 for x in xy['norm_errors'][kind])):
                raise ValueError('XY_SAVED_CORRECTNESS')
    rows = [p['metrics'] for p in packets]; validate_rows(rows, 32)
    for key in ('head', 'X', 'Y'): gate_tree(d['PR_shell_'+key])
    events = [json.loads(line) for line in journal.splitlines()]
    for name in ('trajectory_sample','raw_operator_gate','compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics'):
        selected = [x for x in events if x['kind'] == 'PR.C.'+name]
        if (Counter(x['event'] for x in selected) != {'attempt':32,'COMPLETE':32}
                or any(Counter(x['label'] for x in selected if x['event']==event)
                    != {f'sample{i:02d}':1 for i in range(32)} for event in ('attempt','COMPLETE'))):
            raise ValueError('NO_RETRY_EXCLUSION_OR_MISSING_NATIVE:'+name)
    return dict(status='PASS', samples=32, complete_kernels=32, microevents_per_kernel=32,
        duplicate_realized_event_sequences=32-len(set(fingerprints)), dropped=0,
        reweighting=0, resampling=0, new_PRNG_draws=0, new_build_compile=0,
        seed_keys=32, same_observations_reused_for_both_fixed_thresholds=True,
        independence='separate hash-derived PCG64 streams; IID model assumption, not proof',
        support_tail_coverage_qualified=False, ideal_law_transfer_qualified=False)


def primary_statistics(rows, *, weight=W, shell=SHELL, xdf=TOTALS['XDF'], policy=None):
    """Scientific entry: real rows require separate user execution permission."""
    from scipy.stats import ttest_1samp
    from .c_r32_price_statistics import price_statistics
    policy = POLICY if policy is None else policy
    if policy['confidence'] != .95 or policy['one_sided_alpha'] != .025:
        raise ValueError('CI_TEST_LEVEL_CORRESPONDENCE')
    value = price_statistics(rows, dict(W=weight, shell={'rz_count':shell}, D={'rz_count':xdf}),
        expected_n=policy['samples'], statistics=policy)
    value.update(selection_scope=POLICY['selection_scope'], confirmatory=False,
        primary='XDF', one_sided_alpha=.025, hypothesis='H0:mu_PR>=C_XDF;H1:mu_PR<C_XDF',
        one_sided_test=None, test_interval_consistent=None,
        one_sided_upper_confidence=.975, one_sided_upper_bound=None)
    decision = value['decision']
    if value['kernel_SE'] > 0:
        test = ttest_1samp([r['rz_count'] for r in rows], (xdf-shell)/weight,
            alternative='less', nan_policy='raise')
        p, statistic = float(test.pvalue), float(test.statistic)
        if not all(math.isfinite(x) for x in (p,statistic)): raise ValueError('NONFINITE_T_TEST')
        reject = p < .025; upper = value['total_interval'][1]
        consistent = reject == (upper < xdf)
        value.update(one_sided_test=dict(statistic=statistic,pvalue=p,df=len(rows)-1,
            alternative='less',raw_model_reject=reject), test_interval_consistent=consistent,
            one_sided_upper_bound=upper)
        if not consistent: decision = 'UNRESOLVED_TEST_INTERVAL_NUMERICAL_BOUNDARY'
    value['decision'] = 'POSTHOC_EXPLORATORY_'+decision
    return value


def secondary_statistics(primary, *, cdf=TOTALS['CDF'], xdf=TOTALS['XDF']):
    if not (math.isfinite(cdf) and cdf > xdf > 0): raise ValueError('NESTED_FIXED_THRESHOLDS')
    interval = primary['total_interval']; decision = 'UNRESOLVED_PRIMARY_QUALITY_OR_SIGN'
    quality = (primary['precision_met'] and interval is not None
        and primary['test_interval_consistent'] is True
        and not primary['diagnostics']['influence_warning']
        and not primary['diagnostics']['skewness_warning'])
    if quality and interval[1] < cdf: decision = 'SHARED_BOUND_PR_LOWER'
    elif quality and interval[0] > cdf: decision = 'SHARED_BOUND_PR_HIGHER'
    return dict(secondary='CDF', method=POLICY['secondary'], additional_test_count=0,
        pvalue=None, difference_point=primary['total_point']-cdf,
        ratio_point=primary['total_point']/cdf,
        difference_interval=None if interval is None else [x-cdf for x in interval],
        ratio_interval=None if interval is None else [x/cdf for x in interval],
        primary_lower_implies_secondary_lower=primary['decision']=='POSTHOC_EXPLORATORY_NOMINAL_MODEL_PR_LOWER',
        decision='POSTHOC_EXPLORATORY_'+decision, confirmatory=False,
        actual_CA_achieved=None, guaranteed_resource_superiority=None,
        ideal_law_cost_transfer_qualified=False, equivalence=None, noninferiority=None)


def preflight(root, path):
    root = Path(root).resolve(); c = io.read(path)
    for key, value in fixed().items():
        if c[key] != value: raise ValueError('FIXED_SCOPE:'+key)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    needed = set(roles().values()) | {p for row in native_paths() for p in row.values()}
    old = io.read(root/roles()['PR_contract'])
    identity_inputs = {old['endpoints']['PR']['candidate']: CANDIDATE,
        old['input']: '1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'}
    needed.update(identity_inputs)
    if not needed <= c['saved_inputs'].keys() or ENTRY not in source['files']:
        raise ValueError('UNPINNED_INPUT_OR_ENTRY')
    if any(c['saved_inputs'][p] != digest for p,digest in identity_inputs.items()):
        raise ValueError('CANDIDATE_HAMILTONIAN_INPUT')
    for group in (source['files'], c['saved_inputs']):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != digest: raise ValueError('SOURCE_INPUT:'+name)
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('ENVIRONMENT:'+name)
    d = metadata(root,c); scope = scope_gate(d)
    audit = native_gate(d,load_native(root,c),(root/c['records']['PR_journal']).read_text())
    if (root/FRESH).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    return c, dict(status='PASSIVE_READY', scientific_calls=0, new_real_statistics=0,
        maximum_logical_units=MAX_CALLS, scope=scope, native_audit=audit)


def run(root, path, *, execute_science=False):
    if not execute_science: raise PermissionError('SEPARATE_USER_SCIENTIFIC_PERMISSION_REQUIRED')
    # The opt-in flag is a caller boundary, not authentication/proof of permission.
    from .c1_stage3_integrated import Calls
    root = Path(root).resolve(); c, _ = preflight(root,path)
    out = root/FRESH; out.mkdir(parents=True,exist_ok=False)
    call = Calls(out,LIMITS,MAX_CALLS); start = time.monotonic(); completed = False
    try:
        io.save(out/'contract.json',c); io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        d = call('saved_metadata_load','fixed saved metadata only',lambda:metadata(root,c))
        io.save(out/'scope.json',call('fixed_scope_gate','common ideal lane',lambda:scope_gate(d)))
        packets = call('saved_native32_load','reuse 32 saved kernels, no new samples',lambda:load_native(root,c))
        io.save(out/'audit.json',call('native_row_gate','no exclusion or reweighting',lambda:native_gate(
            d,packets,(root/c['records']['PR_journal']).read_text())))
        primary = call('primary_one_sample_comparison','single PR mean, fixed XDF',lambda:primary_statistics([p['metrics'] for p in packets]))
        io.save(out/'primary.json',primary)
        secondary = call('secondary_fixed_threshold','same PR interval, no second test',lambda:secondary_statistics(primary))
        io.save(out/'secondary.json',secondary)
        result = call('result_summary','conditional exploratory inference only',lambda:dict(
            status='EXPLORATORY_DIRECT_COMPARISON_RECORDED',primary=primary,secondary=secondary,
            source_manifest_sha256=c['source_manifest_sha256'], PR_scientific_SOURCE=PR_SOURCE,
            ideal_scientific_SOURCE=IDEAL_SOURCE,new_native_sampling=0, actual_CA_achieved=None,
            guaranteed_resource_superiority=None,equivalence=None,noninferiority=None,mandatory_stop=True))
        io.save(out/'result.json',result)
        completed = True
        return result
    except BaseException as exc:
        io.save(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),first_failure=call.first_failure))
        raise
    finally:
        io.save(out/'STOP.json',dict(failed=not completed or call.stop.is_set(),
            scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),
            maximum_logical_units=MAX_CALLS,first_failure=call.first_failure,
            wall_seconds=time.monotonic()-start,retry=0,resume=0,mandatory_stop=True))
