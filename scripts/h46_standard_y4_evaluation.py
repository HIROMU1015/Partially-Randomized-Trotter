"""Fixed .001 saved-shot accounting, using frozen paired statistics unchanged."""
import csv
from pathlib import Path
from statistics import NormalDist
import h46_standard_y4_contract as c


def validate_pairs(rows, *, system, candidate, n, source):
    c.require(len(rows) == n and sorted(int(r['sample']) for r in rows) == list(range(n)),
        'EXACT_N_UNIQUE_SAMPLES_NO_TOPUP')
    c.require(all(r['system'] == system and r['candidate_id'] == candidate and
        r['SOURCE'] == source and r['cohort'] == 'CONFIRMATION' and
        r['stratum'] in ('S0', 'S2_single', 'S2_multi', 'S4') and
        r['conditional'] is False and r['paired'] is True and
        r['axes'] == ['cosine', 'sine'] for r in rows), 'NO_COHORT_CANDIDATE_AXIS_MIXING')
    pairs = [r['pair'] for r in rows]
    c.require(all(len(p) == 2 and all(type(x) is int and x > 0 for x in p) for p in pairs),
        'POSITIVE_INTEGER_AXIS_PAIRS')
    return pairs


def compare_statistics(left, right, *, z):
    from h246_confirmation_design import classify_confirmation
    result = classify_confirmation(left, right, z=z)
    result.update(ratio_Y4_over_S2=right['mean'] / left['mean'], ratio_interval=None,
        decision='INCONCLUSIVE' if result['approximate_direction'] == 'UNRESOLVED' else
            'REPRODUCED_APPROXIMATE' if result['approximate_direction'] == 'POSITIVE' else
            'REVERSED_APPROXIMATE',
        all_zero_sample_variance_warning=left['zero_observed_variance'] or right['zero_observed_variance'])
    return result


def load_bundle(package, configs):
    a = Path(package['assets'])
    with (a / 'fixed/fixed_evaluation/paired_samples_118.csv').open() as f:
        saved = list(csv.DictReader(f))
    bundle = {}
    joins = {}
    for cfg in configs:
        s = cfg['stage'][:2]
        g = c.GROUPS[s]
        old = [r for r in saved if r['system'] == s and r['candidate_id'] == g['S2']]
        rows_s2 = [dict(system=s, candidate_id=g['S2'], sample=int(r['sample']),
            SOURCE=r['SOURCE'], cohort='CONFIRMATION', stratum=r['stratum'],
            conditional=False, paired=True, axes=['cosine', 'sine'],
            pair=[int(r['cosine_RZ']), int(r['sine_RZ'])]) for r in old]
        root = Path(cfg['run_root'])
        end = c.read(root / 'end_audit.json')
        ctx = c.read(root / 'context_end_audit.json')
        c.require(end['status'] == 'ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP' and
            not end['errors'] and end['SOURCE'] == cfg['SOURCE'] and end['plan_sha256'] ==
            cfg['plan_sha256'] and end['completed_trajectories'] == g['n'] and
            end['accepted_compile_axes'] == 2 * g['n'] and
            ctx['status'] == 'CONTEXT_AND_ACQUISITION_AUDIT_PASS' and not ctx['errors'] and
            ctx['end_audit_sha256'] == c.sha(root / 'end_audit.json'),
            'FULL_NATIVE_CONTEXT_END_AUDIT_REQUIRED')
        accepted = {r['id']: r['result_sha256'] for r in end['results']}
        c.require(set(accepted) == {t['id'] for t in cfg['tasks']}, 'EXACT_NEW_RESULT_SET')
        rows_y4 = []
        for t in cfg['tasks']:
            path = Path(cfg['output']) / t['id'] / 'result.json'
            r = c.read(path)
            c.require(c.sha(path) == accepted[t['id']] and r['task'] == t and
                r['SOURCE'] == cfg['SOURCE'] and r['plan_sha256'] == cfg['plan_sha256'] and
                r['status'] == 'UNIT_COST_PASS_NOT_ACCOUNTING' and r['seed_cohort'] ==
                'CONFIRMATION' and r['paired_trajectory'] and not r['conditional_sampling'] and
                not r['cohort_pooling'] and [x['axis'] for x in r['axis_costs']] ==
                ['cosine', 'sine'], 'NEW_RESULT_IDENTITY_AND_PAIRING')
            rows_y4.append(dict(system=s, candidate_id=g['Y4'], sample=t['sample'],
                SOURCE=r['SOURCE'], cohort=r['seed_cohort'], stratum=r['events']['stratum'],
                conditional=r['conditional_sampling'], paired=r['paired_trajectory'],
                axes=[x['axis'] for x in r['axis_costs']],
                pair=[x['metrics']['rz_count'] for x in r['axis_costs']]))
        for kind, rows, n, source in [('S2', rows_s2, g['n_s2'], c.OLD_SOURCE),
                ('Y4', rows_y4, g['n'], cfg['SOURCE'])]:
            validate_pairs(rows, system=s, candidate=g[kind], n=n, source=source)
        bundle[s] = dict(S2=rows_s2, Y4=rows_y4)
        joins[s] = dict(end_audit_sha256=c.sha(root / 'end_audit.json'),
            context_audit_sha256=c.sha(root / 'context_end_audit.json'),
            S2_saved_pairs_sha256=c.sha(a / 'fixed/fixed_evaluation/paired_samples_118.csv'))
    return bundle, joins


def evaluate(package_path, config_paths):
    package = c.verify_package(package_path)
    configs = [c.read(p) for p in config_paths]
    c.require(len(configs) == 2 and {v['stage'] for v in configs} == set(c.STAGES),
        'TWO_FIXED_NATIVE_STAGES')
    for cfg in configs:
        c.authorize(cfg)
        c.consumed(cfg)
        c.require(cfg['package_path'] == str(Path(package_path).resolve()), 'SAME_PACKAGE')
    c.require(len({cfg['approval_receipt_sha256'] for cfg in configs}) == 1,
        'ONE_COMBINED_EXPLICIT_APPROVAL')
    root = Path(package['evaluation_root'])
    c.require(not root.exists(), 'NO_EVALUATION_RETRY_OR_USED_GRANT')
    root.mkdir(exist_ok=False)
    ledger = root / 'ledger'
    ledger.mkdir()
    grant = dict(SOURCE=package['SOURCE'], package_sha256=c.sha(package_path),
        approval_sha256=configs[0]['approval_receipt_sha256'],
        call_scope=c.evaluation_scope(), retry=False, next_phase_authorized=False)
    c.save(root / 'CONSUMED.json', dict(state='CONSUMED_NO_RETRY_NO_REFUND', grant=grant))
    counts = {k: 0 for k in c.evaluation_scope()}

    def call(kind, fn, *args, **kwargs):
        i = counts[kind]
        c.require(i < c.evaluation_scope()[kind], 'EVALUATION_CALL_BUDGET:' + kind)
        c.save(ledger / (kind + '__' + str(i) + '.json'),
            dict(kind=kind, index=i, state='RESERVED_NONREFUNDABLE', **grant))
        counts[kind] += 1
        return fn(*args, **kwargs)

    try:
        call('bootstrap', lambda: None)
        bundle, joins = call('bundle_load', load_bundle, package, configs)
        z = call('family_constant', lambda: NormalDist().inv_cdf(1 - .05 / (2 * 2)))
        from h246_confirmation_evaluation import paired_statistics
        results = {}
        for s, g in c.GROUPS.items():
            groups = {}
            for kind, n, source in [('S2', g['n_s2'], c.OLD_SOURCE), ('Y4', g['n'], package['SOURCE'])]:
                rows = bundle[s][kind]
                pairs = validate_pairs(rows, system=s, candidate=g[kind], n=n, source=source)
                stats = call('paired_statistics', paired_statistics, pairs, *g['N_' + kind])
                strata = {name: sum(r['stratum'] == name for r in rows) for name in c.law.STRATA}
                groups[kind] = dict(candidate=g[kind], formula='2nd' if kind == 'S2' else '4th',
                    q=8 if kind == 'S2' else 4, cohort='CONFIRMATION',
                    statistics=stats, N_cos=g['N_' + kind][0], N_sin=g['N_' + kind][1],
                    observed_strata=strata, unobserved_strata=[k for k, n in strata.items() if not n],
                    unseen_costs=None, unseen_tail_bound=None)
            comparison = call('direction_comparison', compare_statistics,
                groups['S2']['statistics'], groups['Y4']['statistics'], z=z)
            results[s] = dict(groups=groups, comparison=comparison)
        # Literal saved Full and exploration endpoints; no additional accounting
        # or ranking is performed here. Their results remain a separate cohort.
        with (c.ROOT / 'artifacts/resource_applicability/track_a_h246_reversal_batch_run01/2026-10-24/primary_endpoints_60.csv').open() as f:
            quotes = [r for r in csv.DictReader(f) if r['system'] in c.GROUPS and
                r['epsilon'] == '0.001']
        expected = {kind + '__' + str(i) + '.json' for kind, n in c.evaluation_scope().items()
                    for i in range(n)}
        c.require(counts == c.evaluation_scope() and {p.name for p in ledger.glob('*.json')}
            == expected, 'EXACT_NINE_EVALUATION_EVENTS')
        c.verify_package(package_path)
        result = dict(status='FIXED_STANDARD_Y4_CONFIRMATION_COMPLETE_MANDATORY_STOP',
            SOURCE=package['SOURCE'], package_sha256=c.sha(package_path),
            input_manifest_sha256=package['input_manifest_sha256'], results=results,
            quoted_saved_endpoints=quotes, quoted_SOURCE=c.BASE, joins=joins,
            policy=c.analysis_scope(), z=z, ledger_events=counts, new_shots=0,
            ratio_interval=None, rare_tail_bound=None, strict_bias_certificate=None,
            cohort_pooling=False, automatic_top_up=False, retry=False, mandatory_STOP=True)
        c.save(root / 'results.json', result)
        c.save(root / 'STOP.json', dict(status='MANDATORY_STOP', SOURCE=package['SOURCE'],
            results_sha256=c.sha(root / 'results.json'), next_phase_authorized=False))
        return result
    except BaseException as exc:
        c.save(root / 'STOP.json', dict(status='STOP', reason=str(exc), retry=False,
            next_phase_authorized=False, mandatory_STOP=True))
        raise
