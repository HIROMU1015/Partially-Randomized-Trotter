"""Fresh approved COST22 and ACCOUNT stages; interrupted grants are read-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from fractions import Fraction
import gzip
import json
import math
import multiprocessing
from pathlib import Path

import pf_common_run_contract as c
from prepare_h6_pf_cost_recovery import verify_seals
from run_pf_common_run import cost_caps, best_complete

NEW = ('scripts/h6_pf_recovery_execution.py', 'scripts/run_h6_pf_recovery_pipeline.py',
       'scripts/run_detached_local_job.py', 'tests/test_h6_pf_recovery_execution.py')


def remaining(plan, prices, eligible):
    rows = [r for r in plan['rows'] if c.cid(r) in eligible and c.cid(r) not in prices]
    c.require(len(rows) == 22 and sorted(c.cid(r) for r in rows) == plan['recovery_ids'], 'EXACT22_NO_RECOMPUTE')
    c.require({c.cid(r) for r in rows} <= set(plan['compile_pool']), 'REGISTERED_POOL')
    return rows


def validate_dependencies(plan):
    inherited = Path(plan['inherited_private'])
    for stage in ('SIGNAL', 'QUALIFY'):
        proof = c.read(inherited/stage/'completion.json')
        c.require(proof['status'] == 'PASS' and proof['SOURCE'] == plan['inherited_SOURCE'], 'INHERITED_PASS:'+stage)
    c.require((inherited/'COST/STOP.json').exists() and not (inherited/'COST/completion.json').exists(), 'OLD_STOP_NOT_REWRITTEN')
    c.require(plan['approval']['additional_internal_budget_approved'] is True, 'ADDITIONAL_BUDGET_APPROVAL')
    c.require(plan['recovery_caps']['native_piece'] == 1520 and plan['recovery_caps']['wrapper_compile'] == 44, 'FIXED_EXTRA_CAPS')


def prepare(private, inactive_private):
    p, inactive_private = Path(private).resolve(), Path(inactive_private).resolve()
    c.require(not p.exists(), 'FRESH_APPROVED_PACK')
    inactive = c.read(inactive_private/'recovery_contract_UNAPPROVED.json')
    proof = c.read(inactive_private/'preparation_validation.json')
    c.require(proof['inactive_contract_sha256'] == c.sha(inactive_private/'recovery_contract_UNAPPROVED.json'), 'INACTIVE_PLAN_SHA')
    verify_seals(inactive)
    old = Path(inactive['original_private']); parent = c.read(old/'plan_APPROVED.json'); c.verify(parent)
    with gzip.open(c.ROOT/'artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/available_prices_UNACCOUNTED.json.gz', 'rt') as f:
        prices = json.load(f)
    plan = copy.deepcopy(parent)
    plan.update(SOURCE=c.source(), private_root=str(p), inherited_private=str(old), inherited_SOURCE=parent['SOURCE'],
        inactive_preparation=str(inactive_private), recovery_ids=inactive['expected22_ids'],
        recovery_caps=inactive['fresh_budget']['fresh_caps'], cumulative_budget_proposal=inactive['fresh_budget']['cumulative_comparison'],
        recovery_track=True, metadata_private=str(Path(parent['metadata_completion']).parent.parent),
        approval=dict(execution_authorized=True, no_intermediate_reapproval=True,
            additional_internal_budget_approved=True, exact_user_instruction='計算を再開して',
            authorization_context='Resume requested immediately after the concrete fresh22 internal-budget proposal and inactive specification. '
                'Accept exactly that proposal: original scientific conditions and candidate set unchanged; no prior grant retry/resume; '
                'finish fixed accounting/audit/reference integration/local commit and mandatory STOP; no push.'))
    # Preserve all old seals, but check original numerical SOURCE files against
    # the new committed SOURCE as well. New orchestration files are separate.
    plan['closure'] = list(inactive['seals'])
    for rel in NEW:
        path = c.ROOT/rel
        c.require(path.read_bytes() == c.gitbytes(plan['SOURCE'], rel), 'NEW_COMMITTED_SOURCE')
        plan['closure'].append(c.seal(path, 'SOURCE:'+rel))
    for e in plan['closure']:
        if e['role'].startswith('SOURCE:'):
            c.require(Path(e['path']).read_bytes() == c.gitbytes(plan['SOURCE'], e['role'][7:]), 'UNCHANGED_NUMERICAL_SOURCE')
    eligible = set(c.read(old/'QUALIFY/output/eligible_ids.json'))
    rows = remaining(plan, prices, eligible)
    prefixes = sorted({r['prefix'] for r in rows if r['method'] != 'FULL'})
    c.require(cost_caps(rows, prefixes) == plan['recovery_caps'], 'EXACT_PREVIOUSLY_PROPOSED_CAPS')
    validate_dependencies(plan)
    p.mkdir(parents=True)
    c.save(p/'saved_prices.json', prices)
    c.save(p/'cost_identity_audit.json', c.read(old/'cost_identity_audit.json'))
    plan['closure'].extend([c.seal(p/'saved_prices.json', 'fixed_recovery_prices'),
                            c.seal(p/'cost_identity_audit.json', 'fixed_cost_identity')])
    c.save(p/'plan_APPROVED.json', plan)
    c.save(p/'approval_receipt.json', dict(plan['approval'], SOURCE=plan['SOURCE'], plan_sha256=c.sha(p/'plan_APPROVED.json')))
    c.verify(plan)
    return dict(status='APPROVED_FRESH22_SEALED', SOURCE=plan['SOURCE'], new_grants=0,
        caps=plan['recovery_caps'], retained_complete_pairs=187, missing_pairs=22)


def run_cost(private):
    private = Path(private); plan = c.read(private/'plan_APPROVED.json'); validate_dependencies(plan)
    inherited = Path(plan['inherited_private']); prices = c.read(private/'saved_prices.json')
    eligible = set(c.read(inherited/'QUALIFY/output/eligible_ids.json'))
    rows = remaining(plan, prices, eligible)
    stage = c.Stage(private, 'COST', plan['recovery_caps'])
    try:
        c.save(stage.root/'selected_candidates.json', rows)
        import pf_common_run_native as native
        prefixes = sorted({r['prefix'] for r in rows if r['method'] != 'FULL'})
        native.STAGE = stage; native.CTX = native.restore(stage, prefixes)
        native.SELECTED = c.read(inherited/'SIGNAL/output/all_signals.json')
        obs = c.resources(); workers = c.workers(obs, len(rows), 2*2**30, 512*2**20)
        c.require(workers > 0, 'NO_IDLE_RECOVERY_RESOURCE')
        c.save(stage.root/'resource_receipt.json', dict(observation=obs, workers=workers,
            worker_RSS_proxy_not_cap=True, compiler_processes=1, BLAS_threads=1))
        results = []; errors = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = {pool.submit(native.cost_one, r):r for r in sorted(rows,
                key=lambda r:r['q']*c.PIECES[r['formula']]*(r['prefix']+1), reverse=True)}
            processes = list(pool._processes.values())
            for future in as_completed(futures):
                try:
                    result = future.result(); results.append(result)
                    print('RECOVERY_PAIR_PASS', len(results), c.cid(result['candidate']), flush=True)
                except BaseException as exc:
                    errors.append(dict(candidate=futures[future], reason=str(exc)))
                    for pending in futures: pending.cancel()
            c.require(not errors, 'RECOVERY_WORKER_FAILURE:'+str(errors))
        c.require(all(p.exitcode == 0 and not p.is_alive() for p in processes), 'OWN_EXIT_REAP')
        c.save(stage.root/'worker_exit_receipt.json', dict(workers=[dict(pid=p.pid, exit_code=p.exitcode, reaped=True) for p in processes],
            same_parent_join=True, remaining_owned_workers=0, scope='NEW22_ONLY_OLD_PARENT_REAP_UNKNOWN'))
        for r in results:
            identity = c.cid(r['candidate'])
            c.require(identity not in prices, 'NEVER_OVERWRITE_ACCEPTED_PRICE')
            prices[identity] = dict(cosine=r['prices']['cosine']['metrics']['rz_count'],
                sine=r['prices']['sine']['metrics']['rz_count'], kind='NEW_RECOVERY_NATIVE',
                result_sha256=c.sha(stage.out/identity/'result.json'))
        c.require(eligible <= set(prices), 'ALL239_ELIGIBLE_COSTS')
        c.save(stage.out/'all_prices.json', prices)
        return stage.finish(dict(new_pairs=22, new_axes=44, inherited_complete_pairs=187,
            reused_pairs=30, old_interrupted_stage_success=False, old_parent_exit_reap=None))
    except BaseException as exc:
        stage.stop(exc); raise


def account_saved(endpoints, prices, pr, take):
    endpoints = copy.deepcopy(endpoints)
    for ep in endpoints:
        take('weighted_PF_endpoint', c.cid(ep['candidate'])+'_'+str(ep['epsilon']))
        price = prices.get(c.cid(ep['candidate'])); ep['price'] = price; ep['total_RZ_exact'] = None
        if ep['status'] == 'ELIGIBLE':
            c.require(price is not None, 'NO_MISSING_ELIGIBLE_PRICE')
            total = ep['N_cos']*price['cosine']+ep['N_sin']*price['sine']
            c.require(total == int(Fraction(ep['N_cos'])*price['cosine']+Fraction(ep['N_sin'])*price['sine']), 'EXACT_PF_ACCOUNTING')
            ep['total_RZ_exact'] = str(total)
    comparisons = []
    for epsilon in c.EPS:
        subset = [e for e in endpoints if e['epsilon'] == epsilon]
        for formula in [*c.FORMULAS, None]:
            take('comparison', str(epsilon)+'_'+str(formula)); pf = best_complete(subset, formula)
            candidates = [r for r in pr if float(r['epsilon']) == epsilon and (formula is None or r['formula'] == formula) and r['status'] == 'ELIGIBLE']
            cost = lambda r:Fraction(int(r['total_RZ_numerator']), int(r['total_RZ_denominator']))
            winner = min(candidates, key=lambda r:(cost(r), r['candidate_id'])) if candidates else None
            diff = ratio = None
            if pf['best'] and winner:
                v, u = Fraction(pf['best']['total_RZ_exact']), cost(winner); diff, ratio = v-u, u/v
                c.require(math.isclose(float(ratio), float(u)/float(v), rel_tol=256*2**-52), 'RATIO_ARITHMETIC')
            comparisons.append(dict(system='H6', epsilon=epsilon, formula=formula or 'ALL_FORMULAS', PF=pf,
                PR_saved=winner, PR_over_PF_exact=None if ratio is None else str(ratio),
                PF_minus_PR_exact=None if diff is None else str(diff), PR_reduction_percent=None if ratio is None else float(100*(1-ratio)),
                point_direction='UNRESOLVED' if diff is None else 'PR_LOWER' if diff > 0 else 'PF_LOWER' if diff < 0 else 'TIE',
                exploratory_not_population_guarantee=True, independent_confirmation=False))
    return endpoints, comparisons


def run_account(private):
    private = Path(private); plan = c.read(private/'plan_APPROVED.json'); validate_dependencies(plan)
    stage = c.Stage(private, 'ACCOUNT', dict(weighted_PF_endpoint=1380, comparison=20), ['COST'])
    try:
        endpoints = c.read(Path(plan['inherited_private'])/'QUALIFY/output/endpoints.json')
        prices = c.read(private/'COST/output/all_prices.json')
        with (c.ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv').open() as f:
            pr = [r for r in csv.DictReader(f) if r['system'] == 'H6' and r['method'] == 'PR' and r['mode'] == 'USUAL_SIGNED']
        c.require(len(endpoints) == 1380 and len(pr) == 60, 'FIXED_ENDPOINT_AND_PR_COUNT')
        endpoints, comparisons = account_saved(endpoints, prices, pr, stage.take)
        c.save(stage.out/'PF_endpoint_costs.json', endpoints); c.save(stage.out/'comparisons.json', comparisons)
        c.save(stage.out/'scope_limits.json', dict(PR_new_sampling=0, PR_repricing=0, PR_statistics_recomputed=0,
            confirmation_pooling=False, finite_sample_selection_effect_uncorrected=True,
            unobserved_rare_event_cost_bound=None, empirical_allowance_not_certificate=True, full_RPE_energy_cost=False, q16_included=False))
        return stage.finish(dict(comparison_rows=20, PF_endpoints=1380))
    except BaseException as exc:
        stage.stop(exc); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('prepare', 'cost', 'account'))
    p.add_argument('--private', required=True); p.add_argument('--inactive-private'); a = p.parse_args()
    result = prepare(a.private, a.inactive_private) if a.mode == 'prepare' else globals()['run_'+a.mode](a.private)
    print(c.encoded(result).decode(), flush=True)
