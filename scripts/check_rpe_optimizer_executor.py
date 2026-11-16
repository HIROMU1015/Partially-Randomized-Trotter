"""Executor acceptance through the real entry (run_rpe_method_optimizer.py --round ... --run-dir ...).

Artificial worker routes (labelled ARTIFICIAL) and saved results only; the real routes are used only
in a budget-0 round, where no job may be launched.  Checks:
  request -> execute -> ingest -> next      saved reuse                budget 0 launches nothing
  budget exhausted -> resume (no relaunch)  idempotent rerun           running job not launched twice
  dead job: reported, then collected        UNKNOWN distinct from model failure
  incumbent scope CONFIGURATION (PR-Pauli saved configurations, artificial worker, UNLIMITED approved round):
  a more expensive configuration still descends to its own end; lookahead/concurrency gives the same
  per-configuration trajectories as one-at-a-time; a rerun waits for nothing and launches nothing.
  interrupted record (search-policy change): no launch, stop label, not-adopted / REQUEST-only jobs listed; resume reuses a
  REQUEST-only directory; a V2 next round loads the prior round's jobs (and refuses a different route); V2 artificial round.
Writes EXECUTOR_ACCEPTANCE.json into --output.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O

PROBLEM = ROOT / 'configs/rpe_method_optimizer/h4_df10_problem.json'
SEED = 'B0I_P7_S2_16h_over_7__B_PRICE'
P7_S2 = dict(formulas=['2nd'], delta_over_h_range=['16/7', '16/7'])


def cli(run_dir, round_file, background=False):
    cmd = [sys.executable, '-B', str(ROOT / 'scripts/run_rpe_method_optimizer.py'), '--problem', str(PROBLEM),
           '--round', str(round_file), '--run-dir', str(run_dir)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))
    if background:
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return O.load_json(run_dir / 'ROUND_RESULT.json')['methods']


def launches(run_dir):
    return {str(p.parent.name): p.stat().st_mtime_ns for p in (run_dir / 'jobs').glob('*/*/LAUNCH.json')}


def artificial_route(problem, **extra):
    tau = O.tau(problem);weights = [(m + 1) ** 2 for m in range(13)]
    seed = [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]
    scale = tau * Fraction(3, 5) / sum(Fraction(w, n) for w, n in zip(weights, seed))
    return dict(kind='artificial', interpreter=sys.executable, configuration_ids='*', view_commit=None,
                artificial=dict(weights=weights, scale=str(scale), tau=str(tau), **extra), existing_caller='ARTIFICIAL')


def write_round(path, round_id, budget, route=None, methods=('TRUNCATED_PF',), scope=True, initial=True):
    spec = dict(schema='RPE_OPTIMIZER_SEARCH_ROUND_V1', round_id=round_id, methods=list(methods),
                max_new_q3_per_method={m: budget for m in methods}, approved_by='ACCEPTANCE_TEST_ARTIFICIAL')
    if initial:
        spec['initial'] = {'TRUNCATED_PF': [SEED]}
    if scope:
        spec['method_overrides'] = {'TRUNCATED_PF': P7_S2}
    if route is not None:
        spec['routes_override'] = {m: [route] for m in methods}
    path.write_text(json.dumps(spec, indent=1) + '\n')
    return path


def pauli_round(path, round_id, route, lookahead, concurrent, scope='CONFIGURATION'):
    spec = dict(schema='RPE_OPTIMIZER_SEARCH_ROUND_V1', round_id=round_id, methods=['PR_PAULI'], science_approved=True,
                max_new_q3_per_method={'PR_PAULI': 'UNLIMITED'}, approved_by='ACCEPTANCE_TEST_ARTIFICIAL',
                method_overrides={'PR_PAULI': dict(saved_configurations_only=True,
                                                   generation=dict(formulas=['2nd'], delta_over_h_values=['1'], kappa_multipliers=['1']))},
                search_overrides=dict(incumbent_scope=scope, seed_saved_same_L_D=True, lookahead=lookahead),
                executor=dict(max_concurrent_jobs=concurrent, wait_for_running_jobs=True), routes_override={'PR_PAULI': [route]})
    path.write_text(json.dumps(spec, indent=1) + '\n')
    return path


def cli_interrupted(run_dir, round_file, stop_file):
    cmd = [sys.executable, '-B', str(ROOT / 'scripts/run_rpe_method_optimizer.py'), '--problem', str(PROBLEM),
           '--round', str(round_file), '--run-dir', str(run_dir), '--interrupted', str(stop_file)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return O.load_json(run_dir / 'ROUND_RESULT_INTERRUPTED.json')['methods']


def pauli_budget_round(path, round_id, route, budget, rule=None, prior=None, approved=True, scope='CONFIGURATION'):
    search = dict(incumbent_scope=scope, seed_saved_same_L_D=True, lookahead=1)
    if rule:
        search = dict(rule=rule, seed_saved_same_L_D=True)
    spec = dict(schema='RPE_OPTIMIZER_SEARCH_ROUND_V1', round_id=round_id, methods=['PR_PAULI'], science_approved=approved,
                max_new_q3_per_method={'PR_PAULI': budget}, approved_by='ACCEPTANCE_TEST_ARTIFICIAL',
                method_overrides={'PR_PAULI': dict(saved_configurations_only=True,
                                                   generation=dict(formulas=['2nd'], delta_over_h_values=['1'], kappa_multipliers=['1']))},
                search_overrides=search, executor=dict(max_concurrent_jobs=1, wait_for_running_jobs=True,
                                                       **({'prior_run_dirs': [str(prior)]} if prior else {})),
                routes_override={'PR_PAULI': [route]})
    path.write_text(json.dumps(spec, indent=1) + '\n')
    return path


def check(name, ok, **detail):
    return dict(check=name, status='PASS' if ok else 'FAIL', **detail)


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output;out.mkdir(parents=True, exist_ok=False)
    problem = O.load_json(PROBLEM);art = artificial_route(problem);checks = []

    # Real routes, budget 0: nothing is launched.
    r0 = out / 'real_budget0';rf = write_round(out / 'real_budget0.json', 'REAL_BUDGET_0', 0,
                                               methods=('FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF'), scope=False, initial=False)
    res = cli(r0, rf)
    checks.append(check('budget0_real_routes_launch_nothing', not launches(r0) and all(
        r['stop_reason'] == 'NEW_SCIENCE_NOT_APPROVED' and r['pending_request'] for r in res.values()),
        stops={m: r['stop_reason'] for m, r in res.items()}))

    # Budget 2 with the artificial worker: saved answers first, then request -> execute -> ingest -> next.
    r1 = out / 'artificial_round';f1 = write_round(out / 'artificial_round.json', 'ART_RESUME', 2, art)
    a = cli(r1, f1)['TRUNCATED_PF'];first = launches(r1)
    saved = [t for t in a['trace'] if t['answer_source'] == 'SAVED'];jobs = [t for t in a['trace'] if t['answer_source'] == 'ARTIFICIAL_JOB']
    nxt = all(t2['origin'].get('base_vector') in (None, t1['vector']) or not t1['improved']
              for t1, t2 in zip(a['trace'], a['trace'][1:]))
    checks += [
        check('saved_results_reused_without_budget', saved and all(t['calls'] == 0 for t in saved), saved_steps=len(saved)),
        check('request_execute_ingest_next', len(jobs) == 2 and len(first) == 2 and all(t['job'] for t in jobs) and nxt
              and a['pending_request'] is not None, job_steps=len(jobs), statuses=[t['status'] for t in jobs]),
        check('budget_exhausted_stop', a['stop_reason'] == 'Q3_BUDGET_EXHAUSTED' and a['new_calls'] == 2, stop=a['stop_reason']),
        check('scope_kept_to_final_request', all(':a6e55777' in t['candidate'] for t in a['trace'])
              and ':a6e55777' in a['pending_request']['candidate'], pending=a['pending_request']['candidate'])]

    # Resume after a raised budget (same round): completed jobs reused, not relaunched.
    write_round(f1, 'ART_RESUME', 4, art)
    b = cli(r1, f1)['TRUNCATED_PF'];second = launches(r1)
    reused = [t for t in b['trace'] if t['answer_source'] == 'ARTIFICIAL_JOB' and t['calls'] == 0]
    checks.append(check('resume_after_budget_end_without_relaunch', len(second) == 4 and all(second[k] == v for k, v in first.items())
                        and len(reused) == 2 and b['new_calls'] == 2 and b['budget']['spent_before'] == 2,
                        launched=len(second), reused=len(reused), new_calls=b['new_calls']))
    c = cli(r1, f1)['TRUNCATED_PF']
    checks.append(check('idempotent_rerun_launches_nothing', launches(r1) == second and c['new_calls'] == 0, new_calls=c['new_calls']))

    # A running job is never launched twice.
    r2 = out / 'running';f2 = write_round(out / 'running.json', 'ART_RUNNING', 1, artificial_route(problem, sleep_seconds=20))
    bg = cli(r2, f2, background=True)
    for _ in range(600):
        if launches(r2):
            break
        time.sleep(0.1)
    mid = cli(r2, f2)['TRUNCATED_PF'];during = launches(r2);bg.wait()
    after = cli(r2, f2)['TRUNCATED_PF']
    checks.append(check('running_job_not_launched_twice', mid['stop_reason'] == 'JOB_IN_PROGRESS' and len(during) == 1
                        and launches(r2) == during and any(t['answer_source'] == 'ARTIFICIAL_JOB' for t in after['trace']),
                        during=mid['stop_reason'], after=after['stop_reason']))

    # Dead job without worker artifact: reported for review; after the artifact appears it is collected (no recompute).
    r3 = out / 'stale';f3 = write_round(out / 'stale.json', 'ART_STALE', 0, art)
    pend = cli(r3, f3)['TRUNCATED_PF']['pending_request'];write_round(f3, 'ART_STALE', 1, art)
    ex = O.Executor(ROOT, r3, {'TRUNCATED_PF': [art]}, {}, problem).for_method('TRUNCATED_PF')
    prop = dict(configuration_id=pend['configuration_id'], vector=tuple(pend['vector']), cost=O.frac(pend['cost']), origin=pend['origin'])
    job = ex.job_dir(prop);job.mkdir(parents=True)
    (job / 'REQUEST.json').write_text(json.dumps(O.jsonable(ex.request(prop, job, art))))
    dead = subprocess.Popen([sys.executable, '-c', 'pass']);dead.wait()
    (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=dead.pid, started=time.time(), kind='artificial')))
    s1 = cli(r3, f3)['TRUNCATED_PF']
    tau = O.tau(problem);value = O.enc(tau * Fraction(1, 2))
    (job / 'ARTIFICIAL_RESULT.json').write_text(json.dumps(dict(status=O.PASS, MSE_lower=value, MSE_upper=value, scientific_evaluation_calls=1)))
    s2 = cli(r3, f3)['TRUNCATED_PF']
    collected = [t for t in s2['trace'] if t['job'] and t['calls'] == 0]
    checks.append(check('dead_job_reported_then_collected', s1['stop_reason'] == 'STALE_JOB_NEEDS_REVIEW' and collected
                        and (job / 'OUTCOME.json').exists(), first=s1['stop_reason'], second=s2['stop_reason']))

    # UNKNOWN (gate) stops; a model failure would continue.
    r4 = out / 'unknown';f4 = write_round(out / 'unknown.json', 'ART_UNKNOWN', 2, artificial_route(problem, force_unknown=True))
    u = cli(r4, f4)['TRUNCATED_PF']
    checks.append(check('unknown_distinct_from_model_failure', u['stop_reason'] == 'UNKNOWN_RESULT' and len(launches(r4)) == 1
                        and u['held'] and u['held'][0]['status'] == 'UNKNOWN', stop=u['stop_reason']))

    # Incumbent scope CONFIGURATION: artificial MSE scaled so that LD5's saved incumbent sits at tau/2.
    tau = O.tau(problem);w = [(m + 1) ** 2 for m in range(13)];ld5 = [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2]
    pr = artificial_route(problem);pr['artificial']['scale'] = str(tau * Fraction(1, 2) / sum(Fraction(a, n) for a, n in zip(w, ld5)))
    seq = cli(out / 'scope_seq', pauli_round(out / 'scope_seq.json', 'ART_SCOPE', pr, 1, 1))['PR_PAULI']
    par_round = pauli_round(out / 'scope_par.json', 'ART_SCOPE', pr, 3, 8)
    par = cli(out / 'scope_par', par_round)['PR_PAULI']
    def path(r):
        return {b['configuration_id']: [(t['vector'], t['status']) for t in r['trace'] if t['configuration_id'] == b['configuration_id']]
                for b in r['by_configuration']}
    by = {b['configuration_id']: b for b in par['by_configuration']}
    cheapest = min(O.frac(b['best']['cost']) for b in by.values() if b['best'])
    dearer = [b for b in by.values() if b['best'] and O.frac(b['best']['cost']) > cheapest]
    checks.append(check('configuration_scope_dearer_configuration_descends_to_its_own_end', len(by) == 3
                        and all(b['stop_reason'] in ('NO_IMPROVING_PROPOSAL', 'NO_FEASIBLE_POINT_IN_RANGE') for b in by.values())
                        and dearer and all(b['steps'] > 0 for b in dearer) and par['stop_reason'] == 'NO_IMPROVING_PROPOSAL',
                        stops={k[:8]: (b['stop_reason'], b['steps'], b['best'] and b['best']['cost_display']) for k, b in by.items()}))
    checks.append(check('lookahead_concurrency_same_trajectory_as_one_at_a_time', path(seq) == path(par)
                        and {k: b['best']['vector'] for k, b in by.items()} == {b['configuration_id']: b['best']['vector'] for b in seq['by_configuration']}
                        and len(launches(out / 'scope_par')) >= len(launches(out / 'scope_seq')),
                        jobs=dict(one_at_a_time=len(launches(out / 'scope_seq')), lookahead=len(launches(out / 'scope_par')))))
    before = launches(out / 'scope_par');again = cli(out / 'scope_par', par_round)['PR_PAULI']
    checks.append(check('configuration_scope_rerun_reuses_all_jobs', launches(out / 'scope_par') == before and again['new_calls'] == 0
                        and path(again) == path(par), new_calls=again['new_calls']))

    # Interrupted record (search-policy change): a budget-stopped artificial round stands in for the stopped round; one
    # completed job off the trajectory (lookahead stand-in) and one REQUEST-only directory (stopped while waiting for a slot).
    ri = out / 'interrupted';fi = pauli_budget_round(out / 'interrupted.json', 'ART_INTERRUPT', pr, 6)
    first_run = cli(ri, fi)['PR_PAULI'];pend = first_run['pending_request']
    expected = {}
    for r in (ri / 'jobs' / 'PR_PAULI').glob('*/REQUEST.json'):  # the round's own expected identities (from C's catalog)
        e = json.loads(r.read_text())['expected'];expected[e['configuration_id']] = {k: v for k, v in e.items() if k != 'configuration_id'}
    ex = O.Executor(ROOT, ri, {'PR_PAULI': [pr]}, expected, problem).for_method('PR_PAULI')
    q = pend['queue'][1];extra = dict(configuration_id=next(b['configuration_id'] for b in first_run['by_configuration'] if b['candidate'] == q['candidate']),
                                      vector=tuple(q['vector']), cost=O.frac(q['cost']), origin=q['origin'])
    ex.launch(extra, pr);ex.finish()
    nxt = dict(configuration_id=pend['configuration_id'], vector=tuple(pend['vector']), cost=O.frac(pend['cost']), origin=pend['origin'])
    req_only = ex.job_dir(nxt);req_only.mkdir(parents=True)
    (req_only / 'REQUEST.json').write_text(json.dumps(O.jsonable(ex.request(nxt, req_only, pr))))
    stop = out / 'STOP.json';stop.write_text(json.dumps(dict(stop_reason='INTERRUPTED_SEARCH_POLICY_CHANGE', note='ARTIFICIAL')))
    before = launches(ri);rec = cli_interrupted(ri, fi, stop)['PR_PAULI']
    na = rec['interrupted']['computed_not_adopted'];rq = rec['interrupted']['requested_not_completed']
    checks.append(check('interrupted_record_labels_lists_and_launches_nothing', launches(ri) == before
                        and rec['stop_reason'] == 'INTERRUPTED_SEARCH_POLICY_CHANGE'
                        and any(b['stop_reason'] == 'INTERRUPTED_SEARCH_POLICY_CHANGE' for b in rec['by_configuration'])
                        and [tuple(O.interleave(x['N_X'], x['N_Y'])) for x in na] == [extra['vector']]
                        and [x['status'] for x in rq] == ['REQUESTED_NOT_LAUNCHED']
                        and [t['vector'] for t in rec['trace']] == [t['vector'] for t in first_run['trace']]
                        and all(b['cheapest_feasible_evaluated'] for b in rec['by_configuration'] if b['best']),
                        stop=rec['stop_reason'], not_adopted=len(na), requested=[x['status'] for x in rq]))
    pauli_budget_round(fi, 'ART_INTERRUPT', pr, 8);resumed = cli(ri, fi)['PR_PAULI']
    checks.append(check('resume_reuses_request_only_directory', (req_only / 'OUTCOME.json').exists()
                        and any(tuple(t['vector']) == nxt['vector'] and t['calls'] == 1 for t in resumed['trace']),
                        stop=resumed['stop_reason'], new_calls=resumed['new_calls']))

    # Next round (V2 rule) loads the stopped round's completed jobs: nothing launched for known allocations, same incumbents.
    done = [p.parent for p in (ri / 'jobs' / 'PR_PAULI').glob('*/OUTCOME.json')]
    rn = out / 'next_round';fn = pauli_budget_round(out / 'next_round.json', 'ART_NEXT_V2', pr, 0, rule='METHOD_MINIMUM_PRIORITY_V2',
                                                        prior=ri, approved=False)
    nr = cli(rn, fn)['PR_PAULI'];after = cli_interrupted(ri, fi, stop)['PR_PAULI']  # cheapest feasible over every evaluation so far
    prior_best = {b['configuration_id']: b['cheapest_feasible_evaluated'] and list(O.interleave(b['cheapest_feasible_evaluated']['N_X'],
                  b['cheapest_feasible_evaluated']['N_Y'])) for b in after['by_configuration']}
    checks.append(check('next_round_loads_prior_jobs_and_launches_nothing', not launches(rn) and nr['prior_rounds']['loaded'] == len(done)
                        and nr['prior_rounds']['refused'] == 0 and nr['stop_reason'] == 'NEW_SCIENCE_NOT_APPROVED'
                        and {b['configuration_id']: b['best'] and b['best']['vector'] for b in nr['by_configuration']} == prior_best
                        and nr['pending_request'] is not None and not ex.has_job(dict(configuration_id=nr['pending_request']['configuration_id'],
                                                                                       vector=tuple(nr['pending_request']['vector']))),
                        loaded=nr['prior_rounds']['loaded'], completed_prior_jobs=len(done), stop=nr['stop_reason']))
    # The same prior jobs with a route that also names a growing registry file (configurations_file) are still reused.
    rr = out / 'next_round_registry';fr = pauli_budget_round(out / 'next_round_registry.json', 'ART_NEXT_REGISTRY',
                                                              dict(pr, configurations_file=str(out / 'REGISTRY.json')), 0,
                                                              rule='METHOD_MINIMUM_PRIORITY_V2', prior=ri, approved=False)
    nrr = cli(rr, fr)['PR_PAULI']
    checks.append(check('prior_jobs_reused_when_route_names_a_registry_file', nrr['prior_rounds']['loaded'] == len(done)
                        and nrr['prior_rounds']['refused'] == 0 and not launches(rr), loaded=nrr['prior_rounds']['loaded']))
    other = artificial_route(problem);other['artificial']['scale'] = str(Fraction(pr['artificial']['scale']) * 2)
    rm = out / 'next_round_other_route';fm = pauli_budget_round(out / 'next_round_other_route.json', 'ART_NEXT_OTHER', other, 0,
                                                                 rule='METHOD_MINIMUM_PRIORITY_V2', prior=ri, approved=False)
    mm = cli(rm, fm)['PR_PAULI']
    checks.append(check('prior_job_of_a_different_route_is_refused', mm['prior_rounds']['loaded'] == 0
                        and mm['prior_rounds']['refused'] == len(done) and not launches(rm), refused=mm['prior_rounds']['refused_reasons']))

    # V2 with the artificial worker in an approved UNLIMITED round: runs to its end; a rerun launches nothing.
    rv = out / 'v2_round';fv = pauli_budget_round(out / 'v2_round.json', 'ART_V2', pr, 'UNLIMITED', rule='METHOD_MINIMUM_PRIORITY_V2')
    v2 = cli(rv, fv)['PR_PAULI'];l1 = launches(rv);v2b = cli(rv, fv)['PR_PAULI']
    checks.append(check('v2_round_runs_to_end_and_rerun_launches_nothing', v2['stop_reason'] == 'NO_CANDIDATE_LEFT_IN_RANGE'
                        and launches(rv) == l1 and v2b['new_calls'] == 0 and v2b['best']['cost'] == v2['best']['cost'],
                        stop=v2['stop_reason'], jobs=len(l1), best=v2['best'] and v2['best']['cost_display']))

    # V2 with a finite budget and concurrent lookahead (rounds driven in batches): same adopted trajectory as one at a time,
    # unconsumed lookahead jobs become knowledge of the next batch, and a rerun launches nothing.
    def v2_round(path, budget, lookahead, concurrent):
        spec = json.loads(pauli_budget_round(path, 'ART_V2_BATCH', pr, budget, rule='METHOD_MINIMUM_PRIORITY_V2').read_text())
        spec['search_overrides'].update(lookahead=lookahead);spec['executor'].update(max_concurrent_jobs=concurrent)
        path.write_text(json.dumps(spec, indent=1) + '\n');return path
    one = cli(out / 'v2_batch_seq', v2_round(out / 'v2_batch_seq.json', 20, 1, 1))['PR_PAULI']
    par_file = v2_round(out / 'v2_batch_par.json', 20, 8, 8);par = cli(out / 'v2_batch_par', par_file)['PR_PAULI']
    seq_path = [(t['configuration_id'], tuple(t['vector']), t['status']) for t in one['trace']]
    par_path = [(t['configuration_id'], tuple(t['vector']), t['status']) for t in par['trace']]
    l1 = launches(out / 'v2_batch_par');again = cli(out / 'v2_batch_par', par_file)['PR_PAULI']
    checks.append(check('v2_finite_budget_lookahead_same_trajectory_and_resume', seq_path == par_path and len(seq_path) == 20
                        and len(l1) >= len(launches(out / 'v2_batch_seq')) and launches(out / 'v2_batch_par') == l1 and again['new_calls'] == 0,
                        steps=len(seq_path), jobs=dict(one_at_a_time=len(launches(out / 'v2_batch_seq')), lookahead=len(l1)),
                        stops=[one['stop_reason'], par['stop_reason']]))

    counts = {}
    for k in checks:
        counts[k['status']] = counts.get(k['status'], 0) + 1
    result = dict(schema='RPE_OPTIMIZER_EXECUTOR_ACCEPTANCE_V1', optimizer_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                  cwd=ROOT, text=True).strip(), counts=counts, checks=checks,
                  science=dict(real_Q3_launched=0, artificial='LABELLED_ARTIFICIAL_WORKER'))
    (out / 'EXECUTOR_ACCEPTANCE.json').write_text(json.dumps(O.jsonable(result), indent=1, ensure_ascii=False) + '\n')
    for k in checks:
        print('%-45s %s' % (k['check'], k['status']))
    print(json.dumps(counts))
    return 0 if set(counts) == {'PASS'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
