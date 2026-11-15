"""Executor acceptance through the real entry (run_rpe_method_optimizer.py --round ... --run-dir ...).

Artificial worker routes (labelled ARTIFICIAL) and saved results only; the real routes are used only
in a budget-0 round, where no job may be launched.  Checks:
  request -> execute -> ingest -> next      saved reuse                budget 0 launches nothing
  budget exhausted -> resume (no relaunch)  idempotent rerun           running job not launched twice
  dead job: reported, then collected        UNKNOWN distinct from model failure
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
        r['stop_reason'] == 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET' and r['pending_request'] for r in res.values()),
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
