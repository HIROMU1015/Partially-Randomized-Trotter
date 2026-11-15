"""A's acceptance cases run through B's $rpe-cost-compare Skill entry (saved results and artificial workers only).

The Skill (``.agents/skills/rpe-cost-compare`` at --skill-commit) tells the agent to pass the user's
method/range/budget/known-start choices as a B2 round file (``RPE_OPTIMIZER_SEARCH_ROUND_V1``) and to
run, and resume, with exactly

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B scripts/run_rpe_method_optimizer.py \
      --problem configs/rpe_method_optimizer/h4_df10_problem.json --round <round.json> --run-dir <run>

in an own checkout of the program SOURCE named in its references/program-entry.md. This runner does
exactly that for the cases of ACCEPTANCE.md and judges the files the Skill tells the agent to read
(ROUND_RESULT.json methods, NEEDS.json, INVOCATIONS.jsonl, jobs/). No real Q3, signal, price or compile:
budget-0 rounds launch nothing; positive budgets use only B2's labelled ``artificial`` route.

    python scripts/acceptance_rpe_cost_compare_skill.py --skill-commit <B skill commit> \
        --program-root <own checkout of the program SOURCE> --python <program runtime> --output <new dir>
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.set_int_max_str_digits(0)          # long exact MSE bounds are written as decimal strings by the program
SKILL_DIR = '.agents/skills/rpe-cost-compare/'
PROBLEM = 'configs/rpe_method_optimizer/h4_df10_problem.json'
CLI = 'scripts/run_rpe_method_optimizer.py'
METHOD = 'TRUNCATED_PF'
SEED_4 = 'B0I_P7_S2_16h_over_7__B_PRICE'
AFTER_3 = 'P7_S2_LOCAL_R12_D1 (m12=3)'
P7_S2 = dict(formulas=['2nd'], delta_over_h_range=['16/7', '16/7'])
COARSE = 'dd8bfc24'
D_PRICE = dict(commit='c7c8c1edd0996ad9ef97629961f67ca508a182f0',
               path='docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json')
D_PRICE_TOTAL = 372035410
PRICE_WAIT = 'e6754564'                     # TRUNCATED_PF 2nd, delta/h = 1: Q3 signal present, 26-axis price missing
RULE_STOPS = {'NO_IMPROVING_PROPOSAL', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', 'Q3_BUDGET_EXHAUSTED', 'STEP_LIMIT'}
BUDGET_STOPS = {'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', 'Q3_BUDGET_EXHAUSTED'}
# B2's labelled artificial worker parameters (a4b4d0e1f executor_acceptance_a2e2ef50a/artificial_round.json):
# MSE = scale * sum_m w_m (1/N_X + 1/N_Y) / 2 with the adopted threshold tau = eps_E^2.
ARTIFICIAL = dict(weights=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169],
                  scale='13717125719597395476268286997213/653735135813562572002410888671875000000',
                  tau='15136964828156312001/5960464477539062500000000')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def frac(value):
    if isinstance(value, dict):
        return Fraction(int(value['numerator']), int(value['denominator']))
    return Fraction(value)


def check(name, ok, **detail):
    return dict(check=name, status='PASS' if ok else 'FAIL', **detail)


# ---------------------------------------------------------------- Skill entry
def load_skill(commit):
    files = {}
    for name in ('SKILL.md', 'references/program-entry.md', 'references/h4-round-budget0.json'):
        raw = git(HERE, 'show', commit + ':' + SKILL_DIR + name)
        files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode())
    entry = files['references/program-entry.md']['text']
    source = re.search(r'実行SOURCE：`([0-9a-f]{40})`', entry).group(1)
    command = re.search(r'```bash\n(.*?)```', entry, re.S).group(1)
    return dict(commit=git(HERE, 'rev-parse', commit).decode().strip(), files=files, program_source=source,
                documented_command=command, template=json.loads(files['references/h4-round-budget0.json']['text']))


def round_file(skill, round_id, *, budget, initial, overrides=None, artificial=None):
    """The Skill's round-field table: round_id, methods, per-method new-Q3 total cap, initial, method_overrides."""
    spec = deepcopy(skill['template'])
    spec.update(round_id=round_id, methods=[METHOD], max_new_q3_per_method={METHOD: budget}, initial=initial,
                method_overrides=overrides or {}, resume_from=[])
    if artificial:
        spec['routes_override'] = {METHOD: [dict(kind='artificial', interpreter=artificial, configuration_ids='*', view_commit=None,
                                                 artificial=ARTIFICIAL, existing_caller='ARTIFICIAL')]}
    return spec


class Program:
    """The documented command, in an own checkout of the Skill's program SOURCE."""

    def __init__(self, root, python, out):
        self.root, self.python, self.out, self.calls = Path(root), python, Path(out), []

    def run(self, round_path, run_dir, expect_failure=False):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='src')
        cmd = [self.python, '-B', CLI, '--problem', PROBLEM, '--round', str(round_path), '--run-dir', str(run_dir)]
        start = time.monotonic()
        done = subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True)
        call = dict(cmd=cmd, cwd=str(self.root), env=dict(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='src'),
                    returncode=done.returncode, wall_seconds=round(time.monotonic() - start, 2),
                    stdout=done.stdout[-4000:], stderr=done.stderr[-4000:])
        self.calls.append(call)
        log = Path(run_dir).parent / (Path(run_dir).name + '.console_%d.log' % len([c for c in self.calls if c['cmd'][-1] == str(run_dir)]))
        log.write_text(done.stdout + ('\n--- stderr ---\n' + done.stderr if done.stderr else ''))
        if expect_failure:
            return call
        if done.returncode != 0:
            raise RuntimeError('PROGRAM_NONZERO_EXIT %d: %s' % (done.returncode, done.stderr[-1500:]))
        return call


def outputs(run_dir):
    run = Path(run_dir)
    result = json.loads((run / 'ROUND_RESULT.json').read_text())
    needs = json.loads((run / 'NEEDS.json').read_text())
    invocations = [json.loads(line) for line in (run / 'INVOCATIONS.jsonl').read_text().splitlines() if line.strip()]
    jobs = sorted(p for p in (run / 'jobs').glob('*/*') if p.is_dir()) if (run / 'jobs').exists() else []
    return dict(result=result, method=result['methods'][METHOD], needs=needs, invocations=invocations, jobs=jobs,
                round_bytes=(run / 'ROUND.json').read_bytes())


# ---------------------------------------------------------------- checks on the Skill-read outputs
def in_range(name, formulas, lo, hi):
    m = re.search(r'formula=([^,]+),delta_over_h=([^:]+):', name)
    return m is not None and m.group(1) in formulas and Fraction(lo) <= Fraction(m.group(2)) <= Fraction(hi)


def range_check(o, scope):
    rows = [c for c in o['needs']['candidates'] if c['method'] == METHOD]
    names = [c['name'] for c in rows] + o['method']['search_range']['candidates'] + [t['candidate'] for t in o['method']['trace']]
    p = o['method']['pending_request']
    if p is not None:
        names += [p['candidate']] + [q['candidate'] for q in p.get('queue', [])]
    outside = sorted({n for n in names if not in_range(n, scope['formulas'], *scope['delta_over_h_range'])})
    return check('search_space_follows_requested_range', not outside and bool(rows), searched=sorted({c['name'] for c in rows}),
                 outside=outside)


def start_check(o, labels):
    r = o['method']; given = [e for e in r['evaluated'] if e['source'] == 'SAVED' and e.get('label') in labels]
    fit = [frac(e['cost']) for e in given if e['status'] == 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD']
    first = r['trace'][0]['incumbent_before'] if r['trace'] else (r['best'] and r['best']['cost'])
    return check('starts_from_given_saved_point_only', bool(fit) and first is not None and frac(first) == min(fit),
                 given_labels=sorted(e['label'] for e in given), first_incumbent=None if first is None else str(frac(first)))


def reveal_check(o, labels):
    """Observable form: every saved result the program knows is a given start label or an allocation it selected."""
    r = o['method']; chosen = {(t['configuration_id'], tuple(t['vector'])) for t in r['trace']}
    extra = [dict(candidate=e['candidate'], label=e.get('label')) for e in r['evaluated']
             if e['source'] == 'SAVED' and e.get('label') not in labels and (e['configuration_id'], tuple(e['vector'])) not in chosen]
    saved_steps = [t['step'] for t in r['trace'] if t['answer_source'] == 'SAVED']
    return check('saved_Q3_revealed_only_after_selection', not extra, unselected_saved_known=extra, saved_answer_steps=saved_steps,
                 observability='CLI output only (the call-level responder log is not exposed by the Skill entry)')


def next_action_check(o):
    r = o['method']; p = r['pending_request']
    concrete = p is None or (bool(p.get('configuration_id')) and len(p.get('vector', ())) == 26 and p.get('cost') is not None
                             and p.get('route') is not None)
    return check('next_action_decided_without_asking', r['stop_reason'] in RULE_STOPS and concrete, stop_reason=r['stop_reason'],
                 next=None if p is None else dict(candidate=p['candidate'], cost=str(frac(p['cost'])), route=p['route']))


def improvement_check(o):
    r = o['method']; trace = r['trace']; imp = [i for i, t in enumerate(trace) if t['improved']]
    if not imp:
        return check('continues_after_improvement', False, reason='NO_IMPROVEMENT_OBSERVED')
    last = imp[-1]; p = r['pending_request']
    cheaper = p is not None and frac(p['cost']) < frac(trace[last]['cost'])
    return check('continues_after_improvement', last < len(trace) - 1 or cheaper or r['stop_reason'] == 'NO_IMPROVING_PROPOSAL',
                 improvements=len(imp), steps_after_last_improvement=len(trace) - 1 - last, pending_cheaper=cheaper)


def launches(o):
    return dict(invocation_launched=o['invocations'][-1]['launched'][METHOD], job_dirs=len(o['jobs']),
                launch_files=sum((j / 'LAUNCH.json').exists() for j in o['jobs']),
                worker_artifacts=sum((j / 'ARTIFICIAL_RESULT.json').exists() for j in o['jobs']),
                launch_kinds=sorted({json.loads((j / 'LAUNCH.json').read_text())['kind'] for j in o['jobs'] if (j / 'LAUNCH.json').exists()}),
                new_calls=o['method']['new_calls'], spent_before=o['method']['budget']['spent_before'])


def nothing_launched(o):
    n = launches(o)
    return check('budget0_launches_nothing', n['invocation_launched'] == 0 and n['job_dirs'] == 0 and n['new_calls'] == 0, **n)


def skill_table(label, o, round_path, run_dir):
    """The Skill's 'return the result' row for this method, from ROUND_RESULT.json methods."""
    r = o['method']; b = r['best']; p = r['pending_request']; n = launches(o)
    best = '—' if b is None else '%s<br>X=%s<br>Y=%s' % (b['candidate'], b['N_X'], b['N_Y'])
    mse = '—' if b is None else '[%.6e, %.6e] %s' % (float(frac(b['MSE_lower'])), float(frac(b['MSE_upper'])), b['status'])
    cost = '—' if b is None else '%s（exact %s）' % (b['cost_display'], frac(b['cost']))
    nxt = '—' if p is None else '%s：%s RZ' % (p['candidate'], frac(p['cost']))
    return ('| %s | %s | %s | %s | %s | %s | %s | %s | 再利用 %d件・新規call %d・予算 %s（消費済み %s） |'
            % (label, METHOD, best, mse, cost, b and b['evidence'], r['stop_reason'], nxt, r['evaluated_count'] - n['new_calls'],
               n['new_calls'], r['budget']['max_new_q3'], r['budget']['spent_before']))


# ---------------------------------------------------------------- cases
def case1(skill, prog, out):
    res = dict(case='1_P7_S2_RANGE', runs={}, checks=[])
    plans = (('1a_from_m12_4_budget0', [SEED_4], 0, False), ('1b_after_m12_3_budget0', [SEED_4, AFTER_3], 0, False),
             ('1b_after_m12_3_budget1_artificial', [SEED_4, AFTER_3], 1, True))
    for label, start, budget, art in plans:
        spec = round_file(skill, 'A_SKILL_ACCEPT_' + label.upper(), budget=budget, initial={METHOD: start}, overrides={METHOD: P7_S2},
                          artificial=prog.python if art else None)
        rp = out / ('round_' + label + '.json'); rp.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n')
        run = out / ('run_' + label); prog.run(rp, run); o = outputs(run)
        res['runs'][label] = dict(round=str(rp.relative_to(out)), run_dir=str(run.relative_to(out)), table=skill_table(label, o, rp, run),
                                  steps=[dict(step=t['step'], candidate=t['candidate'], cost=str(frac(t['cost'])), status=t['status'],
                                              source=t['answer_source'], improved=t['improved']) for t in o['method']['trace']],
                                  **launches(o))
        cs = [range_check(o, P7_S2), start_check(o, start), reveal_check(o, start), next_action_check(o)]
        if label.startswith('1a'):
            cs += [improvement_check(o), nothing_launched(o)]
        elif budget == 0:
            r = o['method']
            cs += [check('next_action_from_m12_3_state_without_budget', r['pending_request'] is not None and r['stop_reason'] in BUDGET_STOPS,
                         stop_reason=r['stop_reason']), nothing_launched(o)]
        else:
            r = o['method']; n = launches(o)
            cs.append(check('remaining_budget_used_then_rule_stop',
                            n['invocation_launched'] == 1 and n['worker_artifacts'] == 1 and n['new_calls'] == 1
                            and n['launch_kinds'] == ['artificial'] and r['stop_reason'] in BUDGET_STOPS and r['pending_request'] is not None,
                            stop_reason=r['stop_reason'], **n))
        res['checks'] += [dict(c, run=label) for c in cs]
    return res


def case2(skill, prog, out):
    res = dict(case='2_Y4_COARSE_PRICE_REFLECTED', runs={}, checks=[])
    problem = json.loads((prog.root / PROBLEM).read_text())
    cat = problem['evidence']['saved_catalog_inputs']
    sources = json.loads(git(prog.root, 'show', cat['commit'] + ':' + cat['path']))['sources']
    res['checks'].append(check('price_input_is_D_saved_file', any(s['commit'] == D_PRICE['commit'] and s['path'] == D_PRICE['path'] for s in sources),
                               catalog_inputs=cat, price_input=D_PRICE))
    for label, overrides in (('2_p7_all_budget0', {}), ('2_y4_32h5_only_budget0', {METHOD: dict(formulas=['4th'], delta_over_h_range=['32/5', '32/5'])})):
        spec = round_file(skill, 'A_SKILL_ACCEPT_' + label.upper(), budget=0, initial='all', overrides=overrides)
        rp = out / ('round_' + label + '.json'); rp.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n')
        run = out / ('run_' + label); prog.run(rp, run); o = outputs(run); r = o['method']
        coarse = [c for c in o['needs']['candidates'] if c['method'] == METHOD and (c['configuration_id'] or '').startswith(COARSE)]
        rows = [e for e in r['evaluated'] if e['configuration_id'].startswith(COARSE) and e['cost'] is not None]
        res['runs'][label] = dict(round=str(rp.relative_to(out)), run_dir=str(run.relative_to(out)), table=skill_table(label, o, rp, run),
                                  coarse_rows=[dict(cost=str(frac(e['cost'])), status=e['status'], source=e['source'], label=e.get('label'))
                                               for e in rows], **launches(o))
        b = r['best']
        res['checks'] += [dict(c, run=label) for c in (
            check('coarse_Y4_ready_with_own_26_axis_price', len(coarse) == 1 and coarse[0]['ready'] and coarse[0]['needs'] == []
                  and coarse[0]['price_status'] == 'SAVED_UNIT_PRICES_REPRODUCE_ALL_SAVED_TOTALS', coarse=coarse),
            check('saved_Q3_reused_without_new_call', any(e['source'] == 'SAVED' for e in rows) and r['new_calls'] == 0, new_calls=r['new_calls']),
            nothing_launched(o),
            check('D_price_total_in_comparison_table', any(frac(e['cost']) == D_PRICE_TOTAL for e in rows) and b is not None
                  and frac(b['cost']) == D_PRICE_TOTAL and b['configuration_id'].startswith(COARSE),
                  best=None if b is None else dict(candidate=b['candidate'], cost=str(frac(b['cost'])), evidence=b['evidence'])),
            next_action_check(o))]
    return res


def case3(skill, prog, out):
    res = dict(case='3_ARTIFICIAL_BUDGET_END_AND_RESUME', runs={}, checks=[])
    scope = dict(formulas=['2nd'], delta_over_h_range=['1', '16/7'])          # A = p7 S2 (ready), B = 2nd 1h (price wait)
    rp = out / 'round_3_artificial.json'; run = out / 'run_3_artificial'
    spec = round_file(skill, 'A_SKILL_ACCEPT_3_ARTIFICIAL', budget=2, initial={METHOD: [SEED_4]}, overrides={METHOD: scope}, artificial=prog.python)
    rp.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n')
    prog.run(rp, run); first = outputs(run); r1 = first['method']; n1 = launches(first)
    old = {j.name: dict(launch=(j / 'LAUNCH.json').read_bytes(), artifact=(j / 'ARTIFICIAL_RESULT.json').read_bytes(),
                        outcome=(j / 'OUTCOME.json').read_bytes()) for j in first['jobs']}
    res['runs']['3_first_budget2'] = dict(table=skill_table('3 first (budget 2)', first, rp, run), **n1,
                                          steps=[dict(step=t['step'], candidate=t['candidate'], cost=str(frac(t['cost'])), status=t['status'],
                                                      source=t['answer_source'], calls=t['calls']) for t in r1['trace']])
    wait = [c for c in r1['not_ready_candidates'] if (c['configuration_id'] or '').startswith(PRICE_WAIT)]
    b1 = r1['best']; p1 = r1['pending_request']
    res['checks'] += [dict(c, run='3_first_budget2') for c in (
        range_check(first, scope),
        check('budget_end_returns_best_unevaluated_and_reason_separately',
              r1['stop_reason'] == 'Q3_BUDGET_EXHAUSTED' and b1 is not None and b1['status'] == 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
              and bool(b1['evidence']) and p1 is not None and len(wait) == 1 and wait[0]['needs'] == ['PRICE_26AXIS'],
              stop_reason=r1['stop_reason'], best=b1 and dict(candidate=b1['candidate'], cost=str(frac(b1['cost'])), evidence=b1['evidence']),
              next=p1 and dict(candidate=p1['candidate'], cost=str(frac(p1['cost']))), price_wait=wait),
        check('no_science_call_beyond_budget', n1['invocation_launched'] == 2 and n1['worker_artifacts'] == 2 and n1['new_calls'] == 2
              and n1['launch_kinds'] == ['artificial'], **n1),
        check('resume_input_recoverable_from_output',
              len([x for x in r1['resume_records'] if x['source'] == 'ARTIFICIAL_JOB']) == 2
              and all(x['configuration_id'] and len(x['N_X']) == 13 and len(x['N_Y']) == 13 and x['status'] and x['MSE_lower'] is not None
                      and x['MSE_upper'] is not None for x in r1['resume_records']),
              resume_records=[dict(configuration_id=x['configuration_id'][:8], status=x['status'], source=x['source'], job=x['job'])
                              for x in r1['resume_records']]))]
    # Skill: an explicitly allowed budget increase is written into the SAME round file as the method's total cap,
    # then the SAME command is re-run on the SAME run directory.
    spec['max_new_q3_per_method'][METHOD] = 4
    rp.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n')
    prog.run(rp, run); second = outputs(run); r2 = second['method']; n2 = launches(second)
    new_jobs = [j for j in second['jobs'] if j.name not in old]
    new_jobs.sort(key=lambda j: json.loads((j / 'LAUNCH.json').read_text())['started'])
    first_new = json.loads((new_jobs[0] / 'REQUEST.json').read_text()) if new_jobs else None
    first_new_vector = None if first_new is None else [v for pair in zip(first_new['N_X'], first_new['N_Y']) for v in pair]
    step_new = next((t for t in r2['trace'] if t['calls'] == 1), None)
    unchanged = all((run / 'jobs' / METHOD / name / 'LAUNCH.json').read_bytes() == v['launch']
                    and (run / 'jobs' / METHOD / name / 'ARTIFICIAL_RESULT.json').read_bytes() == v['artifact']
                    and (run / 'jobs' / METHOD / name / 'OUTCOME.json').read_bytes() == v['outcome'] for name, v in old.items())
    res['runs']['3_resume_total_cap4'] = dict(table=skill_table('3 resume (total cap 4)', second, rp, run), **n2,
                                              new_job_dirs=[j.name for j in new_jobs])
    res['checks'] += [dict(c, run='3_resume_total_cap4') for c in (
        check('resume_does_not_repeat_completed_calls', unchanged and n2['invocation_launched'] == 4 and n2['worker_artifacts'] == 4
              and len(new_jobs) == 2 and n2['spent_before'] == 2 and n2['new_calls'] == 2 and second['round_bytes'] == first['round_bytes'],
              old_jobs_unchanged=unchanged, round_json_unchanged=second['round_bytes'] == first['round_bytes'],
              invocation_budgets=[i['budget'][METHOD] for i in second['invocations']], **n2),
        check('resume_starts_from_previous_state',
              first_new_vector is not None and p1 is not None and first_new_vector == list(p1['vector'])
              and first_new['configuration_id'] == p1['configuration_id'] and step_new is not None
              and frac(step_new['incumbent_before']) == frac(b1['cost']),
              previous_request=p1 and dict(candidate=p1['candidate'], cost=str(frac(p1['cost']))),
              first_new_job=None if first_new is None else dict(configuration_id=first_new['configuration_id'][:8], job=new_jobs[0].name),
              incumbent_before_first_new=None if step_new is None else str(frac(step_new['incumbent_before']))))]
    prog.run(rp, run); third = outputs(run); n3 = launches(third)
    res['runs']['3_same_command_again'] = dict(table=skill_table('3 same command again', third, rp, run), **n3)
    res['checks'].append(dict(check('idempotent_rerun_launches_nothing', n3['invocation_launched'] == 4 and n3['job_dirs'] == 4
                                    and n3['new_calls'] == 0 and third['method']['best'] == r2['best']
                                    and [i['budget'][METHOD] for i in third['invocations']] == [2, 4, 4], **n3), run='3_same_command_again'))
    # Skill: a changed contract (range/initial/routes) must not be pushed into the same run directory.
    changed = deepcopy(spec); changed['method_overrides'] = {METHOD: P7_S2}
    cp = out / 'round_3_artificial_changed_scope.json'; cp.write_text(json.dumps(changed, ensure_ascii=False, indent=1) + '\n')
    call = prog.run(cp, run, expect_failure=True); fourth = outputs(run)
    res['checks'].append(dict(check('changed_round_contract_refused_in_same_run_dir',
                                    call['returncode'] != 0 and 'different approved round' in call['stderr'] and len(fourth['jobs']) == 4
                                    and len(fourth['invocations']) == 3, returncode=call['returncode'], stderr_tail=call['stderr'][-200:]),
                              run='3_changed_scope_same_dir'))
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skill-commit', required=True); parser.add_argument('--program-root', required=True)
    parser.add_argument('--python', required=True); parser.add_argument('--output', required=True)
    a = parser.parse_args(); out = Path(a.output).resolve()
    if out.exists():
        raise SystemExit('A new output directory is required (new run directories and round_ids)')
    out.mkdir(parents=True)
    skill = load_skill(a.skill_commit); root = Path(a.program_root).resolve()
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    status = git(root, 'status', '--porcelain', '--untracked-files=no').decode()
    entry = [check('program_checkout_is_skill_program_source', head == skill['program_source'] and not status,
                   program_head=head, skill_program_source=skill['program_source'], tracked_changes=status),
             check('documented_cli_present', all(x in skill['documented_command'] for x in ('--problem', '--round', '--run-dir', CLI))
                   and all(x in git(root, 'show', 'HEAD:' + CLI).decode() for x in ("'--round'", "'--run-dir'")), command=skill['documented_command'])]
    prog = Program(root, a.python, out); cases = []; error = None
    try:
        for case in (case1, case2, case3):
            cases.append(case(skill, prog, out))
    except Exception as exc:
        import traceback
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()[-2000:])
    checks = entry + [c for x in cases for c in x['checks']]
    counts = dict(PASS=sum(c['status'] == 'PASS' for c in checks), FAIL=sum(c['status'] == 'FAIL' for c in checks))
    result = dict(schema='A_SKILL_ACCEPTANCE_RESULT_V1', skill=dict(commit=skill['commit'], files={k: v['sha256'] for k, v in skill['files'].items()},
                  program_source=skill['program_source']), program_root=str(root), program_head=head, python=a.python,
                  entry_checks=entry, cases=cases, counts=counts, error=error, invocations=prog.calls,
                  science=dict(real_Q3=0, signal=0, price=0, compile=0, artificial_worker_jobs='labelled ARTIFICIAL route only'))
    (out / 'SKILL_ACCEPTANCE_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + '\n')
    lines = ['| run | 方式 | 最安適合構成・shot配分 X/Y | MSE区間・判定 | 総compiled RZ | 価格・証拠 | 停止理由 | 次候補 | 再利用・新規call・予算 |',
             '|---|---|---|---|---|---|---|---|---|']
    lines += [r['table'] for x in cases for r in x['runs'].values()]
    (out / 'SKILL_OUTPUT_TABLE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(counts=counts, error=error and error['message'],
                          fails=[(c.get('run'), c['check']) for c in checks if c['status'] == 'FAIL']), ensure_ascii=False, indent=1))
    sys.exit(0 if not counts['FAIL'] and error is None else 1)


if __name__ == '__main__':
    main()
