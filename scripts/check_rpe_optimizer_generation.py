"""Acceptance of optimizer-generated PR configurations (saved material + artificial fixtures, no science).

Checks: the current S2 / Delta_PF = h / common kappa schedule is reproduced for every saved PR configuration;
changing Delta_PF or kappa changes the candidates, their missing inputs and the next request; physically
identical schedules are never evaluated twice; stop reasons distinguish an exhausted approved range, missing
inputs, an unapproved round and a count limit.  Writes GENERATION_ACCEPTANCE.json.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O

PROBLEM = ROOT / 'configs/rpe_method_optimizer/h4_df10_problem.json'
LD7, LD5 = 'ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1', '4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb'


def check(name, ok, **detail):
    return dict(check=name, status='PASS' if ok else 'FAIL', **detail)


def row(gen, method, base, **variant):
    want = dict(dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'), **variant)
    return next(r for r in gen[method]['variants'] if r['base_configuration_id'] == base and r['variant'] == want)


def make_ready(cand, base):
    """ARTIFICIAL readiness for a generated candidate: reused unit prices where the axis is physically identical,
    the base axis price otherwise (labelled), and an artificial signal flag.  Never science."""
    c = deepcopy(cand);c.update(prices=list(base['prices']), signal_source='ARTIFICIAL_FIXTURE', needs=[], ready=True,
                                price_status='ARTIFICIAL_FIXTURE')
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output;out.mkdir(parents=True, exist_ok=False)
    problem = O.load_json(PROBLEM);ctx = O.prepare(ROOT, problem);gen = ctx['generation'];checks = []
    cands = ctx['candidates'];by_id = {c['configuration_id']: c for c in cands if c['configuration_id']}

    base_rows = [r for m in gen for r in gen[m]['variants'] if r['variant'] == gen[m]['baseline']]
    checks.append(check('baseline_schedule_reproduced_for_every_saved_PR_configuration',
        base_rows and all(r['status'] == 'GENERATED' and r['base_reproduced'] and r['physically_identical_to'] == r['base_configuration_id']
                          for r in base_rows), configurations=[(r['L_D'], r['base_configuration_id'][:8]) for r in base_rows]))
    ld7_base = by_id[LD7]
    checks.append(check('LD7_baseline_rounds_q_2pow_m_R_1', [(x['q'], x['R']) for x in row(gen, 'PR_DF', LD7)['rounds']]
                        == [(1 << m, 1) for m in range(13)]))

    d = row(gen, 'PR_DF', LD7, delta_over_h='1/2');new = by_id.get(d.get('configuration_id'))
    checks.append(check('delta_change_new_schedule_and_needs', d['disposition'] == 'NEW_SCHEDULE_CANDIDATE'
        and [x['q'] for x in d['rounds']] == [2 << m for m in range(13)] and new is not None and not new['ready']
        and 'SIGNAL_ROUNDS:' + ','.join(map(str, range(13))) in new['needs'] and 'PRICE_AXES:26_of_26' in new['needs'],
        q=[x['q'] for x in d['rounds']], needs=new and new['needs']))

    k = row(gen, 'PR_DF', LD7, kappa_multiplier='2')
    checks.append(check('kappa_change_without_schedule_change_reuses_saved_evaluation',
        k['disposition'] == 'IDENTICAL_SCHEDULE_REUSES_SAVED_EVALUATIONS' and k['physically_identical_to'] == LD7
        and k['configuration_id'] not in by_id and any(a['configuration_id'] == k['configuration_id'] for a in ld7_base.get('physical_aliases', [])),
        alias=k['configuration_id'][:8]))
    p5 = row(gen, 'PR_PAULI', LD5, kappa_multiplier='2')
    checks.append(check('kappa_change_with_schedule_change_partial_reuse',
        p5['disposition'] == 'NEW_SCHEDULE_CANDIDATE' and p5['rounds_differing_from_base'] == [10, 11, 12]
        and p5['reusable']['signal_rounds'] == list(range(10)) and p5['reusable']['price_axes'] == 20, needs=p5.get('needs')))

    keys = {}
    for m in gen:
        for r in gen[m]['variants']:
            if r['disposition'] == 'NEW_SCHEDULE_CANDIDATE':
                keys.setdefault(r['configuration_id'], 0);keys[r['configuration_id']] += 1
    dup_saved = [r for m in gen for r in gen[m]['variants'] if r['disposition'] == 'NEW_SCHEDULE_CANDIDATE' and r['physically_identical_to']]
    plain = O.prepare(ROOT, problem, generate_configurations=False)
    a = O.optimize(ROOT, problem, context=plain, methods=['PR_DF', 'PR_PAULI'])
    b = O.optimize(ROOT, problem, context=ctx, methods=['PR_DF', 'PR_PAULI'])
    same = all(a[m]['evaluated_count'] == b[m]['evaluated_count'] and a[m]['pending_request']['vector'] == b[m]['pending_request']['vector']
               for m in a)
    checks.append(check('physically_identical_candidates_not_evaluated_twice', not dup_saved and all(v == 1 for v in keys.values()) and same,
                        new_schedules=len(keys), evaluated={m: b[m]['evaluated_count'] for m in b}))

    pauli = [c for c in cands if c['method'] == 'PR_PAULI']
    knowledge = O.initial_knowledge(ctx['saved'], pauli)
    before, _ = O.proposals([c for c in pauli if c['ready']], knowledge, ctx['tau'], problem['search'], ctx['generate'])
    ready_new = make_ready(by_id[p5['configuration_id']], by_id[LD5])
    after_c = [ready_new if c['configuration_id'] == ready_new['configuration_id'] else c for c in pauli]
    after, _ = O.proposals([c for c in after_c if c['ready']], knowledge, ctx['tau'], problem['search'], ctx['generate'])
    checks.append(check('next_requests_change_when_generated_candidate_becomes_ready',
        not any(x['configuration_id'] == ready_new['configuration_id'] for x in before)
        and any(x['configuration_id'] == ready_new['configuration_id'] and x['origin']['kind'] == 'GENERATOR_SEED' for x in after),
        before=len(before), after=len(after)))

    tau = ctx['tau']
    def mse(cid, v):
        u = tau * Fraction(1, 2) * sum(Fraction(1, n) for n in v) / sum(Fraction(1, n) for n in by_id[LD7]['caps'] for _ in 'XY')
        return u, u
    fx = O.ChainBackend(ctx['saved'], O.FixtureBackend(mse, tau))
    big = dict(problem, search=dict(problem['search'], max_steps_per_method=5000))
    unl = O.optimize(ROOT, big, context=ctx, methods=['PR_DF'], backend=fx, max_new='UNLIMITED')['PR_DF']
    noap = O.optimize(ROOT, big, context=ctx, methods=['PR_DF'], backend=fx, max_new='UNLIMITED', science_approved=False)['PR_DF']
    cap = O.optimize(ROOT, big, context=ctx, methods=['PR_DF'], backend=fx, max_new=2)['PR_DF']
    checks.append(check('stop_reasons_distinguished', unl['stop_reason'] == 'NO_IMPROVING_PROPOSAL_INPUTS_MISSING'
        and noap['stop_reason'] == 'NEW_SCIENCE_NOT_APPROVED' and noap['new_calls'] == 0
        and cap['stop_reason'] == 'Q3_BUDGET_EXHAUSTED' and cap['new_calls'] == 2,
        unlimited=dict(stop=unl['stop_reason'], new_calls=unl['new_calls'], best=unl['best']['cost_display']),
        not_approved=noap['stop_reason'], capped=cap['stop_reason'], artificial='LABELLED_ARTIFICIAL_FIXTURE'))

    counts = {}
    for c in checks:
        counts[c['status']] = counts.get(c['status'], 0) + 1
    result = dict(schema='RPE_OPTIMIZER_GENERATION_ACCEPTANCE_V1', optimizer_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                  cwd=ROOT, text=True).strip(), counts=counts, checks=checks, generation=gen,
                  parameter_assessment={m: b[m]['parameter_assessment'] for m in b},
                  science=dict(new_signal=0, new_price=0, compile=0, new_Q3=0, artificial='LABELLED'))
    (out / 'GENERATION_ACCEPTANCE.json').write_text(json.dumps(O.jsonable(result), indent=1, ensure_ascii=False) + '\n')
    for c in checks:
        print('%-72s %s' % (c['check'], c['status']))
    print(json.dumps(counts))
    return 0 if set(counts) == {'PASS'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
