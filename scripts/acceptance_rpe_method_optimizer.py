"""User-side acceptance cases for the per-method RPE optimizer (B2) and its Skill entry (B).

Defines, from the user's standpoint, the input given, the evaluation results returned during
the run and the behaviour to observe, and runs them against an actual optimizer entry
(``optimize``/``prepare`` of ``src/trottertracks/rpe_method_optimizer.py`` in --optimizer-root).
Saved results and artificial fixtures only: no signal, price, compile or Q3 is computed.

Cases (docs/research/a_rpe_optimizer_acceptance_20261116/ACCEPTANCE.md):
  1  p7 S2 shot search from the saved m12 = 4-shot feasible point (saved Q3 revealed only on selection)
  2  p7 Yoshida4 coarse 32h/5 A_SHAPE: price missing, then D's saved price given (372,035,410 RZ)
  3  artificial budget exhaustion and resume without repeating completed calls

    python scripts/acceptance_rpe_method_optimizer.py --optimizer-root <checkout> --output <dir>
"""
import argparse
import copy
from fractions import Fraction
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parents[1]
PROBLEM = 'configs/rpe_method_optimizer/h4_df10_problem.json'
METHOD = 'TRUNCATED_PF'
P7_S2 = dict(formula='2nd', delta_over_h='16/7')
SEED_4 = 'B0I_P7_S2_16h_over_7__B_PRICE'                 # saved m12 = 4 feasible point
AFTER_3 = 'P7_S2_LOCAL_R12_D1 (m12=3)'                    # saved m12 = 3 feasible point
COARSE = dict(configuration_prefix='dd8bfc248f', label='B0I_P7_Y4_32h_over_5__A_SHAPE')
D_PRICE = dict(commit='c7c8c1edd0996ad9ef97629961f67ca508a182f0',
               path='docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json')
D_PRICE_TOTAL = 372035410
CATALOG_WITHOUT_PRICE = '08c6a694bb05b4d09822bded996c9a1ca35e96a7'   # B2's problem input (Q3 fit, no price)
CATALOG_WITH_PRICE = '417a73e73'                                   # C's INPUTS that list the D price file
RULE_STOPS = {'NO_IMPROVING_PROPOSAL', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', 'Q3_BUDGET_EXHAUSTED', 'STEP_LIMIT',
              'NEW_Q3_REQUIRED_NO_EXECUTOR_IN_THIS_VERSION'}
BUDGET_STOPS = {'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', 'Q3_BUDGET_EXHAUSTED'}


# ---------------------------------------------------------------- instrumented answer sources
class RevealLog:
    """Saved Q3 answers are returned only for the allocation the optimizer has just selected."""

    def __init__(self, inner, fixture=None):
        self.inner, self.fixture, self.queries = inner, fixture, []

    def answer(self, proposal):
        hit = self.inner.answer(proposal) if self.inner is not None else None
        if hit is None and self.fixture is not None:
            hit = self.fixture(proposal)
        self.queries.append(dict(configuration_id=proposal['configuration_id'], vector=list(proposal['vector']),
                                 answered=hit is not None, source=None if hit is None else hit['source'],
                                 calls=None if hit is None else hit['calls']))
        return hit


def artificial_answer(threshold, pass_ratio=Fraction(19, 20)):
    """ARTIFICIAL Q3 for unsaved allocations (labelled; never science): always one new call."""
    def answer(proposal):
        value = threshold * pass_ratio
        return dict(status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', MSE_lower=value, MSE_upper=value,
                    source='ARTIFICIAL_FIXTURE', calls=1)
    return answer


# ---------------------------------------------------------------- generic checks on one method result
def check(name, ok, **detail):
    return dict(check=name, status='PASS' if ok else 'FAIL', **detail)


def reveal_after_selection(result, log):
    """Every backend query is the selected proposal of a step (or the final pending request), in order."""
    chosen = [t['vector'] for t in result['trace']]
    pending = result['pending_request']
    expected = chosen + ([list(pending['vector'])] if pending is not None and len(log.queries) > len(chosen) else [])
    asked = [q['vector'] for q in log.queries]
    return check('saved_Q3_revealed_only_after_selection', asked == expected, queries=len(asked), steps=len(chosen))


def no_user_choice(result):
    """The run ends by a rule; if it needs more, it returns a concrete machine-readable next action."""
    p = result['pending_request']
    concrete = p is None or (p.get('configuration_id') and len(p.get('vector', ())) == 26 and p.get('cost') is not None
                             and p.get('route') is not None)
    return check('next_action_decided_without_asking', result['stop_reason'] in RULE_STOPS and bool(concrete),
                 stop_reason=result['stop_reason'], pending_route=None if p is None else p.get('route'))


def continues_after_improvement(result):
    """After any improvement the run either takes another step or requests one from the updated incumbent."""
    trace = result['trace']; improved = [i for i, t in enumerate(trace) if t['improved']]
    if not improved:
        return check('continues_after_improvement', False, reason='NO_IMPROVEMENT_OBSERVED')
    last = improved[-1]
    went_on = last < len(trace) - 1
    requested = result['pending_request'] is not None and result['pending_request']['cost'] < trace[last]['cost']
    return check('continues_after_improvement', went_on or requested or result['stop_reason'] == 'NO_IMPROVING_PROPOSAL',
                 improvements=len(improved), steps_after_last_improvement=len(trace) - 1 - last,
                 pending_cheaper_than_new_incumbent=requested)


def candidates_in_range(context, method, wanted):
    """The searched candidates are exactly those inside the requested formula/width range."""
    searched = [c for c in context['candidates'] if c['method'] == method]
    outside = [c['name'] for c in searched if c['key'].get('formula') != wanted['formula']
               or (c['key'].get('delta_over_h') is not None and Fraction(c['key']['delta_over_h']) != Fraction(wanted['delta_over_h']))
               or c['key'].get('delta_over_h') is None]
    return check('search_space_follows_requested_range', not outside, outside_candidates=outside)


# ---------------------------------------------------------------- cases
def problem_variant(base, *, catalog=None, p7_s2_only=False):
    p = copy.deepcopy(base)
    if catalog is not None:
        p['evidence']['saved_catalog_inputs']['commit'] = catalog
    if p7_s2_only:
        p['methods'][METHOD]['formulas'] = [P7_S2['formula']]
        p['methods'][METHOD]['delta_over_h_range'] = [P7_S2['delta_over_h'], P7_S2['delta_over_h']]
    return p


def case1(O, root, base):
    """p7 S2 shot search: start from the saved m12 = 4 point; saved Q3 only after selection."""
    out = dict(case='1_P7_S2_SHOT_SEARCH', checks=[], runs={})
    problem = problem_variant(base, p7_s2_only=True)
    context = O.prepare(root, problem)
    out['checks'].append(candidates_in_range(context, METHOD, P7_S2))
    for label, initial in (('1a_from_m12_4', [SEED_4]), ('1b_after_m12_3', [SEED_4, AFTER_3])):
        log = RevealLog(context['saved'])
        r = O.optimize(root, problem, context=context, methods=[METHOD], initial={METHOD: initial}, backend=log, max_new=0)[METHOD]
        out['runs'][label] = summary(r, log)
        seed_cost = None if not r['trace'] else r['trace'][0]['incumbent_before']
        known = [p for p in context['catalog']['candidates'] if p['label'] in initial]
        expected_seed = min(Fraction(p['conditional_cost']['numerator'], p['conditional_cost']['denominator']) for p in known)
        out['checks'] += [dict(c, run=label) for c in (
            check('starts_from_given_saved_point_only', seed_cost == expected_seed, seed_cost=show(seed_cost)),
            reveal_after_selection(r, log), no_user_choice(r))]
        if label == '1a_from_m12_4':
            out['checks'].append(dict(continues_after_improvement(r), run=label))
        else:
            # The next action from the m12 = 3 state follows from the remaining budget: none -> concrete request;
            # one artificial call -> it is taken without asking, then the run stops by the budget rule.
            base_vec = [p for p in known if p['label'] == AFTER_3]
            pend = r['pending_request']
            out['checks'].append(dict(check('next_action_from_m12_3_state_without_budget',
                pend is not None and r['stop_reason'] in BUDGET_STOPS and (r['trace'] or pend['origin'].get('base_vector') is not None),
                stop_reason=r['stop_reason']), run=label))
            log2 = RevealLog(context['saved'], fixture=artificial_answer(context['tau']))
            r2 = O.optimize(root, problem, context=context, methods=[METHOD], initial={METHOD: initial}, backend=log2, max_new=1)[METHOD]
            out['runs'][label + '_budget_1_artificial'] = summary(r2, log2)
            out['checks'].append(dict(check('remaining_budget_used_then_rule_stop',
                r2['new_calls'] == 1 and r2['stop_reason'] in BUDGET_STOPS and r2['pending_request'] is not None,
                new_calls=r2['new_calls'], stop_reason=r2['stop_reason']), run=label + '_budget_1_artificial'))
    out['not_a_pass_condition'] = 'The saved 4 -> 3 -> 2 -> 1 order of m12 shots is not required.'
    return out


def case2(O, root, base):
    """Y4 coarse 32h/5 A_SHAPE: price missing is identified; D's saved price is then used with the saved Q3."""
    out = dict(case='2_Y4_COARSE_A_SHAPE_PRICE_WAIT', checks=[], runs={})
    before = problem_variant(base, catalog=CATALOG_WITHOUT_PRICE)
    ctx = O.prepare(root, before)
    coarse = [c for c in ctx['candidates'] if (c['configuration_id'] or '').startswith(COARSE['configuration_prefix'])]
    saved = [p for p in ctx['catalog']['candidates'] if p['label'] == COARSE['label']]
    out['checks'].append(check('price_missing_identified',
        len(coarse) == 1 and not coarse[0]['ready'] and coarse[0]['needs'] == ['PRICE_26AXIS']
        and len(saved) == 1 and saved[0]['q3_status'] == 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',
        needs=coarse[0]['needs'] if coarse else None, saved_q3=saved[0]['q3_status'] if saved else None))
    r0 = O.optimize(root, before, context=ctx, methods=[METHOD], max_new=0)[METHOD]
    listed = [c for c in r0['not_ready_candidates'] if (c['configuration_id'] or '').startswith(COARSE['configuration_prefix'])]
    out['checks'].append(check('price_wait_reported_in_output', len(listed) == 1 and listed[0]['needs'] == ['PRICE_26AXIS'],
                               listed=listed))
    after = problem_variant(base, catalog=CATALOG_WITH_PRICE)
    try:
        sources = json.loads(subprocess.check_output(['git', 'show', after['evidence']['saved_catalog_inputs']['commit'] + ':'
                                                      + after['evidence']['saved_catalog_inputs']['path']], cwd=root))['sources']
        out['checks'].append(check('price_input_is_D_saved_file', any(s['commit'] == D_PRICE['commit'] and s['path'] == D_PRICE['path']
                                                                         for s in sources), price_input=D_PRICE))
        ctx2 = O.prepare(root, after)
        log = RevealLog(ctx2['saved'])
        r = O.optimize(root, after, context=ctx2, methods=[METHOD], backend=log, max_new=0)[METHOD]
        out['runs']['2b_with_D_price'] = summary(r, log)
        hit = [e for e in r['evaluated'] if e['candidate'].endswith(COARSE['configuration_prefix'][:8]) and e['cost'] is not None]
        out['checks'] += [
            check('saved_Q3_reused_without_new_call', r['new_calls'] == 0 and any(e['source'] == 'SAVED' for e in hit),
                  new_calls=r['new_calls']),
            check('D_price_total_in_comparison_table', any(Fraction(e['cost']) == D_PRICE_TOTAL for e in hit)
                  and r['best'] is not None and r['best']['cost'] <= D_PRICE_TOTAL,
                  best=r['best'] and r['best']['cost_display'], coarse_rows=[show(e['cost']) for e in hit])]
    except Exception as exc:
        out['checks'].append(check('D_price_total_in_comparison_table', False, error=type(exc).__name__ + ': ' + str(exc),
                                   traceback=traceback.format_exc()[-600:]))
    return out


def artificial_context(O):
    """Two-configuration artificial method: A ready with prices, B waiting for a price."""
    def prices(scale):
        return [Fraction(scale * (m + 1)) for m in range(13) for _ in O.AXES]
    tau = Fraction(1, 100)
    a = dict(method='ART', family='PR', key=dict(L_D=1), configuration_id='A' * 16, name='ART:L_D=1:AAAAAAAA',
             prices=prices(1), caps=[6] * 13, ready=True, needs=[], labels=[], price_status='ARTIFICIAL',
             signal_source='ARTIFICIAL', saved_points=[])
    b = dict(a, key=dict(L_D=2), configuration_id='B' * 16, name='ART:L_D=2:BBBBBBBB', prices=None, ready=False,
             needs=['PRICE_26AXIS'], price_status='NO_SAVED_26_AXIS_PRICE')
    seed = tuple([6] * 26)
    saved = O.SavedBackend([], {})
    saved.table[(a['configuration_id'], seed)] = dict(status=O.PASS, MSE_lower=tau / 10, MSE_upper=tau / 10, source='SAVED',
                                                      point_id='ART_SEED', label='ART_SEED', result_sources=[], cost=O.cost(seed, a['prices']))
    problem = dict(methods={'ART': dict(family='PR')}, q3_routes={'ART': dict(caller='ARTIFICIAL_ROUTE')},
                   search=dict(decrement_steps=[1, 2], large_step_headroom='9/10', seed_rules=['B_PRICE'],
                               max_steps_per_method=40, max_new_q3_per_method=0))
    def generate(caps, p, budget, rule):
        return None
    context = dict(candidates=[a, b], saved=saved, tau=tau, generate=generate)
    def mse(cid, v):
        u = sum(Fraction(1, n) for n in v) * tau / 40
        return u, u
    return problem, context, mse


class CallLog:
    def __init__(self, fixture):
        self.fixture, self.calls = fixture, []

    def answer(self, proposal):
        hit = self.fixture.answer(proposal)
        if hit is not None and hit['calls']:
            self.calls.append(dict(configuration_id=proposal['configuration_id'], vector=list(proposal['vector']),
                                   status=hit['status'], MSE_lower=hit['MSE_lower'], MSE_upper=hit['MSE_upper']))
        return hit


def case3(O, root):
    """Artificial budget exhaustion and resume (no repeated completed calls)."""
    out = dict(case='3_BUDGET_END_AND_RESUME', checks=[], runs={})
    problem, context, mse = artificial_context(O)
    chain = lambda: O.ChainBackend(context['saved'], O.FixtureBackend(mse, context['tau']))
    first_log = CallLog(chain())
    r1 = O.optimize(root, problem, context=context, methods=['ART'], initial='all', backend=first_log, max_new=2)['ART']
    out['runs']['3a_budget_2'] = summary(r1, None)
    best = r1['best']; pend = r1['pending_request']
    out['checks'].append(check('budget_end_returns_best_unevaluated_and_reason_separately',
        r1['stop_reason'] in BUDGET_STOPS and best is not None and best['status'] == O.PASS
        and pend is not None and pend['cost'] < best['cost']
        and any(c['needs'] == ['PRICE_26AXIS'] for c in r1['not_ready_candidates']),
        stop_reason=r1['stop_reason'], best_cost=show(best and best['cost']), pending_cost=show(pend and pend['cost']),
        not_ready=[c['candidate'] for c in r1['not_ready_candidates']], new_calls=r1['new_calls']))
    consumed = [t['vector'] for t in r1['trace'] if t['calls']]
    out['checks'].append(check('no_science_call_beyond_budget', len(first_log.calls) <= 2,
        backend_calls_made=len(first_log.calls), calls_counted_by_optimizer=r1['new_calls'], budget=2,
        note='an answer computed and then discarded beyond the budget would be an unapproved Q3 with a real executor'))
    # Resume from the run's own JSON output only (what a user/Skill would have after the process ends).
    output = json.loads(json.dumps(O.jsonable(r1)))
    names = {c['name']: c['configuration_id'] for c in context['candidates']}
    missing = []
    records = []
    for t in output['trace']:
        if t.get('calls'):
            need = {k: t.get(k) for k in ('MSE_lower', 'MSE_upper', 'status')}
            if need['MSE_lower'] is None or names.get(t['candidate']) is None:
                missing.append(dict(step=t['step'], missing=[k for k, v in need.items() if v is None]
                                    + ([] if names.get(t['candidate']) else ['configuration_id'])))
    out['checks'].append(check('resume_input_recoverable_from_output', not missing, missing=missing))
    # Resume with the completed calls as recorded by the call log (stands in for a persisted result store).
    records = [dict(configuration_id=c['configuration_id'], N_X=O.split(c['vector'])[0], N_Y=O.split(c['vector'])[1],
                    status=c['status'], MSE_lower=O.enc(c['MSE_lower']), MSE_upper=O.enc(c['MSE_upper']), source='RESUMED')
               for c in first_log.calls if c['vector'] in consumed]
    second_log = CallLog(chain())
    r2 = O.optimize(root, problem, context=context, methods=['ART'], initial='all', extra_results=records,
                    backend=second_log, max_new=2)['ART']
    out['runs']['3b_resume_budget_2'] = summary(r2, None)
    done = {(c['configuration_id'], tuple(c['vector'])) for c in first_log.calls if c['vector'] in consumed}
    repeated = [c for c in second_log.calls if (c['configuration_id'], tuple(c['vector'])) in done]
    first_new = second_log.calls[0]['vector'] if second_log.calls else None
    out['checks'] += [
        check('resume_does_not_repeat_completed_calls', not repeated and len(consumed) == r1['new_calls'],
              first_run_calls=len(first_log.calls), resumed_calls=len(second_log.calls), repeated=len(repeated)),
        check('resume_starts_from_previous_state', r2['trace'] and r2['trace'][0]['incumbent_before'] == best['cost']
              and (pend is None or first_new == list(pend['vector'])),
              previous_best=show(best['cost']), resumed_first_incumbent=show(r2['trace'][0]['incumbent_before'] if r2['trace'] else None))]
    return out


def show(value):
    return None if value is None else str(Fraction(value))


def summary(r, log):
    return dict(stop_reason=r['stop_reason'], new_calls=r['new_calls'],
                best=None if r['best'] is None else dict(candidate=r['best']['candidate'], cost=show(r['best']['cost']),
                                                       N_X=r['best']['N_X'], N_Y=r['best']['N_Y'], status=r['best']['status']),
                trace=[dict(step=t['step'], candidate=t['candidate'], vector=t['vector'], cost=show(t['cost']), status=t['status'],
                            source=t['answer_source'], origin=t['origin'].get('kind'), improved=t['improved']) for t in r['trace']],
                pending=None if r['pending_request'] is None else dict(candidate=r['pending_request']['candidate'],
                        vector=list(r['pending_request']['vector']), cost=show(r['pending_request']['cost']),
                        origin=r['pending_request']['origin'].get('kind'), route=r['pending_request'].get('route')),
                not_ready=[dict(candidate=c['candidate'], needs=c['needs']) for c in r['not_ready_candidates']],
                backend_queries=None if log is None else log.queries)


def run(optimizer_root, output):
    root = Path(optimizer_root).resolve()
    sys.path.insert(0, str(root / 'src'))
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    O = importlib.import_module('trottertracks.rpe_method_optimizer')
    base = O.load_json(root / PROBLEM)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    cases = []
    for fn, args in ((case1, (O, root, base)), (case2, (O, root, base)), (case3, (O, root))):
        try:
            cases.append(fn(*args))
        except Exception as exc:
            cases.append(dict(case=fn.__name__, checks=[dict(check='case_runs', status='FAIL',
                         error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc()[-800:])]))
    cases.append(dict(case='SKILL_ENTRY_B', checks=[dict(check='skill_entry_runs_the_same_cases', status='PENDING_NOT_PUBLISHED',
                      expected='The Skill passes the case input file to the optimizer entry and returns the same output fields; '
                               'the predicates above are applied to its output.')]))
    counts = {}
    for c in cases:
        for k in c['checks']:
            counts[k['status']] = counts.get(k['status'], 0) + 1
    result = dict(schema='A_RPE_OPTIMIZER_ACCEPTANCE_RESULT_V1', optimizer_root=str(root), optimizer_commit=head,
                  entry='trottertracks.rpe_method_optimizer.prepare/optimize', counts=counts, cases=cases,
                  science=dict(new_signal=0, new_price=0, compile=0, new_Q3=0, artificial_answers='LABELLED_ARTIFICIAL_FIXTURE'))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / 'ACCEPTANCE_RESULT.json').write_text(json.dumps(result, indent=1, default=str, ensure_ascii=False) + '\n')
    for c in cases:
        for k in c['checks']:
            print('%-34s %-58s %s' % (c['case'][:34], (k['check'] + (' [' + k['run'] + ']' if k.get('run') else ''))[:58], k['status']))
    print(json.dumps(counts))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--optimizer-root', default=str(HERE)); ap.add_argument('--output', required=True)
    a = ap.parse_args()
    res = run(a.optimizer_root, a.output)
    sys.exit(0 if set(res['counts']) <= {'PASS', 'PENDING_NOT_PUBLISHED'} else 1)
