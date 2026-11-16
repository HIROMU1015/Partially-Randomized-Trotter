"""Connection of A's PR-Pauli candidate preparation entry (fixed SOURCE 6e4b6bd6a) to the optimizer.

Saved results (real entry on the pinned view, metadata only -- no dictionary, mean, price fit, compile or Q3):
  unregistered L_D of the search range go to the entry before generate_variants keeps only configurations with an id:
  LD1 (saved configuration, no saved price) is registered; LD4/6/7/8/9 stay preparation requests with A's next actions;
  saved LD2/3/5 and the kappa variants are unchanged.  A kappa proposal for the published LD3 kappa x2 (and LD5 x1)
  returns the already registered configuration.  The search never evaluates a registered configuration that lacks
  signals / price / route.
Artificial (labelled stand-in entry): new complete configuration without price -> registered; preparation request kept;
  signal rounds to request carried as needs; existing id -> no duplicate; failed call -> kept with a failure need.
Writes PREPARATION_CONNECTION.json into --output.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'));sys.path.insert(0, str(ROOT / 'scripts'))
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402
from trottertracks import rpe_candidate_preparation as R  # noqa: E402
import run_rpe_method_optimizer as CLI  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
PROBLEM = ROOT / 'configs/rpe_method_optimizer/h4_df10_problem.json'
LD1 = 'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27'


def check(name, ok, **detail):
    return dict(check=name, status='PASS' if ok else 'FAIL', **detail)


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output;out.mkdir(parents=True, exist_ok=True);checks = []
    base_problem = O.load_json(PROBLEM)
    grid = dict(formulas=['2nd'], delta_over_h_values=['1'], kappa_multipliers=['2/3', '1', '2'])
    saved_only = O.prepare(ROOT, CLI.scoped(base_problem, {'PR_PAULI': dict(saved_configurations_only=True, generation=grid)}), methods=['PR_PAULI'])
    problem = CLI.scoped(base_problem, {'PR_PAULI': dict(saved_configurations_only=False, generation=grid)})
    ctx = O.prepare(ROOT, problem, methods=['PR_PAULI'], preparation_dir=out / 'saved_results_preparation')
    pp = [c for c in ctx['candidates'] if c['method'] == 'PR_PAULI'];prep = problem['methods']['PR_PAULI']['generation']['preparation']
    by_ld = {}
    for c in pp:
        by_ld.setdefault(c['key']['L_D'], []).append(c)
    ld1 = next((c for c in pp if c['configuration_id'] == LD1), None)
    pending = {c['key']['L_D']: c for c in pp if c['configuration_id'] is None}
    rows = ctx['generation']['PR_PAULI'].get('preparation') or []
    checks.append(check('LD1_registered_without_price_requirement', ld1 is not None and ld1.get('prepared_by') == 'A_PAULI_CANDIDATE_PREPARATION'
                        and not ld1['ready'] and 'Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION' in ld1['needs']
                        and ld1['preparation']['signal_counts'] == {'REUSE': 13} and ld1.get('configuration_source') is not None,
                        needs=ld1 and ld1['needs'], price_status=ld1 and ld1['price_status'], price_quote=ld1 and ld1.get('price_quote'),
                        readiness=ld1 and ld1['preparation']['readiness']))
    acts = {ld: c['needs'] for ld, c in pending.items()}
    checks.append(check('unregistered_L_D_kept_as_preparation_requests', sorted(pending) == [4, 6, 7, 8, 9]
                        and all(n[0].startswith('SPLIT_PREPARATION:') for n in acts.values())
                        and all(acts[ld][0].startswith('SPLIT_PREPARATION:REUSE_DESCRIPTOR') for ld in (7, 9))
                        and all(acts[ld][0].startswith('SPLIT_PREPARATION:DERIVE_DESCRIPTOR') for ld in (4, 6, 8))
                        and all(pending[ld].get('preparation_request') for ld in pending),
                        needs=acts, rows=[(r['L_D'], r['status'], r['disposition']) for r in rows]))
    same = {c['configuration_id']: c['prices'] for c in saved_only['candidates'] if c['method'] == 'PR_PAULI' and c['configuration_id']}
    now = {c['configuration_id']: c['prices'] for c in pp if c['configuration_id'] in same}
    checks.append(check('saved_and_kappa_variant_candidates_unchanged', now == same and len(same) == 9,
                        configurations=len(same)))

    # Kappa proposals of the rule for saved/published configurations return the registered candidates.
    points = {}
    for pt in ctx['catalog']['candidates']:
        points.setdefault(pt['configuration_id'], pt)
    bases = {c['key']['L_D']: dict(source=points[c['configuration_id']]['configuration_source'], configuration_id=c['configuration_id'])
             for c in pp if c['configuration_id'] in points and not c.get('generated')}
    proposals = [dict(kind='KAPPA_PROPOSAL', family=dict(L_D=3, formula='2nd', delta_over_h='1'), kappa_multiplier='2'),
                 dict(kind='KAPPA_PROPOSAL', family=dict(L_D=5, formula='2nd', delta_over_h='1'), kappa_multiplier='1')]
    n_before = len(pp);k = R.prepare_kappa(ROOT, 'PR_PAULI', prep, ctx['candidates'], proposals, bases, out / 'saved_results_preparation')
    ids = {c['configuration_id'] for c in ctx['candidates'] if c['method'] == 'PR_PAULI'}
    checks.append(check('kappa_proposal_path_returns_registered_configurations', [r['disposition'] for r in k] == ['ALREADY_REGISTERED'] * 2
                        and k[0]['configuration_id'].startswith('23ec1791') and k[1]['configuration_id'].startswith('4d6e2c9c')
                        and k[0]['signal_counts'] == {'REUSE': 13} and len([c for c in ctx['candidates'] if c['method'] == 'PR_PAULI']) == n_before,
                        rows=[(r['item_id'], r['status'], r['disposition'], r['configuration_id'][:8], r['signal_counts']) for r in k],
                        configurations=len(ids)))

    # The search never evaluates a registered configuration that lacks signals / price / route.
    calls = []

    class Spy:
        def answer(self, p, allow_new=True):
            calls.append(p['configuration_id'])
            return None
    kn = O.initial_knowledge(ctx['saved'], pp, 'all')
    r = P.run_priority('PR_PAULI', pp, kn, Spy(), dict(problem['search'], max_steps_per_method=50), ctx['tau'], ctx['generate'], None,
                       baseline=problem['methods']['PR_PAULI']['generation']['baseline'], on_missing='SKIP',
                       free=lambda p: True, facts=ctx['facts'])
    reqs = {q.get('candidate'): q for q in r['input_requests'] if q['kind'] in ('NEW_L_D', 'CONFIGURATION_INPUTS_MISSING')}
    checks.append(check('registered_unready_configuration_never_evaluated', LD1 not in calls and calls
                        and any(q.get('configuration_id') == LD1 for q in reqs.values())
                        and sum(1 for q in reqs.values() if q['kind'] == 'NEW_L_D') == 5,
                        asked=len(calls), requests=sorted((q['kind'], q.get('candidate')) for q in reqs.values())))

    # Artificial entry.
    art = dict(kind='artificial', entry='scripts/artificial_pauli_preparation_entry.py', interpreter=sys.executable, K=2,
               artificial_responses={'L_D=4': dict(status='GENERATED', configuration_id='ART4' * 16, configuration_source=None),
                                     'L_D=6': dict(status='NEEDS_SPLIT_PREPARATION', complete=False, next_actions=['DERIVE_DESCRIPTOR', 'P0']),
                                     'L_D=7': dict(status='GENERATED', configuration_id='ART7' * 16, configuration_source={'commit': 'x'},
                                                   request_rounds=[11, 12]),
                                     'L_D=8': dict(status='GENERATED', configuration_id='EXISTING' * 8)})
    caps = [12] * 13

    def cand(ld, cid=None):
        return dict(method='PR_PAULI', family='PR', key=dict(L_D=ld), configuration_id=cid, ready=cid is not None, prices=None, caps=caps,
                    name='PR_PAULI:L_D=%d:%s' % (ld, (cid or 'NONE')[:8]), needs=[] if cid else ['CONFIGURATION_AND_SCHEDULE'], labels=[])
    cs = [cand(4), cand(6), cand(7), cand(8), cand(5, 'EXISTING' * 8)]
    rows = R.prepare_unregistered(ROOT, 'PR_PAULI', art, cs, dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'), out / 'artificial')
    got = {c['key']['L_D']: c for c in cs}
    checks.append(check('artificial_registration_dispositions', [r['disposition'] for r in rows] == ['REGISTERED_WITHOUT_PRICE_REQUIREMENT',
                        'PREPARATION_REQUEST', 'REGISTERED_WITHOUT_PRICE_REQUIREMENT', 'ALREADY_REGISTERED']
                        and got[4]['configuration_id'] == 'ART4' * 16 and got[4]['prices'] is None and not got[4]['ready']
                        and 'CONFIGURATION_PUBLICATION' in got[4]['needs'] and 'PRICE_AXES:pending_quote' in got[4]['needs']
                        and got[6]['configuration_id'] is None and got[6]['needs'][0] == 'SPLIT_PREPARATION:DERIVE_DESCRIPTOR,P0'
                        and 'SIGNAL_ROUNDS:11,12' in got[7]['needs'] and 'CONFIGURATION_PUBLICATION' not in got[7]['needs']
                        and 8 not in got and len(cs) == 4,
                        rows=[(r['L_D'], r['status'], r['disposition']) for r in rows], needs={ld: c['needs'] for ld, c in got.items()}))
    fail = dict(art, artificial_responses={'*fail*': True});cf = [cand(4), cand(6)]
    rf = R.prepare_unregistered(ROOT, 'PR_PAULI', fail, cf, dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'), out / 'artificial_fail')
    checks.append(check('failed_entry_call_keeps_candidates_with_failure_need', all(c['configuration_id'] is None and not c['ready']
                        and c['needs'] == ['PREPARATION_PREPARATION_CALL_FAILED'] for c in cf) and len(rf) == 2,
                        needs=[c['needs'] for c in cf]))
    try:
        R.check_view(dict(prep, view_commit='0' * 40));wrong = False
    except ValueError:
        wrong = True
    checks.append(check('entry_refused_unless_view_at_fixed_source', wrong, view=prep['view'], fixed=prep['view_commit']))

    counts = {}
    for c in checks:
        counts[c['status']] = counts.get(c['status'], 0) + 1
    res = dict(schema='RPE_PREPARATION_CONNECTION_V1', optimizer_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               preparation_source=prep['view_commit'], handoff=prep.get('handoff'), counts=counts, checks=checks,
               science='saved results (A entry metadata only: no dictionary, mean, price fit, compile, Q3) + ARTIFICIAL entry')
    (out / 'PREPARATION_CONNECTION.json').write_text(json.dumps(O.jsonable(res), indent=1, ensure_ascii=False, default=str) + '\n')
    for c in checks:
        print('%-60s %s' % (c['check'], c['status']))
    print(json.dumps(counts))
    return 0 if set(counts) == {'PASS'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
