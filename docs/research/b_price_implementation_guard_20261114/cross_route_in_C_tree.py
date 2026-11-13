"""C route vs B direct route on C's own artificial fixtures (no real inputs).

Run from a checkout of C 3a8b37f1a (tests importable, PYTHONPATH=src:tests):
    python -B cross_route_in_C_tree.py <C_root> <B_commit> <out.json>

C route : rpe_optimization_pipeline.compare/select (C's guard, then pinned B W).
B route : the B commit's rpe_shot_allocation.compare + rpe_price_implementation_guard,
          executed in the SAME private package as C's pinned controller, so both
          routes share one B.cost/B.select object. Prices, N and G are fictional.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve(); B_COMMIT = sys.argv[2]; OUT = Path(sys.argv[3])
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import check_price_implementation_guard as CT  # C's guard tests: fixtures, bind(), policy()

I, B, G, T = CT.I, CT.B, CT.G, CT.T
API = I.providers(str(ROOT)); PKG = sys.modules[API.B.__package__]


def load(short, path):
    name = PKG.__name__ + '.' + short
    module = types.ModuleType(name); module.__package__ = PKG.__name__; module.__file__ = str(ROOT/path)
    sys.modules[name] = module; setattr(PKG, short, module)
    exec(compile(G.C.blob(ROOT, B_COMMIT, path), B_COMMIT + ':' + path, 'exec'), module.__dict__)
    return module


R = load('rpe_price_implementation_guard', 'src/trottertracks/rpe_price_implementation_guard.py')
WB = load('rpe_shot_allocation_b_direct', 'src/trottertracks/rpe_shot_allocation.py')
RG = R.guard(ROOT)
assert WB.B is API.B and R.B is API.B and RG is not G
assert RG.check.__code__.co_code == G.check.__code__.co_code
FIXTURES = {m: T.fixture(m) for m in ('PR', 'FULL_H_RTE', 'FULL', 'B0I')}


def c_route(v, proofs, **extra):
    with patch.object(G, '_document', return_value=proofs):
        return I.compare(ROOT, *v[:5], **dict(v[5], **extra))


# I.compare/I.select pass their own default scope labels; give B the same inputs.
SCOPE = dict(compare='OUTSIDE_EXPLICIT_ALLOCATION_MENU', select='OUTSIDE_EXPLICIT_CONFIGURATION_MENU')


def b_route(v, proofs, **extra):
    o = dict(v[5], unexplored_scope=SCOPE['compare'], **extra); policy = o.pop('comparison_policy', None)
    with patch.object(RG, '_document', return_value=proofs):
        return WB.compare(v[0]['configuration'], v[2], v[3], [p for p in v[4] if p.get('mean') is not None],
                          root=ROOT, comparison_policy=policy, **o)


def outcome(call):
    calls = []
    original = B.cost
    def counted(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    with patch.object(B, 'cost', side_effect=counted):
        try:
            out = call(); error = None
        except ValueError as exc:
            out = None; error = str(exc)
    return out, error, len(calls)


def strip(evaluation):
    return None if evaluation is None else {k: v for k, v in evaluation.items() if k != 'price_implementation_check'}


cases = []; c_results = []; b_outer = []; configs = []
for method in FIXTURES:
    v, proofs = CT.bind(FIXTURES[method])
    c, ce, cn = outcome(lambda: c_route(v, proofs)); b, be, bn = outcome(lambda: b_route(v, proofs))
    shared = sorted(set(c['comparison']) - {'outer_evaluation'})
    same = (ce is None and be is None and all(c['comparison'][k] == b[k] for k in shared)
            and strip(c['comparison']['outer_evaluation']) == strip(b['outer_evaluation'])
            and c['price_implementation_check'] == b['price_implementation_check'])
    cases.append(dict(case='VALID_SAME_DOMAIN', method=method, consistent=same,
        G_C=[r['G'] for r in c['comparison']['allocations']], G_B=[r['G'] for r in b['allocations']],
        checked_cells=len(b['price_implementation_check']['checked']), B_cost_calls_C=cn, B_cost_calls_B=bn))
    c_results.append(c); b_outer.append(b['outer_evaluation']); configs.append(v[0]['configuration'])

c_select, cse, _ = outcome(lambda: I.select(ROOT, c_results))
b_select, bse, _ = outcome(lambda: R.select(configs, b_outer, unexplored_scope=SCOPE['select']))
cases.append(dict(case='SELECT_ONE_COMMON_POLICY', consistent=cse is None and bse is None and c_select == b_select,
    winner=b_select['winner'], menu_complete=b_select['menu_complete']))


def bind_old(method):
    v, proofs = CT.bind(FIXTURES[method]); o = v[5]
    o['comparison_policy']['id'] = 'ART_OLD_SCENARIO'
    for m in ('PR', 'FULL_H_RTE'):
        o['comparison_policy']['by_method'][m]['rte_rotation_lowering'] = CT.OLD
    o['price_extension']['lowering_id'] = CT.OLD
    for side in o['price_bindings']:
        proof = proofs[side['implementation_evidence']['selector'][0]]
        for target in (side, proof):
            target['implementation'].update(rte_rotation_lowering=CT.OLD, rte_lowering_source=CT.OLD_CODE)
            target['profile_extension']['lowering_id'] = CT.OLD
    return v, proofs


v, proofs = bind_old('FULL_H_RTE')
c_old, _, _ = outcome(lambda: c_route(v, proofs)); b_old, _, _ = outcome(lambda: b_route(v, proofs))
_, ce, _ = outcome(lambda: I.select(ROOT, [c_results[0], c_old]))
_, be, _ = outcome(lambda: R.select([configs[0], v[0]['configuration']], [b_outer[0], b_old['outer_evaluation']],
                                    required_methods=('PR', 'FULL_H_RTE')))
cases.append(dict(case='SELECT_NEW_PR_WITH_OLD_FULLH_POLICY', consistent=ce is not None and ce == be, C=ce, B=be))


def mutate(method, change):
    v, proofs = CT.bind(FIXTURES[method]); change(v, proofs)
    return v, proofs


def fact(name, value, index=0):
    def change(v, proofs):
        side = v[5]['price_bindings'][index]
        for target in (side, proofs[side['implementation_evidence']['selector'][0]]):
            target['implementation'][name] = deepcopy(value)
    return change


def physical(key, value):
    def change(v, proofs):
        side = v[5]['price_bindings'][0]
        for target in (side, proofs[side['implementation_evidence']['selector'][0]]):
            target['physical_context']['physical_candidate'][key] = value
    return change


def all_old(v, proofs):
    for i in range(len(v[5]['price_bindings'])):
        fact('rte_rotation_lowering', CT.OLD, i)(v, proofs)


REJECT = [('OLD_FULLH_RZZ_IN_NEW_POLICY', 'FULL_H_RTE', all_old),
          ('ONE_POLICY_DECLARES_OLD_AND_NEW', 'PR',
           lambda v, p: v[5]['comparison_policy']['by_method']['FULL_H_RTE'].update(rte_rotation_lowering=CT.OLD)),
          ('UNKNOWN_LOWERING', 'PR', fact('rte_rotation_lowering', 'UNKNOWN')),
          ('NEW_ID_ON_OLD_EMITTER', 'PR', fact('rte_lowering_source', CT.OLD_CODE)),
          ('MISSING_IMPLEMENTATION', 'PR', lambda v, p: v[5]['price_bindings'][0].pop('implementation')),
          ('ALTERED_PINNED_ASSERTION', 'PR', lambda v, p: p['0X']['implementation'].update(native_policy='ALTERED')),
          ('FOREIGN_PRICE_SOURCE', 'PR', lambda v, p: v[4][0].update(source={'fixture': 'FOREIGN'})),
          ('SCALAR_TAIL_ONLY', 'PR', fact('scalar_quantity', 'TAIL_ONLY')),
          ('SCALAR_TWICE', 'B0I', fact('scalar_application', 'TWICE')),
          ('WRAPPER_WITHOUT_MEASUREMENT', 'FULL', fact('profile', dict(CT.policy()['profile'], wrapper='NO_MEASUREMENT'))),
          ('AXIS_MEASUREMENT', 'PR', fact('measurement', {'axis': 'FOREIGN'})),
          ('FULLH_PHYSICAL_T', 'FULL_H_RTE', physical('T', 999)),
          ('PR_CANDIDATE_ID', 'PR', physical('id', 'FOREIGN_ID')),
          ('PF_WITH_RTE_SOURCE', 'FULL', fact('rte_lowering_source', CT.CODE)),
          ('TRUNCATED_WITH_RTE_SOURCE', 'B0I', fact('rte_lowering_source', CT.CODE))]
for label, method, change in REJECT:
    v, proofs = mutate(method, change)
    _, ce, cn = outcome(lambda: c_route(v, proofs)); _, be, bn = outcome(lambda: b_route(v, proofs))
    cases.append(dict(case=label, method=method, consistent=ce is not None and ce == be and cn == bn == 0,
                      C=ce, B=be, B_cost_calls_C=cn, B_cost_calls_B=bn))

v, proofs = CT.bind(FIXTURES['PR']); v[4].pop()
c, ce, cn = outcome(lambda: c_route(v, proofs)); b, be, bn = outcome(lambda: b_route(v, proofs))
cases.append(dict(case='MISSING_PRICE_STAYS_UNKNOWN', consistent=(ce is None and be is None and cn == bn == 0
    and c['comparison']['point_minimum']['cost'] is None and b['point_minimum']['cost'] is None
    and c['price_implementation_check'] == b['price_implementation_check']),
    checked_cells=len(b['price_implementation_check']['checked'])))

v = deepcopy(FIXTURES['PR'])
for evaluation in v[3]:
    evaluation['n_packet']['evidence_kind'] = 'SCIENTIFIC'
c, ce, cn = outcome(lambda: I.compare(ROOT, *v[:5], **v[5]))
o = dict(v[5], unexplored_scope=SCOPE['compare'])
b, be, bn = outcome(lambda: WB.compare(v[0]['configuration'], v[2], v[3], v[4], **o))
flag = 'MISSING_PRICE_COMPARISON_IMPLEMENTATION_POLICY'
cases.append(dict(case='SCIENTIFIC_N_WITHOUT_POLICY_PENDING', consistent=(ce is None and be is None and cn == bn == 0
    and flag in c['blockers'] and all(flag in r['evaluation']['reasons'] for r in b['allocations'])
    and c['comparison']['outer_evaluation'] is None and b['outer_evaluation'] is None),
    C_blockers=c['blockers'], B_pending=b['pending_allocation_ids']))

import subprocess
result = dict(scope='ARTIFICIAL_FIXTURES_ONLY_NO_REAL_INPUT_PRICE_N_G',
    C_checkout=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    B_commit=B_COMMIT, aligned_scope_labels=SCOPE, cases=cases, all_consistent=all(c['consistent'] for c in cases))
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + '\n')
print(json.dumps(dict(cases=len(cases), all_consistent=result['all_consistent'],
                      inconsistent=[c['case'] for c in cases if not c['consistent']])))
