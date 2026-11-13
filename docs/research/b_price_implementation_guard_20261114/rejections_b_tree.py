"""Record each B-direct rejection message and B.cost/B.select calls (artificial only).

Run from the B worktree root:  python -B <this file> <out.json>
Reuses the fixtures of tests/check_rpe_price_implementation_guard.py.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import check_rpe_price_implementation_guard as T

B, R, W = T.B, T.R, T.W
OLD, OLD_CODE, CODE = T.OLD, T.OLD_CODE, T.CODE


def counted(call):
    calls = dict(cost=0, select=0); cost, select = B.cost, B.select
    def c(*a, **k): calls['cost'] += 1; return cost(*a, **k)
    def s(*a, **k): calls['select'] += 1; return select(*a, **k)
    with patch.object(B, 'cost', side_effect=c), patch.object(B, 'select', side_effect=s):
        try:
            call(); error = None
        except ValueError as exc:
            error = str(exc)
    return error, calls


def fact(name, value, index=0):
    return lambda v: T.mutate_fact(v, name, value, index)


def physical(key, value):
    def change(v):
        side = v['options']['price_bindings'][0]
        side['physical_context']['physical_candidate'][key] = value
        v['proofs']['0X']['physical_context']['physical_candidate'][key] = value
    return change


def all_old(v):
    for i in range(len(v['prices'])):
        T.mutate_fact(v, 'rte_rotation_lowering', OLD, i); T.mutate_fact(v, 'rte_lowering_source', OLD_CODE, i)


def extension_only(v, assertion=False):
    for side in v['options']['price_bindings']:
        side['profile_extension']['lowering_id'] = OLD
        if assertion:
            v['proofs'][side['implementation_evidence']['selector'][0]]['profile_extension']['lowering_id'] = OLD
    v['options']['price_extension']['lowering_id'] = OLD


CASES = [
    ('1', 'OLD_FULLH_GENERIC_RZZ_IN_NEW_SCENARIO', 'FULL_H_RTE', all_old),
    ('1', 'ONE_POLICY_DECLARES_OLD_AND_NEW_RTE', 'PR',
     lambda v: v['options']['comparison_policy']['by_method']['FULL_H_RTE'].update(rte_rotation_lowering=OLD)),
    ('2', 'SAME_COMPILER_PR_FACTS_OLD_LOWERING', 'PR', fact('rte_rotation_lowering', OLD)),
    ('2', 'SAME_COMPILER_FULLH_NEW_ID_ON_OLD_EMITTER', 'FULL_H_RTE', fact('rte_lowering_source', OLD_CODE)),
    ('2', 'SIDECAR_EXTENSION_ONLY_CLAIMS_OLD', 'PR', extension_only),
    ('2', 'EXTENSION_AND_PINNED_ASSERTION_CLAIM_OLD_FACTS_NEW', 'PR', lambda v: extension_only(v, True)),
    ('3', 'IMPLEMENTATION_MISSING', 'PR', lambda v: v['options']['price_bindings'][0].pop('implementation')),
    ('3', 'UNKNOWN_LOWERING_ID', 'PR', fact('rte_rotation_lowering', 'UNKNOWN_LOWERING')),
    ('3', 'RTE_SOURCE_MISSING', 'PR', fact('rte_lowering_source', None)),
    ('3', 'NATIVE_SOURCE_MISSING', 'PR', fact('native_source', None)),
    ('3', 'WRAPPER_SOURCE_UNRESOLVED', 'PR', fact('wrapper_source', dict(commit='b'*40, path='MISSING.py'))),
    ('3', 'FOREIGN_PRICE_PRODUCER_SOURCE', 'PR', lambda v: v['prices'][0].update(source=dict(fixture='FOREIGN'))),
    ('3', 'ALTERED_PINNED_ASSERTION', 'PR', lambda v: v['proofs']['0X']['implementation'].update(native_policy='ALTERED')),
    ('3', 'UNPINNED_EVIDENCE_LOCATOR', 'PR',
     lambda v: v['options']['price_bindings'][0]['implementation_evidence'].update(commit='HEAD')),
    ('3', 'POLICY_WITHOUT_SOURCE', 'PR', lambda v: v['options']['comparison_policy'].pop('source')),
    ('4', 'SCALAR_TAIL_ONLY', 'PR', fact('scalar_quantity', 'TAIL_ONLY')),
    ('4', 'SCALAR_APPLIED_TWICE', 'TRUNCATED_PF', fact('scalar_application', 'TWICE')),
    ('4', 'RESOLVED_SCALAR_DIFFERS', 'FULL_PF', fact('resolved_scalar', dict(numerator=3, denominator=8))),
    ('4', 'WRAPPER_WITHOUT_MEASUREMENT', 'FULL_H_RTE',
     fact('profile', dict(T.policy()['profile'], wrapper='WITHOUT_MEASUREMENT'))),
    ('4', 'AXIS_MEASUREMENT_FACT', 'PR', fact('measurement', dict(axis='FOREIGN'))),
    ('4', 'PRICE_RECORD_AXIS', 'PR', lambda v: v['prices'][0].update(measurement_axis='sine')),
    ('4', 'METHOD_FACT', 'PR', fact('method', 'FULL_PF')),
    ('4', 'PR_CANDIDATE_T', 'PR', physical('T', 9.)),
    ('4', 'PR_CANDIDATE_ID', 'PR', physical('id', 'FOREIGN_ID')),
    ('4', 'TRUNCATED_CANDIDATE_METHOD', 'TRUNCATED_PF', physical('method', 'FULL')),
    ('4', 'FULL_PF_CANDIDATE_Q', 'FULL_PF', physical('q', 99)),
    ('4', 'FULLH_CANDIDATE_R_OUTER', 'FULL_H_RTE', physical('R_outer', 3)),
    ('4', 'SIDECAR_IDENTITY_OTHER_ROUND', 'PR',
     lambda v: v['options']['price_bindings'][0].update(identity=deepcopy(v['config']['rounds'][1]['identity']))),
    ('5', 'FULL_PF_WITH_RTE_SOURCE', 'FULL_PF', fact('rte_lowering_source', CODE)),
    ('5', 'TRUNCATED_PF_WITH_RTE_SOURCE', 'TRUNCATED_PF', fact('rte_lowering_source', CODE)),
    ('5', 'FULL_PF_POLICY_DEMANDS_RTE_LOWERING', 'FULL_PF',
     lambda v: v['options']['comparison_policy']['by_method']['FULL_PF'].update(rte_rotation_lowering=T.NEW)),
]
rows = []
for item, label, method, change in CASES:
    v = T.fixture(method); change(v)
    for route, call in (('W.compare', lambda: T.compare(v)), ('guarded_cost', lambda: T.direct_cost(v))):
        error, calls = counted(call)
        rows.append(dict(item=item, case=label, method=method, route=route, rejected=error is not None,
                         message=error, B_cost_calls=calls['cost'], B_select_calls=calls['select']))
pr = T.compare(T.fixture('PR'))
old = T.compare(T.fixture('FULL_H_RTE', lowering=OLD, ident='ARTIFICIAL_OLD_RZZ'))
error, calls = counted(lambda: R.select([T.configuration('PR'), T.configuration('FULL_H_RTE')],
                                        [pr['outer_evaluation'], old['outer_evaluation']], required_methods=T.RTE))
rows.append(dict(item='1', case='SELECT_NEW_PR_WITH_OLD_FULLH_POLICY', method='PR+FULL_H_RTE', route='guarded_select',
                 rejected=error is not None, message=error, B_cost_calls=calls['cost'], B_select_calls=calls['select']))
v = T.fixture(kind='SCIENTIFIC'); options = dict(v['options']); options.pop('comparison_policy')
out = None
def science():
    global out
    out = W.compare(v['config'], v['allocations'], v['evaluations'], v['prices'], **options)
error, calls = counted(science)
rows.append(dict(item='7', case='SCIENTIFIC_N_WITHOUT_POLICY', method='PR', route='W.compare', rejected=False,
                 message=None, pending=out['pending_allocation_ids'], G=[r['G'] for r in out['allocations']],
                 reasons=sorted({x for r in out['allocations'] for x in r['evaluation']['reasons']}),
                 B_cost_calls=calls['cost'], B_select_calls=calls['select']))
ok = (all(r['rejected'] and r['B_cost_calls'] == 0 and r['B_select_calls'] == 0 for r in rows[:-1])
      and rows[-1]['B_cost_calls'] == rows[-1]['B_select_calls'] == 0 and rows[-1]['G'] == [None, None])
Path(sys.argv[1]).write_text(json.dumps(dict(scope='ARTIFICIAL_FIXTURES_ONLY', rows=rows, all_rejected_before_arithmetic=ok),
                                        indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(rows=len(rows), all_rejected_before_arithmetic=ok)))
