"""Metadata counting, source binding and the normal publication sidecar only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from trottertracks.h4_search_map import build_index, dimensions
from trottertracks.saved_rpe_cost_catalog import PASS,FAIL
from publish_h4_search_run import prepare


def catalog():
    def row(status,cost,n):
        return dict(method='TRUNCATED_PF',configuration_id='p7',formula='2nd',L_D=7,split={'L_D':7},
            h={'numerator':1,'denominator':10},delta_pf={'numerator':1,'denominator':5},
            N_X=[n]*13,N_Y=[n]*13,q3_status=status,status='FEASIBLE'if status==PASS else'INFEASIBLE',
            conditional_cost={'numerator':cost,'denominator':1},adopted_evaluation={'tau':'fixed'},
            signal_source={'commit':'s','path':'signal.json'},binding_id='b',total_configuration_id='t',
            result_sources=[{'commit':'o','path':'out.json'}],price_axes=[],issues=[])
    return dict(comparison={'accuracy':{'tau':'fixed'}},candidates=[row(FAIL,100,1),row(PASS,120,2)])


def run(rid='r1',new=True):
    ev={k:v for k,v in catalog()['candidates'][1].items()if k not in ('status','result_sources')}
    ev.update(source={'commit':'o','path':'eval.json'},new_evaluation=new,origin='NEW_Q3_JOB'if new else'PRIOR_ROUND_JOB')
    return dict(id=rid,label=rid,source={'commit':'r','path':'ROUND_RESULT.json'},status='STOP',hosts=['executor-A'],
        targets=[dict(method='TRUNCATED_PF',configuration_id='p8',p=8,formula='4th',requested_width='1/5',
                      h='1/10',status='SOURCE_NOT_IMPORTED',missing=['OUTCOME_NOT_PUBLIC'],source={'commit':'r','path':'scope.json'})],
        evaluations=[ev])


class MetadataMap(unittest.TestCase):
    def test_same_evaluation_replay_and_other_run_reuse_do_not_inflate(self):
        c=catalog();a=run();b=run('r2',False)
        index=build_index(c,[a,b]);repeat=build_index(c,[a,b])
        self.assertEqual(index,repeat);self.assertEqual(len(index['points']),2)
        cell=next(x for x in index['views'][0]['cells']if x['row_key']=='7')
        self.assertEqual((cell['physical_evaluated'],cell['evaluated_allocations'],cell['feasible_allocations']),(1,2,1))
        self.assertEqual(cell['min_cost'],{'numerator':120,'denominator':1})
        self.assertEqual(sum(x['evaluated_allocations']for x in next(v for v in index['views']if v['id']=='new:r2')['cells']),0)
        self.assertEqual(sum(x['evaluated_allocations']for x in next(v for v in index['views']if v['id']=='new:r1')['cells']),1)
        self.assertTrue(any(x['state']=='SOURCE_NOT_IMPORTED'for x in index['views'][0]['cells']))

    def test_unevaluated_no_feasible_and_summary_price_not_promoted(self):
        c=catalog();c['candidates']=c['candidates'][:1]
        r=run();e=r['evaluations'][0];e['configuration_id']='summary-only';e['conditional_cost']={'numerator':1,'denominator':1}
        out=build_index(c,[r]);p=next(p for p in out['points']if p['configuration_id']=='summary-only')
        self.assertIsNone(p['conditional_cost']);self.assertEqual(p['saved_summary_cost'],{'numerator':1,'denominator':1})
        self.assertFalse(any(x['min_cost']for x in out['views'][0]['cells']))
        c['candidates'][0]['delta_pf']='1/4'
        out=build_index(c,[]);self.assertEqual(out['views'][0]['cells'][0]['state'],'EVALUATED_NO_FEASIBLE')

    def test_exact_width_and_binary64_width_stay_distinct(self):
        base=dict(method='PR_PAULI',L_D=5,h='1/10')
        self.assertNotEqual(dimensions(dict(base,requested_width=0.2))['width_key'],
                            dimensions(dict(base,requested_width='1/5'))['width_key'])

    def test_one_parameter_delta_uses_admitted_points_and_marks_missing_cost(self):
        c = catalog()
        base = dict(c['candidates'][1], M=12, profile={'compiler':'profile'}, law={'model':'deterministic'})
        lower = dict(base, configuration_id='p6', L_D=6, split={'L_D':6},
                     conditional_cost={'numerator':130,'denominator':1})
        wider = dict(base, configuration_id='p7-wide', delta_pf='1/4',
                     conditional_cost={'numerator':115,'denominator':1}, N_X=[3]*13)
        waiting = dict(wider, configuration_id='p7-far', delta_pf='1/3',
                       status='PRICE_WAITING', conditional_cost=None)
        mismatched = dict(wider, configuration_id='p7-other-profile', delta_pf='2/5',
                          profile={'compiler':'different'})
        summary = run()
        summary['evaluations'][0].update(configuration_id='summary-price', delta_pf='1/6')
        c['candidates'] = [base, lower, wider, waiting, mismatched]
        records = build_index(c, [summary])['direction_records']
        p_edge, *unused = [edge for edge in records if edge['parameter']=='p']
        self.assertEqual((p_edge['from_value'],p_edge['to_value'],p_edge['delta_RZ']),
                         ('6','7',{'numerator':'-10','denominator':'1'}))
        self.assertFalse(unused)
        width_edges = [edge for edge in records if edge['parameter']=='width']
        self.assertEqual([(edge['direction'],edge['delta_RZ']) for edge in width_edges],
                         [('COST_DOWN',{'numerator':'-5','denominator':'1'}),('UNRESOLVED',None)])
        self.assertFalse(width_edges[0]['same_allocation'])

    def test_horizon_direction_and_different_price_conditions(self):
        base = dict(catalog()['candidates'][1], M=12, profile={'compiler':'same'},
                    price_scenario='FIXED_CONDITIONAL_PRICE')
        lower = dict(base, configuration_id='short', M=11, N_X=[2]*12, N_Y=[2]*12)
        other = dict(base, configuration_id='long', M=13, N_X=[2]*14, N_Y=[2]*14,
                     price_scenario='DIFFERENT_PRICE_CONDITION')
        records = build_index(dict(catalog(), candidates=[base, lower, other]), [])['direction_records']
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]['parameter'], records[0]['from_value'], records[0]['to_value'],
                          records[0]['direction']), ('M', '11', '12', 'EQUAL'))
        self.assertFalse(records[0]['same_allocation'])

    def test_kappa_aliases_with_exactly_equal_round_identities_are_one_schedule(self):
        base = dict(catalog()['candidates'][1], method='PR_PAULI', M=12, L_D=7, finite_K=2,
                    profile={'compiler':'profile'}, law={'model':'PR'})
        def row(cid, k, tag, cost):
            return dict(base, configuration_id=cid, kappa={'kind':'PAPER','value':k},
                        schedule=[{'m':m,'physical_id':tag+str(m)} for m in range(13)],
                        conditional_cost={'numerator':cost,'denominator':1})
        data = dict(catalog(), candidates=[row('one',1,'A',100),row('alias',2,'A',100),row('new',3,'B',110)])
        records=build_index(data,[])['direction_records']
        self.assertEqual(len(records),1)
        self.assertEqual((records[0]['parameter'],records[0]['from_value'],records[0]['to_value'],
                          records[0]['direction']),('kappa','1','3','COST_UP'))

    def test_other_snapshot_publication_is_not_joined_to_h4_map(self):
        c=catalog();c['comparison']['binding']={'system':'H4','snapshot':'A'}
        r=run();r['evaluations'][0]['physical_binding']={'system':'H4','snapshot':'B'}
        out=build_index(c,[r]);self.assertTrue(any('DIFFERENT_HAMILTONIAN' in i['reason']for i in out['issues']))
        self.assertTrue(all(not p['occurrences']for p in out['points']))

    def test_publication_sidecar_is_metadata_only_and_idempotent(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);run=root/'docs/research/a/run01';run.mkdir(parents=True)
            (run/'ROUND_RESULT.json').write_text(json.dumps({'schema':'RPE_OPTIMIZER_ROUND_RESULT_V1','methods':{}}))
            before=(run/'ROUND_RESULT.json').read_bytes();out=root/'docs/research/a/SEARCH_PUBLICATION.json'
            status,value=prepare(root,run,out,run_id='controller',execution_hosts=['hostA','hostB'])
            self.assertEqual(status,'UPDATED');self.assertEqual(value['role'],'CONTROLLER')
            self.assertEqual(prepare(root,run,out,run_id='controller',execution_hosts=['hostB','hostA'])[0],'UNCHANGED')
            self.assertEqual(before,(run/'ROUND_RESULT.json').read_bytes())
            (run/'ROUND_RESULT.json').write_text(json.dumps({'artificial_only':True}))
            with self.assertRaises(ValueError):prepare(root,run,out)


if __name__=='__main__':unittest.main(verbosity=2)
