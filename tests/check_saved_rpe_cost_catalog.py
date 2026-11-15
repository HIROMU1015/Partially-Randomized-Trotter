"""Synthetic binding checks and the explicitly requested saved-result acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from trottertracks.saved_rpe_cost_catalog import Catalog, Reader, PASS, FAIL, frac, shots, ref

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json'


class FakeReader:
    def __init__(self,values):self.values=values
    def read(self,r):return self.values[r['path']]


def fixture():
    condition=dict(binding=dict(snapshot='mock',hamiltonian='mockH',state='mockpsi'),h=refnum(1,10),M=12,
        accuracy=dict(estimator='mock_decoder',target='mock_target',loss='mock_loss',epsilon_E_decimal='0.01',MSE_upper=refnum(1,10000)))
    spec=dict(binding=condition['binding'],h=condition['h'],M=12,profile={'P':0,'compiler':'mock'},method='PR',L_D=1,
        formula='mock_formula',delta_pf=refnum(1,10),law={'model':'mock_law'},split={})
    config=dict(id='configuration',spec=spec,rounds=[dict(m=m,q_PF=1,R_outer=2,actual_delta_pf=refnum(1,10),
        T=refnum(2**m,10),candidate={'id':'c'+str(m)})for m in range(13)])
    values={'configuration':config};terms=[]
    for m in range(13):
        for a in ('X','Y'):
            path=f'price{m}{a}';values[path]=dict(m=m,axis=a,physical_id='c'+str(m),q=1,R_outer=2)
            terms.append(dict(m=m,axis=a,N=2,price_source={'path':path}))
    case=dict(configuration={'commit':'a'*40,'path':'configuration'},original_configuration_id='configuration',label='mock')
    alloc=dict(shots=[dict(m=m,N_X=2,N_Y=2)for m in range(13)])
    return Catalog(FakeReader(values),condition),case,alloc,deepcopy(condition['accuracy']),terms,values


def refnum(n,d):return dict(numerator=n,denominator=d)


class Technical(unittest.TestCase):
    def test_missing_q3_and_missing_price_and_saved_failure_are_distinct(self):
        cat,case,alloc,adopted,terms,_=fixture()
        for status,cost in [(None,None),(PASS,None),(FAIL,20),('UNKNOWN',20),(PASS,30)]:
            cat.make(case,alloc,status,cost,None,None,adopted,{'path':str(status)},terms if cost else [])
        self.assertEqual([r['status']for r in cat.rows],['Q3_WAITING','PRICE_WAITING','INFEASIBLE','PENDING','FEASIBLE'])
        self.assertEqual(cat.rows[-1]['conditional_cost'],refnum(30,1))

    def test_snapshot_precision_axis_allocation_physical_and_config_mismatch(self):
        for mode in ('snapshot','precision','axis','N','physical','config','R','compiler','law'):
            cat,case,alloc,adopted,terms,values=fixture()
            if mode=='snapshot':cat.comparison['binding']={'snapshot':'different'}
            if mode=='precision':adopted['MSE_upper']=refnum(1,20000)
            if mode=='axis':values['price0X']['axis']='Y'
            if mode=='N':terms[0]['N']=3
            if mode=='physical':values['price0X']['physical_id']='other'
            if mode=='config':case['original_configuration_id']='other'
            if mode=='R':values['price0X']['R_outer']=3
            if mode=='compiler':cat.comparison['compiler_profile']={'compiler':'other'}
            if mode=='law':case['law']={'model':'other'}
            cat.make(case,alloc,PASS,20,None,None,adopted,{'path':'mock'},terms)
            self.assertEqual(cat.rows[0]['status'],'MISMATCH',mode)
            self.assertIsNone(cat.result()['method_minima']['PR_DF']['conditional_cost'])

    def test_exact_cost_order_ties_and_duplicate_conflicts(self):
        cat,case,alloc,adopted,terms,_=fixture()
        cat.make(case,alloc,PASS,refnum(10000000000000001,10**16),None,None,adopted,{'path':'one'},terms)
        cat.make(case,alloc,PASS,refnum(10000000000000001,10**16),None,None,adopted,{'path':'alias'},terms)
        self.assertEqual(cat.result()['duplicates_removed'],1)
        cat.make(case,alloc,PASS,1,None,None,adopted,{'path':'conflict'},terms)
        self.assertIsNone(cat.result()['method_minima']['PR_DF']['conditional_cost'])


class PublishedAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=json.loads(INPUT.read_text());cls.reader=Reader(ROOT)

    def test_four_m12_points_and_all_required_routes(self):
        cat=Catalog(self.reader,self.inputs['comparison'])
        for source in self.inputs['sources']:cat.add(source)
        result=cat.result()
        self.assertFalse([r for r in result['candidates']if r['status']=='MISMATCH'])
        best=result['method_minima']['TRUNCATED_PF'];self.assertEqual(frac(best['conditional_cost']),395962678)
        selected=next(r for r in result['candidates']if r['point_id']in best['point_ids'])
        self.assertEqual(selected['N_X'][-1],2)
        group=[r for r in result['candidates']if r['configuration_id']==selected['configuration_id']and r['N_X'][-2]==13]
        points={r['N_X'][-1]:(r['status'],frac(r['conditional_cost']))for r in group}
        self.assertEqual(points,{1:('INFEASIBLE',376203472),2:('FEASIBLE',395962678),3:('FEASIBLE',415721884),4:('FEASIBLE',435481090)})
        self.assertEqual(len(next(r for r in group if r['N_X'][-1]==4)['result_sources']),2)
        for method,ld in [('PR_DF',7)]:
            waiting=next(r for r in result['candidates']if r['method']==method and r['L_D']==ld)
            self.assertEqual(waiting['status'],'PRICE_WAITING')
            self.assertEqual(waiting['q3_status'],PASS)
            self.assertIsNone(waiting['conditional_cost'])
            self.assertNotIn(waiting['point_id'],result['method_minima'][method]['point_ids'])
        self.assertEqual(len(result['signal_only_history']),1)
        pauli=[r for r in result['candidates']if r['method']=='PR_PAULI' and r['L_D']==3]
        self.assertEqual(len(pauli),1)
        ld3=pauli[0]
        self.assertEqual(ld3['status'],'FEASIBLE')
        self.assertEqual(frac(ld3['conditional_cost']),frac('173557305044/165'))
        self.assertEqual(result['method_minima']['PR_PAULI']['point_ids'],[ld3['point_id']])
        self.assertFalse(ld3['qualified_for_candidate'])
        self.assertEqual(ld3['price_availability_history'][0]['status'],'PRICE_WAITING')
        self.assertEqual(ld3['price_evidence']['direct_calibration_paths'],2)
        self.assertEqual(ld3['price_evidence']['round_direct_prices'],0)
        self.assertEqual(len(ld3['price_axes']),26)
        for p in ld3['price_axes']:
            self.assertEqual(p['price_kind'],'MODEL')
            self.assertFalse(p['qualified_for_candidate'])
            self.assertIsNotNone(p['q3_axis_source'])
        own_b=next(r for r in result['candidates']if r['method']=='PR_PAULI' and r['L_D']==2 and r['label']=='OWN_B_LD2')
        self.assertEqual(own_b['status'],'FEASIBLE')
        self.assertEqual(frac(own_b['conditional_cost']),frac('220695144335642870/90343469'))
        coarse=next(r for r in result['candidates']if r['label']=='B0I_P7_Y4_32h_over_5')
        self.assertEqual(coarse['status'],'INFEASIBLE')
        self.assertEqual(coarse['q3_status'],FAIL)
        self.assertIsNone(coarse['conditional_cost'])
        self.assertEqual(coarse['execution_history'][0]['status'],'UNKNOWN_STOP_NO_RETRY')
        self.assertNotIn(coarse['point_id'],best['point_ids'])
        self.assertTrue(all(r['MSE_upper'] is not None for r in result['candidates']if r['method']=='FULL_H'))
        self.assertEqual(frac(result['method_minima']['FULL_PF']['conditional_cost']),559457386)
        full=[r for r in result['candidates']if r['point_id']in result['method_minima']['FULL_PF']['point_ids']][0]
        history=[r for r in result['candidates']if r['configuration_id']==full['configuration_id']and r['N_X'][-1]==4]
        self.assertEqual(frac(history[0]['conditional_cost']),615304502)
        one=next(r for r in result['candidates']if r['configuration_id']==full['configuration_id']and r['N_X'][-1]==1)
        self.assertEqual(one['status'],'INFEASIBLE')
        self.assertEqual(frac(one['conditional_cost']),531533828)
        self.assertEqual(one['q3_failure_kind'],'MODEL_THRESHOLD_EXCEEDED_NOT_NUMERICAL_FAILURE')
        self.assertEqual(len([r for r in result['candidates']if r['configuration_id']==full['configuration_id']and r['N_X'][-1]==2]),1)
        pr=next(r for r in result['candidates']if r['label']=='FPN_B_LD2')
        self.assertEqual(frac(pr['historical_budget']),615304502)
        self.assertEqual(pr['historical_PF_benchmarks']['FULL_PF_S2_B_PRICE'],615304502)
        # Additional aliases resolve to the same saved SUMMARY, never new points.
        cat.add(ref(self.inputs['sources'][4]['commit'],'docs/research/d_pr_df_ld_split_q3_resume_20261116/C_HANDOFF.json'))
        self.assertEqual(len(cat.result()['candidates']),len(result['candidates']))

    def test_pauli_price_input_order_and_binding_rejections(self):
        source=next(s for s in self.inputs['sources']if s['path'].endswith('HANDOFF_BC.json'))
        cat=Catalog(self.reader,self.inputs['comparison'])
        cat.add(source)  # The price locator loads its fixed Q3 dependency first.
        self.assertEqual(len(cat.result()['candidates']),1)
        self.assertEqual(cat.result()['candidates'][0]['status'],'FEASIBLE')
        for key,value in [('binding_id','other'),('allocation_id','other'),('total_candidate_id','other'),
                          ('original_candidate_id','other'),('N',49),('R',2)]:
            changed=deepcopy(self.reader.read(source));changed['rows'][0][key]=value
            with self.assertRaisesRegex(ValueError,'Pauli'):
                Catalog(self.reader,self.inputs['comparison']).pauli_model_prices(source,changed)

    def test_same_command_updates_only_by_adding_a_pinned_source(self):
        import os,sys
        with tempfile.TemporaryDirectory()as tmp:
            manifest=Path(tmp)/'INPUTS.json';out=Path(tmp)/'comparison'
            data=deepcopy(self.inputs);data['sources']=data['sources'][:1];manifest.write_text(json.dumps(data))
            cmd=[sys.executable,'-B',str(ROOT/'scripts/collect_saved_rpe_costs.py'),'--inputs',str(manifest),'--output',str(out)]
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
            subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True)
            first=json.loads((out/'CATALOG.json').read_text())
            self.assertEqual(frac(first['method_minima']['TRUNCATED_PF']['conditional_cost']),435481090)
            data['sources'].append(self.inputs['sources'][1]);manifest.write_text(json.dumps(data))
            subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True)
            second=json.loads((out/'CATALOG.json').read_text())
            self.assertEqual(frac(second['method_minima']['TRUNCATED_PF']['conditional_cost']),395962678)


if __name__=='__main__':unittest.main()
