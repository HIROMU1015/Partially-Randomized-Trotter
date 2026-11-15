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
        all_ld7=[r for r in result['candidates']if r['method']=='PR_DF' and r['L_D']==7]
        self.assertEqual(len(all_ld7),7)
        ld7=[r for r in all_ld7 if r['label']in ('LD7_V0','LD7_A_SHAPE','LD7_B_PRICE')]
        self.assertEqual(len(ld7),3)
        self.assertTrue(all(r['status']=='FEASIBLE' and r['q3_status']==PASS for r in ld7))
        selected_df=next(r for r in ld7 if r['label']=='LD7_A_SHAPE')
        self.assertEqual(result['method_minima']['PR_DF']['point_ids'],[selected_df['point_id']])
        self.assertEqual(frac(selected_df['conditional_cost']),frac('155652932221/126'))
        self.assertEqual(selected_df['N_X'],[48,45,42,39,35,32,29,25,22,19,16,12,9])
        self.assertEqual(selected_df['N_X'],selected_df['N_Y'])
        self.assertEqual(selected_df['saved_cost_breakdown']['m7_m12_percent_display'],'95.9453590253993779917152')
        self.assertEqual(frac(selected_df['saved_cost_breakdown']['m7_m12_fraction']),frac('149341764653/155652932221'))
        for r in ld7:
            self.assertFalse(r['qualified_for_candidate'])
            self.assertEqual(len(r['price_axes']),26)
            self.assertTrue(all(p['price_kind']=='MODEL' and not p['qualified_for_candidate']for p in r['price_axes']))
        self.assertEqual(next(r for r in ld7 if r['label']=='LD7_V0')['price_availability_history'][0]['status'],'PRICE_WAITING')
        self.assertEqual(len([r for r in result['candidates']if r['method']=='PR_DF' and r['L_D']in (1,2)]),6)
        ld9=[r for r in result['candidates']if r['method']=='PR_DF' and r['L_D']==9]
        self.assertEqual(len(ld9),1)
        self.assertEqual(ld9[0]['status'],'PRICE_WAITING')
        self.assertEqual(ld9[0]['q3_status'],PASS)
        self.assertIsNone(ld9[0]['conditional_cost'])
        self.assertEqual(ld9[0]['price_axes'],[])
        self.assertEqual(len(ld9[0]['partial_price_axes']),2)
        for p in ld9[0]['partial_price_axes']:
            self.assertEqual(p['m'],6)
            self.assertEqual(frac(p['C_RZ']),492573)
            self.assertEqual(frac(p['reference_C_RZ']),frac('1575349/4'))
            self.assertEqual(frac(p['difference_LD9_minus_LD7']),frac('394943/4'))
            self.assertEqual(p['n_paths'],4)
            self.assertEqual(len(p['path_sources']),4)
            self.assertEqual(p['price_kind'],'DIRECT_FINITE_COHORT_MEAN')
            self.assertFalse(p['qualified_for_candidate'])
        budgets=[r for r in all_ld7 if r['label']not in ('LD7_V0','LD7_A_SHAPE','LD7_B_PRICE')]
        self.assertEqual(len(budgets),4)
        self.assertTrue(all(r['status']=='INFEASIBLE' and r['q3_status']==FAIL for r in budgets))
        self.assertNotIn(ld9[0]['point_id'],result['method_minima']['PR_DF']['point_ids'])
        self.assertEqual(len(result['signal_only_history']),1)
        pauli=[r for r in result['candidates']if r['method']=='PR_PAULI' and r['L_D']==3]
        self.assertEqual(len(pauli),5)
        ld3=next(r for r in pauli if r['label']=='PAULI_LD3_VP')
        self.assertEqual(ld3['status'],'FEASIBLE')
        self.assertEqual(frac(ld3['conditional_cost']),frac('173557305044/165'))
        budget=next(r for r in pauli if r['label']=='FULL_PF__B_PRICE')
        ld5=[r for r in result['candidates']if r['method']=='PR_PAULI' and r['L_D']==5]
        self.assertEqual(len(ld5),2)
        best5=next(r for r in ld5 if r['label']=='LD5__FULL_PF__B_PRICE')
        vp5=next(r for r in ld5 if r['label']=='LD5__VP')
        self.assertEqual(result['method_minima']['PR_PAULI']['point_ids'],[best5['point_id']])
        self.assertEqual(frac(best5['conditional_cost']),frac('3792207339/7'))
        self.assertEqual(frac(vp5['conditional_cost']),frac('6105690627/7'))
        self.assertEqual(frac(best5['saved_split_comparison']['difference_LD5_minus_LD3']),frac('-222210893/1155'))
        self.assertNotEqual(best5['configuration_id'],budget['configuration_id'])
        for r in ld5:
            self.assertEqual(r['status'],'FEASIBLE')
            self.assertIsNotNone(r['MSE_lower'])
            self.assertIsNotNone(r['MSE_upper'])
            self.assertFalse(r['qualified_for_candidate'])
            self.assertEqual(len(r['price_axes']),26)
            self.assertTrue(all(p['price_kind']=='MODEL' and not p['qualified_for_candidate']for p in r['price_axes']))
        self.assertEqual(budget['status'],'FEASIBLE')
        self.assertEqual(frac(budget['conditional_cost']),frac('625936421828/1155'))
        self.assertEqual(budget['N_X'],[48,45,42,39,36,33,29,26,23,19,16,8,2])
        self.assertEqual(budget['N_X'],budget['N_Y'])
        self.assertEqual(frac(budget['saved_PF_comparisons']['difference_T_minus_Full_PF']),frac('-20236859002/1155'))
        self.assertEqual(frac(budget['saved_PF_comparisons']['difference_T_minus_P7']),frac('168599528738/1155'))
        self.assertEqual({r['label']:r['status']for r in pauli if r['label']!='PAULI_LD3_VP'},
            {'P7__A_SHAPE':'INFEASIBLE','P7__B_PRICE':'INFEASIBLE','FULL_PF__A_SHAPE':'INFEASIBLE','FULL_PF__B_PRICE':'FEASIBLE'})
        for r in pauli:
            self.assertFalse(r['qualified_for_candidate'])
            self.assertTrue(all(p['price_kind']=='MODEL' and not p['qualified_for_candidate']for p in r['price_axes']))
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
        coarse_a=next(r for r in result['candidates']if r['label']=='B0I_P7_Y4_32h_over_5__A_SHAPE')
        self.assertEqual(coarse_a['status'],'PRICE_WAITING')
        self.assertEqual(coarse_a['q3_status'],PASS)
        self.assertIsNone(coarse_a['conditional_cost'])
        self.assertEqual(coarse_a['price_axes'],[])
        self.assertEqual(coarse_a['configuration_id'],coarse['configuration_id'])
        self.assertNotEqual(coarse_a['allocation']['allocation_id'],coarse['allocation']['allocation_id'])
        self.assertNotEqual(coarse_a['N_X'],coarse['N_X'])
        self.assertNotIn(coarse_a['point_id'],best['point_ids'])
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

    def test_ld7_source_first_and_republished_six_points_are_reused(self):
        source=next(s for s in self.inputs['sources']if 'd_pr_df_ld7_price_results'in s['path'])
        cat=Catalog(self.reader,self.inputs['comparison']);cat.add(source)
        self.assertEqual(len(cat.result()['candidates']),9)
        for s in self.inputs['sources']:cat.add(s)
        result=cat.result()
        self.assertEqual(len(result['candidates']),49)
        self.assertFalse([r for r in result['candidates']if r['status']=='MISMATCH'])
        self.assertEqual([r['label']for r in result['candidates']if r['status']=='PRICE_WAITING'],
                         ['B0I_P7_Y4_32h_over_5__A_SHAPE','LD9_V0'])
        summary=ref(source['commit'],source['path'].replace('C_HANDOFF.json','SUMMARY.json'))
        modified=deepcopy(self.reader.read(summary));modified['menu'][0]['cost']=refnum(0,1)
        with self.assertRaisesRegex(ValueError,'republication'):
            Catalog(self.reader,self.inputs['comparison']).pr_df_ld7(summary,modified)

    def test_pauli_budget_source_first_and_identity_rejections(self):
        source=next(s for s in self.inputs['sources']if s['path'].endswith('C_COMPARISON_JOIN.json'))
        cat=Catalog(self.reader,self.inputs['comparison']);cat.add(source)
        self.assertEqual(len(cat.result()['candidates']),5)  # Existing VP plus four distinct allocations.
        for s in self.inputs['sources']:cat.add(s)
        self.assertEqual(len(cat.result()['candidates']),49)
        for key,value in [('total_configuration_id','other'),('price_qualified_for_candidate',True)]:
            changed=deepcopy(self.reader.read(source));changed['records'][0][key]=value
            with self.assertRaisesRegex(ValueError,'Pauli budget'):
                Catalog(self.reader,self.inputs['comparison']).pauli_budget(source,changed)

    def test_ld9_unpriced_input_does_not_import_ld7_or_prices(self):
        source=next(s for s in self.inputs['sources']if 'd_pr_df_ld9_v0_results'in s['path'])
        cat=Catalog(self.reader,self.inputs['comparison']);cat.add(source)
        self.assertEqual(len(cat.result()['candidates']),1)
        self.assertEqual(cat.result()['candidates'][0]['L_D'],9)
        for s in self.inputs['sources']:cat.add(s)
        ld7=[r for r in cat.result()['candidates']if r['method']=='PR_DF' and r['L_D']==7]
        self.assertEqual(len(ld7),7)
        self.assertEqual(len([r for r in ld7 if r['label']=='LD7_V0']),1)
        changed=deepcopy(self.reader.read(source));changed['total_model_cost']=refnum(1,1)
        with self.assertRaisesRegex(ValueError,'different split price'):
            Catalog(self.reader,self.inputs['comparison']).pr_df_unpriced_v0(source,changed)

    def test_partial_price_reuses_allocation_and_budget_verdicts_are_preserved(self):
        direct=next(s for s in self.inputs['sources']if 'd_pr_df_ld9_m6_price'in s['path'])
        cat=Catalog(self.reader,self.inputs['comparison']);cat.add(direct)
        rows=cat.result()['candidates']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'PRICE_WAITING')
        self.assertIsNone(rows[0]['conditional_cost'])
        changed=deepcopy(self.reader.read(direct));changed['physical_id']='other'
        with self.assertRaisesRegex(ValueError,'physical candidate'):
            Catalog(self.reader,self.inputs['comparison']).ld9_partial_price(direct,changed)
        budget=next(s for s in self.inputs['sources']if 'pr_df_ld7_pf_budget'in s['path'])
        changed=deepcopy(self.reader.read(budget));changed['rows'][0]['model_status']=PASS
        with self.assertRaisesRegex(ValueError,'Q3/allocation'):
            Catalog(self.reader,self.inputs['comparison']).ld7_budget(budget,changed)

    def test_ld5_input_first_does_not_import_ld3_comparison_rows(self):
        source=next(s for s in self.inputs['sources']if 'pauli_ld5_two_vectors'in s['path'])
        cat=Catalog(self.reader,self.inputs['comparison']);cat.add(source)
        self.assertEqual(len(cat.result()['candidates']),2)
        self.assertTrue(all(r['L_D']==5 for r in cat.result()['candidates']))
        changed=deepcopy(self.reader.read(source));changed['records'][0]['dictionary_identity']['dictionary_sha256']='other'
        with self.assertRaisesRegex(ValueError,'dictionary'):
            Catalog(self.reader,self.inputs['comparison']).pauli_ld5(source,changed)
        changed=deepcopy(self.reader.read(source));changed['records'][0]['axes'][0]['total_candidate_id']='other'
        with self.assertRaisesRegex(ValueError,'total-view axis'):
            Catalog(self.reader,self.inputs['comparison']).pauli_ld5(source,changed)

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
