"""Artificial Git publications through C's unchanged public Catalog adapter."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sp=importlib.util.spec_from_file_location('_comparison_export',ROOT/'scripts/export_pr_candidate_comparison.py')
E=importlib.util.module_from_spec(sp);sp.loader.exec_module(E)
C=types.ModuleType('trottertracks._candidate_export_c');C.__package__='trottertracks'
exec(compile(subprocess.check_output(['git','show','06e0b67634265d6dc818d4032f0cdb8fe8fd5629:src/trottertracks/saved_rpe_cost_catalog.py'],cwd=ROOT),
             'fixed_C:06e0b6763','exec'),C.__dict__)


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def commit(root):
    git(root,'add','.');git(root,'-c','user.name=Artificial fixture','-c','user.email=fixture@example.invalid','commit','-qm','ARTIFICIAL publication')
    return git(root,'rev-parse','HEAD')
def put(root,path,value):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n')
def loc(commit,path):return dict(commit=commit,path=path)


def fixture(root,pauli,status=E.PASS):
    lower,upper=(.25,.25)if status==E.PASS else(2,3)if status==E.FAIL else(.25,2)
    split={'tail_dictionary':{'fixture':'ARTIFICIAL'}}if pauli else{'lambda_source':{'fixture':'ARTIFICIAL'}}
    cfg=dict(id='ARTIFICIAL_CONFIGURATION',spec=dict(method='PR',L_D=4,formula='2nd',finite_K=2,kappa={'value':2},
        split=split,h=.1,M=12,profile={'P':0},binding={'system':'ARTIFICIAL'},law={'model':'ARTIFICIAL_IID'}),
        rounds=[dict(m=m,q_PF=1<<m,R_outer=1,actual_delta_pf=.1,T=.1*(1<<m),
                     candidate={'id':'C'+str(m)},physical_candidate={'id':'P'+str(m)},law_binding={'fixture':'ARTIFICIAL'})for m in range(13)])
    allocation=dict(allocation_id='ARTIFICIAL_ALLOCATION',shots=[dict(m=m,N_X=2,N_Y=3)for m in range(13)])
    case=dict(label='ARTIFICIAL_CASE',original_configuration_id=cfg['id'],configuration=loc('SOURCE','configuration.json'),
        total_configuration_id='ARTIFICIAL_TOTAL',binding_id='ARTIFICIAL_BINDING',
        signal_packet=loc('SOURCE','signals.json'),evaluation_target={'energy_ha':0},allocation=allocation)
    adopted=dict(MSE_upper=1)
    record=dict(configuration_id=case['total_configuration_id'],canonical_binding=case['binding_id'],allocation=allocation,
        evaluation_target=case['evaluation_target'],diagnostic={'deterministic_risk':dict(MSE_lower=lower,MSE_upper=upper)})
    result=dict(status=status,allocation=allocation,record=record,adopted_q3_assessment={'status':status})
    quote=dict(configuration_id=cfg['id'],complete=True,policy='MODEL',rows=[dict(m=m,axis=a,physical_id='P'+str(m),q=1<<m,R=1,
        unit_price_RZ={'numerator':m+1,'denominator':2},classification='MODEL_REFERENCE',source={'fixture':'ARTIFICIAL'},qualified_E_C=False)
        for m in range(13)for a in ('X','Y')])
    put(root,'configuration.json',cfg);put(root,'signals.json',{'fixture':'ARTIFICIAL'})
    src=commit(root);case['configuration']['commit']=src;case['signal_packet']['commit']=src
    outcome=dict(status=status,MSE_lower=lower,MSE_upper=upper,identity=dict(configuration_id=cfg['id'],
        total_configuration_id=case['total_configuration_id'],binding_id=case['binding_id'],contract_matches=True,
        allocation=allocation['shots'],signal_packet=case['signal_packet'],evaluation_target=case['evaluation_target']))
    put(root,'case_inputs.json',dict(cases=[case],adopted_evaluation=adopted));put(root,'result.json',result)
    put(root,'api.json',[record]);put(root,'outcome.json',outcome);put(root,'quote.json',quote);raw=commit(root)
    request=dict(configuration=loc(src,'configuration.json'),**{k:loc(raw,path)for k,path in (
        ('case_inputs','case_inputs.json'),('result','result.json'),('api','api.json'),('outcome','outcome.json'),('quote','quote.json'))})
    return request,dict(binding={'system':'ARTIFICIAL'},h=.1,M=12,accuracy=adopted)


class ExportTests(unittest.TestCase):
    def test_both_pr_methods_through_unchanged_c_catalog(self):
        for pauli,status in ((pauli,status)for pauli in (False,True)for status in (E.PASS,E.FAIL,E.INCONCLUSIVE)):
            with self.subTest(pauli=pauli,status=status),tempfile.TemporaryDirectory()as temp:
                root=Path(temp);git(root,'init','-q');request,comparison=fixture(root,pauli,status)
                axis=E.prices(root,request);put(root,'export/AXIS_PRICES.json',axis);price_commit=commit(root)
                inputs=E.consumer_inputs(root,request,loc(price_commit,'export/AXIS_PRICES.json'))
                put(root,'export/INPUTS.json',inputs);input_commit=commit(root)
                summary,index=E.join(root,loc(input_commit,'export/INPUTS.json'),root/'export')
                put(root,'export/SUMMARY.json',summary);put(root,'export/C_RESULT_JOIN.json',index);final=commit(root)
                catalog=C.Catalog(C.Reader(root),comparison);catalog.add(loc(final,'export/C_RESULT_JOIN.json'));r=catalog.result()['candidates'][0]
                self.assertEqual(r['method'],'PR_PAULI'if pauli else'PR_DF')
                self.assertEqual(r['status'],{E.PASS:'FEASIBLE',E.FAIL:'INFEASIBLE',E.INCONCLUSIVE:'PENDING'}[status]);self.assertEqual(len(r['price_axes']),26)
                self.assertTrue(all(x['price_kind']=='MODEL_REFERENCE'for x in r['price_axes']))
                self.assertEqual(E.frac(r['conditional_cost']),sum((m+1)*5/2 for m in range(13)))
                self.assertIsNone(r['E_C']);self.assertIsNone(r['G_qualified'])
                sp=importlib.util.spec_from_file_location('_c_sync_export_check',ROOT/'scripts/check_pr_candidate_c_sync.py')
                checker=importlib.util.module_from_spec(sp);sp.loader.exec_module(checker)
                put(root,'baseline.json',dict(comparison=comparison,sources=[]));baseline=commit(root)
                checked=checker.check(root,loc(final,'export/C_RESULT_JOIN.json'),loc(baseline,'baseline.json'))
                self.assertEqual(checked['status'],'PASS')
                self.assertEqual(checked['discovered'],2)
                self.assertTrue(checked['original_discover_and_reconcile'])


    def test_missing_prices_and_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);git(root,'init','-q');request,_=fixture(root,True)
            bad=E.read(root,request['quote']);bad['complete']=False;put(root,'bad_quote.json',bad);src=commit(root)
            with self.assertRaisesRegex(ValueError,'Incomplete price quote'):
                E.prices(root,dict(request,quote=loc(src,'bad_quote.json')))
            bad=E.read(root,request['outcome']);bad['identity']['binding_id']='WRONG';put(root,'bad_outcome.json',bad);src=commit(root)
            with self.assertRaisesRegex(ValueError,'Total binding mismatch'):
                E.prices(root,dict(request,outcome=loc(src,'bad_outcome.json')))


if __name__=='__main__':unittest.main(verbosity=2)
