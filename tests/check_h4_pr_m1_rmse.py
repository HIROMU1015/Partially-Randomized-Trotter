"""Synthetic scalar tests + field-only real preflight. No real reanalysis."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import h4_pr_m1_rmse as M
from check_pr_rpe_rmse_s0 import args,fixture as s0_fixture
A=M.A;p=M.p
MOCK=dict(commit='SYNTHETIC_ONLY',path='synthetic/run01/result.json',selector=['records',0])


def fixture():
    _,saved,reference,schedule=s0_fixture()
    req=next(r for r in schedule['rows'] if r['identity']['m']==1 and p.arm(r['identity']['candidate'])=='PR')
    i=req['identity'];saved['identity']=deepcopy(i);d=saved['diagnostics']
    d['candidate']=deepcopy(i['candidate'])
    for k in ('signed_weights','allocation','event_count'):d[k]=deepcopy(i['structure'][k])
    # Deliberately old format, no policy/stages tag. Large synthetic allowance.
    d['allowance']=dict(u=1e-7,log_B_margin=1e-8,certified=False,empirical_not_certified=True)
    reference.update(T=i['candidate']['T'],source=MOCK)
    return req,saved,reference,schedule


def packet_fixture():
    req,saved,ref,schedule=fixture();values,_=args();bridge=M.V.b_metadata(ROOT)
    config=bridge.generate(bridge.a_spec(values,'PR'));ev=A.evaluate(schedule)
    prior=[]
    for m,n in ((0,101),(6,23)):
        r=next(v for v in ev['records'] if v['identity']['m']==m and p.arm(v['identity']['candidate'])=='PR')
        r.update(nominal_N_X=n,nominal_N_Y=n,status='UNQUALIFIED_MODEL_ASSUMPTIONS');prior.append(deepcopy(r))
    old=next(x for x in A.b_handoff(schedule,ev,MOCK,evidence_domain='SYNTHETIC_ONLY')['packets'] if x['spec']['method']=='PR')
    packet=bridge.n_from_a(config,old,inputs=values)
    new=deepcopy(next(v for v in ev['records'] if v['request_id']==req['request_id']))
    new.update(nominal_N_X=37,nominal_N_Y=37,status='UNQUALIFIED_MODEL_ASSUMPTIONS')
    return config,schedule,packet,prior,new


class M1Tests(unittest.TestCase):
    def test_original_allowance_formula_and_S0_geometry_reused_on_toy_only(self):
        req,saved,ref,_=fixture();before=deepcopy(saved)
        out=M.diagnose(req,saved,ref,MOCK,root=ROOT)
        self.assertEqual(saved,before);self.assertTrue(out['numerical_gates_passed'])
        self.assertEqual(out['allowance']['policy'],'R_AWARE_EMPIRICAL_ACTION_COUNT_V1')
        self.assertGreaterEqual(out['allowance']['u'],saved['diagnostics']['allowance']['u'])
        self.assertGreaterEqual(out['allowance']['log_B_margin'],saved['diagnostics']['allowance']['log_B_margin'])
        expected=2*(2*(1+1)+1)+2*8
        self.assertEqual(out['allowance']['stages'],expected)
        source=out['conditional_geometry']['error_source']
        self.assertEqual(source['path'],MOCK['path']);self.assertEqual(source['selector'],MOCK['selector'])
        self.assertNotEqual(source['commit'],M.V.legacy.P1)
        self.assertFalse(out['rigorous_error_bound']);self.assertIsNone(out['actual_implementation']['law_error'])

    def test_missing_allowance_reference_or_numeric_evidence_stops(self):
        for key in ('allowance','reference','norm','closure','candidate'):
            req,saved,ref,_=fixture()
            if key=='allowance':saved['diagnostics']['allowance']['u']=None
            elif key=='reference':ref['evidence']['signal_difference']=1e-8
            elif key=='norm':saved['diagnostics']['signal']['allowance_evidence'].pop('exact_tail_norm_error')
            elif key=='closure':saved['diagnostics']['signal']['allowance_evidence']['raw_corrected_vector_closure']=1e-5
            else:saved['identity']['binding']['state']='SYNTHETIC_FOREIGN'
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):M.diagnose(req,saved,ref,MOCK,root=ROOT)

    def test_no_Hoeffding_wrapper_or_action_called_for_toy_diagnosis(self):
        from trottertracks import pr_r_dependent_n as old
        req,saved,ref,_=fixture()
        with patch.object(old,'analyze',side_effect=AssertionError('old N wrapper')),patch.object(A,'existing_finite_mean',side_effect=AssertionError('action')):
            M.diagnose(req,saved,ref,MOCK,root=ROOT)

    def test_synthetic_model_and_extension_preserve_old_rounds(self):
        req,saved,ref,schedule=fixture();d=M.diagnose(req,saved,ref,MOCK,root=ROOT)
        r=dict(candidate=req['identity']['candidate'],signal=d['signal'],**{k:req['identity']['structure'][k] for k in ('signed_weights','allocation','event_count')})
        e=A.mean_evidence(req,r,MOCK);ev=A.evaluate(schedule,mean_provider=lambda requests:[e])
        new=next(v for v in ev['records'] if v['request_id']==req['request_id'])
        self.assertIsNotNone(new['nominal_N_X']);self.assertIsNone(new['sufficient_N_X'])
        config,schedule,old,prior,_=packet_fixture();snapshot=deepcopy(old)
        new_source=dict(fixture='SYNTHETIC_NEW_M1_SCIENCE_SOURCE')
        packet=M.handoff(config,schedule,old,prior,new,new_source,root=ROOT,evidence_domain='SYNTHETIC_ONLY')
        self.assertEqual(old,snapshot)
        self.assertEqual(packet['rounds'][0]['nominal_N_X'],101);self.assertEqual(packet['rounds'][6]['nominal_N_X'],23)
        self.assertIsNotNone(packet['rounds'][1]['nominal_N_X'])
        self.assertEqual(packet['rounds'][0]['diagnostic_source'],MOCK)
        self.assertEqual(packet['rounds'][1]['diagnostic_source'],new_source)
        self.assertFalse(packet['B_S0_only_validator_applicable'])
        for row in packet['rounds']:
            self.assertIsNone(row['N_X']);self.assertIsNone(row['N_Y'])
            if row['m'] not in (0,1,6):self.assertIsNone(row['nominal_N_X'])
        self.assertIsNone(packet['rmse_evidence']['rmse_bound_ha'])

    def test_foreign_scope_sufficient_N_or_S0_change_rejected(self):
        for key in ('m','SOURCE','sufficient','HP','S0'):
            config,schedule,old,prior,new=packet_fixture()
            if key=='m':new['identity']['m']=2
            elif key=='SOURCE':new['identity']['binding']['state']='FOREIGN'
            elif key=='sufficient':new['sufficient_N_X']=100
            elif key=='HP':new['N_X']=100
            else:prior[0]['nominal_N_X']=100
            with self.subTest(key=key),self.assertRaises(ValueError):M.handoff(config,schedule,old,prior,new,MOCK,root=ROOT)

    def test_real_preflight_is_field_only_and_B_official_V2_matches(self):
        with patch.object(M,'diagnose',side_effect=AssertionError('real science')),patch.object(A,'nominal_axis_shots',side_effect=AssertionError('N')),patch.object(A,'existing_finite_mean',side_effect=AssertionError('mean')):
            plan=M.prepare(ROOT)
        self.assertTrue(plan['B_S0_packet_accepted']);self.assertEqual(plan['m'],1)
        self.assertEqual(plan['physical_candidate']['id'],M.CANDIDATE)
        self.assertEqual(plan['configuration_id'],'587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911')
        self.assertEqual(plan['nominal_coverage'][0]['nominal_N_X'],64)
        self.assertEqual(plan['nominal_coverage'][6]['nominal_N_X'],38)
        for row in plan['nominal_coverage']:
            if row['m'] not in (0,6):self.assertIsNone(row['nominal_N_X'])
        self.assertEqual(plan['science_calls'],0);self.assertIsNone(plan['global_RMSE_bound'])

    def test_future_runner_toy_only_success_and_gate_failure_no_fallback(self):
        import json
        original_blob=M.blob
        mock_commit='0'*40
        def source_blob(root,commit,path):
            return (ROOT/path).read_bytes() if commit==mock_commit and path in (M.MODULE,M.RUNNER) else original_blob(root,commit,path)
        for failed in (False,True):
            c,s,old,prior,_=packet_fixture();req,saved,ref,_=fixture()
            if failed:ref['evidence']['signal_difference']=1e-6
            supplied=dict(configuration=c,schedule=s,N_packet=old,request=req,saved=saved,reference=ref,source_files={'fixture':'SYNTHETIC_ONLY'})
            with tempfile.TemporaryDirectory(prefix='m1-SYNTHETIC-runner-') as tmp,patch.object(M,'_load',return_value=supplied),patch.object(M,'blob',side_effect=source_blob),patch.object(p.SavedArtifacts,'read',return_value={'model_records':prior}),patch.object(A,'existing_finite_mean',side_effect=AssertionError('fresh action')):
                target=Path(tmp)/'run'
                if failed:
                    with self.assertRaises(ValueError):M.execute(ROOT,target,mock_commit)
                    result=json.loads((target/'result.json').read_text())
                    self.assertEqual(result['status'],'STOP_FAILED_NO_FALLBACK')
                    self.assertFalse((target/'N_HANDOFF.json').exists())
                else:
                    result=M.execute(ROOT,target,mock_commit)
                    self.assertEqual(result['status'],'M1_NOMINAL_COMPLETE_UNQUALIFIED')
                    packet=json.loads((target/'N_HANDOFF.json').read_text())
                    self.assertEqual(packet['rounds'][0]['nominal_N_X'],101)
                    self.assertIsNotNone(packet['rounds'][1]['nominal_N_X'])
                self.assertFalse(result['automatic_retry']);self.assertEqual(result['new_finite_means'],0)

    def test_default_CLI_preflight_has_no_science(self):
        with tempfile.TemporaryDirectory(prefix='m1-metadata-') as tmp:
            target=Path(tmp)/'preflight.json'
            subprocess.run([sys.executable,'-B','scripts/run_h4_pr_m1_rmse.py','--out',str(target)],cwd=ROOT,check=True,stdout=subprocess.PIPE)
            import json
            self.assertEqual(json.loads(target.read_text())['science_calls'],0)
            r=subprocess.run([sys.executable,'-B','scripts/run_h4_pr_m1_rmse.py','--execute','--out',str(Path(tmp)/'no-source')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(r.returncode,0);self.assertFalse((Path(tmp)/'no-source').exists())

    def test_D_Full_PF_no_PR_kappa_K_or_truncated_alias(self):
        D=M.full_pf_metadata(ROOT)
        base=dict(binding=dict(system='H4',rank=10,hamiltonian='SYNTHETIC_H'),h='1/10',rmse_policy={'fixture':'NOT_ADOPTED'},profile={'fixture':'MOCK'})
        for formula in ('2nd','4th(m5_best)'):
            s=D.full_pf_spec(base,formula=formula,delta_pf='1/10',M=2,rank=10)
            self.assertEqual(s['split']['deterministic_scope'],'FULL_H');self.assertEqual(s['L_D'],10)
            self.assertIsNone(s['kappa']);self.assertIsNone(s['finite_K']);self.assertIsNone(s['split']['lambda_R'])
            self.assertEqual(s['law']['model'],'deterministic');self.assertEqual(s['event_mapping']['kind'],'NONE')
        with self.assertRaises(ValueError):D.full_pf_spec(base,formula='2nd',delta_pf='1/10',M=2,rank=7)
        self.assertEqual(D.catalog['4th']['v2_status'],'B_V2_FORMULA_EXTENSION_REQUIRED')
        with self.assertRaises(ValueError):D.full_pf_spec(base,formula='4th',delta_pf='1/10',M=2,rank=10)
        self.assertEqual(len(D.catalog['4th']['signed_weights']),3)
        self.assertEqual(len(D.catalog['4th(m5_best)']['signed_weights']),11)

    def test_Full_PF_null_archived_signal_or_allowance_stays_missing(self):
        with patch.object(M,'diagnose',side_effect=AssertionError('science')),patch.object(A,'nominal_axis_shots',side_effect=AssertionError('N')):
            catalog=M.full_pf_inventory(ROOT)
        rows=catalog['saved_carriers']
        self.assertTrue(any(not r['error_fields'] for r in rows))
        # Explicit synthetic missing-signal boundary, not an assertion about
        # the actual archive (whose 136 signal records are present).
        import gzip,json
        c=deepcopy(rows[0]['candidate']);fake={c['id']:dict(candidate=c,signal=None,fixture='SYNTHETIC_ONLY')}
        original=M.blob
        def synthetic(root,commit,path):
            return gzip.compress(json.dumps(fake).encode()) if path.endswith('.json.gz') else original(root,commit,path)
        with patch.object(M,'blob',side_effect=synthetic):missing=M.full_pf_inventory(ROOT)['saved_carriers']
        self.assertTrue(all(not r['signal_record_present'] for r in missing))
        rows+=missing
        for r in rows:
            if not r['signal_record_present']:self.assertFalse(r['signal_fields']);self.assertFalse(r['error_fields'])
            self.assertIsNone(r['nominal_N_X']);self.assertFalse(r['numerical_gate_rechecked'])
        self.assertFalse(catalog['policy_adopted']);self.assertFalse(catalog['configuration_adopted'])

    def test_protected_S0_modules_and_no_scientific_imports(self):
        import run_pr_rpe_rmse_s0 as s0
        for path in s0.CONSUMED+('scripts/run_pr_rpe_rmse_s0.py',):
            self.assertEqual((ROOT/path).read_bytes(),M.blob(ROOT,M.V.S0_SCIENCE,path))
        self.assertFalse(any(v in sys.modules for v in ('numpy','scipy','qiskit','pyscf')))


if __name__=='__main__':unittest.main()
