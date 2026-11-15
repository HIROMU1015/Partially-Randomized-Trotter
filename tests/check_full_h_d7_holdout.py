"""Artificial records only: no H4, sampling, circuit construction, compiler or fit."""
from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_full_h_d7_holdout as H
import run_full_h_kappa2_p2 as P
from check_full_h_kappa2_p2_prep import fictional_paths


class Tests(unittest.TestCase):
    def setUp(self):
        self.plan=P.proposal();self.plan['adopted']=True;D=P.design_api()
        cal=[c for c in self.plan['cells'] if c['role']=='CALIBRATION'];features=list(D.FEATURE_MAPS[self.plan['feature_map']][0])
        models={a:dict(model='PRICE_LINEAR_FEATURE_MODEL_V1',status='CALIBRATED_UNVALIDATED',qualified_for_candidate=False,
            domain=dict(D._domain(cal[0]),axis=a),features=features,coefficients={f:H.exact(0) for f in features},
            design_source=self.plan['design_source'],calibration=[dict(cell_id=c['cell_id'],source=dict(cohort=c['cohort'],
                step_seeds=P.seed_table(self.plan,c['cell_id']))) for c in cal]) for a in ('X','Y')}
        predictions=[]
        for cid in ('M5_ROUND','M6_ROUND'):
            cell=P.selected(self.plan,'HOLDOUT',cid)
            for axis in ('X','Y'):
                predictions.append(dict(cell_id=cid,axis=axis,m=int(cid[1]),q=cell['q'],R=cell['R'],n_paths=cell['n_paths'],
                    physical=P.physical_row(cell),cohort=cell['cohort'],step_seeds=P.seed_table(self.plan,cid),
                    features={k:H.exact(v) for k,v in D.features(cell,self.plan['feature_map']).items()},predicted_RZ=H.exact(H.Fraction(7,3)),
                    observed_RZ=None,tolerance=None))
        self.frozen=dict(models=models,predictions=predictions)

    def bound(self,frozen=None):
        original=P.P1.blob;value=frozen or self.frozen
        with patch.object(P.P1,'blob',side_effect=lambda c,p:json.dumps(value).encode() if p==H.MODEL['path'] else original(c,p)):
            return H.prediction_bindings(self.plan)

    def observations(self):
        return [dict(cell_id=r['cell_id'],axis=r['axis'],index=i,cohort=r['cohort'],physical=r['physical'],
            seeds=s,rz_count=2+i,gate_status='PASS',locators=dict(fixture='ARTIFICIAL',index=i,axis=r['axis']))
            for r in self.bound()['rows'] for i,s in enumerate(r['step_seeds'])]

    def test_preopening_uses_unique_saved_prediction_and_checks_seed(self):
        b=self.bound();self.assertEqual(len(b['rows']),4);self.assertEqual(b['holdout_observation_reads'],0)
        self.assertTrue(all(r['P']==dict(numerator=7,denominator=3) for r in b['rows']))
        for change in ('duplicate','seed','physical'):
            bad=deepcopy(self.frozen)
            if change=='duplicate':bad['predictions'][1]=deepcopy(bad['predictions'][0])
            elif change=='seed':bad['predictions'][0]['step_seeds'][0][0]=0
            else:bad['predictions'][0]['physical']['id']='ARTIFICIAL_FOREIGN'
            with self.assertRaises(ValueError):self.bound(bad)

    def test_exact_mean_signed_errors_and_no_model_verdict(self):
        result=H.compare(self.bound(),self.observations());self.assertEqual(result['status'],'DESCRIPTIVE_COMPLETE')
        a=result['rows'][0];self.assertEqual(a['O'],H.exact(H.Fraction(5,2)))
        self.assertEqual(a['signed_error'],H.exact(H.Fraction(-1,6)))
        self.assertEqual(a['signed_relative_error'],H.exact(H.Fraction(-1,15)))
        self.assertEqual(a['display_signed_percent'],'-6.666667')
        self.assertTrue(all(r['model_verdict']==H.VERDICT and r['tolerance'] is None for r in result['rows']))

    def test_missing_m5_path_is_not_replaced_by_one_path_or_other_axis(self):
        records=self.observations();records=[r for r in records if not(r['cell_id']=='M5_ROUND' and r['axis']=='X' and r['index']==1)]
        value=H.compare(self.bound(),records);self.assertEqual(value['status'],'INCOMPLETE_OR_INVALID')
        self.assertIsNone(value['rows'][0]['O']);self.assertEqual(value['valid_comparisons'],3)

    def test_zero_observation_and_exact_match_do_not_imply_model_pass(self):
        rows=self.observations()
        for r in rows:r['rz_count']=0
        value=H.compare(self.bound(),rows);self.assertEqual(value['status'],'DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED')
        self.assertTrue(all(r['signed_relative_error'] is None and r['relative_error_reason']=='ZERO_OBSERVED_PRICE' for r in value['rows']))
        b=self.bound()
        for r in b['rows']:r['P']=H.exact(0)
        value=H.compare(b,rows);self.assertTrue(all(r['signed_error']==H.exact(0) and r['model_verdict']==H.VERDICT for r in value['rows']))

    def test_gate_failure_and_foreign_axis_binding_are_not_precision_fail(self):
        rows=self.observations();rows[0]['seeds']=[0]
        value=H.compare(self.bound(),rows,{'M6_ROUND':'ARTIFICIAL_ACQUISITION_FAILURE'})
        self.assertEqual(value['valid_comparisons'],1)
        self.assertTrue(all(r['model_verdict']==H.VERDICT for r in value['rows']))

    def test_actual_axis_receipt_validation_reuses_fictional_fixture(self):
        cell=P.selected(self.plan,'HOLDOUT','M5_ROUND');paths=fictional_paths(self.plan,cell)
        for p in paths:
            p['sampling']='ORIGINAL_NORMAL_SAMPLER';p['events']=[['ARTIFICIAL']*cell['R'] for _ in range(cell['q'])]
            p['trial'].update(candidate=p['row'],cohort=p['cohort'],index=p['index'])
            p['trial']['checks']=[dict(check=g,error=0,tolerance=1,phase_alignment=False) for g in H.NUMERIC_GATES] if p['index']==0 else []
            p['checks'].update(scalar=.3,scalar_expected=.3,one_body_applications=1,legacy_rzz_control_instructions=0)
        source=dict(commit='a'*40,caller=H.CALLER,cell=cell['cell_id'],cohort=cell['cohort'],step_seeds=P.seed_table(self.plan,cell['cell_id']))
        result=dict(status='FINITE_CELL_COMPLETE',child=dict(status='PASS',paths=paths),source=source)
        self.assertEqual(len(H.axis_receipts(self.plan,cell,result,'a'*40)),4)
        with self.assertRaises(ValueError):H.axis_receipts(self.plan,cell,result,'b'*40)
        paths[1]['xy_phase_gate']['status']='FAIL'
        with self.assertRaises(ValueError):H.axis_receipts(self.plan,cell,result,'a'*40)

    def test_entry_uses_existing_path_runner_and_saves_progress(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);job=p/'job.json';job.write_text(json.dumps(dict(private=temp)))
            with patch.object(H.Runner,'child',return_value=0) as caller:self.assertEqual(H.child(job),0)
            caller.assert_called_once_with(job)
            self.assertEqual(json.loads((p/'PROGRESS.jsonl').read_text().splitlines()[-1])['stage'],'COMPLETE')

    def test_main_requires_exact_frozen_job_before_existing_executor(self):
        from types import SimpleNamespace
        cell=P.selected(self.plan,'HOLDOUT','M5_ROUND');bound=self.bound()
        job=dict(plan=self.plan,cell_id=cell['cell_id'],frozen_model_source=H.MODEL,comparison_rule=H.RULE,
            new_compile_maximum=4,progress_stages=True,private='ARTIFICIAL_PRIVATE')
        frozen=dict(jobs={cell['cell_id']:job},prediction_bindings=bound)
        original_blob=P.P1.blob
        original_check_output=H.subprocess.check_output
        def check_output(command,*args,**kwargs):
            if command==['git','rev-parse','HEAD']:return 'a'*40
            return original_check_output(command,*args,**kwargs)
        def blob(commit,path):
            if path==H.FROZEN:return json.dumps(frozen).encode()
            if path.startswith('scripts/'):return (ROOT/path).read_bytes()
            return original_blob(commit,path)
        with TemporaryDirectory() as temp:
            path=Path(temp)/'job.json';path.write_text(json.dumps(job))
            args=SimpleNamespace(child=None,execute_approved_job=str(path))
            with patch.object(H.argparse.ArgumentParser,'parse_args',return_value=args), \
                 patch.object(H.subprocess,'check_output',side_effect=check_output),patch.object(P.P1,'blob',side_effect=blob), \
                 patch.object(H,'prediction_bindings',return_value=bound),patch.object(P,'execute',return_value=0) as execute:
                self.assertEqual(H.main(),0);execute.assert_called_once()
                altered=dict(job,new_compile_maximum=6);path.write_text(json.dumps(altered));execute.reset_mock()
                with self.assertRaisesRegex(ValueError,'frozen inputs'):H.main()
                execute.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
