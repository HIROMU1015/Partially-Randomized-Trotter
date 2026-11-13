"""Actual C worktree API + mock N + immutable saved coverage. No real G."""
import ast
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import h6_df19_price_provider as api
from trottertracks import h4_df10_price_provider as h4
from trottertracks import rpe_nc_coverage as coverage


@contextmanager
def no_science():
    with ExitStack() as stack:
        error=AssertionError('No new prediction, N action, or real cost arithmetic')
        for obj,name in [(p,'evaluate'),(p.accounting,'aggregate_supplied_axes'),(api,'_affine'),
                         (api.H6AffinePriceProvider,'predict_affine'),(h4.H4PriceProvider,'predict_path')]:
            stack.enter_context(patch.object(obj,name,side_effect=error))
        stack.enter_context(patch.dict(h4.mq._scope,{'predict':Mock(side_effect=error)}))
        yield


def mock_shot(task,row):
    lane=dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],candidate=deepcopy(row['candidate']),binding=deepcopy(task['binding']),
        base_time=task['configuration']['base_time'],target_id=task['task_id'],status='CONDITIONAL_ROUND_PASS',N_X=3,N_Y=5,
        alpha_X=task['configuration']['alpha_axis'],alpha_Y=task['configuration']['alpha_axis'])
    return dict(identity=p.identity(task,row),plan=lane,actual=dict(N_X=None,N_Y=None),source={'fixture':'MOCK_ONLY'})


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store=p.SavedArtifacts(ROOT)
        with no_science():
            cls.inputs=coverage.load_inputs(ROOT,cls.store)
            cls.snapshot=coverage.build_coverage(ROOT)

    def test_fixed_components_and_actual_module_are_used(self):
        for name in ['src/trottertracks/h6_df19_price_provider.py','tests/check_h6_df19_price_provider.py']:
            self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show','0b5c493f2b2728b0c7c6c848d2cbf084db8f7f76:'+name],cwd=ROOT))
        for name in ['h4_df10_price_provider.py','h4_df10_rpe_connection.py','h6_rpe_batch_provider.py','c1_stage3_accuracy.py']:
            path='src/trottertracks/'+name
            base='7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae' if name=='h4_df10_rpe_connection.py' else coverage.B_BASE
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',base+':'+path],cwd=ROOT))
        baseline=ast.parse(subprocess.check_output(['git','show',coverage.B_BASE+':src/trottertracks/pf_pr_rpe_pipeline.py'],cwd=ROOT))
        # B's integration SOURCE contains the shared module; C's provider-only
        # snapshot 0b5c493f does not. Exercise the live factories below as before.
        current=ast.parse(subprocess.check_output(['git','show','ac3e1ce87b5ed7d9114abc316545366dd097209e:src/trottertracks/pf_pr_rpe_pipeline.py'],cwd=ROOT))
        current.body=[n for n in current.body if not isinstance(n,ast.FunctionDef) or n.name!='h6_saved_price_provider']
        self.assertEqual(ast.dump(baseline),ast.dump(current))
        self.assertEqual(Path(p.__file__).resolve(),ROOT/'src/trottertracks/pf_pr_rpe_pipeline.py')
        self.assertEqual(Path(p.accounting.__file__).resolve(),ROOT/'src/trottertracks/h4_df10_rpe_connection.py')
        self.assertIs(p.h6_saved_price_provider(self.store.read).pipeline,p)

    def test_real_factory_run_missing_uses_saved_quotes_and_never_retries(self):
        task=deepcopy(self.inputs['H6']['task']);task['candidates']=[r for r in task['candidates'] if 4<=r['m']<=7]
        select={r['candidate']['id']:dict(price_unit='cohort_mean',cohort={4:'CONFIRMATION',5:'H6_PRICE_V2_20261110_Q2_CALIBRATION',6:'H6_PRICE_V2_20261110_Q4_INDEPENDENT',7:'NOT_ACQUIRED'}[r['m']]) for r in task['candidates'] if p.arm(r['candidate'])=='PR'}
        provider=p.h6_saved_price_provider(self.store.read,selections=select);callback=Mock(wraps=provider)
        never_N=Mock(side_effect=AssertionError('N is mocked/supplied'))
        with no_science():
            first=p.run_missing(task,[mock_shot(task,r) for r in task['candidates']],[],n_provider=never_N,c_provider=callback)
            second=p.run_missing(task,first['shots'],first['prices'],n_provider=never_N,c_provider=callback)
        callback.assert_called_once();never_N.assert_not_called();self.assertIsNone(first['totals'])
        unknown=[r for r in second['plan']['rows'] if r['m']==7]
        self.assertTrue(all(r['C_axes']==[] and r['C_record_axes']==['X','Y'] for r in unknown))
        self.assertEqual(second['plan']['C_provider_requests'],[])
        self.assertTrue(all(r['provider_quote']['full_RPE_price_adopted'] is False for r in first['prices']))

    def test_model_qualification_finite_policy_and_uncertainty_reach_real_B(self):
        task=deepcopy(self.inputs['H6']['task']);task['candidates']=[r for r in task['candidates'] if r['m'] in (5,6)]
        select={r['candidate']['id']:dict(kind='model_prediction',**(dict(price_unit='cohort_mean',cohort='H6_PRICE_V2_20261110_Q'+str(r['q_PF'])+('_CALIBRATION' if r['m']==5 else '_INDEPENDENT')) if p.arm(r['candidate'])=='PR' else {})) for r in task['candidates']}
        with no_science():
            records=p.h6_saved_price_provider(self.store.read,selections=select)(task,task['candidates'])
            shots=[mock_shot(task,r) for r in task['candidates']]
            finite=p.plan(task,shots,records,price_policy='finite_cohort_scenario');default=p.plan(task,shots,records)
        for row in finite['rows']:self.assertEqual(bool(row['eligible_C_cohorts']),row['m']==6)
        self.assertFalse(next(r for r in default['rows'] if r['m']==6 and r['arm']=='PR')['eligible_C_cohorts'])
        for record in records:
            self.assertIsNone(record['uncertainty']['joint_covariance']);self.assertIsNone(record['uncertainty']['total_CI'])
            self.assertFalse(record['uncertainty']['cross_round_independence_assumed'])
        pr=next(r for r in records if r['identity']['m']==6 and r['identity']['candidate']['method']=='PR')
        self.assertIsNotNone(pr['uncertainty']['nominal_coefficient_covariance']);self.assertEqual(pr['expectation'],'FINITE_COHORT_ONLY')

    def test_foreign_domain_and_early_family_are_not_repriced(self):
        task=deepcopy(self.inputs['H6']['task']);task['candidates']=[r for r in task['candidates'] if r['m']==3]
        with no_science():out=p.h6_saved_price_provider(self.store.read)(task,task['candidates'])
        self.assertTrue(all(v['mean']['rz_count'] is None and v['provider_quote']['validation']=='OUTSIDE_DOMAIN' for v in out))
        # Existing m3 evidence remains present in the legacy inventory.
        rows=[r for r in self.snapshot['systems']['H6']['rows'] if r['m']==3]
        self.assertTrue(all(r['direct_C_available'] and r['saved_model_C_available'] and not r['provider_domain_matches'] for r in rows))

    def test_52_rows_keep_saved_N_and_actual_unknown_integers(self):
        for name,system in self.snapshot['systems'].items():
            self.assertEqual(len(system['rows']),26);self.assertEqual({r['m'] for r in system['rows']},set(range(13)))
            for row in system['rows']:
                self.assertEqual(row['N_status'],'CONDITIONAL_ROUND_PASS');self.assertIsNone(row['actual_N_X']);self.assertIsNone(row['actual_N_Y'])
                self.assertEqual(str(row['N_X']),row['N_X_decimal']);self.assertIsNone(row['total_resources'])
        hand=self.store.read(coverage.H4_PUBLISHED_N)
        for saved in hand['shots']:
            row=next(r for r in self.snapshot['systems']['H4']['rows'] if r['candidate']['id']==saved['identity']['candidate']['id'])
            self.assertEqual(row['N_X'],saved['plan']['N_X']);self.assertEqual(row['N_Y'],saved['plan']['N_Y'])
        m10=next(r for r in self.snapshot['systems']['H4']['rows'] if r['m']==10 and r['arm']=='PR')
        self.assertEqual(m10['N_X_decimal'],'1070910265913026585642')
        self.assertEqual(self.snapshot['scope'],'STATIC_SAVED_COVERAGE_AND_QUALIFICATION_ONLY');self.assertIsNone(self.snapshot['ranking'])

    def test_model_direct_missing_coverage_matches_saved_points(self):
        rows4=self.snapshot['systems']['H4']['rows'];rows6=self.snapshot['systems']['H6']['rows']
        self.assertEqual({(r['m'],r['arm']) for r in rows4 if r['direct_C_available']},{(3,'PF'),(3,'PR'),(4,'PF'),(4,'PR'),(5,'PF')})
        self.assertEqual({(r['m'],r['arm']) for r in rows6 if r['direct_C_available']},{(m,a) for m in (3,4,5,6) for a in ('PF','PR')})
        self.assertTrue(all(r['unvalidated_long_q'] and not r['direct_C_available'] and not r['saved_model_C_available'] for r in rows6 if r['m']>=7))
        self.assertTrue(all(not r['provider_domain_matches'] for r in rows6 if r['m']<=3))
        for data in self.snapshot['systems'].values():
            for row in data['rows']:
                self.assertFalse(row['numerical_model_entry_executed'])
                if row['arm']=='PR':self.assertEqual(row['qualified_expectation_scenarios'],[])
                else:self.assertTrue(all(c['price_unit']=='deterministic' for c in row['price_contexts']))
        self.assertFalse(self.snapshot['missing_covariance_is_zero'])
        self.assertFalse(any('progress.json' in source or '.npz' in source or '/private/' in source for source in self.snapshot['sources']))

    def test_published_N_overlay_keeps_missing_fail_and_conflicts_distinct(self):
        data=self.inputs['H4'];task=data['task'];full=self.store.read(coverage.H4_PUBLISHED_N)
        partial={'shots':[deepcopy(n) for n in full['shots'] if n['identity']['m']==3]}
        with no_science():
            base=coverage.accept_saved_N(task,[],partial,coverage.H4_PUBLISHED_N)
            self.assertEqual(len(base),2)
            filled=coverage.accept_saved_N(task,base,full,coverage.H4_PUBLISHED_N)
        self.assertEqual(len(filled),26)
        invalid=deepcopy(full);invalid['shots'][0]['identity']['binding']['state']='FOREIGN'
        with self.assertRaises(ValueError):coverage.accept_saved_N(task,[],invalid,coverage.H4_PUBLISHED_N)
        conflicting=deepcopy(partial);conflicting['shots'][0]['plan']['N_X']+=1
        with self.assertRaisesRegex(ValueError,'CONFLICTING_SAVED_N'):coverage.accept_saved_N(task,base,conflicting,coverage.H4_PUBLISHED_N)
        fail=deepcopy(partial);fail['shots'][0]['plan'].update(status='NUMERICAL_GATE_FAILURE',N_X=None,N_Y=None)
        accepted=coverage.accept_saved_N(task,[],fail,coverage.H4_PUBLISHED_N)
        self.assertIn(fail['shots'][0]['identity']['candidate']['id'],p.plan(task,accepted,[])['unqualified_N'])


if __name__=='__main__':unittest.main(verbosity=2)
