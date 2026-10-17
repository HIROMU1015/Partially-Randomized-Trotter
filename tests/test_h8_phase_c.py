"""Artificial costs/shots/proofs only; candidate coordinates are fixed metadata."""
from copy import deepcopy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import h8_phase_c_runtime as r
import run_h8_phase_c as runner
import prepare_h8_phase_c as prepare
c = r.c
REGISTRY = json.loads((ROOT/'references/registration_v1.json').read_text())


def account():
    return dict(evidence_kind='EMPIRICAL', allowance_certification_claimed_by_caller=False,
        cost_acquisition_eligible=True, eligibility_under_declared_allowance='ELIGIBLE', N_total=30,
        axes=dict(real=dict(shots=10, headroom=.1, u_over_headroom=1e-6),
                  imag=dict(shots=20, headroom=.1, u_over_headroom=1e-6)))


def fixture():
    model = dict(models=[dict(method=m, axis=a, coefficients=[str(100+10*j), '10', '2'],
        status='S2_ANCHORS_AND_REGISTERED_HOLDOUT_PASS') for j, (m, a) in enumerate(
        (('B0','cosine'), ('B0','sine'), ('B0I','cosine'), ('B0I','sine')))])
    model_sha = c.digest(model)
    by = {(x['method'], x['axis']): x for x in model['models']}
    b2 = []
    for cid, axis in c.B2_AXES:
        method, p, q, _ = cid.split('_'); p=int(p[1:]); q=int(q[1:])
        a,b,d = map(F, by[method,axis]['coefficients']); measured=int(a+3*q*(b+d*p))+1
        b2.append(dict(case_id=cid,axis=axis,model_sha256=model_sha,profile_sha256=c.PROFILE,
            compiler_hash=c.COMPILER,native_exit_code=0,reaped=True,fingerprint_before='fixture',
            fingerprint_after='fixture',max_control_error=1e-12,measured_RZ=measured,
            model_check=dict(PASS=True,exact_relative_error='1/1000')))
    data = dict(fixture_kind='ARTIFICIAL', model=model, B1_result_commit=c.B1_RESULT,
        B1_terminal=dict(status='COMPLETE_MANDATORY_STOP', model_sha256=model_sha),
        B2=dict(terminal=dict(status='COMPLETE_MANDATORY_STOP',SOURCE=c.B2_SOURCE,axes=b2,
            fit_performed=False,next_stage_authorized=False,
            validation=dict(status='VALIDATED_ONLY_ON_FOUR_REGISTERED_S4_AXES')),
            cleanup=dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,all_reaped=True,
                cleanup_errors=[],first_stop=None,max_monitor_gap_seconds=1),
            final=dict(status='PASS_MANDATORY_STOP',SOURCE=c.B2_SOURCE,profile_sha256=c.PROFILE),
            launcher_native_exit=dict(native_exit_code=0,wait_completed=True)),accounting=[],measured_axes=[])
    det=REGISTRY['old']+REGISTRY['additional']
    for cid in [x['id'] for x in det]+REGISTRY['confirmation_ids']:
        for e in c.EPSILONS:
            for f in c.FACTORS:
                row=dict(cell_id=cid,epsilon=e,factor=f,accounting=account())
                if cid in REGISTRY['confirmation_ids']:
                    row.update(sample_phase='confirmation',sample_count=32,PR_metric=dict(
                        G_exact=dict(numerator='300',denominator='1'),paired_samples=32,
                        G_standard_error=1.,population_winner_certified=False))
                data['accounting'].append(row)
    for cell in REGISTRY['old']:
        if cell['method']=='B1':
            for axis in c.AXES:
                data['measured_axes'].append(dict(cell_id=cell['id'],axis=axis,RZ=1000,
                    compiler_hash=c.COMPILER,provenance_verified=True,kind='LEGACY_FULL_PF_DECLARED_LIMITS'))
    return data


def raw_fixture(data):
    old={x['id']:x for x in REGISTRY['old']};extra={x['id']:x for x in REGISTRY['additional']}
    raw=dict(B2=data['B2'],model=data['model'],B1_terminal=data['B1_terminal'],B1_anchors={'rows':[]},
        Phase_A=dict(status='COMPLETE_MANDATORY_STOP',signals=[]),B0I_accounting=[],
        D0=dict(H8_EXPLORATION92=[],H8_CONFIRMATION_UNION6=[]))
    for cell in extra.values():
        raw['Phase_A']['signals'].append(dict(cell=cell,precision_rows={str(e):{str(f):account()
            for f in c.FACTORS} for e in c.EPSILONS}))
    for row in data['accounting']:
        cid=row['cell_id'];z=deepcopy(row)
        if cid in old:
            z.update(status='DERIVED_SAVED_SCALAR')
            if old[cid]['method']=='B0I':raw['B0I_accounting'].append(z)
            else:
                if old[cid]['method']=='B1':z['metrics']={'RZ':{'axis_means':{'cosine':1000,'sine':1000}}}
                raw['D0']['H8_EXPLORATION92'].append(z)
        elif cid in REGISTRY['confirmation_ids']:
            z.update(method='B2',metrics={'RZ':z['PR_metric']});raw['D0']['H8_CONFIRMATION_UNION6'].append(z)
    for m in ('B0','B0I'):
        for p in (14,21):
            for q in ((1,8) if m=='B0' else (1,2,4,8)):
                for axis in c.AXES:
                    raw['B1_anchors']['rows'].append(dict(method=m,prefix=p,q=q,axis=axis,
                        formula='2nd',saved_RZ=777,control='symmetric_directional',evidence={'fixture':True},
                        runtime_profile_sha256=c.PROFILE,compiler_hash=c.COMPILER,
                        origin_SOURCE='db559a518214fd27fa96e2f1451911df05c4f4a8' if m=='B0' else
                        '54ed98c5fa5b7190f77255271f4e3bec8db15630'))
    return raw


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.data=fixture();self.patch=patch.object(c,'MODEL_SHA',c.digest(self.data['model']));self.patch.start()
        self.addCleanup(self.patch.stop)

    def evaluate(self,data=None):
        calls=[];events=[]
        out=c.evaluate(self.data if data is None else data,REGISTRY,take=calls.append,
                       emit=lambda kind,row:events.append((kind,row)))
        return out,calls,events

    def test_exact_domain_caps_scopes_and_labels(self):
        result,calls,events=self.evaluate()
        self.assertEqual(result['accounting_rows'],1900);self.assertEqual(len(result['comparisons']),10)
        self.assertEqual(calls.count('comparison'),20);self.assertEqual(calls.count('reaccounting_pass'),1)
        self.assertLessEqual(calls.count('model_axis_estimate'),352)
        self.assertFalse(result['all_candidates_measured']);self.assertFalse(result['next_stage_authorized'])
        statuses={row['cost_status'] for kind,row in events if kind=='accounting_row'}
        self.assertIn('MODEL_CONDITIONAL_POINT',statuses);self.assertIn('SAVED_PR_FINITE_SAMPLE_MEAN_N32',statuses)

    def test_saved_shot_weighted_exact_rational_cost(self):
        _,_,events=self.evaluate()
        row=next(v for k,v in events if k=='accounting_row' and v['cell_id']=='B0_p7_q1_2nd')
        self.assertEqual(F(row['G_exact']),10*(100+10+2*7)+20*(110+10+2*7))

    def test_B2_measurement_precedes_model_only_after_all_pass(self):
        _,_,events=self.evaluate();cid='B0_p15_q1_4th'
        row=next(v for k,v in events if k=='accounting_row' and v['cell_id']==cid)
        expected=sum((10 if x['axis']=='cosine' else 20)*x['measured_RZ']
                     for x in self.data['B2']['terminal']['axes'] if x['case_id']==cid)
        self.assertEqual(F(row['G_exact']),expected)
        self.assertTrue(all(v['provenance']['kind']=='B2_REGISTERED_S4_MEASUREMENT' for v in row['axis_costs'].values()))

    def test_any_B2_failure_prevents_every_C_call(self):
        for i in range(4):
            data=deepcopy(self.data);data['B2']['terminal']['axes'][i]['model_check']['PASS']=False;calls=[]
            with self.subTest(axis=i),self.assertRaisesRegex(ValueError,'B2_AXIS'):
                c.evaluate(data,REGISTRY,take=calls.append)
            self.assertEqual(calls,[])

    def test_B2_STOP_missing_partial_final_or_native_cleanup_rejected(self):
        mutations=[lambda d:d['B2']['terminal'].update(status='STOP'),
            lambda d:d['B2']['terminal']['axes'].pop(),lambda d:d['B2']['final'].update(status='STOP'),
            lambda d:d['B2']['cleanup'].update(all_reaped=False),
            lambda d:d['B2']['launcher_native_exit'].update(native_exit_code=1),
            lambda d:d['B2']['cleanup'].update(max_monitor_gap_seconds=5.1)]
        for change in mutations:
            data=deepcopy(self.data);change(data)
            with self.subTest(change=change),self.assertRaises(ValueError):self.evaluate(data)

    def test_two_percent_boundary_rejected(self):
        self.data['B2']['terminal']['axes'][0]['model_check']['exact_relative_error']='1/50'
        with self.assertRaisesRegex(ValueError,'B2_AXIS'):self.evaluate()

    def test_changed_model_and_profile_rejected(self):
        for change in (lambda d:d['model'].update(changed=True),
                       lambda d:d['B2']['terminal']['axes'][0].update(profile_sha256='wrong')):
            data=deepcopy(self.data);change(data)
            with self.subTest(change=change),self.assertRaises(ValueError):self.evaluate(data)

    def test_ineligible_is_not_zero_or_a_winner(self):
        cid='B0_p7_q1_2nd'
        for row in self.data['accounting']:
            if row['cell_id']==cid:row['accounting'].update(cost_acquisition_eligible=False,eligibility_under_declared_allowance='INELIGIBLE')
        result,_,events=self.evaluate()
        row=next(v for k,v in events if k=='accounting_row' and v['cell_id']==cid)
        self.assertIsNone(row['G_exact']);self.assertFalse(row['eligible'])
        self.assertNotEqual(result['comparisons'][0]['scopes']['old32']['deterministic']['best']['cell_id'],cid)

    def test_missing_eligibility_prevents_full_domain_minimum(self):
        self.data['accounting'][0]['accounting']=None
        result,_,_=self.evaluate();r0=result['comparisons'][0]
        self.assertIsNone(r0['scopes']['old32']['deterministic']['best'])
        self.assertIsNone(r0['reduction_change_percentage_points_exact'])

    def test_missing_eligible_full_PF_cost_stops(self):
        self.data['measured_axes'].pop()
        with self.assertRaisesRegex(ValueError,'FULL_PF_COST_MISSING'):self.evaluate()

    def test_no_PR_exploration_pooling(self):
        for change in (dict(sample_phase='exploration'),dict(sample_count=8)):
            data=deepcopy(self.data);next(x for x in data['accounting'] if x['cell_id'] in REGISTRY['confirmation_ids']).update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'CONFIRMATION_N32'):self.evaluate(data)

    def test_missing_duplicate_or_wrong_factor_rows_rejected(self):
        for change in (lambda d:d['accounting'].pop(),lambda d:d['accounting'].append(d['accounting'][0]),
                       lambda d:d['accounting'][0].update(factor=.1)):
            data=deepcopy(self.data);change(data)
            with self.subTest(change=change),self.assertRaises(ValueError):self.evaluate(data)

    def test_nonfinite_or_unreliable_shots_rejected(self):
        for field,value in (('shots',2**52),('shots',True),('headroom',float('nan')),('u_over_headroom',.011)):
            data=deepcopy(self.data);data['accounting'][0]['accounting']['axes']['real'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.evaluate(data)

    def test_exact_tie_lexical_and_no_eligible_cases(self):
        rows=[dict(cell_id=x,eligible=True,G_exact='3/2') for x in ('z','a')]
        self.assertEqual(c.minimum(rows)['best']['cell_id'],'a')
        self.assertEqual(c.minimum([dict(cell_id='a',eligible=False)])['status'],'NO_ELIGIBLE_CANDIDATE')

    def test_reduction_formula_and_same_policy_old_vs_new(self):
        result,_,_=self.evaluate();row=result['comparisons'][0]
        for scope in row['scopes'].values():
            det=F(scope['deterministic']['best']['G_exact']);pr=F(scope['confirmed_PR']['best']['G_exact'])
            self.assertEqual(F(scope['PR_reduction_percent_exact']),100*(1-pr/det))
        self.assertLessEqual(F(row['scopes']['strengthened184']['deterministic']['best']['G_exact']),
                             F(row['scopes']['old32']['deterministic']['best']['G_exact']))

    def test_model_axis_cache_shared_across_precision_and_scopes(self):
        _,calls,events=self.evaluate()
        self.assertEqual(calls.count('model_axis_estimate'),352-4)
        self.assertEqual(calls.count('saved_row'),1900)

    def test_adapter_saved_schema_without_new_shot_rule(self):
        normalized=r.inputs.normalize(raw_fixture(self.data),REGISTRY)
        result,_,_=self.evaluate(normalized)
        self.assertEqual(result['accounting_rows'],1900)

    def test_adapter_preserves_legacy_profile_unknown_and_uses_B1_anchors(self):
        normalized=r.inputs.normalize(raw_fixture(self.data),REGISTRY)
        self.assertEqual(sum(x['kind']=='B1_REGISTERED_ANCHOR' for x in normalized['measured_axes']),24)
        self.assertTrue(all(x['profile_sha256'] is None for x in normalized['measured_axes']
                            if x['kind']=='LEGACY_FULL_PF_DECLARED_LIMITS'))

    def test_adapter_rejects_phase_A_domain_or_anchor_change(self):
        for change in (lambda d:d['Phase_A']['signals'].pop(),
                       lambda d:d['B1_anchors']['rows'][0].update(runtime_profile_sha256='wrong')):
            raw=raw_fixture(self.data);change(raw)
            with self.subTest(change=change),self.assertRaises(ValueError):r.inputs.normalize(raw,REGISTRY)

    def test_runtime_draft_rejected_before_inputs_output_or_affinity(self):
        with (patch.object(r,'read',return_value={}), patch.object(r,'verify') as verify,
              patch.object(os,'sched_setaffinity') as affinity):
            with self.assertRaisesRegex(ValueError,'NEW_SEALED_C'):runner.launch('unused_config','unused_grant')
            verify.assert_not_called();affinity.assert_not_called()

    def grant_fixture(self):
        cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(r.plan()),resources=deepcopy(r.RESOURCES),
                 B2_receipt_set_sha256='b'*64)
        g=dict(schema='H8_PHASE_C_NEW_GRANT_V1',scope=r.SCOPE,approved_by_user=True,runtime_authorization=True,
            approval_text='ARTIFICIAL',SOURCE=cfg['SOURCE'],config_sha256=c.digest(cfg),plan_sha256=cfg['plan_sha256'],
            model_sha256=c.MODEL_SHA,B1_result_commit=c.B1_RESULT,B2_receipt_set_sha256=cfg['B2_receipt_set_sha256'],
            resources=deepcopy(r.RESOURCES),call_caps=deepcopy(r.CAPS),one_shot=True,retry=False,resume=False,
            refit=False,next_stage_authorized=False,allowed_cpus=[101,103],driver_cpu=101,observer_cpu=103)
        return cfg,g

    def test_new_C_grant_is_source_model_B2_resources_and_scope_bound(self):
        cfg,g=self.grant_fixture()
        with patch.object(os,'sched_getaffinity',return_value={101,103}):
            self.assertEqual(r.authorize(cfg,g),[101,103])
            for key,value in (('approved_by_user',False),('runtime_authorization',False),('SOURCE','bad'),
                ('model_sha256','bad'),('B2_receipt_set_sha256',None),('scope','B1'),('retry',True),
                ('resources',{}),('call_caps',{})):
                bad=deepcopy(g);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):r.authorize(cfg,bad)

    def test_fresh_gate_staleness_and_binding(self):
        cfg,g=self.grant_fixture()
        fresh=dict(SOURCE=cfg['SOURCE'],config_sha256=c.digest(cfg),grant_sha256=c.digest(g),
            allowed_cpus=g['allowed_cpus'],monotonic_seconds=100,source_input_verified=True,
            physical_cores_verified=True,resources_idle_verified=True,quota_verified=True,
            protected_jobs_unchanged=True,external_monitor_ready=True)
        r.fresh_binding(cfg,g,fresh,now=101)
        with self.assertRaises(ValueError):r.fresh_binding(cfg,g,fresh,now=131)
        fresh['protected_jobs_unchanged']=False
        with self.assertRaises(ValueError):r.fresh_binding(cfg,g,fresh,now=101)

    def test_first_launch_stop_is_saved_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);cfg=dict(output=str(d/'out'),control=str(d/'control'),lock=str(d/'lock'),
                SOURCE='a'*40,protected_files=[],protected_worktrees=[])
            with (patch.object(r,'read',return_value=cfg),patch.object(r,'authorize',return_value=[101,103]),
                  patch.object(r,'verify'),patch.object(runner,'fresh',side_effect=ValueError('FIRST_FIXTURE_STOP')),
                  patch.object(os,'sched_setaffinity')):
                with self.assertRaisesRegex(ValueError,'FIRST_FIXTURE_STOP'):runner.launch(d/'cfg',d/'grant')
            stop=json.loads((d/'C_phase_STOP.json').read_text())
            self.assertEqual(stop['first_stop'],'FIRST_FIXTURE_STOP');self.assertFalse(stop['automatic_retry'])
            self.assertTrue((d/'C_LAUNCH_ATTEMPT').exists());self.assertFalse((d/'out').exists())

    def test_strict_JSON_nan_overflow_and_duplicate_rejected(self):
        for text in ('{"x":NaN}','{"x":1e999}','{"x":1,"x":2}'):
            with self.subTest(text=text),self.assertRaises(ValueError):r.loads(text.encode())

    def test_output_exclusive_and_supervisor_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);cfg=dict(output=str(d/'out'),control=str(d/'control'),lock=str(d/'lock'),
                                       protected_files=[],protected_worktrees=[])
            r.path_set(cfg);(d/'control').mkdir();r.path_set(cfg,supervisor_control_exists=True)
            (d/'lock').touch()
            with self.assertRaisesRegex(ValueError,'SPENT'):r.path_set(cfg,supervisor_control_exists=True)

    def test_output_cannot_overlap_a_protected_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);cfg=dict(output=str(d/'out'),control=str(d/'control'),lock=str(d/'lock'),
                protected_files=[],protected_worktrees=[dict(path=str(d))])
            with self.assertRaisesRegex(ValueError,'OVERLAPS'):r.path_set(cfg)

    def test_default_CLIs_only_print_plan(self):
        for name in ('run_h8_phase_c.py','prepare_h8_phase_c.py'):
            value=json.loads(subprocess.check_output([sys.executable,'-P','-B',str(ROOT/'scripts'/name)]))
            self.assertEqual(value['scope'],r.SCOPE);self.assertFalse(value['new_shot_calculation'])


if __name__ == '__main__': unittest.main()
