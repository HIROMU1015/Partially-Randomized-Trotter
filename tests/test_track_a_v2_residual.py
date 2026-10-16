"""Artificial data only: no molecular inputs, counter, compiler or sampling."""
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import track_a_v2_residual_runner as runner
m=runner.model


def fixture():
    entry=dict(cell_id='B2_p5_q1_2nd_r1_K2',scope='H6_EXPLORATION92',T=.8,
        cell=dict(prefix=5,q=1,formula='2nd',r=1,K=2,R=1),execution_sources=['origin'],
        compiler_hash=m.COMPILER,paired_groups=8,wrapper_phase='exploration')
    values={k:'fixed-'+k for k in (*m.CRITICAL,*m.PROOF)}
    values.update(compiler_hash=m.COMPILER,origin_sources=['origin'],control='symmetric_directional',
                  sample_scope='exploration',reuse_cohort='immutable-wrapper-cohort')
    expected=dict(key=['H6',entry['scope'],entry['cell_id']],observed=deepcopy(entry),
        values=values,metadata_resolved=True,evidence_refs=['external-pinned-receipt'])
    theory=[]
    for model in ('A','B'):
        for axis in ('cosine','sine'):
            theory.append(dict(system='H6',candidate_id=entry['cell_id'],model=model,axis=axis,
                cell=entry['cell'],theory_profile='explicit_rz_sx_x_cx_v1',control='symmetric_directional',
                lower=dict(rz=dict(numerator='100',denominator='1')),
                upper=dict(rz=dict(numerator='101',denominator='1'))))
    connections=[]
    for model in ('A','B'):
        for epsilon in (.05,.01):
            connections.append(dict(system='H6',candidate_id=entry['cell_id'],scope=entry['scope'],model=model,
                metric='RZ',epsilon=epsilon,factor='1.0',sample_count=8,sample_phase='exploration',
                compiled_origin_SOURCE='origin',saved_axis_means=dict(cosine=60,sine=61)))
    return theory,connections,{'H6':[entry]},[expected]


def artificial_fit_rows():
    rows=[]
    for holdout in (False,True):
        for family in ('PR','PF'):
            for i,x in enumerate((2,4)):
                rows.append(dict(id=f'{holdout}-{family}-{i}',system='H6',model='A',axis='cosine',stratum='s',
                    family=family,holdout=holdout,lower=str(x),upper=str(x),compiled=str(3+2*x)))
    return rows


class MetadataTests(unittest.TestCase):
    def test_dedup_epsilon_factor_models(self):
        args=fixture();rows,a=m.prepare_observations(*args)
        self.assertEqual(len(rows),4);self.assertEqual(a['provenance']['status_counts'],{'METADATA_RESOLVED_CONDITIONAL_ONLY':1})
    def test_unresolved_no_imputation(self):
        t,c,i,r=fixture();r[0]['values']['compiler_profile']=None
        rows,a=m.prepare_observations(t,c,i,r)
        self.assertEqual(rows,[]);self.assertIn('MISSING:compiler_profile',a['provenance']['rows'][0]['reasons'])
    def test_registry_cannot_promote_without_evidence(self):
        t,c,i,r=fixture();r[0]['evidence_refs']=[]
        self.assertEqual(m.prepare_observations(t,c,i,r)[0],[])
    def test_mismatched_observed_source_rejected(self):
        t,c,i,r=fixture();i['H6'][0]['execution_sources']=['other']
        with self.assertRaisesRegex(ValueError,'PROVENANCE_MISMATCH'):m.prepare_observations(t,c,i,r)
    def test_profile_and_variant_missing_excluded(self):
        t,c,i,r=fixture();r[0]['values']['basis_variant']=None
        self.assertEqual(m.prepare_observations(t,c,i,r)[0],[])
    def test_duplicate_registry_and_extra_registry_rejected(self):
        t,c,i,r=fixture()
        with self.assertRaisesRegex(ValueError,'DUPLICATE_REGISTRY'):m.prepare_observations(t,c,i,r+r)
        r2=deepcopy(r[0]);r2['key'][2]='foreign'
        with self.assertRaisesRegex(ValueError,'COVERAGE'):m.metadata_audit(i,r+[r2])
    def test_conflicting_repeat_rejected(self):
        t,c,i,r=fixture();c[-1]['saved_axis_means']['cosine']=62
        with self.assertRaisesRegex(ValueError,'CONFLICTING_REPEATED'):m.prepare_observations(t,c,i,r)
    def test_origin_mismatch_rejected(self):
        t,c,i,r=fixture();c[0]['compiled_origin_SOURCE']='foreign'
        with self.assertRaisesRegex(ValueError,'ORIGIN_MISMATCH'):m.prepare_observations(t,c,i,r)
    def test_count_and_phase_cannot_change_between_scopes(self):
        for name,value in [('sample_count',32),('sample_phase','confirmation')]:
            t,c,i,r=fixture();c[0][name]=value
            with self.assertRaisesRegex(ValueError,'SAMPLE_'):m.prepare_observations(t,c,i,r)
    def test_missing_pair_rejected(self):
        t,c,i,r=fixture();c[0]['saved_axis_means']={'cosine':60}
        with self.assertRaisesRegex(ValueError,'AXIS_PAIR'):m.prepare_observations(t,c,i,r)
    def test_zero_negative_nonfinite_missing_never_zero_residual(self):
        for value in (0,-1,float('nan'),float('inf'),None):
            t,c,i,r=fixture()
            for x in c:x['saved_axis_means']['cosine']=value
            rows,a=m.prepare_observations(t,c,i,r)
            self.assertTrue(all(x['axis']=='sine' for x in rows));self.assertTrue(a['skipped'])
    def test_missing_compiled_is_skipped(self):
        t,c,i,r=fixture()
        for x in c:x['saved_axis_means']=None
        self.assertEqual(m.prepare_observations(t,c,i,r)[0],[])
    def test_theory_duplicate_or_missing_axis(self):
        t,c,i,r=fixture()
        with self.assertRaisesRegex(ValueError,'DUPLICATE_THEORY'):m.prepare_observations(t+t[:1],c,i,r)
        with self.assertRaisesRegex(ValueError,'MISSING_REGISTERED_THEORY'):m.prepare_observations(t[:-1],c,i,r)
    def test_theory_interval_and_profile(self):
        t,c,i,r=fixture();t[0]['lower']['rz']['numerator']='200'
        with self.assertRaisesRegex(ValueError,'INTERVAL_ORDER'):m.prepare_observations(t,c,i,r)
        t[0]['theory_profile']='foreign'
        with self.assertRaisesRegex(ValueError,'PROFILE'):m.prepare_observations(t,c,i,r)
    def test_pf_reuse_proof_dedup_and_conflict(self):
        t,c,i,r=fixture();old=i['H6'][0]['cell_id'];new='B0_p5_q1_2nd'
        i['H6'][0]['cell_id']=new;r[0]['key'][2]=new;r[0]['observed']['cell_id']=new
        for row in t:row['candidate_id']=new
        for row in c:row['candidate_id']=new
        second=deepcopy(i['H6'][0]);second['scope']='H6_CONFIRMATION_UNION8';i['H6'].append(second)
        second_r=deepcopy(r[0]);second_r['key'][1]=second['scope'];second_r['observed']['scope']=second['scope'];r.append(second_r)
        extra=deepcopy(c)
        for x in extra:x['scope']=second['scope']
        rows,_=m.prepare_observations(t,c+extra,i,r);self.assertEqual(len(rows),4)
        for x in extra:x['saved_axis_means']['cosine']=100
        with self.assertRaisesRegex(ValueError,'CONFLICTING_REUSED_PF'):m.prepare_observations(t,c+extra,i,r)
    def test_pf_missing_reuse_cohort_not_guessed(self):
        t,c,i,r=fixture();i['H6'][0]['cell_id']='B0_p5_q1_2nd';r[0]['key'][2]=i['H6'][0]['cell_id'];r[0]['observed']['cell_id']=i['H6'][0]['cell_id']
        for x in c:x['candidate_id']=i['H6'][0]['cell_id']
        r[0]['values']['reuse_cohort']=None
        with self.assertRaisesRegex(ValueError,'PF_REUSE'):m.prepare_observations(t,c,i,r)
    def test_holdout_fixed_group(self):
        cell=dict(prefix=11,q=4,formula='4th',r=1,K=2)
        expected=int.from_bytes(hashlib.sha256(b'["H6",11,4,"4th"]').digest(),'big')%5==0
        self.assertEqual(m.holdout(cell,'H6'),expected)
        self.assertEqual(m.holdout(dict(cell,r=8,K=4),'H6'),expected)


class FitTests(unittest.TestCase):
    def test_exact_threshold_and_neighbors(self):
        self.assertEqual(m.threshold_status(Fraction(1,20)),'STOP_RESIDUAL_GAP_GE_005')
        self.assertEqual(m.threshold_status(Fraction(1,20)+Fraction(1,10**12)),'STOP_RESIDUAL_GAP_GE_005')
        self.assertEqual(m.threshold_status(Fraction(1,20)-Fraction(1,10**13)),'BOUNDARY_UNRESOLVED')
        self.assertEqual(m.threshold_status(Fraction(49,1000)),'PASS_POINT_DIAGNOSTIC')
    def test_artificial_affine_fit(self):
        result=m.fit_endpoint(artificial_fit_rows(),'lower')
        self.assertEqual(result['status'],'PASS_POINT_DIAGNOSTIC');self.assertAlmostEqual(result['intercept'],3)
        self.assertAlmostEqual(result['slope'],2)
    def test_small_groups_do_not_import_numpy(self):
        with patch.dict(sys.modules,{'numpy':None}):
            self.assertTrue(m.fit_endpoint(artificial_fit_rows()[:2],'lower')['status'].startswith('INCONCLUSIVE'))
    def test_rank_deficiency(self):
        rows=artificial_fit_rows()
        for x in rows:x['lower']='2'
        self.assertEqual(m.fit_endpoint(rows,'lower')['status'],'INCONCLUSIVE_DISTINCT_X')
    def test_negative_slope(self):
        rows=artificial_fit_rows()
        for x in rows:x['compiled']=str(20-2*int(x['lower']))
        with self.assertRaisesRegex(ValueError,'COEFFICIENT'):m.fit_endpoint(rows,'lower')
    def test_nonpositive_prediction(self):
        rows=artificial_fit_rows()
        for x in rows:
            if not x['holdout']:x['compiled']=str(2*int(x['lower'])-3)
            else:x['lower']='1'
        with self.assertRaisesRegex(ValueError,'PREDICTION'):m.fit_endpoint(rows,'lower')
    def test_design_condition_and_numeric_rank(self):
        import numpy as np
        with patch.object(np.linalg,'cond',return_value=1e11):
            with self.assertRaisesRegex(ValueError,'CONDITION'):m.fit_endpoint(artificial_fit_rows(),'lower',np=np)
        with patch.object(np.linalg,'lstsq',return_value=(np.array([1.,1.]),None,1,None)):
            with self.assertRaisesRegex(ValueError,'RANK'):m.fit_endpoint(artificial_fit_rows(),'lower',np=np)
    def test_nonfinite_fit_input_and_output(self):
        rows=artificial_fit_rows();rows[0]['compiled']='1'+'0'*400
        with self.assertRaises((ValueError,OverflowError)):m.fit_endpoint(rows,'lower')
        import numpy as np
        with patch.object(np.linalg,'lstsq',return_value=(np.array([float('nan'),1.]),None,2,None)):
            with self.assertRaisesRegex(ValueError,'COEFFICIENT'):m.fit_endpoint(artificial_fit_rows(),'lower',np=np)
    def test_first_failed_endpoint_stops_before_upper(self):
        reservations=[];outputs=[]
        def fit(*args):return dict(status='STOP_RESIDUAL_GAP_GE_005',difference=.05)
        result=m.evaluate(artificial_fit_rows(),reserve=lambda *a:reservations.append(a),emit=outputs.append,fit=fit)
        self.assertEqual(result['fits'],1);self.assertEqual(len(reservations),1);self.assertEqual(len(outputs),1)
        self.assertEqual(reservations[0][1],'lower')
    def test_upper_failure_no_next_stratum(self):
        calls=[];rows=artificial_fit_rows();rows+= [dict(r,axis='sine') for r in rows]
        def fit(*args):
            calls.append(args[1]);return dict(status='PASS_POINT_DIAGNOSTIC' if len(calls)==1 else 'BOUNDARY_UNRESOLVED')
        result=m.evaluate(rows,reserve=lambda *a:None,emit=lambda *a:None,fit=fit)
        self.assertEqual(calls,['lower','upper']);self.assertEqual(result['status'],'STOP')
    def test_fit_exception_saved_before_stop(self):
        emitted=[]
        def fail(*a):raise ValueError('artificial failure')
        with self.assertRaisesRegex(ValueError,'artificial failure'):m.evaluate(artificial_fit_rows(),reserve=lambda *a:None,emit=emitted.append,fit=fail)
        self.assertEqual(emitted[0]['status'],'STOP_FIT_ERROR')
    def test_no_assessable_not_pass(self):
        result=m.evaluate([],reserve=lambda *a:self.fail(),emit=lambda *a:self.fail())
        self.assertFalse(result['all_strata_passed']);self.assertEqual(result['scientific_status'],'NO_ASSESSABLE_STRATA')
    def test_stratum_limit_before_any_fit(self):
        rows=[dict(artificial_fit_rows()[0],stratum=str(i)) for i in range(65)]
        with self.assertRaisesRegex(ValueError,'STRATUM_CAP'):
            m.evaluate(rows,reserve=lambda *a:self.fail(),emit=lambda *a:self.fail())
    def test_independent_axes_not_averaged(self):
        calls=[];rows=artificial_fit_rows()+[dict(r,axis='sine') for r in artificial_fit_rows()]
        def fit(group,endpoint):
            calls.append((group[0]['axis'],endpoint))
            return dict(status='PASS_POINT_DIAGNOSTIC' if group[0]['axis']=='cosine' else 'STOP_RESIDUAL_GAP_GE_005')
        result=m.evaluate(rows,reserve=lambda *a:None,emit=lambda *a:None,fit=fit)
        self.assertEqual(calls,[('cosine','lower'),('cosine','upper'),('sine','lower')])
        self.assertEqual(result['status'],'STOP')


class BoundaryTests(unittest.TestCase):
    def grant(self):
        cpus=sorted(os.sched_getaffinity(0))[:2]
        config=dict(policy_sha256=m.digest(m.policy()))
        grant=dict(schema='track_a_v2_residual_grant_v1',approved_by_user=True,runtime_authorization=True,
            metadata_conditional_interpretation_approved=True,scope=m.SCOPE,SOURCE='source',config_sha256='config',
            policy_sha256=config['policy_sha256'],limits=m.LIMITS,caps=m.CAPS,one_shot=True,retry=False,resume=False,
            next_stage_authorized=False,allowed_cpus=cpus,driver_cpu=cpus[0],monitor_cpu=cpus[1])
        fresh=dict(source='source',config_sha256='config',allowed_cpus=cpus,monotonic_seconds=100,
            profile_verified=True,resources_idle_verified=True,protected_jobs_unchanged=True,external_RSS_monitor_ready=True)
        return config,grant,fresh
    def test_approved_shape_and_modified_fields(self):
        c,g,f=self.grant();runner.verify_grant(c,g,f,source='source',config_sha='config',now=101)
        for key,value in [('approved_by_user',False),('runtime_authorization',False),('metadata_conditional_interpretation_approved',False),('SOURCE','other'),('caps',{}),('limits',{}),('retry',True),('scope','other')]:
            bad=deepcopy(g);bad[key]=value
            with self.assertRaises(ValueError):runner.verify_grant(c,bad,f,source='source',config_sha='config',now=101)
    def test_stale_future_and_duplicate_cpus(self):
        c,g,f=self.grant()
        for now in (99,131):
            with self.assertRaisesRegex(ValueError,'FRESH'):runner.verify_grant(c,g,f,source='source',config_sha='config',now=now)
        g['monitor_cpu']=g['driver_cpu'];g['allowed_cpus']=[g['driver_cpu']]*2
        with self.assertRaisesRegex(ValueError,'CPUS'):runner.verify_grant(c,g,f,source='source',config_sha='config',now=101)
    def test_gzip_identity_and_bomb_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.gz';raw=b'{"saved":true}';path.write_bytes(gzip.compress(raw))
            expected=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(runner.bounded_json(path,compressed=True,expected=expected),{'saved':True})
            with self.assertRaisesRegex(ValueError,'IDENTITY'):runner.bounded_json(path,compressed=True,expected=dict(bytes=1,sha256='x'))
            path.write_bytes(gzip.compress(b'x'*(16*2**20+1)))
            with self.assertRaisesRegex(ValueError,'CAP'):runner.bounded_json(path,compressed=True,expected=expected)
    def test_source_runtime_and_input_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source.py';src.write_text('literal source')
            saved=root/'input.json';saved.write_text('{}')
            runtime=root/'runtime.py';runtime.write_text('literal runtime')
            receipt=dict(path=str(saved),bytes=saved.stat().st_size,sha256=runner.base.sha(saved))
            config=dict(inputs={name:dict(receipt) for name in runner.INPUTS},source_hashes={'source.py':runner.base.sha(src)},runtime_file_hashes={str(runtime):runner.base.sha(runtime)})
            with patch.object(runner,'ROOT',root):
                runner.verify_bytes(config,'fake-source')
                for path in (src,runtime,saved):
                    raw=path.read_bytes();path.write_bytes(raw+b'x')
                    with self.assertRaises(ValueError):runner.verify_bytes(config,'fake-source')
                    path.write_bytes(raw)
    def test_default_cli_and_draft_denial_no_numpy(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            for name in ('config','grant','fresh-gate'):(directory/name).write_text('{}')
            command=[sys.executable,'-B',str(ROOT/'scripts/track_a_v2_residual_runner.py')]
            result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0)
            self.assertEqual(json.loads(result.stdout)['new_compile'],0)
            for name in ('config','grant','fresh-gate'):command+=['--'+name,str(directory/name)]
            result=subprocess.run(command+['--launch-approved'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('NEW_APPROVED',result.stderr)
            self.assertEqual(sorted(p.name for p in directory.iterdir()),['config','fresh-gate','grant'])


if __name__=='__main__':unittest.main()
