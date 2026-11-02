"""Small stdlib-only regressions of new domain, accounting and reuse gates."""
import copy
import json
import pickle
import types
import unittest
from unittest.mock import Mock, patch
from trottertracks.resource_applicability.h4_b0i import adapter as science
from trottertracks.resource_applicability.h4_b0i_map import contract as c, runtime as r
from trottertracks.resource_applicability.h4_geometry.identity import Stop


def fake_plan():
    tasks=[]
    for g in c.GEOMETRIES:
        for p in c.PREFIXES:
            for q in c.QS:
                for axis in ('cosine','sine'):
                    t=dict(geometry=g,prefix=p,axis=axis,template=science.template(p,q),
                        input_file=dict(path='/home/AbeHiromu/FAKE_NOT_OPENED.npz',bytes=1,sha256='a'*64),
                        input_identity={k:'a'*64 for k in ('H','DF','state','input','compiler','environment')},
                        source_commit='1'*40,original_science_source=science.ORIGINAL_SOURCE,
                        tail_identity=.1,phase_angle=-.8*.1,trajectory_seed=None,trajectory_index=None,
                        numeric_circuit_fingerprint=None,compiled_cost='UNMEASURED_IN_NEW_MAP',
                        original_B0_template='fake',action='VERIFY_REUSE' if (g,p,q,axis) in c.PILOT_KEYS else 'COMPILE',reuse_receipt=None)
                    if t['action']=='VERIFY_REUSE':t['reuse_receipt']=dict(old_task_id='b'*64,expected_numeric_fingerprint='a'*64)
                    t['task_id']=c.digest(t);tasks.append(t)
    return dict(schema_version='h4-b0i-map240-plan-v1',namespace=c.NAMESPACE,method=science.METHOD,
        semantics=science.SEMANTICS,source_commit='1'*40,original_science_source=science.ORIGINAL_SOURCE,
        tasks=tasks,caps=copy.deepcopy(c.CAPS),carry=copy.deepcopy(c.CARRY),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        output='/home/AbeHiromu/'+c.NAMESPACE,control='/home/AbeHiromu/'+c.NAMESPACE+'-control',
        one_shot_lock='/home/AbeHiromu/'+c.NAMESPACE+'.lock',compiler_options={'num_processes':1},**c.FLAGS)


def approval(plan):
    plan.update(approved=True,runtime_authorization=True,allowed_cpus=[0,1,2],absolute_launch_command='SYNTHETIC_ONLY')
    auth=dict(schema_version='h4-b0i-map240-authorization-v1',namespace=c.NAMESPACE,source_commit=plan['source_commit'],
        plan_digest=c.digest(plan),approved=True,runtime_authorization=True,allowed_cpus=[0,1,2],
        role_cpus=dict(worker=0,driver=1,observer=2),absolute_launch_command='SYNTHETIC_ONLY',user_approval_evidence_sha256='e'*64,**c.FLAGS)
    return auth,dict(approved=True,plan_digest=c.digest(plan),authorization_digest=c.digest(auth))


def old_pair(task):
    old=copy.deepcopy(task);old['source_commit']=c.PILOT_SOURCE;old['task_id']='b'*64
    receipt=dict(status='COMPLETE',task_id=old['task_id'],source_commit=c.PILOT_SOURCE,old_cache_reuse=False,
        numeric_circuit_fingerprint='a'*64,metrics={k:1 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')})
    receipt['validation_wrapper_fingerprint']=science.bound_fingerprint(old['template'],old['tail_identity'],old['axis'],old['input_identity'],c.PILOT_SOURCE,'a'*64)
    return old,receipt


class DomainTests(unittest.TestCase):
    def test_complete_domain_and_234_new(self):
        p=fake_plan();c.validate_plan(p)
        self.assertEqual(sum(t['action']=='COMPILE' for t in p['tasks']),234)
        self.assertEqual({c.key(t) for t in p['tasks'] if t['action']=='VERIFY_REUSE'},c.PILOT_KEYS)
    def test_omission_duplicate_and_reorder_stop(self):
        for operation in ('omit','duplicate','reorder'):
            p=fake_plan()
            if operation=='omit':p['tasks'].pop()
            if operation=='duplicate':p['tasks'][-1]=p['tasks'][0]
            if operation=='reorder':p['tasks'][0],p['tasks'][1]=p['tasks'][1],p['tasks'][0]
            with self.assertRaises(Stop):c.validate_plan(p)
    def test_budget_reset_caps_change_and_retry_stop(self):
        for field,value in [('actual_invocations',73023),('charged_bytes',0)]:
            p=fake_plan();p['carry'][field]=value
            with self.assertRaises(Stop):c.validate_plan(p)
        p=fake_plan();p['caps']['new_compile_calls']=240
        with self.assertRaises(Stop):c.validate_plan(p)
        p=fake_plan();p['retry']=True
        with self.assertRaises(Stop):c.validate_plan(p)
    def test_old_grant_and_draft_refused(self):
        p=fake_plan()
        for auth in ({'schema_version':'h4-b0i-pilot6-authorization-v1','approved':True,'runtime_authorization':True},{}):
            with self.assertRaises(Stop):c.authorize(p,auth,{})
    def test_new_binding_and_tamper(self):
        p=fake_plan();a,v=approval(p);c.authorize(p,a,v);v['authorization_digest']='f'*64
        with self.assertRaises(Stop):c.authorize(p,a,v)
    def test_unapproved_execute_has_no_source_or_process_side_effect(self):
        p=fake_plan()
        with patch.object(r,'load_grant',return_value=(p,{},{})),patch.object(r,'source_gate') as gate,patch.object(r.subprocess,'Popen') as spawn:
            with self.assertRaises(Stop):r.execute(None,None,None,None)
        gate.assert_not_called();spawn.assert_not_called()
    def test_bootstrap_passes_paths_not_full_registry(self):
        p=fake_plan();payload=dict(repo='/home/AbeHiromu/fake',plan_path='/home/AbeHiromu/fake/p.json',auth_path='/home/AbeHiromu/fake/a.json',review_path='/home/AbeHiromu/fake/r.json',plan_digest=c.digest(p))
        self.assertLess(len(pickle.dumps(payload)),c.CAPS['bootstrap_frame_bytes'])
        self.assertGreater(len(json.dumps(p)),c.CAPS['bootstrap_frame_bytes'])
    def test_ledger_never_refunds_and_never_charges_reuse(self):
        p=fake_plan();rows=[];ledger=c.CallLedger(p['tasks'],reserve=rows.append)
        for t in p['tasks']:
            if t['action']=='COMPILE':ledger.before_call(t['task_id'])
            else:
                with self.assertRaises(Stop):ledger.before_call(t['task_id'])
        self.assertEqual(len(ledger.reserved),234)
        with self.assertRaises(Stop):ledger.before_call(ledger.reserved[0])
        with self.assertRaises(Stop):ledger.before_call('f'*64)
        self.assertEqual(len(ledger.reserved),234)
    def test_reservation_io_failure_does_not_dispatch(self):
        p=fake_plan();t=next(t for t in p['tasks'] if t['action']=='COMPILE')
        ledger=c.CallLedger(p['tasks'],reserve=Mock(side_effect=OSError('fixture fsync failure')))
        with self.assertRaises(OSError):ledger.before_call(t['task_id'])
        self.assertEqual(ledger.reserved,[])


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.p=fake_plan();self.t=next(t for t in self.p['tasks'] if t['action']=='VERIFY_REUSE');self.old,self.receipt=old_pair(self.t)
    def test_complete_pilot_matching(self):
        c.verify_reuse_match(self.t,self.old,self.receipt)
    def test_axis_phase_input_compiler_mismatch_stop(self):
        for field in ('axis','tail_identity','input_file','input_identity','template'):
            old=copy.deepcopy(self.old)
            if field=='axis':old[field]='sine'
            elif field=='tail_identity':old[field]=.2
            elif field=='input_file':old[field]['sha256']='c'*64
            elif field=='input_identity':old[field]['compiler']='c'*64
            else:old[field]=science.template(5,2)
            with self.assertRaises(Stop):c.verify_reuse_match(self.t,old,self.receipt)
    def test_partial_old_B0_or_bad_fingerprint_stop(self):
        for change in (dict(status='ERROR'),dict(old_cache_reuse=True),dict(validation_wrapper_fingerprint='f'*64)):
            with self.assertRaises(Stop):c.verify_reuse_match(self.t,self.old,{**self.receipt,**change})
    def test_reuse_signed_zero_is_not_aliased(self):
        t=copy.deepcopy(self.t);old=copy.deepcopy(self.old)
        t['tail_identity']=-0.;old['tail_identity']=0.
        with self.assertRaises(Stop):c.verify_reuse_match(t,old,self.receipt)
    def test_reuse_new_fingerprint_bound_to_namespace_source(self):
        fp=c.bound_fingerprint(self.t,self.p['source_commit'],'a'*64)
        self.assertNotEqual(fp,self.receipt['validation_wrapper_fingerprint'])
        self.assertNotEqual(fp,c.bound_fingerprint(self.t,'2'*40,'a'*64))
    def fake_circuits(self,numeric='a'*64):
        return types.SimpleNamespace(wrapper=lambda *x:'FAKE_CIRCUIT',numerical_fingerprint=lambda *x:numeric,metrics=lambda value:{k:1 for k in self.receipt['metrics']})
    def test_numeric_reuse_never_calls_compiler(self):
        compiler=Mock(side_effect=AssertionError('reuse must not compile'))
        with patch.dict('sys.modules',{'trottertracks.resource_applicability.h4_geometry.circuits':self.fake_circuits()}),patch.object(r,'evolution',return_value='FAKE_EVOLUTION'):
            out=r._compile_task(self.p,self.t,{},compiler,self.receipt)
        compiler.assert_not_called();self.assertEqual(out['new_compile_invocations'],0);r.accept_result(self.p,self.t,out)
    def test_numeric_mismatch_no_automatic_compile_fallback(self):
        compiler=Mock()
        with patch.dict('sys.modules',{'trottertracks.resource_applicability.h4_geometry.circuits':self.fake_circuits('f'*64)}),patch.object(r,'evolution',return_value='FAKE_EVOLUTION'):
            with self.assertRaises(Stop):r._compile_task(self.p,self.t,{},compiler,self.receipt)
        compiler.assert_not_called()
    def test_new_task_calls_compiler_once_and_measures_wall(self):
        t=next(t for t in self.p['tasks'] if t['action']=='COMPILE');compiler=Mock(return_value='FAKE_COMPILED')
        with patch.dict('sys.modules',{'trottertracks.resource_applicability.h4_geometry.circuits':self.fake_circuits()}),patch.object(r,'evolution',return_value='FAKE_EVOLUTION'):
            out=r._compile_task(self.p,t,{},compiler)
        compiler.assert_called_once();self.assertEqual(out['new_compile_invocations'],1);self.assertGreaterEqual(out['transpile_wall_seconds'],0.);r.accept_result(self.p,t,out)
    def test_retirement_requires_all_240_receipts(self):
        with self.assertRaises(Stop):r.retire_observer_after_map(Mock(),Mock(),[{}]*239)
    def test_increased_inode_and_bounded_log_projection(self):
        self.assertLess(240*2+32,c.CAPS['run_inode_cap'])
        log_bound=(c.CAPS['stage_wall_seconds']+5)*8192+16384
        self.assertLess(log_bound,c.CAPS['observer_output_bytes'])
        self.assertLess(c.CAPS['observer_output_bytes']+c.CAPS['journal_bytes']+32*2**20,c.CAPS['stage_output_bytes'])


if __name__=='__main__':unittest.main()
