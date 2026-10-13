"""Small (<=3-qubit) unitary checks and zero-process/zero-transpile mocks."""
import copy
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,Statevector
from trottertracks.resource_applicability.h4_b0i import adapter as a,contract as c,runtime as r
from trottertracks.resource_applicability.h4_geometry import circuits
from trottertracks.resource_applicability.h4_geometry.identity import Stop

ROOT=Path(__file__).absolute().parents[3]


def prep():
    bases=[]
    for angle in (.27,-.42,.19):
        base=QuantumCircuit(2);base.ry(angle,0);base.rx(angle/2,1);bases.append(base)
    return dict(n=2,constant=.11,components=[],deterministic=[
        dict(basis=bases[0],coefficients={():.23,(0,):.37,(1,):-.17}),
        dict(basis=bases[1],coefficients={():.04,(0,):.18,(1,):-.23,(0,1):.13}),
        dict(basis=bases[2],coefficients={():-.03,(0,):-.14,(1,):.31,(0,1):-.08})])


def fake_plan():
    tasks=[]
    for g,p in c.SELECTION:
        for axis in ('cosine','sine'):
            t=dict(geometry=g,prefix=p,axis=axis,template=a.template(p,1),tail_identity=.01,
                source_commit='1'*40,original_science_source=a.ORIGINAL_SOURCE,
                trajectory_seed=None,trajectory_index=None,numeric_circuit_fingerprint=None,compiled_cost='UNMEASURED',
                input_file=dict(path='/home/AbeHiromu/fake.bin',bytes=123,sha256='1'*64),
                input_identity={k:'1'*64 for k in ('H','DF','state','input','compiler','environment')})
            t['task_id']=c.digest(t);tasks.append(t)
    return dict(schema_version='h4-b0i-pilot6-plan-v1',namespace=a.NAMESPACE,original_science_source=a.ORIGINAL_SOURCE,
        method=a.METHOD,semantics=a.SEMANTICS,source_commit='1'*40,tasks=tasks,caps=copy.deepcopy(c.CAPS),carry=copy.deepcopy(c.CARRY),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        output='/home/AbeHiromu/'+a.NAMESPACE,control='/home/AbeHiromu/'+a.NAMESPACE+'-control',
        one_shot_lock='/home/AbeHiromu/'+a.NAMESPACE+'.lock',**c.FLAGS)


def fake_auth(plan):
    # In-memory mock approval only; no real grant is written or issued.
    plan.update(approved=True,runtime_authorization=True,allowed_cpus=[1,2,3],absolute_launch_command='SYNTHETIC_MOCK_ONLY')
    auth=dict(schema_version='h4-b0i-pilot6-authorization-v1',approved=True,runtime_authorization=True,
        plan_digest=c.digest(plan),source_commit=plan['source_commit'],namespace=a.NAMESPACE,allowed_cpus=[1,2,3],
        role_cpus=dict(driver=1,worker=2,observer=3),user_approval_evidence_sha256='2'*64,
        absolute_launch_command='SYNTHETIC_MOCK_ONLY',**c.FLAGS)
    review=dict(approved=True,plan_digest=c.digest(plan),authorization_digest=c.digest(auth))
    return auth,review


def fixture_observation():
    return dict(duration_seconds=.1,scheduler_affinity=[1,2,3],online_cpus=[1,2,3],
        topology=[dict(cpu=i,package=0,core=i,busy_fraction=.0) for i in (1,2,3)],
        memory=dict(observed_at=10.,available=c.CAPS['admission_bytes'],process_cpus=[1,2,3],psi_full_by_scope={'host':0.}),
        filesystem=dict(available_bytes=c.CAPS['storage_required_bytes'],available_inodes=256),quota=dict(status='KNOWN',items=[dict(status='DISABLED')]))


class PhaseTests(unittest.TestCase):
    def test_lower_scalar_ignores_saved_upper(self):
        matrices=[[[1,999],[2,3]],[[.4,float('nan')],[-.7,-.2]]]
        self.assertAlmostEqual(a.tail_scalar_lower(matrices,[.5,-.3],1),-.0915,places=14)
        self.assertAlmostEqual(a.tail_scalar_lower(matrices,[.5,-.3],0),4.25-.0915,places=14)
        self.assertEqual(a.tail_scalar_lower(matrices,[.5,-.3],2),0.)
    def test_nonfinite_lower_stops(self):
        with self.assertRaises(Stop):a.tail_scalar_lower([[[1,0],[float('nan'),2]]],[1.],0)
    def test_no_double_count_and_B0_unchanged(self):
        b=prep();before=circuits.serialize(circuits.build_evolution(b,a.template(2,1),[[]]),'cosine')
        x=a.add_tail_identity(b,a.template(2,1),.13)
        self.assertEqual(b['constant'],.11);self.assertEqual(x['constant'],.24)
        self.assertEqual(b['deterministic'][0]['coefficients'][()],.23)
        self.assertIsNot(x['deterministic'],b['deterministic'])
        with self.assertRaises(Stop):a.add_tail_identity(x,a.template(2,1),.13)
        self.assertEqual(before,circuits.serialize(circuits.build_evolution(b,a.template(2,1),[[]]),'cosine'))
    def test_method_separation(self):
        spec=a.template(2,1);spec['method']='B0'
        with self.assertRaises(Stop):a.add_tail_identity(prep(),spec,.1)
    def test_no_nonidentity_tail(self):
        b=prep();b['components']=[{'support':(0,)}]
        with self.assertRaises(Stop):a.add_tail_identity(b,a.template(2,1),.1)
    def test_kept_prefix_validation(self):
        with self.assertRaises(Stop):a.add_tail_identity(prep(),a.template(1,1),.1)
    def test_unitary_relative_phase_all_q_signs(self):
        for q in (1,2,4,8):
            for scalar in (0.,.137,-.219):
                with self.subTest(q=q,scalar=scalar):
                    b=prep();spec=a.template(2,q);old=Operator(circuits.build_evolution(b,spec,[[] for _ in range(q)])).data
                    new=Operator(a.evolution(b,spec,scalar)).data
                    self.assertTrue(np.allclose(new[:4,:4],np.eye(4),atol=2e-13,rtol=0))
                    self.assertTrue(np.allclose(new[4:,4:],np.exp(-1j*.8*scalar)*old[4:,4:],atol=2e-13,rtol=0))
                    self.assertTrue(np.allclose(new[:4,4:],0,atol=2e-13,rtol=0))
    def test_phase_location_count_and_total(self):
        for q in (1,2,4,8):
            b=prep();spec=a.template(2,q);old=circuits.build_evolution(b,spec,[[] for _ in range(q)]);new=a.evolution(b,spec,.137)
            self.assertEqual([i.operation.name for i in old.data],[i.operation.name for i in new.data])
            diffs=[float(y.operation.params[0])-float(x.operation.params[0]) for x,y in zip(old.data,new.data)
                   if x.operation.name=='p']
            self.assertEqual(sum(abs(d)>1e-15 for d in diffs),q)
            self.assertAlmostEqual(sum(diffs),-.8*.137,places=14)
            self.assertEqual(new.data[0].operation.name,'p');self.assertEqual(new.find_bit(new.data[0].qubits[0]).index,2)
    def test_cosine_sine_saved_relation(self):
        psi=np.array([1,.3j,-.2,.4+.2j]);psi/=np.linalg.norm(psi)
        for q in (1,2,4,8):
            spec=a.template(2,q);old=circuits.build_evolution(prep(),spec,[[] for _ in range(q)])
            z=np.vdot(psi,Operator(old).data[4:,4:]@psi)
            scalar=-.137;expected=a.rotate_saved_signal(z,.8,scalar);new=a.evolution(prep(),spec,scalar)
            for axis in ('cosine','sine'):
                measured=circuits.wrapper(new,axis)
                state=Statevector(np.r_[psi,np.zeros(4)]).evolve(measured.remove_final_measurements(inplace=False))
                result=sum(abs(v)**2 for v in state.data[:4])-sum(abs(v)**2 for v in state.data[4:])
                self.assertAlmostEqual(result,expected.real if axis=='cosine' else expected.imag,places=12)
                if axis=='sine':self.assertEqual(measured.data[-3].operation.name,'sdg')
    def test_controlled_phase_is_not_global(self):
        spec=a.template(2,1);old=circuits.build_evolution(prep(),spec,[[]]);new=a.evolution(prep(),spec,.4)
        self.assertGreater(np.linalg.norm(Operator(new).data-np.exp(-1j*.8*.4)*Operator(old).data),.1)
    def test_zero_scalar_numeric_matches_but_method_identity_differs(self):
        spec=a.template(2,1);old=circuits.build_evolution(prep(),spec,[[]]);new=a.evolution(prep(),spec,0.)
        self.assertEqual(circuits.numerical_fingerprint(old,'cosine'),circuits.numerical_fingerprint(new,'cosine'))
        self.assertNotEqual(a.METHOD,'B0')
    def test_fingerprint_source_input_phase_axis_control(self):
        spec=a.template(2,1);ident={k:'1'*64 for k in ('H','DF','state','input','compiler','environment')};n='2'*64
        base=a.bound_fingerprint(spec,.1,'cosine',ident,'1'*40,n)
        values=[a.bound_fingerprint(spec,.2,'cosine',ident,'1'*40,n),a.bound_fingerprint(spec,.1,'sine',ident,'1'*40,n),
            a.bound_fingerprint(spec,.1,'cosine',ident,'2'*40,n),a.bound_fingerprint(spec,.1,'cosine',{**ident,'input':'3'*64},'1'*40,n),
            a.bound_fingerprint({**spec,'control':'different'},.1,'cosine',ident,'1'*40,n),
            a.bound_fingerprint(a.template(2,2),.1,'cosine',ident,'1'*40,n)]
        self.assertTrue(all(x!=base for x in values));self.assertEqual(len(set(values)),len(values))
    def test_nonfinite_phase_fingerprint(self):
        ident={k:'1'*64 for k in ('H','DF','state','input','compiler','environment')}
        with self.assertRaises(Stop):a.bound_fingerprint(a.template(2,1),float('nan'),'cosine',ident,'1'*40,'2'*64)
    def test_saved_target_bias_three_cases(self):
        path=ROOT/'artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/saved_b0_phase_sensitivity_v1.json'
        rows=json.loads(path.read_text())
        for g,p in c.SELECTION:
            v=next(v for v in rows if v['geometry']==g and v['prefix']==p and '-q1-' in v['original_template_id'])
            z=a.rotate_saved_signal(complex(*v['saved_corrected_signal']),.8,v['tail_identity']);target=complex(*v['target'])
            self.assertAlmostEqual(abs(z.real-target.real),v['counterfactual_bias']['cosine'],places=14)
            self.assertAlmostEqual(abs(z.imag-target.imag),v['counterfactual_bias']['sine'],places=14)
    def test_original_science_source_bytes_unchanged(self):
        d=json.loads((ROOT/'artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/provenance_v1.json').read_text())
        for v in d['static_source_inspection']:
            if v['commit']==a.ORIGINAL_SOURCE:self.assertEqual(hashlib.sha256((ROOT/v['path']).read_bytes()).hexdigest(),v['sha256'])


class ContractTests(unittest.TestCase):
    def test_registry_six_and_in_memory_approval(self):
        plan=fake_plan();auth,review=fake_auth(plan);self.assertEqual(c.authorize(plan,auth,review),[1,2,3])
    def test_unapproved_no_side_effects(self):
        with patch.object(r,'source_gate',side_effect=AssertionError('source called')),patch.object(subprocess,'Popen',side_effect=AssertionError('spawn')):
            with self.assertRaises(Stop):r.execute(ROOT,fake_plan(),{'approved':False},{'approved':False})
    def test_old_grant_rejected(self):
        p=fake_plan();au,re=fake_auth(p);au['schema_version']='h4-newhost-authorization-v2'
        with self.assertRaises(Stop):c.authorize(p,au,re)
    def test_unapproved_plan_even_with_mock_auth(self):
        p=fake_plan();au,re=fake_auth(p);p['approved']=False
        with self.assertRaises(Stop):c.authorize(p,au,re)
    def test_plan_auth_review_tamper(self):
        for location in ('plan','auth','review'):
            p=fake_plan();au,re=fake_auth(p)
            if location=='plan':p['tasks'][0]['tail_identity']+=.1
            elif location=='auth':au['plan_digest']='3'*64
            else:re['authorization_digest']='3'*64
            with self.assertRaises(Stop):c.authorize(p,au,re)
    def test_no_scope_expansion_retry_or_refund(self):
        for key in ('retry','automatic_scope_expansion','next_stage_authorized'):
            p=fake_plan();p[key]=True
            with self.assertRaises(Stop):c.validate_plan(p)
        p=fake_plan();p['carry']['actual_invocations']=0
        with self.assertRaises(Stop):c.validate_plan(p)
    def test_seventh_duplicate_unknown_calls_rejected(self):
        p=fake_plan();events=[];ledger=c.CallLedger(p['tasks'],reserve=events.append)
        for t in p['tasks']:ledger.before_call(t['task_id'])
        self.assertEqual(len(events),6)
        for key in (p['tasks'][0]['task_id'],'9'*64):
            with self.assertRaises(Stop):ledger.before_call(key)
    def test_failed_reservation_never_calls(self):
        p=fake_plan();ledger=c.CallLedger(p['tasks'],reserve=Mock(side_effect=OSError('disk full')))
        with self.assertRaises(OSError):ledger.before_call(p['tasks'][0]['task_id'])
        self.assertEqual(ledger.reserved,[])
    def test_no_seed_namespace_or_compiler_relabel(self):
        p=fake_plan();p['tasks'][0]['trajectory_seed']=17
        with self.assertRaises(Stop):c.validate_plan(p)
        p=fake_plan();p['namespace']='old-H4-run'
        with self.assertRaises(Stop):c.validate_plan(p)
    def test_fresh_admission_and_failure_cases(self):
        p=fake_plan();auth,_=fake_auth(p);c.fresh_gate(p,auth,fixture_observation(),now=10.1)
        for kind in ('age','host','cgroup','memory','bytes','inodes','quota','CPU','busy','core'):
            v=fixture_observation()
            if kind=='age':v['memory']['observed_at']=1.
            elif kind=='host':v['memory']['psi_full_by_scope']['host']=1.
            elif kind=='cgroup':v['memory']['psi_full_by_scope']['/nonroot']=.1
            elif kind=='memory':v['memory']['available']-=1
            elif kind=='bytes':v['filesystem']['available_bytes']-=1
            elif kind=='inodes':v['filesystem']['available_inodes']=255
            elif kind=='quota':v['quota']['status']='UNKNOWN'
            elif kind=='CPU':v['scheduler_affinity']=[1,2]
            elif kind=='busy':v['topology'][0]['busy_fraction']=.3
            else:v['topology'][1]['core']=v['topology'][0]['core']
            with self.subTest(kind=kind),self.assertRaises(Stop):c.fresh_gate(p,auth,v,now=10.1)
    def test_explicit_CPU_rights_required(self):
        p=fake_plan();auth,re=fake_auth(p);auth['allowed_cpus']=[];re['authorization_digest']=c.digest(auth)
        with self.assertRaises(Stop):c.authorize(p,auth,re)
    def test_complete_only_metrics_source_numeric(self):
        p=fake_plan();task=p['tasks'][0];numeric='3'*64
        result=dict(status='COMPLETE',task_id=task['task_id'],source_commit=p['source_commit'],plan_digest=c.digest(p),
            old_cache_reuse=False,numeric_circuit_fingerprint=numeric,
            validation_wrapper_fingerprint=a.bound_fingerprint(task['template'],task['tail_identity'],task['axis'],task['input_identity'],p['source_commit'],numeric),
            metrics=dict(rz_count=4,rz_depth=3,cx_count=2,cx_depth=2,total_depth=8,circuit_size=10),trajectory_seed=None)
        self.assertEqual(r.accept_result(p,task,result),result)
        for key,value in [('status','RESERVED'),('old_cache_reuse',True),('source_commit','2'*40),('numeric_circuit_fingerprint','4'*64)]:
            with self.subTest(key=key),self.assertRaises(Stop):r.accept_result(p,task,{**result,key:value})
    def test_partial_pipe_timeout_preserves_monitoring(self):
        p=fake_plan();process=Mock();process.stdout.fileno.return_value=99;process.poll.return_value=None;monitor=Mock()
        with patch.object(r,'poll_owned',side_effect=lambda m,p:m.poll()),patch.object(r.time,'monotonic',side_effect=[10,10,11,611]),patch.object(r.select,'select',return_value=([process.stdout],[],[])),patch.object(r.os,'read',return_value=b'\0'):
            with self.assertRaises(Stop):r.receive_wait(process,monitor,p,0)
        self.assertGreater(monitor.poll.call_count,0)
    def test_signed_zero_semantic_fingerprint(self):
        p=fake_plan();t=p['tasks'][0]
        self.assertNotEqual(a.bound_fingerprint(t['template'],0.,t['axis'],t['input_identity'],p['source_commit'],'2'*64),
            a.bound_fingerprint(t['template'],-0.,t['axis'],t['input_identity'],p['source_commit'],'2'*64))
    def test_worker_core_two_axes_mock_compiler(self):
        plan=fake_plan();plan['compiler_options']=dict(seed_transpiler=17,num_processes=1)
        invocations=[]
        compiled=Mock();compiled.count_ops.return_value={'rz':11,'cx':3};compiled.depth.return_value=4;compiled.size.return_value=20
        def compiler(circuit,**options):
            self.assertEqual(circuit.num_qubits,3);self.assertEqual(options,plan['compiler_options']);invocations.append(circuit)
            return compiled
        for axis in ('cosine','sine'):
            task={**plan['tasks'][0],'template':a.template(2,1),'prefix':2,'axis':axis}
            result=r._compile_task(plan,task,prep(),compiler)
            self.assertEqual(result['metrics']['rz_count'],11);self.assertFalse(result['old_cache_reuse'])
        self.assertEqual(len(invocations),2)
    def test_worker_core_input_mutation_refused(self):
        plan=fake_plan();plan['compiler_options']={};task={**plan['tasks'][0],'template':a.template(2,1),'prefix':2}
        compiled=Mock();compiled.count_ops.return_value={};compiled.depth.return_value=0;compiled.size.return_value=0
        def compiler(circuit,**options):circuit.p(.1,2);return compiled
        with self.assertRaises(Stop):r._compile_task(plan,task,prep(),compiler)
    def test_combined_inode_limit_and_terminal_reserve(self):
        from types import SimpleNamespace
        plan=fake_plan()
        with patch.object(r,'owned_bytes',return_value=(0,128)),patch.object(Path,'stat',return_value=SimpleNamespace(st_blocks=0)),patch.object(r.os,'statvfs',return_value=SimpleNamespace(f_bavail=10**9,f_frsize=4096,f_favail=1000)):
            with self.assertRaises(Stop):r.disk_guard(plan)
        with patch.object(r,'owned_bytes',return_value=(plan['caps']['stage_output_bytes']//2,1)),patch.object(Path,'stat',return_value=SimpleNamespace(st_blocks=0)),patch.object(r.os,'statvfs',return_value=SimpleNamespace(f_bavail=10**9,f_frsize=4096,f_favail=1000)):
            with self.assertRaises(Stop):r.disk_guard(plan)
    def test_host_oom_change_fail_closed(self):
        monitor=Mock();monitor.host_oom_baseline=7
        with patch.object(r,'disk_guard'),patch.object(r,'host_oom_counter',return_value=8):
            with self.assertRaises(Stop):r.poll_owned(monitor,fake_plan())
    def test_worker_first_error_is_not_hidden_by_secondary_schema(self):
        p=fake_plan()
        with self.assertRaisesRegex(Stop,'original mock compiler failure'):
            r.accept_result(p,p['tasks'][0],{'status':'ERROR','error':'original mock compiler failure'})


class CleanupTests(unittest.TestCase):
    def make(self,count=3):
        pairs=[]
        for _ in range(count):
            process=Mock();process.poll.return_value=None;process.stdin=io.BytesIO();process.stdout=io.BytesIO();owner=Mock();pairs.append((process,owner))
        return pairs
    def test_exit_race_first_middle_all(self):
        for indices in ((0,),(1,),(0,1,2)):
            pairs=self.make();monitor=Mock();first=c.FirstStop()
            for i in indices:pairs[i][1].terminate.return_value=False
            self.assertTrue(c.cleanup(pairs,monitor,first))
            for p,o in pairs:self.assertTrue(p.stdin.closed and p.stdout.closed);p.wait.assert_called_once();o.close.assert_called_once()
    def test_non_ESRCH_continues_and_is_failure(self):
        pairs=self.make();pairs[0][1].terminate.side_effect=PermissionError('not ESRCH');first=c.FirstStop()
        self.assertFalse(c.cleanup(pairs,Mock(),first));self.assertIn('not ESRCH',first.reason)
        pairs[0][0].wait.assert_called_once()
        pairs[2][1].terminate.assert_called_once();pairs[2][0].wait.assert_called_once()
    def test_double_cleanup_and_first_cause(self):
        pairs=self.make(1);pairs[0][0].poll.return_value=0;first=c.FirstStop();first.keep('original STOP')
        self.assertTrue(c.cleanup(pairs,Mock(),first));self.assertTrue(c.cleanup(pairs,Mock(),first));self.assertEqual(first.reason,'original STOP')
    def test_timeout_escalates_only_owned_pidfd(self):
        pairs=self.make(1);pairs[0][0].wait.side_effect=[subprocess.TimeoutExpired('mock',2),0]
        self.assertTrue(c.cleanup(pairs,Mock(),c.FirstStop()));self.assertEqual(pairs[0][1].terminate.call_args_list[-1].args,(signal.SIGKILL,))
    def test_bootstrap_owner_failure_direct_Popen_reap(self):
        pairs=self.make(1);pairs[0]=(pairs[0][0],None)
        self.assertTrue(c.cleanup(pairs,Mock(),c.FirstStop()));pairs[0][0].terminate.assert_called_once()
    def test_old_pidfd_ESRCH_and_non_ESRCH(self):
        from trottertracks.resource_applicability.h4_geometry.observer import OwnedIdentity
        owner=OwnedIdentity.__new__(OwnedIdentity);owner.fd=77;owner.sample=Mock(return_value={})
        with patch.object(signal,'pidfd_send_signal',side_effect=OSError(errno.ESRCH,'exited')):self.assertFalse(owner.terminate())
        with patch.object(signal,'pidfd_send_signal',side_effect=OSError(errno.EPERM,'denied')):
            with self.assertRaises(OSError):owner.terminate()
    def test_ownership_mismatch_never_signals(self):
        from trottertracks.resource_applicability.h4_geometry.observer import OwnedIdentity
        owner=OwnedIdentity.__new__(OwnedIdentity);owner.fd=77;owner.sample=Mock(side_effect=Stop('identity lost'))
        with patch.object(signal,'pidfd_send_signal') as send:self.assertFalse(owner.terminate());send.assert_not_called()
    def test_atomic_consumed_one_shot_lock(self):
        directory=Path(os.environ['H4_B0I_TEST_TMP']);path=directory/'one-shot-test.lock'
        c.claim_lock(path,'3'*64)
        with self.assertRaises(FileExistsError):c.claim_lock(path,'3'*64)
        self.assertTrue(path.exists())
    def test_observer_retired_before_worker_exit_only_after_six(self):
        monitor=Mock();monitor.owner.terminate.return_value=True;monitor.process.returncode=-signal.SIGTERM
        journal=Mock();rows=[{'task_id':str(i)} for i in range(6)]
        r.retire_observer_after_six(monitor,journal,rows)
        monitor.poll.assert_called_once();monitor.process.wait.assert_called_once();journal.write.assert_called_once()
        with self.assertRaises(Stop):r.retire_observer_after_six(monitor,journal,rows[:5])
    def test_unexpected_observer_exit_not_complete(self):
        monitor=Mock();monitor.owner.terminate.return_value=True;monitor.process.returncode=1
        with self.assertRaises(Stop):r.retire_observer_after_six(monitor,Mock(),[{'task_id':str(i)} for i in range(6)])
