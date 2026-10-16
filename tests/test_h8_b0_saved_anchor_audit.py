"""Artificial metadata tampering, Git policy and owned thread-failure receipts."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src/trotterlib'))
import h8_b0_anchor_contract as contract
import h8_b0_anchor_runtime as runtime
import h8_b0_saved_anchor_audit as audit
import launch_h8_strengthened_phase_a as observer


def fixture():
    p=contract.original.proposal(deepcopy(contract.plan()['binding']))
    rows=[];controls=[];workers=[]
    for i,case in enumerate(p['cases']):
        prep=p['binding']['prefix_preparations'][str(case['prefix'])]
        fingerprints={a:hashlib.sha256((case['id']+a).encode()).hexdigest() for a in ('cosine','sine')}
        controls.append(dict(case=case,SOURCE=audit.SOURCE,profile_sha256=audit.PROFILE,max_control_error=1e-12,
            preparation_binding=prep,basis_observed_from_restored_live_operations=True,
            observed_ordered_block_description_sha256=prep['ordered_block_description_sha256'],
            wrapper_fingerprints=fingerprints,scalar_policy=p['scalar_policy']))
        for axis in ('cosine','sine'):
            identity=[100+len(rows),1,1,1]
            rows.append(dict(case_id=case['id'],axis=axis,role=case['role'],SOURCE=audit.SOURCE,
                scope=p['scope'],profile_sha256=audit.PROFILE,compiler_hash=audit.COMPILER,
                control=p['control'],scalar_policy=p['scalar_policy'],preparation_binding=prep,
                max_control_error=1e-12,fingerprint_before=fingerprints[axis],fingerprint_after=fingerprints[axis],
                measured_RZ=100,CX=200,depth=100,size=1000,native_exit_code=0,reaped=True,worker_identity=identity))
            workers.append(dict(task_id=case['id']+'_'+axis,reaped=True,exit_code=0,identity=identity))
    settled={r['case_id']+'_'+r['axis']:{'compile':1} for r in rows}
    b=dict(driver=dict(status='STOP',SOURCE=audit.SOURCE,scope=p['scope'],error='DRIVER_STOP_SIGNAL_OR_DEADLINE',
        unknown_reserved_compile_calls=0,compile_entry_markers=8,irrevocable_reservations=settled,
        settlements=settled,completed_calls_lower_bound=deepcopy(p['call_caps']),retry=False,next_stage_authorized=False),
        launcher=dict(status='STOP',first_stop='RuntimeError:PROCESS_THREADS',all_reaped=True,cleanup_errors=[],native_exit_code=-9),
        native=dict(native_exit_code=1,wait_completed=True),admission=dict(status='STOP'),final_gate_exists=False,
        train=dict(role='TRAIN',axes=rows[:6],fit_performed=False),holdout=dict(role='HOLDOUT',axes=rows[6:],fit_performed=False),
        controls=controls,waves=[dict(status='PASS',all_reaped=True,cleanup_errors=[],automatic_retry=False,workers=workers[:4]),
            dict(status='PASS',all_reaped=True,cleanup_errors=[],automatic_retry=False,workers=workers[4:])],
        accepted=[dict(rows=rows[:4],all_reaped=True,next_stage_authorized=False),
                  dict(rows=rows[4:],all_reaped=True,next_stage_authorized=False)])
    return p,b


class SavedMetadata(unittest.TestCase):
    def test_valid_saved_set_does_not_promote_STOP_or_authorize_B1(self):
        p,b=fixture();r=audit.validate_records(p,b)
        self.assertEqual(r['measured_axes_verified'],8);self.assertEqual(r['original_acquisition_status'],'STOP')
        self.assertFalse(r['B1_authorized']);self.assertFalse(r['historical_thread_compliance_certified'])
        self.assertEqual(r['new_compile'],0)
    def test_missing_duplicate_or_wrong_split_rejected(self):
        for mutate in (lambda b:b['train']['axes'].pop(),
                       lambda b:b['train']['axes'].__setitem__(0,deepcopy(b['train']['axes'][1])),
                       lambda b:b['holdout']['axes'][0].update(role='TRAIN'),
                       lambda b:b['holdout'].update(fit_performed=True)):
            p,b=fixture();mutate(b)
            with self.assertRaises(ValueError):audit.validate_records(p,b)
    def test_profile_SOURCE_control_scalar_preparation_and_fingerprint_fail_closed(self):
        for key,value in [('profile_sha256','bad'),('SOURCE','bad'),('compiler_hash','bad'),
                ('control','ordinary'),('scalar_policy','tail'),('preparation_binding',{}),
                ('fingerprint_after','bad'),('measured_RZ',0),('measured_RZ',100.)]:
            p,b=fixture();b['train']['axes'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):audit.validate_records(p,b)
    def test_control_boundary_nonfinite_and_missing_live_basis_rejected(self):
        for value in (1e-8,float('nan'),float('inf'),-1e-12):
            p,b=fixture();b['train']['axes'][0]['max_control_error']=value
            with self.assertRaises(ValueError):audit.validate_records(p,b)
        p,b=fixture();b['controls'][0]['basis_observed_from_restored_live_operations']=False
        with self.assertRaises(ValueError):audit.validate_records(p,b)
        p,b=fixture();b['controls'][0]['scalar_policy']='tail'
        with self.assertRaises(ValueError):audit.validate_records(p,b)
    def test_failed_worker_unsettled_and_forbidden_calls_rejected(self):
        for mutate in (lambda b:b['waves'][0]['workers'][0].update(exit_code=1),
                lambda b:b['waves'][0].update(all_reaped=False),
                lambda b:b['driver'].update(unknown_reserved_compile_calls=1),
                lambda b:b['driver']['completed_calls_lower_bound'].update(cost_fit=1),
                lambda b:b['driver'].update(next_stage_authorized=True)):
            p,b=fixture();mutate(b)
            with self.assertRaises(ValueError):audit.validate_records(p,b)
    def test_original_STOP_must_not_be_replaced_by_PASS(self):
        for key in ('driver','launcher','admission'):
            p,b=fixture();b[key]['status']='COMPLETE_MANDATORY_STOP'
            with self.assertRaises(ValueError):audit.validate_records(p,b)
        p,b=fixture();b['final_gate_exists']=True
        with self.assertRaises(ValueError):audit.validate_records(p,b)
    def test_unsafe_public_relative_names_rejected(self):
        for name in ('/tmp/a','../a','safe/../../a'):
            with self.assertRaises(ValueError):audit.relative_path(name)
    def test_cli_default_imports_no_scientific_libraries(self):
        code='import runpy,sys;sys.argv=["audit_h8_b0_saved_anchors.py"];runpy.run_path("scripts/audit_h8_b0_saved_anchors.py",run_name="__main__");assert not any(x in sys.modules for x in ("numpy","qiskit","scipy"))'
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(json.loads(r.stdout)['runtime_authorization'])


class AdministrativeGit(unittest.TestCase):
    def test_all_calls_are_read_only_and_single_thread_without_global_config_mutation(self):
        with patch.object(audit.subprocess,'check_output',return_value=b'ok') as child:
            self.assertEqual(audit.git_bytes('/tmp','status','--porcelain=v1','-z'),b'ok')
        command=child.call_args.args[0]
        self.assertEqual(command,['git',*audit.GIT_OPTIONS,'status','--porcelain=v1','-z'])
        self.assertIn('index.threads=1',command);self.assertIn('core.preloadIndex=false',command)
        self.assertIn('--no-optional-locks',command)
    def test_runtime_preservation_uses_the_same_git_policy(self):
        empty=hashlib.sha256(b'').hexdigest()
        with patch.object(runtime,'git_bytes',side_effect=[b'abc\n',b'']) as git:
            runtime.protected(dict(protected_files=[],protected_worktrees=[dict(path='/tmp',HEAD='abc',status_sha256=empty)]))
        self.assertEqual(git.call_count,2)
    def test_worktree_status_matches_and_index_config_bytes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);subprocess.check_call(['git','init','-q',str(path)])
            (path/'tracked').write_text('test\n');subprocess.check_call(['git','add','tracked'],cwd=path)
            (path/'untracked').write_text('keep\n')
            index=(path/'.git/index').read_bytes();config=(path/'.git/config').read_bytes()
            ordinary=subprocess.check_output(['git','--no-optional-locks','status','--porcelain=v1','-z'],cwd=path)
            self.assertEqual(audit.git_bytes(path,'status','--porcelain=v1','-z'),ordinary)
            self.assertEqual((path/'.git/index').read_bytes(),index);self.assertEqual((path/'.git/config').read_bytes(),config)
    def test_resource_caps_and_science_candidates_unchanged(self):
        p=contract.plan();original=contract.original.proposal(p['binding'])
        self.assertEqual(p['cases'],original['cases']);self.assertEqual(p['call_caps'],original['call_caps'])
        self.assertTrue(all(p['resources'][k]==v for k,v in original['resources'].items()))
        self.assertFalse(p['administrative_git']['process_thread_exemption'])
        self.assertEqual(p['thread_violation_receipt']['threshold'],1)


class ThreadEvidence(unittest.TestCase):
    def test_driver_and_direct_child_violations_keep_exact_identity_and_count(self):
        base=dict(identity=[10,20,30,40],zombie=False,RSS_bytes=1,AS_bytes=1,threads=1,executable='/usr/bin/git')
        limits=contract.plan()['resources']
        for position in (0,1):
            rows=[dict(base),dict(base)] if position else [dict(base)]
            rows[position]['threads']=3
            with self.assertRaises(observer.ProcessThreadViolation) as caught:observer.enforce_processes(rows,limits,0)
            receipt=caught.exception.receipt
            self.assertEqual(str(caught.exception),'PROCESS_THREADS');self.assertEqual(receipt['offending_process']['threads'],3)
            self.assertEqual(receipt['offending_process']['identity'],[10,20,30,40])
            self.assertEqual(receipt['offending_process']['executable'],'/usr/bin/git')
            self.assertFalse(receipt['scientific_or_administrative_exemption'])
    def test_real_multithreaded_artificial_driver_STOP_receipt_and_cleanup(self):
        code='''
import json,os,sys
from pathlib import Path
sys.path.insert(0,'scripts')
import launch_h8_strengthened_phase_a as obs
import h8_b0_anchor_contract as c
root=Path(sys.argv[1]);output=root/'output';output.mkdir()
limits=c.plan()['resources'];limits.update(wall_seconds=10,observer_CPU_seconds=10,minimum_free_inodes=1,aggregate_RSS_bytes=2**30,admission_reserve_bytes=1)
child='import threading,time;threading.Thread(target=time.sleep,args=(10,)).start();time.sleep(10)'
try:obs.supervise([sys.executable,'-B','-c',child],env=dict(os.environ),control=root/'control',output=output,limits=limits,allowed_cpus=list(os.sched_getaffinity(0)),probe=lambda:dict(effective_available_bytes=2**50,events={},pressure={}))
except RuntimeError as error:assert str(error)=='RuntimeError:PROCESS_THREADS',str(error)
else:raise AssertionError('multithreaded driver admitted')
terminal=json.loads((root/'control/launcher_terminal.json').read_text())
receipt=json.loads((root/'control/process_thread_violation.json').read_text())
started=json.loads((root/'control/started.json').read_text())
assert receipt==terminal['process_thread_violation']
assert receipt['offending_process']['identity']==started['driver']['identity']
assert receipt['offending_process']['threads']==2 and receipt['offending_process']['executable']
assert terminal['status']=='STOP' and terminal['all_reaped'] and not terminal['cleanup_errors']
assert terminal['automatic_retry'] is False and terminal['next_stage_authorized'] is False
'''
        with tempfile.TemporaryDirectory() as td:
            result=subprocess.run([sys.executable,'-B','-c',code,td],cwd=ROOT,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
