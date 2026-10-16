"""Synthetic H8 adapter/grant/export/worker checks; no real scientific data."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h8_strengthened_contract as c
import h8_strengthened_export as e
import h8_strengthened_runtime as r
import h8_strengthened_signal as adapter
import launch_h8_strengthened_phase_a as observer
import track_a_phase_a_parallel as pool


def fixture(n=2,rank=2):
    blocks=[]
    for i in range(rank+1):
        ops=[dict(name='rz',qubits=[0],parameters=['0.0'],matrix_sha256='0'*64)]
        h=c.digest(dict(num_system_qubits=n,operations=ops,hash_policy='ordered_local_basis_operations_v1'))
        blocks.append(dict(block_id='df-one-body-correction' if i==0 else 'df-fragment-'+str(i-1),
            basis_id='df-basis-'+h,basis_hash=h,operations=ops,diagonal_hex=[float(i).hex()]*n,
            lambda_hex=None if i==0 else float(i).hex()))
    root=dict(blocks=blocks,components=[],bases={},constant_hex=float(.5).hex(),
              extracted_hex=0..hex(),lambda_r_hex=0..hex(),preparation_hash='full')
    return {'faithful:p2':root,'discard:p1':dict(deepcopy(root),blocks=deepcopy(blocks[:2]),preparation_hash='old')}


def config_grant():
    p=c.plan();base='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-only'
    config=dict(SOURCE='source',execution_sealed=True,plan_sha256=c.digest(p),
                output=base+'/output',control=base+'/control',lock=base+'/lock')
    grant=dict(schema='H8_STRENGTHENED_PHASE_A_GRANT_V1',approved_by_user=True,
        runtime_authorization=True,approval_text='synthetic test only',scope=c.SCOPE,SOURCE='source',
        config_sha256='config',plan_sha256=c.digest(p),resources=p['resources'],call_caps=p['call_caps'],
        driver_cpu=0,monitor_cpu=1,worker_cpus=list(range(2,18)),allowed_cpus=list(range(18)),
        one_shot=True,retry=False,resume=False,next_stage_authorized=False)
    fresh=dict(monotonic_seconds=100,source='source',config_sha256='config',grant_sha256=c.digest(grant),
               allowed_cpus=list(range(18)))
    for key in ('resources_idle_verified','profile_verified','protected_jobs_unchanged',
                'external_RSS_monitor_ready','physical_cores_verified','quota_verified','source_input_bytes_verified'):
        fresh[key]=True
    return config,grant,fresh


class Export(unittest.TestCase):
    def setUp(self):self.export=fixture()
    def validate(self):return e.validate_export(self.export,n=2,rank=2,old_prefixes=(1,))
    def test_old_prefix_lineage(self):self.assertEqual(self.validate()['status'],'PASS_METADATA_ONLY')
    def test_wrong_prefix_basis(self):
        self.export['discard:p1']['blocks'][1]['lambda_hex']=3..hex()
        with self.assertRaisesRegex(ValueError,'LINEAGE'):self.validate()
    def test_reordered_fragment(self):
        self.export['faithful:p2']['blocks'][1]['block_id']='df-fragment-1'
        with self.assertRaisesRegex(ValueError,'ORDER'):self.validate()
    def test_matrix_claim_or_angle_tamper(self):
        self.export['faithful:p2']['blocks'][1]['operations'][0]['parameters']=['1.0']
        with self.assertRaisesRegex(ValueError,'HASH'):self.validate()
    def test_nonfinite_hex(self):
        self.export['faithful:p2']['blocks'][1]['lambda_hex']='nan'
        with self.assertRaisesRegex(ValueError,'NONFINITE'):self.validate()
    def test_noncanonical_hex(self):
        with self.assertRaises(ValueError):e.real_hex('0x0p+0')
    def test_nonadjacent_gate(self):
        b=self.export['faithful:p2']['blocks'][1];b['operations'][0].update(name='xx_plus_yy',qubits=[0,2],parameters=['0.0','0.0'])
        with self.assertRaisesRegex(ValueError,'OPERATION'):self.validate()
    def test_unknown_gate(self):
        self.export['faithful:p2']['blocks'][1]['operations'][0]['name']='cx'
        with self.assertRaises(ValueError):self.validate()
    def test_restore_denied_before_any_factory(self):
        factory=Mock();permit=NS(require=Mock(side_effect=ValueError('DENIED')))
        with self.assertRaisesRegex(ValueError,'DENIED'):
            e.restore_blocks(self.export,permit,gate_factory=factory,inspect_operations=Mock(),
                             one_type=NS,fragment_type=NS,n=2,rank=2)
        factory.assert_not_called()
    def restore(self,inspect=None):
        permit=NS(require=Mock(),take=Mock())
        def describe(ops):
            return (),deepcopy(self.export['faithful:p2']['blocks'][0]['operations'])
        return e.restore_blocks(self.export,permit,gate_factory=lambda name,params:NS(name=name,params=params),
                    inspect_operations=describe if inspect is None else inspect,
                    one_type=NS,fragment_type=NS,n=2,rank=2)
    def test_restore_original_order_and_rank(self):
        blocks=self.restore();self.assertEqual([b.order_index for b in blocks],[0,1,2])
        self.assertEqual(blocks[2].rank,1);self.assertEqual(blocks[2].lam,2.)
    def test_restore_matrix_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError,'RUNTIME_BASIS'):
            self.restore(lambda ops:((),[]))
    def test_prefix_view_distinct_identity_shared_blocks(self):
        blocks=self.restore();v=e.prefix_view(blocks,self.export['faithful:p2'],1,export_sha256='export')
        self.assertIs(v.deterministic_blocks[1],blocks[1]);self.assertNotIn(v.preparation_hash,('full','old'))
        self.assertEqual(v.constant_coefficient,.5);self.assertEqual(v.extracted_identity_coefficient,0)


class Contracts(unittest.TestCase):
    def test_grid_and_schedule_counts(self):
        p=c.plan();self.assertEqual(len(c.cells()),76)
        self.assertEqual((len(p['primitive_tasks']),len(p['signal_tasks'])),(22,13))
        self.assertEqual(p['required_primitive_probe_actions'],1650)
        self.assertEqual(sum(t['counts']['reference_matvec'] for wave in c.tasks() for t in wave),48000000)
        self.assertLessEqual(max(len(t['times']) for t in p['primitive_tasks']),32)
    def test_schedule_exact_match_frozen_iterator(self):
        # Same fixed schedule algorithm, loaded as stdlib modules only.
        pf=runpy.run_path(str(ROOT/'src/trotterlib/pf_decomposition.py'))['iter_pf_steps']
        for cell in c.cells():
            w=1/(2-2**(1/3));weights=[1.] if cell['formula']=='2nd' else [1-2*w,w]
            expected=[(i,.8/cell['q']*f) for i,f in pf(cell['prefix']+1,weights)]
            self.assertEqual(c.schedule(cell)['ordinary_one_outer_step'],expected)
    def test_negative_and_directional_times_present(self):
        for cell in c.cells():
            p=c.schedule(cell);times=set(p['unique_primitive_times'])
            for i,t,mode in p['directional_one_outer_step']:
                self.assertIn((i,t),times)
                if mode=='DIRECTIONAL':self.assertIn((i,-t),times)
            self.assertTrue(any(t<0 for i,t in times))
    def test_grant_accepts_only_synthetic_valid_fixture(self):
        cfg,g,f=config_grant()
        self.assertTrue(r.verify_grant(cfg,g,f,config_sha256='config',source='source',now=110,affinity=set(range(18))))
    def test_grant_rejections(self):
        for edit in (lambda g:g.update(approved_by_user=False),lambda g:g.update(scope='compile'),
                     lambda g:g.update(retry=True),lambda g:g.update(next_stage_authorized=True),
                     lambda g:g['call_caps'].update(compile=1),lambda g:g.update(worker_cpus=[2]*16)):
            cfg,g,f=config_grant();edit(g)
            with self.subTest(grant=g),self.assertRaises(ValueError):
                r.verify_grant(cfg,g,f,config_sha256='config',source='source',now=110,affinity=set(range(18)))
    def test_fresh_and_unsealed_rejections(self):
        for edit in (lambda cfg,f:f.update(monotonic_seconds=50),lambda cfg,f:f.update(monotonic_seconds=float('nan')),
                     lambda cfg,f:f.update(quota_verified=False),lambda cfg,f:cfg.update(execution_sealed=False),
                     lambda cfg,f:cfg.update(SOURCE=None),lambda cfg,f:cfg.update(output='/tmp/escape')):
            cfg,g,f=config_grant();edit(cfg,f)
            with self.subTest(fresh=f),self.assertRaises(ValueError):
                r.verify_grant(cfg,g,f,config_sha256='config',source='source',now=110,affinity=set(range(18)))
    def test_json_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x'
            for value in ('{"x":1,"x":2}','{"x":NaN}'):
                p.write_text(value)
                with self.assertRaises(ValueError):r.read(p)
    def test_irrevocable_reservations_and_settlement(self):
        cfg,g,f=config_grant()
        with tempfile.TemporaryDirectory() as td:
            p=r.Permit(cfg,g,td);task=dict(id='a',counts=dict(primitive=3,reference_matvec=10))
            p.reserve(task);p.settle('a',dict(primitive=3,reference_matvec=4))
            self.assertEqual(p.reserved['a']['reference_matvec'],10)
            with self.assertRaises(ValueError):p.reserve(task)
            with self.assertRaises(ValueError):p.settle('a',dict(primitive=3))
    def test_no_science_import_before_permit(self):
        with patch('h8_strengthened_signal.importlib.import_module') as imported:
            with self.assertRaisesRegex(ValueError,'DENIED'):
                adapter.evaluate({},NS(require=Mock(side_effect=ValueError('DENIED'))),None)
            imported.assert_not_called()
    def test_saved_reference_unrotated_and_error_binding(self):
        receipt=dict(metadata=dict(hamiltonian_hash='h',state_hash='s'),state_renormalized=False,
            numerical_allowance_certified=False,reference=dict(real=.3,imag=-.5),
            production_CSR_probe_errors=[1e-13]*3,reference_agreement=dict(signal_difference=2e-13))
        z,error=adapter.saved_reference(receipt,dict(hamiltonian='h',state='s'))
        self.assertEqual(z,.3-.5j);self.assertAlmostEqual(error,3e-13,delta=1e-28)
        receipt['state_renormalized']=True
        with self.assertRaises(ValueError):adapter.saved_reference(receipt,dict(hamiltonian='h',state='s'))
    def test_p21_scalar_reused_without_new_take(self):
        permit=NS(take=Mock());saved=dict(full_Fock_identity=True,new_scientific_certificate=False,
            exact_stored_scalar_numerator='1',exact_stored_scalar_denominator='2',C=.5,scalar_error_upper=0.)
        C,err,receipt=adapter.checked_scalar(saved,21,None,None,permit)
        self.assertEqual((C,err),(.5,0.));permit.take.assert_not_called()
    def test_driver_worker_different_memory_limits(self):
        limits=c.plan()['resources'];base=dict(zombie=False,RSS_bytes=20*2**30,AS_bytes=32*2**30,threads=1)
        observer.enforce_processes([base],limits,0)
        with self.assertRaisesRegex(RuntimeError,'worker'):
            observer.enforce_processes([base,dict(base)],limits,0)
    def test_pressure_and_OOM_fail_closed(self):
        base=dict(effective_available_bytes=2**50,events={'g':dict(oom='0')},pressure={'host':'full avg10=0.00 avg60=0.00 avg300=0.00 total=0'})
        for change in (lambda x:x['events']['g'].update(oom='1'),
                       lambda x:x['pressure'].update(host='full avg10=1.00 avg60=0.00 avg300=0.00 total=0'),
                       lambda x:x['events'].clear()):
            value=deepcopy(base);change(value);self.assertIsNotNone(observer.resource_reason(value,base,1))
    def test_metadata_cli_does_not_import_science(self):
        code='import runpy,sys;sys.argv=["run_h8_strengthened_phase_a.py"];runpy.run_path("scripts/run_h8_strengthened_phase_a.py",run_name="__main__");assert not any(x in sys.modules for x in ("numpy","qiskit","scipy"))'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,
                              env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts')),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['runtime_authorization'])


class Workers(unittest.TestCase):
    def test_observer_success_and_stop_reap_only_synthetic_child(self):
        with tempfile.TemporaryDirectory() as td:
            # Separate executable confines observer RLIMIT/prctl changes.
            code='''
import sys,json,os
from pathlib import Path
sys.path.insert(0,'scripts')
import launch_h8_strengthened_phase_a as obs
import h8_strengthened_contract as contract
root=Path(sys.argv[1]);success=sys.argv[2]=='1';output=root/'output';output.mkdir()
limits=contract.plan()['resources'];limits.update(wall_seconds=5,observer_CPU_seconds=10,
    minimum_free_inodes=1,aggregate_RSS_bytes=2**30,admission_reserve_bytes=1)
if not success:limits['observer_RSS_bytes']=1
child='import json,sys,time;from pathlib import Path;time.sleep(.2);(Path(sys.argv[1])/"terminal.json").write_text(json.dumps(dict(status="COMPLETE_MANDATORY_STOP")))'
try:
    obs.supervise([sys.executable,'-B','-c',child,str(output)],env=dict(os.environ),
        control=root/'control',output=output,limits=limits,allowed_cpus=list(os.sched_getaffinity(0)),
        probe=lambda:dict(effective_available_bytes=2**50,events={},pressure={}))
except RuntimeError:
    if success:raise
receipt=json.loads((root/'control/launcher_terminal.json').read_text())
assert receipt['all_reaped'] and not receipt['cleanup_errors']
assert (receipt['status']=='COMPLETE_MANDATORY_STOP')==success
'''
            for success in (True,False):
                directory=Path(td)/str(success);directory.mkdir()
                completed=subprocess.run([sys.executable,'-B','-c',code,str(directory),'1' if success else '0'],
                    cwd=ROOT,capture_output=True,text=True,timeout=20)
                self.assertEqual(completed.returncode,0,completed.stderr)

    def test_real_owned_success_and_failure_cleanup_with_artificial_callbacks(self):
        with tempfile.TemporaryDirectory() as td:
            cpu=min(os.sched_getaffinity(0));limits=dict(max_workers=1,AS_bytes=2**30,RSS_bytes=2**30,
                worker_CPU_seconds=10,aggregate_CPU_seconds=20)
            for success in (True,False):
                output=Path(td)/str(success)
                def callback(task,directory):
                    if not success:raise ValueError('artificial failure')
                    return dict(ok=True)
                def execute():return pool.map_owned([dict(id='one')],callback,cpus=[cpu],limits=limits,
                    directory=output,deadline=time.monotonic()+10)
                if success:self.assertTrue(execute()['all_reaped'])
                else:
                    with self.assertRaises(RuntimeError):execute()
                receipt=json.loads((output/'parallel_receipt.json').read_text())
                self.assertTrue(receipt['all_reaped']);self.assertFalse(receipt['automatic_retry'])


if __name__=='__main__':unittest.main()
