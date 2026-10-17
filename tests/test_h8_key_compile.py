"""Artificial costs, native API doubles and tiny owned-process cleanup only."""
from copy import deepcopy
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_key_compile_contract as c
import h8_key_compile_adapter as a
import h8_key_compile_controls as controls
import h8_key_compile_parallel as pool
import run_h8_key_compile as run
TEST_ROOT=Path(os.environ['H8_KEY_TEST_ROOT'])


def temp():return tempfile.TemporaryDirectory(dir=TEST_ROOT)


def authorization():
    cpus=[0,1,2]
    cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(c.plan()))
    g=dict(schema='H8_KEY_COMPILE_GRANT_V1',approved=True,runtime_authorization=True,execution_sealed=True,
        approval_text='ARTIFICIAL_ONLY',SOURCE=cfg['SOURCE'],scope=c.SCOPE,plan_sha256=cfg['plan_sha256'],
        config_sha256=c.digest(cfg),call_caps=c.CAPS,one_shot=True,automatic_retry=False,
        cost_reaccount_authorized=False,next_stage_authorized=False,driver_cpu=0,worker_cpus=[1,2],allowed_cpus=cpus)
    return cfg,g


class Contract(unittest.TestCase):
    def test_fixed_priority_axes_and_no_execution_authority(self):
        p=c.plan();self.assertEqual([x['id'] for x in p['cases']],['B0I_p15_q4_4th','B0I_p11_q1_2nd','B0_p15_q4_2nd'])
        self.assertEqual([x['axis'] for x in p['axes']],['cosine','sine']*3)
        self.assertFalse(p['approved']);self.assertFalse(p['runtime_authorization'])
        self.assertIsNone(p['absolute_launch_command']);self.assertEqual(p['allowed_cpus'],[])
        self.assertEqual((p['compile_concurrency'],p['preparation_concurrency']),(1,2))
        self.assertIsNone(p['resource_policy']['agent_imposed_CPU_RAM_CPUtime_wall_caps'])
    def test_denominator_boundary_and_invalid_counts(self):
        self.assertFalse(c.axis_check('102',100)['PASS'])
        self.assertTrue(c.axis_check('10199/100',100)['PASS'])
        self.assertTrue(c.axis_check('98',101)['PASS']is False)
        for invalid in (0,-1,1.0,True,float('nan')):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):c.axis_check('100',invalid)
    def test_explicit_new_grant_and_each_binding(self):
        cfg,g=authorization();self.assertEqual(c.authorize(cfg,g,affinity={0,1,2}),[0,1,2])
        for key,value in [('approved',False),('runtime_authorization',False),('execution_sealed',False),
                ('SOURCE','b'*40),('scope','OLD_SCOPE'),('config_sha256','bad'),('plan_sha256','bad'),
                ('call_caps',{}),('one_shot',False),('automatic_retry',True),('cost_reaccount_authorized',True),
                ('next_stage_authorized',True),('allowed_cpus',[0,1,1]),('worker_cpus',[1])]:
            bad=deepcopy(g);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.authorize(cfg,bad,affinity={0,1,2})
    def test_invalid_grant_rejected_before_side_effects(self):
        with patch.object(c,'read',side_effect=[{},{}]),patch.object(run,'fresh') as fresh,patch.object(pool,'map_owned') as launch:
            with self.assertRaises(ValueError):run.launch('artificial-config','artificial-grant')
            fresh.assert_not_called();launch.assert_not_called()
    def test_no_science_import_before_rejected_adapter_authority(self):
        with patch.object(a.importlib,'import_module') as imported:
            with self.assertRaises((ValueError,KeyError)):a.evaluate({}, {}, 'unused')
            imported.assert_not_called()
    def test_default_cli_no_science(self):
        code='import runpy,sys;sys.argv=["plan"];runpy.run_path("scripts/run_h8_key_compile.py",run_name="__main__");assert not any(k in sys.modules for k in ("numpy","scipy","qiskit"))'
        result=subprocess.run([sys.executable,'-P','-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_nonrefundable_counts_and_forbidden_calls(self):
        with temp() as d:
            p=c.Permit({}, {}, d,synthetic=True)
            for _ in range(6):p.take('compile')
            for key in ('compile','fit','cost_reaccount','new_signal','tail_scalar','extra_compile'):
                with self.subTest(key=key),self.assertRaises(ValueError):p.take(key)
            self.assertEqual(len(list(Path(d).glob('reservation_compile_*.json'))),6)
    def test_duplicate_nonfinite_JSON_rejected(self):
        for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'):
            with temp() as d:
                p=Path(d)/'data.json';p.write_text(text)
                with self.assertRaises(ValueError):c.read(p)


class ScalarsAndSpools(unittest.TestCase):
    def test_only_own_saved_scalar_original_once(self):
        case=dict(id='artificial_p11',method='B0I')
        target=dict(case_id=case['id'],scalar_receipt=dict(policy='EXACT_STORED_BINARY64_TAIL_IDENTITY',
            C=.125,numerator='1',denominator='8',input_operator_error_certificate=False))
        self.assertEqual(a.scalar_for(case,(2.).hex(),target),2.125)
        target['case_id']='artificial_p15'
        with self.assertRaises(ValueError):a.scalar_for(case,(2.).hex(),target)
    def test_scalar_tamper_and_B0_no_tail(self):
        case=dict(id='artificial',method='B0');target=dict(case_id='artificial',scalar_receipt=None)
        self.assertEqual(a.scalar_for(case,(2.).hex(),target),2.)
        target['scalar_receipt']={}
        with self.assertRaises(ValueError):a.scalar_for(case,(2.).hex(),target)
        case['method']='B0I';target['scalar_receipt']=dict(policy='EXACT_STORED_BINARY64_TAIL_IDENTITY',C=.25,
            numerator='1',denominator='8',input_operator_error_certificate=False)
        with self.assertRaises(ValueError):a.scalar_for(case,(2.).hex(),target)
    def test_qpy_byte_identity_path_and_single_circuit(self):
        with temp() as d:
            p=Path(d)/'wrapper.qpy';p.write_bytes(b'ARTIFICIAL')
            receipt=dict(path=str(p),bytes=p.stat().st_size,sha256=c.sha(p));qpy=Mock();qpy.load.return_value=['one']
            self.assertEqual(controls.load_wrapper(receipt,d,qpy),'one')
            qpy.load.return_value=['one','two']
            with self.assertRaises(ValueError):controls.load_wrapper(receipt,d,qpy)
            qpy.load.return_value=['one'];p.write_bytes(b'CHANGED___')
            with self.assertRaises(ValueError):controls.load_wrapper(receipt,d,qpy)
            with self.assertRaises(ValueError):controls.load_wrapper(receipt,str(Path(d)/'other'),qpy)


class BindingAndCompletion(unittest.TestCase):
    def fixture(self,root):
        source=root/'artificial.py';source.write_text('ARTIFICIAL=1\n')
        data=root/'input.json';data.write_text('{"artificial":true}\n')
        spec=dict(logical_name='artificial',bytes=data.stat().st_size,sha256=c.sha(data))
        cfg=dict(schema='H8_KEY_COMPILE_CONFIG_V1',plan_sha256=c.digest({'artificial':True}),
            provider_source=c.PROVIDER,expected_profile_digest=c.PROFILE,provider_root=str(root),
            provider_hashes={},runtime_file_hashes={},python=c.PYTHON,python_sha256=c.sha(c.PYTHON),
            source_hashes={'artificial.py':c.sha(source)},files=[dict(spec,path=str(data))],input_mapping={'artificial':str(data)})
        return cfg,spec
    def test_own_dependency_drift_missing_and_unrelated_progress(self):
        for change in ('none','unrelated','source','input','missing','profile','provider_map','runtime_map'):
            with self.subTest(change=change),temp() as d:
                root=Path(d);cfg,spec=self.fixture(root)
                if change=='source':(root/'artificial.py').write_text('CHANGED\n')
                if change=='input':(root/'input.json').write_text('CHANGED\n')
                if change=='missing':(root/'input.json').unlink()
                if change=='profile':cfg['expected_profile_digest']='bad'
                if change=='provider_map':cfg['provider_hashes']={'unexpected':'bad'}
                if change=='runtime_map':cfg['runtime_file_hashes']={'unexpected':'bad'}
                if change=='unrelated':(root/'unrelated-not-used.txt').write_text('ANOTHER_CHAT_PROGRESS')
                with patch.object(c,'ROOT',root),patch.object(c,'plan',return_value={'artificial':True}),\
                     patch.object(c,'binding',return_value={'inputs':[spec]}),\
                     patch.object(c,'source_paths',return_value=['artificial.py']),\
                     patch.object(c,'PROVIDER_MAP',c.digest({})),patch.object(c,'RUNTIME_MAP',c.digest({})):
                    if change in ('none','unrelated'):self.assertEqual(c.verify(cfg,git=False)['status'],'PASS')
                    else:
                        with self.assertRaises((ValueError,FileNotFoundError)):c.verify(cfg,git=False)
    def test_completion_requires_every_axis_source_and_native_evidence(self):
        rows=[];binding=c.binding()
        for axis in binding['axes']:
            case=next(x for x in binding['cases'] if x['id']==axis['case_id']);rz=round(Fraction(axis['exact_prediction']))
            rows.append(dict(SOURCE='a'*40,case_id=case['id'],axis=axis['axis'],native_exit_code=0,reaped=True,
                preparation_binding=binding['prefix_preparations'][str(case['prefix'])],compiler_hash=c.COMPILER,
                profile_sha256=c.PROFILE,model_sha256=c.MODEL,fingerprint_before='f'*64,fingerprint_after='f'*64,
                max_control_error=0.,model_check=c.axis_check(axis['exact_prediction'],rz),measured_RZ=rz))
        self.assertEqual(c.complete(rows,c.CAPS,'a'*40)['status'],'PASS_ONLY_SIX_REGISTERED_AXES_MANDATORY_STOP')
        for key,value in [('SOURCE','b'*40),('compiler_hash','bad'),('reaped',False),('preparation_binding',{}),
                          ('max_control_error',1e-8),('fingerprint_after','changed')]:
            bad=deepcopy(rows);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.complete(bad,c.CAPS,'a'*40)
        with self.assertRaises(ValueError):c.complete(rows[:-1],c.CAPS,'a'*40)


class AxisAdmission(unittest.TestCase):
    def exercise(self, fail=None, bad_receipt=False):
        with temp() as d:
            permit=c.Permit({},dict(worker_cpus=[0,1]),d,synthetic=True);calls=[];builds=[]
            axes=c.binding()['axes'];ordinal=[0]
            def build(case):builds.append(case['id']);return {}
            def callback(task,built,path):
                i=ordinal[0];ordinal[0]+=1
                return dict(case_id=task['case_id'],axis=task['axis'],measured_RZ=1 if i==fail else round(Fraction(axes[i]['exact_prediction'])))
            def fake(tasks,fn,*,directory,**kwargs):
                self.assertEqual(len(tasks),1);self.assertEqual(len(kwargs['cpus']),1)
                directory.mkdir();task=tasks[0];calls.append(task['id']);path=directory/'task_result.json';c.save(path,fn(task,directory))
                return dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=[dict(task_id=task['id'],directory=str(directory),
                    exit_code=1 if bad_receipt else 0,reaped=True,result_sha256=c.sha(path),identity=[1,2,3,4],peak_RSS_bytes=1,cpu_seconds=.1)])
            if fail is not None or bad_receipt:
                with self.assertRaises(ValueError):a.serial_axes(permit,build,callback,map_owned=fake)
                self.assertEqual(len(calls),1 if bad_receipt else fail+1)
                if fail is not None:
                    row=c.read(Path(d)/('axis_'+str(fail+1).zfill(2)+'_measured.json'))
                    self.assertFalse(row['model_check']['PASS'])
            else:
                rows=a.serial_axes(permit,build,callback,map_owned=fake);self.assertEqual(len(rows),6)
                self.assertEqual(builds,[case['id'] for case in c.binding()['cases']])
    def test_six_successful_serial_axes(self):self.exercise()
    def test_every_failure_position_stops_before_next_compile(self):
        for i in range(6):
            with self.subTest(i=i):self.exercise(fail=i)
    def test_bad_native_exit_not_accepted(self):self.exercise(bad_receipt=True)


class NativeControlDoubles(unittest.TestCase):
    def test_formula_control_parallel_and_small_spool(self):
        import numpy as np
        cpus=sorted(os.sched_getaffinity(0))[:2]
        if len(cpus)<2:self.skipTest('two CPUs unavailable')
        for formula in ('2nd','4th'):
            with temp() as d:
                native=Mock();native.build_deterministic_native.return_value=SimpleNamespace(circuit='evolution')
                def wrapper(evolution,*,axis,**kwargs):
                    return SimpleNamespace(axis=axis,remove_final_measurements=lambda inplace:axis)
                native.build_native_hadamard_wrapper.side_effect=wrapper
                m=SimpleNamespace(np=np,simulate_statevector=lambda circuit,state:state,
                    hadamard_expectation=lambda state,n:1.,canonical_qiskit_circuit_fingerprint=lambda w:'fingerprint-'+w.axis)
                # Both axes get the same artificial expected overlap (1+1j).
                # Instead use a real overlap 1+0j and a probe double encoding axis.
                m.simulate_statevector=lambda circuit,state:state if circuit=='evolution' else circuit
                m.hadamard_expectation=lambda state,n:1. if state=='cosine' else 0.
                qpy=SimpleNamespace(dump=lambda w,f:f.write(('ARTIFICIAL-'+w.axis).encode()))
                permit=c.Permit({},dict(worker_cpus=cpus),d,synthetic=True)
                case=dict(id='synthetic_'+formula,formula=formula,q=4)
                target={k:dict(real=1.,imag=0.) for k in ('corrected','independent_forward')}
                with patch.dict(sys.modules,{'qiskit':SimpleNamespace(qpy=qpy)}):
                    built=controls.build_case(case,(),0.,np.array([1.,0.]),target,permit,native,m,n=1,cap=99)
                self.assertEqual(set(built['spools']),{'cosine','sine'})
                self.assertEqual(native.build_deterministic_native.call_args.kwargs['formula'],formula)
                self.assertEqual(native.build_deterministic_native.call_args.kwargs['control_policy'],'symmetric_directional')
                self.assertEqual(permit.calls['control_probe'],4)
                self.assertEqual(permit.calls['qpy_dump'],2)


class OwnedProcesses(unittest.TestCase):
    def test_artificial_launcher_driver_success_and_original_failure_receipt(self):
        cpus=sorted(os.sched_getaffinity(0))[:3]
        if len(cpus)<3:self.skipTest('three CPUs unavailable')
        for failure in (False,True):
            with self.subTest(failure=failure),temp() as d:
                root=Path(d);cfg=dict(SOURCE='a'*40,python=sys.executable,font_seed=None,
                    output=str(root/'output'),control=str(root/'control'),lock=str(root/'lock'))
                g=dict(driver_cpu=cpus[0],worker_cpus=cpus[1:],allowed_cpus=cpus)
                c.save(root/'config.json',cfg);c.save(root/'grant.json',g)
                def driver(task,directory,config,grant):
                    if failure:
                        c.save(Path(cfg['output'])/'terminal.json',dict(status='STOP',first_stop='ARTIFICIAL_FIRST_REASON'))
                        raise RuntimeError('ARTIFICIAL_FIRST_REASON')
                    c.save(Path(cfg['output'])/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP'))
                    return dict(status='ARTIFICIAL_DRIVER_ONLY')
                with patch.object(c,'authorize',return_value=cpus),patch.object(run,'fresh',return_value={'monotonic_seconds':time.monotonic()}),\
                     patch.object(c,'paths',return_value=tuple(Path(cfg[k]) for k in ('output','control','lock'))),\
                     patch.object(c,'verify',return_value={'status':'ARTIFICIAL'}),patch.object(run,'driver',side_effect=driver),patch.dict(os.environ):
                    if failure:
                        with self.assertRaises(RuntimeError):run.launch(root/'config.json',root/'grant.json')
                        self.assertEqual(c.read(root/'control/STOP.json')['first_stop'],'ARTIFICIAL_FIRST_REASON')
                    else:
                        run.launch(root/'config.json',root/'grant.json')
                        self.assertEqual(c.read(root/'control/final_gate.json')['status'],'PASS_MANDATORY_STOP')
                native=c.read(root/'control/driver/parallel_receipt.json')
                self.assertTrue(native['all_reaped']);self.assertFalse(native['cleanup_errors'])
    def test_two_tiny_workers_success_failure_and_reap(self):
        cpus=sorted(os.sched_getaffinity(0))[:2]
        if len(cpus)<2:self.skipTest('two CPUs unavailable')
        for fail in (False,True):
            with temp() as d:
                root=Path(d)
                def callback(task,path):
                    (root/task['id']).touch();until=time.monotonic()+5
                    while not all((root/x).exists() for x in ('left','right')):
                        if time.monotonic()>until:raise ValueError('NO_OVERLAP')
                        time.sleep(.01)
                    if fail and task['id']=='left':raise ValueError('ARTIFICIAL_FAILURE')
                    return dict(both_started=True)
                def execute():return pool.map_owned([dict(id=x) for x in ('left','right')],callback,cpus=cpus,
                    limits=dict(max_workers=2),directory=root/'workers',deadline=math.inf)
                if fail:
                    with self.assertRaises(RuntimeError):execute()
                else:execute()
                receipt=c.read(root/'workers/parallel_receipt.json')
                self.assertTrue(receipt['all_reaped']);self.assertFalse(receipt['cleanup_errors'])
                self.assertTrue(all(w['reaped'] for w in receipt['workers']))
                if not fail:self.assertTrue(all(w.get('result_sha256') for w in receipt['workers']))
                for w in receipt['workers']:self.assertIsNone(pool.identity(w['pid']))
    def test_ESRCH_only_is_exit_race_and_FD_closed(self):
        h=dict(reaped=False,pid=123,identity=(123,1,2,3))
        with patch.object(pool,'identity',return_value=h['identity']),patch.object(os,'pidfd_open',return_value=99),patch.object(os,'close') as close:
            with patch.object(signal,'pidfd_send_signal',side_effect=ProcessLookupError()):pool.send_owned(h,signal.SIGTERM)
            close.assert_called_with(99)
            with patch.object(signal,'pidfd_send_signal',side_effect=PermissionError()):
                with self.assertRaises(PermissionError):pool.send_owned(h,signal.SIGTERM)
    def test_owner_mismatch_no_signal(self):
        h=dict(reaped=False,pid=123,identity=(123,1,2,3))
        with patch.object(pool,'identity',return_value=(123,2,2,3)),patch.object(signal,'pidfd_send_signal') as send:
            with self.assertRaises(RuntimeError):pool.send_owned(h,signal.SIGTERM)
            send.assert_not_called()


if __name__=='__main__':unittest.main()
