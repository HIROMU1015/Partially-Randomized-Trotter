"""Artificial concurrent preparation, delegated accounting and nested cleanup."""
from copy import deepcopy
from fractions import Fraction
import math
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_key_compile_contract as c
import h8_key_compile_preparation as prep
import h8_key_compile_adapter as adapter
import h8_key_compile_controls as controls
import h8_key_compile_parallel as pool
from test_h8_key_compile import temp,authorization


def artificial_permit(directory,cpus):
    return c.Permit(dict(SOURCE='a'*40),dict(worker_cpus=list(cpus)),directory,synthetic=True)


def wait_markers(root,n,seconds=5):
    until=time.monotonic()+seconds
    while len(list(root.glob('*.ready')))<n:
        if time.monotonic()>until:raise RuntimeError('ARTIFICIAL_CONCURRENCY_NOT_REACHED')
        time.sleep(.01)


class PreparationContract(unittest.TestCase):
    def test_plan_six_workers_same_science_counts_and_no_authority(self):
        p=c.plan();self.assertEqual((p['preparation_concurrency'],p['max_cases_live']),(6,3))
        self.assertEqual(p['preparation']['case_build_concurrency'],3)
        self.assertEqual(p['preparation']['compile_admission'],'ALL_THREE_PREPARATIONS_PASS')
        self.assertEqual(p['compile_concurrency'],1)
        self.assertEqual(c.CAPS['compile'],6);self.assertEqual(c.CAPS['control_probe'],12)
        for key,count in c.CASE_PREPARATION_CAPS.items():self.assertEqual(c.CAPS[key],3*count)
        self.assertFalse(p['approved']);self.assertFalse(p['runtime_authorization'])
        self.assertEqual(p['allowed_cpus'],[])
        self.assertIsNone(p['absolute_launch_command'])
        self.assertEqual(p['resource_policy']['estimates']['aggregate_peak_RSS_bytes'],192*2**30)

    def test_old_grant_and_six_duplicate_or_unavailable_cpus_rejected(self):
        cfg,g=authorization()
        for change in (dict(schema='H8_KEY_COMPILE_GRANT_V1'),dict(worker_cpus=[1,2],allowed_cpus=[0,1,2]),
                       dict(worker_cpus=[1,2,3,4,5,5],allowed_cpus=[0,1,2,3,4,5,5])):
            bad=deepcopy(g);bad.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):c.authorize(cfg,bad,affinity=set(range(7)))
        with self.assertRaises(ValueError):c.authorize(cfg,g,affinity=set(range(6)))

    def fake_map(self,*,mutate=None):
        def execute(tasks,callback,*,directory,reserve,**kwargs):
            self.assertEqual(kwargs['limits']['max_workers'],3)
            self.assertEqual(len(kwargs['cpus']),3)
            directory.mkdir();workers=[]
            for task in tasks:
                reserve(task);work=directory/task['id'];work.mkdir()
                if mutate:mutate(task)
                row=callback(task,work);path=work/'task_result.json';c.save(path,row)
                workers.append(dict(task_id=task['id'],directory=str(work),exit_code=0,reaped=True,
                                    result_sha256=c.sha(path)))
            result=dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=workers)
            c.save(directory/'parallel_receipt.json',result)
            return result
        return execute

    @staticmethod
    def consume(case,permit,cpus):
        for key,count in c.CASE_PREPARATION_CAPS.items():
            for _ in range(count):permit.take(key)
        return dict(artificial=True,worker_cpus=cpus)

    def test_parent_reservations_exact_once_before_child_use(self):
        with temp() as d:
            permit=artificial_permit(d,range(6))
            result=prep.prepare_cases(permit,self.consume,map_owned=self.fake_map())
            self.assertEqual(list(result),[case['id'] for case in c.binding()['cases']])
            for key,count in c.CASE_PREPARATION_CAPS.items():
                self.assertEqual(permit.calls[key],3*count)
                rows=list(Path(d).glob('reservation_'+key+'_*.json'))
                self.assertEqual(len(rows),3*count)
            self.assertEqual(permit.calls['compile'],0)
            self.assertEqual(c.read(Path(d)/'preparation_complete.json')['status'],'PASS_THREE_CASES_PREPARED')
            for row in result.values():self.assertEqual(row['preparation_call_counts'],c.CASE_PREPARATION_CAPS)

    def test_incomplete_child_duplicate_use_or_forbidden_compile_stops(self):
        for failure in ('missing','duplicate','compile'):
            with self.subTest(failure=failure),temp() as d:
                permit=artificial_permit(d,range(6))
                def build(case,child,cpus):
                    if failure=='missing':return {}
                    result=self.consume(case,child,cpus)
                    child.take('compile' if failure=='compile' else 'native_evolution_build')
                    return result
                with self.assertRaises(ValueError):prep.prepare_cases(permit,build,map_owned=self.fake_map())
                self.assertEqual(permit.calls['compile'],0)
                self.assertFalse((Path(d)/'preparation_complete.json').exists())

    def test_ticket_tamper_or_parent_reservation_change_stops(self):
        for mode in ('ticket','reservation','pair'):
            with self.subTest(mode=mode),temp() as d:
                permit=artificial_permit(d,range(6))
                def mutate(task):
                    if mode=='ticket':Path(task['ticket_path']).write_text('{}')
                    elif mode=='pair':task['worker_cpus']=[8,9]
                    else:
                        path=Path(d)/'reservation_native_evolution_build_1.json';path.write_text('{}')
                with self.assertRaises((ValueError,KeyError)):
                    prep.prepare_cases(permit,self.consume,map_owned=self.fake_map(mutate=mutate))
                self.assertEqual(permit.calls['compile'],0)

    def test_native_result_drift_or_failed_exit_not_accepted(self):
        for kind in ('hash','case','source','exit','reap','calls'):
            with self.subTest(kind=kind),temp() as d:
                permit=artificial_permit(d,range(6));normal=self.fake_map()
                def fake(*args,**kwargs):
                    receipt=normal(*args,**kwargs);w=receipt['workers'][1]
                    path=Path(w['directory'])/'task_result.json';row=c.read(path)
                    if kind=='hash':w['result_sha256']='bad'
                    elif kind=='exit':w['exit_code']=1
                    elif kind=='reap':w['reaped']=False
                    else:
                        row[dict(case='case_id',source='SOURCE',calls='preparation_call_counts')[kind]]='bad'
                        path.write_text(__import__('json').dumps(row));w['result_sha256']=c.sha(path)
                    return receipt
                with self.assertRaises(ValueError):prep.prepare_cases(permit,self.consume,map_owned=fake)
                self.assertEqual(permit.calls['compile'],0)

    def test_prepared_descriptors_compile_serial_and_stop_at_each_failure(self):
        axes=c.binding()['axes']
        for fail in (None,*range(6)):
            with self.subTest(fail=fail),temp() as d:
                permit=artificial_permit(d,range(6))
                prepared=prep.prepare_cases(permit,self.consume,map_owned=self.fake_map());calls=[]
                def compile_axis(task,built,path):
                    i=len(calls);calls.append(task['id']);self.assertEqual(built['case_id'],task['case_id'])
                    return dict(case_id=task['case_id'],axis=task['axis'],measured_RZ=1 if i==fail else round(Fraction(axes[i]['exact_prediction'])))
                def compile_map(tasks,callback,*,directory,**kwargs):
                    self.assertEqual(len(tasks),1);self.assertEqual(len(kwargs['cpus']),1)
                    directory.mkdir();path=directory/'task_result.json';c.save(path,callback(tasks[0],directory))
                    return dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=[dict(task_id=tasks[0]['id'],
                        directory=str(directory),exit_code=0,reaped=True,result_sha256=c.sha(path),identity=[1,2,3,4],
                        peak_RSS_bytes=1,cpu_seconds=.1)])
                def execute():return adapter.serial_axes(permit,lambda case:prepared.pop(case['id']),compile_axis,map_owned=compile_map)
                if fail is None:self.assertEqual(len(execute()),6)
                else:
                    with self.assertRaisesRegex(ValueError,'AXIS_MODEL_ERROR_GE_2_PERCENT'):execute()
                self.assertEqual(len(calls),6 if fail is None else fail+1)


class ConcurrentProcesses(unittest.TestCase):
    def cpus(self):
        values=sorted(os.sched_getaffinity(0))
        if len(values)<6:self.fail('six OS CPUs required for artificial overlap test')
        # These tests use separate, previously idle cores; never the live run's 2/3/4.
        values=[v for v in values if v not in (2,3,4)]
        self.assertGreaterEqual(len(values),6)
        return values[:6]

    def test_both_wrappers_are_built_in_parallel_children_with_identical_spool_bytes(self):
        import numpy as np
        from types import SimpleNamespace
        cpus=self.cpus()[:2]
        with temp() as d:
            root=Path(d);barrier=root/'barrier';barrier.mkdir();parent=os.getpid()
            def wrapper(evolution,*,axis,**kwargs):
                c.save(root/(axis+'_builder.json'),dict(pid=os.getpid(),cpu=list(os.sched_getaffinity(0))))
                (barrier/(axis+'.ready')).touch();wait_markers(barrier,2)
                return SimpleNamespace(axis=axis,remove_final_measurements=lambda inplace:axis)
            native=SimpleNamespace(build_deterministic_native=lambda *a,**k:SimpleNamespace(circuit='evolution'),
                                   build_native_hadamard_wrapper=wrapper)
            m=SimpleNamespace(np=np,simulate_statevector=lambda circ,state:state if circ=='evolution' else circ,
                hadamard_expectation=lambda state,n:1. if state=='cosine' else 0.,
                canonical_qiskit_circuit_fingerprint=lambda w:'unchanged-'+w.axis)
            qpy=SimpleNamespace(dump=lambda w,f:f.write(('FROZEN-ARTIFICIAL-'+w.axis).encode()))
            permit=artificial_permit(d,cpus)
            target={key:dict(real=1.,imag=0.) for key in ('corrected','independent_forward')}
            with patch.dict(sys.modules,{'qiskit':SimpleNamespace(qpy=qpy)}):
                built=controls.build_case(dict(id='artificial_wrapper',q=1,formula='2nd'),(),0.,np.array([1.,0.]),
                    target,permit,native,m,n=1,cap=99,worker_cpus=cpus)
            records=[c.read(root/(axis+'_builder.json')) for axis in ('cosine','sine')]
            self.assertEqual(len({row['pid'] for row in records}),2)
            self.assertNotIn(parent,{row['pid'] for row in records})
            for axis,cpu,row in zip(('cosine','sine'),cpus,records):
                self.assertEqual(row['cpu'],[cpu])
                self.assertEqual(Path(built['spools'][axis]['path']).read_bytes(),('FROZEN-ARTIFICIAL-'+axis).encode())
                self.assertEqual(built['fingerprints'][axis],'unchanged-'+axis)
            self.assertEqual(permit.calls['measured_wrapper_build'],2)

    def test_three_cases_and_six_inner_workers_overlap_and_reap(self):
        cpus=self.cpus()
        with temp() as d:
            root=Path(d);barrier=root/'barrier';barrier.mkdir();permit=artificial_permit(d,cpus)
            def build(case,child,pair):
                (barrier/(case['id']+'.ready')).touch();wait_markers(barrier,3)
                for key,count in c.CASE_PREPARATION_CAPS.items():
                    for _ in range(count):child.take(key)
                def inner(task,path):
                    (barrier/(case['id']+'_'+task['id']+'.ready')).touch();wait_markers(barrier,9)
                    return dict(overlap=True,cpu=list(os.sched_getaffinity(0)))
                values=controls.batch([dict(id=x) for x in ('cosine','sine')],inner,child,'artificial_inner',
                    reserve=lambda task:None,worker_cpus=pair)
                return dict(artificial=True,inner=values)
            result=prep.prepare_cases(permit,build)
            self.assertEqual(len(result),3);self.assertEqual(len(list(barrier.glob('*.ready'))),9)
            for p in (root/'parallel_preparation').rglob('parallel_receipt.json'):
                receipt=c.read(p);self.assertEqual(receipt['status'],'PASS');self.assertTrue(receipt['all_reaped'])
                self.assertFalse(receipt['cleanup_errors'])
                for worker in receipt['workers']:self.assertIsNone(pool.identity(worker['pid']))
            self.assertEqual(permit.calls['compile'],0)

    def test_case_failure_cleans_nested_workers_and_never_compiles(self):
        cpus=self.cpus()
        with temp() as d:
            root=Path(d);barrier=root/'barrier';barrier.mkdir();permit=artificial_permit(d,cpus)
            first=c.binding()['cases'][0]['id']
            def build(case,child,pair):
                def inner(task,path):
                    (barrier/(case['id']+'_'+task['id']+'.ready')).touch();wait_markers(barrier,6)
                    if case['id']==first and task['id']=='left':raise RuntimeError('ARTIFICIAL_INNER_FAILURE')
                    while True:time.sleep(.02)
                controls.batch([dict(id=x) for x in ('left','right')],inner,child,'artificial_inner',
                    reserve=lambda task:None,worker_cpus=pair)
                return {}
            with self.assertRaises(RuntimeError):prep.prepare_cases(permit,build)
            receipts=list((root/'parallel_preparation').rglob('parallel_receipt.json'))
            self.assertEqual(len(receipts),4)
            for p in receipts:
                receipt=c.read(p);self.assertTrue(receipt['all_reaped']);self.assertFalse(receipt['cleanup_errors'])
                for worker in receipt['workers']:self.assertIsNone(pool.identity(worker['pid']))
            self.assertIn('ARTIFICIAL_INNER_FAILURE',c.read(root/'parallel_preparation/parallel_receipt.json')['first_error'])
            self.assertEqual(permit.calls['compile'],0)
            self.assertFalse((root/'preparation_complete.json').exists())


if __name__=='__main__':unittest.main()
