"""Synthetic acceleration tests only. One explicit small-toy transpiler regression; no H4."""
from pathlib import Path
import json
import multiprocessing as mp
import os
import sys
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import check_full_h_kappa2_p1_circuits as T
import run_full_h_kappa2_p2 as P
import full_h_p2_acceleration as A
import run_full_h_kappa2_p2_accelerated as Runner
import continue_full_h_p2_checkpoint as Transfer
import complete_full_h_p2_checkpoint as Completion
from qiskit.quantum_info import Operator
import numpy as np


class Tests(unittest.TestCase):
    def setUp(self):T.Tests.setUp(self)

    def test_cached_controlled_gates_preserve_full_legacy_bytes_and_unitary(self):
        native=self.full.setup();prep=self.ctx.preparations[0];row=T.ORDER2_ROW
        reqs,_,_=native.sample_requests(row,prep,P.P1.COHORT,0)
        scalar=self.ctx.ham.constant+prep.extracted_identity_coefficient
        old=native.compose(row,prep,reqs,scalar,lambda q:'old')
        provider=__import__('trotterlib.df_rte_qiskit',fromlist=['x'])
        before=provider.QiskitDFRTEEventCircuitBuilder._append_central_application
        with A.memoized_controlled_rotations(provider) as stats:
            fast=native.compose(row,prep,reqs,scalar,lambda q:'fast')
        self.assertIs(before,provider.QiskitDFRTEEventCircuitBuilder._append_central_application)
        from h46_b3_fast_fingerprint import FingerprintSession
        fp=FingerprintSession().fingerprint
        self.assertEqual(fp(old.circuit),fp(fast.circuit))
        np.testing.assert_allclose(Operator(old.circuit).data,Operator(fast.circuit).data,atol=1e-12,rtol=0)
        self.assertGreater(stats['reused'],0)

    def parallel_fake(self):
        barrier=mp.get_context('fork').Barrier(2)
        def compiler(wrapper,fp,representative):
            barrier.wait(timeout=15)
            return dict(metrics={k:1 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')},
                        fingerprint=fp,compiler='ARTIFICIAL',worker_pid=os.getpid()),None,False
        return compiler

    def test_XY_compile_in_distinct_processes_and_keep_existing_phase_gates(self):
        row=T.ORDER2_ROW;seeds=[P.P1.seed_function()(row,P.P1.COHORT,0,g,0) for g in range(row['q'])]
        result=A.accelerated_path(P,self.full,row,P.P1.COHORT,0,seeds,compiler=self.parallel_fake())
        pids={c['record']['worker_pid'] for c in result['compile_receipts'].values()}
        self.assertEqual(len(pids),2);self.assertNotIn(os.getpid(),pids)
        self.assertEqual(result['xy_phase_gate']['status'],'PASS')
        self.assertEqual(result['checks']['parity']['status'],'PASS')
        self.assertTrue(result['checks']['scalar_event_phases']['event_phases'])
        self.assertGreater(result['acceleration']['gate_cache']['reused'],0)

    def test_control_synthesis_count_reduced_without_changing_canonical_bytes(self):
        from qiskit import QuantumCircuit
        from h46_b3_fast_fingerprint import FingerprintSession
        provider=__import__('trotterlib.df_rte_qiskit',fromlist=['x'])
        base=provider.QiskitDFRTEEventCircuitBuilder
        event=SimpleNamespace(unsigned_rotation_angle=.123)
        app=SimpleNamespace(role='rotation',diagonal_pauli_support=(0,1))
        n=100;old=QuantumCircuit(3);fast=QuantumCircuit(3)
        start=time.monotonic()
        for _ in range(n):base._append_central_application(old,event,app,controlled=True,ancilla_qubit=2)
        old_time=time.monotonic()-start;start=time.monotonic()
        with A.memoized_controlled_rotations(provider) as stats:
            for _ in range(n):base._append_central_application(fast,event,app,controlled=True,ancilla_qubit=2)
        fast_time=time.monotonic()-start
        self.assertEqual(stats,dict(constructed=1,reused=n-1))
        fp=FingerprintSession().fingerprint;self.assertEqual(fp(old),fp(fast))
        print('\nTOY_SYNTHESIS_BENCHMARK',json.dumps(dict(repetitions=n,baseline_seconds=old_time,cached_seconds=fast_time,**stats)))

    def test_checkpoint_restoration_does_not_sample_or_compile(self):
        row=T.ORDER2_ROW;cohort='ARTIFICIAL_CHECKPOINT'
        seeds=[P.P1.seed_function()(row,cohort,0,g,0) for g in range(row['q'])]
        native=self.full.setup()
        def fake(wrapper,fp,representative):
            record=dict(metrics={k:1 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')},
                compiler=native.COMPILER_HASH,fingerprint=fp)
            P.P1.write(native.PRIVATE/'compiled_cache'/(fp+'.json'),record)
            return record,wrapper.copy() if representative else None,False
        original=P.acquire_path(self.full,row,cohort,0,seeds,compile_cached=fake)
        directory=native.PRIVATE/'cost'/cohort/row['id']/'0'
        files=[directory/'result.json',directory/'events_PRIVATE.json',*list((native.PRIVATE/'compiled_cache').glob('*.json'))]
        cp=dict(private=str(native.PRIVATE),trial=str(directory/'result.json'),events=str(directory/'events_PRIVATE.json'),
            source_commit='ARTIFICIAL',files={str(p):A.file_hash(p) for p in files})
        from trottertracks.resource_applicability import ax2b_stream_fingerprint_v5 as legacy
        with patch.object(native,'sample_requests',side_effect=AssertionError('CHECKPOINT_RESAMPLED')) as sampler, \
             patch.object(native,'compile_cached',side_effect=AssertionError('CHECKPOINT_RECOMPILED')) as compiler, \
             patch.object(legacy._Writer,'_circuit',side_effect=AssertionError('SLOW_FINGERPRINT_FALLBACK')):
            stages=[]
            restored=A.resume_saved_path(P,self.full,row,cohort,0,seeds,cp,progress=stages.append)
            sampler.assert_not_called();compiler.assert_not_called()
        self.assertEqual(stages[0],'checkpoint_hashes_and_receipts')
        self.assertEqual(stages[-1],'saved_path_checks_complete')
        self.assertLess(stages.index('wrapper_and_fingerprint_cosine'),stages.index('wrapper_and_fingerprint_sine'))
        self.assertEqual(original['events'],restored['events'])
        self.assertEqual(original['fingerprints'],restored['fingerprints'])
        self.assertEqual(restored['xy_phase_gate']['status'],'PASS')
        self.assertEqual(restored['acceleration']['new_compiles'],0)
        finished=self.tmp/'finished_path.json';P.P1.write(finished,dict(status='PASS',path=original))
        receipt=dict(path=str(finished),sha256=A.file_hash(finished),source_commit='ARTIFICIAL')
        with patch.object(native,'sample_requests',side_effect=AssertionError('FINISHED_RESAMPLED')) as sampler, \
             patch.object(native,'compile_cached',side_effect=AssertionError('FINISHED_RECOMPILED')) as compiler:
            reused=Completion.finished_path(receipt,row,cohort,0,seeds,native.COMPILER_HASH)
            sampler.assert_not_called();compiler.assert_not_called()
        self.assertEqual(json.dumps(reused['events'],sort_keys=True),json.dumps(original['events'],sort_keys=True))
        self.assertTrue(all(r['reused'] and r['producer_commit']=='ARTIFICIAL' for r in reused['compile_receipts'].values()))
        with self.assertRaisesRegex(ValueError,'identity/correctness'):
            Completion.finished_path(receipt,row,cohort,0,[0]*len(seeds),native.COMPILER_HASH)
        with self.assertRaisesRegex(ValueError,'receipt changed'):
            Completion.finished_path(dict(receipt,sha256='0'*64),row,cohort,0,seeds,native.COMPILER_HASH)

    def test_completion_child_reuses_finished_path_and_records_progress(self):
        private=self.tmp/'completion';private.mkdir()
        job=dict(private=str(private),output=str(private/'public'),result=str(private/'child.json'),
            plan={},stage='CALIBRATION',cell_id='ARTIFICIAL',step_seeds=[[1],[2]],checkpoint0={},finished1={})
        path=private/'job.json';P.P1.write(path,job)
        def restore(*args,progress,**kwargs):progress('ARTIFICIAL_NO_H4');return dict(index=0)
        with patch.object(P,'selected',return_value=dict(cohort='ARTIFICIAL',profile=dict(compiler='ARTIFICIAL'))), \
             patch.object(P,'physical_row',return_value={}),patch.object(P.P1,'provider_checks'), \
             patch.object(P.P1,'full_h_module',return_value=self.full), \
             patch.object(A,'resume_saved_path',side_effect=restore), \
             patch.object(Completion,'finished_path',return_value=dict(index=1)):
            self.assertEqual(Completion.child(path),0)
        value=json.loads(Path(job['result']).read_text())
        self.assertEqual(value['new_compiles'],0);self.assertEqual([r['index'] for r in value['paths']],[0,1])
        stages=[json.loads(line)['stage'] for line in (private/'PROGRESS.jsonl').read_text().splitlines()]
        self.assertEqual(stages[-1],'COMPLETE');self.assertIn('reuse_completed_path1',stages)

    def test_completion_refuses_active_worker_or_scientific_failure(self):
        terminal=self.tmp/'terminal.json'
        job=dict(old_run=dict(source_commit='ARTIFICIAL',processes=[dict(pid=os.getpid())],result=str(terminal)),
            handoff_reason='CONTROLLED_SAVED_PATH_HANDOFF')
        with self.assertRaisesRegex(ValueError,'still active'):Completion.stopped(job)
        job['old_run']['processes']=[]
        value=dict(source=dict(commit='ARTIFICIAL'),status='STOP_CELL',child=dict(status='STOP'),error=None)
        P.P1.write(terminal,value)
        with self.assertRaisesRegex(ValueError,'no failure retry'):Completion.stopped(job)
        value['child']=None;terminal.write_text(json.dumps(value));Completion.stopped(job)

    def test_path_workers_are_parallel_and_return_ordered_receipts(self):
        plan=P.proposal();plan['adopted']=True
        private=self.tmp/'parallel_child';private.mkdir();barrier=mp.get_context('fork').Barrier(2)
        job=dict(plan=plan,stage='CALIBRATION',cell_id='G566',step_seeds=P.seed_table(plan,'G566'),
            private=str(private),result=str(private/'child.json'))
        path=private/'job.json';path.write_text(json.dumps(job))
        def artificial(job,index):
            barrier.wait(timeout=15)
            P.P1.write(Path(job['private'])/('path_result_'+str(index)+'.json'),
                dict(status='PASS',path=dict(index=index,worker_pid=os.getpid(),scope='ARTIFICIAL_NO_H4')))
        with patch.object(Runner,'path_worker',artificial):self.assertEqual(Runner.child(path),0)
        saved=json.loads(Path(job['result']).read_text())
        self.assertEqual([p['index'] for p in saved['paths']],[0,1])
        self.assertEqual(len({p['worker_pid'] for p in saved['paths']}),2)

    def test_continuation_refuses_an_already_started_unclaimed_path(self):
        plan=P.proposal();plan['adopted']=True;cell=P.selected(plan,'CALIBRATION','G36223')
        row=P.physical_row(cell);private=self.tmp/'old_private';private.mkdir()
        previous=dict(plan=plan,cell_id=cell['cell_id'],step_seeds=P.seed_table(plan,cell['cell_id']),private=str(private))
        old_job=self.tmp/'previous_job.json';old_job.write_text(json.dumps(previous))
        result=self.tmp/'previous_result.json';result.write_text(json.dumps(dict(source=dict(commit='d'*40),status='STOP_CELL',child=None,error=None)))
        trial=self.tmp/'saved_trial.json';trial.write_text(json.dumps(dict(status='PASS',candidate=row,cohort=cell['cohort'],index=0,
            checks=[dict(check=k,error=0.,tolerance=1e-9,phase_alignment=False) for k in ('low_control','high_control_leakage',
                'independent_occupation_vector','Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine')])))
        job=dict(previous,stage='CALIBRATION',previous_run=dict(worker_pid=999999999,job=str(old_job),result=str(result),source_commit='d'*40,
                termination_reason='CONTROLLED_AFTER_NATIVE_CHECKPOINT'),
            resume={'0':dict(source_commit='d'*40,trial=str(trial),files={str(trial):A.file_hash(trial)})},new_compile_maximum=2)
        Runner.check_continuation(job)
        job['previous_run']['worker_pid']=os.getpid()
        with self.assertRaisesRegex(ValueError,'still active'):Runner.check_continuation(job)
        job['previous_run']['worker_pid']=999999999
        terminal=json.loads(result.read_text());terminal['child']=dict(status='STOP',error='ARTIFICIAL_GATE_FAILURE')
        result.write_text(json.dumps(terminal))
        with self.assertRaisesRegex(ValueError,'no failure retry'):Runner.check_continuation(job)
        terminal['child']=None;result.write_text(json.dumps(terminal))
        directory=private/'cost'/cell['cohort']/row['id']/'1';directory.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Unclaimed old path'):Runner.check_continuation(job)

    def test_compile_failure_has_no_retry_and_restores_the_builder(self):
        native=self.full.setup();provider=__import__('trotterlib.df_rte_qiskit',fromlist=['x'])
        builder=provider.QiskitDFRTEEventCircuitBuilder;before=builder._append_central_application
        barrier=mp.get_context('fork').Barrier(2)
        def failing(wrapper,fp,representative):
            axis='Y' if wrapper.data[-3].operation.name=='sdg' else 'X'
            with (self.tmp/('attempt_'+axis)).open('x') as f:f.write(str(os.getpid()))
            barrier.wait(timeout=15)
            if axis=='X':raise ValueError('ARTIFICIAL_COMPILE_FAILURE')
            return dict(metrics={k:1 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')}),None,False
        row=T.ORDER2_ROW;cohort='ARTIFICIAL_FAILURE';seeds=[P.P1.seed_function()(row,cohort,0,g,0) for g in range(row['q'])]
        with self.assertRaisesRegex(ValueError,'Parallel cosine compiler STOP'):
            A.accelerated_path(P,self.full,row,cohort,0,seeds,compiler=failing)
        self.assertEqual(len(list(self.tmp.glob('attempt_*'))),2)
        self.assertIs(before,builder._append_central_application)

    def test_real_toy_transpiles_return_QPY_and_keep_compiled_vector_gates(self):
        from functools import partial
        import qiskit
        import trotterlib.rte_compiled_cost as costs
        row=T.ORDER2_ROW;cohort='ARTIFICIAL_REAL_TRANSPILER_REGRESSION'
        seeds=[P.P1.seed_function()(row,cohort,0,g,0) for g in range(row['q'])]
        with patch.object(costs,'transpile',partial(qiskit.transpile,num_processes=1)):
            result=A.accelerated_path(P,self.full,row,cohort,0,seeds)
        self.assertEqual(len(result['trial']['checks']),7)
        self.assertEqual(result['trial']['cache_hits'],0)
        self.assertEqual(result['xy_phase_gate']['status'],'PASS')
        self.assertTrue(all(c['record']['compiler']==self.full.setup().COMPILER_HASH for c in result['compile_receipts'].values()))

    def test_transfer_uses_only_completed_native_boundaries(self):
        plan=P.proposal();plan['adopted']=True;cell=P.selected(plan,'CALIBRATION','G36223');row=P.physical_row(cell)
        private=self.tmp/'transfer';cache=private/'compiled_cache';cache.mkdir(parents=True)
        previous=dict(plan=plan,stage='CALIBRATION',cell_id=cell['cell_id'],step_seeds=P.seed_table(plan,cell['cell_id']),private=str(private))
        d=private/'cost'/cell['cohort']/row['id']/'0';d.mkdir(parents=True)
        checks=[dict(check=k,error=0.,tolerance=1e-9,phase_alignment=False) for k in ('low_control','high_control_leakage',
            'independent_occupation_vector','Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine')]
        (d/'result.json').write_text(json.dumps(dict(status='PASS',candidate=row,cohort=cell['cohort'],index=0,checks=checks)))
        (d/'events_PRIVATE.json').write_text(json.dumps(dict(cohort=cell['cohort'],seeds=previous['step_seeds'][0],scope='ARTIFICIAL_METADATA')))
        for a in ('X','Y'):(cache/(a+'.json')).write_text(json.dumps(dict(scope='ARTIFICIAL_METADATA',axis=a)))
        self.assertEqual(Transfer.completed_paths(previous),('READY',[0]))
        cp=Transfer.snapshot(previous,'ARTIFICIAL_SOURCE',[0]);self.assertEqual(set(cp),{'0'})
        d1=d.parent/'1';d1.mkdir()
        self.assertEqual(Transfer.completed_paths(previous),('NATIVE_WORK_IN_PROGRESS',[]))
        d1.rmdir();(d/'STOP.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Old scientific STOP'):Transfer.completed_paths(previous)


if __name__=='__main__':unittest.main(verbosity=2)
