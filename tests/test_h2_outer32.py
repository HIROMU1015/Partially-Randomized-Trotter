"""Artificial design/authority/storage and unchanged-native-path tests for H2."""
import copy
from functools import partial
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import CodeType
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_outer32_contract as c
import h2_outer32_runtime as r
import h2_outer32_pipeline as p

class DesignTests(unittest.TestCase):
    def test_fixed12_groups_and_outer32(self):
        groups=c.groups();self.assertEqual(len(groups),12)
        self.assertEqual({(g['formula'],g['q']) for g in groups},{(f,q) for f in c.FORMULAS for q in (1,2,4,8)})
        self.assertTrue(all(g['system']=='H2' and g['prefix']==1 and g['K']==4 and g['r_value']==32 for g in groups))
        self.assertTrue(all(g['r_policy']=='ABS_TIME_PER_OUTER' for g in groups))
    def test_tasks_and_call_budget(self):
        tasks=c.tasks_for(c.STAGES[1],{g['id']:33 for g in c.groups()});scope=c.scope(c.STAGES[1],tasks)
        self.assertEqual((len(tasks),scope['compile_axes']),(396,792));self.assertFalse(scope['retry'])
        self.assertEqual(scope['rte_events'],33*3*32*(1+2+4+8))
        with self.assertRaises(ValueError):c.tasks_for(c.STAGES[1],{g['id']:8 for g in c.groups()})
    def test_group_copy_and_registry_cohort_separation(self):
        a=c.groups();a[0]['q']=99;self.assertEqual(c.groups()[0]['q'],1)
        t=c.tasks_for(c.STAGES[1],{g['id']:33 for g in c.groups()})[:1]
        a=c.registry(t,'a'*40,'TEST','DESIGN_PILOT',[]);b=c.registry(t,'a'*40,'TEST','EXPLORATION',[])
        self.assertNotEqual(a,b)
        with self.assertRaisesRegex(ValueError,'COLLISION'):c.registry(t,'a'*40,'TEST','EXPLORATION',[next(iter(next(iter(b.values())).values()))])
    def evaluation(self,zero=False):
        return dict(status='H2_PILOT_DESIGN_PASS',groups=[dict(candidate=g,statistics=dict(n=33,n_E_sensitivity=[
            dict(relative_MCSE_target=.02,variance_inflation=4,n_E_nominal_only=None if zero else 81)])) for g in c.groups()])
    def test_main_counts_from_H2_not_H4_H6(self):
        counts,unknown=c.counts_from_evaluation(self.evaluation());self.assertEqual(set(counts.values()),{81});self.assertFalse(unknown)
        v=self.evaluation();v['groups'].pop()
        with self.assertRaises(ValueError):c.counts_from_evaluation(v)
    def test_zero_observed_variance_is_unresolved_minimum(self):
        counts,unknown=c.counts_from_evaluation(self.evaluation(True));self.assertEqual(set(counts.values()),{33});self.assertEqual(len(unknown),12)
    def test_no_grant_refuses_before_inputs(self):
        with patch.object(c,'read',side_effect=AssertionError('INPUT_READ')):
            for cfg in ({},{'authorized':False,'grant':{'valid':True}}):
                with self.assertRaisesRegex(ValueError,'EXPLICIT_H2'):c.authorize(cfg)
    def test_CLI_refuses_all_entrypoints_before_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'draft.json';path.write_text('{"authorized":false,"grant":{"valid":false}}')
            for flag in ('--execute','--supervise','--audit'):
                result=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h2_outer32_stage.py'),flag,'--config',str(path)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2);self.assertIn('EXPLICIT_H2',result.stdout)
            self.assertEqual(len(list(Path(tmp).iterdir())),1)
    def test_pipeline_refuses_before_any_input_read(self):
        with patch.object(c,'read',side_effect=AssertionError('INPUT_READ')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_H2'):p.require_parent({})
    def test_no_double_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=c.tasks_for(c.STAGES[1],{g['id']:33 for g in c.groups()})[0]
            cfg=dict(SOURCE='a'*40,stage=task['stage'],tasks=[task],ledger=tmp,marker=str(Path(tmp)/'marker'),grant={})
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            c.reserve(cfg,task,'compile_axes',0)
            with self.assertRaises(FileExistsError):c.reserve(cfg,task,'compile_axes',0)
            with self.assertRaises(ValueError):c.reserve(cfg,task,'compile_axes',2)
    def test_other_stage_reservation_released_only_after_reap(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(resources=dict(other_fixed_growth_reserve_bytes=100,other_controller_finished=str(Path(tmp)/'done.json')))
            self.assertEqual(p.other_reservation(cfg),100)
            c.save(Path(tmp)/'done.json',dict(stages={'A':{'reaped':True},'B':{'reaped':True}}))
            self.assertEqual(p.other_reservation(cfg),0)
    def test_native_code_operations_and_constants_unchanged(self):
        import h46_pr_pilot_native as original
        def identical(a,b,label):
            self.assertEqual(a.co_code,b.co_code);self.assertEqual(a.co_names,b.co_names)
            for x,y in zip(a.co_consts,b.co_consts):
                if isinstance(x,CodeType):identical(x,y,label)
                else:self.assertEqual(label if x=='DESIGN_PILOT' else x,y)
        for stage in (c.STAGES[1],c.STAGES[3]):
            r.RUNTIME=None;rt=r.get_runtime(dict(stage=stage,signal_path='/synthetic/readonly.json'))
            identical(original.run_task.__code__,rt.native.run_task.__code__,c.cohort(stage))
        r.RUNTIME=None

    def test_fresh_stage_packs_do_not_pool_pilot_or_mutate_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);signal=root/'signals.json';signal.write_text('{}');ref=root/'ref.npz';ref.write_bytes(b'ARTIFICIAL_ONLY')
            exclude=root/'exclude.json';c.save(exclude,dict(values=[1,2,3]));approval=root/'approval.json';approval.write_text('{}')
            parent=dict(SOURCE='a'*40,runtime={},provider_root='ARTIFICIAL',inputs={},binding={'synthetic':True},
                resources={'runtime_caps':False,'host_PSI_kill':False},storage=c.storage.storage_policy(),
                compiler_phase_policy='ARTIFICIAL',approval_path=str(approval),private=str(root),sealed_files=[],initial_exclusions=str(exclude))
            before=copy.deepcopy(parent)
            with patch.object(c,'authorize'):
                _,pilot=p.make_stage(parent,c.STAGES[1],str(signal),str(ref),'TOY_PREP')
                design=root/'design.json';c.save(design,self.evaluation())
                _,main=p.make_stage(parent,c.STAGES[3],str(signal),str(ref),'TOY_PREP',str(design))
            self.assertEqual(parent,before)
            old={v for row in c.read(pilot['registry_path']).values() for v in row.values()}
            new={v for row in c.read(main['registry_path']).values() for v in row.values()}
            self.assertFalse(old&new);self.assertTrue(old<=set(c.read(main['excluded_seeds_path'])['values']))
            self.assertFalse(Path(pilot['run_root']).exists());self.assertFalse(Path(main['run_root']).exists())
            self.assertNotEqual(pilot['plan_sha256'],main['plan_sha256'])
            main['reference_path']='CHANGED'
            self.assertNotEqual(main['plan'],c.plan_fields(main))

@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'explicit artificial provider required')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))
    def test_three_formulas_both_cohorts_artificial_native_and_archive(self):
        from test_h46_m5_cost_readiness import NativeToyTests
        import h46_m5_native as original
        import trotterlib.rte_compiled_cost as cost
        for stage in (c.STAGES[1],c.STAGES[3]):
            for formula in c.FORMULAS:
                with self.subTest(stage=stage,formula=formula),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir();ctx=NativeToyTests.toy(self)
                    row=next(g for g in c.groups() if g['formula']==formula and g['q']==1)
                    task=dict(id='ARTIFICIAL_H2',candidate=row,sample=0,axes=['cosine','sine'],stage=stage)
                    cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=stage,tasks=[task],run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),
                        marker=str(root/'CONSUMED.json'),grant={},storage=c.storage.storage_policy(),resources={'process_start_method':'fork'},
                        systems={'H2':{'binding':{'synthetic':True}}},registry_path=str(root/'registry.json'),signal_path=str(root/'signals.json'),sealed_files=[])
                    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                    c.save(cfg['registry_path'],c.registry([task],cfg['SOURCE'],'ARTIFICIAL_ONLY',c.cohort(stage),[]))
                    signal=original.signal(ctx,dict(row,policy=row['r_policy']));c.save(cfg['signal_path'],dict(records=[dict(candidate=row,signal=signal)]))
                    for path in (cfg['registry_path'],cfg['signal_path']):cfg['sealed_files'].append(c.seal(path,'synthetic'))
                    r.RUNTIME=None
                    with patch.object(c,'authorize'),patch.object(c,'verify'):
                        rt=r.get_runtime(cfg);rt.runner_namespace.update(WORKER_CFG=cfg,WORKER_STOP=threading.Event())
                        with patch.dict(rt.native_namespace,{'context':lambda *a:ctx}),patch.object(cost,'transpile',partial(r.phase_transpile,num_processes=1)):
                            r.worker(task)
                        audit=rt.auditor.task_audit(cfg,task)
                    self.assertEqual(audit['archives'],6)
                    result=c.read(root/'output'/task['id']/'result.json');self.assertEqual(result['seed_cohort'],c.cohort(stage))
                    self.assertTrue(all(g['PASS'] for g in result['checks']))
                    for i in range(2):self.assertTrue(c.read(root/'output'/task['id']/('compiler_phase_guard_'+str(i)+'.json'))['unchanged_gate_stream'])
        r.RUNTIME=None

    def test_saved_H2_layout_restore_and_independent_signal_artificial(self):
        import hashlib
        import numpy as np
        from scipy.linalg import expm
        from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import _state_hash
        from h46_new2_native import OccupationOracle
        import h46_m5_native as native
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
            full=np.zeros(16,complex);full[sector.basis_indices[0]]=1/np.sqrt(2);full[sector.basis_indices[-1]]=1j/np.sqrt(2)
            state=full[sector.basis_indices]
            one=np.diag([.11,.13,.17,.19]);lambdas=np.array([.2,.3,-.1]);gs=np.array([np.diag([.2,.3,.4,.5]),np.diag([.1,.3,-.4,.2]),np.diag([.3,-.2,.1,.4])])
            metadata={'synthetic':True};ham=DFHamiltonian(.37,one,lambdas,tuple(gs),metadata)
            arrays=dict(constant=np.array(.37),one_body=one,lambdas=lambdas,g_matrices=gs,
                state_vector=full,sector_state_vector=state,sector_basis_indices=sector.basis_indices)
            snapshot=root/'snapshot.npz'
            with snapshot.open('xb') as f:np.savez(f,**arrays)
            records={}
            for key,value in arrays.items():
                a=np.ascontiguousarray(value);records[key]=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
            receipt=root/'receipt.json';c.save(receipt,dict(sha256=c.sha(snapshot),arrays=records,hamiltonian_metadata=metadata))
            oracle=OccupationOracle(ham,full,prefix=3);mats=[(a[1]*a[0])@a[2] for a in oracle.specs]
            target=expm(-.8j*(sum(mats)+.37*np.eye(len(oracle.initial))))@oracle.initial;z=complex(np.vdot(oracle.initial,target))
            fp=root/'fp.json';c.save(fp,dict(reference={'reference':dict(real=z.real,imag=z.imag)}))
            cfg=dict(SOURCE='ARTIFICIAL_H2_LAYOUT',stage=c.STAGES[0],inputs={
                'snapshot':{'path':str(snapshot)},'snapshot_receipt':{'path':str(receipt)},'FP_result':{'path':str(fp)}},
                binding=dict(snapshot=c.sha(snapshot),hamiltonian=df_hamiltonian_hash(ham),state=_state_hash(full,state)))
            r.CTX.clear()
            with patch.object(r,'install'):ctx=r.context(cfg)
            self.assertTrue(np.array_equal(ctx.qstate[ctx.oracle.qindices],ctx.oracle.initial));self.assertEqual(ctx.ham.n_blocks,3)
            for row in [g for g in c.groups() if g['q']==1]:
                saved=native.signal(ctx,dict(row,policy=row['r_policy']));values,logB=r.independent_signal(ctx,row)
                for k,value in zip(('exact_tail','corrected','raw'),values):self.assertLess(abs(value-complex(**{'real':saved[k]['real'],'imag':saved[k]['imag']})),1e-9)
                self.assertLess(abs(logB-saved['log_B']),1e-9)
            self.assertTrue(np.array_equal(full,arrays['state_vector']));r.CTX.clear()

if __name__=='__main__':unittest.main()
