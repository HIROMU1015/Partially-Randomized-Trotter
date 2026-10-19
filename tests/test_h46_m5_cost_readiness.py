"""Synthetic metadata, child processes and four-mode DF only; no H4/H6 arrays."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_m5_cost_design as d
import h46_m5_cost_contract as c
import h46_m5_cost_native as native
import run_h46_m5_cost as launch

def row(method='B0',formula='2nd',q=1,id='TOY',system='H4'):
    return dict(id=id,method=method,formula=formula,q=q,prefix=1,system=system,role='PRIMARY',kind='PR_SIGNAL' if method=='PR' else 'DET_SIGNAL',K=4 if method=='PR' else None,r_policy='ABS_TIME_PER_OUTER' if method=='PR' else None,r_value=32 if method=='PR' else None)

def saved(status='ELIGIBLE',method='B0',id='TOY'):
    return dict(status='SIGNAL_DIAGNOSTIC_PASS',candidate=row(method,id=id),preparation_hash='toy',signal_origin='NEW_FINITE_MEAN_SIGNAL',empirical_eligibility=dict(rows=[dict(epsilon=e,axis=a,status_under_proposed_empirical_policy=status,shots=None,cost=None) for e in d.EPSILONS for a in ('real','imag')]))

def fake_worker(task):
    cfg=launch.WORKER_CFG;stop=launch.WORKER_STOP
    if cfg.get('failure')==task['id']:stop.set();raise RuntimeError('SYNTHETIC_STOP')
    if stop.is_set():raise RuntimeError('OWN_STAGE_STOP')
    for kind,n in c.task_limits(task).items():
        for i in range(n):c.reserve(cfg,task,kind,i)
    out=Path(cfg['output'])/task['id'];out.mkdir()
    c.save(out/'result.json',dict(status='UNIT_COST_PASS_NOT_ACCOUNTING',task=task,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],stage=cfg['stage'],axis_costs=[dict(axis=a,compiler_settings_hash=d.COMPILER_HASH,native_policy=d.NATIVE_POLICY) for a in ('cosine','sine')],shot_accounting=False,cost_ranking=False,conditional_sampling=False,cohort_pooling=False))
    return dict(id=task['id'],result_sha256=c.sha(out/'result.json'),peak_RSS_KiB=1)

class DesignTests(unittest.TestCase):
    def test_same_endpoint_both_axes_required(self):
        x=saved('INELIGIBLE');x['empirical_eligibility']['rows'][0]['status_under_proposed_empirical_policy']='ELIGIBLE';x['empirical_eligibility']['rows'][3]['status_under_proposed_empirical_policy']='ELIGIBLE'
        self.assertEqual(d.selection([x])[0]['eligible_epsilons'],[])
        x['empirical_eligibility']['rows'][1]['status_under_proposed_empirical_policy']='ELIGIBLE'
        self.assertEqual(d.selection([x])[0]['eligible_epsilons'],[.05])

    def test_unknown_not_excluded_or_eligible(self):
        x=d.selection([saved('UNDETERMINED')])[0];self.assertEqual(x['proposal'],'HOLD_UNDETERMINED');self.assertEqual(len(x['unresolved_epsilons']),5)

    def test_duplicate_missing_axes_shots_labels_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):d.selection([saved(),saved()])
        for change in [lambda x:x['empirical_eligibility']['rows'].pop(),lambda x:x['empirical_eligibility']['rows'][0].update(shots=1),lambda x:x['empirical_eligibility']['rows'][0].update(status_under_proposed_empirical_policy='PASS')]:
            x=saved();change(x)
            with self.assertRaises(ValueError):d.selection([x])

    def test_full_pair_reused_not_recompiled(self):
        domain=d.selection([saved()]);self.assertEqual(d.tasks_for('H4_DET',domain,{('TOY','cosine'),('TOY','sine')}),[])
        with self.assertRaisesRegex(ValueError,'PARTIAL_PAIR'):d.tasks_for('H4_DET',domain,{('TOY','cosine')})

    def test_no_endpoint_and_other_system_not_dispatched(self):
        self.assertEqual(d.tasks_for('H4_DET',d.selection([saved('INELIGIBLE')]),set()),[])
        self.assertEqual(d.tasks_for('H6_DET',d.selection([saved()]),set()),[])
        with self.assertRaises(ValueError):d.tasks_for('PF788',[],set())

    def test_PR_n_null_and_missing_q_refused(self):
        domain=d.selection([saved(method='PR')])
        for counts in [{},{'TOY':None},{'TOY':1},{'TOY':True},{'TOY':2,'OTHER':2}]:
            with self.assertRaises(ValueError):d.tasks_for('H4_PR',domain,set(),counts)
        self.assertEqual(len(d.tasks_for('H4_PR',domain,set(),{'TOY':3})),3)

    def test_call_bounds_include_pieces_events_paired_axes(self):
        task=dict(id='toy__0',candidate=row('PR','4th(m5_best)',8),sample=0,axes=['cosine','sine'],stage='H4_PR')
        scope=d.call_scope([task],'PR');self.assertEqual(scope['compile_axes'],2);self.assertEqual(scope['request_sampling_calls'],88);self.assertEqual(scope['rte_events'],256)
        self.assertEqual(scope['new_signal'],0);self.assertEqual(scope['cost_accounting'],0)

    def test_cohort_campaign_and_collision_separation(self):
        from h46_m5_contract import core
        tasks=[dict(id='toy__0',candidate=row('PR','4th(m5_best)',2))]
        a=d.registry(tasks,'a'*40,'run-A','EXPLORATION',[],core.schedule)
        self.assertEqual(a,d.registry(tasks,'a'*40,'run-A','EXPLORATION',[],core.schedule))
        self.assertNotEqual(a,d.registry(tasks,'a'*40,'run-B','EXPLORATION',[],core.schedule))
        self.assertNotEqual(a,d.registry(tasks,'a'*40,'run-A','CONFIRMATION',[],core.schedule))
        with self.assertRaisesRegex(ValueError,'COLLISION'):d.registry(tasks,'a'*40,'run-A','EXPLORATION',[next(iter(a['toy__0'].values()))],core.schedule)
        with self.assertRaisesRegex(ValueError,'NAMESPACE'):d.seed('a'*40,'','EXPLORATION','T',0,0)

    def test_n_proposals_do_not_invent_variance_or_tail_cost(self):
        p=d.sampling_proposal(d.selection([saved(method='PR')]))[0]
        self.assertIsNone(p['n_E']);self.assertIsNone(p['unseen_rare_variance']);self.assertIsNone(p['unseen_rare_cost']);self.assertFalse(p['old_piece_r2_samples_usable_for_new_mean'])

    def test_rare_strata_exclusive_priority(self):
        for orders,name in [([0,0],'s0'),([0,2],'s2_single'),([2,2],'s2_multi'),([2,4],'s4')]:
            self.assertEqual(d.event_stratum([[dict(taylor_order=k) for k in orders]])['stratum'],name)
        with self.assertRaises(ValueError):d.event_stratum([[dict(taylor_order=6)]])

class ContractTests(unittest.TestCase):
    def test_draft_refused_before_seals_and_molecular_import(self):
        with patch.object(c,'verify',side_effect=AssertionError('must not reach')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_UNIT_COST'):c.authorize(dict(authorized=False,grant=dict(valid=False)))

    def test_both_CLI_paths_reject_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';c.save(p,dict(authorized=False,grant=dict(valid=False)))
            for mode in ('--execute','--supervise'):
                result=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h46_m5_cost.py'),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2);self.assertIn('EXPLICIT_UNIT_COST',result.stdout)
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_unadopted_decisions_and_foreign_stage_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNADOPTED_COST_DECISIONS'):c.authorize(dict(authorized=True,grant=dict(valid=True),stage='H4_DET',decisions={}))
        with self.assertRaisesRegex(ValueError,'FOUR_ISOLATED'):c.authorize(dict(authorized=True,grant=dict(valid=True),stage='PF788'))

    def test_reservations_exact_and_nonrefundable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dict(id='toy',candidate=row(),stage='H4_DET');cfg=dict(tasks=[task],SOURCE='a'*40,stage='H4_DET',ledger=tmp,marker=str(root/'CONSUMED.json'),grant={})
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            c.reserve(cfg,task,'compile_axes',0)
            with self.assertRaises(FileExistsError):c.reserve(cfg,task,'compile_axes',0)
            with self.assertRaisesRegex(ValueError,'EXACT_CALL'):c.reserve(cfg,task,'compile_axes',2)
            with self.assertRaisesRegex(ValueError,'EXACT_CALL'):c.reserve(cfg,task,'trajectories',0)

    def test_gate_equality_fails_and_quarantine_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'NUMERIC_DISAGREEMENT'):native.gate(tmp,'boundary',1e-7,1e-7)
            self.assertFalse(c.read(Path(tmp)/'boundary.json')['PASS'])
            self.assertTrue(native.gate(tmp,'below',1e-9,1e-7)['PASS'])
            path=Path(tmp)/'raw.bin';native.atomic_binary(path,lambda f:f.write(b'raw'))
            with self.assertRaises(ValueError):native.atomic_binary(path,lambda f:f.write(b'changed'))
            self.assertEqual(path.read_bytes(),b'raw')

    def test_pool_complete_and_failure_preserve_separate_stage(self):
        for failure in (None,'T1',None):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';tasks=[dict(id=x,candidate=row(id=x),sample=0,axes=['cosine','sine'],stage='H4_DET') for x in ('T1','T2')]
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage='H4_DET',tasks=tasks,run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},runtime=dict(python=sys.executable),failure=failure)
                with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(launch,'fresh',return_value=dict(workers=2)):
                    code=launch.execute(cfg,target=fake_worker)
                    self.assertEqual(code,0 if failure is None else 1)
                    report=c.read(root/'completion_witness.json');self.assertTrue(all(r['exit']==0 and r['reaped'] for r in report['workers'].values()));self.assertFalse(report['next_phase_started'])
                    with self.assertRaisesRegex(ValueError,'UNUSED_COST_RUN_ROOT'):launch.execute(cfg,target=fake_worker)

@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'explicit synthetic provider required')
class NativeToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))

    def toy(self):
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian
        from h46_new2_native import prepare,OccupationOracle
        from h46_new2_core import spectrum,matrix_only_tail_coefficients
        from h46_m5_native import signal
        rng=np.random.default_rng(10817)
        def matrix():
            a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));return .1*(a+a.conj().T)
        ham=DFHamiltonian(.37,matrix(),np.array([.2,-.3]),(matrix(),matrix()),{})
        state=np.zeros(16,complex);state[3]=1/np.sqrt(2);state[12]=1j/np.sqrt(2)
        oracle=OccupationOracle(ham,state,prefix=2);prep=prepare(ham,1);full=prepare(ham,2)
        matrices=[(s[1]*s[0])@s[2] for s in oracle.specs];scalar,lam=matrix_only_tail_coefficients(ham,1)
        qstate=np.zeros(16,complex);qstate[oracle.qindices]=oracle.initial
        return NS(np=np,ham=ham,oracle=oracle,cut=prep,fullprep=full,qstate=qstate,scalar=ham.constant,tail_identity=scalar,lambda_R=lam,tail_spec=spectrum(matrices[2]-scalar*np.eye(len(oracle.initial))),reference=oracle.initial.copy())

    def test_actual_native_cost_adapter_all_three_PR_formulas_and_DET(self):
        import h46_m5_native as original
        for formula,method,inject_failure in [('2nd','PR',False),('4th','PR',False),('4th(m5_best)','PR',False),('4th(m5_best)','B0I',False),('4th','B0I',True)]:
            with self.subTest(formula=formula,method=method,failure=inject_failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir();ctx=self.toy();r=row(method,formula,id='ARTIFICIAL_ONLY');task=dict(id='ARTIFICIAL_ONLY__0',candidate=r,sample=0,axes=['cosine','sine'],stage='H4_PR' if method=='PR' else 'H4_DET')
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=task['stage'],tasks=[task],output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},systems=dict(H4=dict(binding={'synthetic':True})),registry_path=str(root/'registry.json'))
                c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                group=dict(r,policy=r['r_policy']);sig=original.signal(ctx,group)
                c.save(cfg['registry_path'],d.registry([task],cfg['SOURCE'],'artificial','EXPLORATION',[],original.c.core.schedule))
                read=c.read
                def read_override(path):
                    if str(path).endswith('/signal_records.json'):return dict(records=[dict(candidate=r,signal=sig)])
                    return read(path)
                from trotterlib.df_trotter.circuit import simulate_statevector
                calls=[0]
                def vector(circuit,initial):
                    calls[0]+=1;value=simulate_statevector(circuit,initial)
                    return value+1e-4 if inject_failure and calls[0]==4 else value
                with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(native,'context',return_value=ctx),patch.object(c,'read',side_effect=read_override),patch('trotterlib.df_trotter.circuit.simulate_statevector',side_effect=vector):
                    if inject_failure:
                        stopped=threading.Event()
                        with self.assertRaisesRegex(ValueError,'compiled_vector_cosine'):native.run_task(cfg,task,stopped)
                        output=root/'output'/task['id'];self.assertTrue(stopped.is_set());self.assertTrue((output/'quarantine_cosine/compiled.qpy').is_file());self.assertTrue((output/'quarantine_cosine/vectors.npz').is_file());self.assertFalse((output/'cost_cosine.json').exists());self.assertFalse((output/'cost_sine.json').exists());self.assertEqual(calls[0],4)
                        continue
                    result=native.run_task(cfg,task,threading.Event())
                report=read(root/'output'/task['id']/'result.json');self.assertEqual(report['status'],'UNIT_COST_PASS_NOT_ACCOUNTING');self.assertEqual([x['axis'] for x in report['axis_costs']],['cosine','sine']);self.assertFalse(report['cohort_pooling']);self.assertFalse(report['shot_accounting']);self.assertFalse(report['conditional_sampling'])
                self.assertTrue(all(x['metrics']['rz_count']>0 for x in report['axis_costs']))
                expected=sum(c.task_limits(task).values());self.assertEqual(len(list((root/'ledger').glob('*.json'))),expected)
                self.assertEqual(result['result_sha256'],c.sha(root/'output'/task['id']/'result.json'))

if __name__=='__main__':unittest.main()
