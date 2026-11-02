"""Artificial fixtures only; no H4 sampling, quantum build or saved-cost analysis."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_tol_p_contract as p
import h4_tol_p_science as science
import run_h4_tol_p as runner
import audit_h4_tol_p_completed as completed
c = p.c


def fixtures():
    eps = []; preps = {}; signals = {}
    for target in p.plan()['targets']:
        cell = target['cell']; cid = cell['id']
        eps.append(dict(cell=cell, n=p.N[cid], epsilon=target['epsilon'],
            saved_shots=dict(cosine=1, sine=2), PF_G_exact='100', expected_direction=1))
        preps[cid] = dict(preparation_hash='p'*64)
        signals[cid] = dict(log_B=0., allowance=dict(log_B_margin=1e-9))
    reg = p.registry('a'*40, eps, preps, signals)
    return eps, reg


def pair(task, cosine=10, sine=20):
    row = dict(cell_id=task['cell']['id'], replica=task['replica'], seed=task['seed'],
        event_digest='e'*64, cohort=p.COHORT, stratum='S0_ONLY',
        calls=dict(trajectory=1, occurrence=task['cell']['q'], RTE_event=task['cell']['R'],
            native_evolution=1, wrapper_compile=2, control_probe=4 if task['replica'] == 0 else 0,
            independent_event_oracle=1 if task['replica'] == 0 else 0))
    row['axes'] = {a: dict(axis=a, RZ=v, seed=row['seed'], event_digest=row['event_digest'],
        compiler_hash=p.main.BINDING['compiler'], fingerprint='f'*64)
        for a,v in [('cosine',cosine),('sine',sine)]}
    return row


class PlanAndSeeds(unittest.TestCase):
    def test_exact_scope(self):
        plan = p.plan()
        self.assertEqual([t['n'] for t in plan['targets']], [1108, 78])
        self.assertEqual([t['PF_cell_id'] for t in plan['targets']], ['B0I_p5_q1_2nd','B0_p7_q8_2nd'])
        self.assertTrue(plan['no_pooling'] and plan['no_optional_stopping'] and plan['no_reselection'])

    def test_scientific_budget_arithmetic(self):
        plan=p.plan(); calls=plan['calls']; targets=plan['targets']
        self.assertEqual(sum(t['n'] for t in targets), calls['trajectory'])
        self.assertEqual(2*calls['trajectory'], calls['wrapper_compile'])
        self.assertEqual(sum(t['n']*t['cell']['q'] for t in targets), calls['occurrence'])
        self.assertEqual(sum(t['n']*t['cell']['R'] for t in targets), calls['RTE_event'])

    def test_no_old_science_calls(self):
        calls=p.plan()['calls']
        for key in ('signal','PF_compile','SCF_DF_state_reference','bootstrap','shot_recalculation','old_sample_aggregation'):
            self.assertEqual(calls[key],0)

    def test_no_resource_caps(self):
        resources=p.plan()['parallel']
        self.assertIsNone(resources['agent_imposed_CPU_RAM_worker_wall_caps'])
        self.assertTrue(resources['PSI_diagnostic_only'])
        self.assertFalse(resources['unrelated_worktree_changes_are_gates'])

    def test_no_population_claim_or_zero_filling(self):
        plan=p.plan()
        self.assertFalse(plan['population_mean_certified'] or plan['actual_power_certified'])
        self.assertIn('unknown',plan['rare_policy'])

    def test_fixed_registry_coverage(self):
        _, reg=fixtures(); self.assertEqual(len(reg['tasks']),1186); self.assertEqual(len(reg['values']),1810)
        self.assertEqual(len(set(reg['values'])),1810)
        for cid,n in p.N.items():
            self.assertEqual({t['replica'] for t in reg['tasks'] if t['cell']['id']==cid},set(range(n)))

    def test_q1_and_q8_hierarchy(self):
        _,reg=fixtures()
        for task in reg['tasks']:
            h=c.seed_hierarchy(task['seed'],task['cell']['q'])
            self.assertEqual(task['step_seeds'],h if task['cell']['q']==1 else h[1:])

    def test_source_changes_every_master(self):
        eps,reg=fixtures(); preps={x['cell']['id']:dict(preparation_hash='p'*64) for x in eps}
        signals={cid:dict(log_B=0.,allowance=dict(log_B_margin=1e-9)) for cid in preps}
        other=p.registry('b'*40,eps,preps,signals)
        self.assertFalse(set(reg['values'])&set(other['values']))

    def test_exploration_namespace_is_different(self):
        eps,reg=fixtures(); t=reg['tasks'][0]
        old=p.main.seed(p.main.SCOPE+'_EXPLORATION','a'*40,c.digest(p.plan()),p.main.BINDING,t['cell']['id'],0)
        self.assertNotEqual(t['seed'],old)

    def test_collision_cannot_redraw(self):
        eps,reg=fixtures(); occupied={reg['tasks'][0]['seed']}
        with self.assertRaisesRegex(ValueError,'COLLISION'):
            c.reserve_seed(occupied,reg['tasks'][0]['seed'],1)
        self.assertFalse(reg['redraw_on_collision'])


class PairingAndAccounting(unittest.TestCase):
    def setUp(self):
        self.eps,self.reg=fixtures(); self.tasks=self.reg['tasks']; self.rows=[pair(t) for t in self.tasks]

    def test_exact_fixed_n_accounting_artificial(self):
        result=science.account(self.rows,self.eps,self.tasks)
        self.assertEqual([r['result']['n'] for r in result['comparisons']],[1108,78])
        self.assertEqual([r['result']['G_exact'] for r in result['comparisons']],['50','50'])
        self.assertEqual(result['actual_worker_calls']['wrapper_compile'],2372)
        self.assertFalse(result['old_samples_pooled'])

    def test_missing_trajectory_rejected(self):
        with self.assertRaisesRegex(ValueError,'EXACT_FIXED_N'): science.validate_pairs(self.rows[:-1],self.tasks)

    def test_duplicate_trajectory_rejected(self):
        self.rows[-1]=deepcopy(self.rows[0])
        with self.assertRaisesRegex(ValueError,'EXACT_FIXED_N'): science.validate_pairs(self.rows,self.tasks)

    def test_exploration_cohort_rejected(self):
        self.rows[0]['cohort']='EXPLORATION_N8'
        with self.assertRaisesRegex(ValueError,'ONLY_NEW_P_COHORT'): science.validate_pairs(self.rows,self.tasks)

    def test_other_seed_rejected(self):
        self.rows[0]['seed']+=1
        with self.assertRaisesRegex(ValueError,'ONLY_NEW_P_COHORT'): science.validate_pairs(self.rows,self.tasks)

    def test_mismatched_axis_seed_rejected(self):
        self.rows[0]['axes']['sine']['seed']+=1
        with self.assertRaisesRegex(ValueError,'UNPAIRED'): science.validate_pairs(self.rows,self.tasks)

    def test_event_mutation_rejected(self):
        self.rows[0]['axes']['sine']['event_digest']='x'*64
        with self.assertRaisesRegex(ValueError,'UNPAIRED'): science.validate_pairs(self.rows,self.tasks)

    def test_compiler_mutation_rejected(self):
        self.rows[0]['axes']['sine']['compiler_hash']='x'*64
        with self.assertRaisesRegex(ValueError,'COMPILE_IDENTITY'): science.validate_pairs(self.rows,self.tasks)

    def test_call_count_mismatch_rejected(self):
        self.rows[0]['calls']['wrapper_compile']=1
        with self.assertRaisesRegex(ValueError,'EXACT_P_TASK_CALLS'): science.validate_pairs(self.rows,self.tasks)

    def test_direction_is_outcome_not_execution_gate(self):
        for ep in self.eps:ep['PF_G_exact']='49'
        result=science.account(self.rows,self.eps,self.tasks)
        self.assertEqual([r['result']['status'] for r in result['comparisons']],['NOT_REPRODUCED']*2)

    def test_tie_is_inconclusive(self):
        for ep in self.eps:ep['PF_G_exact']='50'
        result=science.account(self.rows,self.eps,self.tasks)
        self.assertEqual([r['result']['status'] for r in result['comparisons']],['INCONCLUSIVE']*2)

    def test_unseen_stratum_unknown(self):
        result=science.account(self.rows,self.eps,self.tasks)
        for row in result['comparisons']:
            self.assertIsNone(row['unseen_stratum_cost']);self.assertIsNone(row['unseen_stratum_variance'])
            self.assertEqual(row['strata']['S4_PRESENT'],0)
        self.assertFalse(result['unseen_strata_zero_filled'])

    def test_first_controls_before_remaining(self):
        first=0;phases=[]
        def batch(tasks,callback,spec,directory):
            nonlocal first
            phases.append(directory.name)
            if directory.name=='first_control_workers':
                for task in tasks:
                    c.save(directory/task['id']/'control.json',dict(status='PASS',control_probes=4,independent_event_oracles=1))
                first=len(tasks)
            else:self.assertEqual(first,2)
            return [pair(t) for t in tasks],dict(status='PASS')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c.save(root/'reg.json',self.reg);c.save(root/'eps.json',self.eps)
            spec=dict(seed_registry=str(root/'reg.json'),endpoints=str(root/'eps.json'))
            with patch.object(science,'setup',return_value=NS()),patch('h4_native_science.batch',side_effect=batch):
                result=science.execute(spec,root)
            self.assertEqual(phases,['first_control_workers','remaining_workers'])
            self.assertEqual(result['endpoint_accounting_pass'],1)

    def test_failed_control_does_not_start_remaining(self):
        phases=[]
        def batch(tasks,callback,spec,directory):
            phases.append(directory.name)
            for task in tasks:
                c.save(directory/task['id']/'control.json',dict(status='FAIL',control_probes=4,independent_event_oracles=1))
            return [pair(t) for t in tasks],dict(status='PASS')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.save(root/'reg.json',self.reg)
            with patch.object(science,'setup',return_value=NS()),patch('h4_native_science.batch',side_effect=batch):
                with self.assertRaisesRegex(ValueError,'FIRST_CONTROLS'):
                    science.execute(dict(seed_registry=str(root/'reg.json')),root)
        self.assertEqual(phases,['first_control_workers'])


class Authorization(unittest.TestCase):
    def test_invalid_grant_before_any_input(self):
        with patch.object(c,'read',side_effect=AssertionError('INPUT_LOADED')):
            with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_P'):
                p.authorize({},dict(approved_by_user=False,runtime_authorization=False))

    def test_invalid_runner_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=dict(output=str(root/'out'),lock=str(root/'ONE_SHOT.json'))
            with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_P'):runner.execute(spec,{})
            self.assertEqual(list(root.iterdir()),[])

    def test_invalid_witness_starts_no_process(self):
        with patch.object(runner.subprocess,'Popen',side_effect=AssertionError('LAUNCH')):
            with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_P'):runner.witness({}, {}, NS())

    def test_seal_status_refused(self):
        with self.assertRaisesRegex(ValueError,'P_PENDING_ACTIVATION'):
            p.authorize(dict(status='TECHNICALLY_SEALED_PENDING_EXPLICIT_APPROVAL'),
                dict(approved_by_user=True,runtime_authorization=True,scope=p.SCOPE,stage='P'))

    def test_wrong_scope_refused(self):
        with self.assertRaisesRegex(ValueError,'P_ONLY_GRANT'):
            p.authorize({},dict(approved_by_user=True,runtime_authorization=True,scope='S_STATS',stage='S_STATS'))

    def test_failed_science_keeps_lock_and_stop_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=dict(output=str(root/'output'),lock=str(root/'ONE_SHOT.json'),SOURCE='a'*40)
            with patch.object(p,'authorize'),patch.object(science,'execute',side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError,'synthetic failure'):runner.execute(spec,{})
                self.assertTrue((root/'ONE_SHOT.json').exists());self.assertTrue((root/'output/STOP.json').exists())
                with self.assertRaises(FileExistsError):runner.execute(spec,{})


class SavedRank10Adapter(unittest.TestCase):
    def setup_fixture(self, root, rank=10, prep_hash='p'*64):
        eps,reg=fixtures()
        c.save(root/'config.json',dict(binding=p.main.BINDING))
        c.save(root/'eps.json',eps)
        c.save(root/'preps.json',dict(rows={e['cell']['id']:dict(preparation_hash='p'*64) for e in eps}))
        ham=NS(n_qubits=8,lambdas=[0]*rank);sector=NS(dimension=36)
        load=lambda config:(ham,sector,'synthetic_full','synthetic_state',dict(
            hamiltonian_hash=p.main.BINDING['hamiltonian'],state_hash=p.main.BINDING['state']))
        m=NS(checked_basis_bridge=lambda *a:('indices','synthetic_qstate'),
            _prepare=lambda *a:NS(preparation_hash=prep_hash,hamiltonian_hash=p.main.BINDING['hamiltonian']))
        mods={name:NS() for name in ('pr_cost_native','trottertracks.resource_applicability.ax2b_native_df_v5')}
        mods['trottertracks.resource_applicability.h6_matched_port_v1']=m
        spec=dict(import_paths=[],native_config=str(root/'config.json'),endpoints=str(root/'eps.json'),
            saved_preparations=str(root/'preps.json'))
        return spec,load,mods

    def test_only_saved_rank10_loader_and_two_preparations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,load,mods=self.setup_fixture(root)
            with patch.dict(sys.modules,{'h4_tol_only_input':NS(load=load),
                'trottertracks.resource_applicability.h6_hash_reuse_v1':NS(use_fingerprint_reuse=lambda:None)}),\
                patch('h4_dp_science.compiler',return_value='synthetic_compiler'),\
                patch.object(science.importlib,'import_module',side_effect=lambda name:mods[name]):
                context=science.setup(spec,root)
            self.assertEqual(sorted(context.preparations),[2,3])
            self.assertEqual(c.read(root/'preparations.json')['reconstructions'],2)

    def test_rank12_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,load,mods=self.setup_fixture(root,rank=12)
            with patch.dict(sys.modules,{'h4_tol_only_input':NS(load=load),
                'trottertracks.resource_applicability.h6_hash_reuse_v1':NS(use_fingerprint_reuse=lambda:None)}),\
                patch.object(science.importlib,'import_module',side_effect=lambda name:mods[name]):
                with self.assertRaisesRegex(ValueError,'DF10_SAVED_CONTEXT'):science.setup(spec,root)

    def test_saved_preparation_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,load,mods=self.setup_fixture(root,prep_hash='x'*64)
            with patch.dict(sys.modules,{'h4_tol_only_input':NS(load=load),
                'trottertracks.resource_applicability.h6_hash_reuse_v1':NS(use_fingerprint_reuse=lambda:None)}),\
                patch.object(science.importlib,'import_module',side_effect=lambda name:mods[name]):
                with self.assertRaisesRegex(ValueError,'SAVED_A_PREPARATION_IDENTITY'):science.setup(spec,root)


class CompletionAudit(unittest.TestCase):
    def artificial_audit(self, mutate=None):
        eps,reg=fixtures();tasks=reg['tasks'];rows=[pair(t) for t in tasks]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'run01/output';run=out.parent
            spec=dict(output=str(out),lock=str(run/'ONE_SHOT.json'),seed_registry=str(root/'reg.json'),
                endpoints=str(root/'eps.json'),SOURCE='a'*40,status='SEALED_FOR_EXECUTION',pending=[])
            digest=c.digest(spec)
            data={str(root/'reg.json'):reg,str(root/'eps.json'):eps,
                str(run/'ONE_SHOT.json'):dict(contract_digest=digest)}
            receipts={name:dict(status='PASS',workers=[],all_reaped=True,cleanup_errors=[]) for name in
                ('first_control_workers','remaining_workers')}
            for task,row in zip(tasks,rows):
                phase='first_control_workers' if task['replica']==0 else 'remaining_workers'
                work=out/phase/task['id']; events=[[dict(taylor_order=0)]*task['cell']['r'] for _ in range(task['cell']['q'])]
                event_digest=c.digest(events);row['event_digest']=event_digest
                for axis in c.AXES:row['axes'][axis]['event_digest']=event_digest
                data[str(work/'task_result.json')]=row
                data[str(work/'trajectory.json')]=dict(cell=task['cell'],seed=task['seed'],step_seeds=task['step_seeds'],
                    conditional=False,quantum_weights_changed=False,cohort=p.COHORT,event_digest=event_digest,events=events)
                data[str(work/'calls_ledger.json')]=dict(nonrefundable=True,reserved=row['calls'])
                for axis in c.AXES:data[str(work/('measured_'+axis+'.json'))]=row['axes'][axis]
                if task['replica']==0:data[str(work/'control.json')]=dict(status='PASS',max_error=0.,tolerance=1e-9,
                    control_probes=4,independent_event_oracles=1)
                receipts[phase]['workers'].append(dict(started=1 if task['replica']==0 else 2))
            result=science.account(rows,eps,tasks)
            result.update(SOURCE=spec['SOURCE'],contract_digest=digest,status='COMPLETE_MANDATORY_STOP')
            data.update({str(run/'native_exit.json'):dict(exit_code=0,same_parent_reaped=True,result_sha256='h'*64,SOURCE=spec['SOURCE']),
                str(out/'terminal.json'):dict(status='COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],result_sha256='h'*64,
                    own_workers_reaped=True,final_closure_checked=True,automatic_retry=False,next_stage_authorized=False),
                str(out/'result.json'):result,str(out/'paired_measurements.json'):dict(rows=rows,old_cohorts_not_pooled=True),
                str(out/'preparations.json'):dict(reconstructions=2),
                str(out/'endpoint_accounting_reserved.json'):dict(passes=1,comparisons=2,nonrefundable=True),
                str(root/'prior_preservation_manifest_PRIVATE.json'):[],str(root/'technical_seal_preservation_manifest_PRIVATE.json'):[]})
            for phase,receipt in receipts.items():data[str(out/phase/'accepted_native_receipt.json')]=receipt
            if mutate:mutate(data,out,run,result)
            grant=dict(approved_by_user=True,runtime_authorization=True,one_shot=True,automatic_retry=False,
                next_stage_authorized=False,scope=p.SCOPE,stage='P',approval_receipt='synthetic_receipt',
                approval_receipt_sha256='h'*64,contract_digest=digest)
            with patch.object(c,'read',side_effect=lambda path:data[str(path)]),patch.object(c,'sha',return_value='h'*64),\
                patch.object(p,'verify_bytes'),patch.object(completed,'owned_receipt',side_effect=lambda directory,expected:list(expected)):
                return completed.audit(spec,grant)

    def test_complete_fixed_n_native_receipt_schema(self):
        result=self.artificial_audit();self.assertEqual(result['workers_reaped'],1186)
        self.assertFalse(result['numeric_statistics_recomputed'])

    def test_nonzero_native_exit_rejected(self):
        with self.assertRaisesRegex(ValueError,'NATIVE_EXIT_REAP_SHA'):
            self.artificial_audit(lambda data,out,run,result:data[str(run/'native_exit.json')].update(exit_code=1))

    def test_pooled_result_rejected(self):
        with self.assertRaisesRegex(ValueError,'SCOPE_LIMITS'):
            self.artificial_audit(lambda data,out,run,result:result.update(old_samples_pooled=True))


if __name__=='__main__':unittest.main()
