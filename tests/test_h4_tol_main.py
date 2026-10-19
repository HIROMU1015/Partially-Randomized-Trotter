"""Synthetic fixtures only: no molecular arrays, real costs, or science workers."""
from copy import deepcopy
import importlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_tol_main_contract as c
import h4_tol_main_science as science
import run_h4_tol_main as runner


def record(cell, eligible=True):
    ep = dict(cost_acquisition_eligible=eligible,
        eligibility_under_declared_allowance='ELIGIBLE' if eligible else 'INELIGIBLE',
        axes={a: dict(shots=1 if eligible else None) for a in ('real', 'imag')})
    return dict(cell=deepcopy(cell), precision_rows={str(e): {'1.0': deepcopy(ep)} for e in c.EPS},
        log_B=0., allowance=dict(log_B_margin=1e-10))


def pair(replica, cosine=1, sine=2):
    return dict(replica=replica, seed=replica+1, event_digest='e'*64, cohort='EXPLORATION_N8',
        axes={a: dict(RZ=n, seed=replica+1, event_digest='e'*64,
            compiler_hash=c.BINDING['compiler'], fingerprint='f'*64)
            for a, n in [('cosine', cosine), ('sine', sine)]})


def saved_rows(eligible=True):
    return [dict(cell_id=c.FP_PR, epsilon=e, bias_mode='USUAL_SIGNED', allowance_factor=1.,
        eligible=eligible, G_RZ_exact='30' if eligible else None,
        G_RZ_SE=2.5 if eligible else None, n=32 if eligible else None,
        shots={'cosine': 1, 'sine': 1}) for e in c.EPS]


class Metadata(unittest.TestCase):
    def test_primitive_budget_covers_actual_signed_and_yoshida_time_schedule(self):
        sys.path.insert(0,str(c.ROOT/'src'))
        from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
        for stage in ('D','A'):
            cells=[x for x in c.registrations()[stage] if x['method']!='B0I']
            pairs={('one' if i==0 else str(i-1),t) for x in cells
                for i,t in primitive_time_schedule(x,T=.8)['unique_primitive_times']}
            self.assertEqual(3*len(pairs),c.budgets()[stage]['primitive_probe_maximum'])

    def test_frozen_a_event_oracle_project_import_is_bound(self):
        closure=c.source_closure()
        self.assertIn('scripts/h6_dba_science.py',closure)
        self.assertIn('scripts/h6_dba_native.py',closure)
        self.assertIn('scripts/h6_dba_signal.py',closure)

    def test_rank10_grid_and_full_rule(self):
        reg = c.registrations()
        self.assertEqual((len(reg['D']), len(reg['A'])), (190, 50))
        self.assertEqual({x['prefix'] for x in reg['D'] if x['method']=='B1'}, {10})
        self.assertEqual({x['prefix'] for x in reg['D'] if x['method']=='B0I'}, set(range(1, 10)))
        self.assertEqual({x['prefix'] for x in reg['A']}, {1, 2, 3})
        self.assertFalse(reg['result_dependent_grid'])

    def test_reuse_and_new_call_budgets_are_separate(self):
        b = c.budgets()
        self.assertEqual((b['D']['saved_axes'], b['A']['reused_FP_axes']), (8, 64))
        self.assertEqual((b['A']['trajectory_maximum'], b['A']['wrapper_compile_maximum']), (392, 784))
        self.assertEqual(b['A']['reused_FP_anchor_n'], 32)
        self.assertEqual(b['D']['new_full_H_reference'], 0)
        self.assertEqual(b['D']['exact_truncated_reference_maximum'], 88)

    def test_p_n_source_seeds_are_pending(self):
        p = c.plan('P')['P_contract']
        for k in ('n', 'SOURCE', 'seeds', 'candidates', 'PF'): self.assertIsNone(p[k])
        self.assertEqual(p['primary_epsilons'], [.05, .0005])
        self.assertFalse(c.budgets()['S_STATS']['rank12_variance_transferred'])

    def test_historical_resource_caps_not_inherited(self):
        r = c.plan('D')['resources']
        self.assertIsNone(r['agent_imposed_CPU_RAM_worker_wall_caps'])
        self.assertTrue(r['PSI_diagnostic_only'])
        self.assertFalse(r['unrelated_worktree_changes_are_gates'])

    def test_rank_scalar_compiler_binding_mutations_rejected(self):
        for key, value in [('rank', 12), ('scalar', 'twice'), ('compiler', 'f'*64), ('state', 'b'*64)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'BINDING'):
                c.same_binding(dict(c.BINDING, **{key: value}))

    def test_complete_signals_and_no_unknown_exclusion(self):
        x = c.registrations()['A'][0]; r = record(x, False)
        self.assertEqual(c.selected([r], [x]), [])
        r['precision_rows']['0.05']['1.0']['eligibility_under_declared_allowance']='UNRESOLVED'
        with self.assertRaisesRegex(ValueError, 'UNRESOLVED'): c.selected([r], [x])

    def test_missing_signal_and_shot_are_errors(self):
        x = c.registrations()['D'][0]; r = record(x)
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SIGNAL'): c.selected([], [x])
        r['precision_rows']['0.05']['1.0']['axes']['imag']['shots']=None
        with self.assertRaisesRegex(ValueError, 'SHOTS'): c.selected([r], [x])

    def test_duplicate_grid_does_not_pass(self):
        x = c.registrations()['D'][0]
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SIGNAL'): c.selected([record(x)]*2, [x]*2)

    def reuse(self):
        x = next(x for x in c.registrations()['A'] if x['id']==c.FP_PR)
        row = dict(cell_id=x['id'], replica=0, rows=[dict(axis=a, metrics=dict(RZ=5),
            compiler_hash=c.BINDING['compiler'], fingerprint='f'*64) for a in c.old.AXES])
        return x, row

    def test_saved_compile_identity_and_cohort(self):
        x, row = self.reuse()
        out = c.reuse_measurement(row, c.BINDING, x, accepted_result_sha256='a'*64, source='b'*40)
        self.assertEqual(out['cohort'], 'SAVED_FP_PANEL_N32_ANCHOR')
        self.assertFalse(out['newly_compiled'])

    def test_wrong_saved_axis_candidate_fingerprint_and_compiler_rejected(self):
        for change in ('axis', 'candidate', 'fingerprint', 'compiler'):
            x, row = self.reuse()
            if change=='axis': row['rows'][1]['axis']='cosine'
            if change=='candidate': row['cell_id']='other'
            if change=='fingerprint': row['rows'][0]['fingerprint']=''
            if change=='compiler': row['rows'][0]['compiler_hash']='a'*64
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.reuse_measurement(row, c.BINDING, x, accepted_result_sha256='a'*64, source='b'*40)

    def test_seed_namespace_source_binding_and_q_hierarchy(self):
        args = [c.SCOPE+'_EXPLORATION', 'a'*40, 'p'*64, c.BINDING, 'candidate', 0]
        a = c.seed(*args); args[1]='b'*40
        self.assertNotEqual(a, c.seed(*args))
        self.assertEqual(c.old.seed_hierarchy(a, 1), [a])
        self.assertEqual(len(c.old.seed_hierarchy(a, 4)), 5)
        with self.assertRaisesRegex(ValueError, 'SEED_SCOPE'): c.seed('other', *args[1:])

    def test_seed_registry_collision_stops(self):
        plan = c.plan('A'); source='a'*40
        tasks=[]
        for x in c.registrations()['A']:
            if x['id']==c.FP_PR: continue
            for i in range(8):
                seed=c.seed(c.SCOPE+'_EXPLORATION', source, c.digest(plan), c.BINDING, x['id'], i)
                hierarchy=c.old.seed_hierarchy(seed, x['q'])
                tasks.append(dict(cell=x, replica=i, seed=seed, step_seeds=hierarchy if x['q']==1 else hierarchy[1:]))
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); c.save(d/'registry', dict(SOURCE=source, plan_digest=c.digest(plan), binding=c.BINDING, tasks=tasks))
            c.save(d/'old', dict(values=[tasks[0]['seed']]))
            with self.assertRaisesRegex(ValueError, 'SEED_COLLISION'):
                c.validate_seed_registry(dict(SOURCE=source, plan=plan, seed_registry=str(d/'registry'), old_seed_exclusions=str(d/'old')))


class Accounting(unittest.TestCase):
    def test_saved_fp_mean_se_n_copied_without_raw_aggregation(self):
        x=next(x for x in c.registrations()['A'] if x['id']==c.FP_PR)
        with patch.object(c.old, 'evaluate_pairs', side_effect=AssertionError('no raw recomputation')):
            result=c.account_stage([record(x)], {c.FP_PR: [object()]}, saved_FP_rows=saved_rows())
        self.assertEqual([(r['G_RZ_exact'], r['G_RZ_SE'], r['n']) for r in result['rows']], [('30', 2.5, 32)]*5)
        self.assertFalse(any(r['statistics_recomputed'] for r in result['rows']))
        self.assertEqual(result['new_cohort_covariance'], [])

    def test_missing_saved_fp_stats_blocked(self):
        x=next(x for x in c.registrations()['A'] if x['id']==c.FP_PR); rows=saved_rows(); rows[0]['G_RZ_SE']=None
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SAVED_FP'): c.account_stage([record(x)], {}, saved_FP_rows=rows)

    def test_paired_anticorrelation_and_covariance(self):
        x=c.registrations()['A'][0]; values=[pair(i, i+1, 8-i) for i in range(8)]
        out=c.account_stage([record(x)], {x['id']: values})
        self.assertEqual(out['rows'][0]['G_RZ_exact'], '9')
        self.assertEqual(out['rows'][0]['G_RZ_SE'], 0)
        self.assertEqual(out['new_cohort_covariance'][0]['sample_covariance_exact'], [['0']*5]*5)

    def test_n32_cannot_be_new_n8(self):
        x=c.registrations()['A'][0]
        with self.assertRaisesRegex(ValueError, 'COHORT'): c.account_stage([record(x)], {x['id']: [pair(i) for i in range(32)]})

    def test_duplicate_new_replicas_and_unpaired_axes_rejected(self):
        x=c.registrations()['A'][0]
        values=[pair(i) for i in range(8)]; values[-1]['replica']=0
        with self.assertRaisesRegex(ValueError, 'REPLICAS'): c.account_stage([record(x)], {x['id']: values})
        values=[pair(i) for i in range(8)]; values[0]['axes']['sine']['seed']=100
        with self.assertRaisesRegex(ValueError, 'UNPAIRED'): c.account_stage([record(x)], {x['id']: values})

    def test_missing_eligible_cost_blocks_selection(self):
        x=c.registrations()['D'][0]
        with self.assertRaisesRegex(ValueError, 'MISSING_ELIGIBLE_COST'): c.account_stage([record(x)], {})

    def test_exact_cost_and_lexical_tie(self):
        a,b=c.registrations()['D'][:2]
        out=c.account_stage([record(b), record(a)], {x['id']: [dict(axes={z: dict(RZ=2**54+1) for z in c.old.AXES})] for x in (a,b)})
        self.assertEqual(out['winners'][0]['best']['cell_id'], min(a['id'],b['id']))
        self.assertEqual(out['winners'][0]['best']['G_RZ_exact'], str(2*(2**54+1)))

    def test_ineligible_is_absent_and_not_zero_cost(self):
        out=c.account_stage([record(c.registrations()['D'][0], False)], {})
        self.assertTrue(all(x['best'] is None for x in out['winners']))


class GatesAndAdapters(unittest.TestCase):
    def artificial_seal(self, tmp, stage='D'):
        source_hash=hashlib.sha256(b'synthetic source').hexdigest()
        contract=dict(stage=stage,status='SEALED_FOR_EXECUTION',pending=[],plan=c.plan(stage),SOURCE='a'*40,
            binding=c.BINDING,source_hashes={'scripts/synthetic.py':source_hash},
            inputs=[dict(path='synthetic_input',sha256='input_hash',role='synthetic')],
            runtime_hashes={'synthetic_runtime':'runtime_hash'},provider_hashes={'synthetic_provider':'provider_hash'},
            native_config='synthetic_input', FP_evidence='synthetic_input',saved_reference='synthetic_input',
            font_seed=dict(path='synthetic_input'),
            python_sha256='python_hash',fresh_resources=dict(admitted=True,receipt='synthetic_resources',sha256='resource_hash'),
            worker_cpus=[min(os.sched_getaffinity(0))],output=str(tmp/'output'),lock=str(tmp/'lock'))
        grant=dict(approved_by_user=True,runtime_authorization=True,scope=c.SCOPE,stage=stage,one_shot=True,
            automatic_retry=False,next_stage_authorized=False,contract_digest=c.digest(contract),SOURCE='a'*40,
            approval_receipt='synthetic_approval',approval_receipt_sha256='approval_hash')
        mapping={'synthetic.py':source_hash,'synthetic_input':'input_hash','synthetic_runtime':'runtime_hash',
            'synthetic_provider':'provider_hash',Path(sys.executable).name:'python_hash',
            'synthetic_resources':'resource_hash','synthetic_approval':'approval_hash'}
        return contract,grant,mapping

    def test_sealed_source_input_provider_runtime_mutations_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract,grant,mapping=self.artificial_seal(Path(tmp))
            for changed,reason in [('synthetic.py','SOURCE_CHANGED'),('synthetic_input','INPUT_CHANGED'),
                                   ('synthetic_provider','provider_hashes_CHANGED'),('synthetic_runtime','runtime_hashes_CHANGED')]:
                hashes=dict(mapping); hashes[changed]='changed'
                with self.subTest(changed=changed), patch.object(c,'sha',side_effect=lambda p:hashes[Path(p).name]), \
                     patch.object(c,'verify_provider_tree'), \
                     patch.object(c,'source_closure',return_value=contract['source_hashes']), \
                     patch.object(c.subprocess,'check_output',return_value=b'synthetic source'):
                    with self.assertRaisesRegex(ValueError,reason): c.authorize(contract,grant)

    def test_used_attempt_and_pending_a_or_p_refuse_even_with_mock_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for stage,reason in [('A','PENDING_D_RESULT'),('P','PENDING_D_A_RESULT')]:
                contract,grant,mapping=self.artificial_seal(tmp,stage)
                with self.subTest(stage=stage), patch.object(c,'sha',side_effect=lambda p:mapping[Path(p).name]), \
                     patch.object(c,'verify_provider_tree'), \
                     patch.object(c,'source_closure',return_value=contract['source_hashes']), \
                     patch.object(c.subprocess,'check_output',return_value=b'synthetic source'):
                    with self.assertRaisesRegex(ValueError,reason): c.authorize(contract,grant)
            contract,grant,mapping=self.artificial_seal(tmp); (tmp/'lock').write_text('prior claim')
            with patch.object(c,'sha',side_effect=lambda p:mapping[Path(p).name]), \
                 patch.object(c,'verify_provider_tree'), \
                 patch.object(c,'source_closure',return_value=contract['source_hashes']), \
                 patch.object(c.subprocess,'check_output',return_value=b'synthetic source'):
                with self.assertRaisesRegex(ValueError,'USED_ATTEMPT'): c.authorize(contract,grant)

    def test_untracked_provider_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); (tmp/'approved.py').write_text('synthetic')
            contract=dict(provider_roots=[str(tmp)],provider_hashes={str(tmp/'approved.py'):c.sha(tmp/'approved.py')})
            c.verify_provider_tree(contract)
            (tmp/'unregistered.py').write_text('synthetic')
            with self.assertRaisesRegex(ValueError,'COMPLETE_PROVIDER_FILE_SET'): c.verify_provider_tree(contract)

    def test_invalid_grant_refuses_before_file_provider_or_output_access(self):
        for stage in ('D','A','P','S_STATS'):
            with self.subTest(stage=stage), patch.object(c, 'sha', side_effect=AssertionError('input touched')):
                with self.assertRaisesRegex(ValueError, 'SEPARATE_EXPLICIT'):
                    runner.execute(dict(stage=stage), c.invalid_grant(stage))

    def test_all_direct_and_witness_cli_paths_refuse_without_import_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            # A Python audit hook would expose an attempted import before rejection.
            (tmp/'sitecustomize.py').write_text("import sys\ndef hook(event,args):\n if event=='import' and (args[0].startswith(('numpy','qiskit','trottertracks')) or args[0]=='h4_tol_main_science'): raise AssertionError('SCIENCE_IMPORT')\nsys.addaudithook(hook)\n")
            for stage in ('D','A','P','S_STATS'):
                c.save(tmp/'contract', dict(stage=stage, output=str(tmp/'output'), lock=str(tmp/'lock')))
                c.save(tmp/'grant', c.invalid_grant(stage))
                for witness in (False, True):
                    command=[sys.executable, str(c.ROOT/'scripts/run_h4_tol_main.py'), '--stage', stage,
                        '--contract', str(tmp/'contract'), '--grant', str(tmp/'grant')]+(['--witness'] if witness else [])
                    result=subprocess.run(command, capture_output=True, text=True,
                        env=dict(os.environ, PYTHONPATH=str(tmp), PYTHONDONTWRITEBYTECODE='1'))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED', result.stderr)
                    self.assertNotIn('SCIENCE_IMPORT', result.stderr)
                    self.assertFalse((tmp/'output').exists()); self.assertFalse((tmp/'lock').exists())
                    self.assertFalse((tmp/'stdout.log').exists())

    def test_stop_retains_claim_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); contract=dict(SOURCE='a'*40, output=str(tmp/'output'), lock=str(tmp/'lock'))
            with patch.object(c, 'authorize'), patch.object(runner, 'run_claimed', side_effect=RuntimeError('synthetic failure')) as run:
                with self.assertRaisesRegex(RuntimeError, 'synthetic'): runner.execute(contract, {})
            self.assertEqual(run.call_count, 1); self.assertTrue((tmp/'lock').exists())
            self.assertTrue(c.read(tmp/'output/STOP.json')['reservations_retained'])

    def test_compile_adapter_delegates_to_original_h4_function(self):
        sentinel=object(); original=NS(compile_task=lambda *a: sentinel)
        with patch.dict(sys.modules, h4_followup_science=original):
            self.assertIs(science.h_compile('port', 'config', 'task', 'directory'), sentinel)

    def test_setup_reuses_reference_and_sets_counter_before_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); c.save(tmp/'config', dict(provider_root='synthetic'))
            c.save(tmp/'ref', dict(T=.8, reference=dict(real=1., imag=0.), error=1e-12,
                metadata=dict(hamiltonian_hash=c.BINDING['hamiltonian'], state_hash=c.BINDING['state'])))
            class Port:
                def __init__(self, root, manifest, writer, progress):
                    self.plan=manifest['plan']; self.preps={}
                    self.primitive_limit=self.plan['caps_proposed']['primitive']
                def actions(self, *args, **kwargs): pass
                def primitives(self):
                    self.checked=self.primitive_limit==3
            prep=NS(preparation_hash='p', hamiltonian_hash='h')
            m=NS(MatchedPort=Port, _prepare_discard=lambda *x: prep, _prepare=lambda *x: prep,
                checked_basis_bridge=lambda *x: ('indices','qstate'),
                primitive_time_schedule=lambda *a, **k: dict(unique_primitive_times=[(0,.1)], ordinary_one_outer_step=[(0,.1)]))
            h=NS(provider=lambda: m, Progress=lambda: None, accounting_plan=lambda: dict(
                numerical={}, matrix_free={}, coverage={}, caps_proposed=dict(tail_matvecs_corrected_and_raw={})))
            loader=NS(load=lambda x: (NS(n_qubits=8,lambdas=[0]*10), NS(dimension=36), [], [],
                dict(hamiltonian_hash=c.BINDING['hamiltonian'], state_hash=c.BINDING['state'])))
            with patch.dict(sys.modules, h4_native_science=h, h4_tol_only_input=loader):
                p,_,_,_=science.setup(dict(stage='D', native_config=str(tmp/'config'), import_paths=[], saved_reference=str(tmp/'ref')),
                    c.registrations()['D'][:1], tmp)
            self.assertTrue(p.checked); self.assertEqual(p.qindices, 'indices')
            self.assertEqual(p.reference, 1+0j); self.assertEqual(p.plan['caps_proposed']['reference_matvec'], 0)

    def test_mock_d_complete_grid_without_any_eligible_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); reg=c.registrations()['D']; byid={x['id']:x for x in reg}
            c.save(tmp/'fp', dict(signals=[record(byid[cid],False) for cid in c.FP_SIGNALS_D],
                preparation_identities={}, measurements=[]))
            def batch(contract,tasks,callback,directory):
                self.assertEqual(len(tasks), 96)
                return [record(t['cell'],False) for t in tasks]
            def overlay(r,*args):
                x=dict(r['cell'],method='B0I',id=r['cell']['id'].replace('B0_','B0I_',1)); return record(x,False)
            with patch.object(science,'setup',return_value=(NS(preps={}),{},None,None)), patch.object(science,'batch',side_effect=lambda *a: batch(*a) if a[1] else []), patch.object(science,'overlay',side_effect=overlay):
                result=science.execute(dict(stage='D',FP_evidence=str(tmp/'fp'),SOURCE='a'*40),tmp)
            self.assertTrue(result['registered_set_complete']); self.assertEqual(len(result['rows']),950)
            self.assertEqual(result['actual_new_wrapper_compiles'],0)

    def test_mock_a_controls_finish_before_remaining_paired_replicas(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); reg=c.registrations()['A']; fp=next(x for x in reg if x['id']==c.FP_PR)
            c.save(tmp/'fp', dict(signals=[record(fp)],preparation_identities={},saved_rows=saved_rows()))
            tasks=[dict(cell=x,replica=i,seed=i+1,id=x['id']+'_'+str(i)) for x in reg if x!=fp for i in range(8)]
            c.save(tmp/'registry',dict(tasks=tasks)); c.save(tmp/'d',dict(binding=c.BINDING,registered_set_complete=True,
                winners=[dict(epsilon=e,best=dict(G_RZ_exact='1000')) for e in c.EPS]))
            calls=[]
            def batch(contract,tasks,callback,directory):
                calls.append((directory.name,len(tasks)))
                if directory.name=='signal_workers': return [record(t['cell']) for t in tasks]
                out=[]
                for task in tasks:
                    directory_=directory/task['id']; directory_.mkdir(parents=True)
                    out.append(callback(task,directory_))
                return out
            def compile_(ctx,t,d):
                c.save(d/'trajectory.json',dict(events=[[]]))
                return dict(pair(t['replica']),cell_id=t['cell']['id'])
            port=NS(qstate='synthetic',preps={x['id']:NS(preparation_hash='p') for x in reg})
            mods={'pr_cost_native':NS(compile_task=compile_), 'h4_dp_science':NS(compiler=lambda *x: 'compiler'),
                'trottertracks.resource_applicability.ax2b_native_df_v5':NS()}
            contract=dict(stage='A', FP_evidence=str(tmp/'fp'), SOURCE='a'*40, seed_registry=str(tmp/'registry'),
                accepted_D=dict(path=str(tmp/'d'),sha256=c.sha(tmp/'d')))
            with patch.object(science,'setup',return_value=(port,{},None,None)), patch.object(science,'batch',side_effect=batch), patch.dict(sys.modules,mods):
                out=science.execute(contract,tmp)
            self.assertEqual(calls,[('signal_workers',49),('first_control_workers',49),('remaining_workers',343)])
            self.assertEqual(out['actual_new_wrapper_compiles'],784)
            self.assertEqual(len(out['new_cohort_covariance']),49)
            self.assertFalse(out['saved_FP_raw_aggregation'])


if __name__=='__main__': unittest.main()
