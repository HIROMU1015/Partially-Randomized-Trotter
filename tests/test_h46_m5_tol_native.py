"""Artificial contracts, saved-input toys and process tests; no molecular files."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import h46_m5_contract as old_c
import h46_m5_native as old_n
import h46_m5_tol_contract as c
import h46_m5_tol_native as n
import launch_h46_m5_tol_pilot as launch


def fake_reference(lane):
    return {'status': 'PASS'}


def fake_worker(task):
    cfg = launch.CONFIGS[task['lane']]
    if cfg.get('simulate_failure'):
        return dict(status='STOP', lane=task['lane'], task=task['id'], reason='ARTIFICIAL_FAILURE')
    for axis in ('cosine', 'sine'):
        (Path(cfg['ledger'])/(task['id']+'__'+axis+'.json')).write_text('{}')
    return dict(status='PASS', lane=task['lane'], task=task['id'])


class MetadataTests(unittest.TestCase):
    def test_old_module_globals_and_numeric_code_unchanged(self):
        self.assertEqual(old_c.core.SYSTEMS['H4']['rank'], 12)
        self.assertEqual(c.core.SYSTEMS['H4']['rank'], 10)
        self.assertEqual(c.core.SYSTEMS['H6']['rank'], 19)
        self.assertNotEqual(c.LANES, old_c.LANES)
        for name in ('compose', 'make_requests', 'signal', '_run_task', 'preflight'):
            self.assertIs(getattr(n, name).__code__, getattr(old_n, name).__code__)
            self.assertIs(getattr(n, name).__globals__['c'], c)
            self.assertIs(getattr(old_n, name).__globals__['c'], old_c)

    def test_scopes_and_series_ids(self):
        self.assertEqual(sum(c.scope(l)['compile_axes'] for l in c.LANES), 60)
        self.assertEqual(sum(c.scope(l)['reference_jobs'] for l in c.LANES), 2)
        groups = [g for l in c.LANES for g in c.scope(l)['groups']]
        self.assertTrue(all('TOL_ONLY' in g['input_series'] for g in groups))
        self.assertTrue(all(g['prefix'] == (10 if g['system']=='H4' else 19)
                            for g in groups if g['method']=='FULL'))
        self.assertTrue(all(c.scope(l)['shot_accounting'] is False for l in c.LANES))

    def test_seed_namespace_and_old_pilot_exclusion(self):
        old = old_c.registry('a'*40, set())
        excluded = {v for lane in old.values() for task in lane.values() for v in task.values()}
        new = c.registry('a'*40, excluded)
        values = [v for lane in new.values() for task in lane.values() for v in task.values()]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(len(values), 120)
        self.assertFalse(set(values) & excluded)
        with self.assertRaisesRegex(ValueError, 'COLLISION'):
            c.registry('a'*40, {values[0]})

    def test_loader_function_bodies_match_fixed_original(self):
        original = subprocess.check_output(['git', 'show',
            '97d25cc431516d720ce91c7cb4f148e061d8ea34:scripts/h4_tol_only_input.py'], cwd=ROOT, text=True)
        fixed = (ROOT/'scripts/h46_m5_tol_saved_input.py').read_text()
        functions = lambda s: {x.name: ast.dump(x, include_attributes=False)
                               for x in ast.parse(s).body if isinstance(x, ast.FunctionDef)}
        for name, node in functions(fixed).items():
            self.assertEqual(node, functions(original)[name])
        self.assertEqual(set(functions(fixed)), {'array_record', 'load'})

    def test_draft_refused_before_hash_or_science(self):
        with patch.object(c, 'verify_seals', side_effect=AssertionError('must refuse first')):
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_PILOT'):
                c.authorize(dict(authorized=False, grant={'valid':False}))

    def test_cli_invalid_bundle_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'config.json'
            config.write_text(json.dumps({'authorized':False, 'grant':{'valid':False}}))
            bundle = root/'bundle.json'
            bundle.write_text(json.dumps({'configs':{l:str(config) for l in c.LANES}}))
            before = set(root.iterdir())
            p = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/launch_h46_m5_tol_pilot.py'),
                                '--execute', '--bundle', str(bundle)], capture_output=True, text=True)
            # The lane identity is also fail-closed if absent, before any science.
            self.assertEqual(p.returncode, 2)
            self.assertEqual(before, set(root.iterdir()))
            self.assertIn('REFUSED', p.stdout)

    def test_FP_binding_not_legacy_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(system='H4', binding=copy.deepcopy(c.TARGETS['H4']['binding']))
            good = dict(status='COMPLETE_MANDATORY_STOP', sha256=cfg['binding']['snapshot'],
                hamiltonian_hash=cfg['binding']['hamiltonian'], state_hash=cfg['binding']['state'], rank=10,
                hamiltonian_metadata=dict(input_policy='TOL_ONLY_NO_CONFIG_FALLBACK',
                    df_tol_requested=1e-8, final_rank_supplied=False,
                    coefficient_order='decomposer_generation_order', subsequent_coefficient_cutoff=0))
            receipt = Path(tmp)/'receipt.json'
            receipt.write_text(json.dumps(good))
            cfg['inputs'] = {'H4_tol_snapshot.npz':{'sha256':cfg['binding']['snapshot']},
                             'H4_tol_receipt.json':{'path':str(receipt)}}
            c.validate_inputs(cfg)
            cfg['binding']['rank'] = 12
            with self.assertRaisesRegex(ValueError, 'FIXED_TOL_SERIES'):
                c.validate_inputs(cfg)

    def test_rank_fixed_generation_policy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            binding = copy.deepcopy(c.TARGETS['H4']['binding'])
            receipt = dict(status='COMPLETE_MANDATORY_STOP', sha256=binding['snapshot'],
                hamiltonian_hash=binding['hamiltonian'], state_hash=binding['state'], rank=10,
                hamiltonian_metadata=dict(input_policy='TOL_ONLY_NO_CONFIG_FALLBACK',
                    df_tol_requested=1e-8, final_rank_supplied=True,
                    coefficient_order='decomposer_generation_order', subsequent_coefficient_cutoff=0))
            path = Path(tmp)/'receipt.json';path.write_text(json.dumps(receipt))
            cfg = dict(system='H4', binding=binding, inputs={
                'H4_tol_snapshot.npz':{'sha256':binding['snapshot']}, 'H4_tol_receipt.json':{'path':str(path)}})
            with self.assertRaisesRegex(ValueError, 'TOL_POLICY'):
                c.validate_inputs(cfg)

    def test_original_contract_cannot_be_used_as_tol_lane(self):
        with self.assertRaisesRegex(ValueError, 'FIXED_TOL_FOUR_LANES'):
            c.scope('H4_P2_DET')

    def test_ledger_is_new_scope_and_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope = c.scope(c.LANES[0]);task = scope['tasks'][0]
            cfg = dict(ledger=tmp, plan={'scope':scope}, SOURCE='a'*40, plan_sha256='b'*64)
            c.reserve(cfg, task, 'cosine', 'fp')
            with self.assertRaises(FileExistsError):
                c.reserve(cfg, task, 'cosine', 'fp')
            with self.assertRaises(ValueError):
                c.reserve(cfg, old_c.scope(old_c.LANES[0])['tasks'][0], 'sine', 'fp')

    def test_used_input_path_must_belong_to_sealed_inventory(self):
        cfg = dict(sealed_files=[dict(path='/tmp/frozen_input', sha256='a'*64, bytes=4)],
                   inputs={'snapshot':dict(path='/tmp/other_input', sha256='a'*64, bytes=4)})
        with patch.object(c, '_verify_original', side_effect=AssertionError('reject before other reads')):
            with self.assertRaisesRegex(ValueError, 'USED_INPUT_PATH_MUST_BE_SEALED'):
                c.verify_seals(cfg)

    def test_duplicate_sealed_path_rejected(self):
        cfg = dict(sealed_files=[dict(path='/tmp/frozen_input'), dict(path='/tmp/frozen_input')])
        with self.assertRaisesRegex(ValueError, 'UNIQUE_SEALED_PATHS'):
            c.verify_seals(cfg)


class ProcessTests(unittest.TestCase):
    def artificial_pool(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            configs = {}
            for lane in c.LANES:
                out, ledger = Path(tmp)/(lane+'_out'), Path(tmp)/(lane+'_ledger')
                out.mkdir();ledger.mkdir()
                configs[lane] = dict(lane=lane, system=lane[:2], SOURCE='a'*40,
                    plan={'scope':c.scope(lane)}, run_root=tmp, output=str(out), ledger=str(ledger),
                    RSS_proxy_bytes=1, simulate_failure=lane==failure)
            with patch.object(c, 'verify_seals'), patch.object(launch, 'observe', return_value={'PSI':'diagnostic_only'}):
                return launch.run_pool(configs, {'workers':4}, worker=fake_worker, reference=fake_reference)

    def test_spawn_four_lanes_and_reap(self):
        result = self.artificial_pool()
        self.assertTrue(result['all_reaped'])
        self.assertTrue(all(x['status']=='COMPLETE_MANDATORY_STOP' for x in result['lanes'].values()))
        self.assertEqual(sum(x['compile_entries'] for x in result['lanes'].values()), 60)

    def test_failed_lane_does_not_kill_other_lanes(self):
        failed = c.LANES[0];result = self.artificial_pool(failed)
        self.assertEqual(result['lanes'][failed]['status'], 'STOP')
        self.assertTrue(all(x['status']=='COMPLETE_MANDATORY_STOP'
                            for l,x in result['lanes'].items() if l!=failed))
        self.assertTrue(result['all_reaped'])


@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'), 'explicit frozen synthetic-test provider required')
class SavedInputToys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))

    def fixture(self, root):
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import _state_hash
        from h46_m5_tol_saved_input import array_record
        one = np.diag(np.linspace(-.03,.05,8)).astype(complex)
        one[0,1] = .002j;one[1,0] = -.002j
        lambdas = np.array([(-1)**i*.001*(i+1) for i in range(10)])
        gs = np.array([np.diag(np.linspace(.01,.08,8)+i*.001) for i in range(10)])
        ham = DFHamiltonian(.137,one,lambdas,tuple(gs),{})
        sector = PhysicalSector.spin_sector(n_qubits=8,nelec_alpha=2,nelec_beta=2)
        state = np.zeros(len(sector.basis_indices), complex);state[0] = 1
        full = np.zeros(256, complex);full[sector.basis_indices] = state
        arrays = dict(constant=np.array(.137),one_body=one,lambdas=lambdas,g_matrices=gs,
            sector_basis_indices=sector.basis_indices,state_vector=full,sector_state_vector=state)
        snapshot = root/'synthetic_tol.npz';np.savez(snapshot,**arrays)
        receipt = dict(status='COMPLETE_MANDATORY_STOP',sha256=c.sha(snapshot),
            arrays={k:array_record(v) for k,v in arrays.items()},hamiltonian_metadata={},
            hamiltonian_hash=df_hamiltonian_hash(ham),state_hash=_state_hash(full,state),rank=10,projection={})
        path = root/'synthetic_receipt.json';path.write_text(json.dumps(receipt))
        return snapshot,path,receipt

    def test_actual_synthetic_saved_loader_and_tamper(self):
        from h46_m5_tol_saved_input import load
        with tempfile.TemporaryDirectory() as tmp:
            snap,path,receipt = self.fixture(Path(tmp))
            cfg = {'mapping':{'tol_snapshot':str(snap),'tol_receipt':str(path)}}
            result = load(cfg)
            self.assertEqual(result[0].n_blocks, 10)
            self.assertEqual(result[4]['hamiltonian_hash'], receipt['hamiltonian_hash'])
            with snap.open('ab') as f:f.write(b'artificial_corruption')
            with self.assertRaisesRegex(ValueError, 'TOL_INPUT_COMPLETION'):
                load(cfg)

    def test_context_and_actual_paired_native_task_synthetic_only(self):
        provider = os.environ['M5_TEST_PROVIDER']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot,path,receipt=self.fixture(root)
            (root/'out').mkdir();(root/'ledger').mkdir()
            lane = 'H4_TOL_PILOT_PR'
            scope = c.scope(lane)
            task = next(t for t in scope['tasks'] if t['sample']==0 and t['group']['formula']=='4th(m5_best)')
            cfg = dict(lane=lane,system='H4',SOURCE='a'*40,provider_root=provider,
                runtime={'provider_root':provider},inputs={
                    'H4_tol_snapshot.npz':{'path':str(snapshot)},'H4_tol_receipt.json':{'path':str(path)}},
                binding=dict(hamiltonian=receipt['hamiltonian_hash'],state=receipt['state_hash'],
                             snapshot=receipt['sha256'],rank=10),
                plan={'scope':scope},plan_sha256='b'*64,reference_root=str(root/'ref'),
                output=str(root/'out'),ledger=str(root/'ledger'),registry_path=str(root/'registry.json'))
            c.save(root/'registry.json',dict(lanes=c.registry('a'*40,set())))
            det = copy.deepcopy(cfg);det['plan']['scope']=c.scope('H4_TOL_PILOT_DET')
            with patch.object(c,'authorize'),patch.object(c,'consumed'),patch.object(c,'verify_seals'):
                ref=n.preflight(det)
                self.assertEqual(ref['status'],'PASS')
                result=n.run_task(cfg,task,threading.Event())
            self.assertEqual(result['status'],'PASS',result.get('reason'))
            self.assertFalse(result['shot_accounting'])
            self.assertEqual(len(list((root/'ledger').glob('*.json'))),2)
            self.assertEqual(len(result['checks']),8)
            self.assertEqual(len({x['event_digest'] for x in result['records']}),1)


if __name__ == '__main__':
    unittest.main()
