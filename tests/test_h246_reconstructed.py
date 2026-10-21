"""New contract wiring and native toy checks; no molecular acquisition."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import h246_reconstructed_contract as c


class BoundaryTests(unittest.TestCase):
    def tearDown(self):c.RT=None

    def test_draft_refused_before_read(self):
        with patch.object(c,'read',side_effect=AssertionError('must not read inputs')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_RECONSTRUCTED'):
                c.authorize(dict(authorized=False,grant={'valid':False}))

    def test_fixed_budget(self):
        scopes=[c.law.call_scope(c.tasks_for(s),'PR') for s in c.STAGES]
        self.assertEqual(sum(r['trajectories'] for r in scopes),118)
        self.assertEqual(sum(r['compile_axes'] for r in scopes),236)
        self.assertEqual(sum(r['request_sampling_calls'] for r in scopes),846)

    def test_runtime_contract_and_supervisor_use_new_entry(self):
        r=c.runtime(dict(stage='H4_CONFIRMATION',signal_path='synthetic'))
        self.assertIs(r.native_namespace['c'].authorize,c.authorize)
        self.assertIs(r.runner_namespace['c'].verify,c.verify)
        self.assertIs(r.audit_namespace['c'].authorize,c.authorize)
        self.assertEqual(Path(r.runner.supervise.__globals__['__file__']).name,'run_h246_reconstructed.py')
        self.assertEqual(r.native_namespace['c'].design.COHORT,'CONFIRMATION')
        import pickle
        self.assertIs(pickle.loads(pickle.dumps(r.runner.execute.__kwdefaults__['target'])),c.worker)

    def test_importable_worker_through_real_mock_pool(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        c.RT=SimpleNamespace(worker=lambda x:x+1)
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('fork')) as pool:
            self.assertEqual(list(pool.map(c.worker,[1,2,3])),[2,3,4])

    def test_cli_drafts_all_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';p.write_text('{"authorized":false,"grant":{"valid":false}}')
            for mode in ('execute','supervise','audit','evaluate'):
                r=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h246_reconstructed.py'),'--'+mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_RECONSTRUCTED',r.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[p])

    @unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'toy provider required')
    def test_new_runtime_native_three_formulas(self):
        sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))
        from test_h246_confirmation_readiness import toy_task,options
        from test_h46_m5_cost_readiness import NativeToyTests
        import h46_m5_native as original
        toy=NativeToyTests.toy(self)
        for formula in ('2nd','4th','4th(m5_best)'):
            c.RT=None
            with self.subTest(formula=formula),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir();t=toy_task(formula=formula)
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage='H4_CONFIRMATION',tasks=[t],run_root=tmp,
                    output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
                    storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options(),
                    systems={'H4':{'binding':{'synthetic':True}}},registry_path=str(root/'seeds.json'),sealed_files=[],signal_path=str(root/'signal.json'))
                c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                c.save(cfg['registry_path'],c.law.registry([t],cfg['SOURCE'],'ARTIFICIAL_MIGRATION','CONFIRMATION',[]))
                c.save(cfg['signal_path'],dict(records=[dict(candidate=t['candidate'],signal=original.signal(toy,dict(t['candidate'],policy=t['candidate']['r_policy'])))]))
                cfg['sealed_files']=[c.seal(p,'artificial') for p in (cfg['registry_path'],cfg['signal_path'])]
                r=c.runtime(cfg);view=r.native_namespace['c']
                with patch.object(view,'authorize'),patch.object(view,'verify'),patch.dict(r.native_namespace,{'context':lambda *a:toy}):
                    r.native.run_task(cfg,t,threading.Event())
                self.assertEqual(r.auditor.task_audit(cfg,t)['archives'],6)
                result=c.read(root/'output'/t['id']/'result.json')
                self.assertEqual(result['seed_cohort'],'CONFIRMATION')
                self.assertTrue(all(g['PASS'] for g in result['checks']))
