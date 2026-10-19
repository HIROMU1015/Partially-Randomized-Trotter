"""Synthetic authorization, stage and native action-path checks."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_m5_phase_batch_contract as c
import h6_m5_phase_batch_native as native
import run_h6_m5_phase_batch as runner


def fake_worker_PASS(cfg,stop):
    for stage in c.STAGES:c.reserve(cfg,stage)
    checks=[dict(PASS=True,raw_error=0.,strictly_below=1e-9,reused=True),
        dict(PASS=True,raw_error=0.,strictly_below=1e-9),dict(PASS=True,raw_error=0.,strictly_below=1e-9)]
    path=Path(cfg['output'])/'phase_batches.jsonl';path.write_text('{}\n')
    c.save(Path(cfg['output'])/'report.json',dict(status='PHASE_BATCH_SINGLE_STATE_PASS_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],checks=checks,artifacts={},compiler_policy=c.POLICY,new_compile=1,statevector_replays=1,reused_initial_raw_gate=1,accepted_cost_rows=0,mandatory_STOP=True,next_phase_started=False,phase_batches_sha256=c.sha(path)))


def fake_worker_STOP(cfg,stop):
    c.reserve(cfg,c.STAGES[0]);c.save(Path(cfg['output'])/'STOP.json',dict(status='STOP',reason='ARTIFICIAL_NUMERIC_DISAGREEMENT'))


class ContractTests(unittest.TestCase):
    def test_draft_cli_refused_before_science(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';c.save(p,dict(authorized=False,grant=dict(valid=False)))
            for mode in ('--execute','--supervise'):
                result=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h6_m5_phase_batch.py'),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2);self.assertIn('EXPLICIT_PHASE_BATCH_EXECUTION_APPROVAL_REQUIRED',result.stdout)
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_old_or_expanded_grant_refused(self):
        base=dict(authorized=True,SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64)
        grant=dict(valid=True,grant_id='synthetic',scope_id=c.SCOPE,SOURCE=base['SOURCE'],plan_sha256=base['plan_sha256'],pipeline_identity=base['pipeline_identity'],maximum_new_compile=1,maximum_statevector_replays=1,sine=0,sampling=0,accepted_cost_rows=0,retry=False,next_phase_authorized=False)
        for key,value in [('scope_id',c.prior.SCOPE),('maximum_new_compile',2),('maximum_statevector_replays',2),('sine',1),('retry',True)]:
            cfg=copy.deepcopy(base);cfg['grant']=dict(grant,**{key:value})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'EXACT_NEW_PHASE_BATCH_GRANT'):c.authorize(cfg)

    def test_stage_order_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ledger').mkdir()
            cfg=dict(marker=str(root/'CONSUMED.json'),grant=dict(id='synthetic'),ledger=str(root/'ledger'),SOURCE='a'*40,plan_sha256='b'*64)
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
            with self.assertRaises(ValueError):c.reserve(cfg,'COMPILE_COSINE')
            for name in c.STAGES:c.reserve(cfg,name)
            with self.assertRaises(ValueError):c.reserve(cfg,'LOAD_SAVED_PASS_INPUT')
            with self.assertRaises(ValueError):c.reserve(cfg,'LOWER_NATIVE')

    def test_fixed_reduced_scope_and_three_checks(self):
        s=c.scope()
        self.assertEqual(s['maximum_new_compile'],1);self.assertEqual(s['maximum_statevector_replays'],1)
        self.assertEqual(s['new_lowering'],0);self.assertEqual(s['new_lowered_replay'],0)
        self.assertEqual(s['reused_initial_raw_gate'],1);self.assertEqual(len(s['numerical_checks']),3)
        self.assertEqual(s['raw_strictly_below'],1e-9);self.assertFalse(s['phase_alignment_for_acceptance'])
        self.assertTrue(s['old_STOP_remains_STOP']);self.assertEqual(s['AD03'],'DEFERRED')

    def test_one_compile_one_action_in_native_AST(self):
        tree=ast.parse(Path(native.__file__).read_text())
        compiles=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='compile_with_phase_batch']
        actions=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='ordinary']
        self.assertEqual(len(compiles),1);self.assertEqual(len(actions),1)
        self.assertNotIn('lower_native',Path(native.__file__).read_text())

    def test_raw_fail_still_raises_no_alignment(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'NUMERIC_DISAGREEMENT'):
                native.gate_error('synthetic',np.array([1+0j]),np.array([1+2e-9j]),d)
            row=c.read(Path(d)/'synthetic.json')
            self.assertFalse(row['PASS']);self.assertFalse(row['phase_alignment'])

    def test_spawn_exit_reap_one_shot_and_scientific_STOP(self):
        for target,expected in [(fake_worker_PASS,0),(fake_worker_STOP,1)]:
            with self.subTest(target=target.__name__),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'run01'
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64,
                    runtime=dict(python=sys.executable),run_root=str(root),output=str(root/'output'),
                    ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant=dict(id='synthetic'))
                with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(runner.observed,'fresh_resources',return_value=dict(workers=1,host_PSI_kill=False)):
                    self.assertEqual(runner.execute(cfg,target=target),expected)
                    witness=c.read(root/'completion_witness.json')
                    self.assertEqual(witness['worker_exit'],0);self.assertTrue(witness['worker_reaped'])
                    self.assertEqual(witness['status']=='STOP',expected==1)
                    with self.assertRaisesRegex(ValueError,'UNUSED_BATCH_RUN_ROOT'):runner.execute(cfg,target=target)


if __name__=='__main__':unittest.main()
