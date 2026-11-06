"""Artificial SOURCE-handoff/remaining-task tests, no scientific execution."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_track_d_c6b_accelerated as execution


class HandoffTests(unittest.TestCase):
    def fixture(self,root):
        def write(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value)+'\n');return p
        ids=[f'q8_sample{i:02d}' for i in range(32)]
        write('result.json',dict(source_commit=execution.OLD_SOURCE,status='STOP_FAILURE',planned_ids=ids,compile_calls_started=1))
        write('STOP.json',dict(mandatory_stop=True))
        write('failure.json',dict(error_type='KeyboardInterrupt'))
        for n in ['SOURCE.json','seed_plan.json','frozen_model.json']:write(n,dict(synthetic=True))
        bindings=[]
        for sid in ids:
            p=write('local/events/'+sid+'.json',dict(sample_id=sid,synthetic=True))
            bindings.append(dict(sample_id=sid,record_sha256=execution.sha(p)))
        write('event_bindings.json',bindings)
        p=root/'local/car_context.npz';p.write_bytes(b'artificial-placeholder-not-decoded')
        prefix='local/q8_sample00/'
        write(prefix+'precompile_checks.json',dict(status='PASS',peak_RSS_KiB=7))
        write(prefix+'reference_X_compile.json',dict(status='PASS',compile_calls=1))
        write(prefix+'reference_X_check.json',dict(status='PASS'))
        p=root/prefix/'circuits/reference_X_compiled.qpy';p.parent.mkdir();p.write_bytes(b'artificial-compiled-placeholder')
        (root/prefix/'reference_X_observable.npz').write_bytes(b'artificial-observable-placeholder')
        return bindings

    def test_remaining_tasks_never_repeat_completed_raw_or_compile_and_total_is_128(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);events=self.fixture(root);plan=execution.inheritance_plan(root)
            self.assertEqual(plan['raw_reused'],['q8_sample00'])
            self.assertEqual(plan['wrappers_reused'],[['q8_sample00','reference','X']])
            estimates={x+'_task_memory_estimate_bytes':10 for x in ['raw','reference','direct']}
            tasks=execution.initial_tasks(events,plan,estimates)
            self.assertEqual(len([x for x in tasks if x['kind']=='raw']),31)
            compiles=[x for x in tasks if x['kind']=='compiled']
            self.assertEqual(len(compiles),3)
            self.assertNotIn(('q8_sample00','reference','X'),[(x['sample_id'],x['method'],x['axis']) for x in compiles])
            self.assertEqual(1+len(compiles)+31*4,128)
            execution.verify_inheritance(root,plan)
            self.assertEqual((root/'result.json').read_text(),json.dumps(dict(source_commit=execution.OLD_SOURCE,
                status='STOP_FAILURE',planned_ids=[f'q8_sample{i:02d}' for i in range(32)],compile_calls_started=1))+'\n')

    def test_partial_or_failed_work_rejected_without_retry(self):
        for kind in ['raw_partial','compile_started','post_missing','failure','wrong_drain']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.fixture(root);directory=root/'local/q8_sample00'
                if kind=='raw_partial':(root/'local/q8_sample01').mkdir()
                elif kind=='compile_started':
                    (directory/'reference_X_compile.json').write_text(json.dumps(dict(status='STARTED',compile_calls=1)))
                elif kind=='post_missing':(directory/'reference_X_check.json').unlink()
                elif kind=='failure':(directory/'reference_X_failure.json').write_text('{}')
                else:(root/'failure.json').write_text(json.dumps(dict(error_type='ValueError')))
                with self.assertRaises(ValueError):execution.inheritance_plan(root)

    def test_inherited_original_mutation_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);plan=execution.inheritance_plan(root)
            (root/'local/events/q8_sample00.json').write_text('changed')
            with self.assertRaises(ValueError):execution.verify_inheritance(root,plan)

    def test_worker_selection_is_explicit_and_only_changes_own_runtime_alias(self):
        execution.frozen.c2.old().bootstrap(Path(os.environ['TRACK_D_ORIGIN']))
        from trottertracks import c1_h6_state_action as primitive
        from trottertracks.track_d_state_action_acceleration import BatchedStateActions
        original=primitive.StateActions
        with patch.object(primitive,'StateActions',original):
            execution.install_state_actions()
            self.assertIs(primitive.StateActions,BatchedStateActions)
        self.assertIs(primitive.StateActions,original)

    def test_actual_worker_count_overrides_historical_observer_metadata_without_launch(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(execution,'ProcessPoolExecutor') as executor:
            root=Path(temp)
            execution.pipeline([],dict(raw_remaining=[],raw_reused=[],wrappers_reused=[]),root,root,
                dict(numerical_threads=8,worker_processes=2,memory={'MemAvailable':100}),
                {k+'_task_memory_estimate_bytes':1 for k in ['raw','reference','direct']},{},{},None,root)
            self.assertEqual(json.loads((root/'resources.json').read_text())['worker_processes'],1)
            self.assertEqual(executor.call_args.kwargs['max_workers'],1)
            self.assertFalse(executor.return_value.__enter__.return_value.submit.called)


if __name__=='__main__':unittest.main()
