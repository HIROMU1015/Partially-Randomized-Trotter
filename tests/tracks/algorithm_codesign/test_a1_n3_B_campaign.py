"""B orchestration tests: invented journals and mocks, no scientific input/RNG."""
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from trottertracks.algorithm_codesign import a1_n3_B_campaign as b


class BCampaignTests(unittest.TestCase):
    def test_cli_imports_own_source_without_pythonpath_or_working_directory(self):
        import os,subprocess
        root=Path(__file__).resolve().parents[3]
        env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','SHELL','PATH','LANG') if k in os.environ}
        self.assertNotIn('PYTHONPATH',env)
        result=subprocess.run([b.PYTHON,'-B',str(root/b.ENTRY),'--help'],cwd='/tmp',env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--submit',result.stdout)
        self.assertNotIn('Traceback',result.stderr)

    def test_new_fixed_identity_seed_and_output(self):
        self.assertNotEqual(b.SEED,b.c.SEED)
        self.assertEqual(b.SEED,20261107)
        self.assertNotEqual(b.OUTPUT,b.c.OUTPUT)
        self.assertIn('B-',b.CAMPAIGN)
        self.assertEqual(len(b.c.domains()),60)

    def test_semantic_failure_even_when_process_exit_zero(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            self.assertEqual(b.semantic_status(out,0),1)
            b.c.save(out/'STOP.json',dict(campaign_status='STOPPED'))
            self.assertEqual(b.semantic_status(out,0),1)
            b.c.save(out/'STOP.json',dict(campaign_status='COMPLETED'))
            self.assertEqual(b.semantic_status(out,0),0)
            self.assertEqual(b.semantic_status(out,-9),1)

    def test_duplicate_dispatch_cannot_launch_manager(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/b.CONTROL;out.mkdir(parents=True)
            b.managed.durable_new(out/'CONSUMED.json',{'consumed':True})
            with patch.object(b,'read_spec',return_value=(root,{})),patch.object(b.cron,'record_manager') as launch:
                self.assertEqual(b.dispatch('unused'),72)
            launch.assert_not_called()

    def test_consumption_and_cron_removal_precede_manager(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/b.CONTROL;out.mkdir(parents=True)
            spec={'python':'mock-python'}
            def launch(*args):
                self.assertTrue((out/'CONSUMED.json').exists())
                self.assertTrue((out/'CRONTAB_RESTORED.json').exists())
                return dict(raw_returncode=0)
            with patch.object(b,'read_spec',return_value=(root,spec)),patch.object(b.cron,'remove_own_entry') as removal,patch.object(b.cron,'own_crontab',return_value=None),patch.object(b.obs,'process_identity',return_value={'cgroup':'0::/system.slice/cron.service'}),patch.object(b,'verify'),patch.object(b.os,'chdir'),patch.object(b.cron,'record_manager',side_effect=launch),patch.object(b,'semantic_status',return_value=0):
                self.assertEqual(b.dispatch('unused'),0)
                removal.assert_called_once_with(spec)

    def test_session_cgroup_rejected_before_science(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/b.CONTROL;out.mkdir(parents=True)
            with patch.object(b,'read_spec',return_value=(root,{})),patch.object(b.cron,'remove_own_entry'),patch.object(b.cron,'own_crontab',return_value=None),patch.object(b.obs,'process_identity',return_value={'cgroup':'0::/user.slice/session.scope'}),patch.object(b.cron,'record_manager') as launch:
                with self.assertRaisesRegex(RuntimeError,'cgroup'):b.dispatch('unused')
            launch.assert_not_called();self.assertTrue((out/'DISPATCH_STOP.json').exists())

    def test_output_collision_cannot_decode_or_sample(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/b.OUTPUT;out.mkdir(parents=True)
            (out/'INVOCATION.json').write_text('{}')
            with patch.object(b,'read_spec',return_value=(root,{})),patch.object(b.c.n3,'load_certified_services') as bind:
                with self.assertRaises(FileExistsError):b.parent('unused')
            bind.assert_not_called()

    def test_stop_saved_before_drain_and_planned_attempt_classification(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);rt=b.managed.Runtime(out)
            pool=Mock()
            pool.shutdown.side_effect=lambda **kw:self.assertTrue((out/'STOP_REQUESTED.json').exists())
            b.c.append(out/'JOURNAL.jsonl',dict(event='TRIAL_PLANNED',ticket='toy',domain={'method':'toy'},trial=0))
            with patch.object(b.c,'reconcile',return_value={}):
                self.assertEqual(b.finalize(out,rt,None,pool,'STOPPED','invented',b.time.monotonic()),1)
            row=json.loads((out/'CAMPAIGN_LEDGER.json').read_text())['trials'][0]
            self.assertEqual(row['classification'],'PLANNED_NO_DURABLE_WORKER_START')
            self.assertEqual(row['campaign'],b.CAMPAIGN)

    def test_completed_requires_all_confidence_and_prices(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);rt=b.managed.Runtime(out)
            with patch.object(b.c,'reconcile',return_value={}):
                self.assertEqual(b.finalize(out,rt,None,None,'COMPLETED',None,b.time.monotonic()),1)
            self.assertEqual(json.loads((out/'STOP.json').read_text())['campaign_status'],'STOPPED')

    def test_heartbeat_coalescing_preserves_calls_exits_and_stop(self):
        with TemporaryDirectory() as tmp:
            log=Path(tmp)/'log'
            with b.heartbeat_storage():
                b.obs.event(log,'HEARTBEAT',toy=True)
                b.obs.event(log,'HEARTBEAT',toy=True)
                for event in ('CALL_START','CALL_RETURN','SIGNAL_RECEIVED','STOP_REQUESTED','CHILD_WAIT_STATUS'):
                    b.obs.event(log,event,toy=True)
            events=[json.loads(line)['event'] for line in log.read_text().splitlines()]
            self.assertEqual(events.count('HEARTBEAT'),1)
            self.assertEqual(events[-1],'CHILD_WAIT_STATUS')
            self.assertEqual(len(events),6)

    def test_parent_uses_fresh_seed_once_and_freezes_source(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/b.OUTPUT;out.mkdir(parents=True)
            counts={'B0':1,'B1-O':2,'B1-OL':5,'A1':1}
            sampler=NS(cells=lambda method:tuple((str(i),1) for i in range(counts[method])))
            services=(None,None,sampler,None,{'total_engine_coordinate_upper_exact':'1/10000'})
            methods={m:dict(realized_RZ_complete=True,results={a:dict(status='SUFFICIENT_CONFIDENCE_COMPLETED') for a in ('X','Y')}) for m in counts}
            spec=dict(source_commit='toy',manager_SOURCE='toy-manager')
            pool=Mock();campaign=Mock();campaign.classical={}
            with patch.object(b,'read_spec',return_value=(root,spec)),patch.object(b,'verify',return_value={'toy':'binding'}),patch.object(b.c,'forbidden_science',return_value=nullcontext()),patch.object(b.managed,'connect_legacy',return_value=nullcontext()),patch.object(b.c.n3,'load_certified_services',return_value=services) as bind,patch.object(b.c.n3,'pcg64_streams') as streams,patch.object(b.c,'available_workers',return_value=1),patch.object(b.c,'ProcessPoolExecutor',return_value=pool),patch.object(b.c,'Campaign',return_value=campaign),patch.object(b.c,'reconcile',return_value=methods):
                self.assertEqual(b.parent('unused'),0)
                bind.assert_called_once_with(root)
                streams.assert_called_once_with(20261107,b.c.domains(),path_word_cap=64)
                campaign.execute.assert_called_once_with()
                pool.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
            self.assertEqual(json.loads((out/'SOURCE.json').read_text())['old_measurements_reused'],0)
            self.assertEqual(json.loads((out/'STOP.json').read_text())['campaign_status'],'COMPLETED')

    def test_binding_failure_stops_without_streams_or_retry(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/b.OUTPUT;out.mkdir(parents=True)
            with patch.object(b,'read_spec',return_value=(root,dict(source_commit='toy',manager_SOURCE='toy'))),patch.object(b,'verify',return_value={}),patch.object(b.c,'forbidden_science',return_value=nullcontext()),patch.object(b.managed,'connect_legacy',return_value=nullcontext()),patch.object(b.c.n3,'load_certified_services',side_effect=ValueError('toy mismatch')) as bind,patch.object(b.c.n3,'pcg64_streams') as streams,patch.object(b.c,'reconcile',return_value={}):
                self.assertEqual(b.parent('unused'),1)
                bind.assert_called_once();streams.assert_not_called()
            self.assertEqual(json.loads((out/'STOP.json').read_text())['campaign_status'],'STOPPED')

    def test_old_run_and_old_seed_book_are_not_called(self):
        import inspect
        source=inspect.getsource(b)
        self.assertNotIn('c.run(',source)
        self.assertIn('pcg64_streams(SEED,c.domains()',source)
        self.assertNotIn('20261106,',source)


if __name__=='__main__':unittest.main()
