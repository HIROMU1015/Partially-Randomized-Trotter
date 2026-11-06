"""Activation tests: mocks/invented metadata only, no scientific inputs/RNG."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
from trottertracks.algorithm_codesign import a1_n3_pilot_activation as a

class ActivationTests(unittest.TestCase):
    def test_minimal_cron_environment_bootstrap(self):
        root=Path(__file__).resolve().parents[3]
        env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','SHELL','PATH','LANG') if k in os.environ}
        result=subprocess.run([a.PYTHON,'-B',str(root/a.ENTRY),'--help'],cwd='/tmp',env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--submit',result.stdout)

    def test_adopted_scope_and_independent_domains(self):
        self.assertEqual(a.SEED,20261108);self.assertEqual(a.CAMPAIGN,'A1-N3-PILOT-H4-m0-q1-20261106-v1')
        ds=a.pilot.pilot_domains();self.assertEqual(len(ds),32);self.assertEqual(len({d.token for d in ds}),32)
        self.assertEqual({d.method for d in ds},{'A1','B1-OL'});self.assertEqual({d.stage for d in ds},{'train','prod'})
        self.assertEqual(a.pilot.design.BUDGET.total,752);self.assertNotEqual(a.OUTPUT,a.c.OUTPUT)

    def test_duplicate_dispatch_never_launches(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/a.CONTROL;control.mkdir(parents=True)
            a.managed.durable_new(control/'CONSUMED.json',{})
            with patch.object(a,'read_spec',return_value=(root,{})),patch.object(a.cron,'record_manager') as launch:
                self.assertEqual(a.dispatch('unused'),72)
            launch.assert_not_called()

    def test_cron_removed_before_manager_and_same_output(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/a.CONTROL;control.mkdir(parents=True);spec={'python':'toy'}
            def launch(argv,out,cwd):
                self.assertEqual(out,root/a.OUTPUT)
                self.assertTrue((control/'CONSUMED.json').exists());self.assertTrue((control/'CRONTAB_RESTORED.json').exists())
                return dict(raw_returncode=0)
            with patch.object(a,'read_spec',return_value=(root,spec)),patch.object(a.cron,'remove_own_entry') as removal,patch.object(a.cron,'own_crontab',return_value=None),patch.object(a.obs,'process_identity',return_value={'cgroup':'0::/system.slice/cron.service'}),patch.object(a,'verify'),patch.object(a.os,'chdir'),patch.object(a.cron,'record_manager',side_effect=launch),patch.object(a,'semantic_status',return_value=0):
                self.assertEqual(a.dispatch('unused'),0);removal.assert_called_once_with(spec)

    def test_session_manager_refused_without_loading(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/a.CONTROL).mkdir(parents=True)
            with patch.object(a,'read_spec',return_value=(root,{})),patch.object(a.cron,'remove_own_entry'),patch.object(a.cron,'own_crontab',return_value=None),patch.object(a.obs,'process_identity',return_value={'cgroup':'session.scope'}),patch.object(a.cron,'record_manager') as launch:
                with self.assertRaisesRegex(RuntimeError,'persistent cron'):a.dispatch('unused')
            launch.assert_not_called();self.assertTrue((root/a.CONTROL/'DISPATCH_STOP.json').exists())

    def test_output_collision_rejected_before_input(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/a.OUTPUT;out.mkdir(parents=True);(out/'STOP.json').write_text('{}')
            with patch.object(a,'read_spec',return_value=(root,{})),patch.object(a.c.n3,'load_certified_services') as load:
                with self.assertRaises(FileExistsError):a.parent('unused')
            load.assert_not_called()

    def test_binding_exception_stop_before_rng(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/a.OUTPUT;out.mkdir(parents=True)
            with patch.object(a,'read_spec',return_value=(root,dict(source_commit='toy',manager_SOURCE='toy'))),patch.object(a,'verify',return_value={}),patch.object(a.c,'forbidden_science',return_value=nullcontext()),patch.object(a.managed,'connect_legacy',return_value=nullcontext()),patch.object(a.c.n3,'load_certified_services',side_effect=ValueError('invented identity mismatch')) as load,patch.object(a.c.n3,'pcg64_streams') as rng,patch.object(a.c,'reconcile',return_value={}):
                self.assertEqual(a.parent('unused'),1)
            load.assert_called_once();rng.assert_not_called()
            self.assertEqual(json.loads((out/'STOP.json').read_text())['campaign_status'],'STOPPED')

    def test_one_parent_binding_stream_book_and_driver(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/a.OUTPUT;out.mkdir(parents=True)
            services=(None,None,None,None,{'total_engine_coordinate_upper_exact':'1/999'})
            driver=Mock();driver.collector.raw.classical={}
            def run():
                a.c.save(out/'STOP.json',dict(purpose=a.pilot.PURPOSE,campaign_status='COMPLETED',pilot_status='PILOT_DIAGNOSTIC_COMPLETED'))
            driver.run.side_effect=run
            with patch.object(a,'read_spec',return_value=(root,dict(source_commit='toy',manager_SOURCE='toy'))),patch.object(a,'verify',return_value={'binding':'toy'}),patch.object(a.c,'forbidden_science',return_value=nullcontext()),patch.object(a.managed,'connect_legacy',return_value=nullcontext()),patch.object(a.c.n3,'load_certified_services',return_value=services) as load,patch.object(a.c.n3,'pcg64_streams') as rng,patch.object(a.c,'available_workers',return_value=1),patch.object(a.c,'ProcessPoolExecutor') as pool,patch.object(a.pilot,'connect_legacy_pipeline',return_value=driver) as connect:
                self.assertEqual(a.parent('unused'),0)
                load.assert_called_once_with(root);rng.assert_called_once_with(20261108,a.pilot.pilot_domains(),path_word_cap=64)
                driver.run.assert_called_once_with();connect.assert_called_once();pool.return_value.shutdown.assert_not_called()
            source=json.loads((out/'SOURCE.json').read_text())
            self.assertEqual(source['old_measurements_reused'],0);self.assertEqual(source['conditions']['total_cap'],752)
            streams=json.loads((out/'STREAMS.json').read_text());self.assertEqual(streams['children'],64)
            self.assertEqual([d['path_spawn_key'] for d in streams['domains']],[[2*i] for i in range(32)])

    def test_semantic_failure_overrides_raw_zero(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);self.assertEqual(a.semantic_status(out,0),1)
            a.c.save(out/'STOP.json',dict(purpose=a.pilot.PURPOSE,campaign_status='STOPPED',pilot_status='STOPPED'))
            self.assertEqual(a.semantic_status(out,0),1)
            a.c.save(out/'STOP.json',dict(purpose=a.pilot.PURPOSE,campaign_status='COMPLETED',pilot_status='PILOT_DIAGNOSTIC_COMPLETED'))
            self.assertEqual(a.semantic_status(out,0),0);self.assertEqual(a.semantic_status(out,-9),1)

    def test_setup_failure_persists_stop_before_drain(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);rt=a.managed.Runtime(out);pool=Mock()
            pool.shutdown.side_effect=lambda **kw:self.assertTrue((out/'STOP_REQUESTED.json').exists())
            with patch.object(a.c,'reconcile',return_value={}):
                self.assertEqual(a.finish_metadata(out,rt,None,pool,a.time.monotonic(),'toy',None),1)
            pool.shutdown.assert_called_once();self.assertEqual(json.loads((out/'RUN_FINALIZATION.json').read_text())['scientific_retry'],False)

if __name__=='__main__':unittest.main()
