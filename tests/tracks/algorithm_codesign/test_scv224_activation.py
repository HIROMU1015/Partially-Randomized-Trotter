"""Mock activation, toy metadata/fixed bits only; never submit a science job."""
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from trottertracks.algorithm_codesign import scv224_activation as a
from test_scv_scalar import metadata,scope
from test_scv224_pilot import Collector


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.md,self.law,self.cat,self.kw=metadata();self.control=a.scalar.select_fragment_control(self.md)
        self.scopes={axis:scope(self.md,axis) for axis in ('X','Y')}

    def test_adopted_constants_and_finite_scope(self):
        self.assertEqual(a.BASE,'560c6092bfb5e71af8eedc4a4be5416922764f98')
        self.assertEqual(a.SEED,20261109);self.assertEqual(a.CAMPAIGN,'O2-SCV224-H4-m0-q1-20261107-v1')
        self.assertEqual(len(a.pilot.domains()),12);self.assertEqual(a.pilot.preparation_plan()['total_physical_shots'],224)
        self.assertNotEqual(a.OUTPUT,a.routing.OUTPUT)

    def test_cron_minimal_environment_bootstrap(self):
        root=next(p for p in Path(__file__).parents if (p/'PROJECT_MAP.md').exists())
        env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','SHELL','PATH','LANG') if k in os.environ}
        r=subprocess.run([a.PYTHON,'-B',str(root/a.ENTRY),'--help'],env=env,cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--submit',r.stdout)

    def test_metadata_only_toy_binding_selected_exactly_once(self):
        from trottertracks.algorithm_codesign import a1_new_target_n1 as n1
        one=self.root/a.c.kernel.P1;one.mkdir(parents=True)
        (one/'LAW.json').write_bytes(self.law);(one/'PREPARATION.json').write_bytes(self.cat)
        out=self.root/'out';out.mkdir()
        ids=dict(law_sha256=self.kw['law_sha256'],catalog_sha256=self.kw['catalog_sha256'],hamiltonian_hash='toy-H',preparation_hash='toy-preparation')
        with patch.object(n1,'A','a'),patch.object(n1,'B','b'),patch.object(a.scalar,'select_fragment_control',wraps=a.scalar.select_fragment_control) as select, \
             patch.object(a.n3,'load_certified_services',side_effect=AssertionError('metadata must not load state')):
            control=a.metadata_once(self.root,dict(N1_identities=ids),out)
        self.assertEqual(control.identity,self.control.identity);self.assertEqual(select.call_count,1)
        self.assertTrue((out/'METADATA_CONTROL.json').exists())

    def test_sampler_object_binding_requires_exact_same_law(self):
        md,*_=metadata();services=(None,None,md.sampler,self.scopes,{})
        bound=a.bind_sampler(services,self.control);self.assertIs(bound[2],self.control.metadata.sampler)
        md.sampler.a='b'
        with self.assertRaisesRegex(ValueError,'sampler mismatch'):a.bind_sampler(services,self.control)

    def test_ordered_driver_collects_both_training_before_any_fit(self):
        out=self.root/'out';out.mkdir();a.c.save(out/'SOURCE.json',a.pilot.source_fields(a.CAMPAIGN,self.control))
        collector=Collector(out,self.control)
        driver=a.OrderedDriver(out,a.CAMPAIGN,self.control,self.scopes,collector)
        real=a.scalar.fit_cells;fits=[]
        def fit(*args,**kwargs):
            self.assertEqual([(axis,stage) for axis,stage,_ in collector.calls],[('X','train'),('Y','train')])
            fits.append(1);return real(*args,**kwargs)
        with patch.object(a.scalar,'fit_cells',side_effect=fit):r=driver.run()
        self.assertEqual(r['status'],'COMPLETE',r);self.assertEqual(len(fits),2)
        self.assertEqual(sum(sum(n for _,n in q) for _,_,q in collector.calls),224)
        self.assertEqual(len(driver.envelopes),4);self.assertEqual(len(driver.seen),224)

    def test_manager_routing_uses_qualified_path_and_restores_globals(self):
        original=a.routing.OUTPUT
        with patch.object(a.routing,'manager',side_effect=lambda p:(a.routing.OUTPUT,a.routing.ENTRY,p)):
            self.assertEqual(a.manager('toy'),(a.OUTPUT,a.ENTRY,'toy'))
        self.assertEqual(a.routing.OUTPUT,original)

    def test_consumed_dispatch_rejects_without_launch(self):
        control=self.root/a.CONTROL;control.mkdir(parents=True);a.managed.durable_new(control/'CONSUMED.json',{})
        with patch.object(a,'read_spec',return_value=(self.root,{})),patch.object(a.cron,'record_manager') as launch:
            self.assertEqual(a.dispatch('unused'),72)
        launch.assert_not_called()

    def test_session_scope_rejected_before_scientific_calls(self):
        (self.root/a.CONTROL).mkdir(parents=True)
        with patch.object(a,'read_spec',return_value=(self.root,{})),patch.object(a.cron,'remove_own_entry'),patch.object(a.cron,'own_crontab',return_value=None),patch.object(a.obs,'process_identity',return_value={'cgroup':'session.scope'}),patch.object(a,'metadata_once') as metadata_call,patch.object(a.cron,'record_manager') as launch:
            with self.assertRaisesRegex(RuntimeError,'persistent cron'):a.dispatch('unused')
        launch.assert_not_called();metadata_call.assert_not_called()

    def test_existing_stop_cannot_be_resumed(self):
        out=self.root/a.OUTPUT;out.mkdir(parents=True);(out/'STOP.json').write_text('{}')
        with patch.object(a,'read_spec',return_value=(self.root,{})),patch.object(a,'verify') as verify:
            with self.assertRaises(FileExistsError):a.parent('unused')
        verify.assert_not_called()

    def parent_context(self,resource_error=False,metadata_error=False):
        # A stack of mocks lets the real activation/driver/finalizers run on toy records.
        from contextlib import ExitStack
        out=self.root/a.OUTPUT;out.mkdir(parents=True)
        stack=ExitStack();self.addCleanup(stack.close)
        spec=dict(source_commit='toy',manager_SOURCE='toy')
        stack.enter_context(patch.object(a,'read_spec',return_value=(self.root,spec)))
        stack.enter_context(patch.object(a,'verify',return_value={'toy':'binding'}))
        stack.enter_context(patch.object(a.c,'forbidden_science',return_value=nullcontext()))
        stack.enter_context(patch.object(a.managed,'connect_legacy',return_value=nullcontext()))
        stack.enter_context(patch.object(a.admission,'wait_for_resources',side_effect=RuntimeError('toy no resources') if resource_error else None,return_value=1))
        meta=stack.enter_context(patch.object(a,'metadata_once',side_effect=ValueError('toy metadata failed') if metadata_error else None,return_value=self.control))
        load=stack.enter_context(patch.object(a.c.n3,'load_certified_services',return_value=(None,None,self.md.sampler,self.scopes,{'total_engine_coordinate_upper_exact':'1/999'})))
        rng=stack.enter_context(patch.object(a.c.n3,'pcg64_streams'))
        stack.enter_context(patch.object(a.c,'ProcessPoolExecutor'))
        collector=Collector(out,self.control);collector.raw=NS(classical={})
        stack.enter_context(patch.object(a.pilot,'connect_legacy_pipeline',return_value=collector))
        stack.enter_context(patch.object(a.c,'reconcile',return_value={}))
        return out,meta,load,rng

    def test_resource_failure_is_zero_metadata_decode_rng_and_no_retry(self):
        out,meta,load,rng=self.parent_context(resource_error=True)
        self.assertEqual(a.parent('toy'),1);meta.assert_not_called();load.assert_not_called();rng.assert_not_called()
        self.assertTrue((out/'STOP_REQUESTED.json').exists());self.assertTrue((out/'STOP.json').exists())

    def test_metadata_failure_stops_without_decode_or_rng(self):
        out,meta,load,rng=self.parent_context(metadata_error=True)
        self.assertEqual(a.parent('toy'),1);self.assertEqual(meta.call_count,1);load.assert_not_called();rng.assert_not_called()
        calls=json.loads((out/'SCIENCE_SETUP_CALL_COUNTS.json').read_text())['calls']
        self.assertEqual(calls['metadata_selection_means']['returned'],0)

    def test_one_mock_parent_224_records_single_metadata_decode_stream_book(self):
        out,meta,load,rng=self.parent_context()
        self.assertEqual(a.parent('toy'),0)
        self.assertEqual(meta.call_count,1);self.assertEqual(load.call_count,1)
        rng.assert_called_once_with(20261109,a.pilot.domains(),path_word_cap=64)
        accounting=json.loads((out/'SCV_ACCOUNTING.json').read_text());self.assertEqual(accounting['physical_shots_returned'],224)
        self.assertEqual(json.loads((out/'STREAMS.json').read_text())['children'],24)
        self.assertEqual(a.semantic_status(out,-9),1)


if __name__=='__main__':unittest.main()
