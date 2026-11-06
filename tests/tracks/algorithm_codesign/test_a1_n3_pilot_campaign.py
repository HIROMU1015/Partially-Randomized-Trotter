"""Fixed invented outcomes/prices; no real engine, PRNG, state or compiler."""
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from dataclasses import replace
from unittest.mock import patch,Mock
import json,os,subprocess,unittest
from trottertracks.algorithm_codesign import a1_n3_pilot_campaign as p


def fixture():
    cv=p.cv
    law=cv.OccurrenceLaw(cv.CategoricalLaw((0,2),(F(3,4),F(1,4)),'toy'),cv.CategoricalLaw(('a','b'),(F(1,4),F(3,4)),'toy'))
    sampler=p.n3.ConditionalSampler(law,'toy','a','b')
    scopes={a:cv.Scope('toy-target','toy','toy-wrapper','m0',a,F(1)) for a in ('X','Y')}
    return sampler,scopes


class MockCollector:
    def __init__(self,sampler,scopes,campaign='toy-pilot',failure=None):
        self.s,self.scopes,self.campaign=sampler,scopes,campaign;self.failure=failure
        self.calls=[];self.received=[];self.lifecycle=[]

    def collect(self,method,axis,stage,quotas):
        self.calls.append((method,axis,stage,quotas));rows=[];costs={}
        for cell,count in quotas:
            sampling='B1-OL' if method=='A1' and stage=='train' else method
            path=self.s.draw(sampling,cell,lambda:0)
            token=p.n3.Domain(method,axis,stage,cell).token
            records=tuple(p.cv.Measurement(token,str(i),(path.event,),i%2,F(1),'toy-path','toy-wrapper') for i in range(count))
            rows.append((cell,p.cv.Batch(self.scopes[axis],stage,records)));costs[cell]=tuple(11 for _ in range(count))
        envelope=p.Envelope(self.campaign,p.PURPOSE,method,axis,stage,tuple(rows),costs)
        if self.failure=='foreign':envelope=replace(envelope,campaign='old-campaign')
        if self.failure=='missing_price':
            costs=dict(costs);k=quotas[0][0];costs[k]=(None,)+costs[k][1:];envelope=replace(envelope,rz_by_cell=costs)
        if self.failure=='quota':envelope=replace(envelope,rows=envelope.rows[:-1])
        self.received.append(envelope);return envelope

    def request_stop(self,reason):self.lifecycle.append('STOP_REQUESTED')
    def finalize(self):self.lifecycle.append('DRAIN_AND_RECONCILE');return dict(mock_only=True)


class PilotCampaignTests(unittest.TestCase):
    def test_full_752_mock_protocol_and_serialization(self):
        s,scopes=fixture();backend=MockCollector(s,scopes)
        with TemporaryDirectory() as tmp,patch.object(p.n3,'load_certified_services',side_effect=AssertionError('real bind forbidden')),patch.object(p.n3,'pcg64_streams',side_effect=AssertionError('real RNG forbidden')):
            driver=p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
            self.assertEqual(driver.run(),'PILOT_DIAGNOSTIC_COMPLETED')
            self.assertEqual(sum(sum(n for _,n in q) for m,a,g,q in backend.calls),752)
            self.assertEqual({m for m,a,g,q in backend.calls},{'A1','B1-OL'})
            self.assertEqual(len(driver.seen),752)
            for axis in ('X','Y'):
                data=json.loads((Path(tmp)/f'B1-OL_{axis}_ALLOCATION.json').read_text())
                self.assertEqual(sum(data['counts']),128);self.assertGreaterEqual(min(data['counts']),4)
                frozen=json.loads((Path(tmp)/f'A1_{axis}_FROZEN.json').read_text())
                self.assertEqual(frozen['rank'],4);self.assertEqual(frozen['training_count'],80)
            account=json.loads((Path(tmp)/'PILOT_ACCOUNTING.json').read_text())['received_group_methods']
            self.assertEqual(account['A1']['bits'],416);self.assertEqual(account['B1-OL']['bits'],336)
            self.assertEqual(account['A1']['total_RZ'],416*11)
            self.assertEqual(backend.lifecycle,['DRAIN_AND_RECONCILE'])
            self.assertEqual(json.loads((Path(tmp)/'PILOT_RESULT.json').read_text())['formal_precision'],'NOT_CERTIFIED')
            self.assertEqual(p.semantic_exit_code(tmp),0)
            self.assertEqual(json.loads((Path(tmp)/'STOP.json').read_text())['formal_precision'],'NOT_CERTIFIED')

    def test_foreign_campaign_stops_before_fit_and_next_group(self):
        s,scopes=fixture();backend=MockCollector(s,scopes,failure='foreign')
        with TemporaryDirectory() as tmp,patch.object(p.cv,'fit_and_freeze',side_effect=AssertionError('must not fit foreign data')):
            driver=p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
            self.assertEqual(driver.run(),'STOPPED');self.assertEqual(len(backend.calls),1)
            self.assertEqual(backend.lifecycle,['STOP_REQUESTED','DRAIN_AND_RECONCILE'])

    def test_missing_price_keeps_bits_and_UNKNOWN_cost(self):
        s,scopes=fixture();backend=MockCollector(s,scopes,failure='missing_price')
        with TemporaryDirectory() as tmp:
            driver=p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
            self.assertEqual(driver.run(),'STOPPED');self.assertEqual(len(backend.calls),1)
            self.assertTrue((Path(tmp)/'A1_X_train_DATA.json').exists())
            account=json.loads((Path(tmp)/'PILOT_ACCOUNTING.json').read_text())['received_group_methods']['A1']
            self.assertEqual(account['bits'],80);self.assertEqual(account['priced'],79);self.assertEqual(account['total_RZ'],'UNKNOWN')
            self.assertEqual(p.semantic_exit_code(tmp),1)
            self.assertEqual(backend.lifecycle,['STOP_REQUESTED','DRAIN_AND_RECONCILE'])

    def test_bad_quota_and_range_stop_without_extra_acquisition(self):
        s,scopes=fixture()
        for failure in ('quota',None):
            backend=MockCollector(s,scopes,failure=failure)
            with TemporaryDirectory() as tmp:
                driver=p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
                if failure:
                    self.assertEqual(driver.run(),'STOPPED')
                else:
                    original=p.cv.fit_and_freeze
                    def large(bank,training):return replace(original(bank,training),coefficients=(F(9),F(0),F(0)))
                    with patch.object(p.cv,'fit_and_freeze',side_effect=large):self.assertEqual(driver.run(),'STOPPED')
                    self.assertTrue((Path(tmp)/'A1_X_FROZEN.json').exists())
                self.assertEqual(len(backend.calls),1)

    def test_duplicate_trial_cannot_enter_probe_or_evaluation(self):
        s,scopes=fixture();backend=MockCollector(s,scopes)
        with TemporaryDirectory() as tmp:
            driver=p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
            q=[(k,16) for k,_ in s.cells('B1-OL')]
            driver.acquire('A1','X','train',q)
            with self.assertRaisesRegex(ValueError,'reused'):driver.acquire('A1','X','train',q)

    def test_zero_variance_is_inconclusive_not_improvement(self):
        s,scopes=fixture();backend=MockCollector(s,scopes)
        original=backend.collect
        def constant(*args):
            e=original(*args)
            if e.stage=='prod':e=replace(e,rows=tuple((k,replace(b,records=tuple(replace(r,bit=0) for r in b.records))) for k,b in e.rows))
            return e
        backend.collect=constant
        with TemporaryDirectory() as tmp:
            driver=p.PilotDriver(tmp,'toy-pilot',s,scopes,backend);driver.run()
            result=json.loads((Path(tmp)/'PILOT_RESULT.json').read_text())
            self.assertEqual(set(result['classifications'].values()),{'INCONCLUSIVE'})

    def test_legacy_adapter_rejects_old_SOURCE_without_collecting(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'SOURCE.json').write_text(json.dumps(dict(campaign='old-B',purpose='old-campaign')))
            raw=NS(out=out,group=Mock(),trials=[],costs=[])
            with self.assertRaisesRegex(ValueError,'old campaign'):p.LegacyCollector(raw,'toy-pilot',NS(out=out))
            raw.group.assert_not_called()

    def test_pending_cli_in_minimal_environment_is_nonexecuting(self):
        root=Path(__file__).resolve().parents[3];entry=root/'scripts/tracks/algorithm_codesign/run_a1_n3_pilot.py'
        python='/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python'
        env={k:os.environ[k] for k in ('HOME','PATH','LANG') if k in os.environ}
        with TemporaryDirectory() as tmp:
            result=subprocess.run([python,'-B',str(entry),'--show-plan'],cwd=tmp,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['total'],752)
            blocked=subprocess.run([python,'-B',str(entry),'--run'],cwd=tmp,env=env,capture_output=True,text=True)
            self.assertEqual(blocked.returncode,2);self.assertIn('PREPARATION_ONLY',blocked.stderr)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_legacy_connection_uses_existing_worker_bit_before_price_with_toy_services(self):
        from concurrent.futures import Future
        import threading
        sampler,scopes=fixture();facade=p.n3.MeasurementFacade(lambda *args:F(1,2),scopes)
        metrics=NS(rz_count=11,rz_depth=2,cx_count=3,cx_depth=2,total_depth=9,circuit_size=20,qubit_count=3,compiler_settings_hash='toy',actual_circuit_fingerprint='toy')
        services=(facade,NS(price=Mock(return_value=(metrics,True))),sampler,scopes,{})
        class ImmediatePool:
            def submit(self,fn,job):
                f=Future();f.set_result(fn(job));return f
            def shutdown(self,**kwargs):pass
        streams=p.n3.Streams(64)
        for key,_ in sampler.cells('B1-OL'):streams.bind(p.n3.Domain('B1-OL','X','train',key),lambda:0,lambda:0)
        with TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'SOURCE.json').write_text(json.dumps(dict(campaign='toy-pilot',purpose=p.PURPOSE)))
            stop=threading.Event();runtime=NS(out=out,stop=stop,request_stop=Mock())
            with patch.object(p.legacy,'_SERVICES',services),patch.object(p.legacy,'_STOP_EVENT',stop),patch.object(p.legacy,'_WORKER',(out,out/'worker_toy.jsonl',None,None)):
                driver=p.connect_legacy_pipeline(out,'toy-pilot',services,streams,ImmediatePool(),2,runtime)
                rows,prices=driver.acquire('B1-OL','X','train',[(k,8) for k,_ in sampler.cells('B1-OL')])
                self.assertEqual(sum(len(b.records) for _,b in rows),40)
                result=driver.collector.finalize();self.assertEqual(result['B1-OL']['bits_saved'],40)
            events=[json.loads(line)['event'] for line in (out/'worker_toy.jsonl').read_text().splitlines()]
            self.assertLess(events.index('BIT_SAVED'),events.index('PRICE_SAVED'))
            self.assertFalse(runtime.request_stop.called)

    def test_output_invocation_is_exclusive(self):
        s,scopes=fixture();backend=MockCollector(s,scopes)
        with TemporaryDirectory() as tmp:
            p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
            before=(Path(tmp)/'PILOT_DRIVER_INVOCATION.json').read_bytes()
            with self.assertRaises(FileExistsError):p.PilotDriver(tmp,'toy-pilot',s,scopes,backend)
            self.assertEqual(before,(Path(tmp)/'PILOT_DRIVER_INVOCATION.json').read_bytes())

    def test_legacy_adapter_refuses_warm_cache_and_changed_SOURCE(self):
        import threading
        with TemporaryDirectory() as tmp:
            out=Path(tmp);source=out/'SOURCE.json';source.write_text(json.dumps(dict(campaign='toy-pilot',purpose=p.PURPOSE)))
            raw=NS(out=out,group=Mock(),trials=[],costs=[]);stop=threading.Event();runtime=NS(out=out,stop=stop)
            (out/'cache').mkdir();(out/'cache/old.sqlite').write_bytes(b'old-cache')
            with patch.object(p.legacy,'_STOP_EVENT',stop):
                with self.assertRaisesRegex(ValueError,'fresh independent'):p.LegacyCollector(raw,'toy-pilot',runtime)
                (out/'cache/old.sqlite').unlink();collector=p.LegacyCollector(raw,'toy-pilot',runtime)
                source.write_text('{}')
                with self.assertRaisesRegex(ValueError,'SOURCE changed'):collector.collect('B1-OL','X','train',())
            raw.group.assert_not_called()

    def test_domains_are_32_unique_and_disjoint_by_axis_stage_method(self):
        domains=p.pilot_domains();self.assertEqual(len(domains),32);self.assertEqual(len(set(domains)),32)
        self.assertFalse(any(d.method in ('B0','B1-O') or d.stage=='cal' for d in domains))


if __name__=='__main__':unittest.main()
