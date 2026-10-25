"""B1 mocks and toy-law/statistics tests; no physical input/operator/circuits."""
from collections import Counter,OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json,subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.b2d_evaluator import b1_native as b1

METRICS=dict(rz_count=10,rz_depth=5,cx_count=4,cx_depth=4,total_depth=13,circuit_size=20,measurement_count=1)

class Stop:
    def is_set(self):return False

class QC:
    def remove_final_measurements(self,inplace=False):return self

class FakeNative:
    def __init__(self):self.cache={};self.compile_cache={};self.log=[];self.Operator=lambda c:NS(data=[[0j]])
    def render(self,*args,**kw):self.log.append('build');return QC()
    def reference(self,*args,**kw):return [[0j]]
    def controlled(self,u):return u
    def wrapper(self,*args):return QC()
    def wrapper_reference(self,*args):return [[0j]]
    def compile(self,c):self.log.append('compile');return c
    def check(self,*args):self.log.append('gate');return 0.
    def expectation(self,*args,**kw):self.log.append('expectation');return dict(actual=0.,reference=0.,error=0.)
    def measure(self,c):self.log.append('measure');return dict(METRICS)

class MockInput:
    def __init__(self,run,ledger):
        self.input_diagnostics=dict(mock=True)
        self.prep=NS(rte_preparation=NS(symbolic_tail=NS(components=[NS(component_id='opaque',probability=1.,coefficient_sign=1,diagonal_pauli_support=(0,))])))

def mock_draw(native,row,index,stages=None):
    details=[]
    for occurrence,(spec,word) in enumerate(stages):
        details.append(dict(occurrence=occurrence,spec={},orders=(0,2,4) if spec.cutoff==4 else (0,2),order_probabilities=(1.,0.,0.) if spec.cutoff==4 else (1.,0.),events=[dict(order=0,indices=(0,)) for _ in word]))
    return dict(id=b1.sample_id(row,index),row=row,index=index,words=stages,details=details,key=b1.b0.case_key(row)+':'+str(index%3))

def mock_worker(slot,jobs,run,resources,queue,stop):
    queue.put(dict(type='ready',worker=slot,input_diagnostics=dict(mock=True)))
    for job in jobs:
        if stop.is_set():break
        queue.put(dict(type='sample',record=dict(id=job['id'],candidate=job['row']['candidate_id'],T=str(job['row']['case']['total_time']),sample=job['index'],worker=slot,axes={axis:dict(metrics=dict(METRICS)) for axis in ('cosine','sine')})))
    queue.put(dict(type='done',worker=slot,closure_calls={},cache={},wall_seconds=0.,cpu_seconds=0.,peak_RSS_KiB=0))

def failed_worker(slot,jobs,run,resources,queue,stop):
    queue.put(dict(type='ready',worker=slot,input_diagnostics=dict(mock=True)))
    if slot==0:
        stop.set();queue.put(dict(type='failed',worker=slot,reason='OPAQUE_SCIENTIFIC_FAILURE',context=dict(mock=True)))
    queue.put(dict(type='done',worker=slot,closure_calls={},cache={}))

def mock_statistics(records):
    return dict(groups={b1.b0.case_key(row)+':'+axis:{} for row in b1.b0.cases() for axis in ('cosine','sine')},mock=True)

class B1Tests(unittest.TestCase):
    def test_fixed_metadata_and_scope_rejection(self):
        s=b1.scope(2)
        self.assertEqual((s['trajectories'],s['compile_requests'],s['operator_gates'],s['expectations'],s['sampled_short_steps']),(3072,6144,18432,12288,51200))
        ledger=b1.ParentLedger(s)
        with self.assertRaises(b1.Stage1Failure):ledger.call('b1.receive',lambda:None,identity='unknown')
        native=b1.NativeLedger(bounds=dict(fixture=1))
        native.call('b0.fixture',lambda:None)
        with self.assertRaises(b1.Stage1Failure):native.call('b0.fixture',lambda:None)
        with self.assertRaises(b1.Stage1Failure):native.call('b0.analytic_mean',lambda:None)

    def test_toy_PCG64_law_signed_K4_and_schedule_independent_seeds(self):
        from trotterlib.rte import RTEComponent,finite_rte_distribution,_make_event
        import math
        components=tuple(RTEComponent('c'+str(i),p,p,(-1 if i==1 else 1)) for i,p in enumerate((.2,.3,.5)))
        class Toy:
            run=dict(source='MOCK',inputs=dict(input_data_sha256='MOCK'))
            prep=NS(rte_preparation=NS(symbolic_tail=NS(components=components)))
            def config(self,spec):return None,finite_rte_distribution(math.copysign(.2,float(spec.duration)),spec.cutoff)
            def event(self,spec,descriptor):
                order,indices=descriptor;law=self.config(spec)[1]
                return _make_event(indices,components,law,law.orders.index(order))
        row=next(x for x in b1.b0.cases() if x['candidate_id']=='R4' and x['case']['total_time']<0)
        stages=b1.b0.trajectory(row,0);toy=Toy()
        forward=[b1.draw(toy,row,i,stages=stages) for i in (0,1,2)]
        reverse={j['id']:j for j in (b1.draw(toy,row,i,stages=stages) for i in (2,0,1))}
        for job in forward:
            self.assertEqual(job,reverse[job['id']])
            for item in job['details']:
                for event in item['events']:
                    self.assertIn(event['order'],(0,2,4));self.assertEqual(len(event['indices']),event['order']+1)
                    self.assertGreater(event['probability'],0.)
                    self.assertEqual(math.copysign(1,event['rotation_angle']),math.copysign(1,float(item['spec']['duration'])))
        seeds={b1.subseed(row,i,o,t) for i in range(5) for o in range(6) for t in range(2)}
        self.assertEqual(len(seeds),60)
        self.assertEqual(len([j for batch in b1.allocate(forward,2) for j in batch]),3)

    def test_paired_actual_circuit_cache_and_all_gates_before_metrics(self):
        native=FakeNative();memo=OrderedDict();counts=Counter();job=dict(key='physical-key',id='B3:9/50:0',index=0,row=b1.b0.cases()[0],words=())
        first=b1.evaluate(native,job,memo,2,counts,Stop());second=b1.evaluate(native,dict(job,id='B3:9/50:1',index=1),memo,2,counts,Stop())
        self.assertEqual(native.log.count('compile'),2);self.assertEqual(native.log.count('gate'),12)
        self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit'])
        self.assertEqual(counts['hits'],2);self.assertEqual(counts['misses'],2)
        self.assertLess(max(i for i,x in enumerate(native.log[:native.log.index('measure')]) if x in ('gate','expectation')),native.log.index('measure'))
        self.assertIn('compiled',memo['physical-key']['axes']['cosine'])

    def test_correctness_failure_preserves_context_and_never_measures(self):
        native=FakeNative();native.check=lambda *args:(_ for _ in ()).throw(b1.Stage1Failure('OPAQUE_FAILURE'))
        job=dict(key='k',id='B3:9/50:0',index=0,row=b1.b0.cases()[0],words=())
        with patch.object(b1,'circuit_blob',lambda qc:dict(mock=True)):
            with self.assertRaises(b1.Stage1Failure) as error:b1.evaluate(native,job,OrderedDict(),1,Counter(),Stop())
        self.assertEqual(error.exception.b1_context['id'],job['id']);self.assertNotIn('measure',native.log)
        self.assertTrue(error.exception.b1_context['circuits'])

    def test_unweighted_statistics_sample_count_variance_and_intervals(self):
        records=[]
        for row in b1.b0.cases():
            for i in range(256):
                m={k:(1 if k=='measurement_count' else i) for k in METRICS}
                records.append(dict(candidate=row['candidate_id'],T=str(row['case']['total_time']),sample=i,axes={axis:dict(metrics=m) for axis in ('cosine','sine')}))
        stats=b1.statistics(list(reversed(records)))
        x=stats['groups']['B3:9/50:cosine']['rz_count']
        self.assertEqual(x['sample_count'],256);self.assertEqual(x['mean'],127.5)
        self.assertAlmostEqual(x['unbiased_sample_variance'],sum((i-127.5)**2 for i in range(256))/255)
        self.assertAlmostEqual(x['standard_error'],(x['unbiased_sample_variance']/256)**.5)
        self.assertEqual(stats,b1.statistics(records))

    def test_rare_event_unobserved_is_not_zero_cost(self):
        row=next(x for x in b1.b0.cases() if x['candidate_id']=='R4')
        job=mock_draw(None,row,0,stages=b1.b0.trajectory(row,0));c=b1.coverage([job])[b1.b0.case_key(row)]
        self.assertFalse(c['max_order_observed']);self.assertEqual(c['status'],'INCONCLUSIVE_RARE_EVENT_COVERAGE')

    def test_spawn_streaming_full_mock_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as d:
            resources=dict(workers=2,memo_entries=1,stop_available_bytes=0)
            run=dict(root=ROOT,output=Path(d)/'fresh',scope=b1.scope(2),source='MOCK',inputs={})
            status=b1.run_once(run,resources,native_factory=MockInput,draw_function=mock_draw,worker_function=mock_worker,statistics_function=mock_statistics,mock_only=True)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',status)
            self.assertEqual(status['mode'],'MOCK_ONLY')
            self.assertEqual(len((run['output']/'samples.jsonl').read_text().splitlines()),3072)
            self.assertTrue((run['output']/'STOP.json').exists())
            with self.assertRaises(FileExistsError):b1.run_once(run,resources,mock_only=True)

    def test_spawn_failure_stops_without_retry_or_final_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            resources=dict(workers=2,memo_entries=1,stop_available_bytes=0)
            run=dict(root=ROOT,output=Path(d)/'fail',scope=b1.scope(2),source='MOCK',inputs={})
            status=b1.run_once(run,resources,native_factory=MockInput,draw_function=mock_draw,worker_function=failed_worker,statistics_function=mock_statistics,mock_only=True)
            self.assertEqual(status['status'],'FAILED_MANDATORY_STOP',status)
            self.assertTrue((run['output']/'failure.json').exists());self.assertFalse((run['output']/'result.json').exists())

    def test_inspect_no_science_import_or_output(self):
        code="import sys;sys.path.insert(0,'src');from trottertracks.b2d_evaluator import b1_native as b; b.scope(2);assert not any(x.split('.')[0] in ('numpy','qiskit','scipy','mpmath') for x in sys.modules)"
        result=subprocess.run(['/usr/bin/python3.11','-I','-B','-S','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
