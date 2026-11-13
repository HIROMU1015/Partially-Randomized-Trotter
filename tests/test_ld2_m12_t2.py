"""T2 scope and persistence on mocked inputs only; no H4 numerical leaf."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from trottertracks import ld2_m12_t2 as T

ROOT=Path(__file__).resolve().parents[1]
SOURCE='ARTIFICIAL_EXECUTION_SOURCE'


@pytest.fixture
def metadata():
    p=T.plan(ROOT)
    r=T.L.request_for(ROOT,'PR-LD2',12,execution_source={'commit':SOURCE})
    return p,r


def seal(r):
    r['context_key']=T.D._key(r['physical_context'])
    r['request_id']=T.D._key({k:v for k,v in r.items() if k!='request_id'})


@pytest.mark.parametrize('change',[
    lambda r:r.update(m=11),lambda r:r.update(axes=['X']),
    lambda r:r['physical_context']['physical_candidate'].update(prefix=1)])
def test_only_fixed_scope_is_accepted(metadata,change):
    _,r=metadata;change(r);seal(r)
    with pytest.raises(ValueError,match='T2_FIXED'):T.check_scope(r)


def test_source_mismatch_precedes_scientific_processing(metadata,monkeypatch):
    p,_=metadata
    monkeypatch.setattr(T.subprocess,'check_output',lambda *a,**k:b'OTHER_SOURCE')
    monkeypatch.setattr(T,'restore_h4',lambda *a:pytest.fail('no physical restore'))
    with pytest.raises(ValueError,match='SOURCE_HEAD'):T.preflight(ROOT,SOURCE,p)


def test_resource_stop_uses_real_headroom_and_self_growth_only():
    before=dict(process=dict(VmRSS_bytes=100,write_bytes=100),memory=dict(MemAvailable=1000),disk_available_bytes=1000)
    current=deepcopy(before);current.update(PSI='HIGH')
    assert not T.resource_exhaustion(before,current)
    current['process']['VmRSS_bytes']=200;current['memory']['MemAvailable']=150
    assert T.resource_exhaustion(before,current)
    current=deepcopy(before);current['process']['write_bytes']=200;current['disk_available_bytes']=150
    assert T.resource_exhaustion(before,current)


def fixture_run(tmp_path,metadata,monkeypatch,*,bad=None,fail_Y=False):
    p,r=metadata;calls=[]
    pub=tmp_path/'public';private=tmp_path/'private';pub.mkdir();private.mkdir()
    tail=NS(tail_hash='INVENTED_TAIL',lambda_r=float(T.D.p.accounting.exact_scalar(r['physical_context']['split']['lambda_R'])),components=[])
    prep=NS(preparation_hash='INVENTED_PREP',rte_preparation=NS(symbolic_tail=tail),
            deterministic_fragment_indices=(0,1),randomized_block_indices=tuple(range(2,10)))
    ctx=NS(prep=prep,scalar=r['physical_context']['expected_scalar'],
           compile=lambda *a,**kw:NS(transpiled_circuit=NS(data=[1])))
    if bad=='lambda':tail.lambda_r+=.1
    if bad=='scalar':ctx.scalar+=.1
    if bad=='fragments':prep.deterministic_fragment_indices=(0,)
    class Events:
        def __len__(self):return 2559  # metadata length, no events constructed
    requests=[NS(rte_occurrence=NS(events=Events()))]*4096
    seed=T.P.trial_seed_function(ROOT)
    seeds=[seed(r['physical_context']['physical_candidate'],T.COHORT,0,o,0) for o in range(4096)]
    monkeypatch.setattr(T.D,'_law_records',lambda *a:(seeds,[]))
    class Runtime:
        def __init__(self,*a,**kw):calls.append('runtime')
        def context(self,req):assert req is r;return ctx
        def sample(self,req,cohort,index):
            assert req is r and cohort==T.COHORT and index==0
            calls.append('sample');return requests
    class Adapter:
        def __init__(self,root,runtime,**kw):self.observer=kw['observer']
        def acquire(self,req,path,*,cohort,index,directory,conditional):
            assert req is r and path is requests and not conditional
            assert cohort==T.COHORT and index==0
            calls.append('acquire');directory.mkdir()
            for axis,name in [('X','cosine'),('Y','sine')]:
                if axis=='Y' and fail_Y:raise RuntimeError('INVENTED_Y_FAILURE')
                self.observer(name+'_compile','STARTED')
                ctx.compile(NS(data=[name]),None)
                T.write(directory/(name+'.json'),dict(axis=axis,fixture='ARTIFICIAL_FULL_MEASURED_WRAPPER'))
            return dict(new_compiles=2,records=[dict(axis='X'),dict(axis='Y')])
    def restore(*a):calls.append('restore');return object(),{'fixture':'NO_H4_INPUT'}
    def execute():
        T.run_worker(ROOT,SOURCE,p,r,{},pub,private,restore=restore,runtime_class=Runtime,adapter_class=Adapter)
    return execute,calls,pub,private


def test_one_sample_shared_by_two_distinct_axis_compiles(tmp_path,metadata,monkeypatch):
    execute,calls,pub,private=fixture_run(tmp_path,metadata,monkeypatch)
    execute()
    assert calls==['restore','runtime','sample','acquire']
    out=T.json.loads((pub/'T2_PATH_RESULT.json').read_text())
    assert [x['axis'] for x in out['compile_log']]==['X','Y']
    assert out['mean_compiled_price'] is None and out['qualified_expectation'] is False


@pytest.mark.parametrize('bad',['lambda','scalar','fragments'])
def test_bad_preparation_stops_before_sampling(tmp_path,metadata,monkeypatch,bad):
    execute,calls,pub,private=fixture_run(tmp_path,metadata,monkeypatch,bad=bad)
    with pytest.raises(ValueError,match='CONTEXT_DIFFERS'):execute()
    assert calls==['restore','runtime'] and (pub/'WORKER_STOP.json').exists()


def test_partial_axis_survives_exception_without_retry(tmp_path,metadata,monkeypatch):
    execute,calls,pub,private=fixture_run(tmp_path,metadata,monkeypatch,fail_Y=True)
    with pytest.raises(RuntimeError,match='INVENTED_Y_FAILURE'):execute()
    assert calls.count('sample')==calls.count('acquire')==1
    assert (private/'path/cosine.json').exists() and not (private/'path/sine.json').exists()
    assert (pub/'WORKER_STOP.json').exists() and not (pub/'T2_PATH_RESULT.json').exists()


def test_observer_keeps_original_artificial_wrapper_fingerprints_and_prices(tmp_path,monkeypatch):
    from compiled_price_artificial import fixture, F
    from test_compiled_price_fast import run
    s,ctx,_,cell,p,requests=fixture(q=1,R=2)
    p['runtime']['old_cache']=str(tmp_path/'empty_cache')
    before,old_compiles=run(s,ctx,cell,p,requests,tmp_path/'old',False)
    events=[];original=F.make_backend_class
    def observed(*a,**kw):
        return original(*a,**kw,observer=lambda name,status,seconds,value:events.append((name,status)))
    monkeypatch.setattr(F,'make_backend_class',observed)
    after,new_compiles=run(s,ctx,cell,p,requests,tmp_path/'new',True)
    assert old_compiles==new_compiles
    for key in before:
        if key!='source':assert before[key]==after[key]
    for axis in ('cosine','sine'):
        assert (axis+'_wrapper','COMPLETE') in events
        assert (axis+'_metrics_extract','COMPLETE') in events
