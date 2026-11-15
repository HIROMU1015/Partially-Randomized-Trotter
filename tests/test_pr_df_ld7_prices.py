"""Ten invented DF fragments, supplied events, fake compiler; no molecular I/O."""
import json
from pathlib import Path
import sys
import threading
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_pr_df_ld7_prices as L
R=L.R
v=types.ModuleType('_ld7_artificial');v.__file__=str(ROOT/'tests/test_candidate_price_adapter.py')
s=R.blob(R.BASE,'tests/test_candidate_price_adapter.py').decode()
s=s.replace('from trottertracks import candidate_price_adapter as A','from run_pr_df_ld_split import D as A')
s=s.replace('(.07,-.13,.11)','tuple(.007*(i+1) for i in range(10))')
s=s.replace('np.array([.23,-.37,.19])','np.array([.023*(-1)**i for i in range(10)])')
s=s.replace('for k in (1,2)','for k in (1,2,7)').replace('rank=3','rank=10')
exec(compile(s,v.__file__,'exec'),v.__dict__)
env=v.env

@pytest.mark.parametrize('bad_seed',[False,True])
def test_prefix7_new_request_and_reused_actual_path_worker(env,tmp_path,monkeypatch,bad_seed):
    runtime,preps,_=env
    config,orig=v.request(env,prefix=7,q=2,R=1)
    total=runtime.ham.constant+preps[7].extracted_identity_coefficient
    # The generic invented B fixture has no fixed-grid physical_candidate
    # field. Supply its explicit physical row to the identical from_b leaf.
    req=R.D.from_b(config,0,orig['physical_context']['physical_candidate'],input_source=v.INPUT,execution_source=v.SOURCE,expected_scalar=total)
    runtime.input_source=req['physical_context']['input_source'];runtime.execution_source=req['physical_context']['execution_source']
    ctx=runtime.context(req);events=v.path(runtime,req)
    seeds=[e.seed for e in events];laws=R.D._law_records(ctx,req['physical_context']['physical_candidate'],events)[1]
    ctx.h=dict(ctx.h,sample_requests=lambda *args:(events,seeds,laws))
    original=ctx.compile;calls=[]
    def fake(qc,*a,**kw):
        matching=[x for x in qc.data if x.operation.name=='p'and len(x.qubits)==1 and qc.find_bit(x.qubits[0]).index==8 and float(x.operation.params[0])==-total*.25]
        assert len(matching)==1;calls.append(qc);return original(qc,*a,**kw)
    ctx.compile=fake
    monkeypatch.setattr(R,'PUBLIC',str(tmp_path));monkeypatch.setattr(R,'FAILURE',threading.Event())
    monkeypatch.setattr(R,'STATE',dict(contexts={0:ctx},runtimes={0:runtime},inputs=dict(cache_root=str(tmp_path/'cache'))))
    (tmp_path/'acquisition').mkdir()
    cell=dict(m=0,request=req,cohort='ARTIFICIAL',seeds=[seeds if not bad_seed else [99]*len(seeds)])
    result=R.acquire_path((cell,0))
    if bad_seed:
        assert result['status']=='STOP'and not calls
    else:
        assert result['status']=='VALID'and len(calls)==2
        assert result['record']['resolved_context']['deterministic_fragments']==list(range(7))
        assert result['record']['resolved_context']['randomized_fragments']==[7,8,9]
        assert result['record']['resolved_context']['scalar']==total
        for rec in result['record']['records']:
            raw=json.loads((tmp_path/'acquisition/m0/p0'/(R.D.AXES[rec['axis']]+'.json')).read_text())
            assert raw['metrics']==rec['metrics']and raw['fingerprint']==rec['fingerprint']
