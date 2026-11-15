"""Reuse ten-fragment invented fixture; prefix9 context and real path worker."""
import json
from pathlib import Path
import sys
import threading
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_pr_df_ld9_m6_price as L
from test_pr_df_ld7_prices import env,v
R=L.R

@pytest.mark.parametrize('bad_seed',[False,True])
def test_prefix9_actual_context_worker_and_axis_receipts(env,tmp_path,monkeypatch,bad_seed):
    runtime,preps,_=env
    preps[9]=runtime._tools().h['prepare'](runtime.ham,9)
    config,old=v.request(env,prefix=9,q=2,R=1)
    total=runtime.ham.constant+preps[9].extracted_identity_coefficient
    req=R.D.from_b(config,0,old['physical_context']['physical_candidate'],input_source=v.INPUT,execution_source=v.SOURCE,expected_scalar=total)
    ctx=runtime.context(req);events=v.path(runtime,req);seeds=[r.seed for r in events]
    ctx.h=dict(ctx.h,sample_requests=lambda *a:(events,seeds,R.D._law_records(ctx,req['physical_context']['physical_candidate'],events)[1]))
    compile=ctx.compile;seen=[]
    def fake(qc,*a,**kw):
        scalar=[op for op in qc.data if op.operation.name=='p'and len(op.qubits)==1 and qc.find_bit(op.qubits[0]).index==8 and float(op.operation.params[0])==-total*.25]
        assert len(scalar)==1;seen.append(qc);return compile(qc,*a,**kw)
    ctx.compile=fake
    monkeypatch.setattr(R,'PUBLIC',str(tmp_path));monkeypatch.setattr(R,'FAILURE',threading.Event());monkeypatch.setattr(R,'STATE',dict(contexts={0:ctx},runtimes={0:runtime},inputs=dict(cache_root=str(tmp_path/'cache'))))
    (tmp_path/'acquisition').mkdir()
    cell=dict(m=0,request=req,cohort='ARTIFICIAL',seeds=[seeds if not bad_seed else [99,99]])
    result=R.acquire_path((cell,0))
    if bad_seed:assert result['status']=='STOP'and not seen
    else:
        assert result['status']=='VALID'and len(seen)==2
        assert result['record']['resolved_context']['deterministic_fragments']==list(range(9))and result['record']['resolved_context']['randomized_fragments']==[9]
        assert result['record']['resolved_context']['scalar']==total
        assert len({r['fingerprint']for r in result['record']['records']})==2
        for r in result['record']['records']:
            raw=json.loads((tmp_path/'acquisition/m0/p0'/(R.D.AXES[r['axis']]+'.json')).read_text())
            assert r['metrics']==raw['metrics']and r['fingerprint']==raw['fingerprint']
