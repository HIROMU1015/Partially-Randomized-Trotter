"""Only invented matrices/events and fake compiler; no saved H4 load or sampler."""
from copy import deepcopy
from pathlib import Path
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_pr_df_ld_split as R

# Reuse the published synthetic fixture, preserving its no-sampler/input traps.
v=types.ModuleType('_split_artificial_fixture');v.__file__=str(ROOT/'tests/test_candidate_price_adapter.py')
source=R.blob(R.BASE,'tests/test_candidate_price_adapter.py').decode()
source=source.replace('from trottertracks import candidate_price_adapter as A','from run_pr_df_ld_split import D as A')
exec(compile(source,v.__file__,'exec'),v.__dict__)
env=v.env

@pytest.mark.parametrize('axes',[('X','Y'),('Y','X')])
def test_actual_prefix2_adapter_and_corrected_scalar(env,tmp_path,axes):
    runtime,preps,binding=env
    config,req=v.request(env,prefix=2,q=1,R=2,axes=axes)
    total=runtime.ham.constant+preps[2].extracted_identity_coefficient
    req=R.D.from_b(config,0,req['physical_context']['physical_candidate'],input_source=v.INPUT,execution_source=v.SOURCE,axes=axes,expected_scalar=total)
    ctx=runtime.context(req);calls=[];original=ctx.compile
    def fake(qc,*a,**kw):
        assert qc.num_qubits==9 and qc.count_ops().get('measure')==1
        # Match the single total-T scalar in each measured X/Y wrapper.
        ops=[x for x in qc.data if x.operation.name=='p' and len(x.qubits)==1 and qc.find_bit(x.qubits[0]).index==8 and float(x.operation.params[0])==-total*req['physical_context']['physical_candidate']['T']]
        assert len(ops)==1;calls.append(qc);return original(qc,*a,**kw)
    ctx.compile=fake
    result=R.D.Adapter(ROOT,runtime,cache_root=tmp_path/'none').acquire(req,v.path(runtime,req),cohort='ARTIFICIAL',index=0,directory=tmp_path/'out')
    assert len(calls)==2 and result['new_compiles']==2
    assert result['resolved_context']['deterministic_fragments']==[0,1]
    assert result['resolved_context']['randomized_fragments']==[2]
    assert result['resolved_context']['scalar']==total
    assert [r['axis']for r in result['records']]==list(axes)
    import json
    for r in result['records']:
        raw=json.loads((tmp_path/'out'/(R.D.AXES[r['axis']]+'.json')).read_text())
        assert r['metrics']==raw['metrics'] and r['fingerprint']==raw['fingerprint']


def test_tail_only_scalar_rejected_before_compile(env):
    runtime,preps,_=env;config,r=v.request(env,prefix=2,q=1,R=2)
    bad=R.D.from_b(config,0,r['physical_context']['physical_candidate'],input_source=v.INPUT,execution_source=v.SOURCE,expected_scalar=preps[2].extracted_identity_coefficient)
    with pytest.raises(ValueError,match='scalar'):runtime.context(bad)


def test_foreign_source_rejected_before_compile(env):
    runtime,_,_=env;_,r=v.request(env,prefix=2,source={'fixture':'FOREIGN'})
    with pytest.raises(ValueError,match='SOURCE'):runtime.context(r)
