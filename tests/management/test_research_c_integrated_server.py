"""Server paths and contract invariants; no physical arrays are loaded."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_integrated_server as server
from trottertracks.c_integrated_execution import Run


def test_effective_contract_preserves_registered_scientific_scope(tmp_path):
    base=json.loads((ROOT/server.BASE_CONTRACT).read_text())
    path=tmp_path/server.BASE_CONTRACT;path.parent.mkdir(parents=True)
    path.write_text(json.dumps(base));before=path.read_bytes()
    c=server.effective_contract(tmp_path,base)
    metadata={'date_JST','status','source_code_commit','technical_source',
              'fresh_outputs','environment','server_delta'}
    assert {k:v for k,v in c.items() if k not in metadata} == {
        k:v for k,v in base.items() if k not in metadata}
    env=deepcopy(c['environment'])
    for key in ('fit_python','evaluation_python'):env[key]=base['environment'][key]
    assert env==base['environment']
    assert path.read_bytes()==before
    assert c['maximum_calls']['total']==172835
    assert c['sampling']['selection_samples_per_candidate_r']==32
    assert c['sampling']['final_samples_per_selected_pair']==64


def test_Run_uses_separate_contract_and_progress_family(tmp_path):
    contract=tmp_path/'technical/contract.json';contract.parent.mkdir()
    contract.write_text(json.dumps({'scope':'ARTIFICIAL_SERVER'}))
    source=tmp_path/'SOURCE.json';source.write_text('{}')
    run=Run(tmp_path,source,contract_path='technical/contract.json',family='fresh_server')
    assert run.c['scope']=='ARTIFICIAL_SERVER'
    assert run.progress_path==tmp_path/'fresh_server/execution_progress.jsonl'
    assert not run.native_timing_seen


def test_prepare_refuses_used_output_without_changing_old_contract(tmp_path):
    base=json.loads((ROOT/server.BASE_CONTRACT).read_text())
    path=tmp_path/server.BASE_CONTRACT;path.parent.mkdir(parents=True)
    path.write_text(json.dumps(base));before=path.read_bytes()
    (tmp_path/server.FAMILY/'development/run01').mkdir(parents=True)
    with pytest.raises(ValueError,match='NO_RETRY_RESUME'):server.prepare(tmp_path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('shared_candidate',[False,True])
def test_D_deduplication_does_not_serialize_independent_candidates(tmp_path,monkeypatch,shared_candidate):
    from types import SimpleNamespace
    from collections import Counter
    from trottertracks import c_integrated_downstream_execution as downstream
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps(dict(maximum_calls=dict(
        stage_totals={'development_downstream':63896},total=172835))))
    source=tmp_path/'SOURCE.json';source.write_text('{}')
    family=tmp_path/'family';family.mkdir()
    run=Run(tmp_path,source,contract_path='contract.json',family='family')
    barrier=threading.Barrier(2)
    seen=[]
    def save_units(out,units):
        folder=tmp_path/out;folder.mkdir(parents=True)
        (folder/'call_summary.json').write_text(json.dumps({'scientific_units':units}))
    class Worker:
        def __init__(self,candidate):self.candidate=candidate
        def request(self,operation,**kw):
            seen.append((id(self),operation))
            if operation=='D':
                if not shared_candidate:barrier.wait(timeout=10)
                save_units(kw['out'],141)
                return {'result':{'candidate':self.candidate}}
            if operation=='attach_D':return {'status':'ATTACHED'}
            assert operation=='final'
            save_units(kw['out'],618)
            return {'final':{'invented':True}}
        def close(self):pass
    workers=[Worker('same' if shared_candidate else arm) for arm in ('PR','ENTRY')]
    selected={};owners={}
    for arm,worker in zip(('PR','ENTRY'),workers):
        row=dict(id=arm,candidate_sha256=worker.candidate,input_sha256='I',r=32)
        selected['C0',64,arm]={'common_r32':row,'fair_retuned':row}
        owners[arm]=worker
    counts=Counter()
    def common_call(kind,label,fn):counts[kind]+=1;return fn()
    common_call.counts=counts
    out=tmp_path/'development';out.mkdir()
    g=dict(geometry='H4_100',selected=selected,evalworkers=workers,owners=owners,
        selection_calls=0,downstream_output=out/'downstream',
        downstream_common_call=common_call,output=out,physical=SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(downstream,'compare_final',lambda *a,**kw:{'synthetic':True})
    run.finals(g)
    assert sum(op=='D' for _,op in seen)==(1 if shared_candidate else 2)
    assert sum(op=='attach_D' for _,op in seen)==(1 if shared_candidate else 0)
    assert sum(op=='final' for _,op in seen)==2
