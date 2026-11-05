"""Top-level dependency tests with no numerical engines or molecular inputs."""
from pathlib import Path
from unittest.mock import Mock
import json
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_integrated_execution as m


def fixture(tmp_path):
    c=dict(fresh_outputs=['dev/run01','target/run01'],maximum_calls=dict(
        stage_totals=dict(freeze_transfer_recipe=1,integrated_summary=1),total=172835))
    contract=tmp_path/m.CONTRACT;contract.parent.mkdir(parents=True);contract.write_text(json.dumps(c))
    source=tmp_path/'SOURCE.json';source.write_text('{}')
    (tmp_path/m.FAMILY).mkdir(parents=True)
    return m.Run(tmp_path,source)


@pytest.mark.parametrize('transfer',[True,False])
def test_transfer_is_not_opened_before_frozen_development_recipe(tmp_path,monkeypatch,transfer):
    run=fixture(tmp_path);events=[]
    def initialize(geometry,out,strengths=None):
        events.append('init:'+geometry)
        if geometry=='H4_115':assert events.index('freeze')<events.index('init:H4_115')
        return dict(geometry=geometry,rows=[])
    run.initialize=initialize
    run.optimize=lambda g:(events.append('opt:'+g['geometry']) or g)
    run.selection=lambda g:(events.append('selection:'+g['geometry']) or g)
    def finals(g):events.append('final:'+g['geometry']);run.summary[g['geometry']]={'status':'ARTIFICIAL'}
    run.finals=finals
    def freeze(rows):
        events.append('freeze')
        if not transfer:raise ValueError('NO_COMMON_COMPLETE_STRENGTH_FOR_TRANSFER')
        return {'PR':.1,'ENTRY':.3}
    monkeypatch.setattr(m.plan,'freeze_transfer_strength',freeze)
    run.execute()
    stop=json.loads((tmp_path/m.FAMILY/'INTEGRATED_STOP.json').read_text())
    assert stop['mandatory_stop'] and not stop['failed'] and stop['completed_stage_units']==2
    assert ('init:H4_115' in events)==transfer
    assert events[:3]==['init:H4_100','opt:H4_100','selection:H4_100']


def test_dependency_failure_preserves_stop_and_does_not_advance(tmp_path):
    run=fixture(tmp_path)
    run.initialize=Mock(side_effect=ValueError('ARTIFICIAL_IDENTITY_FAILURE'))
    with pytest.raises(ValueError,match='ARTIFICIAL_IDENTITY'):run.execute()
    assert run.initialize.call_count==1
    stop=json.loads((tmp_path/m.FAMILY/'INTEGRATED_STOP.json').read_text())
    assert stop['failed'] and stop['mandatory_stop'] and stop['retry']==stop['resume']==0


def test_target_initialization_uses_one_paired_setup_per_start_and_exact_budget(tmp_path,monkeypatch):
    from trottertracks import c_integrated_execution_rpc as rpc
    run=fixture(tmp_path)
    run.c['environment']={'evaluation_python':sys.executable}
    run.c['maximum_calls']['stage_totals']['transfer_initialization']=26
    run.c['maximum_calls']['components']={'transfer_initialization':dict(
        scf=1,spatial_integrals=1,physical_context=1,neutral_scale=1,
        paired_start_identity=3,objective_setup=3,start_summary=1,neutral_restore=15)}
    class Physical:
        def __init__(self,*a):pass
        def request(self,op,**kw):
            assert op=='initialize'
            return {'common_pickle':'invented.pkl'}
    monkeypatch.setattr(rpc,'RpcProcess',Physical)
    seen=[]
    def fit(task,physical=None,affinity=None):
        op=task['operation'];seen.append(op)
        if op=='generate_input':
            p=tmp_path/task['out']/'input.npz';p.parent.mkdir(parents=True);p.write_bytes(b'INVENTED_NOT_AN_ARRAY')
            return dict(input_path=str(p.relative_to(tmp_path)),input_sha256=m.io.sha(p),scientific_units=2)
        if op=='restore':
            p=tmp_path/task['out']/'restored.npz';p.parent.mkdir(parents=True);p.write_bytes(b'INVENTED_RESTORE')
            (p.parent/'STOP.json').write_text(json.dumps({'actual_calls':{'mock_scientific_work':5}}))
            return dict(eligible=True,candidate_path=str(p.relative_to(tmp_path)),physical={'budget_PASS':True})
        if op=='tensor_scale':return {'objective_from_tensor':.125}
        if op=='start_identity':return {'status':'ARTIFICIAL_PASS'}
        if op=='specs':return {'PR':{'gamma':task['strength']},'ENTRY':{'strength':task['strength']}}
        raise AssertionError(op)
    run.fit=fit
    result=run.initialize('H4_115','target/run01',{'PR':.1,'ENTRY':.3})
    assert seen.count('specs')==3 and seen.count('restore')==3
    assert run.stages['transfer_initialization']==26
    for start in ('C0','P1','P2'):
        assert result['specs'][start,'transfer']=={'PR':{'gamma':.1},'ENTRY':{'strength':.3}}
