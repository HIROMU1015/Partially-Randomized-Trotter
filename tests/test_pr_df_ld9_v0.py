"""Invented schedule and failure records only; no saved molecular input."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_pr_df_ld9_v0 as L

def invented(R=1):
    rows=[]
    for m in range(13):
        T=.1*(1<<m);num,den=T.as_integer_ratio()
        rows.append(dict(m=m,T=dict(numerator=num,denominator=den),R_outer=R,
            physical_candidate=dict(prefix=9,method='PR',formula='2nd',K=2,delta=.1,q=1<<m,T=T,R_outer=R,id='ARTIFICIAL_'+str(m)),
            structure=dict(formula_id='S2',signed_weights=[1.],allocation=[R],event_count=(1<<m)*R)))
    return dict(rounds=rows)

def test_mapper_gate_accepts_actual_R_without_LD7_copy():
    for R in (1,3):assert L.inspect(invented(R))['status']=='PASS'
    bad=invented();bad['rounds'][4]['physical_candidate']['R_outer']=3
    assert L.inspect(bad)['status']=='STOP_IDENTITY_OR_MAPPER'

def test_signal_entry_stops_at_first_b1_and_persists(tmp_path,monkeypatch):
    config=invented();gate=L.inspect(config);calls=[];scalar=dict(hamiltonian_constant=dict(value=.17),tail_identity=dict(value=.003),total_hex=(.17+.003).hex())
    monkeypatch.setattr(L,'OUT',tmp_path);monkeypatch.setattr(L,'fixed',lambda *a:None);monkeypatch.setattr(L,'delta',lambda *a:None)
    monkeypatch.setattr(L,'local',lambda n:dict(source='ARTIFICIAL',configuration=dict(commit='ARTIFICIAL',path='config')))
    def read(commit,path):return config if path=='config'else gate if path.endswith('SCHEDULE_GATE.json')else scalar
    monkeypatch.setattr(L,'read',read)
    def evaluate(ctx,job,ref,**kw):
        calls.append(job['m']);return dict(status='NUMERICAL_FAILURE',m=job['m'],candidate=config['rounds'][job['m']]['physical_candidate'],numerical=dict(normalizer=1.,gate_pass=False),error=dict(message='ARTIFICIAL finite-normalizer precision unresolved'))
    provider=SimpleNamespace(load_h4_context=lambda *a:SimpleNamespace(constant=.17,tails={9:dict(scalar=.003)}),numerical_recipe=lambda *a:{},saved_h4_reference=lambda *a:dict(fixture='REFERENCE'),evaluate=evaluate)
    metadata=SimpleNamespace(plan=lambda *a,**kw:dict(jobs=[dict(m=m)for m in range(13)]))
    monkeypatch.setattr(L.S,'signal_modules',lambda:(None,metadata,provider,SimpleNamespace(validate=lambda *a:None)))
    assert L.signals()==2 and calls==[0]
    failure=json.loads((tmp_path/'signals/FIRST_GATE_FAILURE.json').read_text());result=json.loads((tmp_path/'signals/RESULT.json').read_text())
    assert failure['normalizer']==1. and failure['normalizer_hex']==float(1).hex()
    assert failure['Q3_status']=='NOT_STARTED_SIGNAL_GATE_FAILURE'
    assert result['not_started_rounds']==list(range(1,13))and result['valid_signal_rounds']==0
