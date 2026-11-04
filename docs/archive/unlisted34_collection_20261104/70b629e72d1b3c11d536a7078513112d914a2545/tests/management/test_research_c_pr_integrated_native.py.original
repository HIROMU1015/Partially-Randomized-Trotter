"""Mock dispatch only: fixed32/r128, phase/call/QPY/STOP plumbing, no H4 data."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_integrated_performance as m


@pytest.mark.parametrize('fail_compile',[False,True])
@pytest.mark.parametrize('finite',[False,True])
def test_fixed32_native_dispatch_and_failure(tmp_path,monkeypatch,fail_compile,finite):
    from qiskit import QuantumCircuit,qpy
    from trottertracks import c1_h4_strong_baseline as strong,c1_h4_native_fair as fair,c_pr_aware_df_m2 as m2
    from trottertracks.pr_dictionary_validation import f2_execution
    for label in ('head','X','Y'):
        with (tmp_path/(label+'.qpy')).open('wb') as f:qpy.dump(QuantumCircuit(9),f)
    c=dict(shells={x:{k:x+'.qpy' for k in ('raw','compiled')} for x in ('head','X','Y')},source_manifest_sha256='SYNTHETIC')
    class Builder:
        def __init__(self,**kw):self.call=kw['call']
        def transition(self,*a):self.call('relative_gaussian','synthetic',lambda:None)
    monkeypatch.setattr(strong,'StrongBuilder',Builder);monkeypatch.setattr(fair,'basis_request',lambda *a:None)
    def compile(q,p):
        if fail_compile:raise ValueError('SYNTHETIC_COMPILE_FAIL')
        return q.copy()
    monkeypatch.setattr(fair,'compile_checked',compile)
    monkeypatch.setattr(m.b.stage1,'car_bilinears',lambda n:{})
    monkeypatch.setattr(m.b.stage1,'raw_base',lambda *a:np.eye(2))
    monkeypatch.setattr(m.b.stage1,'raw_frame',lambda *a:(np.eye(2),None))
    monkeypatch.setattr(m2,'sector_match',lambda *a:(np.arange(36),{}))
    monkeypatch.setattr(m.b2,'reference_context',lambda *a:{})
    monkeypatch.setattr(m.b2,'law_gate',lambda *a:{'PASS':True})
    monkeypatch.setattr(m.b2,'shell_assessment',lambda *a:(None,{'PASS':True}))
    monkeypatch.setattr(m.b,'build_kernel',lambda *a,**k:QuantumCircuit(9))
    monkeypatch.setattr(m.b2,'trajectory_reference',lambda *a:np.eye(2))
    monkeypatch.setattr(m.b2,'raw_gate',lambda *a:(np.eye(2),np.eye(2),{'PASS':True}))
    monkeypatch.setattr(m.b2,'compiled_gate',lambda *a:(np.eye(2),{'PASS':True}))
    monkeypatch.setattr(m.b2,'adjoint_pair_gate',lambda raw,*a:({'raw':raw.copy(),'compiled':raw.copy()},{'PASS':True}))
    monkeypatch.setattr(m.b2,'XY_gate',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(f2_execution,'circuit_metrics',lambda *a:{k:1 for k in m.m4.METRICS})
    seen=[]
    def occurrence(*a,**kw):
        seen.append(kw['shot']);return tuple(SimpleNamespace(taylor_order=0,to_dict=lambda:{'synthetic':True}) for _ in range(128))
    monkeypatch.setattr(m.b,'independent_occurrence',occurrence)
    arm=SimpleNamespace(J=np.zeros((8,4,4)),U=np.broadcast_to(np.eye(4),(8,4,4)),
        model=SimpleNamespace(blocks=[SimpleNamespace(block_id=str(i)) for i in range(9)],source_identity='SYNTHETIC'),
        plan=SimpleNamespace(extracted_identity=0.,atoms=[]))
    a=dict(arm=arm,p=dict(h=np.eye(4),g=np.zeros((4,)*4),constant=0.),sector={},ground=np.ones(36)/6,
        tail=SimpleNamespace(lambda_r=1.,components=[]),dist=None,config=None)
    out=tmp_path/'out';out.mkdir();call=m.Calls(out,dict(m.LIMITS,fixed_word_events=4,finite_summary=1),m.MAX_CALLS)
    options={}
    if finite:
        a['config']=SimpleNamespace(rte_steps=32)
        event=SimpleNamespace(taylor_order=0,to_dict=lambda:{'synthetic':True})
        options=dict(sampling=dict(samples=4,events_per_sample=32),fixed_words=[(event,)*32 for _ in range(4)])
    if fail_compile:
        with pytest.raises(Exception):m.native(tmp_path,c,out,call,a,**options)
        assert call.stop.is_set() and not (out/'result.json').exists()
    else:
        rows,cal=m.native(tmp_path,c,out,call,a,**options)
        n=4 if finite else 32
        assert len(rows)==n and sorted(seen)==([] if finite else list(range(32)))
        assert len(list((out/'circuits').glob('*.qpy')))==4*n
        assert call.counts['compile']==call.counts['fixed_word_events' if finite else 'trajectory_sample']==n
        assert cal['cost_support'] is None
        if finite:
            import json
            value=json.loads((out/'samples/word00_events.json').read_text())
            assert value['sampling'] is None and not value['coherent_occurrences_independent']
            assert cal['status']=='FIXED_FINITE_WORD_CORRECTNESS_COMPLETE' and cal['population_mean'] is None
