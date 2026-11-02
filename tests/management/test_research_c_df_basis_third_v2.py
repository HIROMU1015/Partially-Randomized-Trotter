"""Invented J/U and real B2 law gate: no saved molecular arrays are read."""
from pathlib import Path
from copy import deepcopy
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_third as old,c_df_basis_third_v2 as v2


def artificial(root,monkeypatch):
    from test_research_c_df_structural_pr import fixture
    c,saved,p,d=fixture(root)
    saved['diagnostic']=saved['P4']['final_diagnostic'];saved['sector']=saved['P4_sector']
    saved['M4_native']=saved['native'];saved['P4']=dict(native=saved['native'])
    monkeypatch.setattr(old,'EXPORT','JU.npz');monkeypatch.setattr(old,'EXPORT_SHA','ARTIFICIAL_GAMMA0')
    monkeypatch.setattr(old,'evidence',lambda *a:saved)
    return c,saved,p,d


def test_real_reference_to_unmocked_B2_components_gate(tmp_path,monkeypatch):
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b2 as b2
    c,_,p,_=artificial(tmp_path,monkeypatch)
    out=tmp_path/'reference';out.mkdir();call=old.integrated.Calls(out,old.LIMITS,old.MAX_CALLS)
    with v2.bindings():a=old.stage_reference(tmp_path,c,out,call)
    assert dict(call.counts)==old.REFERENCE_LIMITS
    report=a['tail_info'];tail=a['tail'];arm=a['arm']
    assert old.io.read(out/'tail.json')['components']==report['components']
    assert old.io.read(out/'result.json')['tail']['components']==report['components']
    assert [x['probability'] for x in report['components']]==[x.probability for x in tail.components]
    assert len(report['components'])==len(tail.components)
    car=b.stage1.car_bilinears(8)
    G=[b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)]
    G.extend(b.stage1.raw_frame(J,U,car)[0] for J,U in zip(arm.J,arm.U,strict=True))
    ctx=b2.reference_context(arm,G,car,__import__('fractions').Fraction(1,10))
    checked=b2.law_gate(arm,tail,ctx,report) # deliberately NOT mocked
    assert checked['max_entry']<checked['tolerance']
    legacy=dict(report);legacy.pop('components')
    with pytest.raises(KeyError,match='components'):b2.law_gate(arm,tail,ctx,legacy)
    broken=deepcopy(report);broken['components'][0]['probability']+=.01
    with pytest.raises(ValueError,match='LAW_OR_IDENTITY'):b2.law_gate(arm,tail,ctx,broken)
    with pytest.raises(ValueError,match='METADATA_CONFLICT'):v2.with_components(broken,tail)


def test_binding_restores_all_frozen_engine_functions():
    before=(old.VERSION,old.SCOPE,old.FRESH,old.stage_reference)
    with pytest.raises(RuntimeError):
        with v2.bindings():
            assert old.FRESH.endswith('run02') and old.stage_reference is v2.repaired_reference
            assert old.stage_accuracy is old.p4.stage_accuracy
            raise RuntimeError('ARTIFICIAL')
    assert (old.VERSION,old.SCOPE,old.FRESH,old.stage_reference)==before


def test_revision_rejects_scientific_or_saved_input_changes(tmp_path):
    c={k:{} for k in v2.SCIENCE_KEYS};c.update(saved_inputs={'data':'fixed'},implementation='old')
    old.io.save(tmp_path/'old.json',c)
    c=dict(c,implementation=v2.VERSION,scope=v2.SCOPE,technical_relaunch=1,
           previous_contract='old.json',previous_contract_sha256=old.io.sha(tmp_path/'old.json'))
    old.io.save(tmp_path/'new.json',c)
    assert v2.check_revision(tmp_path,tmp_path/'new.json')==c
    c['target']={'different_CA':1};old.io.save(tmp_path/'bad.json',c)
    with pytest.raises(ValueError,match='CHANGED_SCIENCE:target'):v2.check_revision(tmp_path,tmp_path/'bad.json')
    c['target']={};c['saved_inputs']={'data':'changed'};old.io.save(tmp_path/'bad_input.json',c)
    with pytest.raises(ValueError,match='CHANGED_INPUTS'):v2.check_revision(tmp_path,tmp_path/'bad_input.json')


def test_repaired493_dispatch_uses_real_full_law_gate(tmp_path,monkeypatch):
    import numpy as np
    from qiskit import QuantumCircuit,qpy
    from test_research_c_df_basis_third import rows
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b2 as b2
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    from trottertracks import c_df_basis_transfer as t
    c,saved,_,_=artificial(tmp_path,monkeypatch)
    def train(root,c,out,call):
        data=call('training_rows_load','invented two cohorts',lambda:dict(M4_Proposed=rows(),tail_P4=rows(32)))
        call('training_identity','artificial',lambda:None)
        return {metric:call('transfer_training',metric,lambda metric=metric:t.freeze_training(data,metric)) for metric in old.old.METRICS}
    monkeypatch.setattr(old,'training',train)
    c['shells']={}
    for label in ('head','X','Y'):
        qc=b2.shell_build(8,label);c['shells'][label]={}
        for kind in ('raw','compiled'):
            name=label+kind+'.qpy';c['shells'][label][kind]=name
            with (tmp_path/name).open('xb') as f:qpy.dump(qc,f)
        saved['shell_reports'][label]['metrics']=circuit_metrics(qc)
    monkeypatch.setattr(b,'build_kernel',lambda *a,**kw:QuantumCircuit(9))
    # Keep reference_context and law_gate real, including independent full CAR.
    monkeypatch.setattr(b2,'trajectory_reference',lambda *a:np.eye(2,dtype=complex))
    monkeypatch.setattr(b2,'raw_gate',lambda *a:(None,None,dict(status='MOCK')))
    monkeypatch.setattr(b2,'compiled_gate',lambda *a:(None,dict(status='MOCK')))
    monkeypatch.setattr(b2,'adjoint_pair_gate',lambda *a:({'raw':QuantumCircuit(9),'compiled':QuantumCircuit(9)},dict(status='MOCK')))
    monkeypatch.setattr(b2,'XY_gate',lambda *a:dict(status='MOCK'))
    out=tmp_path/'all';out.mkdir();call=old.integrated.Calls(out,old.LIMITS,old.MAX_CALLS)
    with v2.bindings():result=old.execute(tmp_path,c,out,call)
    assert dict(call.counts)==old.LIMITS and sum(call.counts.values())==493
    assert not call.stop.is_set() and (out/'native/law_gate.json').exists()
    assert len(list((out/'native/circuits').glob('*.qpy')))==128
    assert len(result['native']['descriptor_rows'])==32
    assert result['native']['all_law_synthesis'] is result['native']['cost_support'] is None
