"""Invented current-shaped arrays/small circuits; real metadata is passive only."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from trottertracks import c_iid_grid_bridge as m
from trottertracks import c_iid_grid_accuracy as grid_accuracy
from trottertracks import c_pr_aware_df_stage_c2 as table

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def invented():
    from test_research_c_pr_aware_df_c2_bridge import artificial
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_c2_bridge as bridge
    x=artificial();a=x['arm'];tail,cfg,dist=b.configure(a,Fraction(1,10),32,0)
    operators=b.atom_operators(x['ctx'],a)
    tab=table.analyze_table([abs(atom.coefficient) for atom in a.plan.atoms],[z.probability for z in tail.components])
    means={key:bridge.event_mean(a,tail,dist,operators,dict(ideal_probabilities=prob))
        for key,prob in [('ideal',tab['ideal_probabilities']),('grid',tab['grid']['probabilities_exact'])]}
    kernels=b.kernels(a,x['G'],Fraction(1,10),cfg)  # invented input only
    saved=dict(generators=x['G'],corrected=kernels['corrected'])
    context=m.kernel_bridge(a,cfg,means,m.heads(x['G']),saved)
    powers=m.power_cache(context);psi=np.eye(36,dtype=complex)[0]
    from trottertracks.c1_stage3_accuracy import _legacy
    rows=[]
    for i in range(13):
        A=_legacy('rte').finite_rte_attenuation(cfg,tail_evolutions=1<<i)
        z=np.vdot(psi,A*powers['corrected'][i]@psi)
        rows.append(dict(m=i,reference=dict(attenuation=A,raw_signal_parts=[z.real,z.imag],
            pf_operator_error=.001,rte_corrected_operator_error=.002)))
    return dict(x=x,tail=tail,config=cfg,table=tab,means=means,saved=saved,context=context,
        powers=powers,psi=psi,rounds=rows)


def test_exact_grid_boundary_zero_and_underflow_distinction():
    v=table.choice_grid([0.,.5,.5]);D=1<<53
    assert v['mapping']=='searchsorted_right' and v['counts']==[0,D//2,D//2]
    assert v['zero_random_draws']
    tiny=table.analyze_table([1,Fraction(1,10**400)],[1.,0.])
    assert tiny['lost_positive_ideal_atoms']==[1] and Fraction(tiny['eta_exact'])>0
    zero=table.analyze_table([1,0],[1.,0.]);assert Fraction(zero['eta_exact'])==0


@pytest.mark.parametrize('probabilities',[[.5,.5],[.501,.499]])
def test_grid_signal_for_signed_phase_and_independent_occurrences(probabilities):
    t=table.analyze_table([1,1],probabilities);eta=Fraction(t['eta_exact'])
    for i in (0,12):
        r=m.previous.categorical_preview(t,i)
        assert r['events']==32*(1<<i)
        assert Fraction(r['conditional_raw_bound_exact'])==min(2,2*32*(1<<i)*eta)
        assert r['actual_implemented_law_error'] is None and not r['PCG64_IID_proven']


def test_current_routing_signed_atoms_scalar_and_both_laws(invented):
    v=invented;c=v['context']
    assert c['report']['h']=='1/10' and c['report']['r']==32 and c['report']['s']==1
    assert np.max(abs(c['ideal']-c['raw']))<1e-10
    assert v['config'].rte_steps==32
    for i in (0,3,12):
        row=m.round_bridge(c,v['powers'],i,v['rounds'][i],v['psi'],v['table'])
        assert row['kernel_occurrences']==1<<i and row['events']==32*(1<<i)
        assert row['saved_signal_replay_error']<2e-10
        assert row['error_evidence'] is None and row['reference_numerical_uncertainty'] is None
        assert row['shots'] is None and row['actual_CA_achieved'] is None
        assert row['PCG64_joint_stream_qualified'] is False


@pytest.mark.parametrize('fault',['lambda','identity','atoms','tau'])
def test_current_saved_law_binding_not_just_candidate_name(invented,fault):
    v=invented;law=dict(lambda_R=v['tail'].lambda_r,identity=v['x']['arm'].plan.extracted_identity,atoms=216)
    norm=dict(r=32,tau=v['config'].dimensionless_step_time)
    if fault=='lambda':law['lambda_R']*=2
    elif fault=='identity':law['identity']+=1
    elif fault=='atoms':law['atoms']=215
    elif fault=='tau':norm['tau']*=2
    with pytest.raises(ValueError,match='NORMALIZATION_BINDING'):m.prepare_law(v['x']['arm'],law,norm)


@pytest.mark.parametrize('bad',[dict(rte_steps=16,finite_taylor_order=0,evolution_time=.1),
    dict(rte_steps=32,finite_taylor_order=0,evolution_time=.025),dict(rte_steps=32,finite_taylor_order=1,evolution_time=.1)])
def test_old_r16_s4_or_wrong_K_routing_refused(invented,bad):
    v=invented
    with pytest.raises(ValueError,match='ROUTING'):m.kernel_bridge(v['x']['arm'],SimpleNamespace(**bad),v['means'],{},v['saved'])


def test_reused_kernel_is_not_fresh_IID_and_P0_cross_event_compilation():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    X=np.array([[0.,1.],[1.,0.]]);Z=np.diag([1.,-1.]);M=(X+Z)/2
    assert not np.allclose(M@M,(X@X+Z@Z)/2)
    q=QuantumCircuit(2);q.rz(.1,0);q.rz(.2,0);q.p(.17,1)
    # First two rotations represent different event fragments without a fence.
    lock=SharedQiskitCalls(lambda k,l,fn:fn())
    compiled=lock('compile','two invented events',lambda:compiled_wrapper(q,CompilerProfile()))
    assert np.max(abs(Operator(q).data-Operator(compiled).data))<1e-12
    assert compiled.count_ops().get('rz',0)<=2
    assert m.fixed()['Stage_B']=='P0_UNIFORM_NOT_ESTABLISHED'  # finite example is not coverage


def qualified(comp=None,coverage=None):
    from test_research_c_all_law_signal_ca import parts,premises,packet
    components={k:v for k,v in parts().items() if k!='stream_model'} if comp is None else comp
    return grid_accuracy.qualify(packet(),expected_identity={'candidate':'toy'},observed_identity={'candidate':'toy'},
        components=components,coverage=premises() if coverage is None else coverage)


def test_iid_lane_uses_existing_arithmetic_without_actual_law_label(tmp_path):
    from trottertracks import c_pr_aware_df_stage_c as c1
    from trottertracks.c1_stage3_integrated import Calls
    p,q=qualified();assert q['kind']=='IID_GRID_CONDITIONAL_NUMERICAL_BOUND'
    assert p['error_evidence'] is None and q['PCG64_joint_stream_error'] is None
    assert c1.qualification(p,'evidence')['kind']=='UNQUALIFIED'
    with pytest.raises(ValueError):c1.qualification(dict(p,error_evidence=q),'evidence')
    call=Calls(tmp_path,grid_accuracy.LIMITS,16)
    out=grid_accuracy.assess(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q,call)
    assert dict(call.counts)==grid_accuracy.LIMITS
    assert out['status']=='IID_GRID_CONDITIONAL_NUMERICAL_SUFFICIENT_CONDITION'
    r=out['rounds'][0]
    assert r['raw_signal_error']==pytest.approx(.8*(.002+.003)+4e-7)
    assert r['law_model']==m.MODEL['id'] and r['engine_reference_law']=='ideal_finite_RTE_law'
    assert set(r['raw_error_components'])==set(grid_accuracy.PARTS)
    assert out['actual_CA_achieved'] is None and out['rigorous_CA'] is None and not out['PCG64_joint_stream_qualified']
    assert out['energy']['numerical_pilot']


@pytest.mark.parametrize('missing',grid_accuracy.PARTS)
def test_unknown_bound_not_zero_or_accuracy_PASS(missing):
    from test_research_c_all_law_signal_ca import parts
    from trottertracks import c_pr_aware_df_stage_c as c1
    comp={k:v for k,v in parts().items() if k!='stream_model'};comp[missing]=None
    p,q=qualified(comp);assert q['kind']=='UNQUALIFIED_IID_GRID' and missing in q['missing']
    out=grid_accuracy.assess(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q)
    assert out['rounds']==[] and out['energy'] is None


def test_finite_family_missing_round_and_phase_failure():
    from test_research_c_all_law_signal_ca import parts,premises
    from trottertracks import c_pr_aware_df_stage_c as c1
    _,q=qualified(coverage=premises(final_whole_kernel_P0_covered=False))
    assert q['kind']=='UNQUALIFIED_IID_GRID'
    comp={k:v for k,v in parts().items() if k!='stream_model'};comp['categorical'].pop()
    with pytest.raises(ValueError):qualified(comp)
    comp={k:v for k,v in parts().items() if k!='stream_model'};comp['synthesis'][0]['bound']='1'
    p,q=qualified(comp);out=grid_accuracy.assess(c1.accuracy.AccuracyTarget(**c1.TARGET),p,q)
    assert len(out['rounds'])==1 and out['rounds'][0]['shots'] is None and out['energy'] is None


def test_wrong_random_model_not_treated_as_actual_or_IID():
    from test_research_c_all_law_signal_ca import parts,premises,packet
    with pytest.raises(ValueError,match='MODEL'):
        grid_accuracy.qualify(packet(),expected_identity={},observed_identity={},
            components={k:v for k,v in parts().items() if k!='stream_model'},coverage=premises(),model=dict(m.MODEL,bits=52))


def synthetic_run(root,invented):
    from trottertracks.c_pr_aware_df_stage_b_runner import save_arrays
    from trottertracks import c_pr_aware_df_stage_c as c1
    x=invented['x'];c=m.fixed();c.update(candidate_sha256='toy',input_sha256='toy-input',
        compiler={'profile':'toy'},target=c1.TARGET,source_manifest='SOURCE.json')
    c['input_roles']={k:k+'.npz' for k in ('input','candidate','factors','PF_RTE_reference','own_ground')}
    save_arrays(root/c['input_roles']['input'],**x['packet'])
    save_arrays(root/c['input_roles']['candidate'],**x['data'])
    save_arrays(root/c['input_roles']['factors'],**x['data'],generators=x['G'])
    save_arrays(root/c['input_roles']['PF_RTE_reference'],corrected=invented['saved']['corrected'])
    save_arrays(root/c['input_roles']['own_ground'],sector_state=invented['psi'])
    pr=dict(candidate='toy',PR=dict(r=32,rounds=invented['rounds'],all_law_synthesis=None,actual_CA_achieved=None))
    metadata={k:{} for k in m.previous.records()}
    metadata.update(PR_result=dict(arms={'PR':pr}),PR_contract=dict(candidate_sha256={'PR':'toy'},compiler=c['compiler'],target=c['target'],
        policy=dict(s=1,r=32,K=0,h='1/10',ld=2)),
        PR_law=dict(lambda_R=invented['tail'].lambda_r,identity=x['arm'].plan.extracted_identity,atoms=216),
        PR_normalization=dict(r=32,tau=invented['config'].dimensionless_step_time))
    for key,n in [('PR_STOP',1231),('statistics_STOP',7)]:metadata[key]=dict(scientific_units=n,failed=False,retry=0,resume=0,mandatory_stop=True)
    for key,name in m.previous.records().items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);m.io.save(p,metadata[key])
    m.io.save(root/'SOURCE.json',dict(files={},synthetic=True))
    return c


@pytest.mark.parametrize('fault',['none','candidate','saved_signal'])
def test_artificial_38_call_run_partial_STOP_and_preservation(tmp_path,monkeypatch,invented,fault):
    c=synthetic_run(tmp_path,invented)
    if fault=='candidate':
        path=tmp_path/m.previous.records()['PR_result'];d=m.io.read(path);d['arms']['PR']['candidate']='wrong';path.write_text(json.dumps(d))
    if fault=='saved_signal':
        path=tmp_path/m.previous.records()['PR_result'];d=m.io.read(path);d['arms']['PR']['PR']['rounds'][0]['reference']['raw_signal_parts']=[0.,0.];path.write_text(json.dumps(d))
    before=(tmp_path/c['input_roles']['candidate']).read_bytes()
    monkeypatch.setattr(m,'preflight',lambda *a:(c,{}))
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'toy-code-source\n')
    if fault=='none':
        result=m.run(tmp_path,'unused',execute_science=True)
        assert result['status']=='IID_GRID_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE'
        assert result['Stage_C']=='NOT_READY_BOUND_INPUTS' and result['error_evidence'] is None
    else:
        with pytest.raises((ValueError,RuntimeError)):m.run(tmp_path,'unused',execute_science=True)
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert stop['failed']==(fault!='none') and (tmp_path/c['input_roles']['candidate']).read_bytes()==before
    if fault=='none':assert stop['call_counts']==m.LIMITS and stop['scientific_units']==38
    else:assert stop['first_failure'] is not None and not (out/'result.json').exists()
    with pytest.raises(FileExistsError):m.run(tmp_path,'unused',execute_science=True)


def test_permission_precedes_inputs_and_old_budget_not_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'preflight',lambda *a:pytest.fail('before permission'))
    with pytest.raises(PermissionError):m.run(tmp_path,'absent')
    assert not list(tmp_path.iterdir()) and m.MAX_CALLS==38


@pytest.mark.parametrize('fault',['none','model','source','input','fresh','retry','resume'])
def test_passive_current_contract_drift(monkeypatch,fault):
    read,sha=m.io.read,m.io.sha;cp=ROOT/m.CONTRACT
    def changed(path):
        d=deepcopy(read(path))
        if Path(path)==cp:
            if fault=='model':d['model']['independent_coherent_occurrences']=False
            if fault in ('retry','resume'):d[fault]=1
        return d
    monkeypatch.setattr(m.io,'read',changed)
    bad=ROOT/('src/trottertracks/c_iid_grid_bridge.py' if fault=='source' else m.previous.records()['PR_result'])
    if fault in ('source','input'):monkeypatch.setattr(m.io,'sha',lambda p:'wrong' if Path(p)==bad else sha(p))
    if fault=='fresh':
        exists=Path.exists;monkeypatch.setattr(Path,'exists',lambda p:True if p==ROOT/m.FRESH else exists(p))
    if fault=='none':assert m.preflight(ROOT,cp)[1]['scientific_calls']==0
    else:
        with pytest.raises(ValueError):m.preflight(ROOT,cp)
