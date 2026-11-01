"""Invented arrays/circuits and mocked finite cohorts only; no stored H4 decode."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_rc2_pr_native as m


def write(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False)+"\n")


def invented(root):
    from test_research_c_rc2_pr_conditional_shots import invented as old
    c=old(root);B=c['records'];identities=c['identities']
    c['records']={'B_'+k:v for k,v in B.items()}
    shots_result=dict(status='STAGE_C_SHOTS_COMPLETE',arms={r:{lane:dict(identity=identities[r],
        shots_weight=dict(status='EVIDENCE_UNQUALIFIED' if lane=='evidence' else 'CONDITIONAL_SHOTS_AVAILABLE'))
        for lane in m.shots.LANES} for r in m.ROLES})
    for key,value in dict(result=shots_result,STOP=dict(failed=False,scientific_units=105,
            call_counts=m.shots.LIMITS,retry=0,resume=0,mandatory_stop=True),
            SOURCE=dict(execution_commit=m.SHOTS_SOURCE),contract=dict(identities=identities)).items():
        name='shots_'+key+'.json';write(root/name,value);c['records']['shots_'+key]=name
    rng=np.random.default_rng(441);a=rng.normal(size=(8,4,4))*.002;J=a+a.transpose(0,2,1)
    U=np.broadcast_to(np.eye(4),(8,4,4)).copy()
    c.update(scope=m.SCOPE,implementation=m.VERSION,base_commit=m.BASE,sampling=m.SAMPLING,
        allowances=m.ALLOWANCES,compiler=m.b2.profile_dict(),policy=m.stage_b.POLICY,
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,
        input='input.npz',exports={},sector_references={},grounds={},tails={},laws={})
    np.savez(root/'input.npz',h=np.diag([-.2,-.1,.1,.2]),g=np.zeros((4,)*4),constant=np.array(.31))
    for role in m.ROLES:
        np.savez(root/(role+'.npz'),J=J,U=U,x=np.zeros(128),jac=np.zeros(128))
        identities[role].update(sha256=m.io.sha(root/(role+'.npz')),
            physical_input=m.io.sha(root/'input.npz'))
        c['exports'][role]=role+'.npz'
        for group,suffix,values in [('sector_references','reference',dict(generators=np.zeros((9,36,36)),H=np.zeros((36,36)))),
                ('grounds','ground',dict(energy=np.array(-.1),sector_state=np.ones(36)/6))]:
            name=role+'_'+suffix+'.npz';np.savez(root/name,**values);c[group][role]=name
        for group in ('tails','laws'):
            name=role+'_'+group+'.json';write(root/name,dict(status='PASS'));c[group][role]=name
    # JSON provenance must carry the same invented identities after replacing opaque NPZ.
    for key in ('B_result','B_contract','shots_result','shots_contract'):
        p=root/c['records'][key];d=m.io.read(p)
        if key=='B_result':
            for role in m.ROLES:d['arms'][role]['identity']=identities[role]
        elif key=='shots_result':
            for role in m.ROLES:
                for lane in m.shots.LANES:d['arms'][role][lane]['identity']=identities[role]
        else:d['identities']=identities
        write(p,d)
    names={c['input'],*c['exports'].values(),*c['sector_references'].values(),*c['grounds'].values(),
        *c['tails'].values(),*c['laws'].values(),*c['records'].values()}
    c['saved_inputs']={p:m.io.sha(root/p) for p in names}
    write(root/'contract.json',c);return c


def test_passive_import_and_655_scope():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_rc2_pr_native as m;assert m.MAX_CALLS==655;assert not any(n in sys.modules for n in ('numpy','scipy','qiskit','ffsim','jax'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert m.LIMITS['compile']==67 and m.LIMITS['trajectory_sample']==64
    assert not set(m.LIMITS)&{'fit','optimizer','PF','RTE','shots','total_cost','ground_eigensolve'}


def test_passive_preflight_only_hash_headers_metadata(tmp_path,monkeypatch):
    c=invented(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('real decode in passive preflight'))
    monkeypatch.setattr(m,'execute',lambda *a:pytest.fail('passive execute'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c


@pytest.mark.parametrize('fault',['source','hash','identity','shots_source','shots_status','B_FAIL','profile','sampling','fresh'])
def test_fixed_identity_mismatches_STOP(tmp_path,fault):
    c=invented(tmp_path)
    if fault=='source':c['source_manifest_sha256']='0'*64
    if fault=='hash':c['saved_inputs'][c['exports']['pr']]='0'*64
    if fault=='identity':c['identities']['pr']['role']='proposed'
    if fault=='profile':c['compiler']=dict(c['compiler'],seed_transpiler=18)
    if fault=='sampling':c['sampling']=dict(c['sampling'],samples_per_arm=64)
    if fault=='fresh':(tmp_path/m.FRESH).mkdir(parents=True)
    if fault in ('shots_source','shots_status','B_FAIL'):
        key={'shots_source':'shots_SOURCE','shots_status':'shots_result','B_FAIL':'B_STOP'}[fault]
        p=tmp_path/c['records'][key];d=m.io.read(p)
        if fault=='shots_source':d['execution_commit']='old_source'
        if fault=='shots_status':d['arms']['pr']['evidence']['shots_weight']['status']='CA_PASS'
        if fault=='B_FAIL':d['failed']=True
        write(p,d);c['saved_inputs'][str(p.relative_to(tmp_path))]=m.io.sha(p)
    write(tmp_path/'contract.json',c)
    with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')


def arm2(role):
    from test_research_c_pr_aware_df_stage_b2 import artificial
    arm,p=artificial(n=2,rank=8,role='baseline' if role=='rc2' else 'proposed')
    return replace(arm,role=role),p


def test_new_role_cohort_independent_seed_and_128_occurrences():
    seeds=set()
    for role in m.ROLES:
        arm,_=arm2(role);tail,c,dist=m.b.configure(arm,Fraction(1,10),128,0)
        for source in ('inventedA','inventedB'):
            for i in range(32):
                events,record=m.sample(arm,tail,dist,c,i,source)
                assert len(events)==128 and record['seed'] not in seeds;seeds.add(record['seed'])
                assert not record['paired'] and not record['reused_as_coherent_repetitions']
            assert m.sample(arm,tail,dist,c,0,source)==m.sample(arm,tail,dist,c,0,source)
        with pytest.raises(ValueError):m.sample(arm,tail,dist,c,32,'inventedA')
        _,wrong,d64=m.b.configure(arm,Fraction(1,10),64,0)
        with pytest.raises(ValueError):m.sample(arm,tail,d64,wrong,0,'inventedA')
    assert len(seeds)==128


@pytest.mark.parametrize('role',m.ROLES)
def test_live_invented_r128_reference_P0_phase_adjoint_XY_and_law(tmp_path,role):
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from threadpoolctl import threadpool_limits
    arm,p=arm2(role);h=Fraction(1,10)
    with threadpool_limits(limits=1):
        car=m.b.stage1.car_bilinears(4)
        G=[m.b.stage1.raw_base(m.b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)]
        G.extend(m.b.stage1.raw_frame(J,U,car)[0] for J,U in zip(arm.J,arm.U))
        ctx=m.b2.reference_context(arm,G,car,h);tail,c,dist=m.b.configure(arm,h,128,0)
        catalog=[dict(id=x.component_id,probability=x.probability,sign=x.coefficient_sign) for x in tail.components]
        previous=dict(status='PASS',lambda_R=tail.lambda_r,identity=arm.plan.extracted_identity,atoms=len(arm.plan.atoms))
        write(tmp_path/'tail.json',previous);write(tmp_path/'law.json',dict(status='PASS',orders=[0],event_probabilities=catalog))
        config=dict(tails={role:'tail.json'},laws={role:'law.json'})
        assert m.law_gate(tmp_path,config,role,arm,tail,ctx)['status']=='PASS'
        changed=deepcopy(catalog);changed[0]['sign']*=-1
        write(tmp_path/'law.json',dict(status='PASS',orders=[0],event_probabilities=changed))
        with pytest.raises(ValueError,match='SAVED_COMPONENT'):m.law_gate(tmp_path,config,role,arm,tail,ctx)
        events,_=m.sample(arm,tail,dist,c,0,'invented_native')
        ref=m.b2.trajectory_reference(arm,ctx,h,events)
        np.testing.assert_allclose(ref,m.b.trajectory_reference(arm,G,ctx['atoms'],h,events,c),atol=1e-12,rtol=0)
        raw=m.b.build_kernel(arm,h,events,c);compiled=compile_checked(raw,CompilerProfile())
        R,target,_=m.b2.raw_gate(raw,ref);C,_=m.b2.compiled_gate(compiled,R,target)
        m.b2.adjoint_pair_gate(raw,compiled,R,C,target)
        shells={}
        for label in ('head','X','Y'):
            qc=m.b2.shell_build(4,label);shells[label],_=m.b2.shell_assessment(qc,compile_checked(qc,CompilerProfile()),label)
        psi=np.ones(16)/4
        assert m.b2.XY_gate(R,C,ref,psi,shells)['status']=='PASS'
        bad=raw.copy();bad.p(.03,4)
        with pytest.raises(ValueError):m.b2.raw_gate(bad,ref)
        bad=compiled.copy();bad.rz(.03,0)
        with pytest.raises(ValueError):m.b2.compiled_gate(bad,R,target)


def test_saved_sector_mapping_is_independent_of_PF_payload(tmp_path,monkeypatch):
    from trottertracks.c1_molecule_extension import SectorCAR
    ctx=SectorCAR(4)
    G=np.array([np.diag(np.arange(36)*.001)]*9);H=sum(G)
    np.savez(tmp_path/'ref.npz',generators=G,H=H,pf=np.array('FORBIDDEN'),corrected=np.array('FORBIDDEN'))
    values=m.sector_load(tmp_path/'ref.npz')
    ix=m.b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices]
    np.testing.assert_equal(values['full_input_indices'],ix)
    full=np.zeros((9,256,256),complex)
    for i in range(9):full[i][np.ix_(ix,ix)]=G[i]
    _,report=m.m2.sector_match(full,values)
    assert len(report)==10 and set(values)=={'generators','H','full_input_indices'}


@pytest.mark.parametrize('fault',[False,True])
def test_mock64_end_to_end_counts_QPY_unknowns_FAIL_STOP_and_preservation(tmp_path,monkeypatch,fault):
    from qiskit import QuantumCircuit
    from trottertracks import c1_h4_native_fair as fair
    from trottertracks import c1_h4_strong_baseline as strong
    from trottertracks.pr_dictionary_validation import f2_execution
    from trottertracks.c1_stage3_accuracy import METRICS
    c=invented(tmp_path);protected=dict(c['saved_inputs'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'INVENTED_NATIVE_SOURCE\n')
    monkeypatch.setattr(m.b.stage1,'car_bilinears',lambda *a:None)
    monkeypatch.setattr(m.b.stage1,'raw_base',lambda *a:np.eye(2))
    monkeypatch.setattr(m.b.stage1,'raw_frame',lambda *a:(np.eye(2),None))
    monkeypatch.setattr(m.m2,'sector_match',lambda *a:(np.arange(36),dict(status='PASS')))
    monkeypatch.setattr(m.b2,'reference_context',lambda *a:None)
    monkeypatch.setattr(m,'law_gate',lambda *a:dict(status='PASS'))
    class Builder:
        def __init__(self,relative_fusion,call):self.call=call;self.basis_reports=[];self.fusion_records=[]
        def transition(self,*a):self.call('relative_gaussian','invented transition',lambda:None)
    monkeypatch.setattr(strong,'StrongBuilder',Builder)
    monkeypatch.setattr(fair,'basis_request',lambda *a:None)
    monkeypatch.setattr(fair,'compile_checked',lambda qc,p:qc.copy())
    monkeypatch.setattr(m.b,'build_kernel',lambda *a,**kw:QuantumCircuit(9))
    monkeypatch.setattr(m.b2,'trajectory_reference',lambda *a:np.eye(2))
    def raw_gate(*a):
        if fault:raise ValueError('INVENTED_RELATIVE_PHASE_FAIL')
        return np.eye(2),np.eye(2),dict(status='PASS')
    monkeypatch.setattr(m.b2,'raw_gate',raw_gate)
    monkeypatch.setattr(m.b2,'compiled_gate',lambda *a:(np.eye(2),dict(status='PASS')))
    monkeypatch.setattr(m.b2,'adjoint_pair_gate',lambda raw,compiled,*a:(dict(raw=raw.inverse(),compiled=compiled.inverse()),dict(status='PASS')))
    monkeypatch.setattr(m.b2,'XY_gate',lambda *a:dict(status='PASS'))
    monkeypatch.setattr(f2_execution,'circuit_metrics',lambda qc:{k:7 for k in METRICS})
    if fault:
        with pytest.raises(ValueError,match='INVENTED_RELATIVE_PHASE_FAIL'):m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
    else:
        result=m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
        assert result['status']=='RC2_PR_FINITE_COHORT_NATIVE_COMPLETE'
        assert all(result[k] is None for k in ('whole_law_synthesis_bound','implemented_law_error','cost_support',
            'ideal_law_mean','actual_CA_achieved','shots','conditional_interrogation_totals','guaranteed_advantage'))
        assert all(result['PR_minus_RC2'][k]==0 for k in METRICS)
        assert len(list((tmp_path/m.FRESH/'circuits').glob('*.qpy')))==262
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed']==fault and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    if not fault:assert stop['scientific_units']==655 and stop['call_counts']==m.LIMITS
    else:
        assert stop['scientific_units']<=655 and (tmp_path/m.FRESH/'failure.json').exists()
        assert stop['call_counts'].get('comparison',0)==0
    assert all(m.io.sha(tmp_path/p)==h for p,h in protected.items())
    with pytest.raises(ValueError,match='FRESH_NATIVE'):m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
