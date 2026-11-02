"""Invented DF/operator data only. Never loads Stage A molecular arrays."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks import c_pr_aware_df_stage_b_runner as m


def artificial(n=2,rank=3,scale=1.):
    rng=np.random.default_rng(720+n);a=rng.normal(size=(rank,n,n))*.008
    J=scale*(a+a.transpose(0,2,1));U=np.array([np.linalg.qr(rng.normal(size=(n,n)))[0] for _ in J])
    h=np.diag(np.linspace(-.2,.2,n));g=np.zeros((n,)*4)
    arm=b.prepare_arm(h,g,.31,J,U,role='baseline',source_identity='invented:baseline',input_identity='invented input',ld=1 if n==2 else 2)
    return arm,h,g,J,U


def sector_generators(arm,h,g):
    ctx=m.molecule.SectorCAR(len(h))
    return ctx,ctx.generators(h,g,arm.model.constant,arm.J,arm.U)


def full_generators(arm,h,g):
    car=b.stage1.car_bilinears(arm.model.modes)
    G=[b.stage1.raw_base(b.stage1.corrected_kappa(h,g),arm.model.constant,car)]
    G.extend(b.stage1.raw_frame(J,U,car)[0] for J,U in zip(arm.J,arm.U))
    ctx=SimpleNamespace(d=len(G[0]),bilinears={(p,q):csr_matrix(car[p,q]) for p in range(arm.model.modes)
        for q in range(arm.model.modes) if p%2==q%2})
    return ctx,G


def test_passive_import():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_stage_b_runner as m;assert m.MAX_CALLS==141;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim'])"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


@pytest.mark.parametrize('n,rank',[(2,3),(4,8)])
def test_general_J_car_tail_lambda_spin_scalar_and_original_order(n,rank):
    arm,h,g,_,_=artificial(n,rank);ctx,G=sector_generators(arm,h,g)
    for i,H in enumerate(G):np.testing.assert_allclose(b.candidate_factor(ctx,arm,i),H,atol=1e-12)
    tail,config,dist=b.configure(arm,Fraction(1,40),16,0)
    report=b.tail_gate(ctx,arm,G,tail)
    assert report['status']=='PASS' and report['probability_sum']==pytest.approx(1.)
    assert report['lambda_R']==pytest.approx(np.sum(abs(b.objective.coefficients(arm.J[arm.plan.ld:],np))))
    assert arm.model.blocks[1].block_id=='F0' and arm.model.blocks[-1].block_id=='F'+str(rank-1)
    assert all(a.kind!='IDENTITY' for a in arm.plan.atoms)
    assert any(a.coefficient<0 for a in arm.plan.atoms) and arm.plan.extracted_identity!=0
    assert dist.orders==(0,)


def test_small_nonzero_and_zero_tail_are_not_thresholded_or_randomized():
    arm,h,g,J,U=artificial();J[1:]=0;J[-1,0,0]=1e-200
    tiny=b.prepare_arm(h,g,.31,J,U,role='proposed',source_identity='tiny',input_identity='I',ld=1)
    tail,config,dist=b.configure(tiny,Fraction(1,40),16,0)
    assert tail.lambda_r>0 and len(tiny.plan.atoms)>0 and config is not None
    J[1:]=0
    zero=b.prepare_arm(h,g,.31,J,U,role='proposed',source_identity='zero',input_identity='I',ld=1)
    tail,config,dist=b.configure(zero,Fraction(1,40),16,0)
    assert tail.deterministic_only_certified and config is dist is None
    assert b.independent_occurrence(zero,tail,dist,config,master=1,round_index=0,axis='X',shot=0,occurrence=0)==()
    ctx,G=sector_generators(zero,h,g);k=b.kernels(zero,G,Fraction(1,40),None)
    assert b.reference.reference_round(k,(sum(G),.1),repetitions=4)['attenuation']==1.


def test_law_regeneration_monotone_attenuation_and_occurrence_seeds(monkeypatch):
    from trotterlib.rte import finite_rte_attenuation
    arm,h,g,J,U=artificial()
    other=b.prepare_arm(h,g,.31,J*.5,U,role='proposed',source_identity='different',input_identity='I',ld=1)
    ta,ca,da=b.configure(arm,Fraction(1,40),16,0);tb,cb,db=b.configure(other,Fraction(1,40),16,0)
    assert tb.lambda_r==pytest.approx(ta.lambda_r/2)
    for c in (ca,cb):assert finite_rte_attenuation(c,tail_evolutions=16)==pytest.approx((1+c.dimensionless_step_time**2)**(-16*c.rte_steps/2))
    assert finite_rte_attenuation(cb,tail_evolutions=16)>finite_rte_attenuation(ca,tail_evolutions=16)
    # An unequal coefficient change must also rebuild the probabilities.
    changed=J.copy();changed[-1]*=3
    third=b.prepare_arm(h,g,.31,changed,U,role='proposed',source_identity='third',input_identity='I',ld=1)
    tc,_,_=b.configure(third,Fraction(1,40),16,0)
    assert [x.probability for x in ta.components]!=[x.probability for x in tc.components]
    seeds=[]
    def sample(t,d,n,seed):seeds.append(seed);return tuple(range(n))
    monkeypatch.setattr(b.connection,'sample_occurrence',sample)
    for a,t,c,d in [(arm,ta,ca,da),(other,tb,cb,db)]:
        for occurrence in (0,1):
            events=b.independent_occurrence(a,t,d,c,master=7,round_index=1,axis='X',shot=3,occurrence=occurrence)
            assert len(events)==16  # K0 still has random events.
    assert len(set(seeds))==4


def test_full_space_and_sector_reference_agree_for_complete_factors():
    arm,h,g,_,_=artificial();ctx,G=sector_generators(arm,h,g);full,F=full_generators(arm,h,g)
    # SectorCAR's legacy excitation-table order differs from Qiskit's bit order.
    indices=b.stage1.bit_reverse_indices(4)[ctx.sector.basis_indices]
    for a,f in zip(G,F):np.testing.assert_allclose(a,f[np.ix_(indices,indices)],atol=1e-12)
    tail,c,d=b.configure(arm,Fraction(1,40),16,0)
    ks=b.kernels(arm,G,Fraction(1,40),c);kf=b.kernels(arm,F,Fraction(1,40),c)
    for key in ('pf','corrected'):np.testing.assert_allclose(ks[key],kf[key][np.ix_(indices,indices)],atol=1e-12)


def test_smaller_lambda_can_worsen_PF_for_the_same_total_Hamiltonian():
    from trotterlib import rte
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom,make_tail
    X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]);H=X+Z
    def config(weights):
        atoms=tuple(MixedAtom(name,'PHYSICAL_PAULI',Fraction(w),(name,),(name,),name) for name,w in weights)
        tail=make_tail(atoms,'artificial');c,_=rte.make_rte_config(tail,evolution_time=.2,rte_steps=16,
            truncation_tolerance=1.,finite_taylor_order=0,seed=1)
        return c
    ca=config([('X',2),('Z',2)]);cb=config([('Z',1)])
    ka=b.reference.reference_kernels([-H,2*H],Fraction(1,5),ld=0,rte_config=ca)
    kb=b.reference.reference_kernels([X,Z],Fraction(1,5),ld=0,rte_config=cb)
    ra=b.reference.reference_round(ka,(H,.2),repetitions=1);rb=b.reference.reference_round(kb,(H,.2),repetitions=1)
    assert cb.lambda_r<ca.lambda_r and rb['attenuation']>ra['attenuation']
    assert rb['pf_operator_error']>1e-4 and ra['pf_operator_error']<1e-12


@pytest.mark.parametrize('route',['baseline','proposed'])
def test_native_precise_negative_event_full_controlled_phase_adjoint_XY_P0(route):
    from qiskit.quantum_info import Operator
    from trotterlib import rte
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    arm,h,g,J,U=artificial();arm=b.prepare_arm(h,g,.31,J,U,role=route,source_identity=route,input_identity='I',ld=1)
    ctx,G=full_generators(arm,h,g);ops=b.atom_operators(ctx,arm)
    tail,config,dist=b.configure(arm,Fraction(1,40),2,0)
    events0=rte.enumerate_rte_events(tail.components,dist,max_events=100)
    negative=next(e for e in events0 if e.application_sequence[-1].coefficient_sign<0)
    positive=next(e for e in events0 if e.application_sequence[-1].coefficient_sign>0)
    events=(negative,positive)
    raw=b.build_kernel(arm,Fraction(1,40),events,config)
    ref=b.trajectory_reference(arm,G,ops,Fraction(1,40),events,config)
    expected=np.block([[np.eye(ctx.d),np.zeros((ctx.d,ctx.d))],[np.zeros((ctx.d,ctx.d)),ref]])
    matrix=Operator(raw).data;np.testing.assert_allclose(matrix,expected,atol=1e-11)
    adj=Operator(b.native.actual_adjoint(raw)).data
    np.testing.assert_allclose(adj@matrix,np.eye(2*ctx.d),atol=1e-11)
    compiled=compiled_wrapper(raw,CompilerProfile());M=Operator(compiled).data
    np.testing.assert_allclose(M,expected,atol=1e-10)
    assessed=b.assess_native_probe(raw,compiled,ref)
    assert assessed['whole_law_synthesis_bound'] is None and not assessed['global_phase_fit']
    corrupted=compiled.copy();corrupted.p(.02,arm.model.modes)
    # Keep the corrupted circuit native while checking relative controlled phase.
    corrupted=compiled_wrapper(corrupted,CompilerProfile())
    with pytest.raises(ValueError):b.assess_native_probe(raw,corrupted,ref)
    rng=np.random.default_rng(82);psi=rng.normal(size=ctx.d)+1j*rng.normal(size=ctx.d);psi/=np.linalg.norm(psi)
    observed=[]
    for axis in ('X','Y'):
        head,readout=b.connection.readout_segments(arm.model.modes,axis)
        state=Operator(readout.remove_final_measurements(inplace=False)).data@M@Operator(head).data@np.r_[psi,np.zeros(ctx.d)]
        observed.append(np.sum(abs(state[:ctx.d])**2)-np.sum(abs(state[ctx.d:])**2))
    assert complex(*observed)==pytest.approx(np.vdot(psi,ref@psi),abs=2e-10)
    metrics=circuit_metrics(compiled);point=b.cost_point([metrics],scope='complete_controlled_partial_S2_kernel',independent_trajectories=True)
    assert point['ideal_law_expectation'] is point['cost_support'] is point['lower'] is None
    assert point['sample_mean_standard_error'] is None
    with pytest.raises(ValueError):b.cost_point([metrics],scope='sum_of_individually_compiled_events',independent_trajectories=True)
    # Wrong arm/frame/event metadata must not silently pass the common bridge.
    with pytest.raises((KeyError,ValueError)):b.build_kernel(arm,Fraction(1,20),events,config)


def evidence_fixture(root):
    arm,h,g,J,U=artificial(4,8);initial=root/'input.npz';np.savez(initial,h=h,g=g,constant=np.array(.31))
    records=dict(result='stage_A.json',STOP='old_STOP.json',SOURCE='old_SOURCE.json')
    result=dict(status='STAGE_A_PROXY_AND_DF_PASS',sector={},fit_diagnostics={});exports={}
    for role,scale in [('baseline',1.),('proposed',.8)]:
        name=role+'.npz';np.savez(root/name,J=scale*J,U=U,x=np.zeros(128),jac=np.zeros(128));exports[role]=name
        mass=float(np.sum(abs(b.objective.coefficients(scale*J[2:],np))))
        result['sector'][role]=dict(actual_rank=8,sector_error=1e-8,numerical_guard=1e-9,guard_policy=m.molecule.GUARD_POLICY)
        result['fit_diagnostics'][role]=dict(optimizer=dict(success=True),diagnostic=dict(exact_tail_mass_Ha=mass))
    m.io.save(root/records['result'],result);m.io.save(root/records['STOP'],dict(failed=False,scientific_units=30,retry=0,resume=0,mandatory_stop=True))
    m.io.save(root/records['SOURCE'],dict(execution_commit='e1085c95f5c01b68566e17d543ace5832cb94061'))
    m.io.save(root/'source.json',dict(files={}))
    return dict(stage_A_input='input.npz',stage_A_records=records,final_exports=exports,
        saved_inputs={v:m.io.sha(root/v) for v in exports.values()},original_input=dict(sha256='synthetic'),source_manifest='source.json')


@pytest.mark.parametrize('fault',[None,'representation','mass','alias'])
def test_artificial_B1_end_to_end_141_calls_and_STOP(tmp_path,monkeypatch,fault):
    c=evidence_fixture(tmp_path);monkeypatch.setattr(m,'preflight',lambda *a:c)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_SOURCE\n')
    if fault=='representation':monkeypatch.setattr(b,'candidate_factor',lambda ctx,a,i:np.zeros((ctx.d,ctx.d)))
    if fault=='mass':
        p=tmp_path/'stage_A.json';e=m.io.read(p);e['fit_diagnostics']['baseline']['diagnostic']['exact_tail_mass_Ha']+=.1;p.write_text(__import__('json').dumps(e))
    if fault=='alias':monkeypatch.setattr(m,'ground_gate',lambda *a:(_ for _ in ()).throw(ValueError('ARTIFICIAL_ALIAS')))
    out=tmp_path/m.FRESH
    if fault:
        with pytest.raises(ValueError):m.run(tmp_path,'contract','input',out)
        assert m.io.read(out/'STOP.json')['failed'] and not (out/'baseline_m00.json').exists()
    else:
        m.run(tmp_path,'contract','input',out)
        stop=m.io.read(out/'STOP.json');assert stop['scientific_units']==141 and stop['call_counts']==m.LIMITS
        result=m.io.read(out/'result.json');assert len(result['arms'])==2
        assert all(len(a['rounds'])==13 and a['shots']is None for a in result['arms'].values())
        assert all(r['reference']['raw_signal_parts'] is not None for a in result['arms'].values() for r in a['rounds'])
        assert result['native_build']==result['sampling_samples']==0
        with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,'contract','input',out)


def test_header_only_schema_rejects_wrong_shapes(tmp_path):
    c=evidence_fixture(tmp_path);assert m.export_headers(tmp_path/c['final_exports']['baseline'])['J.npy']['shape']==[8,4,4]
    np.savez(tmp_path/'bad.npz',J=np.zeros((7,4,4)),U=np.zeros((8,4,4)),x=np.zeros(128),jac=np.zeros(128))
    with pytest.raises(ValueError,match='SCHEMA'):m.export_headers(tmp_path/'bad.npz')


def test_preflight_hash_schema_provenance_without_array_decode(tmp_path,monkeypatch):
    import json
    c=evidence_fixture(tmp_path);original=tmp_path/'original.npz';original.write_bytes(b'OPAQUE_ARTIFICIAL_ORIGINAL')
    required={c['stage_A_input'],*c['final_exports'].values(),*c['stage_A_records'].values()}
    c.update(scope=m.SCOPE,status='UNAPPROVED_PROPOSAL',policy=m.POLICY,allowances=m.ALLOWANCES,
        call_limits=m.LIMITS,maximum_logical_units=141,fresh_output=m.FRESH,adapter=b.VERSION,
        native_policy=b.NATIVE_POLICY,original_input=dict(sha256=m.io.sha(original)),
        source_manifest_sha256=m.io.sha(tmp_path/'source.json'),evaluation_packages={},
        saved_inputs={p:m.io.sha(tmp_path/p) for p in required})
    path=tmp_path/'contract.json';m.io.save(path,c)
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_ARRAY_DECODE')))
    assert m.preflight(tmp_path,path,original)==c
    (tmp_path/c['final_exports']['proposed']).write_bytes(b'corrupted export')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,path,original)
