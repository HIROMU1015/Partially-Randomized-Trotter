"""Invented arrays only: no molecular NPZ, QPY, pickle or scientific results."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_h4_relocated_downstream as rel
from trottertracks import c_pr_aware_df_stage_b as b
from test_research_c_pr_aware_df_stage_b import artificial


@pytest.fixture(autouse=True)
def no_saved_array_decode(monkeypatch):
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    monkeypatch.setattr(loader,'load',lambda *a,**k:pytest.fail('saved-array loading forbidden in synthetic tests'))


def fixture(n=2,rank=3,role='baseline',scale=1.):
    _,k,_,J,U=artificial(n,rank,scale)
    return rel.prepare(k,.31,J,U,role=role,source_identity='invented:'+role,input_identity='invented-input',ld=1 if n==2 else 2)


def full_context(modes):
    car=b.stage1.car_bilinears(modes)
    return car,SimpleNamespace(d=car.shape[-1],bilinears={(p,q):csr_matrix(car[p,q])
        for p in range(modes) for q in range(modes) if p%2==q%2})


def test_passive_import_and_logical_work():
    subprocess.run([sys.executable,'-B','-c',"import sys; sys.path.insert(0,'src'); from trottertracks import c_h4_relocated_downstream as r; assert r.logical_work()['native_PR_max']==800; assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"],cwd=ROOT,check=True)
    assert rel.logical_work()['full_controlled_decodes_max']==3212


@pytest.mark.parametrize('role',['baseline','proposed'])
@pytest.mark.parametrize('n,rank',[(2,3),(4,8)])
def test_number_fermion_sector_full_scalar_identity_and_mass(role,n,rank):
    from trottertracks.c1_molecule_extension import SectorCAR
    from trottertracks.c_h4_z_relocation import diagnose
    r=fixture(n,rank,role);a=r.arm
    ctx=SectorCAR(n);old,G=rel.raw_generators(r,ctx=ctx)
    for i,x in enumerate(old):np.testing.assert_allclose(b.candidate_factor(ctx,r.original,i),x,atol=1e-12,rtol=0)
    np.testing.assert_allclose(sum(old),sum(G),atol=2e-12,rtol=0)
    car,fctx=full_context(2*n);fo,fg=rel.raw_generators(r,car=car)
    np.testing.assert_allclose(sum(fo),sum(fg),atol=2e-12,rtol=0)
    ix=b.stage1.bit_reverse_indices(2*n)[ctx.sector.basis_indices]
    for s,f in zip(G,fg):np.testing.assert_allclose(s,f[np.ix_(ix,ix)],atol=2e-12,rtol=0)
    outside=np.setdiff1d(np.arange(len(fg[0])),ix)
    for f in fg:np.testing.assert_allclose(f[np.ix_(outside,ix)],0,atol=1e-12,rtol=0)
    tail,c,d=b.configure(a,Fraction(1,10),8,0)
    assert rel.tail_gate(ctx,a,G,tail)['status']=='PASS'
    assert rel.tail_gate(fctx,a,fg,tail)['status']=='PASS'
    assert tail.lambda_r==pytest.approx(diagnose(a.J,ld=a.plan.ld)['M_ZZ'],abs=1e-13)
    assert a.plan.extracted_identity==r.original.plan.extracted_identity
    assert a.model.constant==pytest.approx(.31-r.rows[a.plan.ld:].sum())
    assert all(not block.diagonal.rz for block in a.model.blocks[a.plan.ld+1:])
    for oldb,newb in zip(r.original.model.blocks[1:],a.model.blocks[1:]):
        assert oldb.rotation==newb.rotation and oldb.diagonal.rzz==newb.diagonal.rzz
        assert oldb.diagonal.global_phase==newb.diagonal.global_phase
    np.testing.assert_array_equal(a.J,r.original.J);np.testing.assert_array_equal(a.U,r.original.U)
    assert not a.J.flags.writeable and not a.U.flags.writeable and a.general is None


@pytest.mark.parametrize('fault',['scalar','identity','tailZ'])
def test_scalar_identity_and_linear_corruption_is_detected(fault):
    r=fixture();car,ctx=full_context(4);_,G=rel.raw_generators(r,car=car)
    tail,_,_=b.configure(r.arm,Fraction(1,10),8,0)
    if fault=='scalar':
        bad=replace(r,arm=replace(r.arm,model=replace(r.arm.model,constant=r.arm.model.constant+.01)))
        with pytest.raises(ValueError):rel.raw_generators(bad,car=car)
    else:
        plan=replace(r.arm.plan,extracted_identity=r.arm.plan.extracted_identity+.01) if fault=='identity' else r.original.plan
        with pytest.raises(ValueError):rel.tail_gate(ctx,replace(r.arm,plan=plan),G,tail)


@pytest.mark.parametrize('role',['baseline','proposed'])
@pytest.mark.parametrize('mode',['PR','D'])
@pytest.mark.parametrize('n,rank',[(2,3),(4,8)])
def test_native_P0_raw_compiled_actual_inverses_phase_shell_XY_leakage(role,mode,n,rank):
    from qiskit.quantum_info import Operator
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    r=fixture(n,rank,role=role);a=r.arm;car,ctx=full_context(2*n);_,G=rel.raw_generators(r,car=car)
    tail,config,dist=b.configure(a,Fraction(1,10),8,0)
    if mode=='PR':
        events=b.independent_occurrence(a,tail,dist,config,master=9,round_index=-1,axis='PRICE_ONLY',shot=0,occurrence=0)
        ref=b.trajectory_reference(a,G,b.atom_operators(ctx,a),Fraction(1,10),events,config)
    else:events=();config=None;ref=b.kernels(a,G,Fraction(1,10),None)['pf']
    raw=b.build_kernel(a,Fraction(1,10),events,config)
    compiled=compile_checked(raw,CompilerProfile())
    R,target,_=b2.raw_gate(raw,ref);C,_=b2.compiled_gate(compiled,R,target)
    circuits,_=b2.adjoint_pair_gate(raw,compiled,R,C,target)
    assert len(circuits)==2
    shells={}
    for label in ('head','X','Y'):
        s=b2.shell_build(2*n,label);cs=compile_checked(s,CompilerProfile())
        shells[label],_=b2.shell_assessment(s,cs,label)
    # A fixed sector state and full operator leakage; no projected circuit.
    dimension=1<<(2*n)
    ix=b.stage1.sector_indices(n,n//2,n//2);psi=np.zeros(dimension,complex);psi[ix[0]]=1
    assert b2.XY_gate(R,C,ref,psi,shells)['status']=='PASS'
    outside=np.setdiff1d(np.arange(dimension),ix)
    np.testing.assert_allclose(ref[np.ix_(outside,ix)],0,atol=1e-11,rtol=0)
    bad=compiled.copy();bad.p(.02,2*n)
    with pytest.raises(ValueError):b2.compiled_gate(bad,R,target)
    wrong=b.native.actual_adjoint(raw);wrong.p(.03,2*n)
    with pytest.raises(AssertionError):np.testing.assert_allclose(Operator(wrong).data@R,np.eye(2*dimension),atol=1e-10,rtol=0)


def synthetic_packet(tmp_path):
    from trottertracks.c1_molecule_extension import SectorCAR
    r=fixture(4,8);ctx=SectorCAR(4);old,G=rel.raw_generators(r,ctx=ctx)
    saved=dict(J=r.arm.J,U=r.arm.U,generators=np.asarray(old),H=sum(old))
    df=dict(candidate_sha256='invented:baseline',input_sha256='invented-input',budget_PASS=True,
        actual_rank=8,budget_Ha=.00019875,kappa='original_g_corrected',scalar='nuclear_once',
        guard_policy='sector_operation_scale_guard8_proposal_v1',sector_error=0.,numerical_guard=1e-12)
    calls=[]
    def call(k,l,f):calls.append(k);return f()
    rr,a,report=rel.prepare_candidate(r.kappa,.31,r.arm.J,r.arm.U,saved,df,
        candidate='invented:baseline',input_identity='invented-input',role='baseline',out=tmp_path/'A',call=call,shell_reports={})
    return rr,a,call,calls


def test_artificial_reference_RPE_13_rounds_all_registered_r_and_binding(tmp_path):
    r,a,call,calls=synthetic_packet(tmp_path);cache=None
    for n in (32,8,16,64,128):
        ar,result,packet,cache=rel.reference_r(a,n,tmp_path/f'r{n}',call,pf_cache=cache)
        assert result['r']==n and result['candidate']==a['arm'].model.source_identity
        assert [x['m'] for x in result['rounds']]==list(range(13))
        for lane,v in result['lanes'].items():
            assert len(v['rounds'])==13
            if lane=='evidence':assert v['total_shots'] is None
    assert calls.count('exact_evolution')==13 and calls.count('kernel_reference')==1
    other=dict(a,arm=replace(a['arm'],model=replace(a['arm'].model,source_identity='other')))
    with pytest.raises(ValueError,match='MIXED_PF'):rel.reference_r(other,16,tmp_path/'bad',call,pf_cache=cache)


def rows(candidate='candidate',split='selection',best=8):
    return [dict(r=r,candidate_sha256=candidate,split=split,samples=32,correctness='PASS',
        eligible=True,total_rz=10. if r==best else 20.) for r in rel.R_VALUES]


def test_fixed_selection_final_no_reselection_ineligible_and_best_D():
    selection={x:rows(x) for x in ('P','E1','E2')}
    ds={x:dict(candidate_sha256=x,eligible=True,total_rz=12.) for x in selection}
    plan=rel.final_plan(selection,ds)
    assert plan['PR_final_pairs']==[('E1',8),('E1',32),('E2',8),('P',8),('P',32)]
    assert plan['best_execution']==dict(P='PR',E1='PR',E2='PR')
    ds['E1']['total_rz']=9.
    assert rel.final_plan(selection,ds)['best_execution']['E1']=='D'
    assert rel.select_r(rows(),'candidate')['selected_r']==8  # boundary retained, never expanded
    with pytest.raises(ValueError):rel.select_r(rows(split='final'),'candidate')
    with pytest.raises(ValueError):rel.select_r(rows()[:-1],'candidate')
    rr=rows()
    for x in rr:x.update(eligible=False,total_rz=None)
    assert rel.select_r(rr,'candidate')['selected_r'] is None
    rr[0]['total_rz']=0
    with pytest.raises(ValueError):rel.select_r(rr,'candidate')


def test_selection_final_candidate_batch_seeds_are_disjoint():
    seeds=[rel.seed(c,s,r,b,i) for c in ('P','E1','E2') for s in ('selection','final')
        for r in rel.R_VALUES for b in range(2) for i in range(32)]
    assert len(seeds)==len(set(seeds))


def test_native_cohort_uses_actual_disjoint_domains_and_no_sample_double_count(tmp_path,monkeypatch):
    from trottertracks import c_integrated_retuning_adapter as adapter
    r,a,call,_=synthetic_packet(tmp_path)
    seen=[]
    class IO:
        save=staticmethod(lambda *args:None)
    def native(root,contract,out,call,ar,*,sampling,prepared):
        for i in range(32):
            seen.append(b.connection.trajectory_seed((sampling['master'],sampling['cohort'],contract['source_manifest_sha256']),
                ar['arm'].role,ar['arm'].model.source_identity,-1,sampling['axis'],i,0))
        return [dict(rz_count=7)]*32,{}
    perf=SimpleNamespace(native=native,b2=SimpleNamespace(calibration=lambda rows:{}),io=IO,
        accounting=SimpleNamespace(cost_calibration=lambda p,expected_n:dict(n=expected_n)))
    monkeypatch.setattr(adapter,'_engines',lambda:(SimpleNamespace(perf=perf,io=IO,
        m3=SimpleNamespace(serial_cost=lambda *a:None)),None,None))
    prepared=dict(arm=a['arm']);aa=dict(a,config=SimpleNamespace(rte_steps=8))
    result=dict(candidate=a['arm'].model.source_identity,r=8,native_admitted=True,lanes={})
    for split in ('selection','final'):
        value,packet=rel.native_cohort(tmp_path,dict(source_manifest_sha256='synthetic-source'),tmp_path/split,call,aa,result,{},split=split,prepared=prepared)
        assert len(packet['kernel_samples'])==(32 if split=='selection' else 64)
    expected=[rel.seed(a['arm'].model.source_identity,s,8,bt,i,role=a['arm'].role)
        for s,batches in [('selection',1),('final',2)] for bt in range(batches) for i in range(32)]
    assert seen==expected and len(set(seen))==96


def test_same_relocated_D_full_pipeline(tmp_path):
    from qiskit import qpy
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    r,a,call,_=synthetic_packet(tmp_path);car,_=full_context(8)
    _,G=rel.raw_generators(r,car=car)
    contract=dict(shells={});reports={}
    for label in ('head','X','Y'):
        raw=b2.shell_build(8,label);compiled=compile_checked(raw,CompilerProfile())
        contract['shells'][label]={}
        for kind,q in [('raw',raw),('compiled',compiled)]:
            p=tmp_path/f'{label}_{kind}.qpy'
            with p.open('wb') as stream:qpy.dump(q,stream)
            contract['shells'][label][kind]=p.name
        reports[label]=dict(metrics=circuit_metrics(compiled))
    a['saved']['shells']=reports
    out=tmp_path/'D';out.mkdir()
    value=rel.strong_d(tmp_path,contract,out,call,a,G)
    assert value['candidate']==a['arm'].model.source_identity
    assert (out/'PF.json').exists() and len(value['PF_rounds'])==13
    assert value['actual_CA_achieved'] is None
    assert value['metrics'] is not None and (out/'adjoint.json').exists()


def test_affine_accounting_single_s1_weight(tmp_path):
    from trottertracks import c_near_r32_statistics as stats
    from trottertracks.c1_stage3_accuracy import METRICS
    r,a,call,_=synthetic_packet(tmp_path)
    ar,result,packet,_=rel.reference_r(a,32,tmp_path/'ref',call)
    packet['shells']={s:dict(metrics={k:2 for k in METRICS}) for s in ('head','X','Y')}
    for lane,value in result['lanes'].items():
        if not value['energy']['same_accuracy']:continue
        W=sum(2*p['shots']*(1<<p['m']) for p in value['rounds'])
        value['weighted_SE']=dict(kernel_reuse_weight=W)
    binding=stats.affine(packet,result)['ideal_exact_native']
    plans=result['lanes']['ideal_exact_native']['rounds']
    assert binding['W']==sum(2*p['shots']*(1<<p['m']) for p in plans)
    assert binding['shell']['rz_count']==4*sum(2*p['shots'] for p in plans)
    assert binding['W']!=32*binding['W']  # r and price-calibration N absent from W


def test_frozen_best_policy_comparisons_never_use_final_for_selection(monkeypatch):
    from trottertracks import c_h4_relocated_execution as execution
    from trottertracks import c_near_r32_statistics as stats
    called=[]
    monkeypatch.setattr(stats,'primary',lambda *args,**kw:called.append(kw['statistics']['confidence']) or dict(status='synthetic'))
    plan=dict(tuned_r=dict(P=8,E1=16,E2=128),best_execution=dict(P='D',E1='D',E2='D'))
    finals={(s,r):dict(rows=[],bindings=dict(ideal_exact_native={})) for s,r in [('P',32),('E1',32),('P',8),('E2',128)]}
    states={s:dict(D=dict(ideal_exact_native=dict(cost=dict(totals=dict(rz_count=100))))) for s in ('P','E1','E2')}
    result=execution.comparisons(dict(statistics=dict(primary=dict(confidence=.975))),plan,finals,states)
    assert called==[.975,.975] and result['secondary_best_policy_P_vs_E1']['choices']==['D','D']
    assert plan['tuned_r']['P']==8


@pytest.mark.parametrize('failure',[False,True])
def test_partial_STOP_fresh_no_retry(tmp_path,failure):
    out=tmp_path/'run01';tasks=[dict(id='P'),dict(id='E1')]
    def work(task,output):
        if failure and task['id']=='E1':raise ValueError('invented phase error')
        return dict(status='SYNTHETIC_PASS')
    if failure:
        with pytest.raises(ValueError):rel.run_fresh(out,tasks,work)
    else:rel.run_fresh(out,tasks,work)
    stop=json.loads((out/'STOP.json').read_text())
    assert stop['failed']==failure and stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert (out/'P.json').exists()
    with pytest.raises(FileExistsError):rel.run_fresh(out,tasks,work)
