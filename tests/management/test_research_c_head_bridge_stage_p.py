"""Invented inputs only; saved molecular arrays/QPY are never decoded."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
from trottertracks import c_head_reference_bridge as h
from trottertracks import c_stage_p_literal as p
from trottertracks import c_head_stage_p_execution as e
from trottertracks import c_per_word_population_range as literal


def population(a='1'):
    return dict(target_id='invented',D_norm_upper_exact=a,d_range_upper_exact=str(1+F(a)),
        accepted_P0_C_norm_exact='1',word_length=32,support=dict(positive=['a']),
        events={'a':dict(event_norm_upper_exact='1')},event_angle_bits={'a':(.125).hex()})


@pytest.mark.parametrize('fault',['none','tiny','phase','nonhermitian'])
def test_head_pairs_exact_tiny_phase_and_nonhermitian_are_enclosed(fault):
    from scipy.linalg import expm
    A=np.array([[.25,.125],[.125,-.25]],complex);B=A.copy()
    if fault=='tiny':B[0,0]+=2**-40
    if fault=='phase':B+=np.eye(2)*.25
    if fault=='nonhermitian':B[0,1]+=.01j
    pairs=[h.half_pair((A,B,F(1,10))) for _ in range(3)]
    result=h.products(pairs)
    observed=float(np.linalg.norm(np.linalg.matrix_power(expm(-.05j*A),3)-np.linalg.matrix_power(expm(-.05j*B),3),2))
    assert observed<=float(F(result['left']['upper_exact']))+1e-15
    if fault=='none':assert F(result['left']['upper_exact'])<F(1,10**80)
    if fault=='phase':assert F(result['left']['lower_exact'])>0
    value=h.kernel_bridge(result,population('1001/1000'))
    assert value['included_error_terms']==[h.TERM]
    rows=[h.round_bridge(value,m) for m in range(13)]
    assert len(rows)==13 and rows[-1]['phase_fit'] is False


@pytest.mark.parametrize('fault',['candidate','leakage','basis','nonfinite'])
def test_head_identity_and_exact_invariance_fail(fault):
    A=[np.eye(4,dtype=complex) for _ in range(3)];B=[np.eye(2,dtype=complex) for _ in range(3)];ix=np.array([0,3])
    candidate='x'
    if fault=='candidate':candidate='y'
    if fault=='leakage':A[0][0,1]=1e-40
    if fault=='basis':ix=np.array([0,0])
    if fault=='nonfinite':B[0][0,0]=np.nan
    with pytest.raises(ValueError):h.identity(A,B,ix,candidate=candidate,expected_candidate='x')


def test_full_and_sector_producers_use_different_rounding_on_invented_input():
    from trottertracks.c1_molecule_extension import SectorCAR
    from trottertracks import c1_stage1 as s
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(4231);ctx=SectorCAR(4);car=s.car_bilinears(8)
    U=np.linalg.qr(rng.normal(size=(4,4)))[0];J=rng.normal(size=(4,4));J=(J+J.T)/2
    with threadpool_limits(limits=1):
        full=s.raw_frame(J,U,car)[0];sector=ctx.raw_frame(J,U)
    ix=s.bit_reverse_indices(8)[ctx.sector.basis_indices]
    block=h.identity([full]*3,[sector]*3,ix,candidate='toy',expected_candidate='toy')[0]
    assert not np.array_equal(block,sector)
    assert np.max(np.abs(block-sector))<1e-12
    bound=h.half_pair((block,sector,F(1,10)))
    assert F(bound['generator_difference']['upper_exact'])>0


def test_B2_connection_has_three_disjoint_terms_and_no_saved_total_addition():
    rows=[dict(m=m,bound='1/100',target_id='toy',included_error_terms=[h.TERM]) for m in range(13)]
    b2=dict(rounds=[dict(m=m,bridge=dict(upper_exact='1/10'),reference_uncertainty=dict(upper_exact='1/5'),total_upper_exact='3/10') for m in range(13)])
    v=h.connect_b2(rows,b2,target_id='toy')
    assert v['rows'][0]['total_upper_exact']=='31/100' and len(v['rows'][0]['included_error_terms'])==3
    bad=deepcopy(rows);bad[0]['included_error_terms']*=2
    with pytest.raises(ValueError):h.connect_b2(bad,b2,target_id='toy')
    bad=deepcopy(rows);bad[0]['target_id']='other'
    with pytest.raises(ValueError):h.connect_b2(bad,b2,target_id='toy')
    rows[0]['bound']=None
    assert h.connect_b2(rows,b2,target_id='toy')['rows'][0]['total_upper_exact'] is None


@pytest.fixture(scope='module')
def word():
    from test_research_c_per_word_population_range import packet,raw
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    arm,heads,events,catalog,table,config=packet(True)
    reference=literal.build_literal_reference(arm,heads,events,r=2,parallel=False)
    qc=raw(arm,events,config);compiled=compile_checked(qc,CompilerProfile())
    R=p.fixed.record(qc);C=p.fixed.record(compiled)
    cr=p.compiled_raw(C,R,parallel=False);rd=literal.compare_literal(R,reference,parallel=False)
    return reference,qc,compiled,R,C,cr,rd


def test_two_leg_adapter_has_no_cap2_and_same_raw_identity(word):
    ref,q,c,R,C,cr,rd=word
    v=p.triangle(R,C,ref,cr,rd)
    assert F(v['upper_exact'])==F(cr['upper_exact'])+F(rd['upper_exact'])
    assert v['Head_bridge_to_B2_included'] is False
    huge=dict(rd,upper_exact='7',lower_exact='3')
    assert F(p.triangle(R,C,ref,cr,huge)['upper_exact'])>7
    wrong=deepcopy(cr);wrong['right_id']='another raw'
    with pytest.raises(ValueError):p.triangle(R,C,ref,wrong,rd)


@pytest.mark.parametrize('fault',['phase','word','target','angle'])
def test_perword_phase_target_and_population_binding(word,fault):
    ref,q,c,R,C,cr,rd=word;reference=deepcopy(ref)
    pop=population('1001/1000');pop['target_id']=ref['binding']['target_id'];pop['word_length']=2
    pop['event_angle_bits']=dict(ref['binding']['word'])
    assert not p.bind_population(reference,pop)['calibration_ready']
    if fault=='phase':
        bad=deepcopy(cr);bad['phase_fit']=True
        with pytest.raises(ValueError):p.triangle(R,C,ref,bad,rd)
    else:
        if fault=='word':reference['binding']['word'].pop()
        if fault=='target':reference['binding']['target_id']='other'
        if fault=='angle':pop['event_angle_bits'][ref['binding']['word'][0][0]]='0x1p-3'
        with pytest.raises(ValueError):p.bind_population(reference,pop)


@pytest.mark.parametrize('label',['head','X','Y'])
def test_shells_phase_and_exact_embedding(label):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(3)
    if label=='Y':qc.sdg(2)
    qc.h(2)
    value=p.shell_certificate(qc,qc,label)
    assert F(value['compiled_ideal']['upper_exact'])<F(1,10**40)
    shifted=qc.copy();shifted.global_phase=.125
    assert F(p.shell_certificate(qc,shifted,label)['compiled_ideal']['lower_exact'])>0
    wrong=qc.copy();wrong.x(0)
    with pytest.raises(ValueError):p.shell_certificate(wrong,wrong,label)


def test_actual_inverse_is_same_realized_circuit_and_not_literal_inverse(word):
    from trottertracks import c1_stage2_native as native
    ref,q,c,R,C,cr,rd=word
    inv=p.fixed.record(native.actual_adjoint(q))
    value=p.fixed.compare(inv,p.fixed.adjoint(R),parallel=False)
    assert F(value['upper_exact'])<F(1,10**10)
    assert ref['input_U_replaced'] is False


def test_finite_signal_uses_both_norms_no_implicit_2_and_unknown_is_not_zero():
    v=dict(binding=dict(target_id='invented'),upper_exact='3')
    pop=population('4');s={k:'1/100' for k in ('head','X','Y')}
    result=p.finite_signal(v,pop,s)
    assert F(result['axes']['X']['upper_exact'])>=F('15.25')
    s['head']=None
    assert p.finite_signal(v,pop,s)['X'] is None
    assert not result['all_law_synthesis_qualified']


def test_full512_artificial_column_certificate_keeps_controlled_phase():
    R=p.fixed.identity(9);C=deepcopy(R)
    C['gates']=[dict(name='p',qubits=[8],params_hex=[(.125).hex()])]
    v=p.compiled_raw(C,R,parallel=False)
    assert v['dimension']==v['columns_evaluated']==512 and F(v['lower_exact'])>0


@pytest.mark.parametrize('stage',['H','P'])
def test_real_passive_contract_cold_no_numeric_imports(stage):
    code="""
import sys
from pathlib import Path
class Block:
 def find_spec(self, name, *args):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_MOLECULAR_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_head_stage_p_execution as m
r=m.preflight(Path.cwd(),m.CONTRACTS[sys.argv[1]])
assert r['scientific_calls']==r['arrays_decoded']==0
assert r['head_bridge_upper_exact'] is None and r['per_word_upper_exact'] is None
"""
    subprocess.run([sys.executable,'-B','-c',code,stage],cwd=Path(__file__).resolve().parents[2],check=True)


@pytest.mark.parametrize('stage',['H','P'])
@pytest.mark.parametrize('failure',[True,False])
def test_mock_lifecycle_partial_FAIL_fresh_and_STOP(tmp_path,monkeypatch,stage,failure):
    c=dict(stage=stage,fresh_output='fresh',source_manifest='SOURCE.json')
    (tmp_path/'SOURCE.json').write_text('{}');(tmp_path/'contract').write_text(json.dumps(c))
    monkeypatch.setattr(e,'preflight',lambda *a:{})
    def toy(root,c,out,calls):
        def body():
            if failure:raise ValueError('INVENTED_FAIL')
            return dict(status='MOCK_COMPLETE')
        return calls('summary','toy',body)
    if failure:
        with pytest.raises(ValueError):e.run(tmp_path,tmp_path/'contract',execute_science=True,executor=toy)
    else:e.run(tmp_path,tmp_path/'contract',execute_science=True,executor=toy)
    stop=json.loads((tmp_path/'fresh/STOP.json').read_text())
    assert stop['scientific_units']==1 and bool(stop['failed']) is failure and stop['mandatory_stop']
    with pytest.raises(FileExistsError):e.run(tmp_path,tmp_path/'contract',execute_science=True,executor=toy)
    with pytest.raises(PermissionError):e.run(tmp_path,tmp_path/'contract')


@pytest.mark.parametrize('stage',['H','P'])
@pytest.mark.parametrize('fault',['identity','basis','precision','source','input','retry','range'])
def test_real_metadata_gate_mutations_FAIL_without_array_decode(tmp_path,stage,fault):
    root=Path(__file__).resolve().parents[2];c=e.io.read(root/e.CONTRACTS[stage])
    if fault=='identity':c['identity']['model']['independent_events']=1
    if fault=='basis':c['basis_order']='unregistered permutation'
    if fault=='precision':c['precision_bits']=53
    if fault=='source':c['source_manifest_sha256']='0'*64
    if fault=='input':c['saved_inputs'][c['full_heads']]='0'*64
    if fault=='retry':c['retry']=1
    if fault=='range':c['population_M_exact']='2'
    path=tmp_path/'bad.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError):e.preflight(root,path)


def pipeline_fixture(root):
    from test_research_c_population_range_schema import metadata_fixture
    from trottertracks import c_pr_aware_df_stage_b as b
    J=np.full((8,4,4),.0009765625);U=np.array([np.eye(4)]*8)
    d=dict(h=np.diag([.125,.25,.375,.5]),g=np.zeros((4,)*4),constant=np.asarray(.125))
    np.savez(root/'input.npz',**d);np.savez(root/'candidate.npz',J=J,U=U)
    digest=e.io.sha(root/'candidate.npz')
    arm=b.prepare_arm(**d,J=J,U=U,role='proposed',source_identity=digest,input_identity='invented',ld=2)
    atoms=[dict(atom_id=a.atom_id,frame_id=a.frame_id,support=list(a.support),coefficient=str(a.coefficient)) for a in arm.plan.atoms]
    c=metadata_fixture(root,atoms,digest)
    table=e.io.read(root/c['grid_table']);table['ideal_probabilities']=['1/216']*216
    (root/c['grid_table']).write_text(json.dumps(table))
    heads=np.zeros((3,256,256),complex);np.savez(root/'heads.npz',heads=heads)
    np.savez(root/'sector.npz',J=J,U=U,generators=np.zeros((9,36,36),complex))
    from trottertracks.c1_molecule_extension import SectorCAR
    np.savez(root/'basis.npz',sector_basis_indices=SectorCAR(4).sector.basis_indices)
    c.update(parent_roles=dict(input='input.npz',candidate='candidate.npz',sector='sector.npz',basis='basis.npz'),
        full_heads='heads.npz',saved_inputs={'input.npz':e.io.sha(root/'input.npz')},
        word=['CDF_proposed:F6:0','CDF_proposed:F7:6,7']*16,
        tail_phase_identity_hex=float(arm.plan.extracted_identity).hex(),normalization='norm.json')
    (root/'norm.json').write_text(json.dumps(dict(tau=np.tan(.015625))))
    binding=literal.validate(arm,heads,tuple(__import__('types').SimpleNamespace(rotation_component_id=k,
        rotation_angle=.015625,taylor_order=0,phase=1,product_component_ids=()) for k in c['word']),h=F(1,10),r=32)
    pop=population();pop['target_id']=binding['target_id'];pop['event_angle_bits']={a.atom_id:(.015625).hex() for a in arm.plan.atoms}
    pop['events']={a.atom_id:dict(event_norm_upper_exact='1') for a in arm.plan.atoms}
    pop['support']['positive']=list(pop['events'])
    return c,pop


def test_artificial_head_runner_exact_26_calls_no_native_or_H4(monkeypatch,tmp_path):
    from concurrent.futures import Future
    from trottertracks.c1_stage3_integrated import Calls
    c,pop=pipeline_fixture(tmp_path)
    monkeypatch.setattr(e,'stage_r',lambda *a:pop)
    b2=dict(rounds=[dict(m=m,bridge=dict(upper_exact='0'),reference_uncertainty=dict(upper_exact='0'),total_upper_exact='0') for m in range(13)])
    monkeypatch.setattr(e,'b2_metadata',lambda *a:b2)
    class Immediate:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,*args):
            future=Future()
            try:future.set_result(fn(*args))
            except BaseException as exc:future.set_exception(exc)
            return future
    import concurrent.futures
    monkeypatch.setattr(concurrent.futures,'ProcessPoolExecutor',Immediate)
    out=tmp_path/'H';out.mkdir();calls=Calls(out,e.LIMITS['H'],e.MAXIMUM['H'])
    result=e.execute_h(tmp_path,c,out,calls)
    assert calls.counts==e.LIMITS['H'] and sum(calls.counts.values())==26
    assert result['status']=='HEAD_BRIDGE_COMPLETE' and not result['Stage_C_ready']


def test_artificial_P_runner_routing_build_compile_records_inverse_shells(monkeypatch,tmp_path):
    # Mathematical engines are tested above; here replace expensive column
    # calculations with explicit mocks to test the complete P orchestration.
    from trottertracks.c1_stage3_integrated import Calls
    from qiskit import QuantumCircuit,qpy
    c,pop=pipeline_fixture(tmp_path);monkeypatch.setattr(e,'stage_r',lambda *a:pop)
    c['shells']={}
    for label in ('head','X','Y'):
        q=QuantumCircuit(9)
        if label=='Y':q.sdg(8)
        q.h(8);c['shells'][label]={}
        for kind in ('raw','compiled'):
            path=label+'_'+kind+'.qpy'
            with (tmp_path/path).open('wb') as f:qpy.dump(q,f)
            c['shells'][label][kind]=path
    def mock_compare(left,right,**kwargs):
        return dict(status='MOCK_PAIR',dimension=1<<left['qubits'],columns_evaluated=1<<left['qubits'],
            phase_fit=False,upper_exact='0',lower_exact='0')
    monkeypatch.setattr(p.fixed,'compare',mock_compare)
    def mock_ref(arm,heads,events,**kwargs):
        return dict(binding=literal.validate(arm,heads,events,h=F(1,10),r=32))
    monkeypatch.setattr(literal,'build_literal_reference',mock_ref)
    monkeypatch.setattr(literal,'compare_literal',lambda raw,ref,**kwargs:dict(
        raw_id=literal._digest(raw),target_id=ref['binding']['target_id'],word_id=ref['binding']['word_id'],
        dimension=512,columns_evaluated=512,phase_fit=False,upper_exact='0',lower_exact='0'))
    out=tmp_path/'P';out.mkdir();calls=Calls(out,e.LIMITS['P'],e.MAXIMUM['P'])
    result=e.execute_p(tmp_path,c,out,calls)
    assert sum(calls.counts.values())<=41 and calls.first_failure is None
    assert calls.counts['kernel_build']==calls.counts['compile']==1
    assert calls.counts['adjoint_certificate']==2 and calls.counts['shell_certificate']==3
    assert len(list(out.glob('*.qpy')))==4
    assert result['status']=='STAGE_P_ONE_WORD_COMPLETE' and result['calibration_ready'] is False
