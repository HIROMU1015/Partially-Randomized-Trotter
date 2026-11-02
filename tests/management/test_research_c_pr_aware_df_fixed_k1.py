"""Invented K1 data only; saved molecular NPZ/QPY are never decoded."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from flint import acb_mat,ctx
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from test_research_c_pr_aware_df_fixed_car import example,build_raw
from trottertracks import c_pr_aware_df_fixed_k1 as k
from trottertracks import c_pr_aware_df_fixed_k1_execution as e
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks.c1_stage3_integrated import Calls
CONTRACT=ROOT/'docs/research/research_c_pr_aware_df_F2_K1_contract_draft_20261027.json'


def dense(reference):
    out=np.zeros((reference['system_dimension'],)*2,complex)
    for block in reference['blocks']:
        indices=block['indices'];a=k.car.unpack(block['entries'],len(indices))
        for i,p in enumerate(indices):
            for j,q in enumerate(indices):out[p,q]=complex(float(a[i,j].real.mid()),float(a[i,j].imag.mid()))
    return out


@pytest.mark.parametrize('anti',[0.,2.**-50,.25])
def test_exact_complex_H_A(anti):
    h=np.array([[.125+anti*1j,.25+.5j],[.25-.5j,-.25-anti*1j]])
    parts=k.exact_kappa(h)
    for i in range(2):
        for j in range(2):
            assert k.diagnostic.add(parts['H'][i][j],parts['A'][i][j])==k.diagnostic.q(h[i,j])
            assert parts['H'][i][j]==(parts['H'][j][i][0],-parts['H'][j][i][1])
            assert parts['A'][i][j]==(-parts['A'][j][i][0],parts['A'][j][i][1])
    assert (parts['anti_spectral_upper']==0)==(anti==0)


def test_rational_H_not_float64_and_subnormal_half():
    parts=k.exact_kappa(np.array([[0.,2.**53],[1.,0.]]))
    assert parts['H'][0][1][0]==F(2**53+1,2)
    assert parts['H'][0][1][0]!=F(float(parts['H'][0][1][0]))
    parts=k.exact_kappa(np.array([[0.,np.nextafter(0.,1.)],[0.,0.]]))
    with ctx.workprec(192):
        ball=k.ball_coefficients(parts['H'])[0][1].real
        assert ball>0 and k.z._fraction(ball.mid())==F(1,2**1075)


def test_full_Fock_H_A_reuses_SectorCAR():
    full=b.stage1.car_bilinears(4)
    E=[[k.car.Matrix(k.z._matrix(full[2*p,2*q]+full[2*p+1,2*q+1])) for q in range(2)] for p in range(2)]
    context=SimpleNamespace(n=2,d=16,E=E)
    parts=k.exact_kappa(np.array([[.125+.01j,.25+.5j],[.25-.4j,-.25-.02j]]))
    with ctx.workprec(192):
        literal=k.head(context,parts,.125,component='literal').ball
        H=k.head(context,parts,.125,component='H').ball;A=k.head(context,parts,component='A').ball
        assert F(k.z._norm(literal-H-A)['upper_exact'])==0
        assert F(k.z._norm(H-H.conjugate().transpose())['upper_exact'])==0
        assert F(k.z._norm(A+A.conjugate().transpose())['upper_exact'])==0


def test_two_half_heads_norm_growth_and_full_particle_count():
    parts=k.exact_kappa(np.diag([.125+.2j,-.25+.1j]))
    with ctx.workprec(192):
        bridge=k.kappa_bridge(parts,F(1,40),dict(upper_exact='1/100'))
        mu=F(bridge['lifted_anti_norm_upper_exact'])
        expected=(1+k.z._real(F(1,100)))*((k.z._real(F(1,40)*mu)).exp()-1)
        assert F(bridge['upper_exact'])>=k.z._fraction(expected.lower())
        assert F(bridge['upper_exact'])>k.z._fraction(((k.z._real(F(1,80)*mu)).exp()-1).upper())
    assert bridge['full_Fock_particle_upper']==4 and bridge['half_head_occurrences']==2
    assert F(bridge['middle_norm_upper_exact'])==F(101,100)
    assert F(bridge['literal_trajectory_norm_upper_exact'])>1


@pytest.mark.parametrize('scaled',[False,True])
def test_noncommuting_literal_and_two_model_bridges(example,scaled):
    arm,h,events,_=example
    if scaled:
        U=arm.U.copy();U[2]*=1.001
        arm=b.prepare_arm(h,np.zeros((2,)*4),.19,arm.J,U,role='baseline',source_identity='a',input_identity='b',ld=2)
    literal=h.astype(complex);literal[0,1]+=.002;literal[1,0]-=.002
    ref=k.build_reference(arm,literal,.19,events,parallel=False)
    H=np.array([[complex(float(x),float(y)) for x,y in row] for row in k.exact_kappa(literal)['H']])
    herm=k.car.build_reference(arm,H,.19,events,parallel=False)
    assert np.linalg.norm(dense(ref)-dense(herm),2)<=float(F(ref['kappa_bridge']['upper_exact']))+1e-14
    assert F(ref['U_bridge']['upper_exact'])>0
    assert ref['target']==k.TARGET!=k.car.TARGET and ref['reference_independent_of_QPY']


@pytest.fixture
def certificate(example):
    arm,h,events,config=example;literal=h.astype(complex);literal[0,0]+=1j*2.**-45
    ref=k.build_reference(arm,literal,.19,events,parallel=False)
    record=k.fixed.record(build_raw(arm,events,config))
    result=k.compare(record,ref,raw_identity='rawhash',parallel=False)
    return record,ref,result


def test_raw_and_compiled_triangle_counts_once(certificate):
    record,ref,result=certificate
    assert result['columns_evaluated']==result['dimension']==32
    total=F(result['literal_CAR_upper_exact'])+F(result['kappa_bridge_upper_exact'])+F(result['U_bridge_upper_exact'])
    assert F(result['upper_exact'])==total<F(1,10**8)
    prior=k.fixed.compare(record,record,parallel=False)
    composed=k.compose(prior,result,raw_identity='rawhash',expected_raw_identity='rawhash')
    assert F(composed['upper_exact'])==F(prior['upper_exact'])+total
    assert composed['target']==k.TARGET and len(composed['counted_terms'])==4


@pytest.mark.parametrize('fault',['raw','norm','target','phase'])
def test_triangle_conditions(certificate,fault):
    record,_,row=certificate;row=deepcopy(row);prior=k.fixed.compare(record,record,parallel=False)
    if fault=='raw':row['raw_identity_sha256']='other'
    if fault=='norm':row['norm']='probe max'
    if fault=='target':row['target']=k.car.TARGET
    if fault=='phase':row['phase_fit']=True
    with pytest.raises(ValueError):k.compose(prior,row,raw_identity='rawhash',expected_raw_identity='rawhash')


def test_uncertainty_retained_and_null_unresolved(certificate):
    record,reference,base=certificate;ref=deepcopy(reference);pair=ref['blocks'][0]['entries'][0][0]
    pair[0]=str(F(pair[0])-F(1,100));pair[1]=str(F(pair[1])+F(1,100))
    inflated=k.compare(record,ref,raw_identity='rawhash',parallel=False)
    assert F(inflated['literal_CAR_upper_exact'])>F(base['literal_CAR_upper_exact'])+F(1,200)
    ref['kappa_bridge']['upper_exact']=None
    row=k.compare(record,ref,raw_identity='rawhash',parallel=False)
    assert row['upper_exact'] is None and row['CA_achieved'] is None
    assert k.compose(None,row,raw_identity='rawhash',expected_raw_identity='rawhash')['upper_exact'] is None


def test_large_literal_not_clipped_to_two(example):
    from qiskit import QuantumCircuit
    arm,h,events,_=example;literal=h.astype(complex)+100j*np.eye(2)
    ref=k.build_reference(arm,literal,.19,events,parallel=False)
    row=k.compare(k.fixed.record(QuantumCircuit(5)),ref,raw_identity='invented',parallel=False)
    assert F(row['literal_CAR_upper_exact'])>2
    assert F(row['upper_exact'])<=2


def test_exact_U_full512_and_relative_vacuum_phase():
    from qiskit import QuantumCircuit
    h=np.zeros((4,4));J=np.zeros((3,4,4));U=np.array([np.eye(4)]*3)
    arm=b.prepare_arm(h,np.zeros((4,)*4),.125,J,U,role='baseline',source_identity='fake',input_identity='fake',ld=2)
    ref=k.build_reference(arm,h,.125,(),parallel=True)
    circuit=QuantumCircuit(9);circuit.p(-.125/40,8)
    row=k.compare(k.fixed.record(circuit),ref,raw_identity='fake',parallel=True)
    assert len(ref['blocks'])==25 and row['columns_evaluated']==512
    assert F(ref['kappa_bridge']['upper_exact'])==F(ref['U_bridge']['upper_exact'])==0
    assert F(row['upper_exact'])<F(1,10**15)
    assert sorted(i for block in ref['blocks'] for i in block['indices'])==list(range(256))
    circuit.p(.01,8)
    bad=k.compare(k.fixed.record(circuit),ref,raw_identity='fake',parallel=False)
    assert F(bad['lower_exact'])>F(1,10000)


@pytest.mark.parametrize('fault',['J','nonfinite','polar','basis','diagnostic'])
def test_input_FAIL_no_fallback(example,fault):
    arm,h,events,_=example;diagnostic=None
    if fault=='J':arm.J[0,0,1]+=.1
    if fault=='nonfinite':h=h.copy();h[0,0]=np.nan
    if fault=='polar':
        arm=b.prepare_arm(h,np.zeros((2,)*4),.19,arm.J,arm.U.copy()*3,role='baseline',source_identity='a',input_identity='b',ld=2)
    if fault=='basis':arm.U[0,0,0]+=.001
    if fault=='diagnostic':diagnostic={'matrices':{name:{'entries':[]} for name in ('kappa_literal','kappa_H_exact_diagnostic_only','kappa_A_exact_diagnostic_only')}}
    with pytest.raises(ValueError):k.build_reference(arm,h,.19,events,parallel=False,diagnostic_record=diagnostic)


def test_partial_input_proof_before_unqualified_polar(example):
    arm,h,events,_=example
    arm=b.prepare_arm(h,np.zeros((2,)*4),.19,arm.J,arm.U.copy()*3,role='baseline',source_identity='a',input_identity='b',ld=2)
    partial={}
    with pytest.raises(ValueError,match='POLAR'):
        k.build_reference(arm,h,.19,events,parallel=False,progress=lambda name,row:partial.update({name:row}))
    assert partial['K1_input_gate']['status']=='PASS' and 'K1_models' not in partial


def test_synthetic_E2E_seven_calls_and_FAIL_STOP(tmp_path,certificate):
    record,ref,row=certificate;prior=k.fixed.compare(record,record,parallel=False);e.io.save(tmp_path/'prior.json',prior)
    c=dict(circuits={'raw':'raw.qpy'},saved_inputs={'raw.qpy':'rawhash'},reuse=dict(compiled_raw='prior.json',raw_sha256='rawhash'))
    def reference(root,c,out,call):
        for kind in ('input_arrays','arm_replay','trajectory_load'):call(kind,'artificial',lambda:None)
        return call('CAR_reference','artificial',lambda:ref)
    call=Calls(tmp_path,e.LIMITS,7)
    result=e.execute(tmp_path,c,tmp_path,call,reference_loader=reference,loader=lambda *a,**kw:record,compare=lambda *a,**kw:row)
    assert result['status']=='COMPLETE' and sum(call.counts.values())==7
    fail=tmp_path/'fail';fail.mkdir();call=Calls(fail,e.LIMITS,7)
    with pytest.raises(ValueError,match='ALLOWANCE_FAIL'):
        e.execute(tmp_path,c,fail,call,reference_loader=reference,loader=lambda *a,**kw:record,
                  compare=lambda *a,**kw:dict(upper_exact='1',lower_exact='1/100'))
    assert call.stop.is_set() and sum(call.counts.values())==6
    assert (fail/'CAR_reference.json').exists() and (fail/'raw_CAR.json').exists()
    assert 'triangle_composition' not in call.counts


def test_passive_source_input_identity_no_decode(monkeypatch):
    if not CONTRACT.exists():pytest.skip('contract freeze follows initial artificial tests')
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('passive molecular decode'))
    assert e.preflight(ROOT,CONTRACT)['scientific_calls']==0
    real=e.io.sha;c=e.io.read(CONTRACT)
    for name in ('src/trottertracks/c_pr_aware_df_fixed_k1.py',c['input']):
        monkeypatch.setattr(e.io,'sha',lambda p,name=name:'0'*64 if Path(p)==ROOT/name else real(p))
        with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):e.preflight(ROOT,CONTRACT)
    monkeypatch.setattr(e.io,'sha',real)


def test_old_exact_gate_not_removed():
    h=np.array([[0.,.125],[.25,0.]])
    with pytest.raises(ValueError,match='EXACT_HERMITIAN'):k.car._hermitian_input(h)
    assert k.exact_kappa(h)['anti_spectral_upper']>0


def test_Arb_encloses_wide_exact_H_and_keeps_operation_radius():
    h=np.array([[0.,1.],[np.nextafter(0.,1.),0.]])
    parts=k.exact_kappa(h)
    with ctx.workprec(192):
        ball=k.ball_coefficients(parts['H'])[0][1].real
        assert ball.is_exact()  # Arb dyadic constructor keeps every input bit
        evaluated=ball.exp()
        assert not evaluated.is_exact() and evaluated.rad()>0
        assert k.z._fraction(ball.lower())<=parts['H'][0][1][0]<=k.z._fraction(ball.upper())


def test_both_bridges_cover_known_exact_polar_target(example):
    from trotterlib.rte import _make_event
    arm,h,_,_=example
    U=np.array([np.diag([1.001, .999])]*4)
    altered=b.prepare_arm(h,np.zeros((2,)*4),.19,arm.J,U,role='baseline',source_identity='a',input_identity='b',ld=2)
    tail,config,dist=b.configure(altered,F(1,40),16,0)
    events=tuple(_make_event((i%len(tail.components),),tail.components,dist,0) for i in range(16))
    literal=h.astype(complex)+np.diag([.01j,-.005j])
    V0=k.build_reference(altered,literal,.19,events,parallel=False)
    # Positive diagonal U has the exact polar Q=I, with no numerical polar solver.
    Q=np.array([np.eye(2)]*4)
    ideal=b.prepare_arm(h,np.zeros((2,)*4),.19,arm.J,Q,role='baseline',source_identity='a',input_identity='b',ld=2)
    V2=k.car.build_reference(ideal,h,.19,events,parallel=False)
    observed=np.linalg.norm(dense(V0)-dense(V2),2)
    bound=F(V0['kappa_bridge']['upper_exact'])+F(V0['U_bridge']['upper_exact'])
    assert observed<=float(bound)+1e-14
    assert np.linalg.norm(dense(V2).conj().T@dense(V2)-np.eye(16),2)<1e-12


def test_runner_partial_STOP_and_no_resume(tmp_path,monkeypatch):
    e.io.save(tmp_path/'SOURCE.json',{'files':{}})
    e.io.save(tmp_path/'contract.json',{'source_manifest':'SOURCE.json'})
    monkeypatch.setattr(e,'preflight',lambda *a:None)
    monkeypatch.setattr(e,'FRESH','run01')
    def fail(root,c,out,call):
        call('input_arrays','invented',lambda:None)
        e.io.save(out/'partial.json',{'input':'invented'})
        call('arm_replay','injected error',lambda:(_ for _ in ()).throw(ValueError('INJECTED_FAIL')))
    monkeypatch.setattr(e,'execute',fail)
    with pytest.raises(ValueError,match='INJECTED_FAIL'):e.run(tmp_path,tmp_path/'contract.json')
    stop=e.io.read(tmp_path/'run01/STOP.json')
    assert stop['scientific_units']==2 and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert (tmp_path/'run01/partial.json').exists() and (tmp_path/'run01/failure.json').exists()
    with pytest.raises(FileExistsError):e.run(tmp_path,tmp_path/'contract.json')
