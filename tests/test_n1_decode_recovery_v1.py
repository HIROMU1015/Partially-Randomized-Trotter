"""Actual-pack adapter regression and new N1-only synthetic checks."""
import ast
from fractions import Fraction as F
import gzip
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from trottertracks.representation_exploration import n1_decode_recovery_v1 as n
from trottertracks.representation_exploration.algorithm_design import rotation
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_n1_checker',ROOT/'scripts/verify_n1_decode_recovery_v1.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_actual_saved_pack_full_preflight_without_science(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('metadata regression consumed science')
    for name in ('prepare_factors','pf_certificate','truth_n1','run_n1'):monkeypatch.setattr(n,name,forbidden)
    monkeypatch.setattr(n.Compiler,'compile',forbidden)
    p=ROOT/'artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json';before=p.read_bytes();report=n.preflight(p);assert report['factor_candidates']==44 and report['pair_combinations']==178;assert report['binary64_ratio_checks']>5000;assert p.read_bytes()==before

@pytest.mark.parametrize('value',[
 {'real':[1.,2.,3.,4.],'imag':[0.,0.,0.,.01]},
 {'real':[1.,2.,3.],'imag':[0.,0.,0.]},
 [True,2.,3.,4.],[float('nan'),2.,3.,4.],[float('inf'),2.,3.,4.],
 {'real':[1.,2.,3.,4.],'imag':[0.,0.,0.,0.],'oracle':1},
 [[1.],2.,3.,4.],{'real':[1.,2.,3.,4.],'imag':[0.,0.,0.]}])
def test_bad_vector_refused(value):
    with pytest.raises(ValueError):n.decode_vector(value)

def test_matrix_shape_finite_and_complex_preservation():
    a=n.mj(np.array([[1.,.13+.07j],[.13-.07j,-.2]]));got=n.decode_matrix(a,2);assert got[0,1].imag==.07 and got[1,0].imag==-.07
    a['real'][0].pop()
    with pytest.raises(ValueError):n.decode_matrix(a,2)

def test_numerical_kernels_ast_identical():
    original=ast.parse((ROOT/'src/trottertracks/representation_exploration/n13_redesign.py').read_text());new=ast.parse((ROOT/'src/trottertracks/representation_exploration/n1_decode_recovery_v1.py').read_text());before={x.name:x for x in original.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))};after={x.name:x for x in new.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    copied=[x.name for x in original.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.lineno<552 and x.name!='load_contexts']
    assert all(ast.dump(before[k],include_attributes=False)==ast.dump(after[k],include_attributes=False) for k in copied)
    assert not {'run_n3','run_n2','n3_inputs','direction_certificate'}&set(after)

def test_oracle_context_and_candidate_refused():
    c=json.loads((ROOT/'artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json').read_bytes())['contexts'][0];c['true_pf_error']=0
    with pytest.raises(ValueError,match='oracle'):n.validate_context(c)
    del c['true_pf_error'];c['generator_candidates'][0][0]['truth_gap']=0
    with pytest.raises(ValueError,match='oracle'):n.validate_context(c)

def test_gaussian_interval_orientation_and_high_precision_enclosure():
    u=rotation(3,0,2,.19)@rotation(3,1,2,-.27);qc=n.gaussian_circuit(u.astype(complex));q=n.gaussian_onebody(qc);numeric=Operator(qc).data;one=numeric[np.ix_([1,2,4],[1,2,4])]/numeric[0,0];assert np.linalg.norm(n.ivmid(q)-one)<2e-15 and np.linalg.norm(n.ivmid(q)-u)<2e-15
    import mpmath as mp
    with mp.workprec(256):
        x=F(.19);y=mp.sin(mp.mpf(x.numerator)/x.denominator);lo,hi=n.endpoints(mp.iv.sin(n.exact_iv(x)));assert mp.mpf(lo.numerator)/lo.denominator<=y<=mp.mpf(hi.numerator)/hi.denominator

def test_interval_pauli_square_against_explicit_jw():
    g=np.array([[.11,.09+.04j],[.09-.04j,-.21]]);terms=n.squared_interval(n.ivmatrix(g),-.31);target=-.31*n.second_quantize_one_body(g)@n.second_quantize_one_body(g)
    from qiskit.quantum_info import Pauli
    built=sum(float(sum(n.endpoints(v.real))/2)*Pauli(k).to_matrix() for k,v in terms.items());assert np.linalg.norm(built-target)<2e-16

def test_exact_pf_commutators_and_independent_checker():
    certs=[{'X':[n.D,n.D]},{'Z':[n.D,n.D]}];assert n.fs(n.pf_certificate(certs)['alpha2'])==audit.independently_pf(certs)==F(1,2);assert n.fs(n.pf_certificate([{'Z':[n.D,n.D]}]*2)['alpha2'])==0

def test_scalar_control_compiler_and_vectorized_reference(tmp_path):
    blocks=[(QuantumCircuit(4),{'ZIII':F(1,9),'IIII':F(1,13)})];qc,err=n.build_product(blocks,.07,2,'Y');target=n.wrapper_reference(n.ideal_operator(blocks,.07,2),'Y');compiler=n.Compiler(tmp_path);cost=compiler.compile(qc,'new_synthetic',target);ir=json.loads(gzip.decompress((tmp_path/cost['ir_file']).read_bytes()));assert err>0;assert np.linalg.norm(n.native_action(ir,np.eye(32,dtype=complex))-audit.native(ir))<3e-9;assert np.linalg.norm(audit.native(ir)-target)<3e-9;assert compiler.compile(qc,'same',target)==cost and compiler.calls==1 and compiler.hits==1

def test_safe_integer_shots():
    assert n.safe_shots(F(0))>=math.ceil(2*math.log(80)/(.04/math.sqrt(2))**2);assert n.safe_shots(F(1)) is None

def test_new_complex_vector_synthetic_n1_end_to_end(tmp_path,monkeypatch):
    monkeypatch.setattr(n.resource,'setrlimit',lambda *a:None);identity=n.mj(np.eye(4));factors=[];inputs=[]
    for j,e in enumerate([[.12,.17,-.02,.03],[.06,-.03,.01,.04]]):
        g=n.mj(np.diag(e));inputs.append(g);factors.append([{'id':f'new_synthetic/factor{j}/exact','family':'exact','eta':sorted(e),'approx_eta':e,'frame':identity,'ungauged_frame':identity,'ungauged_eta':{'real':e,'imag':[0.,0.,0.,0.]},'g':g,'groups':[[i] for i in range(4)]}])
    c={'context':'new_synthetic_reference','modes':4,'particles':2,'df_rank':2,'weights':[.3,-.1],'one_body':n.mj(np.diag([.02,-.01,.04,.05])),'scalar':.09,'input_factors':inputs,'time':.6,'model_budget':.02,'q_grid':[1,2,4,8],'generator_candidates':factors};decoded=n.validate_context(c);result=n.run_n1((decoded,str(tmp_path/'new_synthetic')));saved=json.loads((tmp_path/'new_synthetic/result.json').read_bytes());assert result['status']=='LIMITED_N1_COMPLETE';assert len(saved['rows'])==8 and len(saved['whole_pauli'])==12;assert all(x['pf_error']<2e-15 for x in saved['diagnostics']['rows'])
