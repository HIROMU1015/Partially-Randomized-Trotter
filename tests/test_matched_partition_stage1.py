"""Artificial two/four-qubit operators only. No saved H4 input or circuit."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_h4_p3_stage1 as runner
K=runner.modules()
from stage1.h4_molecular_inputs import spin_lift_unthresholded
from stage1 import df_hamiltonian as DF,pauli_signal_adapter as PA
from openfermion.circuits import low_rank
from qiskit import QuantumCircuit


def toy(weight=.3,g=None):
    g=np.array([[1.]]) if g is None else np.asarray(g)
    h=np.eye(len(g))*.4
    if len(g)==2:h=np.array([[.4,.03],[.03,-.2]])
    # A hand-specified rank-one interaction, not a DF factorization call.
    eri=2*weight*np.einsum('pq,rs->pqrs',g,g)
    bare,two=spin_lift_unthresholded(h,eri);two=two/2
    d=-weight*np.kron(g@g,np.eye(2))
    spin_g=np.kron(g,np.eye(2))
    model=PA.PauliHamiltonian.from_spatial_integrals(h,eri,nuclear_repulsion=.13)
    return dict(spin_one=bare.astype(complex),spin_two=two.astype(complex),
        df_one_body=(bare+d).astype(complex),lambdas=np.array([weight]),
        g_matrices=np.array([spin_g],dtype=complex),constant=.13,
        pauli=dict(constant=model.constant,terms=model.terms),
        sector_indices=np.arange(1<<(len(bare))))


@pytest.mark.parametrize('weight,g',[(.3,None),(-.17,[[.7,.2],[.2,-.3]]),(.11,[[.4,-.1],[-.1,.8]])])
def test_independent_routes_match_on_signed_nondiagonal_artificial_inputs(weight,g):
    data=toy(weight,g)
    with patch.object(low_rank,'low_rank_two_body_decomposition',side_effect=AssertionError('FACTOR')), \
         patch.object(DF,'low_rank_two_body_decomposition',side_effect=AssertionError('FACTOR')), \
         patch.object(QuantumCircuit,'__init__',side_effect=AssertionError('CIRCUIT')):
        p,pi=K.pauli_route(data);d,di=K.df_route(data)
    for key in ('H','T','V','D'):np.testing.assert_allclose(p[key+'_P'],d[key+'_DF'],atol=2e-14,rtol=0)
    np.testing.assert_allclose(pi['correction_tensor'],di['correction_tensor'],atol=1e-15,rtol=0)
    assert di['native_operator_basis_column_actions']==len(d['H_DF'])
    assert di['physical_sector_projection'] is False


def test_one_spatial_orbital_manual_occupation_oracle_and_scalar():
    data=toy();p,_=K.pauli_route(data);d,di=K.df_route(data)
    np.testing.assert_allclose(np.diag(p['H_P']),[.13,.53,.53,1.53],atol=1e-15)
    np.testing.assert_allclose(np.diag(p['V_normal_P']),[0,0,0,.6],atol=1e-15)
    np.testing.assert_allclose(np.diag(p['V_P']),[0,.3,.3,1.2],atol=1e-15)
    assert abs(di['fragment_identity'][0]-.45)<1e-15
    assert abs(np.trace(d['V_DF']).real/4-.45)<1e-15


def test_native_vectorization_matches_original_action_and_endian_mapping():
    data=toy(.21,[[.7,.19],[.19,-.4]]);n=len(data['spin_one']);dim=1<<n
    space=DF.PhysicalSector(n,np.arange(dim));table=DF._build_excitation_table(space)
    fast=K._native_one_body(data['df_one_body'],table,dim).toarray()
    ham=DF.DFHamiltonian(0.,data['df_one_body'],np.array([]),tuple(),{})
    op,_=DF.df_linear_operator(ham,space,backend='python')
    np.testing.assert_allclose(fast,DF._linear_operator_to_dense(op),atol=1e-15,rtol=0)
    p,_=K.pauli_route(data);perm=K.reverse_bits(n)
    np.testing.assert_allclose(p['T_P'],fast[np.ix_(perm,perm)],atol=2e-15,rtol=0)
    assert np.max(np.abs(p['T_P']-fast))>.01


def test_removed_or_doubled_correction_fails_independent_T_V():
    data=toy();p,pi=K.pauli_route(data);d,di=K.df_route(data);dim=4
    for wrong in [data['spin_one'],2*data['df_one_body']-data['spin_one']]:
        test=dict(data,df_one_body=wrong);bad,_=K.df_route(test)
        assert K.compare(p['T_P'],bad['T_DF'],dim)['status']=='FAIL'
    assert K.compare(p['V_normal_P'],d['V_DF'],dim)['status']=='FAIL'


def test_half_factor_and_constant_mutations_are_detected():
    data=toy();p,_=K.pauli_route(data)
    doubled,_=K.pauli_route(dict(data,spin_two=data['spin_two']*2))
    assert K.compare(p['V_P'],doubled['V_P'],4)['status']=='FAIL'
    assert K.compare(p['H_P'],p['H_P']+.13*np.eye(4),4)['status']=='FAIL'


def test_hermiticity_and_sector_leakage_not_hidden_by_projection():
    data=toy();p,pi=K.pauli_route(data);d,di=K.df_route(data);matrices={**p,**d}
    data.update(sector_indices=np.array([1,2]),snapshot_H_P=p['H_P'],snapshot_H_DF=d['H_DF'])
    good=K.diagnose(data,matrices,pi,di,workers=2);assert good['overall']=='PASS'
    bad={k:v.copy() for k,v in matrices.items()};bad['T_DF'][0,1]+=.01
    result=K.diagnose(data,bad,pi,di,workers=2)
    assert result['comparisons']['T_DF_Hermiticity']['status']=='FAIL'
    assert result['comparisons']['T_DF_sector_out']['status']=='FAIL'


def test_predeclared_margin_classification_and_spectral_norm():
    tol=K.tolerance(256,1.)
    assert K.classify(tol/8,tol)=='PASS'
    assert K.classify(tol,tol)=='UNRESOLVED'
    assert K.classify(8*tol,tol)=='FAIL'
    a=np.array([[0,3j],[0,0]])
    assert abs(K.spectral(a)-3)<1e-15
    with pytest.raises(ValueError):K.classify(float('nan'),tol)
