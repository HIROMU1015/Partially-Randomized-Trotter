"""Focused controls for the independent audit; no registered science input."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/tracks/algorithm_codesign'
sys.path.insert(0,str(SCRIPTS))
import audit_df_sparse_correction as a
import audit_df_sparse_correction_native as native

@pytest.mark.parametrize('count',[1,2,3,4])
def test_commuting_quotient_is_exact_through_degree_five(count):
    # Collapse free words to commuting monomials, testing the algebraic control.
    xs=[{(j,):F(j+1,7)} for j in range(count)]
    sweep=[a.exp_series(a.scaled(x,F(1,2))) for x in xs[:-1]]
    out={():F(1)}
    for factor in sweep+[a.exp_series(xs[-1])]+sweep[::-1]:out=a.times(out,factor)
    residual=a.plus(out,a.scaled(a.exp_series(a.plus(*xs)),-1))
    collapsed={}
    for word,value in residual.items():
        key=tuple(sorted(word));collapsed[key]=collapsed.get(key,F(0))+value
    assert all(v==0 for v in collapsed.values())

def test_pure_d_internal_cubic_is_nonzero():
    x={(0,):F(1)};y={(1,):F(1)}
    log=a.log_series(a.times(a.times(a.exp_series(a.scaled(x,F(1,2))),a.exp_series(y)),a.exp_series(a.scaled(x,F(1,2)))))
    assert a.grade(log,3)

@pytest.mark.parametrize('delta',[F(1,8),F(1,1000),F(1,10**80)])
def test_small_eta_stable_angle_identity(delta):
    gamma=F(5,4);b=F(9,8);eta=gamma*delta
    assert b*delta**3/(2*eta)==b*delta**2/(2*gamma)
    assert eta*(b*delta**3/(2*eta))==b*delta**3/2

def test_hierarchical_weights_match_exact_tuple_probability():
    blocks=[[F(1,4),F(1,2)],[F(3,8)],[F(1,8),F(1,16)]]
    total=F(0); tuples=[]
    for j in range(len(blocks)-1):
        tail=sum(blocks[j+1:],[]);lj=sum(blocks[j]);ell=sum(tail)
        for aa,bb,cc,scale in [(blocks[j],blocks[j],tail,24),(tail,blocks[j],tail,12)]:
            mass=sum(aa)*sum(bb)*sum(cc)/scale;total+=mass
            for ha in aa:
                for hb in bb:
                    for hc in cc:
                        c=ha*hb*hc/scale
                        tuples.append((c,mass,ha/sum(aa)*hb/sum(bb)*hc/sum(cc)))
    assert sum(c/total for c,_,_ in tuples)==1
    for c,mass,conditional in tuples:assert mass/total*conditional==c/total
    # Identity rejection is a real probability event, never postselection.
    eta=F(1,8);kappas=[F(0),F(1),F(4)]
    ps=[eta*c*kappas[i%3]/(4*total) for i,(c,_,_) in enumerate(tuples)]
    assert 0<=sum(ps)<=eta
    assert (1-sum(ps))+sum(ps)==1

def test_constant_identity_commutators_vanish():
    import numpy as np
    x=np.array([[0,1],[1,0]]);z=np.diag([1,-1]);eye=np.eye(2)
    assert np.array_equal(a.comm(eye,a.comm(x,z)),np.zeros((2,2)))
    assert np.array_equal(a.comm(x,a.comm(z,eye)),np.zeros((2,2)))

def test_relative_phase_is_not_discarded():
    import numpy as np
    from scipy.linalg import block_diag,norm
    u=np.diag([np.exp(.03j),np.exp(-.03j)])
    # Projectively identical system gates differ as controlled operators.
    assert norm(block_diag(np.eye(2),u)-block_diag(np.eye(2),-u),2)>1.9

def test_nonGaussian_basis_does_not_imply_orbital_access():
    from qiskit.circuit.library import RYGate
    with pytest.raises(ValueError,match='Number-nonconserving'):
        native.one_particle_from_ops(1,[(RYGate(.14),(0,))])

def test_number_preserving_nonGaussian_gate_is_rejected():
    from qiskit.circuit.library import CZGate
    with pytest.raises(ValueError,match='non-Gaussian'):
        native.one_particle_from_ops(2,[(CZGate(),(0,1))])

def test_biased_probability_error_uses_distance_from_identity():
    import numpy as np
    from scipy.linalg import expm,norm
    u=expm(1j*.02*np.diag([1,-1]));p=.1;ptilde=.101
    actual=norm((ptilde-p)*(u-np.eye(2)),2)
    assert actual<=abs(ptilde-p)*.02+1e-15

def test_independence_required_for_first_moment_product():
    import numpy as np
    u=np.diag([1j,-1j]);eye=np.eye(2);mean=(eye+u)/2
    correlated=(eye+u@u)/2
    assert not np.allclose(correlated,mean@mean)
