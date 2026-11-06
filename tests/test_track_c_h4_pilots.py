"""Implementation checks on artificial controls, separate from H4 performance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from trottertracks.track_c_h4 import weights,residual_parameters,shot_rows
from trottertracks.track_c_h4_price import conditional_flags
from trottertracks.track_c_h4_cs_price import residual_realization
from trotterlib.rte import finite_rte_distribution


def test_published_time_and_negative_stages():
    for f in ['S2','Y4','Y8']:assert abs(sum(weights(f))-1)<1e-14
    assert any(x<0 for x in weights('Y8'))


def test_conditional_rare_includes_multiple_orders():
    rng=np.random.default_rng(17)
    assert not conditional_flags([.2,.3],False,rng).any()
    assert conditional_flags([0.,.2,0.],True,rng).tolist()==[False,True,False]
    samples=[conditional_flags([.8,.9],True,rng) for _ in range(500)]
    assert all(x.any() for x in samples)
    assert any(x.all() for x in samples)


def test_raw_corrected_normalizer_once():
    raw=[dict(feasible=True,radius_lower=.4,beta_stat=.3,range=1.) for _ in range(18)]
    corrected=[dict(feasible=True,radius_lower=.8,beta_stat=.3,range=2.) for _ in range(18)]
    assert shot_rows(raw)[0]['shots']==shot_rows(corrected)[0]['shots']


def test_large_finite_range_uses_log_domain():
    rows=[dict(feasible=True,radius_lower=.8,beta_stat=.3,range=math.exp(390),log_range=390.) for _ in range(18)]
    result=shot_rows(rows)[0]
    assert all(n>10**330 for n in result['shots'])
    assert all(math.isfinite(x) for x in result['shots_log10'])


def test_single_residual_coherent_full_joint_mean():
    theta=.72;qc=QuantumCircuit(9,1)
    qc.append(Gate('trackc_res',2,[theta]),[8,0])
    zero,p0,_=residual_realization(qc,1,False,np.random.default_rng(1))
    two,p2,_=residual_realization(qc,1,True,np.random.default_rng(2))
    delta=theta-round(theta/(math.pi/4))*math.pi/4
    B=finite_rte_distribution(delta/2,2).exact_finite_distribution
    inside=np.array([256,257])
    actual=B*(p0*Operator(zero.remove_final_measurements(inplace=False)).data[np.ix_(inside,inside)]
             +p2*Operator(two.remove_final_measurements(inplace=False)).data[np.ix_(inside,inside)])
    z,_,_=residual_parameters(theta,1)
    assert np.linalg.norm(actual-np.diag([z,z.conjugate()]),2)<1e-12
    assert abs(p0+p2-1)<1e-14


def test_processor_measured_equivalence_both_axes():
    x=np.array([[0,1],[1,0]],complex);z=np.diag([1.,-1.]);y=np.array([[0,-1j],[1j,0]])
    p=expm(-.23j*x);u=expm(-.31j*z)
    psi=np.array([.6,.8j]);plus=np.array([1,1])/math.sqrt(2)
    cu=np.block([[np.eye(2),np.zeros((2,2))],[np.zeros((2,2)),u]])
    ref=np.kron(np.eye(2),p.conj().T)@cu@np.kron(np.eye(2),p)@np.kron(plus,psi)
    measured=cu@np.kron(np.eye(2),p)@np.kron(plus,psi)
    for a in (x,y):
        observable=np.kron(a,np.eye(2))
        assert abs(np.vdot(ref,observable@ref)-np.vdot(measured,observable@measured))<1e-13


def test_finite_cdf_retains_rare_support_and_rounding_bound():
    from fractions import Fraction
    from trottertracks.track_c_h4_finite_bits import integer_cdf,proposal_budget
    p=[.1,.9,1e-30];cdf=integer_cdf(p,160)
    actual=[Fraction(b-a,1<<160) for a,b in zip([0]+list(cdf[:-1]),cdf)]
    nominal=[Fraction.from_float(x) for x in p];total=sum(nominal)
    assert all(x>0 for x in actual)
    assert sum(abs(a-b/total) for a,b in zip(actual,nominal))<=Fraction(2*len(p),1<<160)
    budget=proposal_budget(390.,10**9,300)
    assert budget['signal_error_upper']<=math.ulp(1.)
    assert budget['bits_per_uniform_draw']>600


def test_exact_ring_guard_matches_independent_interval_replay():
    import mpmath as mp
    from trottertracks.track_c_h4_synthesis import exact_word_matrix
    from trottertracks.algorithm_codesign.synthesis_placement.economics import sequence_matrix
    mp.mp.dps=100;mp.iv.dps=100
    for word in ['', 'WSS', 'HTtSXWTHT', 'HSTHTHTtW']:
        a=exact_word_matrix(word);b=sequence_matrix(word)
        for i in range(2):
            for j in range(2):
                assert abs(complex(a[i][j].real.mid,a[i][j].imag.mid)-complex(b[i][j].real.mid,b[i][j].imag.mid))<1e-13
