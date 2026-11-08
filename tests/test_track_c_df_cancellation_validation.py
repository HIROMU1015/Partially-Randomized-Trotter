"""Artificial timing/phase checks; molecular conclusions use directed runs."""
from fractions import Fraction as F
import numpy as np
from trottertracks import track_c_df_cancellation_validation as v


def test_fixed_total_scalar_transfer_covers_both_control_branches():
    n=7;h=.17;t=1.190000000001;c=.731
    q=dict(closed=True,rho_lower=.99,beta_stat_lower=.3,total_truncation_PF_phase_upper=.05)
    result=v.scalar_time_transfer(q,n,h,t,c)
    actual=np.diag([1.,np.exp(-1j*t*c)])
    reference=np.diag([1.,np.exp(-1j*n*h*c)])
    assert np.linalg.norm(actual-reference,2)<=result['fixed_total_scalar_time_error_upper']+1e-15
    converted=v.physical_signal([reference[1,1].real,reference[1,1].imag],n,h,t,c)
    assert abs(converted-actual[1,1])<1e-15
    assert result['closed'] and result['beta_stat_lower']<q['beta_stat_lower']


def test_authorized_h_neighborhood_preserves_the_integer_step_count():
    t=F(1,50)*(1<<17)
    for n in (32752,32767,32768,32769,32784):
        h=float(t/n)
        assert round(float(t)/h)==n
        # Exact conversion discrepancy is retained, not replaced by n*h=t.
        discrepancy=F(n)*F.from_float(h)-F.from_float(float(t))
        assert abs(discrepancy)<=n*F.from_float(np.spacing(h))


def test_pair_disk_encloses_independent_complex_signal_perturbations():
    from trottertracks.track_c_df_cancellation_conservative import pair_phase
    zp=.87*np.exp(.13j);zk=.93*np.exp(.27j);ep=.002;ek=.003
    result=pair_phase(zp,ep,zk,ek);centre=complex(*result['centre'])
    for a in np.linspace(0,2*np.pi,13):
        for b in np.linspace(0,2*np.pi,11):
            actual=(zp+ep*np.exp(1j*a))*np.conj(zk+ek*np.exp(1j*b))
            assert abs(actual-centre)<=result['radius']+1e-15
            assert abs(np.angle(actual))<=result['phase_upper']


def test_separate_upper_bounds_do_not_inherit_signed_cancellation():
    from trottertracks.track_c_df_cancellation_conservative import pair_phase
    keep=np.exp(.28j);pf=np.exp(-.034j)
    a=v.s.phase_upper(keep,1e-7);b=pair_phase(pf,1e-7,keep,1e-7)['phase_upper']
    assert abs(np.angle(pf))<.4 and v.s.add(a,b)>.4
    assert pair_phase(.01+0j,.02,1+0j,.001)['phase_upper'] is None


def test_rational_h_source_disk_covers_a_known_diagonal_generator():
    q=dict(closed=True,rho_lower=.99,beta_stat_lower=.3,total_truncation_PF_phase_upper=.05)
    c=dict(blocks=[dict(coeff={(0,):F(3,10)})]);row=dict(h=.17,candidate=dict(drop=[]))
    result=v.h_rounding_supplement(q,c,row,7,6)
    eigenvalues=np.array([.3,-.3]);logical=np.diag(np.exp(-1j*1.28*eigenvalues));actual=np.diag(np.exp(-1j*7*.17*eigenvalues))
    assert np.linalg.norm(logical-actual,2)<=result['logical_h_rounding_source_disk']+1e-15
    assert result['closed'] and result['beta_stat_lower']<q['beta_stat_lower']
