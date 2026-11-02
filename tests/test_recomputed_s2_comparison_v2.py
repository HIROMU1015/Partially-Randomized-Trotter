"""Artificial integration tests; no molecular inputs or scientific run units."""
import importlib.util
from pathlib import Path
import sys
from fractions import Fraction
import math

path=Path(__file__).resolve().parents[1]/'scripts/recomputed_s2_runtime_v2.py'
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('recomputed_test_runtime_v2',path)
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)


def artificial():
    r.bootstrap()
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
    from trotterlib.pr2_s0_s1_validation import _to_qiskit_state
    from trottertracks.resource_applicability.h8_sparse_reference_v1 import OccupationCSR,SpinPrimitive
    from scipy.linalg import expm
    one=np.array([[.2,0,.1,0],[0,.2,0,.1],[.1,0,-.3,0],[0,.1,0,-.3]],complex)
    g=np.diag([.5,.5,-.2,-.2]).astype(complex)
    h=np.array([[.1,0,.2,0],[0,.1,0,.2],[.2,0,.3,0],[0,.2,0,.3]],complex)
    ham=DFHamiltonian(.17,one,np.array([.4,.2]),(g,h),{})
    sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
    state=np.array([1.,0,0,0],complex);full=np.zeros(16,complex);full[sector.basis_indices]=state
    topo=OccupationCSR(sector.basis_indices,4);blocks=tuple(topo.one(g) for g in ham.g_matrices);o=topo.one(one)
    matrix=o.toarray()+ham.constant*np.eye(4)+sum(w*(a@a).toarray() for w,a in zip(ham.lambdas,blocks))
    return dict(ham=ham,sector=sector,state=state,full=full,qstate=_to_qiskit_state(full,4),
        preps={p:_prepare(ham,p) for p in [1,2]},one=o,blocks=blocks,
        spin=[SpinPrimitive(one,sector)]+[SpinPrimitive(g,sector,weight=w) for g,w in zip(ham.g_matrices,ham.lambdas)],
        reference=complex(np.vdot(state,expm(-.8j*matrix)@state)),reference_error=1e-15)


def case(method='B2'):
    return dict(system='ARTIFICIAL',id=method+'_p1_q2_2nd',prefix=1,q=2,r=2,K=4,formula='2nd',method=method)


def test_scalar_once_and_B0I_phase():
    ctx=artificial();a=r.signal(ctx,case('B0'));b=r.signal(ctx,case('B0I'))
    z=lambda d:complex(d['corrected']['real'],d['corrected']['imag'])
    correction=r.scalar(ctx,case('B0I'))-r.scalar(ctx,case('B0'))
    assert abs(z(b)-z(a)*__import__('cmath').exp(-.8j*correction))<1e-12


def test_theory_actual_native_and_raw_closure():
    ctx=artificial();c=case();units,law=r.theory(ctx,c)
    assert all(Fraction(v['lower'])<=Fraction(v['upper']) for v in units.values())
    assert law['order_probabilities'][4]>0
    sig=r.signal(ctx,c)
    assert sig['Horner_forward_error']<1e-12 and sig['raw_corrected_closure']<1e-12
    shot=r.shots(sig,'0.05');assert shot['usable'] and shot['N_cos']>0
    r.CTX,r.CANDIDATE=ctx,c
    cost=r.compile_one(('ARTIFICIAL',0,True))
    assert cost['cosine']>0 and max(cost['validation_errors'])<1e-10


def test_paired_weighted_statistics_and_sample_plan():
    values=[dict(cosine=100+i%2,sine=101+i%2) for i in range(128)]
    shot=dict(usable=True,N_cos=1000,N_sin=7)
    s=r.stats(values,shot)
    assert Fraction(s['mean_exact'])==Fraction(1000*(100+101)+7*(101+102),2)
    assert s['variance']>0 and s['target_met']
    n,plan=r.plan_n(values,{'0.01':shot});assert n==128 and plan['0.01']['n']==128
    assert r.seed('x','MC_pilot',0)!=r.seed('x','MC_main',0)


def test_no_zero_filling_or_ineligible_shots():
    ctx=artificial();sig=r.signal(ctx,case())
    sig['signed_bias']={'real':1.,'imag':1.}
    result=r.shots(sig,'0.0005')
    assert not result['usable'] and result['N_cos'] is None and result['N_sin'] is None
