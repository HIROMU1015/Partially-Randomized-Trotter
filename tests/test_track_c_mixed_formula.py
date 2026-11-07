import numpy as np
from scipy.linalg import expm
from trottertracks.track_c_mixed_formula import coefficients,derive_bch,sequence,cubic_generator

def test_exact_bch_and_ground_minimum():
    result=derive_bch()
    assert result['completed_square']=='0'
    for a in [0.,1/6,.1931833275037836,.25,.5,1.]:
        x,y=coefficients(a)
        assert abs(x-y-(1/96+(a-.25)**2/2))<1e-15

def test_endpoint_and_refinement_are_exact_source_sequences():
    assert sequence('old_endpoint')==sequence('registered_S2')
    from trottertracks.track_c_mixed_formula import compact
    assert sequence('refinement_control')==compact(sequence('registered_S2',.5)+sequence('registered_S2',.5))

def test_cubic_log_includes_multiblock_internal_error():
    rng=np.random.default_rng(7);matrices=[]
    for _ in range(5):
        a=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3));matrices.append((a+a.conj().T)/8)
    h2=cubic_generator(sequence('mixed_2MN'),matrices)
    from scipy.linalg import logm
    errors=[]
    for h in [.02,.01]:
        u=np.eye(3,dtype=complex)
        for kind,j,t in sequence('mixed_2MN',h):u=expm(-1j*t*(matrices[-1] if kind=='R' else matrices[j]))@u
        effective=1j*logm(u)/h
        errors.append(np.linalg.norm((effective-sum(matrices))/h**2-h2))
    assert errors[1]<.4*errors[0]
