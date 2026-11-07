import numpy as np
from trottertracks.track_c_d_internal import schedule,s2
from trottertracks.track_c_mixed_formula import cubic_generator,comm,coefficients

def test_fragment_times_and_time_reversal():
    cases=[dict(kind='ordered'),dict(kind='D2'),dict(kind='D4'),dict(kind='Dtuned',v=1.49),dict(kind='redistributed',fragment=0,a=.2518)]
    for case in cases:
        seq=schedule(case)
        assert seq==list(reversed(seq))
        assert all(abs(sum(t for k,i,t in seq if k=='D' and i==j)-1)<1e-14 for j in range(4))
        assert abs(sum(t for k,_,t in seq if k=='R')-1)<1e-14

def test_joint_D_composition_BCH_identity():
    rng=np.random.default_rng(17);matrices=[]
    for _ in range(5):
        a=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3));matrices.append((a+a.conj().T)/8)
    d=sum(matrices[:4]);r=matrices[-1];alpha,beta=coefficients(.5)
    outer=-alpha*comm(d,comm(d,r))-beta*comm(r,comm(d,r))
    kd=cubic_generator(s2(range(4),tail=False),matrices)
    for v in (0.,.5,1.3512071919596578,1.49):
        full=cubic_generator(schedule(dict(kind='Dtuned',v=v)),matrices)
        predicted=outer+(2*v**3+(1-2*v)**3)*kd/4
        assert np.linalg.norm(full-predicted)<2e-13

def test_modular_D_partition_is_literal_fused_source():
    from trottertracks.track_c_d_resources import split
    from trottertracks.track_c_mixed_formula import compact
    for case in [dict(kind='ordered'),dict(kind='Dtuned',v=1.49),dict(kind='full_Y4',order=list(range(13)))]:
        seq=schedule(case);g=split(case,1.)
        for n in (1,2,8):
            if g['full']:actual=g['prefix']+g['middle']*(n-1)+g['suffix']
            else:
                assert sum(k=='R' for k,_,_ in seq)==1
                actual=g['first']
                for step in range(n):actual=actual+[('R',-1,1.)]+(g['last'] if step==n-1 else g['junction'])
            assert actual==compact(seq*n)
