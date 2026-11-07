import numpy as np
from qiskit.quantum_info import Operator
from trottertracks.track_c_bottleneck_basis import partial_ops,circuit

def test_partial_gaussian_columns_and_fock_involution():
    rng=np.random.default_rng(17);u,_=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    for support in ((2,),(1,3)):
        ops=partial_ops(u,support);b=Operator(circuit(ops,4)).data
        one=b[np.ix_([1<<i for i in range(4)],[1<<i for i in range(4)])]/b[0,0]
        assert np.max(abs(one[:,:len(support)]-u[:,support]))<2e-14
        assert sum(g.name=='xx_plus_yy' for g,_ in ops)<=len(support)*4-len(support)*(len(support)+1)//2
        assert np.linalg.norm(b.conj().T@b-np.eye(16))<2e-14
