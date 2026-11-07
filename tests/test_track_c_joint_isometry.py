import numpy as np
from qiskit.quantum_info import Operator
from trottertracks.track_c_bottleneck_basis import partial_ops,circuit


def test_joint_columns_preserve_each_Fock_operator_and_products():
    rng=np.random.default_rng(20261108)
    u,_=np.linalg.qr(rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)))
    full=Operator(circuit(partial_ops(u,tuple(range(5))),5)).data
    states=np.arange(32)
    for support in ((1,3),(0,2,4),(0,1,2,4)):
        part=Operator(circuit(partial_ops(u,support),5)).data
        old_ops=[];new_ops=[]
        for j,i in enumerate(support):
            old=(full*(1-2*((states>>i)&1)))@full.conj().T
            new=(part*(1-2*((states>>j)&1)))@part.conj().T
            assert np.linalg.norm(old-new)<4e-14
            old_ops.append(old);new_ops.append(new)
        assert np.linalg.norm(old_ops[0]@old_ops[-1]-new_ops[0]@new_ops[-1])<8e-14
        assert np.linalg.norm(part.conj().T@part-np.eye(32))<4e-14
