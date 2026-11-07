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


def test_reversed_labels_preserve_controlled_ZZ_product_and_rotation():
    from types import SimpleNamespace
    from qiskit import QuantumCircuit
    from trotterlib.rte import RTEEventApplication
    from trottertracks.track_c_joint_isometry import append_run
    states=np.arange(512);z=(1-2*((states>>6)&1))*(1-2*((states>>7)&1))
    for role in ('product','rotation'):
        a=RTEEventApplication(0,role,'synthetic',1.,-1,False,None,'synthetic',(0,1),())
        q=QuantumCircuit(9);phase=append_run(q,[(SimpleNamespace(unsigned_rotation_angle=.17),a)],(),{0:7,1:6})
        if phase:q.p(phase,8)
        expected=np.where(states<256,1.,-z if role=='product' else np.exp(-.17j*z))
        assert np.linalg.norm(Operator(q).data-np.diag(expected))<2e-13


def test_reflected_adjacent_Givens_preserve_Fock_orbital_projectors():
    rng=np.random.default_rng(91);u,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    support=(7,1,5);ops=partial_ops(u[::-1],support)
    reflected=tuple((g,tuple(7-i for i in wires)) for g,wires in ops)
    v=Operator(circuit(reflected)).data;b=Operator(circuit(partial_ops(u,tuple(range(8))))).data
    states=np.arange(256)
    for j,i in enumerate(support):
        old=(b*(1-2*((states>>i)&1)))@b.conj().T
        new=(v*(1-2*((states>>(7-j))&1)))@v.conj().T
        assert np.linalg.norm(old-new,'fro')<2e-13
