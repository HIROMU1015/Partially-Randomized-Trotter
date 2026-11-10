"""Small synthetic checks; no molecular input, QC build, or transpilation."""
import math
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import h4_df10_price_v1 as m


class RecordCounter(m.NativeCounter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ops=[]
    def feed(self,op):
        self.ops.append(op)
        return super().feed(op)


def operator_of(ops,n=2):
    result=np.eye(1<<n,dtype=complex)
    sx=np.array([[1+1j,1-1j],[1-1j,1+1j]])/2
    for op in ops:
        if op.name=='cx':
            matrix=np.zeros_like(result)
            for i in range(1<<n):
                j=i^(1<<op.qubits[1]) if i&(1<<op.qubits[0]) else i
                matrix[j,i]=1
        else:
            if op.name=='rz':
                t=op.params[0];local=np.diag([np.exp(-.5j*t),np.exp(.5j*t)])
            elif op.name=='sx':local=sx
            elif op.name=='x':local=np.array([[0,1],[1,0]])
            else:raise AssertionError(op.name)
            matrix=np.array([[1.]],complex)
            for q in reversed(range(n)):
                matrix=np.kron(matrix,local if q==op.qubits[0] else np.eye(2))
        result=matrix@result
    return result


class PriceV1Tests(unittest.TestCase):
    def test_reused_gaussian_descriptor_needs_no_quantum_circuit(self):
        import qiskit
        spec=importlib.util.spec_from_file_location('v1_runner',Path(__file__).resolve().parents[1]/'scripts/run_h4_df10_price_v1.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        scope=dict(np=np,math=math)
        runner.reuse_functions(runner.CARRIER,'scripts/h46_new2_native.py',['gaussian_operations'],scope)
        t=.23;u=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]],complex)
        with patch.object(qiskit.QuantumCircuit,'__init__',side_effect=AssertionError('no QC')):
            operations=scope['gaussian_operations'](u)
        self.assertTrue(operations)
        self.assertTrue(all(g.name in ('p','xx_plus_yy') for g,q in operations))

    def test_local_expansions_preserve_phase_and_action(self):
        from qiskit.circuit.library import HGate,RYGate,XXPlusYYGate
        for name,wires,params,expected in [
            ('h',(0,),(),np.kron(np.eye(2),HGate().to_matrix())),
            ('ry',(0,),(.37,),np.kron(np.eye(2),RYGate(.37).to_matrix())),
            ('xx_plus_yy',(0,1),(.47,-.83),XXPlusYYGate(.47,-.83).to_matrix()),
        ]:
            with patch.object(m,'NativeCounter',RecordCounter):
                a=m.Accounting();a.gate(name,wires,params);a.finish()
            np.testing.assert_allclose(np.exp(1j*a.global_phase)*operator_of(a.native.ops),expected,atol=2e-15,rtol=0)

    def test_only_literal_rz_folding_and_zero_removal(self):
        a=m.Accounting();a.gate('rz',(0,),(.25,));a.gate('rz',(1,),(.5,));a.gate('rz',(0,),(-.25,))
        out=a.finish()
        self.assertEqual(out['expanded_rz_after_basis_cancellation'],3)
        self.assertEqual(out['structural_rz'],1)
        self.assertIsNone(out['compiler_correction'])

    def test_inverse_basis_pairs_only_when_adjacent(self):
        ops=[dict(name='p',qubits=[0],parameters=['.25'])]
        a=m.Accounting();a.basis(ops,False,'basis');a.basis(ops,True,'basis')
        self.assertEqual(a.finish()['features']['adjacent_inverse_basis_pairs_cancelled'],1)
        a=m.Accounting();a.basis(ops,False,'basis');a.gate('sx',(0,));a.basis(ops,True,'basis')
        self.assertEqual(a.finish()['structural_rz'],2)

    def test_tail_shared_basis_and_identity_boundary(self):
        basis=[dict(name='p',qubits=[0],parameters=['.25'])]
        def event(identity=False):
            app=dict(application_index=0,is_identity=identity,role='rotation',coefficient_sign=1,
                     basis_id='synthetic',basis_hash='synthetic',basis_change_operations=basis,diagonal_pauli_support=[0])
            return dict(phase=dict(real=1.,imag=0.),rotation_angle=.2,taylor_order=0,application_sequence=[app])
        a=m.Accounting();a.tail([event(),event()]);out=a.finish()
        self.assertEqual(out['features']['tail_basis_groups'],1)
        self.assertEqual(out['features']['tail_shared_basis_pairs'],1)
        a=m.Accounting();a.tail([event(),event(True),event()]);out=a.finish()
        self.assertEqual(out['features']['tail_basis_groups'],2)
        self.assertEqual(out['features']['identity_applications'],1)

    def test_fixed_scope_and_no_price_input(self):
        with self.assertRaises(ValueError):m.predict([],64,0,6.4,'cosine')
        with self.assertRaises(ValueError):m.predict([],8,0,.8,'Z')
        with self.assertRaises(TypeError):m.predict([],8,0,.8,'cosine',compiled_price=123)
        a=m.Accounting();a.gate('p',(8,),(0.,),'scalar_once');out=a.finish()
        self.assertEqual(out['structural_rz'],0)
        self.assertEqual(out['features']['literal_zero_rz'],1)


if __name__=='__main__':unittest.main()
