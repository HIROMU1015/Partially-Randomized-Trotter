"""Invented 1--3-system-qubit tests; exhaustive vectors, no RNG or science data."""
import ast
from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import unittest

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator, Statevector, partial_trace

from trottertracks.algorithm_codesign import vacuum_reference_cswap as v
from trottertracks.pr_dictionary_validation.gates import matrix_gate

EVIDENCE=[]
TOL=8e-12  # Small binary64 reference tolerance, not a certified science bound.


def fixture(n=2, phase=0.):
    L,R=QuantumCircuit(n),QuantumCircuit(n)
    L.rz(.31,0);R.rz(-.47,n-1)
    L.global_phase=.23+phase;R.global_phase=-.19
    if n>1:
        L.xx_plus_yy(.61,.17,0,1);L.rzz(-.29,0,1)
        R.xx_plus_yy(-.43,.39,0,1);R.rz(.37,0)
    if n==3:
        L.xx_plus_yy(-.23,.11,1,2);R.rzz(.27,1,2)
    # Complex, normalized, includes vacuum/other sectors; not an eigenstate.
    psi=np.array([complex(j+1,(-1)**j) for j in range(2**n)])
    psi/=np.linalg.norm(psi)
    return L,R,psi


def initial(psi,width,target):
    out=np.zeros(2**width,complex)
    for i,c in enumerate(psi):
        at=sum(((i>>j)&1)<<q for j,q in enumerate(target));out[at]=c
    return out


def bit_probability(vector,q):
    return float(sum(abs(c)**2 for i,c in enumerate(vector) if not ((i>>q)&1)))


def known_se(L,R,axis,*,initialize_reference=False):
    """Independent Rowe-style two-CSWAP construction, same phase input."""
    n=L.num_qubits;a=2*n;c=QuantumCircuit(2*n+1)
    if initialize_reference:c.reset(list(range(n,2*n)))
    c.h(a)
    for j in range(n):c.cswap(a,j,n+j)
    c.compose(L.inverse(),qubits=list(range(n)),inplace=True)
    c.compose(R,qubits=list(range(n,2*n)),inplace=True)
    for j in range(n):c.cswap(a,j,n+j)
    c.p(float(v.vacuum_phase(L)+v.vacuum_phase(R)),a)
    if axis=='Y':c.sdg(a)
    if axis is not None:c.h(a)
    if initialize_reference:
        b=ClassicalRegister(1,'vacuum_signal');c.add_register(b);c.measure(a,b[0])
    return c


class VacuumTests(unittest.TestCase):
    def test_full_input_isometry_phase_reference_and_wire_permutations(self):
        maximum=0.
        for n,cat in product((1,2,3),(False,True)):
            L,R,_=fixture(n);lm,rm=Operator(L).data,Operator(R).data
            width=2*n+1+(n-1 if cat else 0)
            for mapping in (tuple(range(width)),tuple(reversed(range(width)))):
                plan=v.build_instrument(L,R,mapping=mapping,fanout=cat)
                for j in range(2**n):
                    psi=np.eye(2**n,dtype=complex)[:,j]
                    actual=Statevector(initial(psi,width,plan.target)).evolve(plan.circuit).data
                    expected=initial(lm.conj().T@psi,width,plan.target)
                    second=initial(rm@psi,width,plan.target)
                    expected=(expected+np.roll(second,1<<plan.ancilla))*np.exp(1j*float(v.vacuum_phase(R)))/math.sqrt(2)
                    error=matrix_gate(actual,expected,TOL,label='ALL_INPUT_ISOMETRY_NO_PHASE_FIT')
                    maximum=max(maximum,error['max_entry'])
                    leakage=sum(abs(c)**2 for i,c in enumerate(actual)
                                if any((i>>q)&1 for q in (*plan.reference,*plan.fanout)))
                    self.assertLess(leakage,TOL)
        EVIDENCE.append(dict(case='all_input_isometry',max_entry_error=maximum,
                             reference_and_fanout_return='VACUUM',phase_fitting=False))

    def test_noncommuting_LR_arbitrary_state_and_binary_distributions(self):
        for n in (2,3):
            L,R,psi=fixture(n)
            lm,rm=Operator(L).data,Operator(R).data;u=lm@rm
            self.assertGreater(np.linalg.norm(lm@rm-rm@lm),.1)
            self.assertGreater(np.linalg.norm(u@psi-np.vdot(psi,u@psi)*psi),.1)
            z=np.vdot(psi,u@psi)
            for axis in ('X','Y'):
                plan=v.build_instrument(L,R,axis=axis)
                vec=Statevector(initial(psi,plan.circuit.num_qubits,plan.target)).evolve(plan.circuit).data
                p=bit_probability(vec,plan.ancilla)
                c=QuantumCircuit(n+1);c.h(n);c.append(R.compose(L).to_gate().control(),[n,*range(n)])
                if axis=='Y':c.sdg(n)
                c.h(n)
                ref=Statevector(initial(psi,n+1,range(n))).evolve(c).data
                self.assertAlmostEqual(p,bit_probability(ref,n),delta=TOL)
                self.assertAlmostEqual(2*p-1,z.real if axis=='X' else z.imag,delta=TOL)
                EVIDENCE.append(dict(case=f'noncommuting_{n}q_{axis}',probability_zero=p,
                    mean=2*p-1,reference_mean=z.real if axis=='X' else z.imag,
                    L_vacuum_phase=str(v.vacuum_phase(L)),R_vacuum_phase=str(v.vacuum_phase(R))))

    def test_actual_phase_ledger_against_independent_vacuum_columns(self):
        for n,extra in product((1,2,3),(-math.pi,0.,.43)):
            L,R,_=fixture(n,extra)
            for c in (L,R,L.inverse(),R.inverse()):
                column=Operator(c).data[:,0]
                expected=np.zeros(2**n,complex);expected[0]=np.exp(1j*float(v.vacuum_phase(c)))
                matrix_gate(column,expected,TOL,label='VACUUM_PHASE_COLUMN')
        negative=QuantumCircuit(1);negative.rz(.4,0)
        self.assertLess(v.vacuum_phase(negative),0)

    def test_finite_actual_law_multiple_paths_preserves_complex_average(self):
        Q=(F(1,8),F(3,8),F(1,2));values=[];expected=0j;actual=0j
        for q,phase in zip(Q,(-math.pi,0.,.61)):
            L,R,psi=fixture(2,phase);z=np.vdot(psi,Operator(L).data@Operator(R).data@psi)
            coords=[]
            for axis in ('X','Y'):
                plan=v.build_instrument(L,R,axis=axis)
                out=Statevector(initial(psi,5,plan.target)).evolve(plan.circuit).data
                coords.append(2*bit_probability(out,plan.ancilla)-1)
            measured=complex(*coords);actual+=float(q)*measured;expected+=float(q)*z
            values.append(dict(Q=str(q),scalar_phase=phase,real=measured.real,imag=measured.imag))
        self.assertAlmostEqual(abs(actual-expected),0.,delta=TOL)
        EVIDENCE.append(dict(case='finite_Q_average',paths=values,real=actual.real,imag=actual.imag,
                             error=abs(actual-expected),sampling_calls=0))

    def test_missing_phase_correction_is_detected(self):
        L,R,psi=fixture(2,.43);plan=v.build_instrument(L,R)
        wrong=plan.circuit.copy();wrong.data.pop() # Remove only final P(ell+r).
        before=Statevector(initial(psi,5,plan.target)).evolve(wrong).data
        after=Statevector(initial(psi,5,plan.target)).evolve(plan.circuit).data
        # Ancilla offdiagonal, before readout.
        def signal(x):
            return 2*sum(np.conj(x[i])*x[i|(1<<plan.ancilla)] for i in range(len(x))
                         if not ((i>>plan.ancilla)&1))
        self.assertGreater(abs(signal(before)-signal(after)),.05)

    def test_known_SE_is_identical_not_a_PR_specific_gadget(self):
        for n,axis in product((1,2,3),('X','Y')):
            L,R,_=fixture(n)
            plan=v.build_instrument(L,R,axis=axis)
            matrix_gate(Operator(plan.circuit).data,Operator(known_se(L,R,axis)).data,
                        TOL,label='KNOWN_SE_IDENTICAL_FULL_UNITARY')
        EVIDENCE.append(dict(case='known_SE_equivalence',operations='IDENTICAL',
                             independent_incremental_gain=0))

    def test_identity_zero_phase_and_measurement_mapping(self):
        L,R=QuantumCircuit(1),QuantumCircuit(1)
        for axis in ('X','Y'):
            plan=v.build_instrument(L,R,axis=axis,measure=True)
            body=plan.circuit.remove_final_measurements(inplace=False)
            out=Statevector(initial(np.array([3.,4j])/5,3,plan.target)).evolve(body).data
            self.assertAlmostEqual(2*bit_probability(out,plan.ancilla)-1,1. if axis=='X' else 0.,delta=TOL)
            self.assertEqual(plan.decode(0),1);self.assertEqual(plan.decode(1),-1)
            self.assertEqual(plan.circuit.find_bit(plan.circuit.data[-1].clbits[0]).index,0)
            self.assertEqual(plan.circuit.metadata['automatic_retries'],0)
            self.assertFalse(plan.circuit.metadata['reference_postselection'])
        with self.assertRaises(ValueError):plan.decode(None)

    def test_nonvacuum_gate_unknown_oracle_and_science_width_rejected(self):
        for name in ('x','h','rx'):
            L=QuantumCircuit(2)
            getattr(L,name)(.2,0) if name=='rx' else getattr(L,name)(0)
            with self.assertRaises(ValueError):v.build_instrument(L,QuantumCircuit(2))
        c=QuantumCircuit(1);c.append(Gate('unknown_unitary_or_state_oracle',1,[]),[0])
        with self.assertRaises(ValueError):v.vacuum_phase(c)
        with self.assertRaises(ValueError):v.vacuum_phase(QuantumCircuit(8))
        L,R,_=fixture(2)
        with self.assertRaises(ValueError):v.build_instrument(L,R,mapping=[0,0,1,2,3])
        with self.assertRaises(ValueError):v.build_instrument(L,R,measure=True)

    def test_multiblock_shared_eigenbasis_arbitrary_superposition(self):
        # Diagonal blocks share their eigenbasis; psi need not be an eigenstate.
        L1,R1,L2,R2=(QuantumCircuit(1) for _ in range(4))
        L1.rz(.23,0);R1.rz(-.41,0);L2.rz(.17,0);R2.rz(-.53,0)
        psi=np.array([3.,4j])/5
        left,right=coherent_two_blocks(L1,R1,L2,R2,psi)
        self.assertLess(np.max(abs(left-right)),TOL)
        EVIDENCE.append(dict(case='shared_eigenbasis_multiblock',density_error=float(np.max(abs(left-right)))))

    def test_multiblock_noncommuting_split_counterexample_even_for_U_powers(self):
        n=2;U=QuantumCircuit(n);U.rz(.71,0)
        L1=QuantumCircuit(n);L1.xx_plus_yy(math.pi/2,0.,0,1)
        R1=U.compose(L1.inverse()) # L1 R1 = U.
        L2=QuantumCircuit(n);R2=U.compose(U)
        psi=np.array([0.,1.,0.,0.],complex) # Exact eigenstate of U.
        left,right=coherent_two_blocks(L1,R1,L2,R2,psi)
        difference=float(np.max(abs(left-right)))
        self.assertGreater(difference,.05)
        # Actual inverse Fourier phase-register measurement distributions.
        ft=np.array([[np.exp(-2j*math.pi*j*k/4)/2 for k in range(4)] for j in range(4)])
        p0=np.real(np.diag(ft@left@ft.conj().T));p1=np.real(np.diag(ft@right@ft.conj().T))
        self.assertGreater(np.max(abs(p0-p1)),.01)
        EVIDENCE.append(dict(case='noncommuting_multiblock_counterexample',density_difference=difference,
            controlled_QPE_probabilities=p0.tolist(),reference_QPE_probabilities=p1.tolist(),
            same_U_powers=True,target_eigenstate=True,single_block_signals_still_valid=True))

    def test_adapter_no_state_Born_RNG_loader_compile_or_inverse_preparation(self):
        tree=ast.parse(Path(v.__file__).read_text())
        forbidden={'Statevector','Operator','SavedEngine','from_fixed_bundle','transpile',
                   'random','default_rng','open','read_text','evolve','initialize'}
        names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
        attrs={x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)}
        self.assertFalse(forbidden&(names|attrs))


def coherent_two_blocks(L1,R1,L2,R2,psi):
    """Independent coherent-QPE check; no state reuse claim from one-shot tests."""
    n=L1.num_qubits;old=QuantumCircuit(n+2);new=QuantumCircuit(2*n+2)
    for j in range(2):old.h(n+j);new.h(2*n+j)
    for j,(L,R) in enumerate(((L1,R1),(L2,R2))):
        old.append(R.compose(L).to_gate().control(),[n+j,*range(n)])
        v.append_gadget(new,L,R,target=range(n),reference=range(n,2*n),ancilla=2*n+j)
    a=Statevector(initial(psi,n+2,range(n))).evolve(old)
    b=Statevector(initial(psi,2*n+2,range(n))).evolve(new)
    return partial_trace(a,list(range(n))).data,partial_trace(b,list(range(2*n))).data


if __name__=='__main__':unittest.main()
