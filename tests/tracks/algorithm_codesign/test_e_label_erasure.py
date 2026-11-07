"""Invented states only; exhaustive Born branches, no RNG/shots or compile.

The small operational enumerator extends the existing M artificial-test
approach to a Clbit condition and a measurement inside its success block.
Matrix targets and classical-label Hadamard references are independent of
the adapter and its decoder. Nothing here loads saved scientific results.
"""
import ast
from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import unittest

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator, Statevector

from trottertracks.algorithm_codesign import e_label_erasure as e
from trottertracks.pr_dictionary_validation.gates import matrix_gate

EVIDENCE = []
TOL = 4e-12  # Artificial binary64 agreement tolerance, not an error certificate.


def fixture(n=1):
    A, B, W0, W1 = (QuantumCircuit(n) for _ in range(4))
    A.rx(.31, 0); A.rz(-.27, 0); A.global_phase = .23
    B.ry(-.42, 0); B.h(0); B.global_phase = -.19
    W0.rx(-.37, 0); W0.rz(.23, 0); W0.global_phase = .41
    W1.ry(.53, 0); W1.global_phase = math.pi-.17  # Signed, phased unitary.
    if n == 2:
        A.cx(0, 1); B.rzz(.18, 0, 1)
        W0.cx(1, 0); W0.ry(.22, 1)
        W1.rzz(-.28, 0, 1); W1.rx(.16, 1)
    psi = np.array([3., 4j])/5 if n == 1 else np.array([1., 1j, -1., 1.])/2
    return A, B, W0, W1, psi


def branches(circuit, psi):
    """Read actual if_else/measure instructions; keep unnormalized branches.

    No sampling or conditioning on success; absent readout remains None.
    This deliberately supports only the one-level artificial adapter scope.
    """
    initial = np.zeros(2**circuit.num_qubits, complex)
    initial[:len(psi)] = psi
    records = [(initial, {}, {})]

    def walk(block, records, qmap, cmap, nested=False):
        for item in block.data:
            op = item.operation
            wires = [qmap[block.find_bit(q).index] for q in item.qubits]
            cbits = [cmap[block.find_bit(b).index] for b in item.clbits]
            updated = []
            for vector, memory, counts in records:
                count = counts.copy()
                label = op.label or op.name
                count[label] = count.get(label, 0)+1
                if op.name == 'measure':
                    for bit in (0, 1):
                        v = vector.copy()
                        v[((np.arange(len(v)) >> wires[0]) & 1) != bit] = 0
                        m = memory.copy(); m[cbits[0]] = bit
                        updated.append((v, m, count.copy()))
                elif op.name == 'if_else':
                    if nested:
                        raise AssertionError('unexpected nested control flow')
                    target, expected = op.condition
                    self_index = block.find_bit(target).index
                    chosen = 0 if memory[cmap[self_index]] == expected else 1
                    if chosen < len(op.blocks):
                        updated += walk(op.blocks[chosen], [(vector, memory, count)],
                                        wires, cbits, nested=True)
                    else:
                        updated.append((vector, memory, count))
                else:
                    v = Statevector(vector).evolve(op, qargs=wires).data
                    updated.append((v, memory, count))
            records = updated
        return records

    return walk(circuit, records, list(range(circuit.num_qubits)),
                list(range(circuit.num_clbits)))


def statistics(plan, psi):
    rows = branches(plan.circuit, psi)
    total = mean = second = success = 0.
    saved = []
    for v, bits, counts in rows:
        prob = float(np.vdot(v, v).real)
        flag = bits[0]
        outcome = plan.decode(flag, bits.get(1))
        total += prob; mean += prob*outcome; second += prob*outcome**2
        success += prob*(flag == 0)
        saved.append(dict(flag=flag, signal_bit=bits.get(1), probability=prob,
                          outcome=outcome, executed_blocks=counts))
    if abs(total-1) > TOL:
        raise AssertionError(f'all-attempt normalization: {total}')
    return dict(mean=mean, second=second, success=success,
                variance=second-mean**2, branches=saved, normalization=total)


def reference(A, B, W0, W1, p, psi, axis):
    """Independent label-specific ordinary controlled Hadamard circuits."""
    means = []
    for W in (W0, W1):
        system = B.compose(W).compose(A)
        n = system.num_qubits
        c = QuantumCircuit(n+1); c.h(n)
        c.append(system.to_gate().control(), [n, *range(n)])
        if axis == 'Y': c.sdg(n)
        c.h(n)
        initial = np.zeros(2**(n+1), complex); initial[:len(psi)] = psi
        v = Statevector(initial).evolve(c).data
        means.append(float(sum(abs(v[:len(psi)])**2)-sum(abs(v[len(psi):])**2)))
    return float(p)*means[0]+float(1-p)*means[1], means


def known_lcu(A, B, W0, W1, p, axis, *, prefix_override=None):
    """Standard positive two-term LCU with the SAME early-flag optimization.

    Independent construction for comparison; not an alias of build_instrument.
    """
    n = A.num_qubits; a, k = n, n+1
    t = 2*math.acos(math.sqrt(float(p)))
    c = QuantumCircuit(n+2)
    c.h(a); c.append(B.to_gate().control(label='common_B'), [a, *range(n)])
    c.ry(t, k)
    c.append(W0.to_gate().control(2, ctrl_state=2, label='SELECT_W0'), [k,a,*range(n)])
    c.append(W1.to_gate().control(2, ctrl_state=3, label='SELECT_W1'), [k,a,*range(n)])
    c.ry(-t, k)
    if prefix_override is not None:
        c = prefix_override.copy()
    bits = ClassicalRegister(2, 'known_outcome'); c.add_register(bits)
    c.measure(k, bits[0])
    with c.if_test((bits[0], 0)):
        c.append(A.to_gate().control(label='common_A'), [a,*range(n)])
        if axis == 'Y': c.sdg(a)
        c.h(a); c.measure(a, bits[1])
    return c


class EInstrumentTests(unittest.TestCase):
    def test_kraus_blocks_on_every_input_and_noncommuting_gates(self):
        for n, p in product((1, 2), (F(2,5), F(3,7))):
            A,B,W0,W1,_ = fixture(n); d = 2**n
            am,bm,w0,w1 = (Operator(c).data for c in (A,B,W0,W1))
            self.assertGreater(np.linalg.norm(w0@w1-w1@w0), .1)
            self.assertGreater(np.linalg.norm(am@bm-bm@am), .1)
            plan = e.build_instrument(A,B,W0,W1,p=p,axis='X')
            m = Operator(plan.prefix).data[:, :d]
            s = float(p)*w0+float(1-p)*w1
            k1 = math.sqrt(float(p*(1-p)))*(w1-w0)
            expected = np.concatenate((np.eye(d),s@bm,np.zeros((d,d)),k1@bm))/math.sqrt(2)
            matrix_gate(m, expected, TOL, label='E_ALL_INPUT_PREFIX_KRAUS')
            matrix_gate(s.conj().T@s+k1.conj().T@k1,np.eye(d),TOL,label='E_KRAUS_COMPLETENESS')

    def test_dynamic_all_branch_mean_and_second_moment(self):
        for n,p,axis in product((1,2),(F(2,5),F(3,7)),('X','Y')):
            A,B,W0,W1,psi=fixture(n)
            plan=e.build_instrument(A,B,W0,W1,p=p,axis=axis)
            actual=statistics(plan,psi)
            ref, labels=reference(A,B,W0,W1,p,psi,axis)
            am,bm,w0,w1=(Operator(c).data for c in (A,B,W0,W1))
            s=float(p)*w0+float(1-p)*w1
            z=np.vdot(psi,am@s@bm@psi)
            ps=(1+float(np.linalg.norm(s@bm@psi)**2))/2
            self.assertAlmostEqual(actual['mean'],ref,delta=TOL)
            self.assertAlmostEqual(ref,z.real if axis=='X' else z.imag,delta=TOL)
            self.assertAlmostEqual(actual['second'],ps,delta=TOL)
            self.assertAlmostEqual(actual['success'],ps,delta=TOL)
            self.assertGreaterEqual(actual['variance']+TOL,(1-ref**2)/2)
            self.assertLessEqual(actual['variance'],1-ref**2+TOL)
            EVIDENCE.append(dict(case=f'noncommuting_{n}q_p{p}_{axis}',axis=axis,
                p=str(p),system_qubits=n,reference_label_means=labels,
                reference_mean=ref,formula_success=ps,**actual))

    def test_failure_has_only_signal_one_and_zero_XY_with_or_without_suffix(self):
        A,B,W0,W1,psi=fixture(2)
        plan=e.build_instrument(A,B,W0,W1,p=F(2,5),axis='Y')
        initial=np.zeros(16,complex);initial[:4]=psi
        v=Statevector(initial).evolve(plan.prefix).data
        # Indices k=1,a=0 vanish on ALL inputs (also tested as an operator block).
        self.assertLess(np.linalg.norm(v[8:12]),TOL)
        failure=v.copy();failure[:8]=0
        self.assertGreater(np.linalg.norm(failure),.1)
        full_A=QuantumCircuit(4);full_A.append(A.to_gate().control(),[2,0,1])
        for state in (failure,Statevector(failure).evolve(full_A).data):
            for axis in ('X','Y'):
                read=QuantumCircuit(4)
                if axis=='Y':read.sdg(2)
                read.h(2)
                out=Statevector(state).evolve(read).data
                signed=sum(abs(out[i])**2*(1-2*((i>>2)&1)) for i in range(16))
                self.assertAlmostEqual(signed,0.,delta=TOL)
        for row in statistics(plan,psi)['branches']:
            if row['flag']:
                self.assertIsNone(row['signal_bit'])
                self.assertNotIn('common_A',row['executed_blocks'])
                self.assertEqual(row['outcome'],0)

    def test_identical_unitaries_have_no_failure_or_M2_gain(self):
        A,B,W0,_,psi=fixture()
        for axis in ('X','Y'):
            plan=e.build_instrument(A,B,W0,W0,p=F(2,7),axis=axis)
            actual=statistics(plan,psi)
            self.assertAlmostEqual(actual['success'],1.,delta=TOL)
            self.assertAlmostEqual(actual['second'],1.,delta=TOL)
            self.assertAlmostEqual(actual['mean'],reference(A,B,W0,W0,F(2,7),psi,axis)[0],delta=TOL)
        EVIDENCE.append(dict(case='identical_W',success=actual['success'],second=actual['second']))

    def test_scalar_signed_relative_phase_and_Y_sign(self):
        A,B,W0,W1=(QuantumCircuit(1) for _ in range(4))
        A.global_phase=.23;B.global_phase=-.19
        W0.global_phase=-.43;W1.global_phase=math.pi+.27
        psi=np.array([1.,0.],complex);p=F(2,5)
        z=np.exp(.04j)*(float(p)*np.exp(-.43j)+float(1-p)*np.exp((math.pi+.27)*1j))
        for axis in ('X','Y'):
            plan=e.build_instrument(A,B,W0,W1,p=p,axis=axis)
            self.assertAlmostEqual(statistics(plan,psi)['mean'],z.real if axis=='X' else z.imag,delta=TOL)
        self.assertGreater(abs(z.imag),.1)
        block=plan.circuit.data[-1].operation.blocks[0]
        self.assertEqual([x.operation.name for x in block.data][-3:],['sdg','h','measure'])
        self.assertEqual((plan.signal_ancilla,plan.pool_ancilla),(1,2))
        self.assertEqual(plan.circuit.find_bit(plan.circuit.data[-2].clbits[0]).index,0)

    def test_decoder_and_all_attempt_denominator(self):
        A,B,W0,W1,_=fixture()
        plan=e.build_instrument(A,B,W0,W1,p=F(2,5),axis='X')
        values=[plan.decode(0,0),plan.decode(1),plan.decode(0,1),plan.decode(1)]
        self.assertEqual(values,[1,0,-1,0])
        self.assertEqual(sum(x*x for x in values)/len(values),.5)
        with self.assertRaises(ValueError):plan.decode(0)
        for flag,bit in ((True,0),(2,0),(0,True),(1,3)):
            with self.assertRaises(ValueError):plan.decode(flag,bit)
        self.assertEqual(plan.circuit.metadata['automatic_retries'],0)
        self.assertTrue(plan.circuit.metadata['fresh_state_consumed_each_attempt'])

    def test_consistent_PREP_probability_error_changes_target(self):
        A,B,W0,W1,psi=fixture()
        p,q=F(2,5),F(9,20)
        differences=[]
        for axis in ('X','Y'):
            actual=statistics(e.build_instrument(A,B,W0,W1,p=q,axis=axis),psi)['mean']
            ref,labels=reference(A,B,W0,W1,p,psi,axis)
            self.assertAlmostEqual(actual-ref,float(q-p)*(labels[0]-labels[1]),delta=TOL)
            differences.append(actual-ref)
        self.assertGreater(max(map(abs,differences)),1e-3)
        EVIDENCE.append(dict(case='PREP_probability_bias',nominal=str(p),effective=str(q),
                             coordinate_bias=differences,certified_bound='UNKNOWN'))

    def test_inconsistent_unPREP_leaks_inactive_branch(self):
        A,B,W0,W1,psi=fixture();p=F(2,5)
        plan=e.build_instrument(A,B,W0,W1,p=p,axis='X')
        bad=plan.prefix.copy()
        item=bad.data[-1]
        bad.data[-1]=item.replace(operation=RYGate(float(item.operation.params[0])+.12))
        initial=np.zeros(8,complex);initial[:2]=psi
        v=Statevector(initial).evolve(bad).data
        self.assertGreater(np.linalg.norm(v[4:6]),.01)
        means=[]
        for axis in ('X','Y'):
            badc=known_lcu(A,B,W0,W1,p,axis,prefix_override=bad)
            result=statistics(e.EInstrument(badc,bad,1,1,2,p,axis,True),psi)
            means.append(result['mean']-reference(A,B,W0,W1,p,psi,axis)[0])
        self.assertGreater(max(map(abs,means)),1e-3)
        EVIDENCE.append(dict(case='unPREP_mismatch',inactive_failure_norm=float(np.linalg.norm(v[4:6])),
                             coordinate_bias=means,certified_bound='UNKNOWN'))

    def test_standard_LCU_same_early_abort_is_identical_instrument_and_cost(self):
        for n,axis in product((1,2),('X','Y')):
            A,B,W0,W1,psi=fixture(n);p=F(2,5)
            plan=e.build_instrument(A,B,W0,W1,p=p,axis=axis)
            known=known_lcu(A,B,W0,W1,p,axis)
            left=branches(plan.circuit,psi);right=branches(known,psi)
            self.assertEqual(len(left),len(right))
            for (v,m,c),(w,m2,c2) in zip(left,right):
                matrix_gate(v,w,TOL,label='STANDARD_LCU_IDENTICAL_BRANCH')
                self.assertEqual(m,m2)
                self.assertEqual(c,c2)
        EVIDENCE.append(dict(case='standard_LCU_same_early_abort',
            operations='IDENTICAL_UP_TO_PREP_ROUNDING',estimator='IDENTICAL',
            logical_branch_cost='IDENTICAL',incremental_gain=0))

    def test_late_flag_same_moments_but_pays_suffix_on_failure(self):
        A,B,W0,W1,psi=fixture(2);p=F(2,5)
        for axis in ('X','Y'):
            early=e.build_instrument(A,B,W0,W1,p=p,axis=axis)
            late=e.build_instrument(A,B,W0,W1,p=p,axis=axis,early_abort=False)
            left,right=statistics(early,psi),statistics(late,psi)
            for key in ('mean','second','success'):
                self.assertAlmostEqual(left[key],right[key],delta=TOL)
            for plan,result in ((early,left),(late,right)):
                count=sum(r['probability']*r['executed_blocks'].get('common_A',0) for r in result['branches'])
                self.assertAlmostEqual(count,left['success'] if plan.early_abort else 1.,delta=TOL)
                self.assertTrue(all(r['executed_blocks']['common_B']==1 for r in result['branches']))
                self.assertTrue(all(r['executed_blocks']['SELECT_W0']==1 for r in result['branches']))
                self.assertTrue(all(r['executed_blocks']['SELECT_W1']==1 for r in result['branches']))
            EVIDENCE.append(dict(case='early_vs_late_'+axis,success=left['success'],
                                 saved_suffix_fraction=1-left['success'],
                                 common_B_calls=1,SELECT_branches=2,fresh_state_per_attempt=1,
                                 compiled_cost='UNKNOWN'))

    def test_direct_grouped_observable_and_optimal_stratification_can_beat_E(self):
        A,B,W0,W1=(QuantumCircuit(1) for _ in range(4));W0.z(0);W1.x(0)
        psi=np.array([1.,0.],complex);p=F(2,5);pf=float(p)
        actual=statistics(e.build_instrument(A,B,W0,W1,p=p,axis='X'),psi)
        radius=math.hypot(pf,1-pf);angle=math.atan2(1-pf,pf)
        c=QuantumCircuit(1);c.ry(-angle,0)
        v=Statevector(psi).evolve(c).data
        mean=radius*(abs(v[0])**2-abs(v[1])**2)
        direct_variance=radius**2-mean**2
        # Analytic optimal fixed-total-N allocation: sigma0=0, sigma1=1.
        strat_variance=(1-pf)**2
        self.assertAlmostEqual(mean,actual['mean'],delta=TOL)
        self.assertAlmostEqual(direct_variance,strat_variance,delta=TOL)
        self.assertLess(direct_variance,actual['variance']-.1)
        EVIDENCE.append(dict(case='strong_baselines_X_Z_pair',target_mean=mean,
            E_variance=actual['variance'],raw_variance=1-mean**2,
            optimal_stratified_coefficient=strat_variance,direct_grouped_variance=direct_variance,
            status='ARTIFICIAL_ONLY_NO_COST_ORACLE'))

    def test_small_angle_reduces_both_erasure_and_suffix_savings(self):
        psi=np.array([1.,0.],complex);p=F(2,5);out=[]
        for angle in (.2,.02):
            A,B,W0,W1=(QuantumCircuit(1) for _ in range(4))
            W0.rz(2*angle,0);W1.rx(2*angle,0)
            r=statistics(e.build_instrument(A,B,W0,W1,p=p,axis='X'),psi)
            expected=float(p*(1-p))*math.sin(angle)**2
            self.assertAlmostEqual(1-r['success'],expected,delta=TOL)
            out.append(1-r['success'])
        self.assertLess(out[1],out[0]/90)
        EVIDENCE.append(dict(case='small_angle_savings',angles=[.2,.02],failure=out))

    def test_nonphysical_unsupported_and_science_inputs_stop(self):
        A,B,W0,W1,_=fixture()
        for p in (0,1,-1,F(3,2),.4,True,float('nan'),F(1,10**500)):
            with self.assertRaises(ValueError):e.build_instrument(A,B,W0,W1,p=p,axis='X')
        bad=[]
        c=QuantumCircuit(1);c.reset(0);bad.append(c)
        c=QuantumCircuit(1,1);c.measure(0,0);bad.append(c)
        c=QuantumCircuit(1);c.rz(Parameter('t'),0);bad.append(c)
        c=QuantumCircuit(1);c.append(Gate('free_state_oracle',1,[]),[0]);bad.append(c)
        c=QuantumCircuit(1);c.append(Gate('rz',1,[.2]),[0]);bad.append(c)
        bad += [QuantumCircuit(3),QuantumCircuit(8)]
        for c in bad:
            with self.assertRaises(ValueError):e.build_instrument(c,B,W0,W1,p=F(2,5),axis='X')
        with self.assertRaises(ValueError):e.build_instrument(A,B,W0,W1,p=F(2,5),axis='Z')
        with self.assertRaises(ValueError):e.build_instrument(A,B,W0,W1,p=F(2,5),axis='X',early_abort=1)

    def test_adapter_has_no_state_Born_rng_loader_or_compiler(self):
        tree=ast.parse(Path(e.__file__).read_text())
        forbidden={'Statevector','Operator','SavedEngine','from_fixed_bundle','qualify_fixed_bundle',
                   'transpile','random','default_rng','read_text','open','evolve','initialize'}
        names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
        attrs={x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)}
        self.assertFalse(forbidden&(names|attrs))
        self.assertFalse(e.build_instrument(*fixture()[:4],p=F(2,5),axis='X').numerically_certified)


if __name__=='__main__':unittest.main()
