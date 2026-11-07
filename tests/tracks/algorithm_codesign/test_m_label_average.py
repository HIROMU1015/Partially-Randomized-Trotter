"""New one/two-qubit instrument references only: no data files, RNG or compile.

The operational enumerator reads the actual Qiskit measure/if_else instructions.
Statevector/Operator are confined to this artificial test, never algorithm inputs.
"""
import ast
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from trottertracks.algorithm_codesign import m_label_average as m
from trottertracks.pr_dictionary_validation.gates import matrix_gate

EVIDENCE = []


def fixture(n=2, *, kind='product', phi=None, trivial=False):
    A, B, V = (QuantumCircuit(n) for _ in range(3))
    if not trivial:
        A.rx(.31, 0); A.rz(-.27, 0); A.global_phase = .23
        B.ry(-.42, 0); B.h(0); B.global_phase = -.19
        V.ry(.63, 0); V.global_phase = .11
        if n == 2:
            A.cx(0, 1); B.rzz(.18, 0, 1); V.rx(-.29, 1)
    law = m.CategoricalLaw(('i', 'nz', 'zz', 'outside', 'zero'),
                           (F(1,8), F(3,8), F(2,8), F(2,8), F(0)), 'invented-grid')
    supports = ((), (0,), (0,1) if n == 2 else (0,))
    rows = tuple(m.SlotLabel(label, 'outer-context', 'slot-0', 'selected-policy', law.source,
        mass, A, B, V, tuple(f's{q}' for q in range(n)), 'selected-V', support, sign,
        kind, phi, original_basis_id='original-V')
        for label, mass, support, sign in zip(law.labels[:3], law.masses[:3], supports, (1,-1,1)))
    return rows, law


def instrument_distribution(plan, psi):
    """Exhaustive operational semantics of this prototype's dynamic instructions.

    Unnormalized post-measurement vectors retain branch probability. No calls to
    a sampler, Statevector.measure, probability oracle or the readout formula.
    """
    circuit = plan.circuit
    initial = np.zeros(2**circuit.num_qubits, complex)
    initial[:len(psi)] = psi
    branches = [(initial, [0]*circuit.num_clbits)]
    for item in circuit.data:
        op = item.operation
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == 'measure':
            cbit = circuit.find_bit(item.clbits[0]).index
            updated = []
            for vector, memory in branches:
                for bit in (0, 1):
                    projected = vector.copy()
                    projected[((np.arange(len(vector)) >> wires[0]) & 1) != bit] = 0
                    saved = memory.copy(); saved[cbit] = bit
                    updated.append((projected, saved))
            branches = updated
        elif op.name == 'if_else':
            register, value = op.condition
            updated = []
            for vector, memory in branches:
                actual = sum(memory[circuit.find_bit(b).index] << k for k,b in enumerate(register))
                block_index = 0 if actual == value else 1
                if block_index < len(op.blocks):
                    block = op.blocks[block_index]
                    for child in block.data:
                        if child.operation.name in ('measure', 'if_else', 'reset'):
                            raise AssertionError('unexpected nested nonunitary readout')
                        qargs = [wires[block.find_bit(q).index] for q in child.qubits]
                        vector = Statevector(vector).evolve(child.operation, qargs=qargs).data
                updated.append((vector, memory))
            branches = updated
        else:
            branches = [(Statevector(v).evolve(op, qargs=wires).data, bits) for v,bits in branches]
    probabilities = {}
    for vector, memory in branches:
        z = sum(memory[k] << k for k in range(plan.system_qubits))
        key = (z, memory[plan.system_qubits])
        probabilities[key] = probabilities.get(key, 0.) + float(np.vdot(vector,vector).real)
    return probabilities


def stats(plan, psi):
    distribution = instrument_distribution(plan, psi)
    total = sum(distribution.values())
    if abs(total-1) > 3e-12 or min(distribution.values()) < 0:
        raise AssertionError('artificial reference normalization/probability')
    mean = sum(p*float(plan.decode(*bits)) for bits,p in distribution.items())
    second = sum(p*float(plan.decode(*bits))**2 for bits,p in distribution.items())
    return mean, second


def original_system(row):
    """Independent label-specific target, using primitive Z/RZ/RZZ instructions."""
    circuit = QuantumCircuit(row.A.num_qubits)
    circuit.compose(row.B, inplace=True); circuit.compose(row.V, inplace=True)
    if row.kind == 'product':
        for q in row.support: circuit.z(q)
        if row.sign < 0: circuit.global_phase += math.pi
    elif not row.support:
        circuit.global_phase -= row.phi.numeric*row.sign
    elif len(row.support) == 1:
        circuit.rz(2*row.phi.numeric*row.sign, row.support[0])
    else:
        circuit.rzz(2*row.phi.numeric*row.sign, *row.support)
    circuit.compose(row.V.inverse(), inplace=True); circuit.compose(row.A, inplace=True)
    circuit.global_phase += row.scalar_phase.numeric + row.event_phase.numeric
    return circuit


def original_stats(rows, law, psi, axis):
    mean = second = 0.
    mass = sum(law.mass(row.label) for row in rows)
    for row in rows:
        system = original_system(row); n = system.num_qubits
        circuit = QuantumCircuit(n+1); circuit.h(n)
        circuit.append(system.to_gate().control(), [n,*range(n)])
        if axis == 'Y': circuit.sdg(n)
        circuit.h(n)
        initial = np.zeros(2**(n+1),complex); initial[:len(psi)] = psi
        vector = Statevector(initial).evolve(circuit).data
        p0 = float(np.sum(abs(vector[:2**n])**2))
        p1 = float(np.sum(abs(vector[2**n:])**2))
        q = float(law.mass(row.label)/mass)
        mean += q*(p0-p1); second += q*(p0+p1)
    return mean, second


def analytic_second(rows, law, psi):
    """Independent Born marginal of z and metadata formula, tests only."""
    first = rows[0]
    u = Operator(first.V).data @ Operator(first.A).data.conj().T @ psi
    v = Operator(first.V).data @ Operator(first.B).data @ psi
    pz = (abs(u)**2+abs(v)**2)/2
    mass = sum(law.mass(row.label) for row in rows)
    total = 0.
    for z,p in enumerate(pz):
        r = sum(law.mass(row.label)/mass*row.sign*(-1)**sum((z>>q)&1 for q in row.support)
                for row in rows)
        square = float(r*r) if first.kind == 'product' else (
            math.cos(first.phi.numeric)**2+math.sin(first.phi.numeric)**2*float(r*r))
        total += p*square
    return float(total)


def ancilla_free_reference(rows, law, psi, axis, *, average):
    """Stronger known direct measurement when A=B†, artificial reference only.

    System z directly estimates the eigenvalue of the normal unitary. Reading
    its real/imaginary coordinates needs no noisy ancillary Pauli measurement.
    """
    first=rows[0]
    matrix_gate(Operator(first.A).data.conj().T,Operator(first.B).data,3e-12,
                label='ANCILLA_FREE_CONTEXT_REQUIRED')
    circuit=first.B.copy();circuit.compose(first.V,inplace=True)
    probabilities=abs(Statevector(psi).evolve(circuit).data)**2
    mass=sum(law.mass(row.label) for row in rows)
    mean=second=0.
    for z,p in enumerate(probabilities):
        values=[]
        for row in rows:
            d=row.sign*(-1)**sum((z>>q)&1 for q in row.support)
            phase=np.exp(1j*(row.scalar_phase.numeric+row.event_phase.numeric))
            value=phase*(d if row.kind=='product' else np.exp(-1j*row.phi.numeric*d))
            values.append(value.real if axis=='X' else value.imag)
        qs=[float(law.mass(row.label)/mass) for row in rows]
        avg=sum(q*v for q,v in zip(qs,values))
        mean+=p*avg
        second+=p*(avg*avg if average else sum(q*v*v for q,v in zip(qs,values)))
    return float(mean),float(second)


class MInstrumentTests(unittest.TestCase):
    def test_exact_actual_law_signed_support_identity_and_partition(self):
        rows, law = fixture()
        group = m.qualify_group(rows, law)
        self.assertEqual(group.mass, F(3,4))
        self.assertEqual(group.conditional, (F(1,6),F(1,2),F(1,3)))
        self.assertEqual(group.r(0), F(0))
        self.assertEqual(group.r(1), F(1,3))
        self.assertEqual(sum(group.conditional), 1)
        # Original basis identity is not the selected basis/gate identity.
        other = replace(rows[1], original_basis_id='unrelated-original', selected_basis_id='another-selected')
        with self.assertRaisesRegex(m.Ineligible,'SELECTED_BASIS_MISMATCH'):
            m.qualify_group((rows[0],other),law)
        groups = m.partition_supplied_rows((rows[0],other,rows[2]),law)
        self.assertEqual([tuple(r.label for r in g.rows) for g in groups], [('i','zz'),('nz',)])
        self.assertEqual(sum(g.mass for g in groups),group.mass)

    def test_noncommon_nearby_or_channel_equivalent_descriptions_are_rejected(self):
        rows, law = fixture(kind='rotation',phi=m.Angle('rad',.37)); row = rows[1]
        close = row.V.copy(); close.rz(1e-14,0)
        phase = row.A.copy(); phase.global_phase += .2
        reverse = QuantumCircuit(2); reverse.rz(-.27,0); reverse.rx(.31,0); reverse.cx(0,1)
        reverse.global_phase = row.A.global_phase
        changes = [('CONTEXT',replace(row,context_id='different')),
            ('SLOT',replace(row,slot_id='another')),('POLICY',replace(row,policy_id='different')),
            ('V_SEQUENCE',replace(row,V=close)),('A_SEQUENCE',replace(row,A=phase)),
            ('A_SEQUENCE',replace(row,A=reverse)),('B_SEQUENCE',replace(row,B=phase)),
            ('PHI',replace(row,phi=m.Angle('rad',.38))),
            ('KIND',replace(row,kind='product',phi=None)),
            ('SCALAR_PHASE',replace(row,scalar_phase=m.Angle('rad',.2))),
            ('EVENT_PHASE',replace(row,event_phase=m.Angle('pi',1))),
            ('QUBIT_ORDER',replace(row,system_order=('s1','s0')))]
        for code, changed in changes:
            with self.subTest(code=code):
                with self.assertRaisesRegex(m.Ineligible,code+'_MISMATCH'):
                    m.qualify_group((rows[0],changed),law)

    def test_zero_support_actual_probability_and_invalid_mapping_stop(self):
        rows, law = fixture(); row = rows[0]
        bad = [('ZERO_ACTUAL_SUPPORT',replace(row,label='zero',actual_mass=F(0))),
            ('ACTUAL_MASS_MISMATCH',replace(row,actual_mass=F(1,4))),
            ('LAW_ID_MISMATCH',replace(row,law_id='ideal-not-actual')),
            ('QUBIT_MAPPING',replace(row,system_order=('s0','s0'))),
            ('SIGNED_DIAGONAL_SUPPORT',replace(row,support=(0,0))),
            ('SIGNED_DIAGONAL_SUPPORT',replace(row,support=(2,))),
            ('SIGNED_DIAGONAL_SUPPORT',replace(row,sign=0))]
        for code, changed in bad:
            with self.subTest(code=code):
                with self.assertRaisesRegex(m.Ineligible,code): m.qualify_group((changed,),law)
        offgrid = m.CategoricalLaw(('i','nz'),(F(1,3),F(2,3)),'not-53-bit')
        with self.assertRaisesRegex(m.Ineligible,'ACTUAL_GRID_MASS'):m.qualify_group(rows,offgrid)
        with self.assertRaisesRegex(m.Ineligible,'ONE_SLOT_REQUIRED'):
            m.partition_supplied_rows((row,replace(rows[1],slot_id='another')),law)

    def test_full_preparation_matrix_has_actual_adjoint_order_and_coherent_phases(self):
        for n in (1,2):
            rows,law=fixture(n); group=m.qualify_group(rows,law)
            plan=m.build_instrument(group,'X')
            A,B,V=(Operator(c).data for c in (rows[0].A,rows[0].B,rows[0].V))
            left,right=V@A.conj().T,V@B
            expected=np.block([[left,left],[right,-right]])/np.sqrt(2)
            matrix_gate(Operator(plan.preparation).data,expected,3e-12,label='M_BRANCH_NO_PHASE_FIT')
            # Copying input circuits protects the qualified description.
            rows[0].A.x(0)
            group.check()
            group.rows[0].A.x(0)
            with self.assertRaisesRegex(m.Ineligible,'QUALIFIED_GROUP_MUTATED'):
                m.build_instrument(group,'X')

    def test_product_instrument_matches_independent_label_hadamard_and_m2(self):
        for n in (1,2):
            rows,law=fixture(n);group=m.qualify_group(rows,law)
            states=(np.eye(2**n,dtype=complex)[0],
                    np.array([3,4j])/5 if n==1 else np.array([1,1j,-1,1])/2)
            for psi,axis in product(states,('X','Y')):
                mean,second=stats(m.build_instrument(group,axis),psi)
                expected,old_second=original_stats(rows,law,psi,axis)
                self.assertAlmostEqual(mean,expected,places=12)
                self.assertAlmostEqual(second,analytic_second(rows,law,psi),places=12)
                self.assertLessEqual(second,old_second+3e-12)

    def test_rotation_dynamic_readout_both_axes_positive_negative_and_zero_angle(self):
        for phi in (m.Angle('rad',.37),m.Angle('rad',-.29),m.ZERO,m.Angle('pi',F(1,2))):
            rows,law=fixture(kind='rotation',phi=phi);group=m.qualify_group(rows,law)
            psi=np.array([1,1j,-1,1])/2
            for axis in ('X','Y'):
                plan=m.build_instrument(group,axis,feed_forward=True)
                mean,second=stats(plan,psi)
                expected,_=original_stats(rows,law,psi,axis)
                self.assertAlmostEqual(mean,expected,places=12)
                self.assertAlmostEqual(second,analytic_second(rows,law,psi),places=12)
                self.assertLessEqual(second,1+3e-12)

    def test_zero_r_zero_m_and_same_eigenvalue_do_not_use_tolerance_snapping(self):
        rows,_=fixture(1,kind='rotation',phi=m.Angle('pi',F(1,2)),trivial=True)
        law=m.CategoricalLaw(('i','nz'),(F(1,2),F(1,2)),'invented-grid')
        rows=tuple(replace(row,actual_mass=F(1,2)) for row in rows[:2])
        group=m.qualify_group(rows,law)
        for axis in ('X','Y'):
            plan=m.build_instrument(group,axis,feed_forward=True)
            self.assertEqual(group.r(0),0);self.assertEqual(plan.readouts[0].weight,0.)
            self.assertEqual(plan.decode(0,0),0.);self.assertEqual(plan.decode(0,1),0.)
            self.assertEqual(stats(plan,np.array([1,0],complex)),(0.,0.))
            self.assertEqual(plan.readouts[1].weight,1.) # No universal range shrink.
        # A raw floating pi/2 is not snapped to an exact quarter turn.
        raw=replace(rows[0],phi=m.Angle('rad',math.pi/2))
        raw2=replace(rows[1],phi=raw.phi)
        self.assertGreater(m.readout(m.qualify_group((raw,raw2),law),'X',0).weight,0.)
        product_group=m.qualify_group(tuple(replace(r,kind='product',phi=None) for r in rows),law)
        self.assertAlmostEqual(stats(m.build_instrument(product_group,'X'),np.array([0,1],complex))[1],1.,places=12)

    def test_product_to_quarter_turn_rotation_preserves_minus_i_relative_phase(self):
        rows,law=fixture();psi=np.array([1,1j,-1,1])/2
        product_group=m.qualify_group(rows,law)
        rotation_group=m.qualify_group(tuple(replace(r,kind='rotation',phi=m.Angle('pi',F(1,2)))
                                             for r in rows),law)
        px,py=(stats(m.build_instrument(product_group,axis),psi)[0] for axis in ('X','Y'))
        rx,ry=(stats(m.build_instrument(rotation_group,axis,feed_forward=True),psi)[0]
               for axis in ('X','Y'))
        self.assertAlmostEqual(rx,py,places=12)
        self.assertAlmostEqual(ry,-px,places=12)

    def test_scalar_event_and_controlled_relative_phases_both_coordinates(self):
        for kind in ('product','rotation'):
            rows,law=fixture(1,kind=kind,phi=m.Angle('rad',-.41) if kind=='rotation' else None)
            rows=tuple(replace(r,scalar_phase=m.Angle('rad',.43),event_phase=m.Angle('pi',1)) for r in rows)
            group=m.qualify_group(rows,law);psi=np.array([3,4j])/5
            values=[]
            for axis in ('X','Y'):
                plan=m.build_instrument(group,axis,feed_forward=kind=='rotation')
                actual,second=stats(plan,psi);reference,_=original_stats(rows,law,psi,axis)
                self.assertAlmostEqual(actual,reference,places=12)
                self.assertAlmostEqual(second,analytic_second(rows,law,psi),places=12)
                values.append(actual)
            dense=sum(float(p)*Operator(original_system(row)).data for p,row in zip(group.conditional,rows))
            signal=np.vdot(psi,dense@psi)
            self.assertAlmostEqual(complex(*values),signal,places=12)

    def test_grouping_increment_is_separate_from_single_label_direct_and_stratification(self):
        # Analytic invented examples: q=(1/4,3/4), z=0 gives r=-1/2.
        for kind,phi,m2 in [('product',None,F(1,4)),('rotation',m.Angle('pi',F(1,4)),F(5,8))]:
            rows,_=fixture(1,kind=kind,phi=phi,trivial=True)
            law=m.CategoricalLaw(('i','nz'),(F(1,4),F(3,4)),'invented-grid')
            rows=tuple(replace(row,actual_mass=law.mass(row.label)) for row in rows[:2])
            group=m.qualify_group(rows,law);psi=np.array([1,0],complex)
            for axis in ('X','Y'):
                grouped=stats(m.build_instrument(group,axis,feed_forward=kind=='rotation'),psi)
                controlled=original_stats(rows,law,psi,axis)
                singles=[stats(m.build_instrument(m.qualify_group((row,),law),axis,
                           feed_forward=kind=='rotation'),psi) for row in rows]
                direct=tuple(sum(float(p)*s[k] for p,s in zip(group.conditional,singles)) for k in (0,1))
                self.assertAlmostEqual(grouped[0],controlled[0],places=12)
                self.assertAlmostEqual(grouped[0],direct[0],places=12)
                self.assertAlmostEqual(direct[1],1.,places=12)
                self.assertAlmostEqual(grouped[1],float(m2),places=12)
                # Known averaged-observable grouping has exactly the same M2.
                self.assertAlmostEqual(grouped[1],analytic_second(rows,law,psi),places=12)
                free=ancilla_free_reference(rows,law,psi,axis,average=False)
                free_grouped=ancilla_free_reference(rows,law,psi,axis,average=True)
                self.assertAlmostEqual(grouped[0],free[0],places=12)
                self.assertAlmostEqual(grouped[0],free_grouped[0],places=12)
                if kind=='rotation':
                    self.assertAlmostEqual(free[1],.5,places=12)
                    self.assertAlmostEqual(free_grouped[1],.5 if axis=='X' else .125,places=12)
                    self.assertGreater(grouped[1],free_grouped[1])
                elif axis=='Y':
                    self.assertEqual(free_grouped[1],0.)
                    self.assertGreater(grouped[1],free_grouped[1])
                EVIDENCE.append(dict(fixture='invented-1q-I-minusZ',kind=kind,axis=axis,
                    mean=grouped[0],grouped_second=grouped[1],expected_second=str(m2),
                    controlled_second=controlled[1],single_label_interferometric_direct_second=direct[1],
                    ancilla_free_direct_second=free[1],ancilla_free_grouped_second=free_grouped[1],
                    known_averaged_observable_second=analytic_second(rows,law,psi)))
            # Same exterior stratum law in both methods, without changing allocation.
            other=tuple(replace(r,context_id='second-outer') for r in rows)
            for axis in ('X','Y'):
                new=sum(float(p)*stats(m.build_instrument(m.qualify_group(rr,law),axis,
                        feed_forward=kind=='rotation'),psi)[0] for p,rr in ((F(3,8),rows),(F(5,8),other)))
                old=sum(float(p)*original_stats(rr,law,psi,axis)[0] for p,rr in ((F(3,8),rows),(F(5,8),other)))
                self.assertAlmostEqual(new,old,places=12)

    def test_wire_bit_y_mapping_feed_forward_and_artificial_scope_guards(self):
        rows,law=fixture();group=m.qualify_group(rows,law);plan=m.build_instrument(group,'Y')
        mappings=[(plan.circuit.find_bit(i.qubits[0]).index,plan.circuit.find_bit(i.clbits[0]).index)
                  for i in plan.circuit.data if i.operation.name=='measure']
        self.assertEqual(mappings,[(0,0),(1,1),(2,2)])
        self.assertEqual([i.operation.name for i in plan.circuit.data][-3:],['sdg','h','measure'])
        for z in range(4):self.assertEqual(plan.decode(z,0),-plan.decode(z,1))
        for z,bit in ((-1,0),(4,0),(True,0),(0,2),(0,True)):
            with self.assertRaises(m.Ineligible):plan.decode(z,bit)
        with self.assertRaisesRegex(m.Ineligible,'UNSUPPORTED_MEASUREMENT'):
            m.build_instrument(group,'X',measurement_mode='hardware-qualified')
        rot=m.qualify_group(tuple(replace(r,kind='rotation',phi=m.ZERO) for r in rows),law)
        with self.assertRaisesRegex(m.Ineligible,'UNSUPPORTED_MEASUREMENT'):
            m.build_instrument(rot,'X')
        self.assertFalse(plan.numerically_certified)
        self.assertFalse(plan.circuit.metadata['hardware_feed_forward_qualified'])
        self.assertEqual(plan.circuit.metadata['numerical_error_bound'],'UNKNOWN')
        n=3;circuits=[QuantumCircuit(n) for _ in range(3)]
        wide=replace(rows[0],A=circuits[0],B=circuits[1],V=circuits[2],system_order=('s0','s1','s2'))
        with self.assertRaisesRegex(m.Ineligible,'ARTIFICIAL_WIDTH_ONLY'):
            m.build_instrument(m.qualify_group((wide,),law),'X')

    def test_adapter_has_no_state_probability_rng_loader_or_compiler(self):
        tree=ast.parse(Path(m.__file__).read_text())
        forbidden={'Statevector','Operator','SavedEngine','from_fixed_bundle','qualify_fixed_bundle',
                   'transpile','random','default_rng','open','read_text','evolve','initialize','sample'}
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertFalse(forbidden&(names|attrs))


if __name__=='__main__':unittest.main()
