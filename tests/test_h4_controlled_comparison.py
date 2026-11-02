"""Artificial SOURCE semantics checks only; no H4 data, compile or accounting."""
import copy
from contextlib import ExitStack
import importlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h4_d1_adapter import A, event
from d1_private import h4_d2_pauli_control as D
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli
import numpy as np


def lift(u):
    result = np.eye(2*len(u), dtype=complex)
    result[len(u):, len(u):] = u
    return result


def rotation(label, angle):
    return math.cos(angle)*np.eye(4)-1j*math.sin(angle)*Pauli(label).to_matrix()


def record(k, *, cancel=False, identity_product=False):
    saved = event(k)
    labels = {str(j):label for j,label in enumerate(('IX','IY','YZ','ZX','XY'))}
    if cancel:
        labels.update({'0':'IX','1':'IY','2':'IX','3':'IY','4':'IZ'})
    if identity_product:
        labels['0'] = 'II'
    apps = []
    for j in range(k+1):
        app = copy.deepcopy(saved['application_sequence'][0])
        app.update(application_index=j, role='rotation' if j==k else 'product',
                   component_id=str(j), coefficient_sign=-1 if j%2==0 else 1,
                   is_identity=identity_product and j==0)
        apps.append(app)
    saved.update(rotation_component_id=str(k), product_component_ids=list(labels)[:k],
                 selected_component_ids=[str(j) for j in range(k+1)], application_sequence=apps)
    return A.event_from_record(saved), labels


def event_oracle(e, labels):
    u = np.eye(4, dtype=complex)
    for app in e.application_sequence:
        p = labels[app.component_id]
        op = app.coefficient_sign*Pauli(p).to_matrix() if app.role=='product' else rotation(p, app.coefficient_sign*e.rotation_angle)
        u = op@u
    return e.phase*u


class SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        for module, name in [('rte','sample_rte_events'), ('rte','_make_event'),
                             ('rte_compiled_cost','transpile'), ('h4_d1_adapter','build'),
                             ('h4_d1_adapter','compile_saved'), ('h4_conditional_cng','shot')]:
            self.stack.enter_context(patch.object(importlib.import_module('d1_private.'+module), name,
                                                  side_effect=AssertionError('FORBIDDEN:'+name)))

    def tearDown(self):
        self.stack.close()

    def same(self, actual, expected):
        self.assertLess(float(np.max(np.abs(actual-expected))), 2e-12)

    def different(self, actual, expected):
        self.assertGreater(float(np.max(np.abs(actual-expected))), 1e-5)

    def test_projector_lift_conjugation_and_product_identities(self):
        b = np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2), np.diag([1,1j]))
        d = rotation('ZZ', .23)
        ib = np.kron(np.eye(2), b)
        self.same(lift(b.conj().T@d@b), ib.conj().T@lift(d)@ib)
        u, v = rotation('XY', .31), rotation('YZ', -.17)
        self.same(lift(v@u), lift(v)@lift(u))
        self.same(lift(np.exp(.37j)*u), np.kron(np.diag([1,np.exp(.37j)]),np.eye(4))@lift(u))

    def test_distinct_noncommuting_signed_products_order2_and4(self):
        for k in (2,4):
            e, labels = record(k)
            body = QuantumCircuit(2)
            A.api().append_pauli_event(body, e, labels)
            body.global_phase += .37
            result = D.central_control_body(body)
            expected = np.exp(.37j)*event_oracle(e, labels)
            self.same(Operator(result.body).data, lift(expected))
            self.same(Operator(result.body).data[:4,:4], np.eye(4))

    def test_product_can_carry_imaginary_phase_without_being_rotation(self):
        body = QuantumCircuit(2)
        body.x(0);body.y(0)
        expected = Pauli('IY').to_matrix()@Pauli('IX').to_matrix()
        self.same(Operator(D.central_control_body(body).body).data, lift(expected))
        self.different(lift(expected), lift(Pauli('IZ').to_matrix()))

    def test_order4_whole_product_minus_identity_is_preserved(self):
        e, labels = record(4, cancel=True)
        body = QuantumCircuit(2)
        A.api().append_pauli_event(body, e, labels)
        self.same(Operator(D.central_control_body(body).body).data, lift(event_oracle(e, labels)))

    def test_identity_product_sign_is_scalar_not_dropped(self):
        e, labels = record(2, identity_product=True)
        body = QuantumCircuit(2)
        A.api().append_pauli_event(body, e, labels)
        self.same(Operator(D.central_control_body(body).body).data, lift(event_oracle(e, labels)))

    def test_full_pf_pr_noncommuting_schedule_constant_and_both_axes(self):
        a = A.api()
        h = np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2),np.eye(4))
        sdg = np.kron(np.diag([1,-1j]),np.eye(4))
        for method in ('PF','PR'):
            body = QuantumCircuit(2);u = np.eye(4,dtype=complex)
            for k in (2,4):
                for label,c,t in [('XY',.7,.11),('YZ',-.4,.11)]:
                    a.append_pauli_evolution(body,label,c,t);u=rotation(label,c*t)@u
                if method=='PR':
                    e,labels=record(k);a.append_pauli_event(body,e,labels);u=event_oracle(e,labels)@u
                for label,c,t in [('YZ',-.4,.11),('XY',.7,.11)]:
                    a.append_pauli_evolution(body,label,c,t);u=rotation(label,c*t)@u
                body.global_phase -= .37*.22;u=np.exp(-1j*.37*.22)*u
            controlled=D.central_control_body(body).body
            for axis in ('cosine','sine'):
                wrapped=a.hadamard_wrapper(controlled,axis,measure=False,already_controlled=True)
                expected=h@(sdg if axis=='sine' else np.eye(8))@lift(u)@h
                self.same(Operator(wrapped).data,expected)

    def test_scalar_lift_and_identity_evolution_domain_rejection(self):
        for phase in (math.pi, .5*math.pi, -.41):
            body=QuantumCircuit(2);body.global_phase=phase
            self.same(Operator(D.central_control_body(body).body).data,lift(np.exp(1j*phase)*np.eye(4)))
        body=QuantumCircuit(2);A.api().append_pauli_evolution(body,'II',-.7,.21)
        with self.assertRaisesRegex(ValueError,'IDENTITY_EVOLUTION'):D.central_control_body(body)

    def test_dropping_products_phase_or_order_is_detected(self):
        e,labels=record(2);expected=lift(event_oracle(e,labels))
        wrong=QuantumCircuit(2)
        app=e.application_sequence[-1]
        A.api().append_pauli_evolution(wrong,labels[app.component_id],app.coefficient_sign,e.rotation_angle)
        self.different(Operator(D.central_control_body(wrong).body).data,expected)
        correct=QuantumCircuit(2);A.api().append_pauli_event(correct,e,labels)
        wrong=correct.copy();wrong.global_phase+=math.pi
        self.different(Operator(D.central_control_body(wrong).body).data,expected)
        wrong=QuantumCircuit(2)
        # Reverse two distinct noncommuting products; leave all signs/phases.
        for app in reversed(e.application_sequence[:-1]):
            for q,axis in enumerate(reversed(labels[app.component_id])):
                if axis!='I':getattr(wrong,axis.lower())(q)
            if app.coefficient_sign<0:wrong.global_phase+=math.pi
        app=e.application_sequence[-1]
        A.api().append_pauli_evolution(wrong,labels[app.component_id],app.coefficient_sign,e.rotation_angle)
        wrong.global_phase+=float(np.angle(e.phase))
        self.different(Operator(D.central_control_body(wrong).body).data,expected)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SemanticsTests))
    if len(sys.argv)==2:
        Path(sys.argv[1]).write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            scope='ARTIFICIAL_2_SYSTEM_QUBITS_ONLY_NO_H4_READ_OR_COMPILE',new_science_grants=0,real_H4_calls=0),indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
