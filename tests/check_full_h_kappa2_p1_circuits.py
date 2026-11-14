"""Artificial circuits for P-1, in the same import order as the real child process.

Also the scalar/event phase gate: order-0 and order-2 toys, provider phase rule, and
negative controls (scalar doubled/missing/misplaced, event phase missing/duplicated/
reordered/moved to another request, X/Y mismatch). No H4 input, sampling of H4 or compile.

The original install() puts the pinned provider root first (only its private
config path is read); then a toy 4-qubit DF Hamiltonian goes through the ORIGINAL
prepare_full_h/full_oracle (9b83 blob), the long-time native sample/compose/
cost_trial and the 01e068af swap. compile_cached is a counter: no H4 input,
transpile or price.
"""
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h46_long_time_signal as signal  # noqa: E402
signal.install()
import run_full_h_kappa2_p1 as P1  # noqa: E402  (no project package at import)

import numpy as np  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

TOY_ROW = dict(system='TOY', method='PR', formula='2nd', prefix=0, T=.8, q=2, R_outer=8, K=2, delta=.4, s=16,
               family='FULL_H_PURE_RTE', L_D=0, id='FULL_H_TOY_ARTIFICIAL_P1')
# Same toy at T=1.6: its index-0 path (fixed seed key) has one order-2 event (phase -1) in each
# request plus 2 / 1 sign -1 products, so the provider appends p(3*pi) and p(2*pi) per request
# (event and product phases accumulated) next to the scalar phase.
ORDER2_ROW = dict(TOY_ROW, T=1.6, delta=.8, id='FULL_H_TOY_ARTIFICIAL_P1_ORDER2_9')


def toy_hamiltonian():
    from trotterlib.df_hamiltonian import DFHamiltonian
    rng = np.random.default_rng(11)
    a = rng.normal(size=(4, 4)); g = []
    for _ in range(2):
        b = rng.normal(size=(4, 4)); g.append(.25 * (b + b.T))
    return DFHamiltonian(.37, .3 * (a + a.T), np.array([.4, .3]), tuple(g), {})


def toy_state(n=4, particles=2):
    v = np.zeros(1 << n, complex); rng = np.random.default_rng(5)
    for i in range(1 << n):
        if bin(i).count('1') == particles:
            v[i] = rng.normal() + 1j * rng.normal()
    return v / np.linalg.norm(v)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='p1_artificial_')); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.full = P1.full_h_module(self.tmp / 'public', self.tmp / 'private')
        ham = toy_hamiltonian(); prep = self.full.prepare_full_h(ham)
        oracle = self.full.full_oracle(ham, toy_state())
        qstate = np.zeros(1 << ham.n_qubits, complex); qstate[oracle.qindices] = oracle.initial
        self.ctx = SimpleNamespace(system='TOY', ham=ham, preparations={0: prep}, oracle=oracle, qstate=qstate)
        self.full.context = lambda system: self.ctx
        self.compiles = []

        def fake_compile(wrapper, fingerprint, representative):
            self.compiles.append(fingerprint)
            return dict(metrics={k: 1 for k in ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')}), None, False
        self.fake_compile = fake_compile

    def evolutions(self):
        native = self.full.setup()
        prep = self.ctx.preparations[0]
        requests, _, _ = native.sample_requests(TOY_ROW, prep, P1.COHORT, 0)
        scalar = self.ctx.ham.constant + prep.extracted_identity_coefficient
        legacy = native.compose(TOY_ROW, prep, requests, scalar, lambda qc: 'x')
        provider = __import__('trotterlib.df_rte_qiskit', fromlist=['x'])
        with P1.zz_module().efficient_controlled_zz(provider):
            new = native.compose(TOY_ROW, prep, requests, scalar, lambda qc: 'x')
        return native, prep, requests, scalar, legacy, new

    def test_provider_root_modules_are_pinned(self):
        files = P1.provider_checks()
        self.assertTrue(all('h46-higher-pf-acquire' in v['file'] or 'provider' in v['file'] or 'src' in v['file']
                            for v in files.values()))

    def test_toy_pilot_end_to_end_without_compile(self):
        trial, checks, ok = P1.pilot(self.full, TOY_ROW, compile_cached=self.fake_compile)
        self.assertTrue(ok, checks)
        self.assertEqual(len(self.compiles), 2)
        self.assertEqual(set(checks['trial_gates']), {'low_control', 'high_control_leakage',
                         'independent_occupation_vector', 'Hadamard_cosine', 'Hadamard_sine'})
        self.assertGreater(checks['parity']['replaced_controlled_zz'], 0)
        self.assertEqual(checks['legacy_rzz_control_instructions'], 0)
        self.assertGreater(checks['one_body_applications'], 0)
        self.assertEqual(checks['scalar_event_phases']['status'], 'PASS')
        self.assertTrue(checks['scalar_once_legacy_diagnostic'])  # order-0 toy: no event phase

    def test_controlled_action_low_identity_scalar_once(self):
        native, prep, requests, scalar, legacy, new = self.evolutions()
        U_old, U_new = Operator(legacy.circuit).data, Operator(new.circuit).data
        np.testing.assert_allclose(U_new, U_old, atol=1e-12, rtol=0)  # incl. global/control phase
        d = 1 << prep.num_system_qubits
        np.testing.assert_allclose(U_new[:d, :d], np.eye(d), atol=1e-12)  # ancilla 0 block
        self.assertLess(np.abs(U_new[:d, d:]).max(), 1e-12)
        np.testing.assert_allclose(U_new[d:, d:] @ self.ctx.qstate, self.ctx.oracle.trajectory(requests), atol=1e-10)
        n = prep.num_system_qubits
        self.assertTrue(P1.scalar_once(new.circuit, n, -scalar * TOY_ROW['T']))
        doubled = new.circuit.copy(); doubled.p(-scalar * TOY_ROW['T'], n)
        self.assertFalse(P1.scalar_once(doubled, n, -scalar * TOY_ROW['T']))
        self.assertGreater(np.abs(Operator(doubled).data - U_new).max(), 1e-3)
        self.assertEqual(P1.op_counts(new.circuit).get('crzz', 0), 0)
        self.assertGreater(P1.op_counts(legacy.circuit).get('crzz', 0), 0)
        self.assertEqual(P1.zz_module().instruction_parity(legacy.circuit, new.circuit)['status'], 'PASS')
        one = [c for c in prep.rte_preparation.symbolic_tail.components if c.df_fragment_id == 'full-h-one-body']
        self.assertTrue(one and all(len(c.diagonal_pauli_support) == 1 for c in one))

    def test_xy_measurement_sign(self):
        native = self.full.setup()
        prep = self.ctx.preparations[0]
        requests, _, _ = native.sample_requests(TOY_ROW, prep, P1.COHORT, 0)
        scalar = self.ctx.ham.constant + prep.extracted_identity_coefficient
        from trottertracks.resource_applicability import ax2b_native_df_v5 as nat
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        from trotterlib.df_trotter.circuit import simulate_statevector
        provider = __import__('trotterlib.df_rte_qiskit', fromlist=['x'])
        z = complex(np.vdot(self.ctx.qstate, self.ctx.oracle.trajectory(requests)))
        low = np.r_[self.ctx.qstate, np.zeros_like(self.ctx.qstate)]; d = len(self.ctx.qstate)
        with use_fingerprint_reuse() as fp, P1.zz_module().efficient_controlled_zz(provider):
            new = native.compose(TOY_ROW, prep, requests, scalar, fp.fingerprint)
            for axis, value in (('cosine', z.real), ('sine', z.imag)):
                w = nat.build_native_hadamard_wrapper(new, axis=axis, include_measurement=True,
                                                      max_instructions=len(new.circuit.data) + 4)
                self.assertEqual(w.data[-1].operation.name, 'measure')
                v = simulate_statevector(w.remove_final_measurements(inplace=False), low)
                self.assertAlmostEqual(float(np.vdot(v[:d], v[:d]).real - np.vdot(v[d:], v[d:]).real), value, delta=1e-10)

    # ---- scalar/event phase gate (replaces the legacy "exactly one ancilla p" scalar_once)

    def phase_case(self, row):
        native = self.full.setup(); prep = self.ctx.preparations[0]
        requests, _, _ = native.sample_requests(row, prep, P1.COHORT, 0)
        scalar = self.ctx.ham.constant + prep.extracted_identity_coefficient
        from trottertracks.resource_applicability import ax2b_native_df_v5 as nat
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        provider = __import__('trotterlib.df_rte_qiskit', fromlist=['x'])
        with use_fingerprint_reuse() as fp, P1.zz_module().efficient_controlled_zz(provider):
            new = native.compose(row, prep, requests, scalar, fp.fingerprint)
            wrappers = {a: nat.build_native_hadamard_wrapper(new, axis=a, include_measurement=True,
                                                             max_instructions=len(new.circuit.data) + 4) for a in ('cosine', 'sine')}
        events = [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
        return SimpleNamespace(circuit=new.circuit, wrappers=wrappers, events=events, scalar=scalar, T=row['T'],
                               n=prep.num_system_qubits, provider=provider, requests=requests)

    def gate(self, c, circuit=None, **kw):
        return P1.scalar_event_phase_gate(c.circuit if circuit is None else circuit, c.n, kw.pop('events', c.events),
                                          c.scalar, kw.pop('expected', c.scalar), c.T, provider=c.provider, **kw)

    @staticmethod
    def rebuilt(circuit, edit):
        items = list(circuit.data); edit(items)
        out = circuit.copy_empty_like()
        for item in items:
            out.append(item.operation, item.qubits, item.clbits)
        return out

    def ancilla_p(self, c, circuit=None):
        circuit = c.circuit if circuit is None else circuit
        return [i for i, item in enumerate(circuit.data) if item.operation.name == 'p'
                and [circuit.find_bit(q).index for q in item.qubits] == [c.n]]

    def test_phase_gate_order0_toy_passes(self):
        c = self.phase_case(TOY_ROW)
        g = self.gate(c)
        self.assertEqual((g['status'], g['event_phases'], g['ancilla_p']), ('PASS', [], 1), g)
        self.assertTrue(g['legacy_scalar_once'])
        self.assertEqual(P1.xy_phase_gate(c.wrappers, c.n, c.events, c.scalar, c.scalar, c.T, provider=c.provider)['status'], 'PASS')

    def test_phase_gate_order2_event_and_scalar_phases_coexist(self):
        c = self.phase_case(ORDER2_ROW)
        orders = [[e['taylor_order'] for e in batch] for batch in c.events]
        self.assertEqual([o.count(2) for o in orders], [1, 1])
        neg = [sum(a['role'] == 'product' and a['coefficient_sign'] == -1 for e in b for a in e['application_sequence'])
               for b in c.events]
        self.assertEqual(neg, [2, 1])
        g = self.gate(c)
        self.assertEqual(g['status'], 'PASS', g)
        self.assertEqual([(p['request'], round(p['angle'] / np.pi, 12)) for p in g['event_phases']], [(0, 3.0), (1, 2.0)])
        self.assertEqual([p['angle'] for p in g['event_phases']], [P1.request_phase(b, c.provider) for b in c.events])
        self.assertEqual([p['minus_one_events'] != [] for p in g['event_phases']], [True, True])
        self.assertEqual(g['ancilla_p'], 3)
        self.assertFalse(g['legacy_scalar_once'])                 # the run02 false negative, reproduced
        self.assertFalse(P1.scalar_once(c.circuit, c.n, -c.scalar * c.T))
        p = self.ancilla_p(c)                                      # scalar is the last instruction and last ancilla p
        self.assertEqual(p[-1], len(c.circuit.data) - 1)
        self.assertEqual(float(c.circuit.data[p[-1]].operation.params[0]), -c.scalar * c.T)
        xy = P1.xy_phase_gate(c.wrappers, c.n, c.events, c.scalar, c.scalar, c.T, provider=c.provider)
        self.assertEqual((xy['status'], xy['same_controlled_evolution']), ('PASS', True), xy)

    def test_central_phase_matches_provider_return(self):
        from qiskit import QuantumCircuit
        c = self.phase_case(ORDER2_ROW)
        builder = c.provider.QiskitDFRTEEventCircuitBuilder
        seen = set()
        for r in c.requests:
            for event in r.rte_occurrence.events:
                for app in event.application_sequence:
                    if app.is_identity:
                        continue
                    got = builder._append_central_application(QuantumCircuit(c.n + 1), event, app, controlled=True, ancilla_qubit=c.n)
                    view = P1._event_view(event.to_dict()).apps[app.application_index]
                    self.assertEqual(P1.central_phase(view), got); seen.add((app.role, app.coefficient_sign))
        self.assertTrue({'rotation', 'product'} <= {role for role, _ in seen}, seen)
        for sign in (1, -1):                                       # product sign rule incl. the -1 branch
            fake = SimpleNamespace(role='product', coefficient_sign=sign, diagonal_pauli_support=(0,))
            self.assertEqual(P1.central_phase(fake), builder._append_central_application(
                QuantumCircuit(c.n + 1), None, fake, controlled=True, ancilla_qubit=c.n))

    def test_phase_gate_rejects_scalar_errors(self):
        c = self.phase_case(ORDER2_ROW)
        last = len(c.circuit.data) - 1
        doubled = c.circuit.copy(); doubled.p(-c.scalar * c.T, c.n)
        missing = self.rebuilt(c.circuit, lambda items: items.pop())
        early = self.rebuilt(c.circuit, lambda items: items.insert(self.ancilla_p(c)[0], items.pop()))
        late_event = self.rebuilt(c.circuit, lambda items: items.insert(last, items.pop(self.ancilla_p(c)[1])))
        for name, circuit in dict(doubled=doubled, missing=missing, before_event_phase=early, event_after_scalar=late_event).items():
            self.assertEqual(self.gate(c, circuit)['status'], 'FAIL', name)
        self.assertEqual(self.gate(c, expected=c.scalar + 1e-12)['reasons'], ['SCALAR_VALUE_DIFFERS_FROM_PREPARATION'])

    def test_phase_gate_rejects_event_phase_errors(self):
        c = self.phase_case(ORDER2_ROW)
        p0, p1, ps = self.ancilla_p(c)
        crz = [i for i, item in enumerate(c.circuit.data) if item.operation.name == 'crz']
        first_crz_r1 = crz[len(c.events[0])]
        dup = self.rebuilt(c.circuit, lambda items: items.insert(p0, items[p0]))
        drop = self.rebuilt(c.circuit, lambda items: items.pop(p0))
        moved = self.rebuilt(c.circuit, lambda items: items.insert(first_crz_r1 + 1, items.pop(p0)))  # into request 1
        def swap(items):
            items[p0], items[p1] = items[p1], items[p0]
        swapped = self.rebuilt(c.circuit, swap)                     # 3*pi <-> 2*pi between requests
        for name, circuit in dict(duplicate=dup, missing=drop, other_request=moved, swapped_order=swapped).items():
            g = self.gate(c, circuit)
            self.assertEqual((g['status'], g['reasons']), ('FAIL', ['ANCILLA_SIGNATURE_DIFFERS']), name)
        flipped = [[dict(e, phase=dict(real=1.0, imag=0.0)) for e in batch] for batch in c.events]
        self.assertEqual(self.gate(c, events=flipped)['status'], 'FAIL')   # event record without its sign
        other = [c.events[1], c.events[0]]
        self.assertEqual(self.gate(c, events=other)['status'], 'FAIL')     # request/event correspondence

    def test_xy_gate_rejects_axis_mismatch(self):
        c = self.phase_case(ORDER2_ROW)
        w = dict(c.wrappers)
        start, stop = P1.wrapper_evolution_bounds(w['sine'], c.n, 'sine')
        p = [i for i in range(start, stop) if w['sine'].data[i].operation.name == 'p'
             and [w['sine'].find_bit(q).index for q in w['sine'].data[i].qubits] == [c.n]]
        w['sine'] = self.rebuilt(w['sine'], lambda items: items.pop(p[0]))
        g = P1.xy_phase_gate(w, c.n, c.events, c.scalar, c.scalar, c.T, provider=c.provider)
        self.assertEqual((g['status'], g['same_controlled_evolution'], g['axes']['cosine']['status']), ('FAIL', False, 'PASS'))
        w2 = dict(c.wrappers)                                       # same phases, one system gate differs
        k = next(i for i in range(start, stop) if w2['sine'].data[i].operation.name == 'xx_plus_yy')
        def bend(items):
            op = items[k].operation; items[k] = items[k].replace(operation=type(op)(op.params[0] + 1e-3, op.params[1]))
        w2['sine'] = self.rebuilt(w2['sine'], bend)
        g2 = P1.xy_phase_gate(w2, c.n, c.events, c.scalar, c.scalar, c.T, provider=c.provider)
        self.assertEqual((g2['status'], g2['same_controlled_evolution']), ('FAIL', False))

    def test_toy_pilot_with_order2_events_keeps_all_existing_gates(self):
        trial, checks, ok = P1.pilot(self.full, ORDER2_ROW, compile_cached=self.fake_compile)
        self.assertTrue(ok, checks)
        self.assertEqual(set(checks['trial_gates']), {'low_control', 'high_control_leakage',
                         'independent_occupation_vector', 'Hadamard_cosine', 'Hadamard_sine'})
        self.assertEqual((checks['parity']['status'], checks['legacy_rzz_control_instructions']), ('PASS', 0))
        self.assertGreater(checks['one_body_applications'], 0)
        self.assertEqual(checks['scalar_event_phases']['status'], 'PASS')
        self.assertFalse(checks['scalar_once_legacy_diagnostic'])
        self.assertEqual(len(checks['scalar_event_phases']['event_phases']), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
