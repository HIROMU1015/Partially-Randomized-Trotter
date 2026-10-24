"""Artificial 2/3-qubit enclosures only; no saved H4 values are loaded."""
import copy
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('artificial_arb_bridge', ROOT / 'src/trotterlib/h4_arb_bridge.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
C = B._backend()


def gate(name, *wires, theta=None):
    return dict(name=name, qubits=list(wires), params_exact=[] if theta is None else [str(Q(theta))])


def program(n=2, gates=(), phase=0., axis='cosine'):
    return dict(kind=B.KIND, domain='ARTIFICIAL', fixture_id='TOY_NATIVE', n_qubits=n, n_system=n-1,
        ancilla=n-1, axis=axis, global_phase_exact=str(Q(phase)), instructions=list(gates),
        measurement=dict(qubit=n-1, clbit=0, terminal_only=True))


def evaluate(p, raw, bits=64):
    return B.evaluate_artificial(p, raw, bits=bits, fixture_id='TOY_NATIVE')


def dense_oracle(p, raw):
    """Independent dense-matrix route at 256 bits; old C.embed is reused."""
    with C.ctx.workprec(256):
        n = p['n_qubits']
        v = C.acb_mat([[C.complex_ball(x)] for x in raw] + [[0] for _ in raw])
        for g in p['instructions']:
            name = g['name']
            if name == 'rz':
                a = C.real(Q(g['params_exact'][0])) / 2
                local = C.acb_mat([[C.phase(-a), 0], [0, C.phase(a)]])
            elif name == 'sx':
                d, o = C.acb(C.real('1/2'), C.real('1/2')), C.acb(C.real('1/2'), C.real('-1/2'))
                local = C.acb_mat([[d, o], [o, d]])
            elif name == 'x': local = C.acb_mat([[0, 1], [1, 0]])
            else:
                local = C.acb_mat(4, 4)
                # local bit 0 = control, bit 1 = target.
                for col in range(4): local[col ^ (2 if col & 1 else 0), col] = 1
            v = C.embed(local, g['qubits'], n) * v
        v = v * C.phase(C.real(Q(p['global_phase_exact'])))
        num = C.arb(0)
        for i in range(1 << n):
            z = v[i, 0]
            num += (-1 if i & (1 << p['ancilla']) else 1) * (z.real**2 + z.imag**2)
        nu = sum((Q(x)**2 + Q(y)**2 for x,y in raw), Q(0))
        return num / C.real(nu)


class EnclosureTests(unittest.TestCase):
    def test_exact_binary64_not_decimal_intention(self):
        self.assertEqual(B.exact_binary64(.1), Q.from_float(.1))
        self.assertNotEqual(B.exact_binary64(.1), Q(1, 10))
        with self.assertRaises(ValueError): B.encoded_parameter('1/10')
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError): B.exact_binary64(value)

    def test_known_sx_rz_sx_includes_native_global_phase(self):
        p = program(gates=[gate('sx',1), gate('rz',1,theta=.125), gate('sx',1)])
        signal, record = evaluate(p, [[1.,0.],[0.,0.]])
        with C.ctx.workprec(256): known = -C.real('1/8').cos()
        self.assertTrue(signal.contains(known))
        self.assertFalse(record['formal_certificate_accepted'])

    def test_sx_square_x_and_fourth_identity(self):
        for count, target in ((2,-1),(4,1)):
            signal, _ = evaluate(program(gates=[gate('sx',1)]*count), [[1.,0.],[0.,0.]])
            self.assertTrue(signal.contains(target))

    def test_control_direction_little_endian_nonadjacent_wire(self):
        for system_bit in range(4):
            raw = [[float(i == system_bit),0.] for i in range(4)]
            signal, _ = evaluate(program(3, [gate('cx',0,2)]), raw)
            self.assertTrue(signal.contains(-1 if system_bit & 1 else 1))
            signal, _ = evaluate(program(3, [gate('cx',2,0)]), raw)
            self.assertTrue(signal.contains(1))

    def test_x_permutation_and_raw_nu_no_rescaling(self):
        raw = [[2.,.5],[.25,-.125]]
        saved = copy.deepcopy(raw)
        signal, record = evaluate(program(gates=[gate('x',1)]), raw)
        self.assertTrue(signal.contains(-1))
        self.assertEqual(raw, saved)
        self.assertEqual(Q(record['raw_norm_exact']), sum((Q(x)**2+Q(y)**2 for x,y in raw), Q(0)))
        self.assertFalse(record['clipping'])

    def test_dense_oracle_two_and_three_qubits_all_precisions(self):
        for n in (2,3):
            anc = n-1
            gates = [gate('sx',anc), gate('rz',anc,theta=.1), gate('cx',anc,0), gate('sx',0),
                     gate('rz',0,theta=-.375), gate('cx',0,anc), gate('x',0), gate('sx',anc)]
            raw = [[float(i+1)/8, float(1-i)/16] for i in range(1 << (n-1))]
            p = program(n, gates, phase=.3, axis='sine')
            ref = dense_oracle(p, raw)
            widths = []
            previous = C.ctx.prec
            for bits in B.PRECISIONS:
                signal, record = evaluate(p, raw, bits)
                self.assertTrue(signal.contains(ref), (n,bits,signal,ref))
                widths.append(Q(record['interval_width_exact']))
                self.assertEqual(C.ctx.prec, previous)
            self.assertTrue(widths[0] > widths[1] > widths[2])

    def test_global_phase_retained_without_affecting_z(self):
        p = program(gates=[gate('sx',1),gate('cx',1,0),gate('sx',1)])
        for phase in (0., .7, -2.):
            p['global_phase_exact'] = str(Q(phase))
            signal, record = evaluate(p, [[1.,.25],[-.5,.125]])
            self.assertTrue(signal.contains(dense_oracle(p, [[1.,.25],[-.5,.125]])))
            self.assertTrue(record['global_phase_preserved'])

    def test_saved_and_large_inputs_rejected_before_backend(self):
        cases = [dict(program(), domain='SAVED_QPY_NO_EXECUTION'), dict(program(), n_system=8, n_qubits=9, ancilla=8)]
        with patch.object(B, '_backend', side_effect=AssertionError('MUST_NOT_LOAD_ARB')):
            for p in cases:
                with self.assertRaisesRegex(ValueError, 'NO_REAL_H4'): evaluate(p, [[1.,0.],[0.,0.]])
        with self.assertRaises(PermissionError): B.execution_request()

    def test_invalid_program_and_state_fail_closed(self):
        for g in (gate('h',0), gate('cx',0,0), gate('rz',3,theta=.1)):
            with self.assertRaises(ValueError): evaluate(program(gates=[g]), [[1.,0.],[0.,0.]])
        p = program(); p['measurement']['qubit'] = 0
        with self.assertRaises(ValueError): evaluate(p, [[1.,0.],[0.,0.]])
        with self.assertRaises(ValueError): evaluate(program(), [[0.,0.],[0.,0.]])
        with self.assertRaises(ValueError): evaluate(program(), [[1.,0.]])

    def test_residual_contains_exact_scaled_difference_no_acceptance(self):
        signal, _ = evaluate(program(), [[2.,0.],[0.,0.]])
        rho, record = B.residual_artificial(signal, dict(lower='1/4',upper='1/2'), B_saved_exact='2',
                                          budget_exact='2', fixture_id='TOY_SCALE')
        self.assertTrue(rho.contains(C.real(1)))
        self.assertTrue(rho.contains(C.real('3/2')))
        self.assertEqual(record['budget_relation'], 'UPPER_WITHIN_BUDGET')
        self.assertFalse(record['formal_certificate_accepted'])
        self.assertFalse(record['LOCAL_BUDGET_MET'])


if __name__ == '__main__': unittest.main(verbosity=2)
