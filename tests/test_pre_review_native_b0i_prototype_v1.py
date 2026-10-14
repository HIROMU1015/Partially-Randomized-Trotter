"""Tiny artificial full-Fock matrices and mock wrappers; no native imports."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('b0i_prototype',
    Path(__file__).resolve().parents[1]/'scripts/pre_review_native_b0i_prototype_v1.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture(control='symmetric_directional'):
    binding = dict(source='a'*40, hamiltonian='b'*64, df='c'*64, state='d'*64,
        input='e'*64, compiler='f'*64, b0_fingerprint='0'*64, fragment_order=['A', 'B', 'C'])
    base = dict(method='B0', binding=copy.deepcopy(binding), scalar=3.0,
        scalar_semantics='ORIGINAL_H_CONSTANT_ONLY', extracted_tail_identity=0.0,
        tail_identity_added=False, retained_fragment_ids=['A'],
        deterministic_blocks=['one_body', 'retained_A'], control=control,
        retained_internal_global_phases=[.125, -.25])
    fragments = [dict(id=x, G=((1, 0), (0, 2)), **{'lambda': v})
        for x, v in [('A', 7.0), ('B', -2.0), ('C', .5)]]
    return base, fragments, binding


def full_fock_identity(g, lam):
    # Explicit dGamma(G) for <=3 artificial modes; fermionic parity is independent
    # of the closed trace formula used by the prototype.
    n = len(g); size = 1 << n
    operator = [[0j]*size for _ in range(size)]
    for ket in range(size):
        for j in range(n):
            if not ket >> j & 1:
                continue
            middle = ket ^ (1 << j)
            sign_j = (-1)**((ket & ((1 << j)-1)).bit_count())
            for i in range(n):
                if middle >> i & 1:
                    continue
                bra = middle | (1 << i)
                sign_i = (-1)**((middle & ((1 << i)-1)).bit_count())
                operator[bra][ket] += sign_j*sign_i*g[i][j]
    return lam*sum(operator[i][j]*operator[j][i] for i in range(size) for j in range(size))/size


class NativeB0IPrototypeTests(unittest.TestCase):
    def test_full_fock_trace_independent_complex_three_mode(self):
        g = ((1, 2j, .5), (-2j, -2, 1-1j), (.5, 1+1j, 3))
        self.assertAlmostEqual(p.fragment_identity(g, -.75), full_fock_identity(g, -.75).real)
        self.assertAlmostEqual(full_fock_identity(g, -.75).imag, 0)

    def test_signed_identity_not_sector_trace(self):
        self.assertEqual(p.fragment_identity(((1, 0), (0, 2)), 2), 7)
        self.assertEqual(p.fragment_identity(((1, 0), (0, 2)), -2), -7)
        # Single-electron sector trace is (1^2+2^2)/2 != full-Fock 3.5.
        self.assertNotEqual(p.fragment_identity(((1, 0), (0, 2)), 1), 2.5)

    def test_tail_prefix_and_original_b0_immutable(self):
        b, f, identity = fixture(); original = copy.deepcopy(b)
        out = p.overlay_b0(b, f, prefix=1, total_time=.8, binding=identity)
        self.assertEqual(out['extracted_tail_identity'], -5.25)
        self.assertEqual(out['scalar'], -2.25)
        self.assertEqual(out['deterministic_blocks'], b['deterministic_blocks'])
        self.assertEqual(out['retained_internal_global_phases'], b['retained_internal_global_phases'])
        self.assertEqual(b, original)
        out['deterministic_blocks'].append('changed_proposal')
        self.assertEqual(b, original)

    def test_empty_tail_keeps_constant(self):
        b, f, identity = fixture(); b['retained_fragment_ids'] = identity['fragment_order'][:]
        out = p.overlay_b0(b, f, prefix=3, total_time=.8, binding=identity)
        self.assertEqual(out['scalar'], 3)
        self.assertEqual(out['added_relative_phase'], 0)

    def test_inconsistent_fragment_dimension_rejected(self):
        b, f, identity = fixture(); f[1]['G'] = ((1,),)
        with self.assertRaisesRegex(ValueError, 'INCONSISTENT_MODE_DIMENSION'):
            p.overlay_b0(b, f, prefix=1, total_time=.8, binding=identity)

    def test_double_overlay_and_ambiguous_scalar_rejected(self):
        b, f, identity = fixture()
        out = p.overlay_b0(b, f, prefix=1, total_time=.8, binding=identity)
        with self.assertRaisesRegex(ValueError, 'ALREADY_OVERLAID'):
            p.overlay_b0(out, f, prefix=1, total_time=.8, binding=identity)
        b['extracted_tail_identity'] = 1
        with self.assertRaises(ValueError):
            p.overlay_b0(b, f, prefix=1, total_time=.8, binding=identity)

    def test_phase_and_axes_direct_ancilla_amplitudes(self):
        phase = p.rotate_saved_signal(1, total_time=.8, tail_identity=.7)
        amplitudes = (1/math.sqrt(2), phase/math.sqrt(2))
        for axis in ('cosine', 'sine'):
            a, b = amplitudes
            if axis == 'sine':
                b *= -1j  # Sdg before final H.
            zero, one = (a+b)/math.sqrt(2), (a-b)/math.sqrt(2)
            expected = phase.real if axis == 'cosine' else phase.imag
            self.assertAlmostEqual(abs(zero)**2-abs(one)**2, expected)

    def test_both_controls_apply_relative_phase_once(self):
        for control in ('ordinary', 'symmetric_directional'):
            b, f, identity = fixture(control)
            out = p.overlay_b0(b, f, prefix=1, total_time=.8, binding=identity)
            d = p.wrapper_descriptor(out, 'sine')
            self.assertEqual(d['native_final_phase_angle'], -out['scalar']*.8)
            self.assertEqual(d['scalar_phase_target'], 'ANCILLA_RELATIVE_PHASE')
            self.assertEqual(d['after_evolution'], ['Sdg(ancilla)'])
            self.assertFalse(d['runtime_authorization'])

    def test_axis_fingerprint_and_binding_change(self):
        b, f, identity = fixture()
        out = p.overlay_b0(b, f, prefix=1, total_time=.8, binding=identity)
        self.assertNotEqual(p.digest(p.wrapper_descriptor(out, 'cosine')),
            p.digest(p.wrapper_descriptor(out, 'sine')))
        bad = copy.deepcopy(identity); bad['input'] = '1'*64
        with self.assertRaisesRegex(ValueError, 'SOURCE_INPUT_BINDING'):
            p.overlay_b0(b, f, prefix=1, total_time=.8, binding=bad)
        with self.assertRaises(ValueError):
            p.overlay_b0(b, f[::-1], prefix=1, total_time=.8, binding=identity)

    def test_hermiticity_and_every_nonfinite_entry_rejected(self):
        for g in (((1, 2), (1, 2)), ((1+1j, 0), (0, 2)),
                ((1, float('nan')), (0, 2)), ((float('inf'), 0), (0, 2))):
            with self.assertRaises(ValueError):
                p.fragment_identity(g, 1)
        for x in (True, float('nan'), float('inf'), 10**1000):
            with self.assertRaises(ValueError):
                p.fragment_identity(((1, 0), (0, 2)), x)
        with self.assertRaises(ValueError):
            p.fragment_identity(((1e308, 0), (0, 2)), 1)

    def test_u_rotation_and_scalar_uncertainty_are_explicit(self):
        out = p.rotated_allowance(1, 2, angle=math.pi/2, signal_abs_upper=1,
            phase_error_upper=.1, rotation_roundoff_upper=.01)
        self.assertAlmostEqual(out['real'], 2.11)
        self.assertAlmostEqual(out['imag'], 1.11)
        self.assertFalse(out['certificate_created'])
        with self.assertRaises(ValueError):
            p.rotated_allowance(1, 2, angle=0, signal_abs_upper=1,
                phase_error_upper=-1, rotation_roundoff_upper=0)


if __name__ == '__main__':
    unittest.main()
