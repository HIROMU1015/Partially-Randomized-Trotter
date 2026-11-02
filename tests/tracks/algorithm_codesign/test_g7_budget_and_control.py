"""Off-domain G7 proofs/semantics tests; no synthesis or registered cost score."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from trottertracks.algorithm_codesign.g7_generator import (ARMS, make_generator,
    raw_reduced_masses, upper_normalizer, quartic_coefficients, log640_upper,
    budget, DeterministicBits, root_interval, ReturnKernel)
from trottertracks.algorithm_codesign.g7_reference import reference_events, resource_reference
from trottertracks.algorithm_codesign.g7_provider import controlled_event_ir

P, X = (F(2, 7), F(5, 7)), F(3, 8)


def independent_reduce(word):
    word = list(word)
    while True:
        j = next((j for j in range(len(word) - 1) if word[j] == word[j + 1]), None)
        if j is None:
            return tuple(word)
        del word[j:j + 2]


def exact_root(v):
    from math import isqrt
    a, b = isqrt(v.numerator), isqrt(v.denominator)
    if a * a != v.numerator or b * b != v.denominator:
        raise AssertionError('expected perfect rational square')
    return F(a, b)


class BudgetAndControl(unittest.TestCase):
    def test_raw_reduced_recursion_independent_oracle(self):
        masses = raw_reduced_masses(P, 7)
        for n in range(8):
            actual = F(0)
            for word in product(range(2), repeat=n):
                weight = F(1)
                for i in word:
                    weight *= P[i]
                if all(word[j] != word[j + 1] for j in range(len(word) - 1)):
                    actual += weight
            self.assertEqual(actual, masses[n])

    def test_U_contains_normalizer_without_budget_oracle(self):
        for m in (1, 3, 5, 7):
            k = ReturnKernel(P, X, m); U = upper_normalizer(k, 256)
            lo, hi = F(0), F(0)
            for l in range(0, m, 2):
                for word in product(range(2), repeat=l):
                    if independent_reduce(word) == word:
                        root = root_interval(k.parent(word).d2, 256)
                        lo += root.lo; hi += root.hi
            ordinary_lo = sum(root_interval(k.t(l) ** 2 + k.t(l + 1) ** 2, 256).lo
                              for l in range(0, m, 2))
            self.assertLessEqual(lo, U.hi)
            if m == 1:
                self.assertEqual(U.lo, lo); self.assertEqual(U.hi, hi)
            else:
                self.assertLess(hi, U.lo)
                self.assertLess(U.hi, ordinary_lo)

    def test_m3_strict_chain(self):
        full = make_generator(P, X, 3, 'full_return')
        partial = make_generator(P, X, 3, 'partial_return_tail')
        closed = make_generator(P, X, 3, 'closed_P3_tail')
        ordinary = make_generator(P, X, 3, 'ordinary')
        self.assertLess(closed.B.hi, full.U.lo)
        self.assertLess(full.U.hi, partial.B.lo)
        self.assertLess(partial.B.hi, ordinary.B.lo)

    def test_quartic_independent_root_expansion(self):
        result = quartic_coefficients(P)
        chi, mu3, mu4 = (sum(pi ** k for pi in P) for k in (2, 3, 4))
        self.assertEqual(result['ordinary'], (F(1), -F(7, 72)))
        self.assertEqual(result['partial'], (1 - chi, -F(7, 72) + chi / 18))
        self.assertEqual(result['full'], (1 - chi, -F(7, 72) - chi / 6 + mu3 / 4 - mu4 / 36))
        self.assertEqual(result['partial_minus_full_x4'], sum(pi * pi * (1 - pi) * (8 - pi) for pi in P) / 36)

    def test_all_arms_same_finite_mean_independent_formal_polynomial(self):
        m = 5
        p = (F(1, 7), F(2, 7), F(4, 7))
        from collections import defaultdict
        from math import factorial
        expected = defaultdict(lambda: [F(0), F(0)])
        def add(table, word, phase, mass):
            axis, sign = ((0, 1), (1, 1), (0, -1), (1, -1))[phase % 4]
            table[independent_reduce(word)][axis] += sign * mass
        for n in range(m + 1):
            for word in product(range(len(p)), repeat=n):
                mass = X ** n / factorial(n)
                for i in word:
                    mass *= p[i]
                add(expected, word, -n, mass)
        for arm in ARMS:
            gen = make_generator(p, X, m, arm)
            actual = defaultdict(lambda: [F(0), F(0)])
            if arm == 'full_return':
                for l in range(0, m, 2):
                    for word in product(range(len(p)), repeat=l):
                        if independent_reduce(word) != word:
                            continue
                        par = gen.kernel.parent(word)
                        add(actual, word, l, par.a)
                        for i, mass in zip(par.children, par.masses):
                            add(actual, (i,) + word, l - 1, mass)
            else:
                for index, g in enumerate(gen.groups):
                    a = exact_root(g.square / (1 + g.ratio * g.ratio))
                    for event in reference_events(gen):
                        if event['ratio'] != g.ratio or event['phase_i_power'] != g.degree % 4:
                            continue
                        # Disambiguate by regenerating exactly this group.
                        try:
                            own = gen.event(index, event['raw_word'], event['child'])
                        except (ValueError, IndexError):
                            continue
                        if own != event:
                            continue
                        label = event['coefficient'] / gen.roots[index].midpoint
                        add(actual, event['raw_word'], g.degree, a * label)
                        add(actual, (event['child'],) + event['raw_word'], g.degree - 1, a * g.ratio * label)
            self.assertEqual({k: v for k, v in actual.items() if any(v)},
                             {k: v for k, v in expected.items() if any(v)}, arm)

    def test_digital_moment_range_and_support(self):
        for arm in ARMS:
            gen = make_generator(P, X, 5, arm); events = list(reference_events(gen)); plan = budget(gen)
            self.assertLessEqual(sum(e['proposal'] * e['weight'] ** 2 for e in events), plan['m2_upper'])
            self.assertLessEqual(max(e['weight'] for e in events), plan['range_upper'])
            self.assertTrue(all(e['proposal'] > 0 for e in events))
            self.assertTrue(all(e['proposal'] * e['weight'] == e['coefficient'] for e in events))

    def test_common_bias_margin_and_familywise_budget(self):
        plans = [budget(make_generator(P, X, 5, arm)) for arm in ARMS]
        self.assertEqual(len({p['remaining'] for p in plans}), 1)
        self.assertEqual(16 * plans[0]['alpha_axis'], plans[0]['alpha_familywise_16_axes'])
        self.assertTrue(all(not p['uses_reference_B_new_or_m2'] and not p['uses_exact_signal'] for p in plans))

    def test_log640_upper_certificate(self):
        import mpmath as mp
        mp.mp.dps = 150
        upper = log640_upper()
        self.assertGreater(mp.mpf(upper.numerator) / upper.denominator, mp.log(640))
        self.assertLess(mp.mpf(upper.numerator) / upper.denominator - mp.log(640), mp.mpf('1e-75'))

    def test_deterministic_interface_reproducible_with_zero_trials(self):
        for arm in ARMS:
            gen = make_generator(P, X, 5, arm)
            a, b = DeterministicBits(), DeterministicBits()
            self.assertEqual([gen.sample(a) for _ in range(24)], [gen.sample(b) for _ in range(24)])
        gen = make_generator(P, X, 5, 'full_return')
        class ForceRawPair:
            def __init__(self): self.position = 0
            def bits(self, n):
                self.position += 1
                if self.position == 1:
                    return int(gen.order[0] * (1 << n))  # choose l=2
                return 0  # two identical labels -> pre-quantum zero
        forced = ForceRawPair()
        self.assertIsNone(gen.sample(forced))
        self.assertEqual(forced.position, 3)  # no acceptance/child/angle work after raw zero

    def test_provider_calls_and_helper_resource(self):
        gen = make_generator(P, X, 5, 'full_return')
        event = gen.event((0, 1), 1)
        ir = controlled_event_ir(event)
        self.assertEqual(sum(e['op'] in ('CQ', 'CQ_actual_adjoint') for e in ir), 4)
        self.assertEqual(sum(e['op'] == 'H' for e in ir), 4)
        self.assertEqual(sum(e['op'] == 'CX' for e in ir), 2)
        self.assertEqual(sum(e['op'] == 'RZ' for e in ir), 2)
        self.assertEqual(sum(event['provider_calls'].values()), 4)
        self.assertEqual(ir[0]['op'], 'Z')

    def test_strict_controlled_semantics_and_helper_reset(self):
        import numpy as np
        from math import atan, cos, sin, sqrt
        qs = [np.array([[0, 1], [1, 0]], complex), np.diag([1, -1]).astype(complex)]
        H = np.array([[1, 1], [1, -1]], complex) / sqrt(2)
        def local(g):
            U = np.zeros((8, 8), complex)
            for column in range(8):
                c, a, s = column // 4, (column // 2) % 2, column % 2
                state = [c, a, s]; op = g['op']
                if op in ('CQ', 'CQ_actual_adjoint'):
                    ctrl = c if g['control'] == 'outer' else a
                    Q = qs[g['label']] if op == 'CQ' else qs[g['label']].conj().T
                    matrix, wire = Q if ctrl else np.eye(2), 2
                elif op == 'H': matrix, wire = H, 1
                elif op == 'Z': matrix, wire = np.diag([1, -1]), 0
                elif op == 'RZ':
                    phi = atan(float(F(g['ratio']))) * g['sign']
                    matrix, wire = np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]), 1
                elif op == 'CX':
                    state[1] ^= c; U[4 * state[0] + 2 * state[1] + state[2], column] = 1; continue
                else: raise AssertionError(op)
                for bit in range(2):
                    target = state.copy(); target[wire] = bit
                    U[4 * target[0] + 2 * target[1] + target[2], column] = matrix[bit, state[wire]]
            return U
        event = {'word': (0, 1), 'child': 0, 'ratio': F(1, 7), 'phase_i_power': 2}
        for sigma in (-1, 1):
            U = np.eye(8, dtype=complex)
            for gate in controlled_event_ir(event, sigma): U = local(gate) @ U
            phi = atan(1 / 7)
            V = -(cos(phi) * np.eye(2) - 1j * sigma * sin(phi) * qs[0]) @ qs[0] @ qs[1]
            for c in (0, 1):
                for s in (0, 1):
                    expected = np.zeros(8, complex)
                    expected[4 * c:4 * c + 2] = (np.eye(2) if c == 0 else V)[:, s]
                    self.assertLess(np.linalg.norm(U[:, 4 * c + s] - expected), 1e-12)
            wrong = U.copy(); wrong *= np.exp(1j / 9)
            self.assertGreater(np.linalg.norm(wrong - U), 0.1)  # phase-sensitive, not projective

    def test_enumeration_separated_from_production(self):
        code = (Path(__file__).resolve().parents[3] / 'src/trottertracks/algorithm_codesign/g7_generator.py').read_text()
        for forbidden in ('itertools', 'reference_events', 'resource_reference', 'pygridsynth', 'numpy'):
            self.assertNotIn(forbidden, code)

    def test_marker_is_exclusive_and_never_consumed_twice(self):
        import tempfile
        from trottertracks.algorithm_codesign.g7_launch import consume_marker
        with tempfile.TemporaryDirectory() as directory:
            marker = consume_marker(directory, {'source': 'off-domain', 'runs': 1})
            before = marker.read_bytes()
            with self.assertRaises(FileExistsError):
                consume_marker(directory, {'runs': 2})
            self.assertEqual(before, marker.read_bytes())

    def test_protected_source_and_append_only_index(self):
        import hashlib, json, tempfile
        from trottertracks.algorithm_codesign.g7_launch import protected_check
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source').write_bytes(b'old source')
            (root / 'index').write_bytes(b'old index\nnew entry')
            ledger = {p: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()}
                      for p, v in [('source', b'old source'), ('index', b'old index\n')]}
            (root / 'ledger').write_text(json.dumps(ledger))
            contract = {'protected_ledger': 'ledger', 'append_only_paths': ['index']}
            self.assertEqual(protected_check(root, contract)['violations'], [])
            (root / 'source').write_bytes(b'changed')
            self.assertEqual(protected_check(root, contract)['violations'], ['source'])

    def test_launch_refuses_dirty_or_different_head_before_science(self):
        from unittest.mock import patch
        from trottertracks.algorithm_codesign.g7_launch import verify_source
        module = 'trottertracks.algorithm_codesign.g7_launch.git'
        with patch(module, return_value='b' * 40):
            with self.assertRaises(PermissionError): verify_source(Path('/tmp'), 'a' * 40, {})
        with patch(module, side_effect=['a' * 40, ' M source.py']):
            with self.assertRaises(PermissionError): verify_source(Path('/tmp'), 'a' * 40, {})

    def test_zero_probability_prevents_readout_cost_but_not_attempt_cap(self):
        import importlib.util
        path = Path(__file__).resolve().parents[3] / 'scripts/tracks/algorithm_codesign/g7_budget_control_economics.py'
        spec = importlib.util.spec_from_file_location('g7_runner_test', path)
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        ref = {'digital_acceptance': F(1, 4), 'per_trial_fixed_cost':
               {'T_Rz': F(3), 'CX_fixed': F(1, 2), '1Q_fixed_no_readout': F(5)},
               'per_trial_provider_calls': [F(1, 2), F(3, 4)]}
        total = runner.totals({'N_per_axis': 20}, ref)
        self.assertEqual(total['expected_quantum_calls_two_axes'], 10)
        self.assertEqual(total['quantum_call_hard_cap_two_axes'], 40)
        self.assertEqual(total['T_Rz'], 120)
        self.assertEqual(total['1Q_fixed_including_Re_Im_readout'], 225)

    def test_keys_are_shared_and_negative_is_actual_adjoint(self):
        import importlib.util
        path = Path(__file__).resolve().parents[3] / 'scripts/tracks/algorithm_codesign/g7_budget_control_economics.py'
        spec = importlib.util.spec_from_file_location('g7_inventory_test', path)
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        gens = [('off-domain', make_generator(P, X, 5, arm)) for arm in ARMS]
        result = runner.inventory(gens)
        self.assertEqual(result['key_count'], len(set(result['positive_tangent_keys'])))
        self.assertLess(result['key_count'], sum(len(row['positive_tangent_keys']) for row in result['rows']))
        self.assertIn('actual adjoint', result['negative_primitives'])

    def test_native_binding_retains_scalar_and_actual_adjoint(self):
        from trottertracks.algorithm_codesign.g7_provider import bind_synthesized_provider_ir
        event = {'word': (), 'child': 0, 'ratio': F(1, 7), 'phase_i_power': 0}
        cache = {'1/7': {'sequence': 'WTStH', 'sequence_sha256': 'synthetic',
                         'strict_operator_error_upper': '0'}}
        native = [g for g in bind_synthesized_provider_ir(event, cache) if g['op'] == 'SYNTHESIZED_RZ']
        self.assertEqual(native[0]['matrix_product_tokens'], list('WTStH'))
        self.assertEqual(native[1]['matrix_product_tokens'], ['H', 'T', 'Sdag', 't', 'Wdag'])
        self.assertTrue(native[1]['actual_adjoint'])
        self.assertTrue(native[1]['global_phase_tokens_retained'])


if __name__ == '__main__': unittest.main()
