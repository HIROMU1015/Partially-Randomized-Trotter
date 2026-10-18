"""Focused F3 artificial/formal semantics only; no science data or stochastic study.

Only this file is selected by the recorded runner. Fixed bit integers are path
witnesses, not IID/acceptance-rate/performance evidence. No word-universe table.
"""
from dataclasses import replace
from fractions import Fraction as F
import json
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import g6_f3_contract as contract
from trottertracks.algorithm_codesign import g6_f3_precision as precision
from trottertracks.algorithm_codesign import g6_f3_generator as api
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, F3Failure
from trottertracks.algorithm_codesign.return_aggregation import root_interval


CONFIG = dict(root_bits=128, probability_bits=96, eta=F(1, 1000), rho=F(1, 10**6),
              angle_tau=F(1, 1 << 32), angle_root_bits=80, angle_terms=48)


def session(p=(F(2, 5), F(3, 5)), x=F(1, 2), m=3, **changes):
    return api.prepare(p, x, m, **(CONFIG | changes))


class FixedBits:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def bits(self, n):
        self.calls.append(n)
        return next(self.values)


class ForbiddenBits:
    def __init__(self):
        self.calls = 0

    def bits(self, n):
        self.calls += 1
        raise AssertionError('bits must not be consumed')


def direct_atan_interval(r, terms=256):
    """Independent direct series; no half-angle/root or F3 enclosure reuse."""
    result = F(0)
    for k in range(terms):
        result += F((-1)**k, 2*k+1) * r**(2*k+1)
    following = F((-1)**terms, 2*terms+1) * r**(2*terms+1)
    return result+min(F(0), following), result+max(F(0), following)


def independent_pi_interval():
    # Classical Machin identity pi = 16 atan(1/5) - 4 atan(1/239).
    a, b = direct_atan_interval(F(1, 5)), direct_atan_interval(F(1, 239))
    return 16*a[0]-4*b[1], 16*a[1]-4*b[0]


class F3Assertions(unittest.TestCase):
    def assertFailure(self, category, function, *args, **kwargs):
        with self.assertRaises(F3Failure) as caught:
            function(*args, **kwargs)
        failure = caught.exception
        self.assertEqual(failure.category, category)
        self.assertTrue(failure.evidence.read()['mandatory_STOP'])
        self.assertFalse(failure.evidence.read()['run_complete'])
        self.assertFalse(failure.evidence.read()['is_random_zero'])
        return failure


class T1Entrance(F3Assertions):
    def test_missing_or_unregistered_precision_arguments_fail_closed(self):
        self.assertFailure('INVALID_INPUT', api.prepare, (1,), F(1, 2), 1)
        self.assertFailure('INVALID_INPUT', session, unknown_precision=1)

    def test_exact_fraction_and_integer_pair_inputs(self):
        ready = session(p=({'num': 2, 'den': 5}, {'num': 3, 'den': 5}),
                        x={'num': 1, 'den': 2})
        self.assertEqual(ready.request.p, (F(2, 5), F(3, 5)))
        self.assertEqual(ready.status()['state'], 'READY')

    def test_float_or_string_is_not_implicitly_a_rational(self):
        for value in (0.5, '1/2', float('nan')):
            with self.subTest(received_type=type(value).__name__):
                self.assertFailure('INVALID_INPUT', session, x=value)
        self.assertFailure('INVALID_INPUT', session, p=(0.4, 0.6))

    def test_boolean_integer_substitutes_are_rejected(self):
        for field in ('m', 'sigma', 'root_bits', 'probability_bits', 'angle_terms',
                      'angle_root_bits', 'eta', 'rho', 'angle_tau', 'x'):
            with self.subTest(field=field):
                self.assertFailure('INVALID_INPUT', session, **{field: True})

    def test_zero_negative_empty_and_implicit_tables_rejected(self):
        for p in ((0, 1), (F(-1, 2), F(3, 2)), (), iter((F(1),)), '1'):
            with self.subTest(received_type=type(p).__name__):
                self.assertFailure('INVALID_INPUT', session, p=p)

    def test_normalization_is_exact(self):
        self.assertFailure('INVALID_INPUT', session, p=(F(1, 3), F(1, 3)))

    def test_rational_denominator_must_be_positive_integer(self):
        for d in (0, -2, True, 2.0):
            with self.subTest(denominator_type=type(d).__name__):
                self.assertFailure('INVALID_INPUT', session, x={'num': 1, 'den': d})

    def test_degree_and_short_step_domains(self):
        for m in (-1, 0, 2, F(3), 3.0):
            with self.subTest(m_type=type(m).__name__):
                self.assertFailure('INVALID_INPUT', session, m=m)
        for x in (F(-1), F(3, 2)):
            self.assertFailure('INVALID_INPUT', session, x=x)

    def test_full_m1_and_canonical_domain_unsupported_are_distinct(self):
        self.assertEqual(session(m=1).status()['state'], 'READY')
        bad_domain = self.assertFailure('INVALID_INPUT', session, m=1, arm='ordinary')
        unsupported = self.assertFailure('INVALID_INPUT', session, arm='ordinary')
        self.assertNotEqual(bad_domain.reason, unsupported.reason)
        self.assertEqual(unsupported.reason, 'CANONICAL_PREFLIGHT_NOT_IMPLEMENTED')
        self.assertFailure('INVALID_INPUT', session, arm='unregistered')

    def test_eta_rho_sigma_tau_domains(self):
        for name, values in [('eta', (F(0), F(1), F(2))), ('rho', (F(0), F(-1))),
                             ('sigma', (0, 2, F(1))), ('angle_tau', (F(0), F(-1)))]:
            for value in values:
                with self.subTest(field=name):
                    self.assertFailure('INVALID_INPUT', session, **{name: value})

    def test_positive_supplied_bit_and_series_parameters(self):
        for key in ('root_bits', 'probability_bits', 'angle_root_bits', 'angle_terms'):
            for value in (0, -1, 2.0):
                with self.subTest(field=key):
                    self.assertFailure('INVALID_INPUT', session, **{key: value})

    def test_parent_helper_validates_external_words(self):
        for word in ((-1, 0), (True, 1), (0,), (0, 0), (0, 1, 0, 1), '01'):
            with self.subTest(word_type=type(word).__name__):
                self.assertFailure('INVALID_INPUT', session().parent_packet, word)

    def test_child_helper_rejects_negative_bool_and_excluded(self):
        for child in (-1, True, 2, 0):
            with self.subTest(child_type=type(child).__name__):
                self.assertFailure('INVALID_INPUT', session().event, (0, 1), child, trial_id='child')


class T2Identity(F3Assertions):
    def test_identity_never_constructs_or_queries_generator_or_angle(self):
        bits = ForbiddenBits()
        with (patch.object(api.legacy, 'FullReturnGenerator', side_effect=AssertionError('constructor')),
              patch.object(api, 'enclose_atan', side_effect=AssertionError('angle'))):
            identity = session(x=F(0))
            result = identity.sample(bits, trial_id='identity').read()
        self.assertEqual(bits.calls, 0)
        self.assertEqual(result['kind'], 'IDENTITY')
        self.assertEqual(result['bits_consumed'], 0)
        self.assertEqual(identity.preflight.read()['positive_preflight'], 'NOT_APPLICABLE')

    def test_identity_is_distinct_from_random_zero(self):
        identity = session(x=0).sample(None, trial_id='id').read()
        zero = session().sample(FixedBits([0, (1 << 96)-1]), trial_id='zero').read()
        self.assertEqual(identity['kind'], 'IDENTITY')
        self.assertEqual(zero['kind'], 'ZERO')

    def test_identity_preserves_context_and_signed_time(self):
        one, two = session(x=0, sigma=1), session(x=0, sigma=-1)
        for ready in (one, two):
            result = ready.sample(None, trial_id='id').read()
            self.assertEqual(result['input_id'], ready.request.input_id)
            self.assertEqual(result['policy_id'], ready.request.policy_id)
            self.assertEqual(result['sigma'], ready.request.sigma)
        self.assertNotEqual(one.request.input_id, two.request.input_id)

    def test_identity_rejects_stochastic_helper_without_query(self):
        ready = session(x=0)
        self.assertFailure('INVALID_INPUT', ready.event, (), 0, trial_id='invalid')
        self.assertEqual(ready.status()['state'], 'STOPPED')


class T3Preflight(F3Assertions):
    def test_P5_rejects_constructor_law_substitution(self):
        original = api.legacy.FullReturnGenerator
        def changed_law(*args, **kwargs):
            generator = original(*args, **kwargs)
            generator.labels = tuple(reversed(generator.labels))
            return generator
        with patch.object(api.legacy, 'FullReturnGenerator', side_effect=changed_law):
            failure = self.assertFailure('CONTRACT_INCONSISTENCY', session)
        self.assertEqual(failure.reason, 'ACTUAL_LAW_BINDING_FAILED')

    def test_predicate_helper_domains_are_checked(self):
        for function, args in (
                (precision.rounding_predicate, (F(-1), F(1), F(1, 2), F(1, 4))),
                (precision.rounding_predicate, (F(0), F(0), F(1, 2), F(1, 4))),
                (precision.coefficient_predicate, (F(0), F(1), F(1))),
                (precision.acceptance_predicate, (F(1, 4), F(0), F(1, 2)))):
            with self.subTest(function=function.__name__):
                self.assertFailure('INVALID_INPUT', function, *args)
        ready = session()
        self.assertFailure('INVALID_INPUT', precision.assess_precision, ready.request, None)

    def test_all_six_predicates_have_exact_bound_evidence(self):
        cert = session().preflight.read()
        self.assertEqual([p['id'] for p in cert['predicates']], ['P0', 'P1', 'P2', 'P3', 'P4', 'P5'])
        self.assertTrue(all(p['passed'] for p in cert['predicates']))
        self.assertTrue(cert['ready'])
        self.assertFalse(cert['all_parent_enumeration_used'])
        dimensions = cert['input_dimensions']
        self.assertEqual((dimensions['L'], dimensions['m_value'], dimensions['m_binary_length']), (2, 3, 2))
        self.assertEqual((dimensions['H_value'], dimensions['K_root_value']), (96, 128))
        for p in cert['predicates']:
            for check in p['checks']:
                self.assertTrue(check['passed'])
                self.assertNotIsInstance(check['left'], float)
                self.assertNotIsInstance(check['right'], float)

    def test_arithmetic_assessment_alone_does_not_pass_P5(self):
        ready = session()
        with patch.object(api.legacy, 'FullReturnGenerator', side_effect=AssertionError('not needed')):
            data = precision.assess_precision(ready.request, ready._source).read()
        self.assertTrue(data['precision_conditions_satisfied'])
        self.assertFalse(data['ready'])
        self.assertEqual(data['predicates'][-1]['status'], 'NOT_BOUND')

    def test_insufficient_H_fails_before_generator_creation(self):
        with patch.object(api.legacy, 'FullReturnGenerator', side_effect=AssertionError('must not construct')) as ctor:
            failure = self.assertFailure('TECHNICAL_PRECISION_FAILURE', session, probability_bits=1)
        ctor.assert_not_called()
        self.assertEqual(failure.stage, 'prepare.preflight')
        self.assertFalse(failure.evidence.read()['details']['predicates'][2]['passed'])

    def test_insufficient_K_e_ge_one_is_safe_fail_closed(self):
        failure = self.assertFailure('TECHNICAL_PRECISION_FAILURE', session, m=5, root_bits=1)
        p1 = failure.evidence.read()['details']['predicates'][1]
        self.assertFalse(p1['passed'])
        self.assertIsNone(p1['bounds']['zeta_o'])

    def test_midpoint_sufficient_inequality_inclusive_boundary(self):
        checks = precision.coefficient_predicate(F(1, 8), F(1, 4), F(1, 2))
        self.assertTrue(all(c['passed'] for c in checks))
        self.assertEqual(checks[1]['left'], checks[1]['right'])
        self.assertFalse(precision.coefficient_predicate(F(1, 4), F(1, 4), F(1))[0]['passed'])

    def test_R_exact_law_inclusive_boundary_and_failure(self):
        good = precision.rounding_predicate(F(0), F(1, 2), F(1, 4), F(1, 8))
        bad = precision.rounding_predicate(F(0), F(1, 2), F(1, 4), F(1, 8)+F(1, 1024))
        self.assertTrue(all(c['passed'] for c in good))
        self.assertTrue(any(not c['passed'] for c in bad))
        self.assertFalse(precision.rounding_predicate(F(1), F(1, 2), F(1, 4), F(1, 8))[0]['passed'])

    def test_acceptance_inclusive_boundary(self):
        good = precision.acceptance_predicate(F(1, 16), F(1, 64), F(9, 20))
        self.assertTrue(all(c['passed'] for c in good))
        self.assertEqual(good[1]['left'], good[1]['right'])
        bad = precision.acceptance_predicate(F(1, 16), F(1, 32), F(9, 20))
        self.assertFalse(bad[1]['passed'])

    def test_root_exact_and_outward_endpoints(self):
        exact = root_interval(F(4), 8)
        irrational = root_interval(F(2), 8)
        self.assertEqual((exact.lo, exact.hi), (F(2), F(2)))
        self.assertLessEqual(irrational.lo**2, 2)
        self.assertGreaterEqual(irrational.hi**2, 2)
        self.assertLessEqual(irrational.hi-irrational.lo, F(1, 256))

    def test_preflight_does_not_visit_nonempty_parents(self):
        original = api.legacy.ReturnKernel.parent
        visited = []
        def spy(kernel, word):
            visited.append(tuple(word))
            return original(kernel, word)
        with patch.object(api.legacy.ReturnKernel, 'parent', spy):
            ready = session(m=5)
        self.assertEqual(visited, [()])  # The legacy U constructor's one empty-parent query.
        self.assertTrue(ready.preflight.read()['ready'])

    def test_selected_unvisited_parent_guards_after_uniform_preflight(self):
        ready = session(p=(F(1, 6), F(1, 3), F(1, 2)), x=F(2, 5), m=5)
        for word in ((), (0, 1), (1, 2), (0, 1, 0, 1)):
            with self.subTest(parent_length=len(word)):
                packet = ready.parent_packet(word).read()
                lo, hi = packet['d_interval']
                self.assertGreater(lo, 0)
                self.assertLessEqual((hi-lo)/2, ready.request.rho*lo)
                self.assertTrue(all(q > 0 for q in packet['child_law']))
                self.assertEqual(sum(packet['child_law']), 1)

    def test_L1_m1_x1_are_supported_existing_boundaries(self):
        ready = session(p=(1,), x=1, m=1)
        record = ready.sample(FixedBits([0, 0, 0]), trial_id='single').read()
        self.assertEqual(record['kind'], 'EVENT')
        self.assertEqual(record['event']['ratio'], 1)


class T4Failures(F3Assertions):
    def test_raw_nonreduced_is_valid_zero(self):
        bits = FixedBits([(1 << 96)-1, 0, 0])
        ready = session()
        result = ready.sample(bits, trial_id='raw').read()
        self.assertEqual(result['kind'], 'ZERO')
        self.assertEqual(result['reason'], 'RAW_NONREDUCED')
        self.assertEqual(len(bits.calls), 3)
        self.assertEqual(ready.status()['state'], 'READY')

    def test_coin_rejection_is_valid_zero(self):
        result = session().sample(FixedBits([0, (1 << 96)-1]), trial_id='coin').read()
        self.assertEqual((result['kind'], result['reason']), ('ZERO', 'COIN_REJECT'))
        self.assertEqual(result['trace']['bits_consumed'], 192)

    def test_law_tampering_is_detected_before_bit_consumption(self):
        ready, bits = session(), ForbiddenBits()
        ready._generator.labels = tuple(reversed(ready._generator.labels))
        failure = self.assertFailure('INVALID_INPUT', ready.sample, bits, trial_id='tamper')
        self.assertEqual(failure.reason, 'ACTUAL_LAW_STATE_CHANGED')
        self.assertEqual(bits.calls, 0)

    def test_local_guard_after_preflight_is_terminal_inconsistency(self):
        ready = session()
        with patch.object(ready._generator, 'packet', side_effect=ValueError('fixed weight precision insufficient')):
            failure = self.assertFailure('CONTRACT_INCONSISTENCY', ready.sample,
                                         FixedBits([0]), trial_id='guard')
        self.assertEqual(failure.reason, 'LEGACY_GUARD_AFTER_PREFLIGHT')
        self.assertEqual(ready.status()['state'], 'STOPPED')
        self.assertFalse(ready.status()['run_complete'])
        original_id = ready.last_failure.digest
        self.assertFailure('INVALID_INPUT', ready.sample, ForbiddenBits(), trial_id='no-retry')
        self.assertEqual(ready.last_failure.digest, original_id)

    def test_constructor_guard_after_preflight_is_preserved(self):
        with patch.object(api.legacy, 'FullReturnGenerator', side_effect=ValueError('fixed interval precision insufficient')):
            failure = self.assertFailure('CONTRACT_INCONSISTENCY', session)
        self.assertEqual(failure.stage, 'prepare.generator')
        self.assertTrue(failure.evidence.read()['details']['assessment']['precision_conditions_satisfied'])

    def test_bit_source_runtime_failure_is_not_zero_or_retry(self):
        bits, ready = ForbiddenBits(), session()
        failure = self.assertFailure('UNEXPECTED_TECHNICAL_FAILURE', ready.sample, bits, trial_id='bits')
        self.assertEqual(failure.reason, 'BIT_SOURCE_FAILED')
        self.assertEqual(bits.calls, 1)
        self.assertFalse(ready.status()['confidence_certified'])

    def test_invalid_bit_integer_is_contract_failure(self):
        for value in (True, -1, 1 << 96, F(0)):
            with self.subTest(received_type=type(value).__name__):
                failure = self.assertFailure('INVALID_INPUT', session().sample,
                                             FixedBits([value]), trial_id='bad-bits')
                self.assertEqual(failure.reason, 'INVALID_H_BIT_INTEGER')

    def test_invalid_trial_id_does_not_consume_bits(self):
        for trial_id in ('', None, 0.5, object(), True, []):
            with self.subTest(trial_id_type=type(trial_id).__name__):
                bits, ready = ForbiddenBits(), session()
                self.assertFailure('INVALID_INPUT', ready.sample, bits, trial_id=trial_id)
                self.assertEqual(bits.calls, 0)
                self.assertEqual(ready.status()['state'], 'STOPPED')

    def test_unexpected_preflight_runtime_failure_is_preserved(self):
        with patch.object(api, 'assess_precision', side_effect=OSError('artificial allocation failure')):
            failure = self.assertFailure('UNEXPECTED_TECHNICAL_FAILURE', session)
        self.assertEqual(failure.reason, 'PREFLIGHT_EVALUATION_FAILED')

    def test_source_identity_change_uses_no_file_mutation(self):
        ready = session()
        data = ready._source.read()
        data['test_only_tamper'] = True
        with patch.object(api, 'source_identity', return_value=Evidence.create(data)):
            failure = self.assertFailure('INVALID_INPUT', ready.sample, ForbiddenBits(), trial_id='source')
        self.assertEqual(failure.reason, 'SOURCE_IDENTITY_CHANGED')

    def test_preflight_evidence_tampering_is_rejected(self):
        ready = session()
        ready._certificate = replace(ready.preflight, document=ready.preflight.document+' ')
        self.assertFailure('INVALID_INPUT', ready.sample, ForbiddenBits(), trial_id='evidence')

    def test_input_policy_mutation_is_rejected(self):
        ready = session()
        object.__setattr__(ready.request, 'sigma', -1)
        failure = self.assertFailure('INVALID_INPUT', ready.sample, ForbiddenBits(), trial_id='input')
        self.assertEqual(failure.reason, 'INPUT_OR_POLICY_IDENTITY_CHANGED')

    def test_angle_precision_failure_stops_session_without_refinement(self):
        ready = session(angle_root_bits=1, angle_terms=1, angle_tau=F(1, 1 << 80))
        event = ready.event((), 0, trial_id='angle')
        self.assertFailure('TECHNICAL_PRECISION_FAILURE', ready.angle_for_event, event)
        self.assertEqual((ready.request.angle_root_bits, ready.request.angle_terms), (1, 1))
        self.assertEqual(ready.status()['state'], 'STOPPED')

    def test_failure_preserves_parent_child_and_source_location(self):
        ready = session()
        with patch.object(ready._generator, 'packet', side_effect=ValueError('artificial local guard')):
            failure = self.assertFailure('CONTRACT_INCONSISTENCY', ready.event,
                                         (0, 1), 1, trial_id='parent-failure')
        details = failure.evidence.read()['details']
        self.assertEqual(details['operation_context'], {'word_type': 'tuple', 'child_type': 'int',
                                                      'parent': (0, 1), 'child': 1})
        self.assertTrue(details['source_locations'])
        self.assertEqual(details['trial_id'], 'parent-failure')


class T5Angle(F3Assertions):
    def test_unserializable_angle_context_is_invalid_input(self):
        self.assertFailure('INVALID_INPUT', precision.enclose_atan, F(1),
                           tau=F(1), root_bits=8, terms=4, context={'bad': 0.5})

    def angle(self, r, **kw):
        return precision.enclose_atan(r, tau=CONFIG['angle_tau'], root_bits=80,
                                      terms=48, **kw).read()

    def assertContainsReference(self, result, reference):
        lo, hi = result['phi_interval']
        self.assertLessEqual(lo, reference[0])
        self.assertGreaterEqual(hi, reference[1])
        self.assertLessEqual(result['representative_error_upper'], result['tau'])
        self.assertEqual(result['representative'], (lo+hi)/2)

    def test_zero_angle_is_exact_and_needs_no_root(self):
        with patch.object(precision, 'root_interval', side_effect=AssertionError('unused')):
            result = self.angle(0)
        self.assertEqual(result['phi_interval'], (F(0), F(0)))
        self.assertEqual(result['representative_error_upper'], 0)

    def test_atan_one_contains_independent_Machin_pi_quarter(self):
        pi = independent_pi_interval()
        self.assertContainsReference(self.angle(F(1)), (pi[0]/4, pi[1]/4))

    def test_small_positive_angle_contains_independent_direct_series(self):
        self.assertContainsReference(self.angle(F(1, 3)), direct_atan_interval(F(1, 3)))

    def test_upper_tangent_boundary_contains_independent_reciprocal_identity(self):
        pi, half = independent_pi_interval(), direct_atan_interval(F(1, 2))
        self.assertContainsReference(self.angle(F(2)), (pi[0]/2-half[1], pi[1]/2-half[0]))

    def test_signed_time_retains_unsigned_and_signed_intervals(self):
        positive, negative = self.angle(F(1, 2), sigma=1), self.angle(F(1, 2), sigma=-1)
        lo, hi = positive['phi_interval']
        self.assertEqual(negative['phi_interval'], (lo, hi))
        self.assertEqual(negative['signed_phi_interval'], (-hi, -lo))
        self.assertEqual(negative['signed_representative'], -positive['representative'])

    def test_invalid_angle_inputs_fail_closed(self):
        for ratio in (-1, F(3), 0.5, True):
            self.assertFailure('INVALID_INPUT', self.angle, ratio)
        for kwargs in ({'tau': F(0)}, {'root_bits': True}, {'terms': 0}, {'sigma': 0}):
            config = dict(tau=F(1, 100), root_bits=16, terms=16, sigma=1) | kwargs
            self.assertFailure('INVALID_INPUT', precision.enclose_atan, F(1), **config)

    def test_insufficient_angle_precision_keeps_candidate_evidence(self):
        failure = self.assertFailure('TECHNICAL_PRECISION_FAILURE', precision.enclose_atan,
                                     F(1), tau=F(1, 1 << 60), root_bits=1, terms=1)
        data = failure.evidence.read()['details']
        self.assertFalse(data['precision_check']['passed'])
        self.assertIn('phi_interval', data)

    def test_one_supplied_root_precision_no_retry(self):
        with patch.object(precision, 'root_interval', wraps=root_interval) as root:
            result = self.angle(F(1, 2))
        self.assertEqual(root.call_count, 1)
        self.assertEqual(result['root_bits'], 80)
        self.assertEqual(result['terms'], 48)
        self.assertFalse(result['native_rotation_acquired'])

    def test_large_integer_evidence_roundtrip_uses_no_decimal_setting_change(self):
        value = F((1 << 20000)+1, 3)
        evidence = Evidence.create({'large': value})
        self.assertEqual(evidence.read()['large'], value)
        self.assertIn('$rat', evidence.document)
        json.dumps(evidence.to_dict())


class T6EventSemantics(F3Assertions):
    def test_rehashed_external_event_tampering_is_contract_failure(self):
        for field in ('weight', 'proposal', 'ratio', 'phase_i_power'):
            with self.subTest(field=field):
                ready = session()
                data = ready.event((), 0, trial_id='tamper').read()
                data['event'][field] += 1
                failure = self.assertFailure('INVALID_INPUT', ready.angle_for_event,
                                             Evidence.create(data))
                self.assertEqual(failure.reason, 'EVENT_CONTENT_IDENTITY_MISMATCH')

    def test_deterministic_events_equal_frozen_legacy_reference(self):
        for sigma in (-1, 1):
            ready = session(m=5, sigma=sigma)
            reference = api.legacy.FullReturnGenerator(ready.request.p, ready.request.x, 5,
                         root_bits=128, probability_bits=96, eta=CONFIG['eta'], rho=CONFIG['rho'])
            for word, child in (((), 0), ((0, 1), 1), ((0, 1, 0, 1), 1)):
                result = ready.event(word, child, trial_id='formal').read()
                self.assertEqual(result['event'], reference.event(word, child))
                self.assertEqual(result['sigma'], sigma)

    def test_fixed_accepted_stream_preserves_exact_legacy_event(self):
        values = [(1 << 96)-1, 0, (1 << 96)-1, 0, 0]
        ready = session()
        reference = api.legacy.FullReturnGenerator(ready.request.p, ready.request.x, 3,
                       root_bits=128, probability_bits=96, eta=CONFIG['eta'], rho=CONFIG['rho'])
        old = reference.sample(FixedBits(values))
        result = ready.sample(FixedBits(values), trial_id='accepted').read()
        self.assertEqual(result['event'], old)
        self.assertEqual(result['event']['proposal']*result['event']['weight'], result['event']['coefficient'])
        self.assertEqual(result['trace']['extra_random_draws_for_audit'], 0)
        self.assertEqual(result['trace']['completed_bit_map_calls_by_stage'],
                         {'order': 1, 'label': 2, 'acceptance': 1, 'child': 1})

    def test_phase_word_child_and_sigma_are_lossless(self):
        result = session(sigma=-1).event((0, 1), 1, trial_id='phase').read()
        event = result['event']
        self.assertEqual((event['word'], event['child'], event['phase_i_power']), ((0, 1), 1, 2))
        self.assertEqual(event['circuit_time_word'], (1, 0))
        self.assertEqual(result['sigma'], -1)

    def test_packet_evidence_keeps_exact_local_quantities(self):
        ready = session()
        data = ready.parent_packet((0, 1)).read()
        self.assertEqual(data['input_id'], ready.request.input_id)
        self.assertEqual(data['preflight_id'], ready.preflight.digest)
        self.assertEqual(data['angle_tangent'], data['s']/data['a'])
        self.assertEqual(sum(data['child_law']), 1)

    def test_returned_event_is_an_immutable_snapshot(self):
        ready = session()
        record = ready.event((), 0, trial_id='snapshot')
        data = record.read()
        data['event']['weight'] *= 2
        self.assertNotEqual(data, record.read())
        self.assertEqual(ready.angle_for_event(record).read()['context']['event_id'], record.digest)

    def test_angle_rejects_event_from_other_input(self):
        old = session().event((), 0, trial_id='old')
        ready = session(p=(F(1, 3), F(2, 3)))
        self.assertFailure('INVALID_INPUT', ready.angle_for_event, old)

    def test_angle_rejects_event_from_other_precision_policy(self):
        old = session().event((), 0, trial_id='old')
        self.assertFailure('INVALID_INPUT', session(probability_bits=95).angle_for_event, old)

    def test_bound_angle_keeps_parent_sigma_and_precision_identity(self):
        ready = session(sigma=-1)
        event = ready.event((0, 1), 1, trial_id='angle')
        angle = ready.angle_for_event(event).read()
        self.assertEqual(angle['context']['parent'], (0, 1))
        self.assertEqual(angle['context']['preflight_id'], ready.preflight.digest)
        self.assertEqual(angle['context']['policy_id'], ready.request.policy_id)
        self.assertEqual(angle['sigma'], -1)


if __name__ == '__main__':
    unittest.main()
