"""Artificial only: Pauli-D2 guard route, level-1 paper G accounting, model validation.

No H4 arrays, saved prices, compiler or sampling. Fixtures reuse the existing C
pipeline/guard tests (fictional matrices, prices and N).
"""
from copy import deepcopy
from fractions import Fraction
import sys
import unittest
from unittest.mock import patch

import check_price_implementation_guard as GT
from trottertracks import paper_nominal_cost as PN
from trottertracks import price_implementation_guard as G
from trottertracks import price_model_validation as V
from trottertracks import rpe_mean_price_model as P

T = GT.T; ROOT = GT.ROOT; I = GT.I; B = GT.B; API = T.API
R = sys.modules[B.__package__].pr_rpe_rmse
MOCK = dict(fixture='ARTIFICIAL_PAPER_G_ROUTES_ONLY')
PAULI_CODE = dict(commit='9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c', path='src/trotterlib/h4_d2_pauli_control.py')
TAIL = dict(kind=G.PAULI_TAIL_KINDS[0], dictionary_sha256='0'*64, source=MOCK)
FIX = {}


def fixture(method):
    if method not in FIX:
        FIX[method] = T.fixture(method)
    return deepcopy(FIX[method])


def route_policy():
    pol = GT.policy(); pol['id'] = 'ARTIFICIAL_DF_ZZ_PLUS_PAULI_D2'
    pol['by_method']['PR_PAULI_TAIL'] = dict(native_policy='D2_candidate_central_rotation_control',
        rte_rotation_lowering=G.PAULI_D2, scalar_quantity='ART_TOTAL_PR_PAULI')
    return pol


def pauli_values():
    """The PR fixture re-generated with a collected-Pauli tail dictionary (B canonical)."""
    req, packet, alloc, evals, prices, opts = fixture('PR')
    spec = deepcopy(req['configuration']['spec'])
    spec['split']['tail_dictionary'] = deepcopy(TAIL); spec['law']['tail_dictionary'] = deepcopy(TAIL)
    config = I.C.configure(ROOT, spec, api=API)
    physical = dict(enumerate(deepcopy(req['pr_physical'])))
    req2 = I.plan(ROOT, config, target=req['target'], signal_source=req['signal_source'], pr_physical=physical)
    rows = {r['m']: r for r in config['rounds']}; prices2 = []; sides = []; cohorts = {}
    for price, side in zip(prices, opts['price_bindings']):
        row = rows[price['identity']['m']]
        p2 = deepcopy(price); p2.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']))
        s2 = deepcopy(side); s2.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                                       physical_context=I.price_context(req2, row['m']))
        prices2.append(p2); sides.append(s2); cohorts[(row['candidate']['id'], price['axis'])] = price['cohort']
    return req2, prices2, sides, cohorts


def bind_route(req, prices, sides, pol, *, route=None, lowering_source=PAULI_CODE):
    config = req['configuration']; key = G.route(B, config['spec']); domain = pol['by_method'][key]
    method = B.comparison_method(config['spec']); proofs = {}
    ext = dict(lowering_id=domain['rte_rotation_lowering'], source=MOCK)
    for price, side in zip(prices, sides):
        price['normalization_is_price_multiplier'] = False
        row = config['rounds'][price['identity']['m']]; side['profile_extension'] = deepcopy(ext)
        facts = dict(domain, method=method, profile=deepcopy(pol['profile']), measurement=B.p.measurement(row, price['axis']),
            native_source=GT.CODE, wrapper_source=GT.CODE, scalar_source=GT.CODE, scalar_application=G.SCALAR_ONCE,
            rte_lowering_source=lowering_source)
        if route is not None:
            facts['route'] = route
        side['implementation'] = facts
        locator = dict(commit='b'*40, path='ARTIFICIAL_FIXED_METADATA.json', selector=[str(row['m'])+price['axis']])
        side['implementation_evidence'] = locator
        proofs[locator['selector'][0]] = {k: deepcopy(side[k]) for k in ('price_source', 'identity', 'law_binding', 'axis',
                                           'cohort', 'physical_context', 'profile_extension', 'implementation')}
    return proofs


def check(req, prices, sides, cohorts, pol, proofs):
    with patch.object(G, '_document', return_value=proofs):
        return G.check(ROOT, req['configuration'], prices, cohorts=cohorts, bindings=sides, policy=pol)


class PauliRoute(unittest.TestCase):
    def test_route_key_from_tail_dictionary(self):
        req, prices, sides, cohorts = pauli_values()
        self.assertEqual(G.route(B, req['configuration']['spec']), 'PR_PAULI_TAIL')
        self.assertEqual(G.route(B, fixture('PR')[0]['configuration']['spec']), 'PR')
        spec = deepcopy(req['configuration']['spec']); spec['split']['tail_dictionary']['kind'] = 'UNKNOWN_DICTIONARY'
        with self.assertRaisesRegex(ValueError, 'Unknown PR tail'):
            G.route(B, spec)

    def test_pauli_D2_accepted_beside_DF_new_ZZ_in_one_policy(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values()
        proofs = bind_route(req, prices, sides, pol, route='PR_PAULI_TAIL')
        self.assertEqual(len(check(req, prices, sides, cohorts, pol, proofs)['checked']), 4)
        v, df_proofs = GT.bind(fixture('PR')); s = v[5]; s['comparison_policy'] = pol
        out = check(v[0], v[4], s['price_bindings'], s['cohorts'], pol, df_proofs)
        self.assertEqual(len(out['checked']), 4)

    def test_pauli_never_falls_back_to_DF_PR_entry(self):
        req, prices, sides, cohorts = pauli_values(); pol = GT.policy()
        proofs = bind_route(req, prices, sides, route_policy(), route='PR_PAULI_TAIL')
        with self.assertRaisesRegex(ValueError, 'outside the explicit implementation policy'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_pauli_sidecar_must_declare_its_route(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values()
        proofs = bind_route(req, prices, sides, pol, route=None)
        with self.assertRaisesRegex(ValueError, 'route differs'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_zz_and_pauli_lowerings_cannot_be_swapped(self):
        req, prices, sides, cohorts = pauli_values()
        bad = route_policy(); bad['by_method']['PR_PAULI_TAIL']['rte_rotation_lowering'] = G.NEW_ZZ
        proofs = bind_route(req, prices, sides, bad, route='PR_PAULI_TAIL', lowering_source=GT.NEW_CODE)
        with self.assertRaisesRegex(ValueError, 'separate registered lowerings'):
            check(req, prices, sides, cohorts, bad, proofs)
        bad = route_policy(); bad['by_method']['PR']['rte_rotation_lowering'] = G.PAULI_D2
        bad['by_method']['FULL_H_RTE']['rte_rotation_lowering'] = G.PAULI_D2
        v, df_proofs = GT.bind(fixture('PR'))
        with self.assertRaisesRegex(ValueError, 'separate registered lowerings'):
            check(v[0], v[4], v[5]['price_bindings'], v[5]['cohorts'], bad, df_proofs)

    def test_pauli_id_with_ZZ_emitter_rejected(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values()
        proofs = bind_route(req, prices, sides, pol, route='PR_PAULI_TAIL', lowering_source=GT.NEW_CODE)
        with self.assertRaisesRegex(ValueError, 'emitter SOURCE'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_old_fullH_generic_RZZ_still_rejected_under_route_policy(self):
        v, proofs = GT.bind(fixture('FULL_H_RTE')); pol = route_policy()
        for side in v[5]['price_bindings']:
            side['implementation']['rte_rotation_lowering'] = GT.OLD; side['implementation']['rte_lowering_source'] = GT.OLD_CODE
            proof = proofs[side['implementation_evidence']['selector'][0]]
            proof['implementation']['rte_rotation_lowering'] = GT.OLD; proof['implementation']['rte_lowering_source'] = GT.OLD_CODE
        with self.assertRaisesRegex(ValueError, 'rte_rotation_lowering'):
            check(v[0], v[4], v[5]['price_bindings'], v[5]['cohorts'], pol, proofs)
        pol['by_method']['FULL_H_RTE']['rte_rotation_lowering'] = GT.OLD
        with self.assertRaisesRegex(ValueError, 'Mixed RTE lowering'):
            check(v[0], v[4], v[5]['price_bindings'], v[5]['cohorts'], pol, proofs)


class PaperLevel(unittest.TestCase):
    def guarded(self, method):
        v, proofs = GT.bind(fixture(method)); return v, proofs

    def paper(self, v, proofs, packet=None):
        req = v[0]; s = v[5]
        packet = packet or PN.nominal_packet(ROOT, req['configuration'], v[1], source=MOCK)
        with patch.object(G, '_document', return_value=proofs), \
                patch.object(B, 'cost', side_effect=AssertionError('paper level must not call B.cost')), \
                patch.object(B, 'select', side_effect=AssertionError('paper level must not call B.select')):
            return PN.paper_cost(ROOT, req['configuration'], packet, v[4], cohorts=s['cohorts'],
                                 price_bindings=s['price_bindings'], comparison_policy=s['comparison_policy'])

    def test_deterministic_PF_nominal_N_and_exact_G(self):
        v, proofs = self.guarded('FULL'); packet = PN.nominal_packet(ROOT, v[0]['configuration'], v[1], source=MOCK)
        self.assertEqual([(r['nominal_N_X'], r['nominal_N_Y']) for r in packet['rounds']], [(15, 15), (11, 11)])
        self.assertIsNone(packet['sufficient_N']); self.assertEqual(packet['status'], PN.N_STATUS)
        self.assertTrue(all(r['sufficient_N_X'] is None for r in packet['rounds']))
        out = self.paper(v, proofs, packet)
        self.assertEqual(B.a.exact_scalar(out['G_paper']), 2*(15*1+11*100))
        self.assertEqual(out['status'], 'PAPER_NOMINAL_G_DETERMINISTIC_DIRECT')
        self.assertEqual([B.a.exact_scalar(x) for x in out['G_paper_interval']], [2230, 2230])
        self.assertIsNone(out['G_qualified']); self.assertIsNone(out['sufficient_N'])

    def test_PR_uses_its_own_finite_B(self):
        v, proofs = self.guarded('PR'); packet = PN.nominal_packet(ROOT, v[0]['configuration'], v[1], source=MOCK)
        expected = [R.nominal_axis_shots(1, m['m'], m['normalization']['log_B'])['nominal_N_X'] for m in v[1]['means']]
        self.assertEqual([r['nominal_N_X'] for r in packet['rounds']], expected)
        out = self.paper(v, proofs, packet)
        self.assertEqual(B.a.exact_scalar(out['G_paper']), 2*(expected[0]*1+expected[1]*100))
        self.assertEqual(out['status'], 'PAPER_NOMINAL_G_QUALIFIED_EXPECTATION')

    def test_nominal_packet_cannot_enter_qualified_B_path(self):
        v, _ = self.guarded('FULL'); packet = PN.nominal_packet(ROOT, v[0]['configuration'], v[1], source=MOCK)
        out = B.assess(v[0]['configuration'], packet, v[4], cohorts=v[5]['cohorts'])
        self.assertFalse(out['eligible']); self.assertIn('UNKNOWN_N_EVIDENCE_KIND', out['reasons'])

    def test_deterministic_nonunit_B_and_missing_normalization(self):
        v, _ = self.guarded('FULL'); packet = deepcopy(v[1]); packet['means'][0]['normalization']['log_B'] = .1
        with self.assertRaisesRegex(ValueError, 'B = 1'):
            PN.nominal_packet(ROOT, v[0]['configuration'], packet, source=MOCK)
        packet = deepcopy(v[1]); packet['means'][1]['normalization'].pop('log_B')
        out = PN.nominal_packet(ROOT, v[0]['configuration'], packet, source=MOCK)
        self.assertEqual(out['status'], 'NOT_EVALUATED_MISSING_NORMALIZATION'); self.assertIsNone(out['rounds'][1]['nominal_N_X'])

    def test_missing_price_is_unknown_not_zero(self):
        v, proofs = self.guarded('FULL'); v[4].pop()
        out = self.paper(v, proofs)
        self.assertIsNone(out['G_paper']); self.assertEqual(out['status'], 'NOT_EVALUATED_MISSING_PRICE_OR_N')
        self.assertEqual(out['missing'], [dict(m=1, axis='Y', reason='MISSING_PRICE')])

    def test_model_and_unpaired_prices_downgrade_status(self):
        v, proofs = self.guarded('PR')
        for price in v[4][:2]:  # both axes of m0, so the X/Y pair stays consistent
            price.update(classification='MODEL_PREDICTION', model=dict(qualified_for_candidate=False))
        self.assertEqual(self.paper(v, proofs)['status'], 'PAPER_NOMINAL_G_UNQUALIFIED_MODEL')
        v, proofs = self.guarded('PR'); v[4][1]['n_paths'] = 7
        out = self.paper(v, proofs)
        self.assertEqual(out['status'], 'PAPER_NOMINAL_G_UNPAIRED_XY_UNQUALIFIED')

    def test_guard_runs_before_any_arithmetic_and_policy_is_mandatory(self):
        v, proofs = self.guarded('FULL_H_RTE')
        for side in v[5]['price_bindings']:
            side['implementation']['rte_rotation_lowering'] = GT.OLD
            proofs[side['implementation_evidence']['selector'][0]]['implementation']['rte_rotation_lowering'] = GT.OLD
        with self.assertRaisesRegex(ValueError, 'rte_rotation_lowering'):
            self.paper(v, proofs)
        v, proofs = self.guarded('FULL'); v[5]['comparison_policy'] = None
        with self.assertRaisesRegex(ValueError, 'comparison policy'):
            self.paper(v, proofs)

    def test_compare_orders_under_one_policy_without_qualified_winner(self):
        results = []
        for method in ('FULL', 'PR', 'B0I', 'FULL_H_RTE'):
            v, proofs = self.guarded(method); results.append(self.paper(v, proofs))
        out = PN.paper_compare(results)
        self.assertEqual(len(out['ranking']), 4); self.assertIsNone(out['qualified_winner'])
        values = [B.a.exact_scalar(r['G_paper']) for r in out['ranking']]; self.assertEqual(values, sorted(values))
        other = deepcopy(results[1]); other['comparison_policy']['id'] = 'OTHER'
        with self.assertRaisesRegex(ValueError, 'common SOURCE-bound'):
            PN.paper_compare([results[0], other])


class ModelValidation(unittest.TestCase):
    def rec(self, cell, role, q, rz, **kw):
        base = dict(cell_id=cell, role=role, route='FULL_PF', axis='X', features=dict(one=1, pf_step_count=q), rz_count=rz,
                    kind='DIRECT_COMPILED_DETERMINISTIC', source=dict(cell=cell, **MOCK),
                    implementation=dict(lowering_id='ART_NATIVE', source=MOCK), profile=dict(B.p.PROFILE))
        base.update(kw); return V.record(**base)

    def test_exact_affine_fit_holdout_pass_and_fail_without_refit(self):
        cal = [self.rec('m4', 'CALIBRATION', 7, 107), self.rec('m5', 'CALIBRATION', 14, 177)]
        model = V.fit(cal, features=['one', 'pf_step_count'], design_source=MOCK)
        self.assertEqual(model['coefficients'], dict(one=dict(numerator=37, denominator=1), pf_step_count=dict(numerator=10, denominator=1)))
        tol = dict(kind='RELATIVE', value=Fraction(1, 100), source=MOCK)
        ok = V.validate(model, [self.rec('m6', 'HOLDOUT', 28, 318)], tolerance=tol)
        self.assertEqual(ok['status'], 'HOLDOUT_WITHIN_DECLARED_TOLERANCE'); self.assertFalse(ok['qualified_for_candidate'])
        bad = V.validate(model, [self.rec('m6', 'HOLDOUT', 28, 400)], tolerance=tol)
        self.assertEqual(bad['status'], 'HOLDOUT_FAILED_NO_REFIT'); self.assertEqual(bad['coefficients'], model['coefficients'])
        with self.assertRaises(ValueError):
            V.validate(bad, [self.rec('m7', 'HOLDOUT', 56, 597)], tolerance=tol)

    def test_overdetermined_singular_overlap_and_mixed_route(self):
        cal = [self.rec('a', 'CALIBRATION', 1, 10), self.rec('b', 'CALIBRATION', 2, 13), self.rec('c', 'CALIBRATION', 3, 15)]
        model = V.fit(cal, features=['one', 'pf_step_count'], design_source=MOCK)
        self.assertEqual(model['method'], 'EXACT_NORMAL_EQUATION_LEAST_SQUARES')
        self.assertEqual(model['coefficients']['pf_step_count'], dict(numerator=5, denominator=2))
        with self.assertRaisesRegex(ValueError, 'singular'):
            V.fit([self.rec('a', 'CALIBRATION', 2, 10), self.rec('b', 'CALIBRATION', 2, 11)], features=['one', 'pf_step_count'], design_source=MOCK)
        tol = dict(kind='ABSOLUTE', value=1, source=MOCK)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            V.validate(model, [self.rec('a', 'HOLDOUT', 1, 10)], tolerance=tol)
        with self.assertRaisesRegex(ValueError, 'differs'):
            V.validate(model, [self.rec('d', 'HOLDOUT', 4, 18, axis='Y')], tolerance=tol)
        with self.assertRaisesRegex(ValueError, 'One route'):
            V.fit([cal[0], self.rec('z', 'CALIBRATION', 5, 30, route='TRUNCATED_PF')], features=['one', 'pf_step_count'], design_source=MOCK)

    def test_terms_feed_existing_quote_model_unqualified(self):
        req = fixture('FULL')[0]; ext = dict(lowering_id='ART_NATIVE', source=MOCK)
        model = V.fit([self.rec('m4', 'CALIBRATION', 7, 107), self.rec('m5', 'CALIBRATION', 14, 177)],
                      features=['one', 'pf_step_count'], design_source=MOCK)
        terms = V.quote_terms(model, dict(one=('SCALAR', 'WRAPPER'), pf_step_count=('PF_BODY', 'SIGNED_PIECE_BOUNDARY', 'OUTER_BOUNDARY')))
        scope = P.context(ROOT, req, 1, 'X', profile_extension=ext)
        out = P.quote_model(ROOT, req, 1, 'X', profile_extension=ext,
            model=dict(mode='ADDITIVE', terms=terms, source=MOCK, context=scope, qualified_for_candidate=False))
        q = scope['physical_context']['physical_candidate']['q']
        self.assertEqual(B.a.exact_scalar(out['mean']['rz_count']), 37+10*q)
        self.assertEqual(out['price_status'], 'UNQUALIFIED_MODEL')


if __name__ == '__main__':
    unittest.main(verbosity=2)
