"""Artificial: Pauli bridge/guard route, level-1 paper G, model validation.

Fictional matrices, dictionary, prices and N for all value paths. The
RealPauliBridge class reads only pinned public metadata (configurations,
dictionary bytes for the hash, carrier physical contexts/SOURCE); it reads no
signal value, price, compiler output, and computes no N or G.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

import check_price_implementation_guard as GT
from trottertracks import paper_nominal_cost as PN
from trottertracks import price_implementation_guard as G
from trottertracks import price_model_validation as V
from trottertracks import rpe_d_price_connection as DC
from trottertracks import rpe_mean_price_model as P
from trottertracks.r_price_phase1_science import functions

T = GT.T; ROOT = GT.ROOT; I = GT.I; B = GT.B; API = T.API
R = sys.modules[B.__package__].pr_rpe_rmse
MOCK = dict(fixture='ARTIFICIAL_PAPER_G_ROUTES_ONLY')
D = DC._d_pauli(str(ROOT))
PAULI_CODE = dict(D.D2_SOURCE)  # D's rte_lowering_source: P3 D2 at 9c4006bf (event emitter checked at the same commit)
DICT_SOURCE = dict(commit='d'*40, path='ARTIFICIAL_PAULI_DICTIONARY.json')
ROUND_SOURCES = {'0': dict(commit='e'*40, path='ARTIFICIAL_REUSED_ROUND.py'),
                 '1': dict(commit='f'*40, path='ARTIFICIAL_NEW_ROUND.py')}
FIX = {}
D_FACTS = dict(deepcopy=deepcopy, **{k: getattr(D, k) for k in DC.D_CONSTANTS})
functions(ROOT, DC.D_PAULI['commit'], DC.D_PAULI['path'],
          ('implementation_facts', 'price_records', 'legacy_physical_candidate'), D_FACTS)
KEEP = object()


def fixture(method):
    if method not in FIX:
        FIX[method] = T.fixture(method)
    return deepcopy(FIX[method])


def route_policy():
    pol = GT.policy(); pol['id'] = 'ARTIFICIAL_DF_ZZ_PLUS_PAULI_D2'
    pol['by_method']['PR_PAULI_TAIL'] = dict(native_policy=D.NATIVE_POLICY,
        rte_rotation_lowering=D.LOWERING_ID, scalar_quantity=D.SCALAR_QUANTITY)
    return pol


def pauli_spec(spec, *, scalar=.004):
    """Fictional hash-bound dictionary; identity built by A's own dictionary_identity."""
    spec = deepcopy(spec)
    payload = dict(binding=deepcopy(spec['binding']), prefix=spec['L_D'], mapping='ARTIFICIAL_MODE_MAPPING',
                   lambda_P=float(B.a.exact_scalar(spec['split']['lambda_R'])), saved_DF_scalar=scalar, terms=[])
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    tail = I._pauli_helpers(str(ROOT)).dictionary_identity(dict(payload=payload,
        dictionary_sha256=hashlib.sha256(raw).hexdigest(), source=deepcopy(DICT_SOURCE)))
    lam = Fraction(payload['lambda_P'])  # A's generate_configuration convention, read by D as a rational
    spec['split'].update(tail_dictionary=deepcopy(tail), lambda_source=dict(DICT_SOURCE, selector=['lambda_P']), scalar=scalar,
                         lambda_R=dict(numerator=lam.numerator, denominator=lam.denominator))
    spec['law']['tail_dictionary'] = deepcopy(tail)
    return spec, payload


def pauli_plan(config, payload, **kw):
    base = fixture('PR')[0]
    args = dict(target=base['target'], signal_source=base['signal_source'],
                implementation_sources=deepcopy(ROUND_SOURCES), pauli_dictionary=deepcopy(payload))
    args.update(kw)
    return I.plan(ROOT, config, **args)


def pauli_fixture():
    """The PR fixture through the Pauli bridge: canonical IDs, dictionary, per-round SOURCE."""
    req, packet, allocations, evaluations, prices, opts = fixture('PR')
    spec, payload = pauli_spec(req['configuration']['spec'])
    config = I.C.configure(ROOT, spec, api=API); req2 = pauli_plan(config, payload)
    rows = {r['m']: r for r in config['rounds']}
    packet2 = deepcopy(packet); packet2['configuration_id'] = config['id']
    for mean in packet2['means']:
        row = rows[mean['m']]; job = req2['pending']['jobs'][mean['m']]; src = ROUND_SOURCES[str(mean['m'])]
        mean.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                    physical_candidate=deepcopy(job['physical_context']['physical_candidate']), implementation_source=deepcopy(src))
        mean['provider_binding'].update(physical_context=deepcopy(job['physical_context']), implementation_source=deepcopy(src))
    evaluations2 = []
    for e in evaluations:
        n, _, _ = T.DATA_SCOPE['data'](config)
        for row, shot in zip(n['rounds'], e['allocation']['shots']):
            row.update(N_X=shot['N_X'], N_Y=shot['N_Y'])
        n['rmse_evidence']['statistical_contract'] = deepcopy(e['statistical_contract'])
        evaluations2.append(dict(e, configuration_id=config['id'], n_packet=n, source=n['source'], signal_input=I.signal_input(packet2)))
    prices2 = []; sides = []; cohorts = {}
    coverage = dict(opts['interval_contract'], configuration_id=config['id'])
    for price, side in zip(prices, opts['price_bindings']):
        row = rows[price['identity']['m']]
        p2 = deepcopy(price); p2.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']))
        s2 = deepcopy(side); s2.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                                       physical_context=I.price_context(req2, row['m']))
        s2['mean_interval'].update(domain=deepcopy(row['identity']['candidate']), coverage=deepcopy(coverage))
        prices2.append(p2); sides.append(s2); cohorts[(row['candidate']['id'], price['axis'])] = price['cohort']
    opts2 = dict(opts, cohorts=cohorts, price_bindings=sides, interval_contract=coverage)
    return [req2, packet2, allocations, evaluations2, prices2, opts2]


def pauli_values():
    v = pauli_fixture(); return v[0], v[4], v[5]['price_bindings'], v[5]['cohorts']


def bind_route(req, prices, sides, pol, *, route=KEEP, lowering_source=PAULI_CODE):
    config = req['configuration']; key = G.route(B, config['spec']); domain = pol['by_method'][key]
    method = B.comparison_method(config['spec']); proofs = {}
    ext = dict(lowering_id=domain['rte_rotation_lowering'], source=MOCK)
    for price, side in zip(prices, sides):
        price['normalization_is_price_multiplier'] = False
        row = config['rounds'][price['identity']['m']]; side['profile_extension'] = deepcopy(ext)
        measurement = B.p.measurement(row, price['axis'])
        if key == 'PR_PAULI_TAIL':
            # D's own sidecar facts (8b28dce0); fictional native/scalar SOURCE blobs.
            facts = D_FACTS['implementation_facts'](deepcopy(pol['profile']), native_source=GT.CODE,
                measurement=measurement, scalar=2.25, scalar_source=GT.CODE)
            facts['rte_lowering_source'] = deepcopy(lowering_source)
            if domain['rte_rotation_lowering'] != D.LOWERING_ID:
                facts.update(rte_rotation_lowering=domain['rte_rotation_lowering'])
        else:
            facts = dict(domain, method=method, profile=deepcopy(pol['profile']), measurement=measurement,
                native_source=GT.CODE, wrapper_source=GT.CODE, scalar_source=GT.CODE, scalar_application=G.SCALAR_ONCE,
                rte_lowering_source=lowering_source)
        if route is None:
            facts.pop('route', None)
        elif route is not KEEP:
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
        proofs = bind_route(req, prices, sides, pol)
        self.assertEqual(len(check(req, prices, sides, cohorts, pol, proofs)['checked']), 4)
        v, df_proofs = GT.bind(fixture('PR')); s = v[5]; s['comparison_policy'] = pol
        out = check(v[0], v[4], s['price_bindings'], s['cohorts'], pol, df_proofs)
        self.assertEqual(len(out['checked']), 4)

    def test_pauli_never_falls_back_to_DF_PR_entry(self):
        req, prices, sides, cohorts = pauli_values(); pol = GT.policy()
        proofs = bind_route(req, prices, sides, route_policy())
        with self.assertRaisesRegex(ValueError, 'outside the explicit implementation policy'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_D_facts_accepted_route_declared_and_emitter_commit_required(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values()
        proofs = bind_route(req, prices, sides, pol)
        self.assertEqual(sides[0]['implementation']['route'], 'PR_PAULI_TAIL')  # D's facts as published
        self.assertEqual(len(check(req, prices, sides, cohorts, pol, proofs)['checked']), 4)
        for route in (None, 'PR'):
            proofs = bind_route(req, prices, sides, pol, route=route)
            with self.assertRaisesRegex(ValueError, 'route differs'):
                check(req, prices, sides, cohorts, pol, proofs)
        proofs = bind_route(req, prices, sides, pol, lowering_source=dict(commit=None, path=PAULI_CODE['path']))
        with self.assertRaisesRegex(ValueError, 'RTE lowering SOURCE missing'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_D2_and_event_emitter_checked_at_sidecar_commit(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values(); real = G.C.blob
        edits = {'src/trotterlib/h4_d2_pauli_control.py': ('def central_control_body(body):', 'def central_control_body(body, extra=0):'),
                 'src/trotterlib/pauli_signal_adapter.py': ('def append_pauli_event(circuit, event, labels):',
                                                            'def append_pauli_event(circuit, event, labels, extra=0):')}
        for i, (target, (old, new)) in enumerate(edits.items()):
            fake = str(i)*40
            def blob(root, commit, path, fake=fake, target=target, old=old, new=new):
                if commit != fake:
                    return real(root, commit, path)
                text = real(root, PAULI_CODE['commit'], path)
                return text.decode().replace(old, new).encode() if path == target else text
            proofs = bind_route(req, prices, sides, pol, lowering_source=dict(PAULI_CODE, commit=fake))
            with patch.object(G.C, 'blob', side_effect=blob):
                with self.assertRaisesRegex(ValueError, 'emitter SOURCE'):
                    check(req, prices, sides, cohorts, pol, proofs)
        proofs = bind_route(req, prices, sides, pol,
                            lowering_source=dict(PAULI_CODE, commit='f2a2c3afd69d9013bc5f2e60973b2820ec8406b3'))
        self.assertEqual(len(check(req, prices, sides, cohorts, pol, proofs)['checked']), 4)

    def test_D_price_records_need_the_canonical_candidate_and_are_not_G_prices(self):
        """D's own price_records/legacy_physical_candidate (8b28dce0) against the C guard."""
        pol = route_policy(); v = pauli_fixture(); req = v[0]; config = req['configuration']
        def d_records(candidate):
            prices, sides, proofs, cohorts = [], [], {}, {}
            for row in config['rounds']:
                p, s, f = D_FACTS['price_records'](config, row['m'], trial_prices=dict(cosine=dict(rz_count=11), sine=dict(rz_count=12)),
                    scalar=2.25, physical_candidate=candidate(row), native_source=dict(DC.D_PAULI), cohort='D_PAULI_ARTIFICIAL',
                    price_source=dict(MOCK, m=row['m']), evidence_commit='b'*40, evidence_path='ARTIFICIAL.json', measurement=B.p.measurement)
                for side in s:  # fictional dictionary SOURCE has no blob; only the scalar SOURCE locator is substituted
                    side['implementation']['scalar_source'] = GT.CODE; f[side['implementation_evidence']['selector'][0]]['implementation']['scalar_source'] = GT.CODE
                prices += p; sides += s; proofs.update(f)
                for axis in B.p.AXES:
                    cohorts[(row['candidate']['id'], axis)] = 'D_PAULI_ARTIFICIAL'
            return prices, sides, proofs, cohorts
        canonical = lambda row: I.price_context(req, row['m'])['physical_candidate']
        prices, sides, proofs, cohorts = d_records(canonical)
        self.assertEqual(len(check(req, prices, sides, cohorts, pol, proofs)['checked']), 4)
        with patch.object(G, '_document', return_value=proofs), self.assertRaisesRegex(ValueError, 'Unknown compiled price kind'):
            PN.paper_cost(ROOT, config, PN.nominal_packet(ROOT, config, v[1], source=MOCK), prices, cohorts=cohorts,
                          price_bindings=sides, comparison_policy=pol)  # wiring pilot labels are not B price kinds
        legacy = lambda row: D_FACTS['legacy_physical_candidate'](dict(T=B.a.exact_scalar(row['T']), prefix=row['candidate']['prefix'],
            q=row['q_PF'], R=row['R_outer'], K=row['candidate']['K']), candidate_id=I.C.p.candidate_id)
        prices, sides, proofs, cohorts = d_records(legacy)
        with self.assertRaisesRegex(ValueError, 'Pauli canonical physical id'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_zz_and_pauli_lowerings_cannot_be_swapped(self):
        req, prices, sides, cohorts = pauli_values()
        bad = route_policy(); bad['by_method']['PR_PAULI_TAIL']['rte_rotation_lowering'] = G.NEW_ZZ
        proofs = bind_route(req, prices, sides, bad, lowering_source=GT.NEW_CODE)
        with self.assertRaisesRegex(ValueError, 'separate registered lowerings'):
            check(req, prices, sides, cohorts, bad, proofs)
        bad = route_policy(); bad['by_method']['PR']['rte_rotation_lowering'] = G.PAULI_D2
        bad['by_method']['FULL_H_RTE']['rte_rotation_lowering'] = G.PAULI_D2
        v, df_proofs = GT.bind(fixture('PR'))
        with self.assertRaisesRegex(ValueError, 'separate registered lowerings'):
            check(v[0], v[4], v[5]['price_bindings'], v[5]['cohorts'], bad, df_proofs)
        self.assertEqual(G.PAULI_D2, D.LOWERING_ID)

    def test_pauli_id_with_ZZ_emitter_rejected(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values()
        proofs = bind_route(req, prices, sides, pol, lowering_source=GT.NEW_CODE)
        with self.assertRaisesRegex(ValueError, 'emitter SOURCE'):
            check(req, prices, sides, cohorts, pol, proofs)

    def test_pauli_price_with_legacy_DF_physical_id_rejected_by_guard(self):
        pol = route_policy(); req, prices, sides, cohorts = pauli_values()
        for side in sides:
            c = side['physical_context']['physical_candidate']
            c['id'] = I.C.p.candidate_id({k: v for k, v in c.items() if k != 'id'})
        proofs = bind_route(req, prices, sides, pol)
        with self.assertRaisesRegex(ValueError, 'Pauli canonical physical id'):
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


class PauliBridge(unittest.TestCase):
    """C planner: canonical B candidate = physical ID; dictionary/law/SOURCE bound."""
    def setUp(self):
        req = fixture('PR')[0]; self.spec, self.payload = pauli_spec(req['configuration']['spec'])
        self.config = I.C.configure(ROOT, self.spec, api=API)

    def test_canonical_ids_dictionary_and_per_round_source(self):
        req = pauli_plan(self.config, self.payload)
        self.assertEqual(req['provider'], 'A_PR_PAULI'); self.assertEqual(req['pending']['bridge'], I.PAULI_BRIDGE)
        for row, job in zip(self.config['rounds'], req['pending']['jobs']):
            ctx = job['physical_context']
            self.assertEqual(ctx['physical_candidate']['id'], row['candidate']['id'])
            self.assertEqual(ctx['dictionary_identity'], self.spec['split']['tail_dictionary'])
            self.assertEqual(ctx['numerical_recipe']['source'], ROUND_SOURCES[str(row['m'])])
            self.assertIsNone(job['N_X']); self.assertIsNone(job['G'])
        self.assertEqual(req['science_calls'], 0)
        I._validate(ROOT, req)  # replanning reproduces the request
        with self.assertRaisesRegex(ValueError, 'Pauli means come from A'):
            I.produce_signals(ROOT, req, ROOT / 'unused')

    def test_legacy_DF_physical_id_never_substituted(self):
        legacy = {}
        for row in self.config['rounds']:
            c = dict(system='H4', method='PR', formula=self.spec['formula'], prefix=1, T=float(B.a.exact_scalar(row['T'])),
                     q=row['q_PF'], delta=float(B.a.exact_scalar(row['actual_delta_pf'])), R_outer=row['R_outer'], K=2)
            c['id'] = B.p.candidate_id(c); legacy[row['m']] = c
        with self.assertRaisesRegex(ValueError, 'no legacy-ID substitution'):
            pauli_plan(self.config, self.payload, pr_physical=legacy)
        with self.assertRaisesRegex(ValueError, 'Pauli-bridge inputs only'):
            I.plan(ROOT, fixture('PR')[0]['configuration'], target=fixture('PR')[0]['target'], signal_source=MOCK,
                   pr_physical=legacy, implementation_sources=deepcopy(ROUND_SOURCES))

    def test_dictionary_hash_lambda_scalar_and_law_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hash/identity differs'):
            pauli_plan(self.config, dict(self.payload, lambda_P=.25))
        spec = deepcopy(self.spec); spec['split']['tail_dictionary']['dictionary_sha256'] = '0'*64
        spec['law']['tail_dictionary'] = deepcopy(spec['split']['tail_dictionary'])
        with self.assertRaisesRegex(ValueError, 'hash/identity differs'):
            pauli_plan(I.C.configure(ROOT, spec, api=API), self.payload)
        spec = deepcopy(self.spec); spec['law']['tail_dictionary']['dictionary_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'one split/law Pauli dictionary'):
            pauli_plan(I.C.configure(ROOT, spec, api=API), self.payload)
        spec, payload = pauli_spec(fixture('PR')[0]['configuration']['spec'], scalar=.004)
        spec['split']['scalar'] = .005
        with self.assertRaisesRegex(ValueError, 'lambda/scalar SOURCE differs'):
            pauli_plan(I.C.configure(ROOT, spec, api=API), payload)
        spec, payload = pauli_spec(fixture('PR')[0]['configuration']['spec'])
        spec['split']['lambda_source'] = MOCK
        with self.assertRaisesRegex(ValueError, 'lambda/scalar SOURCE differs'):
            pauli_plan(I.C.configure(ROOT, spec, api=API), payload)

    def test_per_round_SOURCE_map_required_for_every_round(self):
        for bad in (None, {'0': ROUND_SOURCES['0']}, dict(ROUND_SOURCES, **{'1': dict(commit='short', path='x')})):
            with self.assertRaisesRegex(ValueError, 'per-round numerical SOURCE map'):
                pauli_plan(self.config, self.payload, implementation_sources=bad)

    def test_level2_compare_binds_physical_context_and_round_source(self):
        v = pauli_fixture(); pol = route_policy()
        proofs = bind_route(v[0], v[4], v[5]['price_bindings'], pol); v[5]['comparison_policy'] = pol
        out = GT.compare(v, proofs)
        self.assertEqual(len(out['price_implementation_check']['checked']), 4)
        self.assertEqual(out['request']['provider'], 'A_PR_PAULI')
        bad = deepcopy(v); bad[1]['means'][1]['implementation_source'] = deepcopy(ROUND_SOURCES['0'])
        with self.assertRaisesRegex(ValueError, 'numerical SOURCE differs'):
            GT.compare(bad, proofs)
        bad = deepcopy(v)  # DF-style context without dictionary identity/numerical recipe
        bad[1]['means'][0]['provider_binding']['physical_context'] = I.price_context(v[0], 0)
        with self.assertRaisesRegex(ValueError, 'physical context'):
            GT.compare(bad, proofs)

    def test_D_round_request_shares_schedule_facts(self):
        req = pauli_plan(self.config, self.payload); rows = DC.pauli_requests(ROOT, req)
        self.assertEqual(len(rows), 2*len(self.config['rounds']))
        for row in rows:
            c = row['physical_context']['physical_candidate']; d = row['d_pauli_round']
            self.assertEqual((d['candidate_id'], d['q'], d['R_outer'], d['K']), (c['id'], c['q'], c['R_outer'], c['K']))
            self.assertEqual(row['expected_lowering_id'], G.PAULI_D2); self.assertIsNone(row['mean'])
        spec = deepcopy(self.spec); spec['split']['lambda_R'] = dict(numerator=1, denominator=3)
        with self.assertRaises(ValueError):
            DC.pauli_requests(ROOT, pauli_plan(I.C.configure(ROOT, spec, api=API), self.payload))


class PauliPaperLevel(unittest.TestCase):
    """Artificial Pauli price records through the level-1 path (guard first)."""
    def paper(self, v, proofs, packet=None):
        packet = packet or PN.nominal_packet(ROOT, v[0]['configuration'], v[1], source=MOCK)
        with patch.object(G, '_document', return_value=proofs), \
                patch.object(B, 'cost', side_effect=AssertionError('paper level must not call B.cost')):
            return PN.paper_cost(ROOT, v[0]['configuration'], packet, v[4], cohorts=v[5]['cohorts'],
                                 price_bindings=v[5]['price_bindings'], comparison_policy=route_policy())

    def bound(self):
        v = pauli_fixture(); proofs = bind_route(v[0], v[4], v[5]['price_bindings'], route_policy()); return v, proofs

    def test_pauli_price_accepted_nominal_separate_from_sufficient(self):
        v, proofs = self.bound(); packet = PN.nominal_packet(ROOT, v[0]['configuration'], v[1], source=MOCK)
        expected = [R.nominal_axis_shots(1, m['m'], m['normalization']['log_B'])['nominal_N_X'] for m in v[1]['means']]
        self.assertEqual([r['nominal_N_X'] for r in packet['rounds']], expected)
        self.assertTrue(all(r['sufficient_N_X'] is None and r['sufficient_N_Y'] is None for r in packet['rounds']))
        out = self.paper(v, proofs, packet)
        self.assertEqual(out['route'], 'PR_PAULI_TAIL'); self.assertEqual(out['status'], 'PAPER_NOMINAL_G_QUALIFIED_EXPECTATION')
        self.assertEqual(B.a.exact_scalar(out['G_paper']), 2*(expected[0]*1+expected[1]*100))
        self.assertIsNone(out['G_qualified']); self.assertIsNone(out['sufficient_N'])
        out2 = GT.compare([*v[:5], dict(v[5], comparison_policy=route_policy())], proofs)  # level 2, same guard
        self.assertEqual(out2['price_implementation_check'], out['price_implementation_check'])

    def test_direct_model_finite_cohort_and_missing(self):
        v, proofs = self.bound()
        for price in v[4]:
            price['expectation'] = 'FINITE_COHORT_ONLY'
        self.assertEqual(self.paper(v, proofs)['status'], 'PAPER_NOMINAL_G_FINITE_COHORT_SCENARIO')
        v, proofs = self.bound()
        for price in v[4][:2]:
            price.update(classification='MODEL_PREDICTION', model=dict(qualified_for_candidate=False))
        self.assertEqual(self.paper(v, proofs)['status'], 'PAPER_NOMINAL_G_UNQUALIFIED_MODEL')
        v, proofs = self.bound()
        for price in v[4]:
            price.update(classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC')
        with self.assertRaisesRegex(ValueError, 'outside deterministic PF'):
            self.paper(v, proofs)
        v, proofs = self.bound(); v[4].pop()
        out = self.paper(v, proofs)
        self.assertIsNone(out['G_paper']); self.assertEqual(out['status'], 'NOT_EVALUATED_MISSING_PRICE_OR_N')

    def test_pauli_price_rejected_on_dictionary_law_or_source_mismatch(self):
        v, proofs = self.bound()
        for side in v[5]['price_bindings']:
            side['physical_context']['split']['tail_dictionary']['dictionary_sha256'] = '0'*64
        proofs = bind_route(v[0], v[4], v[5]['price_bindings'], route_policy())
        with self.assertRaisesRegex(ValueError, 'split/scalar construction differs'):
            self.paper(v, proofs)
        v, proofs = self.bound()
        for side in v[5]['price_bindings']:
            side['physical_context']['law']['tail_dictionary']['prefix'] = 2
        proofs = bind_route(v[0], v[4], v[5]['price_bindings'], route_policy())
        with self.assertRaisesRegex(ValueError, 'event law differs'):
            self.paper(v, proofs)
        v, proofs = self.bound(); df = fixture('PR')[0]['configuration']
        for price in v[4]:
            price['law_binding'] = deepcopy(df['rounds'][price['identity']['m']]['law_binding'])
        with self.assertRaisesRegex(ValueError, 'law binding differs'):
            self.paper(v, proofs)
        v, proofs = self.bound(); proofs[next(iter(proofs))]['price_source'] = dict(MOCK, other=True)
        with self.assertRaisesRegex(ValueError, 'Pinned implementation evidence'):
            self.paper(v, proofs)


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

    def test_price_kind_names_must_be_B_enums(self):
        v, proofs = self.guarded('FULL')
        for price in v[4]:  # D 8b28dce0 Full PF/Truncated caller labels
            price.update(classification='DIRECT_DETERMINISTIC', expectation='EXACT_DETERMINISTIC_CIRCUIT',
                         price_status='UNQUALIFIED_FOR_C_MODEL')
        with self.assertRaisesRegex(ValueError, 'Unknown compiled price kind'):
            self.paper(v, proofs)

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


class RealPauliBridge(unittest.TestCase):
    """Pinned public metadata only: A's LD1/LD2 canonical configurations and carriers."""
    HANDOFF = dict(commit=I.A_PAULI_SOURCE, path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')
    IDS = {1: 'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27',
           2: 'd7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'}

    @classmethod
    def setUpClass(cls):
        root = str(ROOT); cls.packets = I._pinned_json(root, cls.HANDOFF['commit'], cls.HANDOFF['path'])['packets']
        cls.records = {}
        for packet in cls.packets:
            for row in packet['rounds']:
                rec = I._pinned_json(root, row['mean_source']['commit'], row['mean_source']['path'])
                for key in row['mean_source']['selector']:
                    rec = rec[key]
                # metadata fields only; the signal/allowance values are not read
                cls.records[packet['prefix'], row['m']] = {k: rec[k] for k in ('source', 'physical_context', 'B_candidate_id',
                    'B_identity', 'B_law_binding', 'B_configuration_id', 'candidate')}

    def plan(self, packet):
        target = I._pinned_json(str(ROOT), '91349e703e0d09ef702ccd108745dfaf1acc29f7',
            'docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD%d.json' % packet['prefix'])['target']
        sources = {str(r['m']): self.records[packet['prefix'], r['m']]['source'] for r in packet['rounds']}
        return I.plan(ROOT, packet['configuration'], target=target, signal_source=self.HANDOFF, implementation_sources=sources)

    def test_all_13_rounds_match_A_records_and_B_canonical_ids(self):
        self.assertEqual([p['prefix'] for p in self.packets], [1, 2])
        for packet in self.packets:
            config = packet['configuration']; req = self.plan(packet)
            self.assertEqual(config['id'], self.IDS[packet['prefix']]); self.assertEqual(len(req['pending']['jobs']), 13)
            for row, job in zip(config['rounds'], req['pending']['jobs']):
                rec = self.records[packet['prefix'], row['m']]; c = job['physical_context']['physical_candidate']
                self.assertEqual(job['physical_context'], rec['physical_context'])
                self.assertEqual((job['B_candidate_id'], c['id']), (rec['B_candidate_id'], row['candidate']['id']))
                self.assertEqual((job['B_identity'], job['B_law_binding']), (rec['B_identity'], rec['B_law_binding']))
                self.assertNotEqual(I.C.p.candidate_id({k: v for k, v in c.items() if k != 'id'}), c['id'])
                # B's intake view (HANDOFF field map) passes C's provider-binding check
                ref = packet['rounds'][row['m']]['mean_source']; src = {k: ref[k] for k in ('commit', 'path', 'selector')}
                I.provider_binding_check(req, dict(m=row['m'], source=src, implementation_source=rec['source'],
                    provider_binding=dict(physical_context=rec['physical_context'], numerical_source=src,
                                          implementation_source=rec['source'])))
            self.assertEqual({j['physical_context']['numerical_recipe']['source']['commit'] for j in req['pending']['jobs']},
                             {'05447d1d4f1e3c0e9e335a7085aa58a877c139f2', '7084fbc9a43cf7b7e24681b4005d37cb59c1e465'})

    def test_D_round_requests_agree_for_26_rounds(self):
        for packet in self.packets:
            rows = DC.pauli_requests(ROOT, self.plan(packet))
            self.assertEqual(len(rows), 26); self.assertTrue(all(r['mean'] is None for r in rows))

    def test_real_round_source_swap_rejected(self):
        packet = self.packets[0]; req = self.plan(packet); rec = self.records[1, 3]; other = self.records[1, 0]['source']
        self.assertNotEqual(rec['source'], other)
        with self.assertRaisesRegex(ValueError, 'numerical SOURCE differs'):
            I.provider_binding_check(req, dict(m=3, source=dict(MOCK), implementation_source=other,
                provider_binding=dict(physical_context=rec['physical_context'], numerical_source=dict(MOCK), implementation_source=other)))


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
