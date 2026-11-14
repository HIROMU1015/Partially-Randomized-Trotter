"""Pauli LD1/m0 pilot, parent side, on C's bridge planner/guard/paper accounting (c628e9e8 in tree).

Metadata, saved P0 representation check and FICTIONAL prices only. No sampling,
build, compile or real price. The artificial circuits run in their own
provider-root process (see check_pauli_ld1_m0_pilot_circuits.py).
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_pauli_ld1_m0_pilot as PILOT  # noqa: E402
from trottertracks import price_implementation_guard as G  # noqa: E402
from trottertracks import rpe_optimization_pipeline as I  # noqa: E402
from trottertracks import paper_nominal_cost as PN  # noqa: E402
P = PILOT.connection()
B = I.providers(str(ROOT)).B
SOURCE = dict(fixture='ARTIFICIAL_PRICE_NOT_MEASURED', commit='1e5c2c15bef7c1ccb92f8d7b3b75805e30019f60')


def guard():
    return G


def fictional(meta):
    return dict(prices={a: dict(rz_count=500 + i) for i, a in enumerate(('cosine', 'sine'))}, scalar=meta['scalar'])


def policy(config, **kw):
    entry = dict(native_policy=P.NATIVE_POLICY, rte_rotation_lowering=G.PAULI_D2, scalar_quantity=P.SCALAR_QUANTITY)
    entry.update(kw)
    return dict(id='ARTIFICIAL_PAULI', source=dict(fixture='ARTIFICIAL'), profile=deepcopy(config['spec']['profile']),
                by_method=dict(PR_PAULI_TAIL=entry))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = PILOT.preflight()
        cls.req, cls.packet = PILOT.canonical_request()
        cls.config = cls.req['configuration']
        cls.prices, cls.sides, cls.proofs = PILOT.records(ROOT, fictional(cls.meta), source=SOURCE,
                                                          evidence_commit='a' * 40, evidence_path='ARTIFICIAL.json')
        cls.cohorts = {(cls.config['rounds'][0]['candidate']['id'], a): cls.meta['cohort'] for a in ('X', 'Y')}

    def check(self, sides=None, proofs=None, prices=None, pol=None):
        with patch.object(G, '_document', return_value=proofs or self.proofs):
            return G.check(ROOT, self.config, prices or self.prices, cohorts=self.cohorts, bindings=sides or self.sides,
                           policy=pol or policy(self.config))

    def mutated(self, fn):
        sides, proofs = deepcopy(self.sides), deepcopy(self.proofs)
        for s in sides:
            fn(s)
            proofs['m0' + s['axis']] = {k: deepcopy(s[k]) for k in proofs['m0' + s['axis']]}
        return sides, proofs

    def test_canonical_candidate_from_C_planner_and_unchanged_seed(self):
        c = self.meta['canonical_physical_candidate']
        self.assertEqual(self.meta['configuration_id'], 'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27')
        self.assertEqual(c, I.price_context(self.req, 0)['physical_candidate'])
        self.assertEqual(c['id'], 'RMSE:62a9a6b68c2c88951f1055d1da7f9b6a89da94a91092c9488cf5fce503288e54')
        self.assertEqual((c['T'], c['q'], c['R_outer'], c['K'], c['prefix']), (.1, 1, 3, 2, 1))
        self.assertEqual(self.meta['route'], G.route(B, self.config['spec']))
        self.assertEqual(self.meta['seeds'], [14788159701244282917])
        self.assertEqual(self.meta['seeds'], [P.seed(c['id'], self.meta['cohort'], 0, 0, 0)])
        legacy = dict(system='H4', method='PR', formula='2nd', prefix=1, T=.1, q=1, delta=.1, R_outer=3, K=2)
        legacy['id'] = I.C.p.candidate_id(legacy)
        self.assertNotEqual(legacy['id'], c['id'])
        self.assertNotEqual(P.seed(legacy['id'], self.meta['cohort'], 0, 0, 0), self.meta['seeds'][0])

    def test_records_use_B_finite_sample_kinds_and_are_not_expectations(self):
        for p in self.prices:
            self.assertEqual((p['classification'], p['expectation'], p['n_paths']), ('DIRECT_COMPILED_FINITE_SAMPLE', 'FINITE_COHORT_ONLY', 1))
            self.assertIn(p['classification'], B.a.PRICE_KINDS)
            self.assertEqual(p['diagnostic']['pilot_role'], 'DIRECT_SINGLE_PATH_WIRING')
            self.assertFalse(p['diagnostic']['law_expectation'])
            self.assertEqual(p['law_context']['ideal_sampler_distance_qualified'], False)
            self.assertEqual(p['physical_context'], I.price_context(self.req, 0))
            self.assertEqual(p['record_schema'], P.RECORD_SCHEMA)
        self.assertEqual(PN._tier(B, self.prices[0], 'PR'), 'FINITE_COHORT_SCENARIO')
        promoted = dict(deepcopy(self.prices[0]), expectation='QUALIFIED_EXPECTATION')
        self.assertEqual(PN._tier(B, promoted, 'PR'), 'FINITE_COHORT_SCENARIO')  # sampler law not qualified
        det = dict(deepcopy(self.prices[0]), classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC')
        with self.assertRaises(ValueError):
            PN._tier(B, det, 'PR')
        for old in (dict(classification='DIRECT_SINGLE_PATH_WIRING', expectation='NOT_A_LAW_MEAN'),):
            with self.assertRaisesRegex(ValueError, 'Unknown compiled price kind'):
                PN._tier(B, dict(deepcopy(self.prices[0]), **old), 'PR')

    def test_C_guard_accepts_canonical_sidecar(self):
        self.assertEqual(len(self.check()['checked']), 2)
        self.assertEqual(P.LOWERING_ID, G.PAULI_D2)

    def test_legacy_physical_id_rejected(self):
        legacy = dict(system='H4', method='PR', formula='2nd', prefix=1, T=.1, q=1, delta=.1, R_outer=3, K=2)
        legacy['id'] = I.C.p.candidate_id(legacy)
        sides, proofs = self.mutated(lambda s: s['physical_context'].update(physical_candidate=deepcopy(legacy)))
        with self.assertRaises(ValueError):
            self.check(sides, proofs)

    def test_dictionary_law_source_scalar_route_lowering_mismatches_rejected(self):
        def dict_hash(s):
            s['physical_context']['split']['tail_dictionary']['dictionary_sha256'] = '0' * 64
        def law(s):
            s['physical_context']['law']['tail_dictionary'] = dict(s['physical_context']['law']['tail_dictionary'], prefix=2)
        def native(s):
            s['implementation']['native_source'] = dict(path=P.MODULE)
        def wrapper(s):
            s['implementation']['wrapper_source'] = dict(commit=None, path='x')
        def scalar(s):
            s['implementation']['resolved_scalar'] = s['implementation']['resolved_scalar'] + 1e-12
        def route(s):
            s['implementation'].pop('route')
        def lowering(s):
            s['implementation'].update(rte_rotation_lowering=G.NEW_ZZ)
        def emitter(s):
            s['implementation'].update(rte_lowering_source=dict(commit='01e068af7e06d9f1f5621a5b3800c30eb32ecb06',
                                                                path='src/trottertracks/efficient_controlled_zz.py'))
        def quantity(s):
            s['implementation'].update(scalar_quantity='HAMILTONIAN_CONSTANT')
        for fn in (dict_hash, law, native, wrapper, scalar, route, lowering, emitter, quantity):
            sides, proofs = self.mutated(fn)
            with self.assertRaises((ValueError, subprocess.CalledProcessError), msg=fn.__name__):
                self.check(sides, proofs)
        with self.assertRaises(ValueError):
            self.check(pol=policy(self.config, rte_rotation_lowering=G.NEW_ZZ))
        dfpol = policy(self.config); dfpol['by_method'] = dict(PR=dfpol['by_method']['PR_PAULI_TAIL'])
        with self.assertRaises(ValueError):
            self.check(pol=dfpol)

    def test_paper_level_accounting_accepts_shape_and_marks_finite_cohort(self):
        means = [dict(m=r['m'], identity=r['identity'], law_binding=r['law_binding'],
                      normalization=dict(log_B=.04)) for r in self.config['rounds']]
        packet = PN.nominal_packet(ROOT, self.config, dict(configuration_id=self.config['id'], target={}, source=SOURCE, means=means),
                                   source=SOURCE)
        with patch.object(G, '_document', return_value=self.proofs):
            out = PN.paper_cost(ROOT, self.config, packet, self.prices, cohorts=self.cohorts, price_bindings=self.sides,
                                comparison_policy=policy(self.config))
        self.assertEqual(out['status'], 'NOT_EVALUATED_MISSING_PRICE_OR_N')  # only m0 has (fictional) prices
        self.assertEqual([c['tier'] for c in out['cells'] if c['m'] == 0], ['FINITE_COHORT_SCENARIO'] * 2)
        self.assertIsNone(out['G_paper']); self.assertIsNone(out['G_qualified'])

    @unittest.skipUnless('--include-saved-numerical' in sys.argv,
                         'Saved H4 numerical revalidation requires a separately permitted scope')
    def test_saved_P0_bit_order_for_A(self):
        import json
        ops, blocks, basis = PILOT.p0_arrays()
        d = json.loads(PILOT.blob(PILOT.P0['commit'], PILOT.P0['directory'] + 'LD1_DICTIONARY.json'))
        out = P.bit_order_checks(d, ops, blocks)
        self.assertTrue(out['pass']); self.assertGreater(out['unconverted_mismatch'], 1.)

    def test_artificial_circuits_in_provider_root_process(self):
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / 'tests/check_pauli_ld1_m0_pilot_circuits.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn('Ran 5 tests', out.stderr)

    def test_recovery_import_chain_and_B0_default_compatibility(self):
        import ast
        import json
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / PILOT.CALLER), '--import-preflight'],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        checked = json.loads(out.stdout)
        self.assertEqual(checked['status'], 'RESTORE_IMPORT_CHAIN_READY_NO_SCIENCE')
        self.assertTrue(all(checked[k] == 0 for k in ('input_restores', 'preparations', 'sampling', 'circuit_builds', 'compiles')))
        self.assertEqual(len(checked['references']), 2)
        old = ast.parse(PILOT.blob('62ae1bd56718287de5d958cf0ed6411f5b7395c9', PILOT.CALLER))
        new = ast.parse((ROOT / PILOT.CALLER).read_bytes())
        # B1 adds prefix/seed/index/diagnostic arguments to canonical_request
        # and pilot_core. Their default behavior is compared on toy circuits
        # against the frozen B0 implementation in the B1 circuit tests.
        for name in ('child', 'records', 'preflight'):
            f = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(f(old)), ast.dump(f(new)), name)
        import inspect
        self.assertEqual(inspect.signature(PILOT.canonical_request).parameters['prefix'].default, 1)
        self.assertEqual(inspect.signature(PILOT.pilot_core).parameters['index'].default, 0)
        self.assertIsNone(inspect.signature(PILOT.pilot_core).parameters['step_seeds'].default)
        self.assertFalse(inspect.signature(PILOT.pilot_core).parameters['event_metadata'].default)
        self.assertEqual(PILOT.OUTPUT, 'docs/research/d_pauli_ld1_m0_pilot_results_20261114/run02')
        self.assertTrue(str(PILOT.PRIVATE_OUTPUT).endswith('/run02'))


if __name__ == '__main__':
    if '--include-saved-numerical' not in sys.argv:
        # The saved-H4 operator check is numerical revalidation, not a metadata
        # regression. Keep it available for separately authorized work, but do
        # not run it in B1 technical preparation.
        names = [n for n in unittest.defaultTestLoader.getTestCaseNames(Tests)
                 if n != 'test_saved_P0_bit_order_for_A']
        suite = unittest.TestSuite(Tests(n) for n in names)
        sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    sys.argv.remove('--include-saved-numerical')
    unittest.main(verbosity=2)
