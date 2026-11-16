"""B2's actual JSON entry: saved artifacts and an explicitly artificial new split.

No real H arrays, mean, normalizer, price prediction or Q3 are executed.
"""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import prepare_pauli_candidate as RUN
from trottertracks import pauli_candidate_preparation as A
from trottertracks import pauli_split_descriptor as D

E = A.E
NEW_PACKET = dict(commit='1537dfdf9d7fbf52c24609ed3a188381cc24eb8c',
    path='docs/research/b2_pauli_kappa_signals_20261116/packets/LD3_S2_delta_h_kappa_x2/MEAN_HANDOFF.json')


def entry(request, tmp, **kwargs):
    req, out = Path(tmp)/'REQUEST.json', Path(tmp)/'OUT.json'
    req.write_text(json.dumps(request))
    with patch('sys.stdout', new=io.StringIO()):
        result = RUN.main(['--request', str(req), '--output', str(out)], **kwargs)
    return result, json.loads(out.read_text())


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = A.Assets(ROOT)

    def no_science(self):
        stack = ExitStack()
        for target in ((np, 'load'), (E.p0, 'load_saved'), (E.provider, 'evaluate'),
                       (D, 'derive_descriptor')):
            stack.enter_context(patch.object(*target, side_effect=AssertionError('No H4 numerical leaf')))
        return stack

    def test_A_saved_splits_from_B2_entry_independent_of_price_catalog(self):
        request = dict(kind='pauli_native', items=[dict(item_id='LD%d' % p,
            configuration=dict(L_D=p), changes=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
            for p in (1, 2, 3, 5)])
        with self.no_science(), tempfile.TemporaryDirectory() as tmp:
            value, serialized = entry(request, tmp, assets_factory=lambda *a, **k: self.assets)
        self.assertEqual(serialized, value)
        for row in value['items']:
            p = row['L_D']; expected, loc = self.assets.baselines[p]
            self.assertEqual(row['preparation_status'], 'REUSED_CONFIGURATION', row.get('error'))
            self.assertEqual(row['configuration'], expected)
            self.assertEqual(row['configuration_source'], loc)
            self.assertEqual(row['dictionary_identity'], expected['spec']['split']['tail_dictionary'])
            self.assertEqual(row['signal_plan']['reuse_rounds'], list(range(13)))
            self.assertEqual(row['signal_plan']['request_rounds'], [])
            self.assertEqual([r['physical_id'] for r in row['schedule']], [r['candidate']['id'] for r in expected['rounds']])
            self.assertTrue(row['readiness']['configuration_complete'])
        self.assertFalse(value['items'][0]['readiness']['price_26axis_complete'])
        # The exact published B2 NEEDS LD1 row used to hide this saved configuration.
        needs = E.read_ref(ROOT, dict(commit=A.B2,
            path='docs/research/rpe_method_optimizer_h4_20261116/run04/NEEDS.json', selector=['candidates', 29]))
        with self.no_science(), tempfile.TemporaryDirectory() as tmp:
            value, _ = entry(needs, tmp, assets_factory=lambda *a, **k: self.assets)
        self.assertEqual(value['items'][0]['configuration_id'], self.assets.baselines[1][0]['id'])

    def test_B_published_changed_kappa_same_dictionary_and_no_duplicate_means(self):
        saved = A.Assets(ROOT, packets=[NEW_PACKET])
        request = dict(kind='pauli_native', view=str(ROOT), view_commit='4e5b8bef6', items=[dict(
            item_id='LD3-kappa-x2', base=saved.baselines[3][1],
            changes=dict(formula='2nd', delta_over_h='1', kappa_multiplier='2'))])
        with self.no_science(), tempfile.TemporaryDirectory() as tmp:
            value, _ = entry(request, tmp, assets_factory=lambda *a, **k: saved)
        row = value['items'][0]
        self.assertEqual(row['preparation_status'], 'REUSED_CONFIGURATION', row.get('error'))
        self.assertEqual(row['configuration_id'], '23ec179150682e0af256fc7c7bfcdf6f39af772f62418d6b45326afd5da1b2a8')
        self.assertEqual(row['configuration'], saved.read(row['configuration_source']))
        self.assertEqual(row['status'], 'GENERATED')
        self.assertEqual(row['structure'], A.generator(str(ROOT))[1].structure_checks(row['configuration'])[1])
        self.assertEqual(row['dictionary_identity'], saved.baselines[3][0]['spec']['split']['tail_dictionary'])
        self.assertEqual(row['signal_plan']['reuse_rounds'], list(range(13)))
        self.assertEqual(row['signal_plan']['request_rounds'], [])
        # Without that extra carrier, only physically identical early rounds are reused.
        no_new = A.prepare(request, self.assets)['items'][0]['signal_plan']
        self.assertEqual(no_new['reuse_rounds'], [0, 1])
        self.assertEqual(no_new['request_rounds'], list(range(2, 13)))

    def test_exact_one_fifth_and_binary64_point_two_remain_distinct(self):
        req = dict(items=[dict(item_id='exact', configuration=dict(L_D=3), changes=dict(delta_pf='1/5')),
                          dict(item_id='binary64', configuration=dict(L_D=3), changes=dict(delta_pf=0.2))])
        with self.no_science(), tempfile.TemporaryDirectory() as tmp:
            value, _ = entry(req, tmp, assets_factory=lambda *a, **k: self.assets)
        x, y = value['items']
        self.assertEqual(A.rational(x['configuration']['spec']['delta_pf']), Fraction(1, 5))
        self.assertEqual(A.rational(y['configuration']['spec']['delta_pf']), Fraction(.2))
        self.assertNotEqual(x['configuration_id'], y['configuration_id'])
        self.assertTrue(x['signal_plan']['unsupported_rounds'])
        self.assertEqual(y['signal_plan']['unsupported_rounds'], [])
        self.assertEqual(x['request']['changes']['delta_pf'], '1/5')
        self.assertEqual(y['request']['changes']['delta_pf'], .2)

    def test_signed_formula_K_and_boundaries_use_B2_mapper(self):
        base = self.assets.baselines[3][0]
        req = dict(items=[dict(item_id='M5', configuration=dict(L_D=3),
                              changes=dict(formula='4th(m5_best)', K=4, kappa_multiplier='3/2'))])
        with self.no_science():
            row = A.prepare(req, self.assets)['items'][0]
        self.assertEqual(row['preparation_status'], 'GENERATED_CONFIGURATION', row.get('error'))
        b, build = A.generator(str(ROOT))
        expected = build(base, dict(formula='4th(m5_best)', kappa_multiplier='3/2'))
        spec = deepcopy(expected['spec']); spec['finite_K'] = 4
        self.assertEqual(row['configuration'], b.generate(spec))
        for r in row['schedule']:
            self.assertEqual(len(r['structure']['allocation']), 11)
            self.assertEqual(sum(r['structure']['allocation']), r['R_outer'])
            self.assertEqual(r['structure']['event_count'], r['q_PF']*r['R_outer'])
        self.assertEqual([r['m'] for r in row['schedule']], list(range(13)))

    def test_missing_real_split_only_plans_and_reports_missing_input(self):
        with self.no_science():
            rows = A.prepare(dict(items=[dict(item_id=str(p), configuration=dict(L_D=p))
                                       for p in (4, 7, 9)]), self.assets)['items']
        self.assertEqual([r['status'] for r in rows], ['NEEDS_SPLIT_PREPARATION']*3)
        self.assertEqual(rows[0]['next_actions'][0], 'DERIVE_DESCRIPTOR')
        self.assertEqual(rows[1]['next_actions'][0], 'REUSE_DESCRIPTOR')
        bad = A.prepare(dict(items=[dict(configuration=dict(L_D=10)),
                                   dict(configuration=dict(L_D=3), snapshot='WRONG')]), self.assets)['items']
        self.assertTrue(all(r['missing_inputs'] and r['configuration'] is None for r in bad))

    def test_missing_price_source_does_not_discard_completed_configuration(self):
        with self.no_science(), patch.object(A, 'price_input', side_effect=FileNotFoundError('D2 candidate evidence')):
            row = A.prepare(dict(items=[dict(configuration=dict(L_D=1))]), self.assets)['items'][0]
        self.assertEqual(row['status'], 'GENERATED')
        self.assertEqual(row['configuration_id'], self.assets.baselines[1][0]['id'])
        self.assertTrue(row['readiness']['configuration_complete'])
        self.assertFalse(row['readiness']['price_26axis_complete'])
        self.assertIn('D2 candidate evidence', row['price_handoff']['needs'][0])

    def test_missing_signal_source_retains_configuration_without_new_requests(self):
        with self.no_science(), patch.object(self.assets, 'signal_plan', side_effect=FileNotFoundError('saved packet')):
            row = A.prepare(dict(items=[dict(configuration=dict(L_D=1))]), self.assets)['items'][0]
        self.assertEqual(row['status'], 'GENERATED')
        self.assertTrue(row['readiness']['configuration_complete'])
        self.assertFalse(row['readiness']['signal_13round_complete'])
        self.assertEqual(row['signal_plan']['request_rounds'], [])
        self.assertEqual(row['signal_plan']['planning_status'], 'SOURCE_UNRESOLVED')

    def test_prepared_descriptor_context_reaches_actual_signal_dispatch(self):
        """Check the CLI context encloses plan AND execution/assembly, no science leaf."""
        import run_pauli_signal_entry as signal
        toy = deepcopy(self.assets.baselines[3][0]); toy['spec']['L_D'] = 4
        descriptor = dict(prefix=4, binding=self.assets.binding,
            descriptor=dict(scalar=.125, lambda_R=.75), source=dict(commit='ARTIFICIAL', path='descriptor.json'))
        previous = E.L.p0_metadata
        seen = []
        def dispatch(args, packets):
            value = E.L.p0_metadata(ROOT, (2, 4))
            seen.append((args.plan, args.execute, args.assemble))
            self.assertEqual(value['saved_tail_descriptors']['4'], descriptor['descriptor'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'DESC.json'; path.write_text(json.dumps(descriptor))
            for mode in ('--plan', '--execute', '--assemble'):
                argv = ['signal', mode, '--config', 'ARTIFICIAL', '--tail-descriptor', str(path),
                        '--source', 'ARTIFICIAL', '--out', str(Path(tmp)/'out')]
                with self.no_science(), patch.object(sys, 'argv', argv), \
                     patch.object(E, 'load_configuration', return_value=(toy, None)), \
                     patch.object(E, 'read_ref', return_value=descriptor['descriptor']), \
                     patch.object(signal, 'dispatch', side_effect=dispatch):
                    signal.main()
                self.assertIs(E.L.p0_metadata, previous)
        self.assertEqual(seen, [(True, False, False), (False, True, False), (False, False, True)])

    def test_materializer_is_lazy_so_reuse_survives_p0_carrier_commit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with patch.object(subprocess, 'check_output', side_effect=AssertionError('No source check before leaf')):
                leaf = RUN.materializer(ROOT, Path(tmp)/'p0', 'OLDER_SCIENCE_SOURCE')
            request = dict(items=[dict(item_id='cached-LD1', configuration=dict(L_D=1))])
            row = A.prepare(request, self.assets, materialize=leaf)['items'][0]
            self.assertEqual(row['preparation_status'], 'REUSED_CONFIGURATION')
        # A source mismatch is still rejected when a genuinely new split is requested.
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            leaf = RUN.materializer(ROOT, Path(tmp)/'p0', 'OLDER_SCIENCE_SOURCE',
                                    builder_id='P0_LEGACY_OPENFERMION_DEFAULT_V1')
            with patch.object(subprocess, 'check_output', return_value=b'CURRENT_CARRIER\n'):
                with self.assertRaisesRegex(ValueError, 'FIXED_CURRENT'):
                    leaf(4, dict(descriptor=None))
            # A used preparation file may not drift behind the claimed scientific SOURCE.
            with patch.object(subprocess, 'check_output', return_value=b'CLAIMED_SOURCE\n'), \
                 patch.object(E.p0, 'blob', return_value=b'different source bytes'), self.no_science():
                leaf = RUN.materializer(ROOT, Path(tmp)/'p0', 'CLAIMED_SOURCE',
                                        builder_id='P0_LEGACY_OPENFERMION_DEFAULT_V1')
                with self.assertRaisesRegex(ValueError, 'USED_PREPARATION_SOURCE_CHANGED'):
                    leaf(4, dict(descriptor=None))

    def test_local_metadata_cache_reopens_through_fresh_B2_entry(self):
        request = dict(items=[dict(item_id='MOCK-SCHEDULE-NO-SCIENCE', configuration=dict(L_D=3),
                                  changes=dict(formula='4th(m5_best)', K=4))])
        with self.no_science(), tempfile.TemporaryDirectory() as tmp:
            req, out, cache = (Path(tmp)/s for s in ('REQ.json', 'OUT.json', 'SPLITS.json'))
            req.write_text(json.dumps(request))
            argv = ['--request', str(req), '--output', str(out), '--cache', str(cache)]
            with patch('sys.stdout', new=io.StringIO()):
                first = RUN.main(argv)['items'][0]
                second = RUN.main(argv)['items'][0]
            self.assertEqual(first['preparation_status'], 'GENERATED_CONFIGURATION')
            self.assertEqual(second['preparation_status'], 'REUSED_LOCAL_METADATA_CONFIGURATION')
            self.assertEqual(first['configuration'], second['configuration'])
            self.assertIsNone(second['configuration_source'])  # no fabricated immutable commit

    def test_C_unregistered_artificial_split_create_then_reuse_through_entry(self):
        assets = ToyAssets(self.assets)
        request = dict(items=[dict(item_id='TOY-LD4', configuration=dict(L_D=4),
                                  changes=dict(formula='2nd', delta_over_h='1', kappa_multiplier='2'))])
        made = []
        def create(prefix, descriptor):
            made.append(prefix)
            # The very same historical descriptor helper and P0 global JW collector, on a toy only.
            d = D.derive_descriptor(ROOT, assets.ham, prefix)
            op = E.p0.square_factor_dictionary(ROOT, assets.ham, prefix)
            model = E.p0.pauli_api().PauliHamiltonian.from_openfermion(op, num_qubits=2, model_id='ARTIFICIAL')
            lam = sum(abs(c) for _, c in model.terms)
            payload = dict(prefix=prefix, binding=assets.binding, lambda_P=lam, pauli_identity=model.constant,
                saved_DF_scalar=d['scalar'], mapping='ARTIFICIAL_QUBIT_ORDER',
                terms=[dict(label_qiskit=k, coefficient=v) for k, v in model.terms])
            self.assertAlmostEqual(model.constant, d['scalar'])
            item = dict(payload=payload, dictionary_sha256=E.L.canonical_sha(payload),
                source=dict(commit='ARTIFICIAL', path='LD4_DICTIONARY.json'), P0_metrics={'artificial': True},
                P0_result_source=dict(commit='ARTIFICIAL', path='P0.json'))
            return dict(status='SPLIT_READY', item=item, validation_scope='ARTIFICIAL_ONLY',
                        descriptor=dict(status='ARTIFICIAL', descriptor=d,
                                        source=dict(commit='ARTIFICIAL', path='descriptor.json')))
        def factory(*_):
            return create
        with tempfile.TemporaryDirectory() as tmp:
            req, out = Path(tmp)/'REQ.json', Path(tmp)/'OUT.json'
            req.write_text(json.dumps(request))
            argv = ['--request', str(req), '--output', str(out), '--cache', str(Path(tmp)/'cache.json'),
                    '--prepare-missing', '--source', 'ARTIFICIAL', '--p0-directory', str(Path(tmp)/'p0')]
            with patch('sys.stdout', new=io.StringIO()):
                first = RUN.main(argv, assets_factory=lambda *a, **k: assets, materializer_factory=factory)['items'][0]
                second = RUN.main(argv, assets_factory=lambda *a, **k: assets, materializer_factory=factory)['items'][0]
            self.assertEqual(made, [4])
            self.assertEqual(first['configuration'], second['configuration'])
            self.assertTrue(first['readiness']['configuration_complete'], first.get('error'))
            self.assertNotEqual(first['lambda_P'], first['lambda_DF'])
            self.assertEqual(A.rational(first['configuration']['spec']['split']['lambda_R']),
                             Fraction(first['lambda_P']))
            self.assertFalse(first['readiness']['signal_13round_complete'])
            self.assertEqual(first['signal_plan']['request_rounds'], list(range(13)))
            self.assertIn('--tail-descriptor', first['signal_plan_command'])
            self.assertIsNone(first['price_handoff']['request']['implementation'])
            # Pass the candidate and a toy-specific contract to D2's exact boundary (empty toy registry).
            scope = dict(__name__='artificial_price_boundary')
            exec(compile(E.p0.blob(ROOT, A.D2, 'src/trottertracks/pr_price_rules.py'), 'D2:price_rules', 'exec'), scope)
            rules = scope['PriceRules'](None, dict(entries=[]))
            price_request = deepcopy(first['price_handoff']['request'])
            price_request['implementation'] = dict(lowering_id='ARTIFICIAL', contract_source='ARTIFICIAL')
            quote = rules.quote(price_request)
            self.assertFalse(quote['complete'])
            self.assertEqual(len(quote['rows']), 26)
        # Genuine missing inputs are returned to B2, with no replacement split.
        unavailable = lambda p, d: dict(status='NEEDS_INPUT', missing_inputs=['fixed_snapshot.g_matrices'])
        empty = ToyAssets(self.assets)
        row = A.prepare(request, empty, materialize=unavailable)['items'][0]
        self.assertEqual(row['missing_inputs'], ['fixed_snapshot.g_matrices'])
        self.assertIsNone(row['configuration'])


class ToyAssets(A.Assets):
    """Artificial materializer/storage and reference boundary; real B mapper and E.plan."""
    def __init__(self, saved):
        self.root = ROOT; self.cache = None; self.fragment_count = 6
        self.binding = dict(saved.binding, system='ARTIFICIAL', hamiltonian='TOY_H', snapshot='TOY_SNAPSHOT', state='TOY_STATE')
        self.context = dict(prefixes={}); self.baselines = {}; self.configurations = []; self.local_configurations = []
        self.extra_splits = {}; self.packets = (); self.price_registry = dict(entries=[])
        self.template = deepcopy(saved.baselines[1][0]); self.template['spec']['binding'] = self.binding
        self.ham = SimpleNamespace(n_blocks=6, n_qubits=2, lambdas=np.array([1., 2., -.2, .3, .5, -.125]),
            g_matrices=[np.diag([1., -.5])]*6)

    def descriptor(self, prefix):
        return deepcopy(self.extra_splits.get(prefix, {}).get('descriptor') or
                        dict(status='NEEDS_DERIVATION', descriptor=None, source=D.DERIVATION_SOURCE, missing_inputs=[]))

    def dictionary(self, prefix):
        return self.extra_splits.get(prefix, {}).get('item')

    def baseline_from_item(self, prefix, item):
        original = deepcopy(self.template)
        original['spec']['L_D'] = prefix; original['spec']['split']['L_D'] = prefix
        return E.L.generate_configuration(ROOT, A.generator(str(ROOT))[0], original, item)

    def signal_plan(self, config, locator, descriptor):
        item = self.dictionary(config['spec']['L_D'])
        with patch.object(E, 'configuration_support', return_value=([], item, {})), \
             patch.object(E, 'round_support', return_value=([], 'ARTIFICIAL_REFERENCE_BOUNDARY')):
            return E.plan(ROOT, config, packets=(), index={}, configuration_source=locator)


if __name__ == '__main__':
    unittest.main()
