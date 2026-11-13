"""Synthetic MC plus immutable A/C metadata only; no H4 statistical analysis."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_rpe_decoder_statistics import toy, MOCK, POLICY, UNDEFINED, count
from trottertracks import rpe_signal_statistics_caller as K
from trottertracks import rpe_decoder_fast as F

B, S, W, p = K.B, K.S, K.W, K.p
A_SOURCE = 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
C_SOURCE = '5cc7c07e112ce97205948465c94441cba0042416'
BUNDLE = '91349e703e0d09ef702ccd108745dfaf1acc29f7'


def fixed_functions(commit, path, names, scope):
    # Test-only selection of unmodified pure functions; excludes scientific
    # imports/providers. Production accepts C's existing module directly.
    tree = ast.parse(subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT))
    nodes = [v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name in names]
    assert {v.name for v in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), commit+':'+path, 'exec'), scope)
    return NS(**{name: scope[name] for name in names})


def c_interface():
    metadata = fixed_functions(A_SOURCE, 'src/trottertracks/pr_rpe_signal_plan.py',
        ('plan',), dict(deepcopy=deepcopy, common=p))
    recipe = fixed_functions(A_SOURCE, 'src/trottertracks/pr_rpe_signal_provider.py',
        ('numerical_recipe',), dict(deepcopy=deepcopy, RECIPE='DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1'))
    api = NS(B=B, W=W, A=NS(metadata=metadata, numerical_recipe=recipe.numerical_recipe))
    scope = dict(deepcopy=deepcopy, C=NS(p=p), providers=lambda root: api,
        MODULE='src/trottertracks/rpe_optimization_pipeline.py', A_SOURCE=A_SOURCE,
        B_SOURCE='b9b9ffa6233db9dd2b413dc07b27e7039324c666', C_SOURCE=C_SOURCE,
        D_SOURCE='24b0a6eb1d806391a3c6ad454fb0c018148d8545')
    return fixed_functions(C_SOURCE, 'src/trottertracks/rpe_optimization_pipeline.py',
        ('plan', '_validate', 'price_context', 'signal_input', 'compare'), scope)


I = c_interface()


def synthetic(M=2):
    config, means, shots, target = toy(M=M)
    # A-shaped target deliberately lacks energy_ha and does NOT adopt a proxy.
    target.pop('energy_ha')
    target.update(reference_solver_energy_ha=12345., truth_proxy_adopted_for_new_simulation=False,
        qualification='UNQUALIFIED', ground_state_certified=False)
    for mean in means:
        mean['target'] = deepcopy(target)
        mean['implementation_source'] = deepcopy(MOCK)
        mean['provider_binding'] = dict(physical_context=dict(fixture='SYNTHETIC_CONTEXT'),
            numerical_source=deepcopy(mean['source']))
    packet = dict(configuration_id=config['id'], source=MOCK, target=target, means=means)
    evaluation_target = dict(kind='SYNTHETIC_TEST_TARGET', energy_ha=-.2, source=MOCK,
        qualification_scope='SYNTHETIC_ONLY_NO_PHYSICAL_CLAIM',
        original_target_source=deepcopy(target['source']), binding=deepcopy(target['binding']))
    allocations = [dict(allocation_id='synthetic-a', source=MOCK, shot_policy='USER_SUPPLIED', shots=shots)]
    contract = dict(id='SYNTHETIC_CONTRACT', source=MOCK, claim_level='SYNTHETIC_ONLY',
        simulation_policy=POLICY, allow_conditional_unqualified=True,
        summary_options=dict(unresolved_loss=UNDEFINED, iid_repetitions=True, confidence=.9))
    return config, packet, allocations, evaluation_target, contract


def prepare(args):
    c, packet, allocations, target, contract = args
    return K.prepare(c, packet, allocations, evaluation_target=target,
        statistical_contract=contract, integration=I)


class CallerTests(unittest.TestCase):
    def test_01_missing_or_unbound_target_stops_before_MC(self):
        args = synthetic()
        with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
            for target in (None, {}, dict(args[3], energy_ha=None), dict(args[3], source=None),
                           dict(args[3], qualification_scope=None), dict(args[3], binding={}),
                           dict(args[3], original_target_source={'fixture': 'OTHER'})):
                with self.assertRaises(ValueError): prepare((*args[:3], target, args[4]))
            entry = prepare(args)[0]; entry.pop('evaluation_target')
            with self.assertRaises(ValueError): K.diagnose(entry, repetitions=5, seed=1, source=MOCK)

    def test_02_synthetic_explicit_target_scalar_and_R3_match_original(self):
        args = synthetic(); before = deepcopy(args); entry = prepare(args)[0]
        with patch.object(p, 'run_missing', side_effect=AssertionError('no physical provider')):
            scalar = K.diagnose(entry, repetitions=31, seed=37, source=MOCK)
            fast = K.diagnose(entry, repetitions=31, seed=37, source=MOCK,
                batch_decoder=F.decode_sampled_counts)
            original = S.simulate_prepared(entry['prepared'], repetitions=31, seed=37,
                policy=POLICY, allow_conditional_unqualified=True)
            expected = S.summarize(original['trials'], target_energy_ha=args[3]['energy_ha'],
                **args[4]['summary_options'])
        self.assertEqual(scalar['diagnostic'], dict(simulation=original, summary=expected))
        self.assertEqual(fast['diagnostic']['summary'], expected)
        self.assertEqual([{k:v for k,v in t.items() if k != 'decoder_acceleration_source'}
            for t in fast['diagnostic']['simulation']['trials']], original['trials'])
        self.assertEqual(scalar['evaluation_target'], args[3])
        self.assertEqual(scalar['signal_input'], I.signal_input(args[1]))
        self.assertNotIn('energy_ha', scalar['signal_input']['target'])
        self.assertEqual(before, args)
        for result in (scalar, fast):
            for key in ('n_packet', 'sufficient_N', 'global_RMSE_bound', 'G', 'winner'):
                self.assertIsNone(result[key])
            self.assertIsNone(result['diagnostic']['summary']['sufficient_N'])

    def test_03_complete13_rounds_unequal_axes_exact_shots(self):
        args = synthetic(M=12)
        entry = prepare(args)[0]
        self.assertEqual([r['shots'] for r in entry['prepared']['rows']], args[2][0]['shots'])
        result = K.diagnose(entry, repetitions=3, seed=14, source=MOCK, batch_decoder=F.decode_sampled_counts)
        consumed = result['diagnostic']['simulation']['input_raw_means']
        self.assertEqual(len(consumed), 13)
        self.assertEqual([(r['N_X'], r['N_Y']) for r in consumed], [(3, 5)]*13)
        huge = deepcopy(args); huge[2][0]['shots'][0]['N_X'] = 2**200 + 1
        prepared = prepare(huge)[0]
        self.assertEqual(prepared['prepared']['rows'][0]['shots']['N_X'], 2**200 + 1)
        with self.assertRaisesRegex(ValueError, 'int64'): K.diagnose(prepared, repetitions=1, seed=1, source=MOCK)

    def test_04_missing_unqualified_and_bad_integer_shots(self):
        for bad in (True, 2., 0, -1, None):
            args = synthetic(); args[2][0]['shots'][1]['N_X'] = bad
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
                with self.assertRaises(ValueError): prepare(args)
        args = synthetic(); args[2][0]['shots'].pop()
        with self.assertRaises(ValueError): prepare(args)
        args = synthetic(); args[4]['allow_conditional_unqualified'] = False
        with patch.object(S, 'simulate', side_effect=AssertionError('MC forbidden')):
            out = K.diagnose(prepare(args)[0], repetitions=3, seed=1, source=MOCK)
        self.assertEqual(out['diagnostic']['simulation']['status'], 'NOT_EVALUATED_UNQUALIFIED_PHYSICAL_MEANS')
        self.assertIsNone(out['diagnostic']['summary'])
        args = synthetic(); args[1]['means'][1]['raw'] = None
        with patch.object(S, 'simulate', side_effect=AssertionError('MC forbidden')):
            out = K.diagnose(prepare(args)[0], repetitions=3, seed=1, source=MOCK)
        self.assertEqual(out['diagnostic']['simulation']['status'], 'NOT_EVALUATED_MISSING_STATISTICAL_INPUT')

    def test_05_target_loss_and_signal_tampering_rejected_before_MC(self):
        args = synthetic(); args[1]['source'] = dict(fixture='OTHER_IMPLEMENTATION')
        with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
            with self.assertRaises(ValueError): prepare(args)
        for change in ('target', 'signal', 'shots', 'loss', 'confidence'):
            entry = prepare(synthetic())[0]
            if change == 'target': entry['evaluation_target']['energy_ha'] = float('nan')
            if change == 'signal': entry['signal_input']['means'][0]['raw']['real'] = .9
            if change == 'shots': entry['allocation']['shots'][0]['N_X'] += 1
            if change == 'loss': entry['statistical_contract']['summary_options']['unresolved_loss'] = {}
            if change == 'confidence': entry['statistical_contract']['summary_options']['confidence'] = 1
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
                with self.assertRaises(ValueError): K.diagnose(entry, repetitions=4, seed=1, source=MOCK)

    def test_06_zero_tie_and_unresolved_loss_unchanged(self):
        # Exercise the caller's summary on original decoder outputs, including
        # exact zero/tie; sampled ensembles cannot reliably produce a chosen tie.
        zero = S.decode_counts([count((1,2),(1,2))], h=1, M=0, prior=[-2,2])
        tie = S.decode_counts([count((2,2),(1,2)),count((0,2),(1,2),1)], h=1, M=1, prior=[-2,2])
        self.assertEqual([zero['status'], tie['status']], ['UNRESOLVED_ZERO_COORDINATES', 'UNRESOLVED_BRANCH_TIE'])
        entry = prepare(synthetic())[0]
        with patch.object(S, 'simulate_prepared', return_value=dict(trials=[zero, tie])):
            out = K.diagnose(entry, repetitions=2, seed=1, source=MOCK)
        self.assertEqual(out['diagnostic']['simulation']['trials'], [zero, tie])
        summary = out['diagnostic']['summary']
        self.assertEqual(summary['unresolved_count'], 2)
        self.assertIsNone(summary['all_observed_trials_RMSE'])
        self.assertIsNone(summary['population_RMSE_CI'])
        entry['statistical_contract']['summary_options'].update(
            unresolved_loss=dict(kind='BOUNDED_INTERVAL', lower=0, upper=1, source=MOCK),
            loss_support=dict(upper=5, source=MOCK))
        with patch.object(S, 'simulate_prepared', return_value=dict(trials=[zero, tie])):
            bounded = K.diagnose(entry, repetitions=2, seed=1, source=MOCK)
        self.assertEqual(bounded['diagnostic']['summary'], S.summarize([zero, tie],
            target_energy_ha=entry['evaluation_target']['energy_ha'],
            **entry['statistical_contract']['summary_options']))
        self.assertIsNone(bounded['n_packet']); self.assertIsNone(bounded['G'])


class SavedMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accepted = {}
        with patch.object(S, 'simulate_prepared', side_effect=AssertionError('H4 MC forbidden')), \
             patch.object(B, 'cost', side_effect=AssertionError('real G forbidden')), \
             patch.object(p, 'run_missing', side_effect=AssertionError('science provider forbidden')):
            for label in ('LD1', 'LD2'):
                source = dict(commit=BUNDLE,
                    path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_'+label+'.json')
                cls.accepted[label] = K.accept_saved(ROOT, bundle_source=source, integration=I)

    def test_07_both_original_A_bundles_preserved_and_C_provenance(self):
        store = p.SavedArtifacts(ROOT)
        ids = []
        for label, accepted in self.accepted.items():
            packet = accepted['signal_packet']; original = store.read(packet['bundle_source'])
            ids.append(packet['configuration_id'])
            self.assertEqual(len(packet['means']), 13)
            self.assertEqual(packet['target'], original['target'])
            self.assertNotIn('energy_ha', packet['target'])
            self.assertFalse(packet['target']['truth_proxy_adopted_for_new_simulation'])
            self.assertEqual(packet['source'], original['numerical_source'])
            self.assertEqual(accepted['signal_input'], I.signal_input(packet))
            for mean, old in zip(packet['means'], original['means']):
                self.assertEqual({k:v for k,v in mean.items() if k != 'provider_binding'}, old)
                self.assertEqual(mean['qualification'], 'UNQUALIFIED')
                bound = mean['provider_binding']
                self.assertEqual(bound['physical_context'], I.price_context(accepted['request'], mean['m']))
                self.assertEqual(bound['numerical_source'], mean['source'])
                self.assertEqual(bound['implementation_source'], packet['numerical_source'])
                self.assertTrue(bound['input_source']); self.assertEqual(bound['reference_source'], old['reference_source'])
            # All old nulls/statuses/STOP remain present and unchanged.
            for key, value in original.items():
                if key != 'means': self.assertEqual(packet[key], value)
            self.assertFalse(accepted['statistics_executed'])
        self.assertNotEqual(*ids)

    def test_08_C_gate_stays_unqualified_without_real_arithmetic(self):
        for label, accepted in self.accepted.items():
            request = accepted['request']; packet = accepted['signal_packet']
            contract = synthetic()[4]
            allocations = [dict(allocation_id='SYNTHETIC_METADATA_ONLY', source=MOCK, shot_policy='USER_SUPPLIED',
                shots=[dict(m=m, N_X=1, N_Y=2) for m in range(13)])]
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('H4 MC forbidden')), \
                 patch.object(B, 'cost', side_effect=AssertionError('real G forbidden')):
                result = I.compare(ROOT, request, packet, allocations, [], [],
                    cohorts={}, statistical_contract=contract)
            self.assertIn('SIGNAL_INPUT_MISSING_OR_PHYSICALLY_UNQUALIFIED', result['blockers'])
            self.assertIsNone(result['comparison']['point_minimum']['allocation_id'])
            self.assertIsNone(result['comparison']['allocations'][0]['G'])

    def test_09_LD_configuration_law_SOURCE_and_carrier_mismatch(self):
        a = self.accepted['LD1']; b = self.accepted['LD2']; store = p.SavedArtifacts(ROOT)
        packet = a['signal_packet']; original = store.read(packet['bundle_source'])
        records = {m['m']:store.read(m['source']) for m in original['means']}
        config = a['request']['configuration']
        for change in ('LD', 'law', 'implementation', 'carrier', 'input', 'signal'):
            bundle = deepcopy(original); recs = deepcopy(records)
            if change == 'LD': bundle['configuration_id'] = b['signal_packet']['configuration_id']
            if change == 'law': bundle['means'][2]['law_binding'] = b['signal_packet']['means'][2]['law_binding']
            if change == 'implementation': bundle['numerical_source']['commit'] = '0'*40
            if change == 'carrier': bundle['means'][0]['source']['commit'] = '0'*40
            if change == 'input': recs[1]['physical_context']['binding'] = {}
            if change == 'signal': recs[1]['signal']['raw'] = None
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('H4 MC forbidden')):
                with self.assertRaises(ValueError):
                    K.bind_bundle(config, bundle, recs, bundle_source=packet['bundle_source'],
                        request=a['request'], integration=I)


if __name__ == '__main__': unittest.main()
