"""Saved metadata and artificial caller inputs only; no real risk/MC/cost."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_rpe_total_binding import T, no_science, pauli, MOCK

FULL_H = ('ce6aacfaca8249c510baa2d958849cffc7649cdc',
    'docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/')
P7 = ('5fb2c32e39e63267ea1748861269c362a93a1b86',
    'docs/research/truncated_fullh_signal_results_20261114/run01/B0I_P7_S2_D01/')


def saved(locator):
    commit, directory = locator
    cs = dict(commit=commit, path=directory+'PLAN.json', selector=['configuration'])
    ps = dict(commit=commit, path=directory+'B_SIGNAL_INPUT.json')
    config, packet = T._selected(ROOT, cs), T._selected(ROOT, ps)
    return T.create(ROOT, config, original_source=cs), packet, ps


class BaselineBindingTests(unittest.TestCase):
    def test_Full_PF_artificial_signal_uses_same_passive_adapter(self):
        with no_science():
            bridge, saved_packet, _ = saved(P7)
            api = T.C.interfaces(str(ROOT)).pipeline
            deps = api.providers(str(ROOT))
            spec = deps.D.full_pf_spec(bridge['original_configuration']['spec'],
                formula='2nd', delta_pf=.1, M=12, rank=10)
            config = T.B.generate(spec)
            target = deepcopy(saved_packet['target'])
            request = api.plan(ROOT, config, target=target, signal_source=MOCK)
            means = []
            for row in config['rounds']:
                context = api.price_context(request, row['m'])
                means.append(dict(m=row['m'], identity=deepcopy(row['identity']),
                    law_binding=deepcopy(row['law_binding']), physical_candidate=deepcopy(context['physical_candidate']),
                    target=deepcopy(target), source=MOCK, raw=dict(real=.25, imag=-.5),
                    corrected=None, normalization=dict(log_B=0.), uncertainty=None,
                    qualification='UNQUALIFIED', provider_binding=dict(physical_context=context, numerical_source=MOCK)))
            packet = dict(configuration_id=config['id'], target=target, source=MOCK, means=means)
            bridge = T.create(ROOT, config, original_source=MOCK)
            with patch.object(T, '_selected', return_value=packet):
                view = T.bind_signal(ROOT, bridge, packet, original_packet_source=MOCK)
            self.assertEqual(T.B.comparison_method(view['configuration']['spec']), 'FULL_PF')
            self.assertTrue(all(r['R_outer'] is None for r in view['configuration']['rounds']))
            self.assertEqual(view['configuration']['spec']['L_D'], 10)

    def test_saved_baselines_keep_all_rows_and_qualifications_without_science(self):
        with no_science():
            for locator in (FULL_H, P7):
                bridge, packet, source = saved(locator)
                original = deepcopy(packet)
                view = T.bind_signal(ROOT, bridge, packet, original_packet_source=source)
                out = view['signal_packet']
                self.assertEqual(packet, original)
                self.assertEqual(out['origin_signal_packet'], packet)
                self.assertEqual(out['origin_packet_source'], source)
                self.assertEqual(len(out['means']), 13)
                self.assertEqual(out['target'], packet['target'])
                self.assertFalse(out['target']['ground_state_certified'])
                for old, new in zip(packet['means'], out['means']):
                    for key in ('raw', 'corrected', 'normalization', 'uncertainty', 'qualification', 'source'):
                        self.assertEqual(old[key], new[key])
                    self.assertEqual(new['implementation_source'], packet['source'])
                    self.assertNotEqual(old['identity'], new['identity'])
                for key in ('sufficient_N', 'global_RMSE_bound', 'G', 'winner'):
                    self.assertIsNone(out[key])
                self.assertEqual(view['science_calls'], 0)

    def test_mutated_raw_source_normalizer_and_law_rejected(self):
        with no_science():
            bridge, packet, source = saved(FULL_H)
            for field in ('raw', 'source', 'normalization', 'law_binding'):
                bad = deepcopy(packet); bad['means'][0][field] = {}
                with self.assertRaisesRegex(ValueError, 'saved SOURCE'):
                    T.bind_signal(ROOT, bridge, bad, original_packet_source=source)

    def test_full_h_R448_cannot_be_relabelled_as_kappa2(self):
        with no_science():
            bridge, _, _ = saved(FULL_H)
            other = dict(commit=P7[0], path='docs/research/truncated_fullh_signal_results_20261114/run01/'
                'FULL_H_K2_QDYADIC_R448/B_SIGNAL_INPUT.json')
            with self.assertRaisesRegex(ValueError, 'configuration'):
                T.bind_signal(ROOT, bridge, T._selected(ROOT, other), original_packet_source=other)

    def test_PR_fixed_grid_uses_versioned_view_and_Pauli_shortcut_is_rejected(self):
        with no_science():
            source = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
                path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json')
            original = T._selected(ROOT, source)
            bridge = T.create(ROOT, original, original_source=source)
            self.assertEqual(bridge['total_configuration']['spec']['schedule_adapter'], T.FIXED_GRID_TOTAL)
            self.assertEqual(bridge['total_configuration']['spec']['decoder_origin_spec'], original['spec'])
            args, _ = pauli(); bridge = T.create(ROOT, args['config'], original_source=MOCK)
            with self.assertRaisesRegex(ValueError, 'baseline-only'):
                T.bind_signal(ROOT, bridge, args['signal_packet'], original_packet_source=MOCK)

    def test_artificial_caller_uses_explicit_kernel_view_and_no_N(self):
        # Real configuration metadata only; every signal coordinate and the
        # evaluation target/shot values below are artificial. Kernel is mocked.
        bridge, original, _ = saved(P7)
        config = bridge['original_configuration']
        bridge = T.create(ROOT, config, original_source=MOCK)
        packet = deepcopy(original)
        packet['source'] = deepcopy(MOCK)
        for mean in packet['means']:
            mean.update(raw=dict(real=.25, imag=-.5), source=deepcopy(MOCK))
            mean['provider_binding']['numerical_source'] = deepcopy(MOCK)
        source = dict(fixture='ARTIFICIAL_SAVED_PACKET')
        with patch.object(T, '_selected', return_value=packet):
            view = T.bind_signal(ROOT, bridge, packet, original_packet_source=source)
            with patch.object(T.Q, 'evaluate', return_value=[dict(deterministic_input=dict(source=MOCK))]) as numerical:
                out = T.evaluate(ROOT, bridge, view['signal_packet'], [], evaluation_target=dict(energy_ha=0.,source=MOCK),
                    statistical_contract=MOCK, source=MOCK, options=dict(estimator=T.T.ESTIMATOR))
                self.assertEqual(out[0]['deterministic_input']['origin_packet_source'], source)
                for key in ('sufficient_N', 'n_packet', 'G', 'G_qualified'):
                    self.assertIsNone(out[0][key])
                self.assertEqual(numerical.call_count, 1)
                consumed = numerical.call_args.args[1]
                self.assertEqual(consumed, view['signal_packet']['kernel_signal_packet'])
                self.assertEqual(consumed['means'][0]['raw'], dict(real=.25, imag=-.5))
                self.assertEqual(consumed['means'][0]['implementation_source'], MOCK)
        self.assertNotIn('implementation_source', original['means'][0])

    def test_existing_C_prepare_body_is_reused(self):
        api = T.C.interfaces(str(ROOT)).pipeline
        fn = api.C.prepare
        self.assertEqual(fn.__code__.co_filename,
            T.C.SOURCE+':src/trottertracks/baseline_signal_pipeline.py')
        with no_science():
            bridge, packet, source = saved(P7)
            bridge = T.create(ROOT, bridge['original_configuration'], original_source=MOCK)
            # The physical context check still rejects a mismatched candidate,
            # even when the supplied fixture is its own source document.
            packet['means'][0]['physical_candidate']['prefix'] = 8
            with patch.object(T, '_selected', return_value=packet):
                with self.assertRaisesRegex(ValueError, 'physical candidate'):
                    T.bind_signal(ROOT, bridge, packet, original_packet_source=source)


if __name__ == '__main__': unittest.main(verbosity=2)
