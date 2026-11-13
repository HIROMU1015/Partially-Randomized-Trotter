"""Saved JSON identity checks and artificial contracts only; no real numerics."""
import ast
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import rpe_pauli_signal_intake as A
from trottertracks import rpe_evaluation_levels as L
from trottertracks import rpe_deterministic_risk as D
from check_rpe_signal_statistics_caller import I, synthetic, prepare
from check_rpe_price_implementation_guard import fixture, G, MOCK

B, S, K, p = A.B, A.S, A.K, A.p


def forbid_science():
    from contextlib import ExitStack
    stack = ExitStack()
    for owner, name in ((S, 'simulate_prepared'), (S, 'simulate'), (D, 'evaluate'),
                        (D, 'binomial_axis'), (B, 'cost'), (B, 'select'), (p, 'run_missing')):
        stack.enter_context(patch.object(owner, name, side_effect=AssertionError('Science forbidden: '+name)))
    return stack


class SavedIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with forbid_science():
            cls.accepted = A.accept_saved(ROOT, integration=I)

    def test_all26_source_bound_and_scientifically_unqualified(self):
        out = self.accepted
        self.assertEqual(out['science_calls'], 0)
        self.assertEqual([v['configuration']['id'] for v in out['results']], [
            'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27',
            'd7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'])
        counts = {True: 0, False: 0}
        store = p.SavedArtifacts(ROOT)
        for item in out['results']:
            packet = item['signal_packet']
            self.assertEqual([m['m'] for m in packet['means']], list(range(13)))
            self.assertEqual(item['signal_input'], I.signal_input(packet))
            self.assertFalse(item['readiness']['statistics_executed'])
            self.assertFalse(item['readiness']['physical_qualification_complete'])
            self.assertNotIn('energy_ha', packet['target'])
            self.assertFalse(packet['target']['truth_proxy_adopted_for_new_simulation'])
            for mean, row in zip(packet['means'], packet['provenance']):
                counts[row['reused_without_reanalysis']] += 1
                rec = store.read(mean['source'])['record']
                self.assertEqual(mean['raw'], rec['signal']['raw'])
                self.assertEqual(mean['uncertainty']['empirical_allowance'], rec['allowance'])
                self.assertEqual(mean['uncertainty']['numerical_observations'], rec['numerical'])
                self.assertEqual(mean['implementation_source'], rec['source'])
                self.assertEqual(mean['qualification'], 'UNQUALIFIED')
                self.assertEqual(mean['provider_binding']['physical_context'], rec['physical_context'])
                self.assertEqual(row['reused_without_reanalysis'], row['m'] in (0, 6))
                for axis in ('X', 'Y'):
                    self.assertIsNone(item['readiness']['rows'][row['m']]['shots'])
            for k in ('sufficient_N', 'global_RMSE_bound', 'G', 'winner'):
                self.assertIsNone(packet[k])
        self.assertEqual(counts, {True: 4, False: 22})

    def test_mixed_candidate_dictionary_SOURCE_law_and_fixedR_rejected(self):
        store = p.SavedArtifacts(ROOT); original = store.read(A.HANDOFF)['packets'][0]
        def bind(packet):
            config = packet['configuration']
            records = {r['m']: store.read(r['mean_source'])['record'] for r in packet['rounds']}
            return A.bind_packet(config, packet, records, store.read(packet['dictionary_identity']['source']),
                target=self.accepted['results'][0]['signal_packet']['target'], target_source=MOCK,
                assembly_source=A.HANDOFF, helpers=A._metadata_helpers(str(ROOT)))
        before = deepcopy(original)
        with forbid_science():
            bind(original)
            for key, value in (('candidate_id', 'OTHER'), ('law_binding', {}),
                               ('identity', {}), ('nominal_N_X', 64)):
                bad = deepcopy(original); bad['rounds'][0][key] = value
                with self.assertRaises(ValueError): bind(bad)
            bad = deepcopy(original); bad['dictionary_identity']['dictionary_sha256'] = 'wrong'
            with self.assertRaises(ValueError): bind(bad)
            bad = deepcopy(original); bad['rounds'][0]['mean_source']['scientific_source'] = 'wrong'
            with self.assertRaises(ValueError): bind(bad)
            bad = deepcopy(original); bad['configuration']['spec']['event_mapping']['kind'] = 'FIXED_R'
            with self.assertRaises(ValueError): bind(bad)
            bad = deepcopy(original); bad['configuration'] = self.accepted['results'][1]['configuration']
            with self.assertRaises(ValueError): bind(bad)
        self.assertEqual(original, before)

    def test_C_projection_unchanged_at_pinned_guard_SOURCE(self):
        path = 'src/trottertracks/rpe_optimization_pipeline.py'
        def function(commit):
            tree = ast.parse(subprocess.check_output(['git','show',commit+':'+path], cwd=ROOT))
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'signal_input'))
        self.assertEqual(function('5cc7c07e112ce97205948465c94441cba0042416'),
                         function('3a8b37f1a282c0c06ac1a363b070f4376e9ff41a'))
        with forbid_science():
            for item in self.accepted['results']:
                packet = item['signal_packet']; config = item['configuration']
                with self.assertRaisesRegex(ValueError, 'Physical recipe differs from B round'):
                    I.plan(ROOT, config, target=packet['target'], signal_source=packet['source'],
                        pr_physical={m['m']: m['provider_binding']['physical_context']['physical_candidate']
                                     for m in packet['means']})


class ArtificialInterfaceTests(unittest.TestCase):
    def test_authoritative_mixed_round_SOURCE_preserves_existing_caller(self):
        args = synthetic(M=12); packet = args[1]
        packet.update(per_round_source_is_authoritative=True, implementation_sources={})
        for mean in packet['means']:
            source = dict(fixture='P1' if mean['m'] in (0,6) else 'COMPLETION')
            packet['implementation_sources'][str(mean['m'])] = source
            mean['implementation_source'] = deepcopy(source)
            mean['provider_binding']['implementation_source'] = deepcopy(source)
        before = deepcopy(args)
        with forbid_science():
            entry = prepare(args)[0]
            self.assertEqual([r['shots'] for r in entry['prepared']['rows']], args[2][0]['shots'])
            for mutate in ('source', 'authoritative', 'missing'):
                bad = deepcopy(args)
                if mutate == 'source': bad[1]['means'][0]['implementation_source'] = dict(fixture='OTHER_LD')
                if mutate == 'authoritative': bad[1]['per_round_source_is_authoritative'] = False
                if mutate == 'missing': bad[1]['implementation_sources'].pop('12')
                with self.assertRaises(ValueError): prepare(bad)
        self.assertEqual(args, before)

    def test_raw_probability_rule_only_on_artificial_coordinates(self):
        self.assertEqual(D._axis_probability(.25, D.EXACT_AFFINE), Fraction(5,8))
        self.assertEqual(D._axis_probability(-1., D.EXACT_AFFINE), 0)
        self.assertEqual(D._axis_probability(1., D.EXACT_AFFINE), 1)
        with self.assertRaises(ValueError): D._axis_probability(1.01, D.EXACT_AFFINE)

    def test_two_levels_missing_prices_and_no_nominal_promotion(self):
        v = fixture(); c = v['config']; n = v['evaluations'][0]['n_packet']
        paper = dict(configuration_id=c['id'], policy=deepcopy(c['spec']['rmse_policy']), source=MOCK,
            status=L.PAPER, rigorous_sufficient_N=False, sufficient_N=None, global_RMSE_bound=None,
            rounds=[dict(m=r['m'], identity=deepcopy(r['identity']), law_binding=deepcopy(r['law_binding']),
                source=MOCK, nominal_N_X=2**200+1, nominal_N_Y=3) for r in c['rounds']])
        with forbid_science():
            missing = L.prepare(c, paper_packet=paper)
            self.assertEqual(missing['paper']['readiness'], 'UNKNOWN')
            self.assertIsNone(missing['mse']['N_input'])
            o = v['options']
            with patch.object(G, '_document', return_value=v['proofs']):
                out = L.prepare(c, paper_packet=paper, mse_packet=n, prices=v['prices'], cohorts=o['cohorts'],
                    root=ROOT, price_bindings=o['price_bindings'], comparison_policy=o['comparison_policy'])
            self.assertEqual([out[k]['readiness'] for k in ('paper','mse')], ['INPUTS_READY']*2)
            self.assertEqual(out['paper']['N_input']['rounds'][0]['nominal_N_X'], 2**200+1)
            for k in ('paper','mse'):
                self.assertIsNone(out[k]['G']); self.assertIsNone(out[k]['sufficient_N'])
                self.assertFalse(out[k]['scientific_qualification_issued'])
            with self.assertRaises(ValueError): L.prepare(c, mse_packet=paper)
            bad = deepcopy(paper); bad['rounds'][0]['nominal_N_X'] = True
            with self.assertRaises(ValueError): L.prepare(c, paper_packet=bad)
            bad = deepcopy(paper); bad['rounds'][0]['law_binding'] = {}
            with self.assertRaises(ValueError): L.prepare(c, paper_packet=bad)
            with patch.object(L.R, 'check', side_effect=ValueError('foreign lowering')), \
                 patch.object(B, 'assess', side_effect=AssertionError('guard must run first')):
                with self.assertRaisesRegex(ValueError, 'foreign lowering'):
                    L.prepare(c, root=ROOT, comparison_policy=MOCK)


if __name__ == '__main__': unittest.main()
