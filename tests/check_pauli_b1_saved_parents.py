"""Saved toy parent events -> direct q2 composition. No H4 inputs or transpile."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import check_pauli_ld1_m0_pilot_circuits as T
import run_pauli_b1_prices as P


class Tests(unittest.TestCase):
    def setUp(self):
        T.Tests.setUp(self)
        from trotterlib import rte
        from trottertracks.resource_applicability import ax2b_native_df_v5
        self.rte, self.lowerer = rte, ax2b_native_df_v5
        self.req = dict(self.req, T=float(self.req['T']))
        self.identity = dict(dictionary_sha256=P.B0.connection().canonical_sha256(self.d), prefix=1, fixture='TOY_ONLY')
        self.calls = []
        def fake(wrapper, fingerprint, representative):
            self.calls.append(fingerprint)
            return dict(metrics={k: 1 for k in self.native.METRICS}, fingerprint=fingerprint,
                        compiler=self.native.COMPILER_HASH), wrapper.copy(), False
        self.fake = fake
        comps, _, _ = P.B0.connection().components(self.d)
        dist = rte.finite_rte_distribution(self.req['lambda_P'] * (self.req['step_time'] / self.req['R']), 2)
        # Existing deterministic event constructor, never a sampler. Positive
        # product signs plus order2 phase=-1 exercise a nonzero D2 step phase.
        rare = rte._make_event([0, 1, 2], comps, dist, 1)
        ordinary = rte._make_event([1], comps, dist, 0)
        self.parents = []
        for i, seed in enumerate((11, 12)):
            batch = (rare, ordinary, ordinary) if i == 0 else (ordinary, rare, ordinary)
            steps = [dict(outer=0, signed_time=self.req['step_time'], seed=seed, events=batch)]
            result = P.B0.pilot_core(self.native, self.ctx, self.d, self.req, self.ops, self.blocks,
                cohort='TOY', compile_cached=self.fake, index=i, step_seeds=[seed], event_metadata=True,
                saved_step_data=(steps, dist))
            self.parents.append(dict(index=i, cohort='TOY', request=self.req, dictionary_identity=self.identity,
                result=result, receipt_source=dict(commit='a'*40, execution_commit='b'*40,
                path='ARTIFICIAL_PARENTS.json', selector=['paths', i])))
        file = self.tmp / 'saved_parents.json'; file.write_text(json.dumps(self.parents))
        self.parents = json.loads(file.read_text())
        self.pair_request = dict(self.req, q=2, T=.2)
        self.calls.clear()

    def acquire(self, parents):
        return P.acquire_path(self.native, self.ctx, self.d, self.pair_request, cohort='TOY_PAIR', index=0,
            seeds=[11, 12], operators=self.ops, blocks=self.blocks, compile_cached=self.fake,
            parents=parents, dictionary_identity=self.identity)

    def test_saved_order2_nonzero_phase_concatenation_sampler_zero_xy_and_scalar_once(self):
        with (patch.object(P.B0.connection(), 'sample_steps', side_effect=AssertionError('sampler forbidden')) as step,
              patch.object(self.rte, 'sample_rte_events', side_effect=AssertionError('sampler forbidden')) as sample):
            result = self.acquire(self.parents)
        step.assert_not_called(); sample.assert_not_called()
        self.assertEqual(result['sampling'], 'SAVED_PARENT_EVENTS_NO_DRAW')
        self.assertEqual(result['paired_check']['status'], 'PAIRED_STEP_METADATA_IDENTICAL')
        self.assertEqual(len(self.calls), 2)
        expected = [s for p in self.parents for s in p['result']['event_metadata']]
        for actual, saved in zip(result['event_metadata'], expected):
            self.assertEqual(json.loads(json.dumps(actual['events'])), saved['events'])
            self.assertEqual(json.loads(json.dumps(actual['D2'])), saved['D2'])
            self.assertNotEqual(saved['D2']['aggregate_source_global_phase'], 0.)
            self.assertTrue(any(e['event']['taylor_order'] == 2 for e in actual['events']))
        self.assertEqual(set(result['fingerprints']), {'cosine', 'sine'})
        self.assertTrue(all(v['measurement'] for v in result['fingerprints'].values()))
        self.assertTrue({'compiled_vector_cosine','compiled_vector_sine'} <= {g['check'] for g in result['checks']})

    def test_parent_order_dictionary_and_phase_fail_before_build_compile(self):
        variants = []
        bad = deepcopy(self.parents); bad.reverse(); variants.append(bad)
        bad = deepcopy(self.parents); bad[0]['dictionary_identity']['dictionary_sha256'] = 'f'*64; variants.append(bad)
        bad = deepcopy(self.parents); bad[0]['result']['event_metadata'][0]['events'][0]['event']['phase']['real'] = 1.; variants.append(bad)
        bad = deepcopy(self.parents); bad[0]['result']['event_metadata'][0]['D2']['aggregate_source_global_phase'] += .2; variants.append(bad)
        bad = deepcopy(self.parents); bad[0]['result']['event_metadata'][0]['D2']['phase_instruction'] += 1; variants.append(bad)
        bad = deepcopy(self.parents); bad[0]['result']['event_metadata'][0]['D2']['aggregate_source_global_phase'] = float('nan'); variants.append(bad)
        bad = deepcopy(self.parents); bad[0]['result']['scalar'] += .1; variants.append(bad)
        for bad in variants:
            with self.subTest(parent=bad[0]['index']), patch.object(self.lowerer, 'build_partial_native') as build:
                with self.assertRaises(ValueError): self.acquire(bad)
                build.assert_not_called(); self.assertEqual(self.calls, [])

    def test_normal_path_still_uses_original_sampler(self):
        connection = P.B0.connection()
        with patch.object(connection, 'sample_steps', wraps=connection.sample_steps) as sampler:
            result = P.acquire_path(self.native, self.ctx, self.d, self.req, cohort='TOY_NORMAL', index=0,
                seeds=[11], operators=self.ops, blocks=self.blocks, compile_cached=self.fake)
        sampler.assert_called_once()
        self.assertEqual(sampler.call_args.kwargs['step_seeds'], [11])
        self.assertEqual(result['sampling'], 'ORIGINAL_NORMAL_SAMPLER')
        self.assertEqual(len(self.calls), 2)

    def test_cache_hit_checks_saved_compiled_qpy_without_recompile(self):
        import gzip, io
        from qiskit import qpy
        cache = self.native.PRIVATE / 'compiled_cache'; cache.mkdir(parents=True)
        original = self.fake
        def cache_hit(wrapper, fingerprint, representative):
            record, qc, _ = original(wrapper, fingerprint, representative)
            stream = io.BytesIO(); qpy.dump(qc, stream)
            (cache / (fingerprint + '_compiled.qpy.gz')).write_bytes(gzip.compress(stream.getvalue()))
            return record, None, True
        self.fake = cache_hit
        with patch.object(P.B0.connection(), 'sample_steps', side_effect=AssertionError('sampler forbidden')):
            result = self.acquire(self.parents)
        self.assertTrue(all(f['cache_reused'] for f in result['fingerprints'].values()))
        self.assertTrue({'compiled_vector_cosine','compiled_vector_sine'} <= {g['check'] for g in result['checks']})


if __name__ == '__main__':
    unittest.main(verbosity=2)
