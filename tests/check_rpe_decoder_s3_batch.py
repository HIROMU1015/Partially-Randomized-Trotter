"""Actual B integration, artificial counts only; saved S3 metadata stays passive."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import numpy as np
from trottertracks import rpe_decoder_statistics as S
from trottertracks import rpe_decoder_fast as F
from rpe_decoder_r3_fixture import b_statistics, strip_acceleration
from check_rpe_decoder_statistics import toy, POLICY, MOCK


def artificial_rows(M=12):
    return [dict(m=m, raw=dict(real=.43*math.cos(.237*(1 << m)),
        imag=.43*math.sin(.237*(1 << m))), N_X=31+m, N_Y=37+m) for m in range(M+1)]


class S3BatchTests(unittest.TestCase):
    def test_01_actual_default_and_batch_all_trials_match_pinned_B(self):
        old = b_statistics()
        for M in (0, 3, 12):
            kw = dict(M=M,h=.1,prior=[-4.,4.],repetitions=33,seed=441,policy=POLICY)
            expected = old.simulate(artificial_rows(M), **kw)
            self.assertEqual(expected, S.simulate(artificial_rows(M), **kw))
            actual = S.simulate(artificial_rows(M), **kw,batch_decoder=F.decode_sampled_counts)
            self.assertEqual(expected, strip_acceleration(actual))
            self.assertTrue(all(t['decoder_acceleration_source']==F.SOURCE for t in actual['trials']))
            for loss in (dict(kind='UNDEFINED',source=MOCK),
                    dict(kind='BOUNDED_INTERVAL',lower=0.,upper=10000.,source=MOCK)):
                summary = dict(target_energy_ha=0.,unresolved_loss=loss,iid_repetitions=True,
                    confidence=.95,loss_support=dict(upper=10000.,source=MOCK))
                self.assertEqual(old.summarize(expected['trials'], **summary), S.summarize(actual['trials'], **summary))

    def test_02_PCG64_counts_consumption_order_and_exact_converter(self):
        rows = artificial_rows(); repetitions = 19; seed = 919
        bitgen = np.random.PCG64(seed); reference = np.random.Generator(bitgen)
        expected = []
        for r in rows:
            expected.append({a:reference.binomial(r['N_'+a],(1.+r['raw'][c])/2.,size=repetitions)
                for a,c in (('X','real'),('Y','imag'))})
        state = deepcopy(bitgen.state)
        scalar_counts = []; original_decode = S.decode_counts
        def scalar_hook(counts, **kw):
            scalar_counts.append(deepcopy(counts)); return original_decode(counts, **kw)
        kwargs = dict(M=12,h=.1,prior=[-4,4],repetitions=repetitions,seed=seed,policy=POLICY)
        with patch.object(S,'decode_counts',side_effect=scalar_hook):
            S.simulate(rows, **kwargs)
        for i, counts in enumerate(scalar_counts):
            for m,r in enumerate(rows):
                for a in ('X','Y'):
                    self.assertEqual(counts[m][a],dict(plus=int(expected[m][a][i]),shots=r['N_'+a]))
        def batch_hook(samples, rounds, **kw):
            for m in range(13):
                for a in ('X','Y'): np.testing.assert_array_equal(samples[m][a],expected[m][a])
            self.assertIs(kw['count_mean'], S.count_mean)
            return F.decode_sampled_counts(samples, rounds, **kw)
        # Capture the actual generator to check its final state, not only outputs.
        original_PCG64 = np.random.PCG64; used = []
        def factory(value):
            used.append(original_PCG64(value)); return used[-1]
        with patch.object(np.random,'PCG64',side_effect=factory):
            S.simulate(rows, **kwargs,batch_decoder=batch_hook)
        self.assertEqual(len(used),1); self.assertEqual(used[0].state,state)

    def test_03_prepared_real_adapter_and_gates(self):
        config,means,shots,target = toy(M=12)
        prepared = S.prepare(config,means,shots,target=target,shot_policy=dict(source=MOCK,meaning='NOMINAL'))
        kwargs = dict(repetitions=23,seed=61,policy=POLICY,allow_conditional_unqualified=True)
        old = S.simulate_prepared(prepared, **kwargs)
        actual = S.simulate_prepared(prepared, **kwargs,batch_decoder=F.decode_sampled_counts)
        self.assertEqual(old,strip_acceleration(actual))
        for k in ('sufficient_N','global_RMSE_bound','G'):self.assertIsNone(actual[k])
        self.assertFalse(actual['physical_qualification_complete'])
        for incomplete in (dict(prepared,input_complete=False),prepared):
            with patch.object(S,'simulate',side_effect=AssertionError('simulation on blocked input')):
                result=S.simulate_prepared(incomplete,repetitions=1,seed=1,policy=POLICY,
                    batch_decoder=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('hook on blocked input')))
            self.assertTrue(result['status'].startswith('NOT_EVALUATED'))

    def test_04_saved_S3_input_nominals_and_scientific_nulls_remain_passive(self):
        path=ROOT/'docs/research/rpe_s3_readiness_20261113/metadata01/STATISTICS_INPUTS.json'
        prepared=json.loads(path.read_text())
        self.assertEqual([r['shots']['N_X'] for r in prepared['rows']],[64,60,56,52,47,43,38,34,30,25,21,17,12])
        self.assertEqual([r['shots']['N_Y'] for r in prepared['rows']],[r['shots']['N_X'] for r in prepared['rows']])
        for k in ('sufficient_N','global_RMSE_bound','G'):self.assertIsNone(prepared[k])
        self.assertFalse(prepared['scientific_execution_ready']);self.assertFalse(prepared['S3_executed'])
        with patch.object(S,'simulate',side_effect=AssertionError('real S3 started')):
            result=S.simulate_prepared(prepared,repetitions=1,seed=1,policy={},batch_decoder=F.decode_sampled_counts)
        self.assertEqual(result['status'],'NOT_EVALUATED_UNQUALIFIED_PHYSICAL_MEANS')

    def test_05_actual_simulation_zero_tie_and_prior_boundary(self):
        rows=[dict(m=m,raw=dict(real=0.,imag=0.),N_X=2,N_Y=2) for m in range(4)]
        kwargs=dict(M=3,h=1.,prior=[-1.,1.],repetitions=128,seed=31,policy=POLICY)
        old=S.simulate(rows, **kwargs)
        actual=S.simulate(rows, **kwargs,batch_decoder=F.decode_sampled_counts)
        self.assertEqual(old,strip_acceleration(actual))
        statuses={t['status'] for t in actual['trials']}
        self.assertTrue({'UNRESOLVED_ZERO_COORDINATES','UNRESOLVED_BRANCH_TIE',S.DECODED} <= statuses)
        for r in actual['trials']:
            self.assertFalse(r['prior_applied_to_decoder']);self.assertFalse(r['alias_qualified'])
            if r['status']!=S.DECODED:self.assertIsNone(r['energy_estimate_ha'])
        for prior in ([math.pi-1,math.pi],[-math.pi,-math.pi+1]):
            kwargs.update(M=0,prior=prior,repetitions=2)
            boundary=[dict(m=0,raw=dict(real=-1.,imag=0.),N_X=2,N_Y=2)]
            self.assertEqual(S.simulate(boundary, **kwargs),strip_acceleration(
                S.simulate(boundary, **kwargs,batch_decoder=F.decode_sampled_counts)))


if __name__ == '__main__': unittest.main()
