"""Artificial decoder equivalence, boundaries and pinned-B S3 hook tests."""
from copy import deepcopy
import itertools
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import numpy as np
from trottertracks import rpe_decoder_fast as F
from trottertracks import pr_rpe_rmse as A
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase as scalar, _vectorized_reconstruction as vector
from rpe_decoder_r3_fixture import b_statistics, strip_acceleration


class FastDecoderTests(unittest.TestCase):
    def compare(self, phases, scalar_rows=None):
        phases = np.asarray(phases, dtype=float)
        q = [1 << m for m in range(len(phases))]
        fast = F.reconstruct_batch(phases, q)
        old = vector(phases, q); new = F.vectorized_reconstruction_fast(phases, q)
        for a, b in zip(old[1], new[1]): np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(old[0], new[0]); np.testing.assert_array_equal(old[2], new[2])
        indices = range(phases.shape[1]) if scalar_rows is None else range(min(scalar_rows, phases.shape[1]))
        for i in indices:
            previous = scalar(phases[:, i], q)
            self.assertEqual(previous, F._decoded_trial(fast, i))
            ambiguous = any(math.isclose(r['distance_from_previous'], r['next_smallest_branch_distance'], rel_tol=0., abs_tol=1e-12)
                            for r in previous['rounds'][1:])
            self.assertEqual(bool(fast['ambiguous'][i]), ambiguous)

    def test_random_M0_through_M12_exact_scalar_and_vector(self):
        rng = np.random.default_rng(20261113)
        for M in range(13):
            with self.subTest(M=M): self.compare(rng.uniform(-math.pi, math.pi, (M+1, 96)), scalar_rows=8)

    def test_small_q_exhaustive_phase_grid(self):
        grid = [-math.pi, -math.pi/2, 0., math.pi/2, math.pi]
        for M in range(4):
            self.compare(np.array(list(itertools.product(grid, repeat=M+1))).T)

    def test_exact_ties_preserve_both_different_reference_rules(self):
        q = 32; previous = np.asarray(2.650718801466388); principal = np.asarray(0.)
        candidates = F._wrap((principal+2*math.pi*np.arange(q))/q)
        distances = np.abs(np.angle(np.exp(1j*(candidates-previous))))
        scalar_choice = int(np.argmin(distances)); vector_choice = int(np.argsort(distances)[0])
        a = F._nearest(principal,previous,q,tie_rule='scalar')
        b = F._nearest(principal,previous,q,tie_rule='legacy_vector')
        self.assertEqual(int(a[1]),scalar_choice); self.assertEqual(int(b[1]),vector_choice)
        self.assertTrue(a[-1]); self.assertEqual(float(a[2]),float(a[3]))
        # Original scalar/vector semantics can differ here: do not silently
        # replace argsort by lower-index argmin in the vector-compatible API.
        self.assertEqual(float(b[0]),float(candidates[vector_choice]))

    def test_latest_published_B_statistics_blob_is_identical(self):
        path = 'src/trottertracks/rpe_decoder_statistics.py'
        refs = ['170047ef6e7a72aef5f4c6ff9c73f942097a694e','bf515a976f1987ad6a19c61d14689b27b0894542']
        blobs = [subprocess.check_output(['git','rev-parse',ref+':'+path],cwd=ROOT) for ref in refs]
        self.assertEqual(blobs[0],blobs[1])

    def test_wrap_midpoint_and_1e12_tie_neighbourhood(self):
        for M in range(1, 13):
            q = 1 << M; cases = []
            for previous in (-math.pi, -1., -0., .7, np.nextafter(math.pi, 0.)):
                for offset in (-2e-12, -5.01e-13, -5e-13, -4.99e-13, 0., 4.99e-13, 5e-13, 5.01e-13, 2e-12):
                    phases = [float(F._wrap((1 << m)*previous)) for m in range(M)]
                    phases.append(float(F._wrap(q*(previous+offset)+math.pi)))
                    cases.append(phases)
            self.compare(np.array(cases).T)

    def test_extreme_phases_signed_zero_and_rounding_boundaries(self):
        cases = [-1e308, -1e20, -math.pi, np.nextafter(-math.pi, -np.inf),
                 np.nextafter(-math.pi, 0.), -0., 0., np.nextafter(0., 1.),
                 np.nextafter(math.pi, 0.), math.pi, np.nextafter(math.pi, np.inf), 1e20, 1e308]
        self.compare(np.array([cases, cases[::-1], cases]))
        # Arbitrary shape, including scalar arrays, retains the tuple API.
        for values in ([np.array(.2), np.array(.4)], [np.zeros((2,3)), np.full((2,3), -math.pi)]):
            a = vector(values, [1,2]); b = F.vectorized_reconstruction_fast(values, [1,2])
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(type(a[0]),type(b[0])); self.assertEqual(a[0].shape,b[0].shape)

    def test_coordinate_wrapper_zero_tie_alias_prior_and_energy(self):
        rng = np.random.default_rng(19)
        for M in range(13):
            xy = rng.uniform(-1,1,(9,M+1,2))
            xy[0,M] = 0; xy[1,:,0] = 1; xy[1,:,1] = 0
            if M: xy[1,M] = [-1,0]
            out = F.decode_coordinates_batch(xy,h=.1)
            for expected, actual in zip((A.decode_supplied_coordinates([tuple(map(float,v)) for v in row],h=.1) for row in xy), out):
                self.assertEqual(expected,actual)
        for energy in (-4., 4., math.pi, -math.pi):
            xy = [(math.cos(-energy*(1 << m)), math.sin(-energy*(1 << m))) for m in range(5)]
            self.assertEqual(A.decode_supplied_coordinates(xy,h=1.),F.decode_supplied_coordinates_fast(xy,h=1.))

    def test_scalar_rule_does_not_enumerate_q_branches(self):
        phases = np.zeros((25,3)); phases[-1] = -math.pi
        # Even exact ties use only constant candidates in the canonical mode.
        with patch.object(np, 'arange', side_effect=AssertionError('all-branch enumeration')):
            result = F.reconstruct_batch(phases, [1 << m for m in range(25)])
        self.assertEqual(result['final_phase_estimate'].shape,(3,))

    def test_invalid_inputs_and_missing_measurement_remain_rejected(self):
        for phases, q in (([],[]),([np.zeros(2)],[2]),([np.zeros(2),np.zeros(3)],[1,2]),([np.array([np.nan])],[1])):
            with self.assertRaises(ValueError): F.reconstruct_batch(phases,q)
        for xy in ([(True,0.)],[(float('nan'),0.)],[(1.01,0.)]):
            with self.assertRaises(ValueError): F.decode_supplied_coordinates_fast(xy,h=1)
        with self.assertRaises(ValueError): F.decode_coordinates_batch(np.zeros((1,0,2)),h=1)

    def test_B_batch_counts_exact_rounding_large_N_and_complete_trial_dicts(self):
        S = b_statistics(); rng = np.random.default_rng(122)
        for n in (3, 17, 2**53+1, 2**62+1):
            rounds = [dict(m=m,N_X=n,N_Y=n) for m in range(3)]
            values = np.array([0,1,n//2,(n+1)//2,n-1,n],dtype=np.int64)
            samples = [dict(X=values,Y=values[::-1]) for m in range(3)]
            actual = F.decode_sampled_counts(samples,rounds,M=2,h=.1,prior=[-4,4],count_mean=S.count_mean,decoder_source=S.SOURCE)
            for i,row in enumerate(actual):
                counts = [dict(m=m,**{a:dict(plus=int(samples[m][a][i]),shots=n) for a in ('X','Y')}) for m in range(3)]
                expected = S.decode_counts(counts,M=2,h=.1,prior=[-4,4]); expected.pop('counts')
                row = deepcopy(row); row.pop('decoder_acceleration_source')
                self.assertEqual(expected,row)
        with self.assertRaises(ValueError):
            F.decode_sampled_counts([dict(X=np.array([True]),Y=np.array([1]))],[dict(m=0,N_X=3,N_Y=3)],M=0,h=1,prior=[-4,4],count_mean=S.count_mean,decoder_source=S.SOURCE)

    def test_pinned_B_patch_same_rng_trials_loss_CI_and_original_fallback(self):
        original = b_statistics(); accelerated = b_statistics(patched=True)
        policy = dict(kind=original.IID,source={'fixture':'ARTIFICIAL'},target_law='ideal_finite_RTE_law')
        for M in (0,3,12):
            rows = [dict(m=m,raw=dict(real=.4*math.cos(.23*(1<<m)),imag=.4*math.sin(.23*(1<<m))),N_X=3,N_Y=5) for m in range(M+1)]
            kw = dict(M=M,h=.1,prior=[-4,4],repetitions=17,seed=123,policy=policy)
            a = original.simulate(rows,**kw); b = accelerated.simulate(rows,**kw)
            self.assertEqual(a,b)  # No hook -> original path.
            b = accelerated.simulate(rows,**kw,batch_decoder=F.decode_sampled_counts)
            self.assertEqual(a,strip_acceleration(b))
            summary = dict(target_energy_ha=0.,unresolved_loss=dict(kind='BOUNDED_INTERVAL',lower=0.,upper=10000.,source={'fixture':'ARTIFICIAL'}),
                iid_repetitions=True,confidence=.95,loss_support=dict(upper=10000.,source={'fixture':'ARTIFICIAL'}))
            self.assertEqual(original.summarize(a['trials'],**summary),accelerated.summarize(b['trials'],**summary))

    def test_prepared_B_missing_and_unqualified_gates_never_call_hook(self):
        S = b_statistics(patched=True)
        kwargs = dict(repetitions=1,seed=1,policy={},batch_decoder=lambda *a,**k:(_ for _ in ()).throw(AssertionError('hook executed')))
        for prepared in (dict(input_complete=False,blockers=['MISSING_RAW_MEAN']),
                         dict(input_complete=True,physical_qualification_complete=False,blockers=['PHYSICAL_MEAN_UNQUALIFIED'])):
            with patch.dict(S.simulate_prepared.__globals__,simulate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('simulation executed'))):
                self.assertTrue(S.simulate_prepared(prepared,**kwargs)['status'].startswith('NOT_EVALUATED'))
        rows = [dict(m=0,mean=dict(raw=dict(real=.4,imag=-.3),source={'fixture':'ARTIFICIAL'},bias={'finite_K':None},uncertainty={'law':None}),
                     shots=dict(N_X=3,N_Y=5),law_binding=dict(definition=dict(model='ideal_finite_RTE_law')))]
        prepared = dict(input_complete=True,physical_qualification_complete=False,rows=rows,
            decoder=S.DECODER,prior=[-4,4],M=0,h=dict(numerator=1,denominator=10),configuration_id='ARTIFICIAL',target={'fixture':'ARTIFICIAL'})
        args = dict(repetitions=19,seed=31,policy=dict(kind=S.IID,source={'fixture':'ARTIFICIAL'},target_law='ideal_finite_RTE_law'),allow_conditional_unqualified=True)
        a = S.simulate_prepared(prepared,**args)
        b = S.simulate_prepared(prepared,**args,batch_decoder=F.decode_sampled_counts)
        self.assertEqual(a,strip_acceleration(b))
        self.assertIsNone(b['global_RMSE_bound']); self.assertIsNone(b['sufficient_N']); self.assertIsNone(b['G'])


if __name__ == '__main__':
    unittest.main()
