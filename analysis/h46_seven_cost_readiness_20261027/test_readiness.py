"""Synthetic/metadata checks only. No molecule, sampler or compiler calls."""
from concurrent.futures import Future
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare as p
import execute as x
from h246_saved_cost_evaluation import paired


class Readiness(unittest.TestCase):
    def test_paired_covariance_is_in_the_design(self):
        stats = paired([(2,4),(4,8)])
        self.assertEqual(p.variance(stats, 2, 3), Fraction(128))
        ep = dict(candidate=dict(method='PR'), N_cos=2, N_sin=3)
        result = p.report.cost_for(ep, dict(unit_statistics=stats, cohort='ARTIFICIAL'))
        self.assertEqual((result['G'], result['G_SE']), (24, 8))

    def test_variance_inflation_is_four_not_sixteen(self):
        self.assertEqual(p.sample_size(.01, .01), 400)
        self.assertEqual(p.sample_size(0, .01), 33)

    def test_scalar_header_matches_existing_receipt_convention(self):
        header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (), }\n"
        data = b'\x93NUMPY\x01\x00' + struct.pack('<H',len(header)) + header + struct.pack('<d', 2.)
        self.assertEqual(p.npy_record(data)['shape'], [1])

    def test_frozen_seven_and_saved_N(self):
        plan = x.read_plan()
        self.assertEqual([v['candidate']['id'] for v in plan['candidates']], list(p.IDS))
        self.assertTrue(all(p.study.cid(v['candidate']) == v['candidate']['id'] for v in plan['candidates']))
        self.assertEqual(sum(v['n_exploration'] for v in plan['candidates']), 1856)
        self.assertEqual(sum(v['n_confirmation'] for v in plan['candidates']), 54833)
        self.assertEqual(plan['scientific_calls']['logical_wrapper_axes'], 113378)
        first = plan['candidates'][0]
        ep = next(e for e in first['saved_endpoints'] if e['epsilon'] == .05)
        self.assertEqual((ep['N_cos'],ep['N_sin']), (12306,8259))
        self.assertEqual(plan['seed_audit']['original_long_time_trials_identified'], 4022)

    def test_new_cohorts_are_disjoint_hash_identifiers(self):
        a = {p.native.trial_seed(dict(id='ARTIFICIAL'), p.COHORTS[0], i, o, 0) for i in range(100) for o in range(3)}
        b = {p.native.trial_seed(dict(id='ARTIFICIAL'), p.COHORTS[1], i, o, 0) for i in range(100) for o in range(3)}
        self.assertEqual(len(a),300);self.assertEqual(len(b),300);self.assertFalse(a & b)

    def test_trial_reuse_never_reexecutes_partial_or_STOP(self):
        c = dict(id='ARTIFICIAL', q=1, R_outer=4)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.assertIsNone(x.accepted_trial(c,'TEST',0,root))
            folder=root/'cost/TEST/ARTIFICIAL/0';folder.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,'Partial trial'):
                x.accepted_trial(c,'TEST',0,root)
            (folder/'STOP.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'scientific STOP'):
                x.accepted_trial(c,'TEST',0,root)

    def test_failure_cancels_pending_and_does_not_submit_more(self):
        futures=[Future(),Future()];futures[0].set_exception(ValueError('synthetic correctness failure'))
        pool=MagicMock();pool.submit.side_effect=futures
        context=MagicMock();context.__enter__.return_value=pool
        with patch.object(x,'ProcessPoolExecutor',return_value=context),patch.object(x,'wait',return_value=({futures[0]}, {futures[1]})):
            with self.assertRaises(ValueError):
                x.process_jobs([('a',),('b',),('c',)],2)
        self.assertEqual(pool.submit.call_count,2);self.assertTrue(futures[1].cancelled())

    def test_confirmation_and_exploration_are_never_pooled(self):
        plan = x.read_plan()
        by_cohort = {}
        for i, cohort in enumerate(plan['cohorts']):
            by_cohort[cohort] = {v['candidate']['id']:dict(unit_statistics=paired([(2+100*i,3+100*i),(4+100*i,5+100*i)]),
                cohort=cohort) for v in plan['candidates']}
        saved = {}
        with tempfile.TemporaryDirectory() as root, patch.object(p, 'HERE', Path(root)):
            (Path(root)/'results').mkdir()
            with patch.object(x.study, 'save', side_effect=lambda path,value:saved.update({Path(path).name:value})):
                rows, comparisons = x.account(plan, by_cohort, Path(root)/'results')
        self.assertEqual(len(comparisons),18)
        target = plan['candidates'][0];identity = target['candidate']['id']
        ep = next(e for e in target['saved_endpoints'] if e['epsilon']==target['primary_epsilon'])
        first = next(v for v in comparisons if v['id']==identity and v['cohort']==plan['cohorts'][0])
        second = next(v for v in comparisons if v['id']==identity and v['cohort']==plan['cohorts'][1])
        self.assertEqual(second['G']-first['G'],100*(ep['N_cos']+ep['N_sin']))
        self.assertEqual(Fraction(first['difference_exact']), Fraction(3*ep['N_cos']+4*ep['N_sin'])-Fraction(target['primary_PF']['G_exact']))
        self.assertFalse(saved['extended_point_summary.json']['independent_confirmation_pooled'])

    def test_representatives_precede_remaining_sampling(self):
        plan = x.read_plan();item = dict(plan['candidates'][0], n_exploration=2)
        small = dict(plan,candidates=[item])
        with tempfile.TemporaryDirectory() as root:
            with patch.object(x.native,'native_context',return_value=object()),patch.object(x,'resource_workers',return_value=(1,{})),patch.object(x,'process_jobs',side_effect=ValueError('representative FAIL')) as process:
                with self.assertRaises(ValueError):
                    x.acquire(small,p.COHORTS[0],Path(root),Path(root))
                self.assertEqual(process.call_count,1)
                self.assertEqual([j[2] for j in process.call_args.args[0]],[0])

    def test_metadata_check_does_not_bind_or_launch_science(self):
        with patch.object(x.native,'native_context',side_effect=AssertionError('science')),patch.object(x.native,'cost_trial',side_effect=AssertionError('science')):
            result=x.verify_static(x.read_plan())
        self.assertEqual(result['status'],'STATIC_READY')


if __name__ == '__main__':
    unittest.main()
