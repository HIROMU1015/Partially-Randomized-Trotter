"""Passive saved PLAN fixtures and synthetic diagnostics; no molecular arithmetic."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_normalization_intake as m


class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intake = m.A0Intake(ROOT)

    def fixture(self):
        records = []
        for expected in self.intake.expected.values():
            row = {k: deepcopy(expected[k]) for k in m.ROW_FIELDS}
            # Deliberately arbitrary diagnostic constants. No exp/log evaluation.
            row.update(normalization=dict(status=m.SCORED, method='EXISTING_K2_FINITE_DISTRIBUTION',
                lambda_source=deepcopy(self.intake.plan['saved_tails'][row['system']]['source']),
                log_B=1.25, optimistic_shot_inflation_log=2.5,
                one_ulp_accumulation_log_sensitivity=1e-12,
                sensitivity_is_rigorous_bound=False, leading_small_tau_term=None),
                attenuation=dict(value=.5, status='FINITE_BINARY64'),
                shot_inflation=dict(value=10., status='FINITE_BINARY64'), N=None, C=None, G=None,
                phase_radius_N_status='NOT_EVALUATED', scientific_R_adopted=False,
                recommended_R=None, scientific_candidate_PASS_FAIL=None)
            records.append(row)
        handoff = dict(status=m.COMPLETE, scientific_source=m.A0, records=records,
            no_threshold_no_candidate_selection=True, normalization_evidence_valid=True,
            N_C_G=None, scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)
        result = {k: deepcopy(handoff[k]) for k in ('status', 'scientific_source', 'records')}
        result.update(source_preserved=True, input_preserved=True,
            source_files=deepcopy(self.intake.plan['sources']), input_hashes=deepcopy(self.intake.plan['input_hashes']),
            real_N_C_G=0, signal_phase_radius_actions=0, H_state_array_loads=0, errors=[])
        return handoff, result

    def sync(self, h, result):
        for key in ('status', 'scientific_source', 'records'):
            result[key] = deepcopy(h[key])

    def test_saved_plan_24_and_unchanged_C1_eight_cells(self):
        ready = self.intake.readiness()
        self.assertEqual(ready['candidate_count'], 24)
        self.assertEqual(len(ready['C1_cells']), 8)
        self.assertEqual(len(ready['C1_overlap']), 2)
        self.assertEqual({(self.intake.expected[c]['system'], self.intake.expected[c]['m'],
                          self.intake.expected[c]['R_outer']) for c in ready['C1_overlap']},
                         {('H6', 4, 16), ('H6', 4, 32)})
        extras = [v for v in ready['C1_cells'] if v['intake_lane'] == 'EXPLICIT_C_CALIBRATION_VALIDATION_ONLY']
        self.assertEqual(len(extras), 2)
        self.assertTrue(all(not v['in_A0'] for v in extras))
        self.assertTrue(all(v['normalization'] is None for v in ready['candidates']))
        for v in ready['candidates']:
            self.assertFalse(v['raw_law_alias_applied'])
            self.assertEqual(v['request']['identity'], self.intake.B_requests[v['candidate_id']]['identity'])
            self.assertNotEqual(v['request']['law_binding'], v['B_provider_law_binding'])

    def test_complete24_retains_raw_diagnostics_without_N_C_G(self):
        h, result = self.fixture(); before = deepcopy((h, result))
        out = self.intake.accept(h, result)
        self.assertEqual((h, result), before)
        self.assertEqual(out['completion'], 'COMPLETE_RECORDED')
        self.assertEqual(out['received_count'], 24)
        self.assertTrue(out['normalization_evidence_usable'])
        self.assertEqual(out['original_handoff'], h)
        self.assertEqual(out['N_records'], []); self.assertEqual(out['C_records'], [])
        self.assertIsNone(out['N_C_G']); self.assertIsNone(out['recommended_R'])
        self.assertFalse(out['full_RPE_qualified'])
        self.assertTrue(all(not v['N_C_G_qualified'] for v in out['rows']))
        href = dict(commit=m.A0, path='synthetic/HANDOFF.json')
        rref = dict(commit=m.A0, path='synthetic/result.json')
        with patch.object(self.intake.store, 'read', side_effect=[deepcopy(h), deepcopy(result)]) as reader:
            published = self.intake.read_published(href, rref)
            self.assertEqual([c.args[0] for c in reader.call_args_list], [href, rref])
        self.assertEqual(published['publication'], dict(handoff=href, result=rref))
        self.assertEqual(published['original_handoff'], h)

    def test_partial_and_STOP_keep_all24_missing_not_zero(self):
        for status in ('PHASE0_RUNNING', 'PHASE0_STOP_TECHNICAL_OR_INPUT'):
            h, result = self.fixture(); h.update(status=status, records=h['records'][:12], normalization_evidence_valid=False)
            self.sync(h, result); result['errors'] = ['synthetic interrupted input']
            out = self.intake.accept(h, result)
            self.assertEqual(out['completion'], 'PARTIAL' if status == 'PHASE0_RUNNING' else 'STOP')
            self.assertEqual(len(out['rows']), 24); self.assertEqual(len(out['missing_ids']), 12)
            self.assertTrue(all(v['record'] is None for v in out['rows'][12:]))
            self.assertFalse(out['normalization_evidence_usable'])
            self.assertEqual(out['original_result']['errors'], result['errors'])
        h, result = self.fixture(); h['records'] = []; h['normalization_evidence_valid'] = False
        self.sync(h, result)
        self.assertEqual(self.intake.accept(h, result)['missing_ids'], list(self.intake.expected))
        h, result = self.fixture(); h['records'][0]['normalization'] = deepcopy(
            next(iter(self.intake.expected.values()))['normalization'])
        self.sync(h, result); out = self.intake.accept(h, result)
        self.assertEqual(out['rows'][0]['normalization_status'], 'NOT_EVALUATED')
        self.assertFalse(out['normalization_evidence_usable'])

    def test_overflow_underflow_null_precision_unresolved_unchanged(self):
        h, result = self.fixture()
        h['records'][0]['attenuation'] = dict(value=0., status='UNDERFLOW_TO_ZERO_KEEP_LOG')
        h['records'][0]['shot_inflation'] = dict(value=None, status='OVERFLOW_BINARY64_KEEP_LOG')
        self.sync(h, result); out = self.intake.accept(h, result)
        self.assertTrue(out['normalization_evidence_usable'])
        self.assertEqual(out['rows'][0]['record'], h['records'][0])
        for normal in (None, dict(status='NUMERICAL_NORMALIZATION_UNRESOLVED',
                method='EXISTING_K2_FINITE_DISTRIBUTION', log_B=None,
                lambda_source=self.intake.plan['saved_tails']['H4']['source'])):
            bad = deepcopy(h); bad['status'] = 'PHASE0_NORMALIZATION_NUMERICAL_UNRESOLVED'
            bad['normalization_evidence_valid'] = False; bad['records'][1]['normalization'] = normal
            bad['records'][1]['attenuation'] = dict(value=None, status='UNAVAILABLE_NORMALIZATION')
            self.sync(bad, result); out = self.intake.accept(bad, result)
            self.assertFalse(out['normalization_evidence_usable'])
            self.assertEqual(out['rows'][1]['record']['normalization'], normal)
        h, result = self.fixture(); h['records'][0]['normalization']['log_B'] = None
        self.sync(h, result)
        self.assertFalse(self.intake.accept(h, result)['normalization_evidence_usable'])

    def test_candidate_H_state_snapshot_source_law_recipe_mismatch(self):
        mutators = [lambda v: v.update(candidate_id='foreign'), lambda v: v.update(T=.2),
            lambda v: v.update(q_PF=2), lambda v: v.update(delta=.2), lambda v: v.update(R_outer=999),
            lambda v: v['structure']['allocation'].__setitem__(0, 999),
            lambda v: v['structure']['signed_weights'].__setitem__(0, -99.),
            lambda v: v['structure'].update(event_count=999),
            lambda v: v['request']['identity']['binding'].update(hamiltonian='foreign'),
            lambda v: v['request']['identity']['binding'].update(state='foreign'),
            lambda v: v['request']['identity']['binding'].update(snapshot='foreign'),
            lambda v: v['request']['identity']['profile'].update(compiler='foreign'),
            lambda v: v['request']['law_binding']['definition'].update(sampling='foreign'),
            lambda v: v['normalization']['lambda_source'].update(context_sha256='foreign')]
        for mutate in mutators:
            h, result = self.fixture(); mutate(h['records'][0]); self.sync(h, result)
            with self.assertRaises(ValueError): self.intake.accept(h, result)
        h, result = self.fixture(); h['records'].append(deepcopy(h['records'][0])); self.sync(h, result)
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.intake.accept(h, result)
        with self.assertRaisesRegex(ValueError, 'Scientific SOURCE'):
            self.intake._source('3b742fff26381514e12c07b51701276b7d1b26ec')

    def test_evidence_companion_and_preservation_required_without_losing_STOP(self):
        h, result = self.fixture()
        self.assertFalse(self.intake.accept(h)['normalization_evidence_usable'])
        for key in ('source_preserved', 'input_preserved'):
            bad = deepcopy(result); bad[key] = False
            h2 = deepcopy(h); h2['status'] = 'PHASE0_STOP_SOURCE_OR_INPUT_CHANGED'; self.sync(h2, bad)
            out = self.intake.accept(h2, bad)
            self.assertFalse(out['normalization_evidence_usable']); self.assertEqual(out['completion'], 'STOP')
        bad = deepcopy(result); bad['input_hashes'] = {}
        with self.assertRaisesRegex(ValueError, 'provenance'): self.intake.accept(h, bad)
        bad = deepcopy(result); bad['status'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'HANDOFF/result'): self.intake.accept(h, bad)

    def test_C1_only_exact_overlap_source_request_and_original_law(self):
        h, result = self.fixture(); accepted = self.intake.accept(h, result)
        for cid in self.intake.overlap:
            request = self.intake.expected[cid]['request']
            link = self.intake.for_c1(accepted, cid, request=request, normalization_source=m.A0)
            self.assertTrue(link['share_eligible']); self.assertIsNone(link['C'])
            with self.assertRaisesRegex(ValueError, 'SOURCE/request/law'):
                self.intake.for_c1(accepted, cid, request=request, normalization_source='f'*40)
            with self.assertRaisesRegex(ValueError, 'SOURCE/request/law'):
                self.intake.for_c1(accepted, cid, request=self.intake.B_requests[cid], normalization_source=m.A0)
        for cid in self.intake.cells.keys() - set(self.intake.overlap):
            with self.assertRaisesRegex(ValueError, 'outside'):
                self.intake.for_c1(accepted, cid, request={}, normalization_source=m.A0)
        cid = self.intake.overlap[0]
        h['status'] = 'PHASE0_STOP_TECHNICAL_OR_INPUT'; h['normalization_evidence_valid'] = False
        self.sync(h, result); stopped = self.intake.accept(h, result)
        self.assertFalse(self.intake.for_c1(stopped, cid, request=self.intake.expected[cid]['request'],
                                          normalization_source=m.A0)['share_eligible'])

    def test_scientific_outputs_nonfinite_and_automatic_calls_rejected(self):
        for field, value in [('N', 2**1200+3), ('C', 1), ('G', 2),
                             ('scientific_candidate_PASS_FAIL', 'PASS'), ('recommended_R', 4)]:
            h, result = self.fixture(); h['records'][0][field] = value; self.sync(h, result)
            with self.assertRaises(ValueError): self.intake.accept(h, result)
        h, result = self.fixture(); h['records'][0]['normalization']['log_B'] = float('inf')
        with self.assertRaises(ValueError): self.intake.accept(h, result)
        h, result = self.fixture()
        with (patch.object(m.r, 'evaluate_menu', side_effect=AssertionError('No G/R')),
              patch.object(m.b.RMenuBatch, '__call__', side_effect=AssertionError('No N')),
              patch.object(m.b.FixedSavedPrices, '__call__', side_effect=AssertionError('No C')),
              patch.object(m.p, 'run_missing', side_effect=AssertionError('No run_missing'))):
            self.intake.accept(h, result); self.intake.readiness()
        env = os.environ.copy(); env['PYTHONPATH'] = 'src'
        code = "import sys; from trottertracks.rpe_normalization_intake import A0Intake; A0Intake('.').readiness(); assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','trottertracks.pr_r_normalization_phase0','trottertracks.pr_r_screening'))"
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
