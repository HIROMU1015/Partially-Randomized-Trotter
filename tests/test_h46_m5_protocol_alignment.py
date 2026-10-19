"""Synthetic plan corruption checks; no scientific module imports."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location('alignment_check', ROOT / 'scripts/check_h46_m5_protocol_alignment.py')
k = importlib.util.module_from_spec(s)
s.loader.exec_module(k)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        names = ('aligned_plan_DRAFT.json', 'target_input_bindings_DRAFT.json',
                 'candidate_domain_PROPOSED.json', 'protocol_gap_matrix.csv', 'evidence_catalog.json')
        for name in [k.BUNDLE + '/' + p for p in names] + [k.DOC]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def tearDown(self):
        self.tmp.cleanup()

    def change(self, name, mutation):
        p = self.root / k.BUNDLE / name
        data = json.loads(p.read_text())
        mutation(data)
        p.write_text(json.dumps(data))

    def check(self):
        return k.validate(self.root, verify_git=False)

    def test_structure(self):
        self.assertEqual(self.check()['status'], 'PASS')

    def test_reject_execution(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d.update(new_science_authorized=True))
        with self.assertRaisesRegex(ValueError, 'NO_AUTHORIZATION'):
            self.check()

    def test_reject_m5_main_adoption(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d.update(AD03_main_comparison_status='ADOPTED'))
        with self.assertRaisesRegex(ValueError, 'AD03_STILL_HELD'):
            self.check()

    def test_reject_old_contract_mutation(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d.update(old_four_contracts_modified=True))
        with self.assertRaisesRegex(ValueError, 'NO_AUTHORIZATION'):
            self.check()

    def test_reject_pilot_as_independent_confirmation(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d['technical_pilot'].update(independent_confirmation=True))
        with self.assertRaisesRegex(ValueError, 'PILOT_NOT_CORE'):
            self.check()

    def test_reject_reused_grant(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d['technical_pilot']['lanes'][0].update(grant_id='OLD_GRANT'))
        with self.assertRaisesRegex(ValueError, 'LANE_UNSEALED'):
            self.check()

    def test_reject_legacy_H4(self):
        self.change('target_input_bindings_DRAFT.json', lambda d: d['systems']['H4'].update(series='H4_LEGACY_DF12'))
        with self.assertRaisesRegex(ValueError, 'TOL_LEGACY_SEPARATE'):
            self.check()

    def test_reject_old_H4_prefix_11(self):
        self.change('candidate_domain_PROPOSED.json', lambda d: d['rows'][0].update(prefix=11))
        with self.assertRaisesRegex(ValueError, 'PREFIX_Q_RANGE'):
            self.check()

    def test_reject_duplicate_candidate(self):
        self.change('candidate_domain_PROPOSED.json', lambda d: d['rows'].append(d['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'DOMAIN_UNIQUE'):
            self.check()

    def test_reject_automatic_phase_transition(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d['stages'][0].update(auto_transition=True))
        with self.assertRaisesRegex(ValueError, 'SEPARATE_APPROVAL_STOP'):
            self.check()

    def test_reject_dag_cycle(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d['stages'][0].update(depends_on=['H4_SIGNAL']))
        with self.assertRaisesRegex(ValueError, 'CLOSED_ACYCLIC'):
            self.check()

    def test_reject_coarse_cgroup_stop(self):
        self.change('aligned_plan_DRAFT.json', lambda d: d['parallel'].update(transient_PSI_CPU_only_stop=True))
        with self.assertRaisesRegex(ValueError, 'NONINTERFERENCE'):
            self.check()


if __name__ == '__main__':
    unittest.main()
