"""Synthetic admission checks; no molecular input or real job dispatch."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from launch_h46_m5_cost_admitted import admission_check


class AdmissionTests(unittest.TestCase):
    def fixtures(self):
        configs = [dict(stage=s, resources=dict(admission_cpu_set=cpus, worker_RSS_proxy_bytes=ram),
                        call_scope=dict(compile_axes=axes)) for s, cpus, ram, axes in
                   [('H4_DET', [1, 2], 10, 38), ('H6_DET', [3, 4], 40, 54)]]
        joint = dict(status='JOINT_ADMISSION_ACCEPTED_FOR_APPROVED_DET_SCOPE', allocation={'H4_DET': [1, 2], 'H6_DET': [3, 4]}, reserved_proxy_bytes=100, scratch_free_proxy_bytes=200)
        return configs, joint

    def test_valid_joint_admission_does_not_change_affinity(self):
        import os
        before = os.sched_getaffinity(0)
        configs, joint = self.fixtures()
        result = admission_check(configs, joint, 100, 200, {1, 2, 3, 4})
        self.assertEqual(result['reserved_proxy_bytes'], 100)
        self.assertEqual(before, os.sched_getaffinity(0))

    def test_other_PR_stage_rejected(self):
        configs, joint = self.fixtures();configs[0]['stage'] = 'H4_PR'
        with self.assertRaisesRegex(ValueError, 'TWO_DET'):admission_check(configs, joint, 100, 200, {1, 2, 3, 4})

    def test_overlap_and_foreign_CPU_rejected(self):
        configs, joint = self.fixtures();configs[1]['resources']['admission_cpu_set'] = [2, 3];joint['allocation']['H6_DET'] = [2, 3]
        with self.assertRaisesRegex(ValueError, 'SUBSET_AND_DISJOINT'):admission_check(configs, joint, 100, 200, {1, 2, 3, 4})
        configs, joint = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'SUBSET_AND_DISJOINT'):admission_check(configs, joint, 100, 200, {1, 2, 3})

    def test_memory_and_disk_not_double_counted(self):
        configs, joint = self.fixtures()
        for ram, disk, reason in [(99, 200, 'RAM'), (100, 199, 'SCRATCH')]:
            with self.assertRaisesRegex(ValueError, reason):admission_check(configs, joint, ram, disk, {1, 2, 3, 4})

    def test_scope_and_allocation_changes_rejected(self):
        configs, joint = self.fixtures();configs[0]['call_scope']['compile_axes'] = 40
        with self.assertRaisesRegex(ValueError, '38_54'):admission_check(configs, joint, 100, 200, {1, 2, 3, 4})
        configs, joint = self.fixtures();joint['allocation']['H4_DET'] = [1]
        with self.assertRaisesRegex(ValueError, 'EXACT_JOINT'):admission_check(configs, joint, 100, 200, {1, 2, 3, 4})


if __name__ == '__main__':unittest.main()
