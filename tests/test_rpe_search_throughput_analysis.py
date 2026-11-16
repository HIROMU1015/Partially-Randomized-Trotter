"""Synthetic checks for causal saved-log analysis, never real Q3 inputs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('search_throughput_analysis', ROOT/'scripts/analyze_rpe_search_throughput.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def job(name, start, end, shots, identity='same', donor=True):
    return dict(job=name, run='fixture', start=start, end=end, shots=tuple((x, x) for x in shots),
                input_identity=identity, donor=donor, runtime_fully_recorded=False)


class CausalPrefix(unittest.TestCase):
    def test_only_already_completed_job_is_available(self):
        jobs = [job('a', 0, 10, [4, 4, 4]), job('b', 1, 4, [4, 4, 3]), job('c', 5, 6, [4, 4, 2])]
        result = A.prefix_opportunities(jobs, [1])
        a, b, c = result['jobs']
        self.assertEqual([r['reusable_rounds'] for r in (a, b, c)], [0, 0, 2])
        self.assertEqual(c['completed_donor'], 'b')
        self.assertEqual(b['shared_with_running_rounds'], 2)
        self.assertEqual(result['running_only_prefix_opportunities'], 1)

    def test_donor_completion_at_launch_is_available(self):
        result = A.prefix_opportunities([job('a', 0, 1, [4, 4]), job('b', 1, 2, [4, 3])], [0])
        self.assertEqual(result['jobs'][1]['reusable_rounds'], 1)

    def test_different_contract_or_signal_does_not_share(self):
        result = A.prefix_opportunities([job('a', 0, 1, [4, 4], 'one'), job('b', 2, 3, [4, 3], 'two')], [0])
        self.assertEqual(result['launches_with_earlier_completed_prefix'], 0)

    def test_failed_or_unverified_job_never_becomes_donor(self):
        result = A.prefix_opportunities([job('a', 0, 1, [4, 4], donor=False), job('b', 2, 3, [4, 3])], [0])
        self.assertEqual(result['launches_with_earlier_completed_prefix'], 0)

    def test_checkpoint_boundary_controls_usable_prefix(self):
        jobs = [job('a', 0, 1, [4, 4, 4, 4]), job('b', 2, 3, [4, 4, 3, 3])]
        self.assertEqual(A.prefix_opportunities(jobs, [2])['jobs'][1]['reusable_rounds'], 0)
        self.assertEqual(A.prefix_opportunities(jobs, [1, 2])['jobs'][1]['reusable_rounds'], 2)

    def test_parallel_jobs_are_not_added_as_cpu_or_wall(self):
        result = A.interval_stats([job('a', 0, 10, [1, 1]), job('b', 0, 10, [1, 1])], 2)
        self.assertEqual(result['span_seconds'], 10)
        self.assertEqual(result['sum_terminal_job_wall_seconds'], 20)
        self.assertEqual(result['slot_utilization'], 1)
        self.assertEqual(result['idle_slot_seconds'], 0)

    def test_identity_excludes_only_per_call_labels(self):
        request = dict(method='ARTIFICIAL', configuration_id='cfg', expected={'binding': 'same'}, N_X=[4, 3], N_Y=[4, 3],
                       route=dict(view_commit='source', interpreter='/fixture/python'))
        case = dict(configuration={'commit': 'cfg-source'}, signal_packet={'commit': 'raw-source'}, evaluation_target={'energy': 0},
                    binding_id='binding', total_configuration_id='total', label='a',
                    allocation=dict(allocation_id='a', shots=[dict(m=0, N_X=4, N_Y=4), dict(m=1, N_X=3, N_Y=3)]))
        inputs = dict(cases=[case], statistical_contract={'iid': True}, options={'frontier': 5}, adopted_evaluation={'tau': 1},
                      kernel_source='fixed', C_source='fixed', acceleration={'source': 'fixed', 'workers': 1})
        original = A.input_identity(request, inputs, {})[0]
        clone = deepcopy(inputs)
        clone['cases'][0]['label'] = 'b'
        clone['cases'][0]['allocation']['allocation_id'] = 'b'
        self.assertEqual(A.input_identity(request, clone, {})[0], original)
        for field in ('options', 'statistical_contract', 'adopted_evaluation'):
            clone = deepcopy(inputs)
            clone[field]['changed'] = True
            self.assertNotEqual(A.input_identity(request, clone, {})[0], original)
        clone = deepcopy(inputs)
        clone['cases'][0]['signal_packet']['commit'] = 'different'
        self.assertNotEqual(A.input_identity(request, clone, {})[0], original)
        clone = deepcopy(inputs)
        clone['cases'][0]['allocation']['shots'][0]['N_Y'] = 1
        with self.assertRaisesRegex(ValueError, 'allocation mismatch'):
            A.input_identity(request, clone, {})

    def test_missing_cpu_is_not_fabricated_at_reach(self):
        a = dict(job('a', 0, 2, [1, 1]), status=A.PASS, cost='10', tree_CPU_seconds=5)
        b = dict(job('b', 1, 3, [1, 1]), status=A.PASS, cost='9', tree_CPU_seconds=None)
        result = A.reach_levels([a, b], ['10'])[0]
        self.assertEqual(result['wall_from_first_launch_seconds'], 2)
        self.assertEqual(result['completed_tree_CPU_lower_bound_seconds'], 5)
        self.assertIsNone(result['eventual_launched_tree_CPU_upper_bound_seconds'])
        self.assertIsNone(result['exact_CPU_at_threshold_seconds'])


if __name__ == '__main__':
    unittest.main()
