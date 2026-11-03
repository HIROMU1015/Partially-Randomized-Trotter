"""Serialized event representation tests, without scientific dispatch."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from repair_h8_long_time_recovery import same_event_metadata


class OriginalEventMetadataTests(unittest.TestCase):
    def test_json_arrays_and_original_tuples_compare_identically(self):
        same_event_metadata(dict(seeds=[17], events=[dict(qubits=[0, 2], phase=1.)]),
                            dict(seeds=[17], events=[dict(qubits=(0, 2), phase=1.)]))

    def test_a_changed_seed_parameter_or_event_order_is_rejected(self):
        saved = dict(seeds=[17], events=[dict(qubits=[0, 2], phase=1.)])
        for actual in [dict(seeds=[18], events=saved['events']),
                       dict(seeds=[17], events=[dict(qubits=(0, 2), phase=-1.)]),
                       dict(seeds=[17], events=[dict(qubits=(2, 0), phase=1.)])]:
            with self.assertRaisesRegex(ValueError, 'EXACT_ORIGINAL_EVENTS_ONLY'):
                same_event_metadata(saved, actual)


if __name__ == '__main__':
    unittest.main()
