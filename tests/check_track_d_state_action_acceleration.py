"""Synthetic equivalence only; never sample/build/compile a molecular path."""
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_track_d_c6b as frozen
frozen.c2.old().bootstrap(Path(os.environ['TRACK_D_ORIGIN']))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from threadpoolctl import threadpool_limits
from trottertracks.c1_h6_state_action import StateActions
from trottertracks.track_d_state_action_acceleration import BatchedStateActions, stage_memory_estimates


class AccelerationTests(unittest.TestCase):
    def circuit(self):
        qc=QuantumCircuit(3)
        qc.global_phase=.371
        for k in range(12):
            qc.rz((-1)**k*.031*(k+1),k%3)
            qc.rx(.017*(k+1),(k+1)%3)
            qc.cx(k%3,(k+1)%3)
        block=QuantumCircuit(3);block.ry(.28,1);block.global_phase=-.19
        qc.compose(block,inplace=True)
        return qc

    def test_all_columns_phase_inverse_and_input_order_match_existing_and_dense(self):
        qc=self.circuit()
        inputs=np.vstack([np.eye(8,dtype=complex),np.eye(8,dtype=complex)[::-1],
                          np.ones((1,8),complex)/np.sqrt(8)])
        expected=inputs@Operator(qc).data.T
        before=Operator(qc).data.copy()
        for threads in [1,3]:
            action=BatchedStateActions(threads)
            with threadpool_limits(limits=1):
                actual=action.apply(qc,inputs)
            old=StateActions(threads).apply(qc,inputs)
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual,old,atol=1e-12,rtol=0)
            np.testing.assert_allclose(action.apply(qc.inverse(),actual),inputs,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(Operator(qc).data,before)
            self.assertLessEqual(max(action.last_batch_sizes),threads)
            self.assertEqual(sum(action.last_batch_sizes),len(inputs))
            if threads > 1:
                self.assertGreater(max(r.get('parallel_experiments',1) for r in action.last_parallel_metadata),1)

    def test_repeat_uses_cached_conversion_without_modifying_any_circuit(self):
        qc=self.circuit();action=BatchedStateActions(2);inputs=np.eye(8,dtype=complex)
        first=action.apply(qc,inputs);entry=action.cache[id(qc)]
        second=action.apply(qc,inputs)
        self.assertIs(action.cache[id(qc)][1],entry[1])
        np.testing.assert_array_equal(first,second)

    def test_invalid_inputs_remain_errors(self):
        qc=self.circuit();action=BatchedStateActions(2)
        for bad in [np.zeros((2,7)),np.array([[float('nan')]*8]),np.empty((0,8))]:
            with self.assertRaises(ValueError):action.apply(qc,bad)

    def test_memory_stages_use_completed_peaks_and_reject_instant_partial_RSS(self):
        evidence=dict(raw=[dict(status='PASS',peak_RSS_KiB=7),dict(status='PASS',peak_RSS_KiB=8)],
                      reference=[dict(status='PASS',peak_RSS_KiB=32)],direct=[dict(status='PASS',peak_RSS_KiB=1)])
        values=stage_memory_estimates(**evidence)
        self.assertEqual(values['raw_task_memory_estimate_bytes'],8*1024)
        self.assertEqual(values['reference_task_memory_estimate_bytes'],32*1024)
        for bad in [[],[dict(status='STARTED',peak_RSS_KiB=2)],[dict(status='PASS',current_RSS_KiB=2)]]:
            with self.assertRaises(ValueError):stage_memory_estimates(**{**evidence,'raw':bad})


if __name__=='__main__':unittest.main()
