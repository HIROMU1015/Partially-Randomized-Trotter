"""Small artificial integration; no H4 or P0 call."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from trottertracks.track_d_native_phase160 import PhaseLedger,optimize_with_ledger
from run_track_d_h4_native_bridge import observed_optimize
from trottertracks.track_d_native_resources import NativeInstruction as I
from trottertracks.track_d_native_emitter import NativeSink
from trottertracks.track_d_p0_1q_bridge import optimize_native
from trottertracks.track_d_stream_resources import evaluate_native_stream,evaluate_saved_raw


class IntegrationTests(unittest.TestCase):
    def test_phase_is_finalized_after_producer_finishes(self):
        ledger=PhaseLedger([.17]);word=[I('cx',(0,1)),I('rz',(1,),params=(0.,)),I('cx',(0,1))]
        def producer():
            for i,op in enumerate(word):ledger.add((-1000.,.31,7.)[i]);yield op
        actual=evaluate_native_stream(producer(),2,0,ledger)
        expected,_,expected_phase,_=optimize_with_ledger(word,2,0,ledger,observed_optimize)
        for key in ('input','output','operations','global_phase','rounds'):self.assertEqual(actual[key],expected[key])
        self.assertEqual(actual['precise_phase'],expected_phase.result())

    def test_raw_adapter_preserves_old_native_and_scalar_results(self):
        import qiskit
        from qiskit import QuantumCircuit
        c=QuantumCircuit(2,1);c.global_phase=.23;c.h(0);c.rzz(.31,0,1)
        c.cx(0,1);c.rz(0.,0);c.cx(0,1);c.p(-.37,1);c.measure(1,0)
        with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
              patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
            word=[];sink=NativeSink(2,1,consumer=word.append);sink.compose(c,inplace=True)
            expected=optimize_native(iter(word),2,1,global_phase=sink.global_phase)
            actual=evaluate_saved_raw(c)
        for key in ('input','output','operations','global_phase','rounds'):self.assertEqual(actual[key],expected[key])
        self.assertGreater(actual['lowering_statistics']['peak_gate_chunk'],0)


if __name__=='__main__':unittest.main(verbosity=2)
