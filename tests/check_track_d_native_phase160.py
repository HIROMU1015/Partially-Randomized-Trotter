"""Small artificial phase accounting only; no P0/H4 simulation."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from trottertracks.track_d_native_phase160 import PhaseLedger,lower_with_ledger,optimize_with_ledger
from trottertracks.track_d_native_emitter import NativeSink
from run_track_d_h4_native_bridge import observed_optimize


class PhaseTests(unittest.TestCase):
    def test_long_accumulation_keeps_lost_small_term(self):
        p=PhaseLedger([1e16,1.,-1e16]);r=p.result()
        self.assertEqual(r['legacy_running'],0.)
        self.assertEqual(r['binary64_full_fsum'],1.)
        self.assertEqual(float(p.total),1.)
        self.assertEqual(r['normalized'],1.)

    def test_exact_binary64_input_not_decimal_shortening(self):
        p=PhaseLedger([.1]);n,d=(.1).as_integer_ratio()
        expected=p.se.RealMPFR(n,160)/p.se.RealMPFR(d,160)
        self.assertEqual(p.total,expected)
        self.assertNotEqual(p.total,p.se.RealMPFR('0.1',160))
        self.assertTrue(0<=PhaseLedger([-1e6]).result()['normalized']<2*math.pi)

    def test_same_native_word_and_phase_aware_artificial_action(self):
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import HGate,PhaseGate,RZGate,SXGate,XGate,CXGate
        from qiskit.quantum_info import Operator
        import numpy as np
        raw=QuantumCircuit(2);raw.global_phase=.19;raw.h(0);raw.p(.31,1)
        raw.cx(0,1);raw.rz(0.,0);raw.cx(0,1);raw.rzz(-.27,0,1)
        with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
              patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
            old=[];sink=NativeSink(2,consumer=old.append);sink.compose(raw,inplace=True)
            word=[];ledger=lower_with_ledger(raw,word.append)
            self.assertEqual(word,old);self.assertEqual(ledger.running,sink.global_phase)
            result,_,final,_=optimize_with_ledger(word,2,0,ledger,observed_optimize)
            expected,_=observed_optimize(old,2,0,sink.global_phase)
            self.assertEqual(result,expected)
            qc=QuantumCircuit(2);qc.global_phase=final.result()['normalized']
            gates=dict(rz=RZGate,sx=SXGate,x=XGate,cx=CXGate)
            for op in result['operations']:qc.append(gates[op.name](*op.params),list(op.qubits))
            self.assertLess(float(np.max(abs(Operator(qc).data-Operator(raw).data))),1e-10)

    def test_full_global_phase_vs_relative_readout_phase(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        import numpy as np
        z=np.diag([1.,1.,-1.,-1.])
        def observable(global_phase,relative):
            c=QuantumCircuit(2);c.h(1);c.cx(1,0);c.p(relative,1);c.h(1)
            c.global_phase=global_phase;u=Operator(c).data
            return (u.conj().T@z@u)[:2,:2]
        self.assertLess(float(np.max(abs(observable(.23,.31)-observable(.47,.31)))),1e-12)
        self.assertGreater(float(np.max(abs(observable(.23,.31)-observable(.23,.32)))),1e-4)


if __name__=='__main__':unittest.main(verbosity=2)
