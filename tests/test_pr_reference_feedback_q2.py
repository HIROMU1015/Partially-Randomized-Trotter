"""Artificial native-helper and reset-layout tests; no scientific input."""
from pathlib import Path
import sys
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import pr_reference_feedback_q2 as q2


class NativeReuseTests(unittest.TestCase):
    def test_sample_selection_uses_existing_json_helper(self):
        ns=q2.helpers();original=ns['archive']
        ns['archive']=lambda path:{str(i):q2.json.dumps(row).encode() for i,row in enumerate([
            dict(q=1,j=0,rare=True),dict(q=1,j=0,rare=False)])}
        try:self.assertEqual([row['rare'] for row in ns['sample_rows']()],[False,True])
        finally:ns['archive']=original

    def test_pinned_native_action_all_gates_and_phase(self):
        q=QuantumCircuit(3)
        q.rz(.37,0);q.sx(2);q.cx(2,0);q.x(1);q.cx(0,2);q.global_phase=.23
        expected=Operator(q).data
        for column in range(8):
            v=np.eye(8,dtype=complex)[:,column]
            actual=q2.native_action(q,v)
            self.assertLess(np.linalg.norm(actual-expected[:,column]),1e-12)

    def test_reference_reset_and_measure_layout(self):
        q=QuantumCircuit(17,1);q.reset(range(8,16));q.rz(.3,16);q.measure(16,0)
        self.assertEqual(q2.layout(q,'Reference_B1')['reset_prefix'],list(range(8,16)))
        bad=QuantumCircuit(17,1);bad.h(16);bad.reset(range(8,16));bad.measure(16,0)
        with self.assertRaises(RuntimeError):q2.layout(bad,'Reference_B1')

    def test_direct_raw_tail_sign(self):
        core=QuantumCircuit(2);core.cp(.375,1,0)
        state=np.array([.5,.5,.5,.5],complex)
        after=Operator(core).data@state
        z=2*np.vdot(after[:2],after[2:])
        for name,angle in q2.ANGLES:
            angle=float.fromhex(angle)
            feedback=q2.q1.FeedbackAngle(angle,None,q2.F(),0)
            raw=q2.q1.feedback_readout(core,feedback,family='Bprime')
            self.assertAlmostEqual(q2.raw_mean(after,raw),float((np.exp(-1j*angle)*z).real),places=13)


if __name__=='__main__':unittest.main()
