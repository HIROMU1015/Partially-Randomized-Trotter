"""Synthetic instrumentation transparency; no H4 or P0 execution."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from run_track_d_h4_native_bridge import observed_optimize
from trottertracks.track_d_native_resources import NativeInstruction as I
from trottertracks.track_d_p0_1q_bridge import optimize_native


class ObserverTests(unittest.TestCase):
    def test_observer_keeps_same_native_word_metrics_phase_and_rounds(self):
        import qiskit
        words=[[],[I('cx',(0,1)),I('rz',(0,),params=(0.,)),I('cx',(0,1))],
               [I('rz',(0,),params=(.31,)),I('sx',(1,)),I('cx',(0,1)),I('cx',(0,1)),I('x',(1,))]]
        with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
              patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
            for word in words:
                actual,observer=observed_optimize(word,2,0,.17)
                expected=optimize_native(iter(word),2,global_phase=.17)
                self.assertEqual(actual,expected)
                self.assertEqual(len(actual['rounds']),len(observer['cx_rounds']))

    def test_uncommitted_prediction_is_rejected_before_oracle_access(self):
        import run_track_d_h4_native_bridge as runner
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/runner.OUT;out.mkdir(parents=True)
            (out/'PREDICTIONS.json').write_text('{}')
            with (patch.object(runner,'ROOT',Path(directory)),
                  patch.object(runner,'git',return_value=b'changed'),
                  patch.object(runner,'archive_manifest',side_effect=AssertionError('oracle access before freeze'))):
                with self.assertRaisesRegex(ValueError,'committed unchanged'):
                    runner.compare('unused-private-archive','synthetic-commit')


if __name__=='__main__':unittest.main(verbosity=2)
