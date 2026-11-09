"""Regression of the keyword interface that failed before old optimization."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from complete_track_d_stream_bridge_benchmark import compatible_baseline,ORIGINAL_BASELINE
from trottertracks.track_d_native_resources import NativeInstruction as I


class AdapterTest(unittest.TestCase):
    def test_common_keyword_call_preserves_original_output(self):
        import qiskit
        word=[I('cx',(0,1)),I('rz',(0,),params=(0.,)),I('cx',(0,1)),I('rz',(1,),params=(.27,))]
        with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
              patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
            actual=compatible_baseline(iter(word),2,0,global_phase=.31)
            expected=ORIGINAL_BASELINE(iter(word),2,0,.31)
        for key in ('input','output','operations','global_phase','rounds','phase_additions'):
            self.assertEqual(actual[key],expected[key])


if __name__=='__main__':unittest.main(verbosity=2)
