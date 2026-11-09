"""Small artificial native words; no P0 compile or molecular input."""
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from trottertracks.track_d_native_resources import NativeInstruction as I, count_native
from trottertracks.track_d_p0_cx_bridge import CXBoundaryBridge


class BridgeTests(unittest.TestCase):
    def test_adjacent_pair_and_odd_run(self):
        cx=I("cx",(0,1))
        self.assertEqual(list(CXBoundaryBridge(2).extend([cx,cx]).operations()),[])
        self.assertEqual(list(CXBoundaryBridge(2).extend([cx,cx,cx]).operations()),[cx])

    def test_rz_blocks_on_either_wire(self):
        cx=I("cx",(0,1))
        for wire in (0,1):
            word=[cx,I("rz",(wire,),params=(.23,)),cx]
            self.assertEqual(list(CXBoundaryBridge(2).extend(word).operations()),word)

    def test_disjoint_operations_and_direction(self):
        cx=I("cx",(0,1)); x=I("x",(2,))
        self.assertEqual(list(CXBoundaryBridge(3).extend([cx,x,cx]).operations()),[x])
        opposite=I("cx",(1,0))
        self.assertEqual(list(CXBoundaryBridge(2).extend([cx,opposite]).operations()),[cx,opposite])

    def test_cascade_and_chunk_invariance(self):
        a,b=I("cx",(0,1)),I("cx",(1,2))
        alphabet=(a,b,I("rz",(0,),params=(-.17,)),I("sx",(2,)))
        for length in range(5):
            for word in product(alphabet,repeat=length):
                expected=CXBoundaryBridge(3).extend(iter(word))
                for cut in range(length+1):
                    split=CXBoundaryBridge(3).extend(iter(word[:cut])).extend(iter(word[cut:]))
                    self.assertEqual(list(split.operations()),list(expected.operations()))
                    self.assertEqual(split.result(),expected.result())
                    self.assertEqual(split.template().result(),split.result()['output'])
        self.assertEqual(list(CXBoundaryBridge(3).extend([a,b,b,a]).operations()),[])

    def test_no_one_qubit_or_parity_rewrite(self):
        word=[I("rz",(0,),params=(.23,)),I("rz",(0,),params=(-.23,)),
              I("x",(1,)),I("x",(1,))]
        bridge=CXBoundaryBridge(2).extend(word)
        self.assertEqual(list(bridge.operations()),word)
        self.assertEqual(bridge.result()['p0_qualification'],'P0_UNQUALIFIED')

    def test_measurement_dependency_and_phase(self):
        word=[I("cx",(0,1)),I("measure",(1,),(0,)),I("cx",(0,1))]
        bridge=CXBoundaryBridge(2,1,global_phase=.37).extend(word)
        self.assertEqual(bridge.result()['output'],count_native(iter(word),2,1,global_phase=.37))

    def test_frontier_can_grow_and_expose_early_pairs(self):
        first=[I("cx",(0,1)),I("cx",(1,2))]*16
        bridge=CXBoundaryBridge(3).extend(first)
        self.assertEqual(bridge.peak_live_nodes,32)
        bridge.extend(reversed(first))
        self.assertEqual(list(bridge.operations()),[])
        self.assertEqual(bridge.cancelled_pairs,32)

    def test_reuse_t0_rejections(self):
        with self.assertRaises(ValueError):CXBoundaryBridge(1).feed(I("cx",(0,1)))
        with self.assertRaises(TypeError):CXBoundaryBridge(2).feed(("cx",(0,1)))


if __name__=='__main__':unittest.main(verbosity=2)
