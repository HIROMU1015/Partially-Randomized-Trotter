"""Exact differential artificial tests; no P0/H4 generation or simulation."""
from itertools import product,chain
from pathlib import Path
import math
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.track_d_native_resources import NativeInstruction as I
from trottertracks.track_d_p0_1q_bridge import optimize_native,oneq_runs
from trottertracks.track_d_stream_bridge_fast import optimize_native_stream,OneQStream,ObservedCX


def signature(word):
    return [(o.name,o.qubits,o.clbits,tuple(v.hex() for v in o.params)) for o in word]


class FastTests(unittest.TestCase):
    def same(self,word,nq=2,nc=0):
        snapshot=signature(word);expected=optimize_native(iter(word),nq,nc,global_phase=.17)
        actual=optimize_native_stream(iter(word),nq,nc,global_phase=.17)
        self.assertEqual(signature(actual['operations']),signature(expected['operations']))
        for key in ('input','output','global_phase','rounds'):self.assertEqual(actual[key],expected[key],key)
        self.assertEqual(signature(word),snapshot)
        return actual

    def test_all_short_native_words(self):
        alphabet=[I('rz',(0,),params=(0.,)),I('rz',(1,),params=(.23,)),I('sx',(0,)),
                  I('x',(1,)),I('cx',(0,1)),I('cx',(1,0))]
        for n in range(4):
            for word in product(alphabet,repeat=n):self.same(list(word))

    def test_signed_zero_special_angles_and_repeated_runs(self):
        word=[I('rz',(0,),params=(v,)) for v in (0.,-0.,2*math.pi,-2*math.pi,.23,-.23)]
        word+=[I('sx',(1,)),I('sx',(1,)),I('x',(1,))]
        self.same(word)
        self.same([I('cx',(0,1)),I('rz',(0,),params=(0.,)),I('cx',(0,1))]*8)

    def test_nested_cancellation_and_classical_dependencies(self):
        word=[I('cx',(i%2,(i+1)%2)) for i in range(15)]
        self.same(word+list(reversed(word)))
        self.same([I('sx',(1,)),I('cx',(0,1)),I('measure',(0,),(0,)),
                   I('rz',(1,),params=(.31,)),I('measure',(1,),(0,))],nc=1)

    def test_oneq_stream_preserves_full_flush_order_and_phase(self):
        word=[I('rz',(1,),params=(.3,)),I('sx',(0,)),I('rz',(1,),params=(-.3,)),
              I('cx',(0,1)),I('x',(1,)),I('rz',(0,),params=(0.,))]
        expected,phase,replaced=oneq_runs(iter(word),2)
        stream=OneQStream(2,{})
        self.assertEqual(signature(list(stream.run(iter(word)))),signature(expected))
        self.assertEqual(math.fsum(stream.phase),phase);self.assertEqual(stream.replaced,replaced)

    def test_all_chunk_splits_are_one_continuous_bridge(self):
        word=[I('cx',(0,1)),I('rz',(1,),params=(0.,)),I('cx',(0,1)),I('sx',(0,)),I('x',(1,))]
        expected=optimize_native_stream(iter(word),2)
        for cut in range(len(word)+1):
            actual=optimize_native_stream(chain(iter(word[:cut]),iter(word[cut:])),2)
            self.assertEqual(signature(actual['operations']),signature(expected['operations']))
            self.assertEqual(actual['phase_additions'],expected['phase_additions'])

    def test_unbounded_frontier_not_silently_truncated(self):
        cx=ObservedCX(2).extend([I('rz',(0,),params=(.1,)) for _ in range(500)])
        self.assertEqual(cx.peak_live_nodes,500);self.assertEqual(cx.statistics()['peak_single_wire_stack'],500)

    def test_invalid_input_is_still_checked_once(self):
        for word in [[object()],[I('cx',(0,2))]]:
            with self.assertRaises((TypeError,ValueError)):optimize_native_stream(iter(word),2)


if __name__=='__main__':unittest.main(verbosity=2)
