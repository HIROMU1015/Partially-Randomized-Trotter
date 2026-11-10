"""Technical differential tests, no molecular generation/P0/simulation."""
from itertools import chain, product
from pathlib import Path
import math
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.track_d_native_resources import NativeInstruction as I
from trottertracks import track_d_stream_bridge_fast as old
from trottertracks.track_d_bridge_run_reuse import optimize_native_runs


def signature(word):
    return [(o.name, o.qubits, o.clbits, tuple(v.hex() for v in o.params)) for o in word]


class ReuseTests(unittest.TestCase):
    def same(self, word, nq=3, nc=0):
        snapshot = signature(word)
        before_trace, after_trace = [], []
        class TraceCX(old.ObservedCX):
            def operations(self):
                result = list(super().operations())
                before_trace.append(signature(result))
                return iter(result)
        with patch.object(old, 'ObservedCX', TraceCX):
            expected = old.optimize_native_stream(iter(word), nq, nc, global_phase=.17)
        actual = optimize_native_runs(iter(word), nq, nc, global_phase=.17,
            trace=lambda i, ops, phase: after_trace.append(signature(ops)))
        self.assertEqual(before_trace, after_trace)
        self.assertEqual(signature(actual['operations']), signature(expected['operations']))
        for k in ('input', 'output', 'rounds'):
            self.assertEqual(actual[k], expected[k], k)
        self.assertEqual(actual['global_phase'].hex(), expected['global_phase'].hex())
        self.assertEqual([v.hex() for v in actual['phase_additions']],
                         [v.hex() for v in expected['phase_additions']])
        self.assertEqual(signature(word), snapshot)
        return actual

    def test_exhaustive_short(self):
        alphabet = [I('rz',(0,),params=(0.,)), I('sx',(1,)), I('x',(0,)),
                    I('cx',(0,1)), I('cx',(1,0))]
        for n in range(5):
            for word in product(alphabet, repeat=n):
                self.same(list(word), nq=2)

    def test_many_deterministic_local_dependencies(self):
        # Fixed technical opcode words, not sampling a Hamiltonian/PR law.
        rng = random.Random(170111)
        alphabet = [I('rz',(w,),params=(p,)) for w in range(3)
                    for p in (.23, -.23, 0., -0., math.pi)]
        alphabet += [I(g,(w,)) for w in range(3) for g in ('sx','x')]
        alphabet += [I('cx',(a,b)) for a in range(3) for b in range(3) if a != b]
        for _ in range(100):
            self.same([rng.choice(alphabet) for _ in range(80)])

    def test_cancellation_chain_exposes_merged_runs(self):
        nested = [I('cx',(i%3,(i+1)%3)) for i in range(40)]
        word = [I('rz',(0,),params=(.3,)), I('sx',(2,))]
        word += nested + [I('rz',(2,),params=(0.,))] + list(reversed(nested))
        word += [I('rz',(0,),params=(-.3,)), I('sx',(2,)), I('x',(2,))]
        self.same(word)

    def test_flush_order_after_disjoint_barrier_removed(self):
        self.same([I('rz',(2,),params=(.4,)), I('sx',(0,)), I('cx',(0,1)),
                   I('rz',(2,),params=(-.4,)), I('cx',(0,1)),
                   I('x',(1,)), I('sx',(0,))])
        self.same([I('sx',(2,)), I('rz',(0,),params=(.1,)), I('cx',(2,0)),
                   I('rz',(1,),params=(.8,)), I('cx',(2,0)), I('sx',(2,))])

    def test_classical_blockers_signed_zero_special_angles(self):
        self.same([I('rz',(w,),params=(p,)) for w in (2,0,1)
                   for p in (0., -0., 2*math.pi, -2*math.pi, .2, -.2)]
                  + [I('measure',(0,),(0,)), I('measure',(2,),(0,)),
                     I('cx',(1,2)), I('measure',(1,),(1,)), I('cx',(1,2))], nc=2)

    def test_run_internals_and_cx_feeds_actually_skipped(self):
        word = [I('rz',(0,),params=(.2,)), I('sx',(0,)), I('rz',(0,),params=(.3,))]
        word += [I('cx',(0,1)), I('rz',(1,),params=(0.,)), I('cx',(0,1))]
        word += [I('rz',(2,),params=(.4,)), I('sx',(2,)), I('rz',(2,),params=(.5,))]
        result = self.same(word)
        rows = result['performance']['rounds']
        self.assertGreater(sum(r['memo_native_skipped'] for r in rows), 0)
        self.assertLess(sum(r['cx']['processed'] for r in rows),
                        sum(r['input_size'] for r in result['rounds']))

    def test_chunk_independence_and_invalid_input(self):
        word = [I('cx',(0,1)), I('rz',(0,),params=(0.,)), I('cx',(0,1))]
        expected = signature(optimize_native_runs(iter(word),2)['operations'])
        for cut in range(len(word)+1):
            actual = optimize_native_runs(chain(word[:cut],word[cut:]),2)
            self.assertEqual(signature(actual['operations']), expected)
        for invalid in ([object()], [I('cx',(0,3))]):
            with self.assertRaises((ValueError,TypeError)):
                optimize_native_runs(iter(invalid),3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
