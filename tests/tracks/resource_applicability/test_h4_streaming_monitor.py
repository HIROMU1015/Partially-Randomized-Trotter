"""JSON metadata and mocked monitoring only; no transpile or science inputs."""
import hashlib
import json
import threading
import unittest
from unittest.mock import patch
from trottertracks.resource_applicability.h4_geometry import identity as ident
from trottertracks.resource_applicability.h4_geometry import execution


class StreamingIdentityTests(unittest.TestCase):
    def test_exact_digest_matches_original_encoder_for_all_identity_types(self):
        values = [None,True,False,0,-123,2**100,'日本語/🙂\n"\\',-0.,0.,1e-300,
                  float.fromhex('0x1.fffffffffffffp+1023'),complex(-0.,1.),
                  ('a',1,1j),{'z':[1.,{'a':-1j}], 'a':3}]
        for value in values:
            payload={'domain':'ARTIFICIAL','value':value}
            original=hashlib.sha256(ident.canonical(payload)).hexdigest()
            self.assertEqual(ident.fingerprint('ARTIFICIAL',{'value':value}),original)

    def test_large_digest_streams_with_cooperative_chunks(self):
        leaf={'complex128_hex':['0x1.123456789abcdep-1','-0x0.0p+0']}
        payload={'matrix':[[leaf]*64 for _ in range(64)],'repeat':[leaf]*8192}
        original=hashlib.sha256(ident.canonical({'domain':'ARTIFICIAL',**payload})).hexdigest()
        with patch.object(ident.time,'sleep') as sleep:
            actual=ident.fingerprint('ARTIFICIAL',payload)
        self.assertEqual(actual,original)
        self.assertGreater(sleep.call_count,4)
        sleep.assert_called_with(0)

    def test_nonfinite_symbolic_and_nonstring_keys_still_rejected(self):
        for value in [float('nan'),float('inf'),complex(1,float('inf')),
                      {1:'bad'},object()]:
            with self.assertRaises(ident.Stop):
                ident.fingerprint('ARTIFICIAL',{'value':value})

    def test_whole_exact_tree_is_not_created_for_fingerprint(self):
        value={'rows':[[{'complex128_hex':['0x1.0p+0','0x0.0p+0']}]*32]*32}
        expected=hashlib.sha256(ident.canonical({'domain':'ARTIFICIAL',**value})).hexdigest()
        with patch.object(ident,'exact',side_effect=AssertionError('whole tree copy')):
            self.assertEqual(ident.fingerprint('ARTIFICIAL',value),expected)

    def test_array_metadata_memo_is_call_local_and_preserves_exact_bytes(self):
        import numpy as np
        from trottertracks.resource_applicability.h4_geometry import circuits
        array=np.array([[1.,-0.],[0.,1.]],dtype=np.complex128)
        baseline=circuits.number(array);memo={}
        first=circuits.number(array,memo);second=circuits.number(array,memo)
        self.assertIs(first,second)
        self.assertEqual(ident.canonical(first),ident.canonical(baseline))
        array[0,0]=2
        self.assertNotEqual(circuits.number(array),first)

    def test_first_monitor_exception_logged_before_owned_cleanup(self):
        run=execution.OwnedRun.__new__(execution.OwnedRun)
        class Event:
            def wait(self,timeout):return False
        class Monitor:
            def poll(self):raise ident.Stop('synthetic monitor interval/freshness')
            def stop_children(self):pass
        run.finished=Event();run.monitor=Monitor();run.failure=None
        with patch('builtins.print') as print_,patch('_thread.interrupt_main') as interrupt:
            run._watch()
        self.assertIsInstance(run.failure,ident.Stop)
        print_.assert_called_once_with('H4 MONITOR STOP: Stop: synthetic monitor interval/freshness',flush=True)
        interrupt.assert_called_once()


if __name__=='__main__':unittest.main()
