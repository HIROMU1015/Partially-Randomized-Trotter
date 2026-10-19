"""Exact artificial codec/driver comparisons; no science inputs or RNG streams."""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import test_g6_w2c_b3_driver as T
import test_g6_w2c_b3_f01_provenance as P
import test_g6_w2c_b3_host_integration as I

from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_driver as D
from trottertracks.algorithm_codesign import g6_w2c_b3_driver as REFERENCE
S=D.serialization


def fused_set_up(self):
    context=patch.object(D,'SERIALIZATION_MODE','fused');context.__enter__()
    self.addCleanup(context.__exit__,None,None,None)


def reference(x):return D.audit.canonical(REFERENCE.exact(x))


class Canonical(unittest.TestCase):
    setUp=fused_set_up
    def equal(self,value):
        a,b=reference(value),S.canonical(value)
        self.assertEqual(a,b)
        self.assertEqual(a.encode(),b.encode())
        self.assertEqual(sha256(a.encode()).hexdigest(),D.identity(value))

    def test_native_tree_and_event_fields(self):
        x={'event':T.event(),'nested':[(),[],{2:F(-1,7),'none':None,'flag':True}], 'zero':0}
        old=deepcopy(x);self.equal(x);self.assertEqual(x,old)

    def test_huge_integer_fraction_hex(self):self.equal([2**30000,-2**27000,F(2**12000+1,2**11000+3)])
    def test_key_order(self):self.assertEqual(S.canonical({'z':1,'a':2}),S.canonical({'a':2,'z':1}))
    def test_list_tuple_normalization(self):self.assertEqual(S.canonical([1,[2,3]]),S.canonical((1,(2,3))))

    def test_provider_integer_keys(self):self.equal({0:1,1:2,2:{3:F(5,7)}})
    def test_collision_overwritten_unsupported_value(self):self.equal({1:object(),'1':F(2,3)})
    def test_collision_nested_normalization_error_still_raised(self):
        class Key:
            def __str__(self):raise ValueError('artificial key conversion')
        value={1:{Key():0},'1':F(1,2)}
        for function in (reference,S.canonical):
            with self.assertRaisesRegex(ValueError,'artificial key conversion'):function(value)

    def test_collision_unsupported_survivor_is_rejected(self):
        for function in (reference,S.canonical):
            with self.assertRaises(D.audit.InvalidEvidence):function({1:0,'1':object()})

    def test_container_subclasses_reference_fallback(self):
        class L(list):pass
        class M(dict):pass
        class U(tuple):pass
        self.equal(M({'a':L([1,U((2,F(3,5)))])}))

    def test_string_subclass_values_rejected(self):
        class V(str):pass
        for function in (reference,S.canonical):
            with self.assertRaises(D.audit.InvalidEvidence):function([V('not-native')])

    def test_unsupported_types(self):
        for value in (0.5,complex(1,2),b'bytes',set((1,2)),object()):
            for function in (reference,S.canonical):
                with self.subTest(value=type(value).__name__,function=function.__name__),self.assertRaises(D.audit.InvalidEvidence):function(value)

    def test_bounded_random_native_tree_1000(self):
        rng=random.Random(19317)  # Artificial fixture, not a scientific bit provider.
        def tree(depth):
            if not depth:return rng.choice([None,False,True,'\x7f\ud800é',rng.randrange(-100,100),F(rng.randrange(-500,500),rng.randrange(1,100))])
            kind=rng.randrange(4)
            if kind==0:return [tree(depth-1)for _ in range(rng.randrange(4))]
            if kind==1:return tuple(tree(depth-1)for _ in range(rng.randrange(4)))
            if kind==2:return {str(i):tree(depth-1)for i in range(rng.randrange(4))}
            return tree(0)
        for i in range(1000):self.equal(tree(4))


    def test_recovered_server_prototype_and_collision_counterexample(self):
        row=json.loads((T.ROOT/'tests/tracks/algorithm_codesign/fixtures/g6_w2c_b3_pilot_exact_reference.json').read_text())
        namespace={'F':F,'D':D,'json':json}
        for name in ('packed_once','canonical_fast'):exec(row['functions'][name],namespace)
        f=namespace['canonical_fast']
        value={'event':T.event(),'unicode':'\x7f\ud800日本語'}
        self.assertEqual(f(value),S.canonical(value))
        collision={1:object(),'1':F(2,3)}
        with self.assertRaises(D.audit.InvalidEvidence):f(collision)
        self.equal(collision)  # Our reference fallback preserves last-wins semantics.

    def test_original_pack_is_not_called_on_normal_path(self):
        with patch.object(D.audit,'_pack',side_effect=AssertionError('reference packing used')):
            self.assertTrue(S.canonical({'a':[1,F(2,3),'é']}))

    def test_stream_full_bytes_chain_and_file_hash(self):
        with tempfile.TemporaryDirectory(prefix='b3-exact-codec-')as temp:
            streams=[]
            for i,module in enumerate((REFERENCE,D)):
                s=module.Stream(Path(temp)/str(i),module.Meter(T.FakeClock()),byte_limit=1000000,flush_rows=2)
                for v in ({'a':T.event()},{'text':'\ud800\x7f\n'},[F(-3,7),2**999]):s.emit(v)
                s.close();streams.append(s)
            a,b=streams
            self.assertEqual(a.path.read_bytes(),b.path.read_bytes());self.assertEqual(a.previous,b.previous)
            self.assertEqual(a.file_hash.hexdigest(),b.file_hash.hexdigest())
            self.assertEqual(a.meter.rows,b.meter.rows)

    def test_cyclic_input_rejected(self):
        value=[];value.append(value)
        for function in (reference,S.canonical):
            with self.assertRaises(RecursionError):function(value)


for i,value in enumerate(['ASCII','日本語 é α','\ud800','\udfff','\ud800\udfff','\x7f','\x00\n\r\t"\\','\U0001f680']):
    def make(v):
        def test(self):self.equal({'text':v,'key'+v:(v,)})
        return test
    setattr(Canonical,'test_unicode_control_'+str(i),make(value))


class F01Exact(unittest.TestCase):
    setUp=fused_set_up
    def exercise(self,module,mode):
        with tempfile.TemporaryDirectory(prefix='b3-exact-failure-')as temp:
            e=P.ToyEndpoint(fail_sample=(lambda t:t.endswith('/X/1'))if mode=='next'else
                           (lambda t:t.endswith('/Y/0'))if mode=='axis'else None,
                           fail_angle=mode=='angle',bad_evidence=mode=='evidence')
            meter=module.Meter(T.FakeClock());s=module.Stream(Path(temp)/'out',meter,byte_limit=1000000,flush_rows=1)
            if mode in ('partial','flush'):s.file=P.FailingFile(s,mode)
            cells=[dict(arm='full_return',layer='B',axis=x,N=2)for x in ('X','Y')]
            try:module.run_pair(e,lambda axis:T.Bits(iter((2,)*10)),cells,s,meter,phase='profile',pass_name='baseline',correctness_trials=16)
            except BaseException as exc:
                result=(type(exc).__name__,str(exc),exc.b3_failure_provenance)
            else:raise AssertionError('failure not injected')
            try:s.close()
            except OSError:pass
            return result,s.path.read_bytes(),meter.rows

for mode in ('next','axis','angle','evidence','partial','flush'):
    def make(m):
        def test(self):self.assertEqual(self.exercise(REFERENCE,m),self.exercise(D,m))
        return test
    setattr(F01Exact,'test_reference_fast_failure_'+mode,make(mode))
