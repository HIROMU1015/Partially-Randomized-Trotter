"""Artificial byte/guard controls only; no registered operator or backend."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign.parent_experiment.evidence import BatchedEvidenceWriter
from trottertracks.algorithm_codesign.rte_development.evidence import EvidenceWriter, verify_chunks
from trottertracks.algorithm_codesign.parent_experiment.design import integer_io_limit


class BufferedEvidenceTests(unittest.TestCase):
    def compare(self, value, chunk_bytes=524288):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';new=Path(tmp)/'new'
            old.mkdir();new.mkdir()
            a=EvidenceWriter(old,8000000,chunk_bytes=chunk_bytes).write('data.json',value)
            b=BatchedEvidenceWriter(new,8000000,chunk_bytes=chunk_bytes).write('data.json',value)
            self.assertEqual(a,b)
            self.assertEqual((old/'data.json').read_bytes(),(new/'data.json').read_bytes())
            self.assertEqual(sorted(p.name for p in old.iterdir()),sorted(p.name for p in new.iterdir()))
            for p in old.iterdir():self.assertEqual(p.read_bytes(),(new/p.name).read_bytes())
            if b['chunks']:self.assertTrue(verify_chunks(new,b))

    def test_same_bytes_for_nested_exact_values(self):
        self.compare(dict(n=17,p=F(2,11),row=[dict(label='off-domain',empty=None,values=[True,False,0,-3,F(1,13)])]*31))

    def test_utf8_escaping_and_block_boundaries(self):
        self.compare(dict(text=('界😀\\\"\n\t'*12000),tail='終'))

    def test_natural_review_chunks_are_identical(self):
        self.compare(dict(text='日本😀'*70000))

    def test_small_review_chunk_boundary_is_identical(self):
        self.compare(dict(text='境😀'*301),chunk_bytes=31)

    def test_many_fragments_have_identical_bytes(self):
        self.compare([dict(i=i,law='artificial',gates=[['R',0,'2/11',1],['X',1]],q=F(i+1,4001)) for i in range(180)])

    def test_large_fraction_decimal_policy_preserved(self):
        with integer_io_limit(65536):self.compare(dict(q=F(10**5500+3,10**5400+7)))

    def test_prewrite_budget_guard_and_partial_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);w=BatchedEvidenceWriter(d,80000)
            with self.assertRaises(RuntimeError):w.write('large.json',dict(text='界'*50000))
            self.assertFalse((d/'large.json').exists())
            self.assertTrue((d/'large.json.partial').exists())
            data=(d/'large.json.partial').read_bytes();data.decode('utf-8')
            self.assertLessEqual(len(data),80000-32768);self.assertEqual(w.files,[])

    def test_failure_does_not_allow_same_output_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=BatchedEvidenceWriter(tmp,33000)
            with self.assertRaises(RuntimeError):w.write('x.json',dict(text='x'*20000))
            with self.assertRaises(FileExistsError):w.write('x.json',{})

    def test_guard_interruption_is_propagated_before_next_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def guard():
                calls.append(1)
                if len(calls)==3:raise RuntimeError('artificial CPU guard')
            w=BatchedEvidenceWriter(tmp,1000000,guard)
            with self.assertRaisesRegex(RuntimeError,'artificial CPU'):w.write('x.json',dict(text='界'*40000))
            self.assertEqual(len(calls),3)
            data=(Path(tmp)/'x.json.partial').read_bytes();data.decode('utf-8')
            self.assertLessEqual(len(data),32768);self.assertFalse((Path(tmp)/'x.json').exists())

    def test_external_output_is_included_in_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);calls=[]
            def guard():
                calls.append(1)
                if len(calls)==2:(d/'external.json').write_bytes(b'x'*45000)
            w=BatchedEvidenceWriter(d,90000,guard)
            with self.assertRaises(RuntimeError):w.write('x.json',dict(text='x'*70000))
            self.assertFalse((d/'x.json').exists());self.assertLessEqual((d/'x.json.partial').stat().st_size,16384)

    def test_same_published_path_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=BatchedEvidenceWriter(tmp,1000000);a=w.write('x.json',{'once':True})
            with self.assertRaises(FileExistsError):w.write('x.json',{'twice':True})
            self.assertEqual(hashlib.sha256((Path(tmp)/'x.json').read_bytes()).hexdigest(),a['sha256'])

    def test_chunk_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=BatchedEvidenceWriter(tmp,1000000,chunk_bytes=31);a=w.write('x.json',dict(text='界'*100))
            p=Path(tmp)/a['chunks'][0]['path'];p.write_bytes(b'x'+p.read_bytes()[1:])
            with self.assertRaises(ArithmeticError):verify_chunks(tmp,a)

    def test_nonfinite_or_unsupported_value_never_published(self):
        for value in (float('nan'),object()):
            with tempfile.TemporaryDirectory() as tmp:
                w=BatchedEvidenceWriter(tmp,1000000)
                with self.assertRaises((TypeError,ValueError)):w.write('x.json',dict(value=value))
                self.assertFalse((Path(tmp)/'x.json').exists());self.assertEqual(w.files,[])

    def test_only_fixed_buffer_is_accepted(self):
        for n in (True,0,4096,32768):
            with self.assertRaises(ValueError):BatchedEvidenceWriter('unused',1000000,write_block_bytes=n)

    def test_file_operation_failure_never_marks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=BatchedEvidenceWriter(tmp,1000000)
            with patch('trottertracks.algorithm_codesign.parent_experiment.evidence.os.fsync',side_effect=OSError('artificial disk failure')):
                with self.assertRaises(OSError):w.write('x.json',dict(value='small'))
            self.assertFalse((Path(tmp)/'x.json').exists());self.assertTrue((Path(tmp)/'x.json.partial').exists());self.assertEqual(w.files,[])


if __name__=='__main__':unittest.main()
