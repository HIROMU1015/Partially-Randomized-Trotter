"""Artificial opaque-byte compression tests; no actual QPY/NPZ decoding."""
import copy
import io
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lossless',ROOT/'scripts/h46_pilot_lossless_storage.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'input.bin';self.source.write_bytes(bytes(range(256))*8192)
        self.raw=s.digest_file(self.source);self.dest=self.root/'copy.zlib'
    def tearDown(self):self.temp.cleanup()
    def archive(self):return s.archive_file(self.source,self.dest,self.raw)

    def test_roundtrip_and_original_preservation(self):
        stat=self.source.stat();r=self.archive()
        self.assertEqual(s.verify_archive(self.dest,r),self.raw)
        out=self.root/'restored.bin';s.restore_new(self.dest,r,out)
        self.assertEqual(out.read_bytes(),self.source.read_bytes())
        self.assertEqual(self.source.stat().st_mtime_ns,stat.st_mtime_ns)
        self.assertFalse(r['scientific_adoption']);self.assertFalse(r['source_deleted'])

    def test_same_bytes_deterministic_archive(self):
        r1=self.archive();r2=s.archive_file(self.source,self.root/'other.zlib',self.raw)
        self.assertEqual(r1['stored'],r2['stored'])

    def test_existing_destination_and_partial_rejected(self):
        self.archive()
        with self.assertRaisesRegex(ValueError,'ALREADY_USED'):self.archive()
        other=self.root/'other.zlib';other.with_name(other.name+'.partial').write_bytes(b'keep')
        with self.assertRaises(ValueError):s.archive_file(self.source,other,self.raw)
        self.assertEqual(s.digest_file(self.source),self.raw)

    def test_changed_source_size_rejected(self):
        bad=dict(self.raw,bytes=self.raw['bytes']+1)
        with self.assertRaisesRegex(ValueError,'ORIGINAL_SIZE'):s.archive_file(self.source,self.dest,bad)

    def test_wrong_original_sha_never_published(self):
        with self.assertRaisesRegex(ValueError,'ORIGINAL_SHA'):
            s.archive_file(self.source,self.dest,dict(self.raw,sha256='0'*64))
        self.assertFalse(self.dest.exists());self.assertTrue(self.source.exists())

    def test_archive_corruption_rejected(self):
        r=self.archive();b=bytearray(self.dest.read_bytes());b[10]^=1;self.dest.write_bytes(b)
        with self.assertRaisesRegex(ValueError,'STORED_BYTES'):s.verify_archive(self.dest,r)

    def test_truncation_even_with_updated_stored_sha_rejected(self):
        r=self.archive();self.dest.write_bytes(self.dest.read_bytes()[:-2]);r['stored']=s.digest_file(self.dest)
        with self.assertRaisesRegex(ValueError,'TRUNCATED'):s.verify_archive(self.dest,r)

    def test_trailing_bytes_or_concatenated_stream_rejected(self):
        r=self.archive();self.dest.write_bytes(self.dest.read_bytes()+b'extra');r['stored']=s.digest_file(self.dest)
        with self.assertRaisesRegex(ValueError,'TRAILING'):s.verify_archive(self.dest,r)

    def test_decompression_bound(self):
        r=self.archive();r['original']['bytes']=100
        with self.assertRaisesRegex(ValueError,'SIZE_EXCEEDED'):s.verify_archive(self.dest,r)

    def test_insufficient_disk_keeps_original(self):
        with patch.object(s.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaisesRegex(ValueError,'SCRATCH_HEADROOM'):s.archive_file(self.source,self.dest,self.raw,min_free_bytes=1)
        self.assertFalse(self.dest.exists());self.assertEqual(s.digest_file(self.source),self.raw)

    def test_empty_input(self):
        self.source.write_bytes(b'');raw=s.digest_file(self.source)
        r=s.archive_file(self.source,self.dest,raw);self.assertEqual(s.verify_archive(self.dest,r),raw)

    def test_restore_cannot_overwrite(self):
        r=self.archive()
        with self.assertRaisesRegex(ValueError,'RESTORE_DESTINATION'):s.restore_new(self.dest,r,self.source)

    def test_format_or_lossy_rejected(self):
        r=self.archive();r['lossy']=True
        with self.assertRaisesRegex(ValueError,'LOSSLESS'):s.verify_archive(self.dest,r)

    def test_forward_only_serialization_matches_original_bytes(self):
        payload=self.source.read_bytes()
        def writer(f):
            self.assertFalse(f.seekable());self.assertEqual(f.tell(),0)
            f.write(payload[:100]);f.write(payload[100:]);self.assertEqual(f.tell(),len(payload))
        r=s.archive_writer(self.dest,writer)
        self.assertEqual(r['original'],self.raw);self.assertFalse(r['full_raw_disk_temporary'])

    def test_seeking_serializer_rejected_without_published_archive(self):
        with self.assertRaisesRegex(ValueError,'SEEK_REQUIRES'):
            s.archive_writer(self.dest,lambda f:f.seek(0))
        self.assertFalse(self.dest.exists());self.assertEqual(s.digest_file(self.source),self.raw)

    def test_serializer_error_retains_partial_not_success(self):
        def writer(f):f.write(b'partial');raise RuntimeError('TOY_WRITER_ERROR')
        with self.assertRaises(RuntimeError):s.archive_writer(self.dest,writer)
        self.assertFalse(self.dest.exists());self.assertTrue(self.dest.with_name(self.dest.name+'.partial').exists())

    def test_peak_accounts_for_retained_final_and_separate_scratch(self):
        systems={'H4':dict(max_measured_compressed_pair_bytes=100,raw_pair_proxy_bytes=1000),
                 'H6':dict(max_measured_compressed_pair_bytes=200,raw_pair_proxy_bytes=2000)}
        p=s.planning_case(systems,n_each=33,raw_buffers={'H4':2,'H6':3},compressed_factor=2,metadata_reserve_per_trajectory=10)
        self.assertEqual(sum(p['final_growth_bytes'].values()),245520)
        self.assertEqual(p['parallel_peak_growth_bytes'],255520)
        self.assertEqual(p['sequential_peak_growth_bytes'],253520)
        self.assertIsNone(p['PR_or_rare_output_upper_bound'])


@unittest.skipUnless(os.environ.get('M5_STORAGE_QPY_TEST'),'explicit existing Qiskit runtime required')
class QpySerializationTests(unittest.TestCase):
    def test_artificial_QPY_stream_is_byte_identical(self):
        from qiskit import QuantumCircuit,qpy
        circuit=QuantumCircuit(3,name='ARTIFICIAL_STORAGE_ONLY')
        circuit.h(0);circuit.cx(0,1);circuit.rz(.123,2);circuit.measure_all()
        raw=io.BytesIO();qpy.dump(circuit,raw);payload=raw.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'toy.zlib';r=s.archive_writer(dest,lambda f:qpy.dump(circuit,f))
            expected=dict(bytes=len(payload),sha256=s.hashlib.sha256(payload).hexdigest())
            self.assertEqual(r['original'],expected)
            out=Path(tmp)/'original.qpy';s.restore_new(dest,r,out);self.assertEqual(out.read_bytes(),payload)
        # No transpile, simulation, sampling, or saved molecular circuit load.

if __name__=='__main__':unittest.main()
