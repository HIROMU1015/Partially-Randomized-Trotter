"""Artificial boundary tests; no scientific fixtures, libraries or workers."""
import gzip
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_h2_common_pf_originals as audit


class RecoveryBoundaries(unittest.TestCase):
    def check_archive(self, stored, original, raw=None):
        receipt=dict(compressed_bytes=len(stored),archive_sha256=hashlib.sha256(stored).hexdigest(),
                     original_bytes=len(original),original_sha256=hashlib.sha256(original).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'artificial.gz';path.write_bytes(stored)
            value=audit.verify_gzip(path,receipt)
            self.assertEqual(value,original if raw is None else raw)

    def test_gzip_raw_roundtrip(self):
        raw=b'artificial identity only';self.check_archive(gzip.compress(raw,mtime=0),raw)

    def test_truncated_gzip_rejected(self):
        raw=b'artificial'
        with self.assertRaises(audit.r.EvidenceError):self.check_archive(gzip.compress(raw)[:-2],raw)

    def test_concatenated_archive_rejected(self):
        raw=b'artificial'
        with self.assertRaises(audit.r.EvidenceError):self.check_archive(gzip.compress(raw)+gzip.compress(b'extra'),raw)

    def test_declared_raw_size_limits_decode(self):
        with self.assertRaises(audit.r.EvidenceError):self.check_archive(gzip.compress(b'x'*100000),b'x')

    def test_wrong_original_sha_rejected(self):
        with self.assertRaises(audit.r.EvidenceError):self.check_archive(gzip.compress(b'abc'),b'abd')

    def test_duplicate_candidate_rejected(self):
        row=dict(candidate=dict(system='H2',method='FULL',prefix=3,q=4,formula='2nd'))
        with self.assertRaises(audit.r.EvidenceError):audit.selected([row,row],'H2_FULL_3_4_2nd')

    def test_private_path_projection_rejected(self):
        with self.assertRaises(audit.r.EvidenceError):audit.safe_json(dict(locator='/home/artificial/private'))

    def test_no_scientific_imports(self):
        self.assertFalse({x.split('.')[0] for x in sys.modules}&{'qiskit','numpy','scipy','symengine','trotterlib','trottertracks'})


if __name__=='__main__':unittest.main()
