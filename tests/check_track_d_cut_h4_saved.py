"""Synthetic archive identity guards, without molecular input or computation."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from trottertracks.track_d_cut_h4_saved import checked_payload,checked_record,sha


class SavedTests(unittest.TestCase):
    def archive(self,body=b"invented event metadata"):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/"toy.tar.gz"
        with tarfile.open(path,"w:gz") as tar:
            info=tarfile.TarInfo("events.json");info.size=len(body);tar.addfile(info,io.BytesIO(body))
        return path,dict(bytes=path.stat().st_size,sha256=sha(path.read_bytes()),members=[dict(path="events.json",bytes=len(body),sha256=sha(body))])

    def test_lossless_member(self):
        p,m=self.archive();self.assertEqual(checked_payload(p,m),{"events.json":b"invented event metadata"})

    def test_archive_identity_stop(self):
        p,m=self.archive();m["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"ARCHIVE_IDENTITY"):checked_payload(p,m)

    def test_member_identity_stop(self):
        p,m=self.archive();m["members"][0]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"MEMBER_IDENTITY"):checked_payload(p,m)

    def test_missing_member_stop(self):
        p,m=self.archive();m["members"][0]["path"]="different.json"
        with self.assertRaisesRegex(ValueError,"MEMBER_SET"):checked_payload(p,m)

    def test_changed_event_bytes_stop_before_decode(self):
        with self.assertRaisesRegex(ValueError,"EVENT_BYTES"):
            checked_record(b"different",dict(record_sha256="0"*64),"s","e","p","l")
