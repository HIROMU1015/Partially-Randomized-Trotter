"""Artificial outputs never enter the ordinary real-result publication tree."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from publish_romutail_search import copy_metadata, publish

class PublicationTests(unittest.TestCase):
    def test_compact_copy_preserves_tokens_and_keeps_previous_file_on_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'source', Path(tmp)/'dest'
            token = '9'*80
            raw = ('{\n' + ' '*200 + '"ratio": ' + token + ',\n' +
                   ' '*200 + r'"text": "  x\n y  "' + '\n}\n').encode()
            source.write_bytes(raw)
            copy_metadata(source, dest, limit=170)
            compact = dest.read_bytes()
            self.assertEqual(json.loads(raw), json.loads(compact))
            self.assertIn(token.encode(), compact)
            with self.assertRaises(ValueError):
                copy_metadata(source, dest, limit=10)
            self.assertEqual(dest.read_bytes(), compact)
            self.assertEqual(source.read_bytes(), raw)

    def test_artificial_results_skip_before_any_git_or_publication_write(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            (run/'PROBLEM.json').write_text(json.dumps({'problem':{'system':'ARTIFICIAL_FIXTURE'}}))
            result=publish(root/'DOES_NOT_EXIST',run,branch='synthetic')
            self.assertEqual(result['status'],'ARTIFICIAL_SKIPPED')
            self.assertFalse((root/'DOES_NOT_EXIST').exists())

if __name__=='__main__':unittest.main()
