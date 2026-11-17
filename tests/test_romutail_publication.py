"""Artificial outputs never enter the ordinary real-result publication tree."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from publish_romutail_search import publish

class PublicationTests(unittest.TestCase):
    def test_artificial_results_skip_before_any_git_or_publication_write(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            (run/'PROBLEM.json').write_text(json.dumps({'problem':{'system':'ARTIFICIAL_FIXTURE'}}))
            result=publish(root/'DOES_NOT_EXIST',run,branch='synthetic')
            self.assertEqual(result['status'],'ARTIFICIAL_SKIPPED')
            self.assertFalse((root/'DOES_NOT_EXIST').exists())

if __name__=='__main__':unittest.main()
