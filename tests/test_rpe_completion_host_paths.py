"""Deployment-only metadata paths never alter the scientific request body."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_candidate_completion import Completion


class HostPathTests(unittest.TestCase):
    def test_instance_paths_and_subprocess_metadata_are_scoped_to_settings(self):
        with tempfile.TemporaryDirectory()as tmp:
            completion=Completion.__new__(Completion)
            completion.settings={'controller_paths':{'NATIVE':'/LOCAL/python','D2_VIEW':'/LOCAL/fixed-view'}}
            received=[]
            completion.call=lambda command,*args:received.append(command)
            request={'configuration':{'id':'FIXED'},'policy':'MODEL'}
            completion.service('quote',request,Path(tmp))
            self.assertEqual(received[0][0],'/LOCAL/python')
            self.assertEqual(completion.path('D2_VIEW'),Path('/LOCAL/fixed-view'))
            stored=json.loads((Path(tmp)/'quote.INPUT.json').read_text())
            self.assertEqual(stored['configuration'],request['configuration'])
            self.assertNotIn('controller_paths',request)
            self.assertEqual(stored['controller_paths'],completion.settings['controller_paths'])


if __name__=='__main__':unittest.main()
