"""Completion refuses missing native evidence; no production input is read."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_h4_followup_results import accepted


class Collection(unittest.TestCase):
    def setUp(self):
        self.end=dict(exit_code=0,same_parent_reaped=True,wait4_complete=True,SOURCE='a'*40)
        self.term=dict(exit_code=0,SOURCE='a'*40,status='COMPLETE_MANDATORY_STOP')
        self.gate=dict(SOURCE='a'*40,status='PASS');self.result=dict(status='COMPLETE_MANDATORY_STOP')

    def test_complete(self):accepted(self.end,self.term,self.gate,self.result)

    def test_failed_or_unreaped_native_rejected(self):
        for key,value in [('exit_code',2),('same_parent_reaped',False),('wait4_complete',False)]:
            end=dict(self.end,**{key:value})
            with self.assertRaisesRegex(ValueError,'NATIVE_COMPLETION'):accepted(end,self.term,self.gate,self.result)

    def test_source_gate_or_result_not_promoted(self):
        with self.assertRaisesRegex(ValueError,'SOURCE_GATE'):accepted(self.end,self.term,dict(self.gate,SOURCE='b'*40),self.result)
        with self.assertRaisesRegex(ValueError,'SOURCE_GATE'):accepted(self.end,self.term,dict(self.gate,status='STOP'),self.result)
        with self.assertRaisesRegex(ValueError,'MANDATORY_STOP'):accepted(self.end,self.term,self.gate,dict(status='PARTIAL'))


if __name__=='__main__':unittest.main()
