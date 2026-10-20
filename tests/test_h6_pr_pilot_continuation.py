import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_pr_pilot_continuation_contract as c
import h46_pr_design_pilot as d

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.full=d.tasks_for('H6_DESIGN_PILOT',{r['id']:33 for r in d.groups() if r['system']=='H6'})
        self.ids=[t['id'] for t in self.full];self.done=self.ids[:74];self.todo=self.ids[74:]
    def test_partition(self):
        self.assertEqual(c.partition(self.full,self.done,self.todo),self.full[74:])
    def test_partition_rejects_any_scope_change(self):
        variants=[(self.done[:-1],self.todo),(self.done,self.todo[:-1]),
          (self.done,self.todo[:-1]+[self.done[0]]),(list(reversed(self.done)),self.todo),
          (self.done,list(reversed(self.todo)))]
        for a,b in variants:
            with self.assertRaises(ValueError):c.partition(self.full,a,b)
    def test_invalid_before_input(self):
        with patch.object(c,'read',side_effect=AssertionError('input read')):
            for cfg in ({},{'authorized':True,'grant':{'valid':False}}):
                with self.assertRaisesRegex(ValueError,'EXPLICIT_PILOT'):c.authorize(cfg)
    def test_registry_rederives_original_source_not_new_source(self):
        cfg=dict(SOURCE='new',campaign_id='old_campaign');old=dict(campaign_id='old_campaign')
        scoped=c.Scope(cfg);sentinel={'original_seed':'not_new'}
        with patch.object(c,'acceptance',return_value=({},old,self.full,self.full[74:])),\
             patch.object(d,'registry',return_value=sentinel) as generate:
            self.assertEqual(scoped.registry(self.full[74:],'new','old_campaign',d.COHORT,[]),sentinel)
            generate.assert_called_once_with(self.full,c.OLD_SOURCE,'old_campaign',d.COHORT,[])
    def test_registry_rejects_new_namespace(self):
        cfg=dict(SOURCE='new');old=dict(campaign_id='old')
        with patch.object(c,'acceptance',return_value=({},old,self.full,self.full[74:])):
            with self.assertRaises(ValueError):c.Scope(cfg).registry(self.full[74:],'new','redraw',d.COHORT,[])
    def test_only_h6(self):
        with self.assertRaisesRegex(ValueError,'ONLY_APPROVED_H6'):
            c.acceptance(dict(continuation_mode=c.MODE,stage='H4_DESIGN_PILOT'))
    def test_cli_disabled_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';p.write_text(json.dumps(dict(authorized=False,grant=dict(valid=False))))
            runner=c.ROOT/'scripts/run_h6_pr_pilot_continuation.py'
            for mode in ('--execute','--supervise'):
                r=subprocess.run([sys.executable,'-B',str(runner),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_PILOT_EXECUTION_APPROVAL_REQUIRED',r.stdout)
            self.assertEqual([v.name for v in Path(tmp).iterdir()],['draft.json'])
    def test_reuses_original_full_checks(self):
        cfg={'authorized':True,'grant':{'valid':True}}
        with patch.object(c,'verify') as verify:
            # The original authorizer still rejects missing/wrong pilot stage.
            with self.assertRaises(KeyError):c.authorize(cfg)
            verify.assert_not_called()

if __name__=='__main__':unittest.main()
