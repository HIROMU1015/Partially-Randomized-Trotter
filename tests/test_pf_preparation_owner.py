"""Original producer metadata is checked before a new owned clone can launch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class OriginalPFOwner(unittest.TestCase):
    def check(self,mode):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)/'original';(root/'scripts').mkdir(parents=True)
            (root/'scripts/prepare_pf_candidate.py').write_text(
                "from pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\nSTUDY='study'\n"
                "def source_gate():return {'commit':'FIXED_ARTIFICIAL'}\n"
                "def selected_configuration(item):return item['configuration_body']\n"
                "def preflight(plan,out):(out/'PROBED.json').write_text('{}')\n")
            package=root/'src/trottertracks';package.mkdir(parents=True)
            (package/'__init__.py').write_text('')
            (package/'pf_evidence_preparation.py').write_text('def make_plan(root,cfg,source):return {"ARTIFICIAL":True}\n')
            folder=root/'study/selected/C';folder.mkdir(parents=True)
            original=None
            if mode=='partial':original=folder/'signal_run01';original.mkdir();original=original/'STATUS.json';original.write_text('{"status":"COMPLETE_THIRTEEN_ROUNDS"}')
            if mode=='stop':original=folder/'STOP.json';original.write_text('{"status":"UNKNOWN"}')
            entry={'signal_packet':{'commit':'OLD','path':'signal'},'prices':{'commit':'OLD','path':'prices'},'q3_inputs':{'commit':'OLD','path':'Q3'}}
            if mode=='complete':(folder/'COMPLETION.json').write_text(json.dumps({'configuration_id':'C','registry_entry':entry}))
            before=original.read_bytes()if original else None
            request=Path(temp)/'REQUEST.json';output=Path(temp)/'CHECK.json'
            request.write_text(json.dumps(dict(selected=[dict(configuration_body=dict(id='C'))],registry_file=str(root/'REGISTRY.json'),output_root=str(root/'other'))))
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_pf_preparation_owner.py'),
                '--provider-root',str(root),'--request',str(request),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            row=json.loads(output.read_text())['rows'][0]
            if original:self.assertEqual(before,original.read_bytes())
            return row

    def test_original_partial_or_unknown_stop_holds_without_producer(self):
        self.assertEqual(self.check('partial')['status'],'PRIOR_OWNER_HOLD')
        self.assertEqual(self.check('stop')['status'],'PRIOR_OWNER_HOLD')

    def test_original_completed_receipt_reused_and_clear_scope_preflighted(self):
        self.assertEqual(self.check('complete')['status'],'REUSE_COMPLETE_ORIGINAL_OWNER')
        self.assertEqual(self.check('clear')['status'],'ORIGINAL_OWNER_PREFLIGHT_PASS')


if __name__=='__main__':unittest.main()
