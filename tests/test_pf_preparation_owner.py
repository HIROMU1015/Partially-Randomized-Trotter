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
            plan=('raise ValueError("One saved axis only: resolve per-axis native execution before compiling")'
                  if mode=='price' else 'return {"ARTIFICIAL":True,"direct_prices":[]}')
            (package/'pf_evidence_preparation.py').write_text('def make_plan(root,cfg,source):'+plan+'\n')
            folder=root/'study/selected/C';folder.mkdir(parents=True)
            original=None
            if mode=='partial':original=folder/'signal_run01';original.mkdir();original=original/'STATUS.json';original.write_text('{"status":"COMPLETE_THIRTEEN_ROUNDS"}')
            if mode=='stop':original=folder/'STOP.json';original.write_text('{"status":"UNKNOWN"}')
            entry={'signal_packet':{'commit':'OLD','path':'signal'},'prices':{'commit':'OLD','path':'prices'},'q3_inputs':{'commit':'OLD','path':'Q3'}}
            if mode=='complete':(folder/'COMPLETION.json').write_text(json.dumps({'configuration_id':'C','registry_entry':entry}))
            before=original.read_bytes()if original else None
            request=Path(temp)/'REQUEST.json';output=Path(temp)/'CHECK.json'
            item=dict(selected=[dict(configuration_body=dict(id='C'))],
                      registry_file=str(root/'REGISTRY.json'),output_root=str(root/'other'))
            options = []
            if mode == 'price_independent':
                (package/'pf_evidence_preparation.py').write_text(
                    "from types import SimpleNamespace\n"
                    "physical={'id':'circuit'}\n"
                    "def price_pool(root):return {}\n"
                    "def accepted_direct(root,prepared):return []\n"
                    "def physical_key(value):return value['id']\n"
                    "I=SimpleNamespace(price_requests=lambda root,prepared:[{'m':3,'axis':'X',"
                    "'physical_context':{'profile':{}}}])\n"
                    "def make_plan(root,cfg,source):\n"
                    " return dict(configuration=cfg,direct_prices=[],new_signal_means=1,"
                    "selected_axis_compile=True,new_compile_cells=[dict(m=3,axes=['X'],role='CALIBRATION')],"
                    "prepared={'pending':{'requests':[dict(m=3,physical_candidate=physical)]}})\n")
                (root/'scripts/run_pf_m0_prices.py').write_text(
                    "def duplicate_check(rounds,selected):\n"
                    " return {'rows':[dict(row,action='HELD',findings=[{'running_job':{'pid':123}}]) "
                    "for row in selected]}\n")
                index = Path(temp)/'SAVED_PRICE_INDEX.json'
                index.write_text(json.dumps(dict(root=str(root),locators=[])))
                options = ['--saved-price-index', str(index)]
            if mode=='prior_axes':
                item['previously_audited_price_axes']=[['old-circuit','X'],['old-circuit','Y']]
            request.write_text(json.dumps(item))
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_pf_preparation_owner.py'),
                '--provider-root',str(root),'--request',str(request),'--output',str(output),*options],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            row=json.loads(output.read_text())['rows'][0]
            if original:self.assertEqual(before,original.read_bytes())
            return row

    def test_exploratory_partial_price_locator_is_not_complete(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('owner_check', ROOT/'scripts/check_pf_preparation_owner.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        entry = dict(signal_packet={'saved': True}, q3_inputs={'saved': True},
                     prices=dict(axes=[{'m': 0, 'axis': 'X'}], coverage='PARTIAL_DIRECT_PRICE_EVIDENCE'))
        self.assertFalse(helper.reusable_entry(ROOT, entry, exploratory=True, selected_axes=True))
        entry['prices']['coverage'] = 'COMPLETE'
        self.assertTrue(helper.reusable_entry(ROOT, entry, exploratory=True, selected_axes=True))

    def test_original_partial_or_unknown_stop_holds_without_producer(self):
        self.assertEqual(self.check('partial')['status'],'PRIOR_OWNER_HOLD')
        self.assertEqual(self.check('stop')['status'],'PRIOR_OWNER_HOLD')

    def test_original_completed_receipt_reused_and_clear_scope_preflighted(self):
        self.assertEqual(self.check('complete')['status'],'REUSE_COMPLETE_ORIGINAL_OWNER')
        self.assertEqual(self.check('clear')['status'],'ORIGINAL_OWNER_PREFLIGHT_PASS')

    def test_price_incompatibility_is_recorded_before_signal_dispatch(self):
        row=self.check('price')
        self.assertEqual(row['status'],'PRICE_EVIDENCE_REQUIRED')
        self.assertTrue(row['signal_deferred_for_price'])

    def test_replan_cannot_recompile_previously_audited_axes(self):
        row=self.check('prior_axes')
        self.assertEqual(row['status'],'PRICE_EVIDENCE_REQUIRED')
        self.assertEqual(row['missing_audited_axes'],[['old-circuit','X'],['old-circuit','Y']])

    def test_pending_price_owner_still_allows_independent_signal_plan(self):
        row = self.check('price_independent')
        self.assertEqual(row['status'], 'ORIGINAL_OWNER_PREFLIGHT_PASS')
        self.assertEqual(row['clear_compile_rounds'], [])
        self.assertFalse(row['deferred_price_rounds']['3'][0]['new_calculation'])


if __name__=='__main__':unittest.main()
