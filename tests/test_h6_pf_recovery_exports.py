"""Artificial multi-SOURCE export checks: old failed COST is never a success."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import finalize_h6_pf_common_results as final


class RecoveryExportTests(unittest.TestCase):
    def test_inherited_signal_qualify_and_new_cost_account_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';new=root/'new';out=root/'export';out.mkdir()
            def save(p,value):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            plan=dict(SOURCE='fresh',inherited_SOURCE='original',inherited_private=str(old),
                recovery_track=True,closure=[],input_paths={})
            save(new/'plan_APPROVED.json',plan);save(new/'approval_receipt.json',dict(approved=True))
            for owner,name,source in [(old,'SIGNAL','original'),(old,'QUALIFY','original'),(new,'COST','fresh'),(new,'ACCOUNT','fresh')]:
                for filename in ('grant.json','CONSUMED.json','reserved_counts.json','completion.json'):
                    save(owner/name/filename,dict(SOURCE=source,stage=name))
                save(owner/name/'ledger/one.json',dict(SOURCE=source))
                save(owner/name/'resource_receipt.json',dict(stage=name,SOURCE=source))
            save(old/'SIGNAL/output/all_signals.json',dict(signal='saved-original-not-recomputed'))
            save(old/'COST/STOP.json',dict(status='MANDATORY_STOP'))
            save(old/'COST/output/accepted/result.json',dict(SOURCE='original',status='PASS'))
            save(new/'COST/output/new/result.json',dict(SOURCE='fresh',status='PASS'))
            snapshot={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
            final.export_controls(new,out)
            with gzip.open(out/'controls_and_ledgers.json.gz','rt') as f:controls=json.load(f)
            self.assertEqual(controls['SIGNAL']['grant.json']['SOURCE'],'original')
            self.assertEqual(controls['QUALIFY']['grant.json']['SOURCE'],'original')
            self.assertEqual(controls['COST']['grant.json']['SOURCE'],'fresh')
            self.assertEqual(controls['ACCOUNT']['grant.json']['SOURCE'],'fresh')
            with gzip.open(out/'native_results.json.gz','rt') as f:native=json.load(f)
            self.assertEqual({r['SOURCE'] for r in native},{'original','fresh'})
            self.assertFalse((old/'COST/completion.json').exists())
            self.assertFalse((new/'SIGNAL').exists())
            self.assertEqual(snapshot,{str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()})
            projection=json.loads((out/'execution_contract_projection.json').read_text())
            self.assertNotIn('inherited_private',projection)


if __name__=='__main__':unittest.main()
