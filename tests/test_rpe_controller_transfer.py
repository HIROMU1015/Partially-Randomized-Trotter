"""Controller transfer preserves opaque saved jobs, STOPs and queue attempts."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transfer_rpe_controller_state',ROOT/'scripts/transfer_rpe_controller_state.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)


class TransferTests(unittest.TestCase):
    def test_snapshot_retains_unknown_attempt_budget_stop_and_result_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);service=base/'service';state=base/'state';prior=base/'prior'
            for p in [service,state,prior]:p.mkdir()
            T.put(prior/'STOP.json',{'reason':'EXISTING_SCIENTIFIC_STOP'})
            T.put(prior/'jobs/PR_DF/job/REQUEST.json',{'request_id':'job','fixed_source':'FIXED'})
            T.put(prior/'jobs/PR_DF/job/OUTCOME.json',{'status':'UNKNOWN','MSE_upper':None})
            T.put(prior/'jobs/PR_DF/job/raw/LARGE_RAW.json',{'not_needed_for_index':True})
            T.put(service/'ROUND.json',{'max_new_q3_per_method':{'PR_DF':3},'executor':{'prior_run_dirs':[str(prior)]}})
            T.put(service/'SERVICE.json',{'state_directory':str(state),'round':str(service/'ROUND.json'),'code_root':str(base)})
            T.put(service/'CURRENT_SERVICE.json',{'service_config':str(service/'SERVICE.json')})
            db=sqlite3.connect(state/'QUEUE.sqlite3')
            db.execute('CREATE TABLE jobs(job_id TEXT,attempt_id TEXT,state TEXT)')
            db.execute("INSERT INTO jobs VALUES('job','original-attempt','UNKNOWN')");db.commit();db.close()
            with patch.object(T,'active_controllers',return_value=[]):T.export(service/'CURRENT_SERVICE.json',base/'snapshot')
            self.assertEqual(T.verify(base/'snapshot')['status'],'PASS')
            manifest=json.loads((base/'snapshot/MANIFEST.json').read_text())
            copied={r['original']:base/'snapshot'/r['path']for r in manifest['files']}
            for p in [prior/'STOP.json',prior/'jobs/PR_DF/job/OUTCOME.json',service/'ROUND.json']:
                self.assertEqual(p.read_bytes(),copied[str(p)].read_bytes())
            new=sqlite3.connect(copied[str(state/'QUEUE.sqlite3')])
            self.assertEqual(new.execute('SELECT * FROM jobs').fetchall(),[('job','original-attempt','UNKNOWN')]);new.close()
            copied[str(prior/'STOP.json')].write_text('{}')
            with self.assertRaisesRegex(ValueError,'Snapshot hash differs'):T.verify(base/'snapshot')


if __name__=='__main__':unittest.main()
