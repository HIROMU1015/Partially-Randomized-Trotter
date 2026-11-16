"""Technical process evidence only; no numerical kernel or science."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('_receipt_adapter',ROOT/'scripts/run_rpe_optimizer_q3_job.py')
Q=importlib.util.module_from_spec(s);s.loader.exec_module(Q)

class Receipts(unittest.TestCase):
    def test_df_uses_original_joined_exitcodes_and_rejects_missing_or_failed(self):
        req={'route':{'kind':'prdf_d2_common_entry'}}
        with tempfile.TemporaryDirectory()as tmp:
            good={'receipts':[{'status':'SCIENTIFIC_RESULT'}],'errors':[], 'process_exitcodes':[{'pid':123,'exitcode':0,'signal':None}]}
            self.assertTrue(Q.process_exit_evidence(req,good,Path(tmp))['all_exit_codes_zero'])
            for exits in ([],[{'pid':123,'exitcode':-9,'signal':9}],[{'pid':123,'exitcode':None,'signal':None}]):
                self.assertFalse(Q.process_exit_evidence(req,dict(good,process_exitcodes=exits),Path(tmp))['all_exit_codes_zero'])
            self.assertFalse(Q.process_exit_evidence(req,dict(good,receipts=[]),Path(tmp))['all_exit_codes_zero'])
    def test_pauli_keeps_installed_exit_log_gate(self):
        req={'route':{'kind':'pauli_b_common_entry'}}
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)
            self.assertFalse(Q.process_exit_evidence(req,{},p)['all_exit_codes_zero'])
            (p/'PROCESS_EXITS_1.jsonl').write_text('{"children":[{"exitcode":0}]}\n')
            self.assertTrue(Q.process_exit_evidence(req,{},p)['all_exit_codes_zero'])
            (p/'PROCESS_EXITS_2.jsonl').write_text('{"children":[{"exitcode":-9}]}\n')
            self.assertFalse(Q.process_exit_evidence(req,{},p)['all_exit_codes_zero'])

class CollectedOutcomes(unittest.TestCase):
    def test_unknown_is_preserved_and_collection_requires_original_and_exit_gate(self):
        import sys,json,hashlib
        sys.path.insert(0,str(ROOT/'src'))
        from trottertracks.rpe_method_optimizer import job_outcome
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);original=p/'OUTCOME.json'
            original.write_text('{"status":"UNKNOWN","reason":"OLD_SCHEMA_STOP"}')
            new=dict(status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',mode='collect',collection_source='fixed',
                     original_outcome_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
            (p/'COLLECTED_OUTCOME.json').write_text(json.dumps(new))
            self.assertEqual(job_outcome(p)[0]['status'],'UNKNOWN')
            (p/'COLLECT_SUPERVISION.json').write_text('{"process_exit_gate":{"all_exit_codes_zero":true}}')
            self.assertEqual(job_outcome(p)[0]['status'],new['status'])
            self.assertEqual(json.loads(original.read_text())['status'],'UNKNOWN')
            original.write_text('{"status":"UNKNOWN","reason":"CHANGED"}')
            self.assertEqual(job_outcome(p)[0]['status'],'UNKNOWN')

if __name__=='__main__':unittest.main(verbosity=2)
