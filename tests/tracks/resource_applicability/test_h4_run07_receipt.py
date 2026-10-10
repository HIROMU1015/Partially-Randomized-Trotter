"""Limited metadata/byte regression of the latest stopped predecessor."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from trottertracks.resource_applicability.h4_geometry import run07_receipt as proof
from trottertracks.resource_applicability.h4_geometry.identity import Stop


class Run07ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.receipt=json.loads(proof.RECEIPT.read_bytes())
        self.evidence={'newhost_run07_predecessor':dict(path=str(proof.RECEIPT),bytes=proof.RECEIPT_BYTES,sha256=proof.RECEIPT_SHA)}
    def test_received_original_metadata(self):
        self.assertEqual(len(proof.validate_metadata(self.receipt)),6)
    def test_cost_reset_or_completion_rewrite_rejected(self):
        for key in ('charged_bytes','actual_invocations_consumed_or_reserved','completed_wrappers','signal_records'):
            value=copy.deepcopy(self.receipt);value[key]=0
            with self.assertRaises(Stop):proof.validate_metadata(value)
    def test_unknown_exit_or_time_cannot_be_backfilled(self):
        for key in ('worker_exit_code','exact_driver_worker_exit_time'):
            value=copy.deepcopy(self.receipt);value[key]=0
            with self.assertRaises(Stop):proof.validate_metadata(value)
    def test_missing_duplicate_or_nonterminal_identity_rejected(self):
        for mode in ('missing','duplicate','nonterminal'):
            value=copy.deepcopy(self.receipt);rows=value['native_identity_audits'][0]['identities']
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            else:rows[0]['classification']='PRESENT'
            with self.assertRaises(Stop):proof.validate_metadata(value)
    def test_two_audits_must_match_and_be_separated(self):
        for mode in ('identity','time'):
            value=copy.deepcopy(self.receipt)
            if mode=='identity':value['native_identity_audits'][1]['identities'][0]['expected']['start']='other'
            else:value['native_identity_audits'][1]['monotonic']=value['native_identity_audits'][0]['monotonic']
            with self.assertRaises(Stop):proof.validate_metadata(value)
    def test_wrong_predecessor_or_forged_proof_reference_rejected_before_IO(self):
        with patch.object(proof,'streaming_sha') as stream:
            for key in ('path','bytes','sha256'):
                value=copy.deepcopy(self.evidence);value['newhost_run07_predecessor'][key]='wrong'
                with self.assertRaises(Stop):proof.verify_run07_stop(value)
            stream.assert_not_called()
    def test_byte_hash_mismatch_stops(self):
        with patch.object(proof,'streaming_sha',return_value={'bytes':proof.RECEIPT_BYTES,'sha256':'0'*64}):
            with self.assertRaises(Stop):proof.verify_run07_stop(self.evidence)
    def test_all_original_byte_files_and_current_native_absence(self):
        result=proof.verify_run07_stop(self.evidence)
        self.assertEqual(result['charged_bytes'],4263870068)
    def test_same_owned_identity_still_present_stops(self):
        identities=proof.validate_metadata(self.receipt)
        with patch('trottertracks.resource_applicability.h4_geometry.observer.process_sample',side_effect=lambda pid:identities[pid]):
            with self.assertRaisesRegex(Stop,'owned identity still exists'):proof.verify_run07_stop(self.evidence)
    def launch_documents(self):
        from trottertracks.resource_applicability.h4_geometry import launch_binding as bind
        from trottertracks.resource_applicability.h4_geometry.identity import fingerprint
        root=Path(__file__).absolute().parents[3]
        old=root/'artifacts/resource_applicability/track_a_h4_production_run07/2026-10-10'
        p,a,r=[json.loads((old/name).read_bytes()) for name in ('plan_authorized_v17.json','authorization_v17.json','review_v17.json')]
        for doc in (p,a,r):doc['run_id']=bind.RUN_ID
        p.update(output_root='/home/AbeHiromu/artificial/output/'+bind.RUN_ID,control_root='/home/AbeHiromu/artificial/control/'+bind.RUN_ID)
        p['caps']['worker_AS_RSS']=32*2**30
        memory=root/'artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/preparation_binding_v1.json'
        a['memory_budget_amendment']=json.loads(memory.read_bytes())['memory_budget_amendment']
        a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
        r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)
        return bind,p,a,r
    def test_fresh_run08_cap32_authorizes_without_runtime_operations(self):
        bind,p,a,r=self.launch_documents()
        self.assertEqual(bind.RUN_ID,'h4-newhost-signal-compile-20261010-run08')
        self.assertEqual(bind.authorize(p,a,r,explicit_launch=True).plan['caps']['worker_AS_RSS'],32*2**30)
    def test_consumed_run07_identity_cannot_be_reused(self):
        from trottertracks.resource_applicability.h4_geometry.identity import fingerprint
        bind,p,a,r=self.launch_documents()
        for doc in (p,a,r):doc['run_id']=proof.RUN
        a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
        r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)
        with self.assertRaisesRegex(Stop,'stage/run/source'):bind.authorize(p,a,r,explicit_launch=True)
