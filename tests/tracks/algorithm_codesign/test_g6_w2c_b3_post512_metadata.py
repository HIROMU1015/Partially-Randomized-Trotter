"""Artificial metadata/closed guard tests. Saved approval is data, never a permit."""
from pathlib import Path
import json,unittest,copy,os,tempfile
from types import SimpleNamespace
from unittest.mock import patch
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import guard as G,metadata as M,contract as K,execution as E,gate,run_control as A,pilot_readiness as P
ROOT=Path(__file__).resolve().parents[3]
PRIOR=ROOT/'docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020'


class Metadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=K.candidate(ROOT);cls.live=json.loads((PRIOR/'approved_manifest.json').read_text())
        cls.fake=K.manifest(cls.c,'ARTIFICIAL_ONLY/RELEASE/METADATA_UNIT','/tmp/b3-release-test-metadata-unit/output')
    def report(self,m,n=0,fail=False):
        return M.stamp_report(dict(states=dict(COMPLETED=n),pairs=[dict(failure=dict(trial_failure_provenance=dict(kind='TYPED_FAILURE')))] if fail else [],artificial_only=True),m)
    def test_artificial_mode(self):
        v=M.context(self.fake);self.assertTrue(v['artificial_only']);self.assertEqual(v['approval_mode'],'MOCK_APPROVAL');self.assertFalse(v['user_approved_live_execution'])
    def test_saved_user_live_provenance_pure_formatter(self):
        v=M.context(self.live);self.assertFalse(v['artificial_only']);self.assertTrue(v['user_approved_live_execution']);self.assertFalse(v['external_authority_used']);self.assertFalse(v['description_is_execution_permission'])
    def test_coordinator_overrides_inherited_default(self):
        obj=object.__new__(E.Coordinator);obj.manifest=self.live
        with patch.object(E.PreviousCoordinator,'report',return_value=dict(artificial_only=True,states=dict(COMPLETED=512),pairs=[])):
            self.assertFalse(obj.report()['artificial_only'])
    def test_artificial_observation_not_live(self):self.assertEqual(self.report(self.fake,512)['live_API_observation']['status'],'ARTIFICIAL_ONLY')
    def test_complete_finite_not_general(self):
        v=self.report(self.live,512);self.assertEqual(v['live_API_observation']['status'],'FIXED_MANIFEST_TRIALS_COMPLETED');self.assertEqual(v['live_API_compatibility'],'NOT_ESTABLISHED');self.assertFalse(v['science_correctness_certified'])
    def test_partial_observation(self):self.assertEqual(self.report(self.live,7)['live_API_observation']['status'],'PARTIAL_TRIAL_PREFIX')
    def test_no_observation(self):self.assertEqual(self.report(self.live)['live_API_observation']['status'],'NO_COMPLETED_TRIALS')
    def test_failure_is_not_general_guarantee(self):
        v=self.report(self.live,2,True)['live_API_observation'];self.assertEqual(v['real_F01_failure_path'],'RECORDED_TRIAL_FAILURE');self.assertEqual(v['general_failure_handling_guarantee'],'NOT_ESTABLISHED')
    def test_artificial_failure_not_real_failure(self):self.assertEqual(self.report(self.fake,2,True)['live_API_observation']['real_F01_failure_path'],'NOT_OBSERVED')
    def test_no_real_failure_observed(self):self.assertEqual(self.report(self.live,512)['live_API_observation']['real_F01_failure_path'],'NOT_OBSERVED')
    def test_wrong_live_boolean_rejected(self):
        with self.assertRaises(Exception):M.context(self.fake,live=True)
    def test_false_artificial_boolean_rejected(self):
        with self.assertRaises(Exception):M.context(self.live,live=False)
    def test_mock_cannot_be_live_approval(self):
        m=copy.deepcopy(self.fake);m.update(SCIENCE_RUN_AUTHORIZED=True,artificial_only=False)
        with self.assertRaises(Exception):M.context(m)
    def test_missing_live_approval(self):
        m=copy.deepcopy(self.live);m['approval_record']=None
        with self.assertRaises(Exception):M.context(m)
    def test_user_text_target_binding(self):
        m=copy.deepcopy(self.live);m['approval_record']['approval_text_sha256']='0'*64
        with self.assertRaises(Exception):M.context(m)
    def test_metadata_value_is_not_permit(self):
        with self.assertRaises(Exception):gate.require_live(M.context(self.live),self.c,self.live)
    def test_unapproved_live_entry_rejected_without_store(self):
        m=K.run_manifest_candidate(self.c)
        with self.assertRaises(Exception):gate.dispatch(self.c,m,'/tmp/b3-post512-must-not-exist')
    def test_old_approval_does_not_authorize_new_source(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import activation
        with self.assertRaises(Exception):activation.admit(self.c,self.live,self.live['state_directory'])
    def test_metadata_semantic_mutations_rejected(self):
        for m in (self.fake,self.live):
            for field,value in [('artificial_only',not m['artificial_only']),('execution_metadata',{}),('science_correctness_certified',True),('live_API_compatibility','PASS'),('live_API_observation',{})]:
                report=self.report(m,512);report[field]=value
                with self.subTest(mode=m['artificial_only'],field=field):
                    with self.assertRaises(Exception):M.validate_bundle(report,m,{})
    def test_receipt_caller_mode_mutations_rejected(self):
        for m in (self.fake,self.live):
            good=dict(execution_metadata=M.context(m),artificial_only=m['artificial_only'],production_permission=False)
            self.assertTrue(M.validate_bundle(self.report(m,512),m,dict(delivery=good,caller=good,receipt=good)))
            for field,value in [('artificial_only',not m['artificial_only']),('execution_metadata',{}),('production_permission',True)]:
                bad=copy.deepcopy(good);bad[field]=value
                with self.subTest(field=field):
                    with self.assertRaises(Exception):M.validate_bundle(self.report(m,512),m,dict(caller=bad))
    def test_store_spent_stop_unknown(self):
        with tempfile.TemporaryDirectory(prefix='b3-release-test-post512-') as tmp:
            root=Path(tmp);m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/'+root.name,root/'output');store=A.SimpleStore(root)
            v=store.snapshot(m);self.assertFalse(v['claim_spent']);self.assertEqual(v['run_status'],'NOT_STARTED')
            store.claim(m,A.mock_approval(m));self.assertTrue(store.snapshot(m)['claim_spent'])
            store.stop(m,unknown=True);v=store.snapshot(m);self.assertEqual(v['run_status'],'UNKNOWN');self.assertFalse(v['external_authority_used']);self.assertFalse(v['real_authority'])
            with self.assertRaises(Exception):store.enter(m)
    def test_snapshot_manifest_mismatch_is_diagnostic(self):
        with tempfile.TemporaryDirectory(prefix='b3-release-test-post512-') as tmp:
            root=Path(tmp);m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/'+root.name,root/'output');store=A.SimpleStore(root)
            store.claim(m,A.mock_approval(m));bad=copy.deepcopy(m);bad['storage_destination']=str(root/'changed')
            v=store.snapshot(bad);self.assertFalse(v['execution_metadata_verified']);self.assertTrue(v['claim_spent'])
            with self.assertRaises(Exception):M.validate_bundle(self.report(bad),bad,dict(store=v))
    def test_core_never_loaded(self):self.assertFalse(G.observation()['core_loaded']);self.assertFalse(G.ATTEMPTS)


class Pilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.value=P.candidate(ROOT);cls.saved_live=json.loads((PRIOR/'approved_manifest.json').read_text())
    def test_closed_candidate(self):self.assertTrue(P.validate(self.value,ROOT))
    def test_exact_schedule(self):
        tasks=self.value['proposal']['tasks'];self.assertEqual(len(tasks),48);self.assertEqual(sum(t['scheduled_sampling'] for t in tasks),16896)
        self.assertEqual(sum(t['scheduled_sampling'] for t in tasks if t['phase']=='correctness'),512)
        for mode in ['S','P1']:self.assertEqual(sum(t['scheduled_sampling'] for t in tasks if t['condition']==mode),8448)
        self.assertEqual(tasks[-1]['offset']+tasks[-1]['scheduled_sampling'],16896)
    def test_old512_not_deducted(self):self.assertEqual(self.value['prior_512_role'],'REFERENCE_ONLY_NOT_REUSED_OR_DEDUCTED')
    def test_preparation_activation_closed(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import activation
        self.assertIsNone(activation.APPROVED_ACTIVATION_ID)
    def test_no_scope_authorization_or_release(self):
        for name in ['global_scope','global_claim','run_id','science_release','science_authorization','approval_record','activation_record','future_pilot_release_SOURCE']:
            self.assertIsNone(self.value[name])
    def test_closed_mutations(self):
        for key,v in [('run_ready',True),('contract_grants_execution',True),('science_release','1'*40),('SCIENCE_RUN_AUTHORIZED',True),('global_claim','FAKE'),('resume',True),('retry',1)]:
            c=copy.deepcopy(self.value);c[key]=v
            with self.subTest(field=key):
                with self.assertRaises(Exception):P.validate(c,ROOT)
    def test_entry_always_denied(self):
        for kwargs in [{},{'approved':True,'run_ready':True},{'candidate':self.value,'permit':M.context(self.saved_live)}]:
            with self.assertRaises(G.Closed):P.production_entry(**kwargs)
    def test_guard_prevents_core_import(self):
        self.assertFalse(G.observation()['core_loaded']);self.assertFalse(G.ATTEMPTS)
