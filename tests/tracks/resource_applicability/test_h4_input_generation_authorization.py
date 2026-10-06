"""Zero-science gate tests. No runner, worker, input, transpile or seed calls."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

PREPARATION = None


class AuthorizationPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = PREPARATION
        cls.gates, cls.identity, cls.resources = cls.helper.production_modules()
        cls.bundle = cls.helper.ROOT/cls.helper.BUNDLE
        cls.saved = tuple(json.loads((cls.bundle/name).read_bytes()) for name in (
            'input_generation_plan_v1.json','authorization_draft_v1.json','stage_review_v1.json'))

    def simulated(self):
        plan, auth, review = copy.deepcopy(self.saved)
        # [0] is a fictional CPU permission used only by a pure in-memory gate.
        # It is never a grant for this server and never saved to a review file.
        auth['allowed_cpus'] = [0]
        review['approved'] = True
        self.bind(plan,auth,review)
        return plan,auth,review

    def bind(self, plan, auth, review):
        auth['plan_fingerprint'] = self.identity.fingerprint('h4-execution-plan-v1',plan)
        review['plan_fingerprint'] = auth['plan_fingerprint']
        review['authorization_digest'] = self.identity.fingerprint('h4-authorization-v1',auth)

    def permit(self, documents):
        return self.gates.authorize('input_generation',*documents,explicit_launch=True)

    def test_exact_saved_production_schema_and_source_bound_scope(self):
        plan,auth,review = self.saved
        self.gates.structural_gate(plan,auth,review)
        self.assertEqual(plan['requested_workers'],6)
        self.assertEqual(plan['binding'],'SOURCE_BOUND')
        self.assertIsNone(plan['inputs']);self.assertIsNone(plan['generation_freeze_digest'])
        self.assertEqual(plan['source_root'],str(self.helper.SCIENCE_ROOT))
        self.assertEqual(plan['source_commit'],self.helper.SOURCE)
        self.assertEqual(len(plan['source_hashes']),19)
        self.assertEqual(len(plan['templates']),218)
        self.assertEqual(plan['distances'],list(self.gates.DISTANCES))
        self.assertEqual(auth['permission'],'input_generation')
        self.assertIs(review['approved'],False)

    def test_saved_false_review_rejected_before_checkout_or_science(self):
        with patch.object(self.gates,'checkout_gate',side_effect=AssertionError('checkout must not run')) as checkout:
            with self.assertRaisesRegex(self.identity.Stop,'review/authorization'):
                self.permit(copy.deepcopy(self.saved))
            checkout.assert_not_called()

    def test_saved_cpu_permission_not_inferred_from_observation(self):
        resource=json.loads((self.bundle/'resource_cpu_review_v1.json').read_bytes())
        self.assertEqual(self.saved[1]['allowed_cpus'],resource['allowed_cpus'])
        self.assertIs(resource['observation_is_permission'],False)
        if not resource['allowed_cpus']:
            p,a,r=copy.deepcopy(self.saved);r['approved']=True
            with self.assertRaisesRegex(self.identity.Stop,'explicit CPU permission'):
                self.permit((p,a,r))
            self.assertIsNone(resource['explicit_cpu_permission_evidence'])
        self.assertIs(resource['execution_ready'],False)

    def test_positive_simulated_authorize_and_metadata_only_checkout(self):
        documents=self.simulated();permit=self.permit(documents)
        contract,options=self.gates.checkout_gate(permit)
        self.assertEqual(permit.stage,'input_generation')
        self.assertEqual(permit.source_root,str(self.helper.SCIENCE_ROOT))
        self.assertEqual(contract['templates'],documents[0]['templates'])
        self.assertEqual(options['num_processes'],1)
        self.assertIs(self.saved[2]['approved'],False)
        self.assertIs(json.loads((self.bundle/'stage_review_v1.json').read_bytes())['approved'],False)

    def test_identity_lineage_all_blobs_environment_and_old_evidence(self):
        contract,audit,report=self.helper.verify_identity()
        self.assertEqual(report['science_source_and_parent_paths_verified'],19)
        self.assertEqual(report['dependency_count'],45)
        self.assertEqual(report['installed_source_paths_verified'],11)
        self.assertEqual(report['contract_manifest_entries_verified'],37)
        self.assertEqual(report['old_sources_verified'],247)
        self.assertEqual(report['saved_JSON_verified'],6)
        self.assertFalse(report['science_source_modified'])
        self.assertEqual(audit['source_commit'],self.helper.SOURCE)

    def test_plan_authorization_review_fingerprint_binding(self):
        p,a,r=self.saved
        self.assertEqual(a['plan_fingerprint'],self.identity.fingerprint('h4-execution-plan-v1',p))
        self.assertEqual(r['plan_fingerprint'],a['plan_fingerprint'])
        self.assertEqual(r['authorization_digest'],self.identity.fingerprint('h4-authorization-v1',a))
        self.assertNotEqual(self.identity.fingerprint('h4-review-v1',r),
                            self.identity.fingerprint('h4-review-v1',{**r,'approved':True}))

    def test_source_closure_complete_from_external_audit(self):
        raw=(self.helper.SCIENCE_ROOT/self.gates.SOURCE_AUDIT).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),self.helper.SOURCE_AUDIT_SHA)
        audit=json.loads(raw)
        self.assertEqual(self.saved[0]['source_hashes'],{**audit['new_source_hashes'],**audit['namespace_parent_hashes']})
        self.assertEqual(Path(self.gates.__file__).absolute(),self.helper.SCIENCE_ROOT/'src/trottertracks/resource_applicability/h4_geometry/gates.py')

    def test_no_explicit_launch_rejected_in_pure_gate(self):
        with self.assertRaisesRegex(self.identity.Stop,'explicit launch required'):
            self.gates.authorize('input_generation',*self.simulated(),explicit_launch=False)

    def test_generation_authorization_not_reused_for_signal_stage(self):
        with self.assertRaises(self.identity.Stop):
            self.gates.authorize('signal_compile',*self.simulated(),explicit_launch=True)

    def test_resource_conditions_synthetic_admission_only(self):
        r=self.resources
        self.assertEqual(r.admission(72*2**30,10,6,list(range(6)),list(range(6)),now=10),6)
        self.assertEqual(r.admission(64*2**30,10,6,list(range(6)),list(range(6)),now=10),5)
        self.assertEqual(r.admission(72*2**30,10,6,[0,1],[0,1],now=10),2)
        for available,at,explicit,process in ((31*2**30,10,[0],[0]),(72*2**30,0,[0],[0]),
                                             (72*2**30,10,[],[0]),(72*2**30,10,[0],[1])):
            with self.assertRaises(self.identity.Stop):r.admission(available,at,6,explicit,process,now=10)

    def test_launch_context_subset_condition_as_written_in_frozen_source(self):
        source=(self.helper.SCIENCE_ROOT/'src/trottertracks/resource_applicability/h4_geometry/execution.py').read_text()
        self.assertIn("require(observation['process_cpus'] <= set(authorization['allowed_cpus']), 'process can use unpermitted CPU')",source)
        # Validate the pure predicate; never construct OwnedRun or change affinity.
        self.identity.require({0,1} <= {0,1,2},'process can use unpermitted CPU')
        with self.assertRaises(self.identity.Stop):
            self.identity.require({0,1,2} <= {0,1},'process can use unpermitted CPU')

    def test_installed_source_tamper_metadata_read_rejected(self):
        audit=json.loads((self.helper.SCIENCE_ROOT/self.gates.SOURCE_AUDIT).read_bytes())
        chosen=next(iter(audit['installed_source_hashes']))
        original=Path.read_bytes
        def read(path):
            return b'ARTIFICIAL_INSTALLED_SOURCE_TAMPER' if str(path)==chosen else original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaises(self.identity.Stop):self.gates.checkout_gate(self.permit(self.simulated()))

    def test_actual_checkout_source_bytes_tamper_mock_rejected(self):
        chosen=self.helper.SCIENCE_ROOT/'src/trottertracks/resource_applicability/h4_geometry/parallel.py'
        original=Path.read_bytes
        def read(path):
            return b'ARTIFICIAL_SOURCE_TAMPER' if path==chosen else original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaisesRegex(self.identity.Stop,'actual checkout source mismatch'):
                self.gates.checkout_gate(self.permit(self.simulated()))

    def test_source_audit_bytes_tamper_mock_rejected(self):
        chosen=self.helper.SCIENCE_ROOT/self.gates.SOURCE_AUDIT;original=Path.read_bytes
        def read(path):
            return b'ARTIFICIAL_AUDIT_TAMPER' if path==chosen else original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaisesRegex(self.identity.Stop,'independent source audit binding'):
                self.gates.checkout_gate(self.permit(self.simulated()))


def mutation_rejected(document_index,key,value,*,checkout=False,rebind=True):
    def test(self):
        documents=self.simulated();documents[document_index][key]=copy.deepcopy(value)
        if rebind:self.bind(*documents)
        with self.assertRaises(self.identity.Stop):
            permit=self.permit(documents)
            if checkout:self.gates.checkout_gate(permit)
    return test


MUTATIONS = [
    ('source_sha_syntax',0,'source_commit','x',False),
    ('source_sha_identity',0,'source_commit','0'*40,True),
    ('source_closure_missing',0,'source_hashes',{},False),
    ('source_closure_incomplete',0,'source_hashes',{'src/fake.py':'0'*64},True),
    ('audit_hash_syntax',0,'source_audit_sha256','bad',False),
    ('audit_hash_identity',0,'source_audit_sha256','0'*64,True),
    ('source_root_relative',0,'source_root','relative',False),
    ('source_root_preparation',0,'source_root','/tmp/artificial-preparation',True),
    ('artifact_anchor',0,'artifact_anchor','/tmp/artificial',False),
    ('output_root',0,'output_root','/tmp/artificial-output',False),
    ('distance_order',0,'distances',['1.60','1.40','1.10','0.90','0.80','0.70'],False),
    ('distance_missing',0,'distances',['0.70'],False),
    ('base_commit',0,'base_commit','0'*40,False),
    ('contract_fingerprint',0,'contract_plan_fingerprint','0'*64,False),
    ('input_placeholder',0,'inputs',{'0.70':'0'*64},False),
    ('freeze_placeholder',0,'generation_freeze_digest','0'*64,False),
    ('binding',0,'binding','INPUT_BOUND',False),
    ('workers_zero',0,'requested_workers',0,False),
    ('workers_excess',0,'requested_workers',13,False),
    ('workers_bool',0,'requested_workers',True,False),
    ('templates_missing',0,'templates',[],True),
    ('compiler_fingerprint',0,'compiler_fingerprint','0'*64,True),
    ('environment_fingerprint',0,'environment_fingerprint','0'*64,True),
    ('plan_stage',0,'stage','signal_compile',False),
    ('plan_run',0,'run_id','other',False),
    ('permission',1,'permission','signal_compile',False),
    ('auth_stage',1,'stage','signal_compile',False),
    ('one_shot',1,'one_shot',False,False),
    ('result_prior',1,'result_prior',False,False),
    ('cpu_empty',1,'allowed_cpus',[],False),
    ('cpu_negative',1,'allowed_cpus',[-1],False),
    ('cpu_duplicate',1,'allowed_cpus',[0,0],False),
    ('cpu_bool',1,'allowed_cpus',[True],False),
    ('cpu_string',1,'allowed_cpus',['0'],False),
    ('review_stage',2,'stage','signal_compile',False),
    ('review_approved_false',2,'approved',False,False),
    ('review_approved_string',2,'approved','true',False),
]
for name,index,key,value,checkout in MUTATIONS:
    setattr(AuthorizationPreparationTests,'test_reject_'+name,mutation_rejected(index,key,value,checkout=checkout))
for name,index,key in [('auth_plan_fp',1,'plan_fingerprint'),('review_plan_fp',2,'plan_fingerprint'),
                       ('review_auth_digest',2,'authorization_digest')]:
    setattr(AuthorizationPreparationTests,'test_reject_'+name,mutation_rejected(index,key,'0'*64,rebind=False))


def extra_field_rejected(index):
    def test(self):
        documents=self.simulated();documents[index]['ARTIFICIAL_EXTRA_FIELD']=True
        self.bind(*documents)
        with self.assertRaises(self.identity.Stop):self.permit(documents)
    return test
for index,name in enumerate(('plan','authorization','review')):
    setattr(AuthorizationPreparationTests,'test_reject_extra_'+name,extra_field_rejected(index))
