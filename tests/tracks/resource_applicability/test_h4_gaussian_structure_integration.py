"""New Gaussian wiring/approval/identity metadata only; no synthetic compile."""
import ast,copy,json,unittest
from pathlib import Path
from unittest.mock import patch,sentinel
from trottertracks.resource_applicability.h4_geometry import gaussian_structure as structure,circuits,signal as science,launch_binding as bind,run09_receipt as proof
from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint,trajectory_seed,wrapper_key

ROOT=Path(__file__).absolute().parents[3]


def rebound(p,a,r):
    a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
    r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)


def documents():
    b=ROOT/'artifacts/resource_applicability/track_a_h4_production_run09/2026-10-10'
    p,a,r=[json.loads((b/n).read_bytes()) for n in ('plan_authorized_v19.json','authorization_v19.json','review_v19.json')]
    for doc in (p,a,r):doc['run_id']=bind.RUN_ID
    p.update(output_root='/home/AbeHiromu/artificial/output/'+bind.RUN_ID,control_root='/home/AbeHiromu/artificial/control/'+bind.RUN_ID,
        gaussian_synthesis_profile=dict(path='artificial-Gaussian-profile.json',sha256='3'*64))
    a['gaussian_amendment']=dict(approved=True,from_semantics='h4-full-gaussian-paired-wrapper-v1',to_semantics=structure.SEMANTICS,
        profile=p['gaussian_synthesis_profile'],authority=dict(path='artificial-Gaussian-authority.json',sha256='4'*64),old_partial_cache_reuse=False)
    rebound(p,a,r);return p,a,r


class GaussianIntegrationTests(unittest.TestCase):
    def test_closed_profile_rejects_approximation_gate_or_semantics_change(self):
        self.assertEqual(structure.verify(structure.PARAMETERS),structure.PARAMETERS)
        for key,value in [('numerical_atol',1e-3),('approximate_pruning',True),('maximum_two_mode_rotations',100),
                          ('phase_gate','rz'),('wrapper_semantics','old'),('old_partial_cache_reuse',True)]:
            p=dict(structure.PARAMETERS);p[key]=value
            with self.assertRaises(Stop):structure.verify(p)
    def test_verified_math_bodies_are_retained_without_reexecuting_campaign(self):
        folder=ROOT/'src/trottertracks/resource_applicability/h4_geometry'
        def functions(name):
            tree=ast.parse((folder/name).read_bytes());values={}
            for n in tree.body:
                if isinstance(n,ast.FunctionDef):
                    if isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body=n.body[1:]
                    for node in ast.walk(n):
                        if isinstance(node,ast.Name) and node.id=='POLICY':node.id='PARAMETERS'
                    values[n.name]=ast.dump(n,include_attributes=False)
            return values
        old,new=functions('gaussian_structure_proposal.py'),functions('gaussian_structure.py')
        for name in ('givens_plan','orbital_matrix','fermionic_pair_matrix','build_basis'):self.assertEqual(old[name],new[name])
    def test_production_basis_dispatch_uses_structured_builder_only(self):
        with patch.object(structure,'build_basis',return_value=sentinel.basis) as build:
            self.assertIs(circuits.gaussian_basis(sentinel.orbital),sentinel.basis);build.assert_called_once_with(sentinel.orbital)
        self.assertIs(circuits.diagonalize_block.__kwdefaults__['basis_builder'],circuits.gaussian_basis)
    def test_candidate_new_cost_series_seeds_and_keys_are_separated(self):
        inputs=dict(geometry='0.70',H='1'*64,DF='2'*64,state='3'*64,input='4'*64);template=dict(method='B2',L_D=3,q=1,T=.8,delta=.8,r=1,K=2)
        new=science.candidate_identity(inputs,template,'5'*40,'6'*64,'7'*64);old={**new,'wrapper_semantics':'h4-full-gaussian-paired-wrapper-v1'}
        self.assertEqual(new['wrapper_semantics'],structure.SEMANTICS)
        self.assertNotEqual(trajectory_seed(new,0,actual_inputs_frozen=True,signal_launch=True),trajectory_seed(old,0,actual_inputs_frozen=True,signal_launch=True))
        self.assertNotEqual(wrapper_key(new,'cosine',None,None),wrapper_key(old,'cosine',None,None))
        self.assertEqual(new['template'],fingerprint('h4-template-v1',template))
    def test_run10_closed_adoption_pure_authorizes_when_bound(self):
        p,a,r=documents();permit=bind.authorize(p,a,r,explicit_launch=True)
        self.assertEqual(permit.plan['run_id'],'h4-newhost-signal-compile-20261010-run10');self.assertEqual(p['caps']['worker_AS_RSS'],32*2**30)
    def test_false_missing_or_old_cost_adoption_stops(self):
        for key,value in [('approved',False),('to_semantics','h4-full-gaussian-paired-wrapper-v1'),('old_partial_cache_reuse',True),
                          ('profile',dict(path='different',sha256='3'*64))]:
            p,a,r=documents();a['gaussian_amendment'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
        p,a,r=documents();a.pop('gaussian_amendment');rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_new_source_still_requires_explicit_launch_and_independent_flags(self):
        p,a,r=documents()
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=False)
        r['approved']=False;rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_source_gate_has_exact_user_authority_and_profile_binding(self):
        tree=ast.parse((ROOT/'src/trottertracks/resource_applicability/h4_geometry/launch_binding.py').read_bytes())
        runtime=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_runtime')
        strings={n.value for n in ast.walk(runtime) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertTrue({'h4-user-gaussian-authority-v1','新方式へ切替・再実行','gaussian_cost_amendment_approved','accepted_profile','old_partial_cache_reuse'}<=strings)
    def test_schema_requires_profile_and_closed_new_series_approval(self):
        s=json.loads((ROOT/'schemas/h4_newhost_launch_v2.json').read_bytes())
        self.assertIn('gaussian_synthesis_profile',s['properties']['plan']['required'])
        a=s['properties']['authorization'];self.assertIn('gaussian_amendment',a['required']);self.assertFalse(a['properties']['gaussian_amendment']['additionalProperties'])
    def test_old_run09_manual_stop_native_cost_and26_byte_files(self):
        r=proof.verify_run09_stop(dict(newhost_run09_predecessor=dict(path=str(proof.RECEIPT),bytes=proof.RECEIPT_BYTES,sha256=proof.RECEIPT_SHA)))
        self.assertEqual(len(proof.validate_metadata(r)),6);self.assertEqual(len(r['files']),26)
        self.assertEqual((r['actual_invocations_consumed_or_reserved'],r['completed_wrappers'],r['signal_records']),(8,4,2))
    def test_old_exit_or_cost_cannot_be_backfilled_or_erased(self):
        original=json.loads(proof.RECEIPT.read_bytes())
        for key,value in [('historical_driver_exit_code',130),('exact_driver_worker_exit_time',0.),('charged_bytes',0),('old_partial_results_reused',True)]:
            r=copy.deepcopy(original);r[key]=value
            with self.assertRaises(Stop):proof.validate_metadata(r)
    def test_wrong_proof_reference_or_same_owned_process_stops(self):
        with patch.object(proof,'streaming_sha') as stream:
            with self.assertRaises(Stop):proof.verify_run09_stop(dict(newhost_run09_predecessor={}))
            stream.assert_not_called()
        identities=proof.validate_metadata(json.loads(proof.RECEIPT.read_bytes()))
        with patch('trottertracks.resource_applicability.h4_geometry.observer.process_sample',side_effect=lambda pid:identities[pid]):
            with self.assertRaises(Stop):proof.verify_run09_stop(dict(newhost_run09_predecessor=dict(path=str(proof.RECEIPT),bytes=proof.RECEIPT_BYTES,sha256=proof.RECEIPT_SHA)))
