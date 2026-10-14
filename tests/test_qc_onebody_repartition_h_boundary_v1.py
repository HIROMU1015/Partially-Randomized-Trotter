"""Unregistered tiny fixtures; no saved molecular array is decoded/evaluated."""
from pathlib import Path
import copy,json,math,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition import qc_onebody_repartition_pilot_v1 as old
from trottertracks.ground_pf_repartition import qc_onebody_repartition_resources_v1 as oldcost
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_protocol_v1 as policy
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_pilot_v1 as new
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_checker_v1 as checker
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_execution_v1 as gate

def contract():return json.loads((ROOT/gate.ARTIFACT/'contract.json').read_text())

def fixture():
    a=old.spin_frame(np.array([[.1,.27],[.27,-.08]]));gs=np.array([old.spin_frame(np.diag([.4,-.2])),old.spin_frame(np.array([[.1,.35],[.35,.25]]))])
    labels=np.array([3,6,9,12]);weights=np.array([.8,.45]);constant=.13
    ladders=old.sector_ladders(4,labels);H=constant*np.eye(4)+old.lift(a,ladders)+sum(w*(old.lift(g,ladders)@old.lift(g,ladders)) for w,g in zip(weights,gs))
    ev,V=np.linalg.eigh(H);budget=old.Budget({k:100 for k in ('RDM_acquisitions','spatial_eigh_calls','direct_leading_evaluations')})
    design=old.build_design(a,weights,gs,constant,labels,V[:,0],budget);lam=old.prepare_actors(design,budget)
    data={'one_body':a,'lambdas':weights,'g_matrices':gs,'constant':np.array(constant),'labels':labels,'state':V[:,0],
          'H':H,'H_eigenvalues':ev,'H_eigenvectors':V,'gamma':design['gamma']}
    seal={'shifts':{},'leading_coefficients':{},'E0_accessed_by_generator':False,'finite_step_truth_accessed_by_generator':False,'selection':design['selector']}
    for c in design['configurations']:
        name=c['name'];data['shift_'+name]=c['shift'];data['fragments_'+name]=np.array(c['fragments'])
        for suffix,field in [('orbital_frames','orbital_frame'),('actor_eta','eta'),('actor_linear','linear'),('actor_pairs','pairs')]:data[suffix+'_'+name]=np.array([a[field] for a in c['actors']])
        seal['shifts'][name]=c['shift'].tolist();seal['leading_coefficients'][name]=c['d']
    c=contract();c['scope_id']='UNREGISTERED_FOUR_MODE_BOUNDARY_FIXTURE'
    c['schedule'].update(h_anchor=.03,Lambda_common=lam,maximum_h=.12,coarse_ratio=1.2,initial_h_cap=256,total_h_cap=384,refinement_intervals_per_round=4)
    return data,seal,{'truth':{'E0':float(ev[0])},'records':[]},c

def record(h,bias=0.,method='S2:original',ambiguous=False):
    return {'method':method,'h':h,'h_index':0,'bias':bias,'overlap':1.,'numerical_allowance':1e-10,'ambiguous':ambiguous}

class PolicyTests(unittest.TestCase):
    def test_real_schedule_metadata_only_caps(self):
        c=contract();g,K=policy.initial_grid(c['schedule']['h_anchor'],c['schedule']['Lambda_common'],c['schedule'])
        self.assertLessEqual(len(g),1024);self.assertEqual(g[-1],1.);self.assertEqual(g[0],c['schedule']['h_anchor']/2)
        self.assertTrue(all(b-a<=math.pi/(8*K)*(1+1e-12) for a,b in zip(g[:-1],g[1:])))
    def test_half_closure_and_signed_S4_guard(self):
        c=contract();g,K=policy.initial_grid(.01,2.,{**c['schedule'],'maximum_h':.1})
        assessed,_=policy.initial_grid(.01,2.,{**c['schedule'],'maximum_h':.1},support=False)
        self.assertGreater(K,2.);self.assertTrue(all(x/2 in g for x in assessed))
    def test_invalid_grid_scale(self):
        for h,lam in [(0.,2.),(.01,-1.),(math.nan,1.)]:
            with self.assertRaises(ValueError):policy.initial_grid(h,lam,contract()['schedule'])
    def test_grid_cap_not_silent_truncation(self):
        s={**contract()['schedule'],'initial_h_cap':2}
        with self.assertRaises(ValueError):policy.initial_grid(.01,2.,s)
    def test_numerical_boundary_is_not_bias_failure(self):
        c=contract();r=record(.1,5e-6-5e-11)
        self.assertEqual(policy.eligibility(r,1e-5,c),(False,'NUMERICAL_BOUNDARY'))
    def test_nonfinite_record_refused(self):
        with self.assertRaises(ValueError):policy.eligibility(record(.1,math.nan),1e-5,contract())
    def test_union_refinement_multiple_eligibility_islands(self):
        c=contract();r=[record(.1,0.),record(.12,1.),record(.14,0.)]
        points,ledger=policy.refinement_proposals(r,[.1,.12,.14],c)
        self.assertEqual(len(ledger['selected_intervals']),2);self.assertIn(math.sqrt(.1*.12),points)
    def test_refinement_cap_tie_and_half_points(self):
        c=contract();c['schedule']['refinement_intervals_per_round']=1
        points,ledger=policy.refinement_proposals([record(.1),record(.12,1.),record(.15)],[.1,.12,.15],c)
        self.assertEqual(len(ledger['selected_intervals']),1);self.assertEqual(ledger['unselected_interval_count'],1)
    def test_upper_edge_is_undetermined(self):
        c=contract();r=record(.1);self.assertFalse(policy.boundary_status(r,[r],1e-5,c)['resolved'])
    def test_boundary_numerical_and_overlap_not_overclaimed(self):
        c=contract();a=record(.1);b=record(.1005,1.)
        self.assertTrue(policy.boundary_status(a,[a,b],1e-5,c)['resolved'])
        b['ambiguous']=True;self.assertFalse(policy.boundary_status(a,[a,b],1e-5,c)['resolved'])

class BranchTests(unittest.TestCase):
    def phase(self,theta,h,c=None,left=None,right=None,state=None,K=10.):
        c=contract() if c is None else c;z=np.exp(-1j*np.asarray(theta));U=np.diag(z);V=np.eye(len(z),dtype=complex)
        return new.phase_from_spectrum(U,V,z,np.array([1.,0.]) if state is None else state,h,0.,c,K,left,right)
    def test_wrap_continuation_without_E0_fit(self):
        a,P=self.phase([-3.12,1.],.39,K=8.)
        a.update(ambiguous=False,theta=-3.12,h=.39)
        b,Q=self.phase([-3.2,1.],.4,left=(a,P),K=8.)
        self.assertEqual(b['alias_integer'],-1);self.assertAlmostEqual(b['theta'],-3.2);self.assertFalse(b['ambiguous'])
    def test_signed_stage_root_guard(self):
        r,_=self.phase([0.,1.],.1,K=10.);self.assertIn('ANCHOR_PHASE_GUARD',r['branch_reasons'])
    def test_no_restart_after_invalid_left(self):
        a,P=self.phase([0.,1.],.01,K=10.);a['ambiguous']=True
        b,_=self.phase([.01,1.],.02,left=(a,P));self.assertIn('LEFT_BRANCH_INVALID_NO_RESTART',b['branch_reasons'])
    def test_circular_cluster_crosses_pi_cut(self):
        z=np.exp(-1j*np.array([math.pi-1e-12,-math.pi+1e-12]))
        groups=new.circular_clusters(z,.1,1e-9);self.assertEqual(len(groups),1)
    def test_equal_state_overlap_is_ambiguous(self):
        r,_=self.phase([0.,.1],.01,state=np.ones(2)/math.sqrt(2));self.assertIn('INPUT_OVERLAP_TIE',r['branch_reasons'])
    def test_step_too_large_refused(self):
        a,P=self.phase([0.,1.],.01)
        b,_=self.phase([.1,1.],.11,left=(a,P));self.assertIn('CONTINUATION_STEP_TOO_LARGE',b['branch_reasons'])
    def test_left_right_alias_disagreement(self):
        a,P=self.phase([0.,1.],.01);r={**a,'h':.03,'theta':2*math.pi}
        b,_=self.phase([.1,1.],.02,left=(a,P),right=(r,P));self.assertIn('LEFT_RIGHT_ALIAS_CONFLICT',b['branch_reasons'])
    def test_spectral_defect_recorded(self):
        c=contract();U=np.diag([1.01,1.]);V=np.eye(2);z=np.diag(U)
        r,_=new.phase_from_spectrum(U,V,z,np.array([1.,0.]),.01,0.,c,10.)
        self.assertIn('NUMERICAL_SPECTRAL_DEFECT',r['branch_reasons'])

class GateTests(unittest.TestCase):
    def test_missing_approval_before_any_input_decode(self):
        with patch('numpy.load',side_effect=AssertionError('REGISTERED_DECODE')):
            with self.assertRaisesRegex(ValueError,'NEW_EXPLICIT'):gate.authorize(ROOT,None)
    def test_consumed_v1_authorization_not_reusable(self):
        p=ROOT/'artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/authorization.json'
        with patch('numpy.load',side_effect=AssertionError('REGISTERED_DECODE')):
            with self.assertRaisesRegex(ValueError,'NEW_SCOPE'):gate.authorize(ROOT,p)
    def test_unissued_target_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';p.write_text(json.dumps({'approved':False,'status':'UNISSUED_APPROVAL_TARGET'}))
            with self.assertRaisesRegex(ValueError,'UNISSUED'):gate.authorize(ROOT,p)
    def test_registered_execution_requires_context(self):
        with self.assertRaisesRegex(ValueError,'AUTHORIZED_CONTEXT'):new.run_registered(None)

class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)
        cls.data,cls.seal,cls.parent,cls.c=fixture();cls.budget=old.Budget(cls.c['limits'])
        with patch.object(old,'select_response',side_effect=AssertionError('RESELECT')):
            cls.stop=new.collect(cls.data,cls.seal,cls.parent,cls.c,cls.path,cls.budget)
        cls.raw=json.loads((cls.path/'raw.json').read_text())
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_independent_full_numeric_and_price_audit(self):
        r=checker.audit(self.data,self.seal,self.raw,self.c,self.path)
        self.assertEqual(r['status'],'INDEPENDENT_BOUNDARY_JW_PADE_PHASE_COST_POLICY_AUDIT_PASS')
    def test_restore_keeps_exact_saved_shifts_and_no_solver(self):
        with patch('numpy.linalg.eigh',side_effect=AssertionError('NEW_EIGH')):
            configs=new.restore(self.data,self.seal)
        for name,c in configs.items():np.testing.assert_array_equal(c['shift'],self.data['shift_'+name])
    def test_spectral_seed_matches_old_and_independent_word(self):
        c=new.restore(self.data,self.seal)['response_selected'];b=old.Budget(self.c['limits']);nodes=old.word(4,.07,len(c['actors']))
        U=new.SpectralCache().seed(c['name'],c['actors'],nodes,c['constant'],.07,b)
        np.testing.assert_allclose(U,old.seed(c['actors'],nodes,c['constant'],.07,b),atol=1e-13)
    def test_strong_control_and_all_methods_same_grid(self):
        primitive=self.raw['primitive_records']
        self.assertEqual(len(primitive),6*len(self.raw['h_grid']));self.assertEqual(len(self.raw['result']['cells']),4)
        self.assertEqual(len(self.raw['result']['secondary_same_family']),16)
        self.assertEqual(len(self.raw['records']),12*len(self.raw['assessment_h_grid']))
    def test_corrupt_price_independent_checker_rejects(self):
        r=copy.deepcopy(self.raw);cost=next(x for x in r['costs'] if x['eligible']);cost['conditional_price_total']+=1
        with self.assertRaisesRegex(ValueError,'PRICE'):checker.audit(self.data,self.seal,r,self.c,self.path)
    def test_corrupt_alias_independent_checker_rejects(self):
        r=copy.deepcopy(self.raw);r['primitive_records'][0]['alias_integer']+=1
        with self.assertRaisesRegex(ValueError,'PHASE_UNWRAP'):checker.audit(self.data,self.seal,r,self.c,self.path)
    def test_corrupt_refinement_independent_checker_rejects(self):
        r=copy.deepcopy(self.raw);r['refinement_ledger'][0]['previous_h_count']+=1
        with self.assertRaisesRegex(ValueError,'REFINEMENT'):checker.audit(self.data,self.seal,r,self.c,self.path)
    def test_removed_numerical_floor_independent_checker_rejects(self):
        r=copy.deepcopy(self.raw);r['primitive_records'][0]['numerical_allowance']=0.
        with self.assertRaisesRegex(ValueError,'NUMERICAL_FLOOR'):checker.audit(self.data,self.seal,r,self.c,self.path)
    def test_corrupt_derived_allowance_independent_checker_rejects(self):
        r=copy.deepcopy(self.raw);point=next(x for x in r['records'] if x['method'].startswith('CS2:'));point['numerical_allowance']=0.
        with self.assertRaisesRegex(ValueError,'CS2_NUMERICAL'):checker.audit(self.data,self.seal,r,self.c,self.path)
    def test_scalar_info_precision_and_no_forced_gamma(self):
        rows=self.raw['costs'];cs=next(r for r in rows if r['method']=='CS2:original' and r['eligible'])
        self.assertEqual(cs['information']['required_leading_absolute_precision'],cs['target_Ha']*.125/cs['h']**2)
        base=next(r for r in rows if r['method']=='S2:original' and r['eligible'])
        self.assertFalse(base['information']['gamma_required']);self.assertFalse(base['information']['leading_coefficient_required'])
    def test_cost_restores_v1_primitive_policy(self):
        configs=new.restore(self.data,self.seal);r=next(x for x in self.raw['records'] if x['method']=='S4:response_selected')
        oldrow=oldcost.account(r,configs,self.c,1e-4,self.c['resources']['pricing'][0])
        current=next(x for x in self.raw['costs'] if x['method']==r['method'] and x['h_index']==r['h_index'] and x['target_Ha']==1e-4 and x['price_id']=='P1')
        if current['eligible']:self.assertEqual(oldrow['primitive_vector'],current['primitive_vector'])
    def test_cached_affine_counts_match_v1_all_configs_orders(self):
        configs=new.restore(self.data,self.seal)
        for c in configs.values():
            for order in (2,4):
                nodes=old.word(order,1.,len(c['actors']))
                one=oldcost.circuit_counts(c['actors'],nodes,c['constant'],1);two=oldcost.circuit_counts(c['actors'],nodes,c['constant'],2)
                for h in (.015,.04,.12):
                    for q in (1,9):
                        actual=oldcost.circuit_counts(c['actors'],old.word(order,h,len(c['actors'])),c['constant'],q)
                        self.assertEqual(actual,{k:one[k]+(q-1)*(two[k]-one[k]) for k in one})
    def test_hash_bound_anchor_cache_avoids_its_solver_calls(self):
        data=copy.deepcopy(self.data);parent=copy.deepcopy(self.parent);parent['records']=[]
        anchors={self.c['schedule']['h_anchor']/2,self.c['schedule']['h_anchor']}
        for r in self.raw['primitive_records']:
            if r['h'] not in anchors:continue
            parent['records'].append(copy.deepcopy(r))
            with np.load(self.path/'cells'/f'{r["key"]}.npz',allow_pickle=False) as z:
                data[r['key']+'_U']=np.array(z['U']);data[r['key']+'_vectors']=np.array(z['vectors']);data[r['key']+'_eigenvalues']=np.array(z['eigenvalues'])
        with tempfile.TemporaryDirectory() as d:
            budget=old.Budget(self.c['limits']);new.collect(data,self.seal,parent,self.c,d,budget)
            raw=json.loads((Path(d)/'raw.json').read_text());self.assertEqual(raw['cached_anchor_cells'],12)
            self.assertEqual(len(raw['primitive_records'])-12,budget.counts['PF_Schur_calls'])

if __name__=='__main__':unittest.main()
