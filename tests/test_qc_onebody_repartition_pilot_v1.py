"""Unregistered four-mode fixtures only; never load a molecular pack."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition import qc_onebody_repartition_pilot_v1 as core
from trottertracks.ground_pf_repartition import qc_onebody_repartition_resources_v1 as costs
from trottertracks.ground_pf_repartition import qc_onebody_repartition_checker_v1 as checker
from trottertracks.ground_pf_repartition import qc_onebody_repartition_execution_v1 as launch
from trottertracks.ground_pf_repartition.qc_onebody_repartition_pipeline_v1 import collect


def contract():
    return json.loads((ROOT/launch.ARTIFACT/'contract.json').read_text())


def fixture():
    """Not a molecule or any saved scientific input; 2 spatial/4 spin modes."""
    a=core.spin_frame(np.array([[.1,.27],[.27,-.08]]))
    gs=np.array([core.spin_frame(np.diag([.4,-.2])),core.spin_frame(np.array([[.1,.35],[.35,.25]]))])
    labels=np.array([3,6,9,12]);weights=np.array([.8,.45]);c=.13
    ladders=core.sector_ladders(4,labels)
    H=c*np.eye(4)+core.lift(a,ladders)+sum(w*(core.lift(g,ladders)@core.lift(g,ladders)) for w,g in zip(weights,gs))
    _,vectors=np.linalg.eigh(H)  # authorized unregistered fixture, not molecular eigensolver
    return a,weights,gs,c,labels,vectors[:,0]


class SelectorTests(unittest.TestCase):
    def test_linear_root(self):
        t,table=core.bounded_candidates(.4,2.,0.,1.)
        self.assertAlmostEqual(t,-.2);self.assertLessEqual(len(table),6)
    def test_quadratic_roots(self):
        t,_=core.bounded_candidates(-.25,0.,1.,1.)
        self.assertAlmostEqual(t,-.5)
    def test_vertex(self):
        t,_=core.bounded_candidates(1.,2.,2.,1.)
        self.assertAlmostEqual(t,-.5)
    def test_zero_tie_fallback(self):
        t,_=core.bounded_candidates(0.,0.,0.,1.)
        self.assertEqual(t,0.)
    def test_roots_outside_bound(self):
        t,table=core.bounded_candidates(10.,1.,0.,.25)
        self.assertEqual(t,-.25);self.assertTrue(all(abs(x['t'])<=.25 for x in table))
    def test_nonfinite_or_negative_bound(self):
        for values in [(math.nan,1,1,1),(1,math.inf,1,1),(1,1,1,-1)]:
            with self.assertRaises(core.PilotError):core.bounded_candidates(*values)
    def test_zero_projected_gradient(self):
        a=np.eye(4);g=np.eye(4)*.5
        x,r=core.select_response(a,g,.2,.1)
        self.assertEqual(r['status'],'ZERO_PROJECTED_GRADIENT');np.testing.assert_array_equal(x,0)
    def test_response_number_gauge(self):
        a=core.spin_frame(np.array([[.2,.3j],[-.3j,-.1]]));gamma=np.eye(4)*.5
        g,Q=core.response(a,gamma)
        np.testing.assert_allclose(Q@np.ones(4),0,atol=1e-14);self.assertAlmostEqual(float(sum(g)),0)
    def test_response_shape_and_hermiticity_reject(self):
        with self.assertRaises(core.PilotError):core.response(np.eye(2),np.eye(3))
        with self.assertRaises(core.PilotError):core.response(np.array([[0,1],[0,0]]),np.eye(2))


class FermionAndPFTests(unittest.TestCase):
    def test_independent_JW_parity_and_endianness(self):
        labels=np.array([3,6,9,12])
        np.testing.assert_allclose(core.sector_ladders(4,labels),checker.jw_transitions(4,labels),atol=0)
    def test_duplicate_sector_refused(self):
        with self.assertRaises(core.PilotError):core.sector_ladders(4,[3,3])
    def test_spin_mixing_refused(self):
        a=np.eye(4,dtype=complex);a[0,1]=a[1,0]=.2
        with self.assertRaises(core.PilotError):core.paired_eigh(a,core.Budget(contract()['limits']))
    def test_invalid_cached_state_refused(self):
        values=list(fixture());values[-1]=values[-1]*2
        with self.assertRaises(core.PilotError):core.build_design(*values,core.Budget(contract()['limits']))
    def test_H_identity_and_response_direct(self):
        d=core.build_design(*fixture(),core.Budget(contract()['limits']))
        for c in d['configurations']:
            np.testing.assert_allclose(sum(c['fragments'])+d['constant']*np.eye(4),d['H'],atol=1e-13)
            self.assertAlmostEqual(float(np.mean(c['shift'])),0)
            np.testing.assert_allclose(c['shift'][::2],c['shift'][1::2],atol=0)
        self.assertFalse(d['selector'].get('E0_accessed',False))
    def test_unitary_exterior_frame(self):
        u=core.spin_frame(np.array([[.8,-.6],[.6,.8]]));F=core.exterior_frame(u,[3,6,9,12])
        np.testing.assert_allclose(F.conj().T@F,np.eye(4),atol=1e-14)
    def test_S4_signed_order_and_fusion(self):
        nodes=core.word(4,.1,3)
        self.assertTrue(any(t<0 for _,t in nodes))
        self.assertTrue(all(a[0]!=b[0] for a,b in zip(nodes,nodes[1:])))
        for j in range(3):self.assertAlmostEqual(sum(t for k,t in nodes if k==j),.1)
    def test_unregistered_PF_order_refused(self):
        with self.assertRaises(core.PilotError):core.word(6,.1,3)
    def test_budget_stops_before_overrun(self):
        budget=core.Budget({'H_truth_eigh_calls':0})
        with self.assertRaises(core.PilotError):budget.tick('H_truth_eigh_calls')
        self.assertEqual(budget.counts,{})
    def test_phase_branch_tie_undetermined(self):
        U=np.diag(np.exp(-1j*np.array([-.2,.2])));state=np.ones(2)/np.sqrt(2)
        rec=core.phase_record(U,state,.1,0.,contract(),core.Budget(contract()['limits']))
        self.assertTrue(rec['ambiguous'])


class CostAndGateTests(unittest.TestCase):
    def test_overlap_repeat_budget(self):
        self.assertEqual(costs.repeats(1.,.01),1);self.assertEqual(costs.repeats(.9,.01),2)
        with self.assertRaises(core.PilotError):costs.repeats(0,.01)
    def test_numeric_boundary_rejected(self):
        r={'bias':.5e-5,'numerical_allowance':1e-10,'overlap':1.,'ambiguous':False}
        self.assertFalse(costs.eligible(r,1e-5,contract())[0])
    def test_query_boundary_fusion(self):
        actor={'orbital_frame':np.eye(4,dtype=complex),'linear':np.ones(4),'pairs':np.zeros((4,4))}
        nodes=core.word(2,.1,2)
        once=costs.circuit_counts([actor,actor],nodes,.2,1)
        twice=costs.circuit_counts([actor,actor],nodes,.2,2)
        self.assertEqual(once['controlled_phase'],13)
        self.assertEqual(twice['controlled_phase'],21)
    def test_known_fixed_number_scalar_elision(self):
        actor={'orbital_frame':core.spin_frame(np.array([[.8,-.6],[.6,.8]])),
               'linear':np.ones(4),'pairs':np.zeros((4,4)),
               'kind':'onebody','eta':np.ones(4),'particles':2}
        counts=costs.circuit_counts([actor,actor],core.word(2,.1,2),0.,100)
        self.assertEqual(counts['controlled_Givens'],0);self.assertEqual(counts['controlled_phase'],1)
    def test_empty_eligible_grid_undetermined(self):
        result=costs.comparison([],contract())
        self.assertEqual(result['classification'],'UNDETERMINED');self.assertFalse(result['formal_research_GO'])
    def test_missing_approval_before_any_preflight_or_science(self):
        with self.assertRaises(launch.LaunchError):launch.authorize(ROOT,None)
        self.assertFalse((ROOT/launch.OUTPUT).exists())
    def test_unissued_target_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'approval.json';p.write_text(json.dumps({'scope_id':launch.SCOPE,'approved':False,'status':'UNISSUED_APPROVAL_TARGET'}))
            with self.assertRaises(launch.LaunchError):launch.authorize(ROOT,p)
        self.assertFalse((ROOT/launch.OUTPUT).exists())
    def test_old_940_call_scope_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'approval.json';p.write_text(json.dumps({'scope_id':'QC_C1_MACROCYCLE_OLD','approved':True,'status':'ISSUED_USER_EXPLICIT_EXECUTION','attempts':1,'retry':0}))
            with self.assertRaises(launch.LaunchError):launch.authorize(ROOT,p)
        self.assertFalse((ROOT/launch.OUTPUT).exists())
    def test_pipeline_requires_authorized_context(self):
        from trottertracks.ground_pf_repartition.qc_onebody_repartition_pipeline_v1 import run_registered
        with self.assertRaises(launch.LaunchError):run_registered(None)


class SyntheticEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='prt-unregistered-onebody-fixture-')
        cls.path=Path(cls.temp.name);cls.contract=contract();budget=core.Budget(cls.contract['limits'])
        design=core.build_design(*fixture(),budget)
        cls.raw=collect(design,cls.contract,budget,cls.path)
        with np.load(cls.path/'raw.npz',allow_pickle=False) as z:cls.data={k:np.array(z[k]) for k in z.files}
        cls.seal=json.loads((cls.path/'selector_seal.json').read_text())
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_independent_fermion_Pade_cost_audit(self):
        result=checker.audit_arrays(self.data,self.raw,self.seal,self.contract)
        self.assertEqual(result['status'],'INDEPENDENT_JW_PADE_DIRECT_RESPONSE_COST_AUDIT_PASS')
        self.assertEqual(result['new_eigensolvers'],0)
    def test_scalar_correction_tamper_rejected(self):
        raw=copy.deepcopy(self.raw);r=next(r for r in raw['records'] if r['method'].startswith('CS2:'));r['energy']+=.01
        with self.assertRaises(checker.AuditError):checker.audit_arrays(self.data,raw,self.seal,self.contract)
    def test_Hamiltonian_tamper_rejected(self):
        data=copy.deepcopy(self.data);data['fragments_response_selected'][0,0,0]+=.01
        with self.assertRaises(checker.AuditError):checker.audit_arrays(data,self.raw,self.seal,self.contract)
    def test_selector_truth_leak_rejected(self):
        seal=copy.deepcopy(self.seal);seal['E0_accessed_by_generator']=True
        with self.assertRaises(checker.AuditError):checker.audit_arrays(self.data,self.raw,seal,self.contract)
    def test_PF_matrix_tamper_rejected(self):
        data=copy.deepcopy(self.data);key=next(k for k in data if k.endswith('_U'));data[key][0,0]+=.01
        with self.assertRaises(checker.AuditError):checker.audit_arrays(data,self.raw,self.seal,self.contract)
    def test_controlled_cost_tamper_rejected(self):
        raw=copy.deepcopy(self.raw);row=next(c for c in raw['costs'] if c['eligible']);row['conditional_price_total']+=1
        with self.assertRaises(checker.AuditError):checker.audit_arrays(self.data,raw,self.seal,self.contract)
    def test_actor_phase_coefficients_tamper_rejected(self):
        data=copy.deepcopy(self.data);data['actor_pairs_original'][1,0,1]+=.01
        with self.assertRaises(checker.AuditError):checker.audit_arrays(data,self.raw,self.seal,self.contract)
    def test_common_h_bound_tamper_rejected(self):
        raw=copy.deepcopy(self.raw);raw['Lambda_common']*=.5
        with self.assertRaises(checker.AuditError):checker.audit_arrays(self.data,raw,self.seal,self.contract)
    def test_grid_coverage_tamper_rejected(self):
        raw=copy.deepcopy(self.raw);raw['records'].pop()
        with self.assertRaises(checker.AuditError):checker.audit_arrays(self.data,raw,self.seal,self.contract)


if __name__=='__main__':unittest.main()
