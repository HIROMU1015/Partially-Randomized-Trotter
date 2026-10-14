"""Repair regressions on unregistered fixtures only; no H4 array load."""
from pathlib import Path
import copy,json,math,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_qc_onebody_repartition_h_boundary_v1 import fixture,contract
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_checker_v1 as previous
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_checker_v2 as fixed
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_pilot_v1 as producer
from trottertracks.ground_pf_repartition import qc_onebody_repartition_pilot_v1 as tiny
from trottertracks.ground_pf_repartition import qc_onebody_repartition_h_boundary_reaudit_v2 as gate

class AllowanceTests(unittest.TestCase):
    def example(self,theta=(.001,.12),h=.001,offset=2e-12):
        z=np.exp(-1j*np.array(theta));V=np.eye(2,dtype=complex);exact=np.diag(z);stored=(1+offset)*exact;state=np.array([1.,0.]);c=contract()
        r,_=producer.phase_from_spectrum(stored,V,z,state,h,0.,c,1.)
        r.update(key='unregistered',config='synthetic',order=2,left_key=None,right_key=None)
        return r,stored,exact,V,z,state,c
    def check(self,x):
        r,stored,exact,V,z,state,c=x
        return fixed.phase_check(r,stored,V,z,state,c,1.,{},independent_U=exact)
    def test_distinct_matrices_old_failure_new_self_consistency(self):
        x=self.example();r,stored,exact,V,z,state,c=x;before=copy.deepcopy(r)
        with self.assertRaisesRegex(ValueError,'NUMERICAL_ALLOWANCE'):previous.phase_check(r,exact,V,z,state,c,1.,{})
        self.check(x);self.assertEqual(r,before)
    def test_unaltered_identical_matrix(self):self.check(self.example(offset=0.))
    def test_allowance_corruption_refused(self):
        x=self.example();x[0]['numerical_allowance']+=1e-7
        with self.assertRaisesRegex(ValueError,'NUMERICAL_ALLOWANCE'):self.check(x)
    def test_residual_corruption_refused(self):
        x=self.example();x[0]['spectral_residual']+=1e-12
        with self.assertRaisesRegex(ValueError,'SAVED_SPECTRAL_RESIDUAL'):self.check(x)
    def test_unitarity_corruption_refused(self):
        x=self.example();x[0]['unitarity_defect']+=1e-12
        with self.assertRaisesRegex(ValueError,'SAVED_UNITARITY_DEFECT'):self.check(x)
    def test_missing_orthogonality_refused(self):
        x=self.example();del x[0]['eigenvector_orthogonality_defect']
        with self.assertRaisesRegex(ValueError,'SAVED_EIGENVECTOR'):self.check(x)
    def test_nonfinite_allowance_refused(self):
        x=self.example();x[0]['numerical_allowance']=math.nan
        with self.assertRaisesRegex(ValueError,'NONFINITE_ALLOWANCE'):self.check(x)
    def test_zero_h_refused(self):
        x=self.example();x[0]['h']=0
        with self.assertRaisesRegex(ValueError,'INVALID_H'):self.check(x)
    def test_large_pade_difference_refused(self):
        x=list(self.example());x[2]=x[2]+np.eye(2)*1e-7
        with self.assertRaisesRegex(ValueError,'INDEPENDENT_PADE_PF'):self.check(x)
    def test_independent_spectral_threshold_not_relaxed(self):
        x=list(self.example(offset=0.));x[2]*=1+4e-10
        with self.assertRaisesRegex(ValueError,'INDEPENDENT_SPECTRAL_DEFECT'):self.check(x)
    def test_corrupted_branch_reasons_refused(self):
        x=self.example();x[0]['branch_reasons']=['ANCHOR_PHASE_GUARD']
        with self.assertRaisesRegex(ValueError,'BRANCH_VALIDITY'):self.check(x)
    def test_wrap_alias_remains_truth_free(self):
        c=contract();V=np.eye(2,dtype=complex);state=np.array([1.,0.])
        z=np.exp(-1j*np.array([-3.12,1.]));U=np.diag(z)
        left,P=producer.phase_from_spectrum(U,V,z,state,.39,0.,c,8.)
        left.update(ambiguous=False,theta=-3.12,h=.39,key='left')
        z=np.exp(-1j*np.array([-3.2,1.]));U=np.diag(z)
        r,_=producer.phase_from_spectrum(U,V,z,state,.4,12345.,c,8.,left=(left,P))
        r.update(left_key='left',right_key=None)
        fixed.phase_check(r,U,V,z,state,c,8.,{'left':(left,P)},independent_U=U)
        self.assertEqual(r['alias_integer'],-1)
    def test_wrong_cluster_refused(self):
        x=self.example();x[0]['cluster_indices']=[0,1]
        with self.assertRaisesRegex(ValueError,'CIRCULAR_CLUSTER'):self.check(x)
    def test_circular_spread_across_pi_cut(self):
        self.check(self.example(theta=(math.pi-1e-13,-math.pi+1e-13),h=.01,offset=0.))
    def test_diagnostic_keeps_both_matrix_allowances(self):
        trace=fixed.AuditTrace();token=fixed._ACTIVE.set(trace)
        try:self.check(self.example())
        finally:fixed._ACTIVE.reset(token)
        self.assertGreater(abs(trace.details['reproduced_allowance']-trace.details['independent_allowance']),1e-10)
        self.assertIn('saved_matrix_metrics',trace.details)
    def test_snapshot_nonfinite_serializable(self):
        trace=fixed.AuditTrace();trace.details={'bad':math.nan}
        json.dumps(trace.snapshot(),allow_nan=False)

class SyntheticAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)
        cls.data,cls.seal,parent,cls.c=fixture()
        producer.collect(cls.data,cls.seal,parent,cls.c,cls.path,tiny.Budget(cls.c['limits']))
        cls.raw=json.loads((cls.path/'raw.json').read_text())
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_complete_synthetic_audit(self):
        result=fixed.audit(self.data,self.seal,self.raw,self.c,self.path)
        self.assertIn('AUDIT_PASS',result['status']);self.assertEqual(result['checker_version'],2)
        self.assertEqual(result['diagnostics']['partial_counts']['cells_completed'],len(self.raw['primitive_records']))
    def test_failure_cell_metrics_partial_counters(self):
        raw=copy.deepcopy(self.raw);raw['primitive_records'][0]['numerical_allowance']+=1e-7
        with self.assertRaises(fixed.AuditFailure) as caught:fixed.audit(self.data,self.seal,raw,self.c,self.path)
        d=caught.exception.diagnostic;self.assertEqual(d['cell']['key'],raw['primitive_records'][0]['key'])
        self.assertEqual(d['predicate'],'NUMERICAL_ALLOWANCE');self.assertGreater(d['partial_counts']['expm_calls'],0)
        self.assertEqual(d['partial_counts']['cells_completed'],0);self.assertIn('independent_allowance',d['details'])
    def test_expm_exception_counts_started_and_completed(self):
        with patch.object(fixed,'expm',side_effect=RuntimeError('synthetic_expm_failure')):
            with self.assertRaises(fixed.AuditFailure) as caught:fixed.audit(self.data,self.seal,self.raw,self.c,self.path)
        d=caught.exception.diagnostic;self.assertEqual(d['partial_counts']['expm_calls'],1)
        self.assertEqual(d['partial_counts']['expm_completed'],0);self.assertEqual(d['stage'],'primitive')
    def test_expm_cap_still_enforced(self):
        c=copy.deepcopy(self.c);c['limits']['independent_expm_calls']=0
        with self.assertRaisesRegex(ValueError,'EXPM_CAP'):fixed.audit(self.data,self.seal,self.raw,c,self.path)
    def test_derived_allowance_corruption_refused(self):
        raw=copy.deepcopy(self.raw);next(r for r in raw['records'] if r['method'].startswith('CS2:'))['numerical_allowance']+=1e-7
        with self.assertRaisesRegex(ValueError,'CS2_NUMERICAL'):fixed.audit(self.data,self.seal,raw,self.c,self.path)
    def test_cost_corruption_still_refused(self):
        raw=copy.deepcopy(self.raw);cost=next(r for r in raw['costs'] if r['eligible']);cost['conditional_price_total']+=1
        with self.assertRaisesRegex(ValueError,'PRICE'):fixed.audit(self.data,self.seal,raw,self.c,self.path)
    def test_h_coverage_corruption_still_refused(self):
        raw=copy.deepcopy(self.raw);raw['primitive_records'].pop()
        with self.assertRaisesRegex(ValueError,'COVERAGE_CAP'):fixed.audit(self.data,self.seal,raw,self.c,self.path)
    def test_synthesis_budget_corruption_still_refused(self):
        raw=copy.deepcopy(self.raw);next(r for r in raw['costs'] if r['eligible'])['uniform_synthesis_angle_allowance']*=2
        with self.assertRaisesRegex(ValueError,'SYNTHESIS'):fixed.audit(self.data,self.seal,raw,self.c,self.path)

class GateTests(unittest.TestCase):
    def test_missing_approval_before_decode(self):
        with patch('numpy.load',side_effect=AssertionError('REGISTERED_ARRAY_DECODE')):
            with self.assertRaisesRegex(ValueError,'NEW_EXPLICIT_REAUDIT'):gate.authorize(ROOT,None)
    def test_consumed_original_scope_rejected(self):
        p=ROOT/'artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/authorization.json'
        with patch('numpy.load',side_effect=AssertionError('REGISTERED_ARRAY_DECODE')):
            with self.assertRaisesRegex(ValueError,'NEW_REAUDIT_SCOPE'):gate.authorize(ROOT,p)
    def test_unissued_target_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';p.write_text(json.dumps({'approved':False}))
            with self.assertRaisesRegex(ValueError,'UNISSUED'):gate.authorize(ROOT,p)
    def test_wrong_issued_identity_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';p.write_text(json.dumps({'approved':True,'status':'ISSUED_USER_EXPLICIT_EXECUTION','scope_id':gate.SCOPE,
                'attempts':1,'retry':0,'issuer':'USER_EXPLICIT_SESSION_INSTRUCTION','user_instruction_verbatim':'synthetic_only','contract_SHA256':'0'*64}))
            with patch('numpy.load',side_effect=AssertionError('REGISTERED_ARRAY_DECODE')):
                with self.assertRaisesRegex(ValueError,'APPROVAL_contract_SHA256'):gate.authorize(ROOT,p)
    def test_identity_corruption_rejected_without_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f';p.write_bytes(b'a')
            with patch('numpy.load',side_effect=AssertionError('REGISTERED_ARRAY_DECODE')):
                with self.assertRaisesRegex(ValueError,'IDENTITY_'):gate.check_files(Path(d),{'f':{'SHA256':'0'*64,'bytes':1}})

if __name__=='__main__':unittest.main()
