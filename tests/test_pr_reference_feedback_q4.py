"""Q4 artificial analytic/mocked fixtures only. No saved H4 numbers are evaluated."""
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_reference_feedback_q4 as g
from trottertracks import pr_reference_feedback_q4_certify as c
from trottertracks import pr_reference_feedback_q4_check as check


def band(lo,hi=None):
    lo,hi=F(lo),F(lo if hi is None else hi)
    return g.Confidence(lo,lo,hi,hi)


def toy(**changes):
    # Deliberately not the H4 saved rho/bias/failure/time/prefix conditions.
    model=c.Model(F(9,10),F(0),F(3,10),F(3,5),F(1,5),F(1),
                  F(0),F(0),F(0),F(1,100))
    return replace(model,**changes)


class Q4ArtificialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=g.Arithmetic()

    def test_common_radius_joint_not_independent_axis_projection(self):
        a=self.a
        # Each strip intersects the unit circle; their common point is inside
        # its hole. No common physical coherence satisfies BOTH exact strips.
        self.assertTrue(g.angular_region(a,[band(F(3,5)),band(-1,1)],F(1)))
        self.assertTrue(g.angular_region(a,[band(-1,1),band(F(3,5))],F(1)))
        self.assertEqual(g.angular_region(a,[band(F(3,5)),band(F(3,5))],F(1)),[])
        self.assertEqual(g.angular_region(a,[band(F(4,5)),band(F(4,5))],F(1,2)),[])

    def test_annulus_components_contact_and_period_cut(self):
        a=self.a
        regions=g.angular_region(a,[band(F(-1,5),F(1,5)),band(-1,1)],F(3,4))
        self.assertEqual(len(regions),2)
        for target in (F(1,4),F(3,4)):
            self.assertTrue(any(lo<=target<=hi for lo,hi in regions))
        tangent=g.angular_region(a,[band(1),band(-1,1)],F(1))
        self.assertTrue(any(lo<=0<=hi or lo<=1<=hi for lo,hi in tangent))
        quarter=g.angular_region(a,[band(0,1),band(0,1)],F(1,2))
        self.assertTrue(any(lo<=F(1,8)<=hi for lo,hi in quarter))
        self.assertFalse(any(lo<=F(1,2)<=hi for lo,hi in quarter))
        cut=g.residual_region(a,[band(-1),band(0)],F(1),F(0),F(3,4),(F(1,2),F(1,2)))
        self.assertTrue(any(lo<=0<=hi for lo,hi in cut))
        self.assertLess(g.width_of(cut),F(1,1000))

    def test_registered_native_axes_and_quadrant_signs(self):
        a=self.a
        x,y=F(3,5),F(4,5)
        native=a.iv.cos(a.r(g.NATIVE_A))*a.r(x)+a.iv.sin(a.r(g.NATIVE_A))*a.r(y)
        lo,hi=a.bounds(native)
        regions=g.angular_region(a,[band(x),band(lo,hi)],F(1),'native')
        expected=a.bounds(a.iv.atan2(a.r(y),a.r(x))/(2*a.iv.pi))
        self.assertTrue(any(l<=expected[0] and h>=expected[1] for l,h in regions))
        # The registered angle has a real nonzero cosine; do not substitute pi/2.
        self.assertGreater(a.bounds(a.iv.cos(a.r(g.NATIVE_A)))[0],0)
        turns=g.point_turns(a,a.r(-1),a.interval(F(-1,100),F(1,100)))
        self.assertLess(sum(h-l for l,h in turns),F(1,10))

    def test_unmeasured_empty_and_sine_inverse(self):
        a=self.a
        self.assertEqual(g.hoeffding(a,0,0,F(0)).outer,(F(-1),F(1)))
        self.assertEqual(g.clopper_pearson(a,0,0,F(0)).outer,(F(-1),F(1)))
        with self.assertRaises(ValueError):g.hoeffding(a,2,1,F(0))
        with self.assertRaises(ValueError):g.clopper_pearson(a,2,3,F(1,5))
        self.assertEqual(g.q_region(a,band(0),F(1,2),F(1,20),F(3,10)),[(F(-1,20),F(1,20))])

    def test_readout_source_algebra_on_artificial_two_dimensional_matrices(self):
        # Pure 2x2 algebra, not an H4 circuit/simulator/compiler call.
        import math
        import numpy as np
        X=np.array([[0,1],[1,0]],complex)
        Y=np.array([[0,-1j],[1j,0]],complex)
        Z=np.diag([1,-1])
        SX=np.array([[1+1j,1-1j],[1-1j,1+1j]])/2
        def rz(v):return np.diag([np.exp(-.5j*v),np.exp(.5j*v)])
        a=.7;delta=math.pi/2-a
        h=rz(a)@SX@rz(a)
        self.assertLess(np.linalg.norm(h.conj().T@Z@h-(math.sin(a)*X+math.cos(a)*Y)),2e-15)
        for phase in (0.,.4,-1.):
            readout=h@rz(-phase)
            expected=math.cos(delta+phase)*X+math.sin(delta+phase)*Y
            self.assertLess(np.linalg.norm(readout.conj().T@Z@readout-expected),2e-15)

    def test_binomial_tail_and_cp_root_enclosures(self):
        a=self.a
        for n in range(1,6):
            for k in range(n+1):
                for p in (F(0),F(1,3),F(1)):
                    lo,hi=a.bounds(g.binomial_survival(a,n,k,p))
                    exact=check.rational_tail(n,k,p)
                    self.assertLessEqual(lo,exact);self.assertLessEqual(exact,hi)
                ci=g.clopper_pearson(a,n,k,F(1,5))
                self.assertTrue(ci.resolved)
                if k:
                    check.root_bracket(n,k,F(1,10),[(ci.lower_lo+1)/2,(ci.lower_hi+1)/2])
                if k<n:
                    check.root_bracket(n,n-k,F(1,10),[(1-ci.upper_hi)/2,(1-ci.upper_lo)/2])
        ci=g.clopper_pearson(a,1,0,F(1,5))
        self.assertLessEqual(ci.upper_lo,F(4,5));self.assertGreaterEqual(ci.upper_hi,F(4,5))
        # Exact finite-distribution coverage at a rational artificial p; this
        # is a regression check, not a proof by finite grid for arbitrary p.
        n=4;p=F(1,3);failure=F(0)
        from math import comb
        for k in range(n+1):
            ci=g.clopper_pearson(a,n,k,F(1,5))
            if not ci.lower_lo<=2*p-1<=ci.upper_hi:
                failure+=comb(n,k)*p**k*(1-p)**(n-k)
        self.assertLessEqual(failure,F(1,5))

    def test_all_count_box_certificate_and_checker(self):
        a=self.a;model=toy()
        result=c.lookup_for_total(a,model,1,'H','native')
        self.assertEqual(result['status'],'PASS_IMPLEMENTED')
        self.assertTrue(check.total_result(a,model,result,'H','native'))
        points=c.select_policy(result['cells'],(F(49,100),F(51,100)),model.center_pad)
        self.assertEqual(points.nx+points.ny,1)
        c.select_policy(result['cells'],(F(-1,1000),F(1,1000)),model.center_pad)
        with self.assertRaises(ValueError):c.select_policy(result['cells'],(F(0),F(1,10)),model.center_pad)
        with self.assertRaises(ValueError):check.count_cover([(0,0,0,0)],1,1)
        with self.assertRaises(ValueError):check.count_cover([(0,1,0,1),(0,0,0,0)],1,1)
        normalized=c.center_turn_interval(a,replace(model,kappa=F(1,1000)),(F(1),F(1001,1000)))
        self.assertLessEqual(normalized[1]-normalized[0],2*model.center_pad)
        with self.assertRaises(ValueError):c.center_turn_interval(a,model,(F(1),F(1001,1000)))

    def test_continuous_failure_share_exclusion_uses_verified_pairs(self):
        a=self.a;model=toy(target=F(1,100))
        row=c.search_share(a,model,1,1,F(1,8),'H','ideal')
        self.assertEqual(row['status'],'FAIL_ALL_SHARES')
        self.assertTrue(check.share_fail(a,model,row,'H','ideal'))
        self.assertEqual(g.merge([tuple(map(F,e['share_box'])) for e in row['exclusions']]),[(F(0),F(1))])

    def test_restoration_and_matched_q_adapter(self):
        a=self.a;model=toy()
        self.assertEqual(c.certify_q(a,model,2)['status'],'PASS_IMPLEMENTED')
        charged=replace(model,kappa=F(1,100))
        self.assertNotEqual(c.certify_q(a,charged,2)['status'],'PASS_IMPLEMENTED')
        result=c.certify_counts(a,charged,c.Allocation(1,0,F(1)),(F(1,8),F(1,8)),'H','ideal')
        self.assertNotEqual(result['status'],'PASS_IMPLEMENTED')

    def test_uniform_budget_lower_bound_includes_smaller_actual_totals(self):
        a=self.a;model=toy(target=F(1,100))
        result=c.lookup_for_total(a,model,2,'H','ideal')
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(len(result['allocations']),5)  # 3 actual-N2 + 2 actual-N1
        self.assertTrue(check.total_result(a,model,result,'H','ideal'))
        result['allocations']=result['allocations'][:3]
        with self.assertRaises(ValueError):check.total_result(a,model,result,'H','ideal')

    def test_fraction_price_correlation_and_failure_budget(self):
        allocation=c.Allocation(2,3,F(2,5));total=F(1,5)
        self.assertEqual(sum(allocation.alphas(total)),total)
        prices={'X':[F(10),F(12)],'Y':[F(11),F(13)]}
        self.assertEqual(c.native_cost([F(100),F(110)],prices,[allocation]),[F(153),F(173)])
        self.assertEqual(c.correlated_saving(4,allocation,prices),[F(7),F(13)])
        with self.assertRaises(ValueError):c.Allocation(1,1,F(0))

    def test_preflight_never_calls_arithmetic_or_campaign(self):
        spec=importlib.util.spec_from_file_location('q4_artificial_entry',ROOT/'scripts/run_pr_reference_feedback_q4.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        mock_data={
            'artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json':dict(target=[17,131072,32,88],arms={'Bprime':{},'Reference_B1':{}}),
            'artifacts/pr_reference_q0/2026-11-09/run01/CHECKS.json':dict(status='Q0_POSTRUN_CHECKS_PASS'),
            'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/C0.json':dict(status='ALL_CONTEXT_NATIVE_TRANSFER_CLOSED_IN_FIXED_MODEL'),
            'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/D.json':dict(status='GUARANTEE_CLOSED_IN_FIXED_MODEL')}
        import json
        def mock_git(args,**kwargs):
            # Pure source reads are genuine, only artificial JSON is mocked.
            path=args[-1].split(':',1)[1]
            return json.dumps(mock_data[path]).encode()
        with patch.object(runner,'INPUT_PATHS',list(mock_data)),patch.object(runner,'SOURCE_PATHS',[]),\
             patch.object(runner.subprocess,'check_output',side_effect=mock_git),\
             patch.object(runner.g,'pinned_text',side_effect=lambda p:(ROOT/p).read_text()),\
             patch.object(runner.g,'Arithmetic',side_effect=AssertionError('scientific arithmetic called')):
            record=runner.preflight()
        self.assertEqual(record['scientific_campaigns_executed'],0)
        self.assertIn('PREPARATION_ONLY',record['status'])

    def test_saved_price_schema_adapter_with_only_mock_prices(self):
        spec=importlib.util.spec_from_file_location('q4_mock_price_entry',ROOT/'scripts/run_pr_reference_feedback_q4.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        metrics=('RZ','CX','size','depth');price_rows=[];solutions=[]
        for arm in ('Bprime','Reference_B1'):
            packed=lambda pair:{'exact':list(map(str,pair))}
            price_rows.extend([
                dict(arm=arm,m=0,XY_cost={m:packed([100,110]) for m in metrics}),
                dict(arm=arm,m=17,per_shot={axis:{m:packed(pair) for m in metrics}
                     for axis,pair in [('X',[10,12]),('Y',[11,13])]},
                     Q_per_shot={m:packed([11,13]) for m in metrics})])
            policy=dict(status='PASS_IMPLEMENTED',cells=[dict(allocation=dict(nx=2,ny=3,share='2/5'))])
            for stage in ('H','E'):
                solutions.append(dict(stage=stage,arm=arm,kind='XY',geometry='native',
                    result=dict(lower=4,feasible=5,results=[policy])))
            solutions.append(dict(stage='E',arm=arm,kind='Q',geometry='native',result=dict(lower=1,feasible=2)))
        with patch.object(runner,'saved',return_value=dict(rows=price_rows,qubits={'mock':1})):
            result=runner.price_connection(solutions)
        self.assertEqual(len(result['rows']),16)
        for row in result['rows']:
            self.assertTrue(row['shared_prefix_cancels'])
            self.assertTrue(row['same_mu_Y_used'])
            expected=('Q_BEATS_CERTIFIED_FAMILY_COST_LOWER_BOUND' if row['family']=='E' else
                      'BASELINE_FEASIBLE_COST_BEATS_Q_FAMILY_LOWER_BOUND')
            self.assertEqual(row['status'],expected)

    def test_driver_replays_artificial_certificates_without_optimizer(self):
        spec=importlib.util.spec_from_file_location('q4_mock_check_entry',ROOT/'scripts/run_pr_reference_feedback_q4.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        a=self.a;model=toy()
        xy=c.solve_xy(a,model,'H','ideal',2)
        q=c.solve_q(a,model,2)
        with patch.object(c,'solve_xy',side_effect=AssertionError('optimizer rerun')),\
             patch.object(c,'solve_q',side_effect=AssertionError('Q optimizer rerun')):
            self.assertTrue(runner.check_solution(a,model,xy,'H','ideal','XY'))
            self.assertTrue(runner.check_solution(a,model,q,'E','native','Q'))
        q['lower']+=1
        with self.assertRaises(ValueError):runner.check_solution(a,model,q,'E','native','Q')


if __name__=='__main__':unittest.main(verbosity=2)
