"""S0 artificial scalar/gate tests. No real saved-value evaluation here."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_pr_rpe_rmse_s0 as s0
from demo_pr_rpe_rmse import artificial_design
A=s0.A;p=s0.p
MOCK=dict(kind='SYNTHETIC_ONLY')


def args():
    design=artificial_design();design.update(system='H4',PF=dict(method='B0',prefix=7))
    design['binding']['system']='H4';design['profile']=dict(p.PROFILE,compiler='MOCK')
    tail=dict(lambda_R=.5,prefix=1,binding=deepcopy(design['binding']),source=MOCK)
    return s0.materialize(design,tail,MOCK)


def fixture():
    values,conditions=args();schedule=A.schedule(**values)
    req=next(r for r in schedule['rows'] if r['identity']['m']==0 and p.arm(r['identity']['candidate'])=='PR')
    c=req['identity']['candidate'];st=req['identity']['structure'];L=.03;attenuation=math.exp(-L)
    z=lambda value:dict(real=value,imag=0.)
    signal=dict(reference=z(1.),corrected=z(1.),raw=z(attenuation),exact_tail=z(1.),
        total_signed_bias=z(0.),outer_PF_or_discard_bias=z(0.),finite_RTE_bias=z(0.),log_B=L,
        allowance_evidence=dict(maximum_intermediate_norm=1.,raw_corrected_vector_closure=0.,exact_tail_norm_error=0.))
    stages=c['q']*(2*(c['prefix']+1)+1)+2*st['event_count']
    diag=dict(candidate=deepcopy(c),signal=signal,reference_solver_energy_ha=0.,state_residual_ha=0.,ground_state_certified=False,
        signed_weights=st['signed_weights'],allocation=st['allocation'],event_count=st['event_count'],
        allowance=dict(u=1e-9,log_B_margin=1e-12,stages=stages,policy='R_AWARE_EMPIRICAL_ACTION_COUNT_V1',certified=False),
        error_disk=dict(unconditional_ground_identification_error=None))
    saved=dict(identity=deepcopy(req['identity']),diagnostics=diag,plan=dict(status='CONDITIONAL_ROUND_PASS',N_X=999999999))
    ref=dict(T=c['T'],binding=deepcopy(req['identity']['binding']),source=MOCK,
             evidence=dict(reference=z(1.),independent_reference=z(1.),signal_difference=0.))
    return req,saved,ref,schedule


class S0Tests(unittest.TestCase):
    def test_unique_formula_materialization_and_saved_recipe(self):
        values,c=args();self.assertEqual(c['dimensionless_x_exact'],'255/128')
        self.assertEqual(Fraction(values['kappa_paper'])*Fraction(.5)**2*Fraction(.1)**2,Fraction(255,128))
        self.assertLessEqual(c['empirical_qpe_horizon_ha'],c['qpe_budget_ha'])
        self.assertGreater(.1*math.pi/math.ldexp(.1,c['M']-1),c['qpe_budget_ha'])
        schedule=A.schedule(**values)
        ids={r['identity']['m']:r['identity']['candidate']['id'] for r in schedule['rows'] if p.arm(r['identity']['candidate'])=='PR' and r['identity']['m'] in s0.IDS}
        self.assertEqual(ids,s0.IDS)

    def test_invalid_outer_or_lambda_not_optimized(self):
        values,_=args()
        for lam in (0.,-1.,float('nan')):
            tail=deepcopy(values['tail_norm']);tail['lambda_R']=lam
            with self.assertRaises(ValueError):s0.materialize(values['design'],tail,MOCK)
        design=deepcopy(values['design']);design['K']=4
        with self.assertRaises(ValueError):s0.materialize(design,values['tail_norm'],MOCK)

    def test_original_scalar_gates_and_unknown_actual_errors(self):
        req,old,ref,_=fixture();r=s0.numeric_diagnostics(req,old,ref)
        self.assertTrue(r['numerical_gates_passed']);self.assertFalse(r['rigorous_error_bound'])
        self.assertIsNone(r['actual_implementation']['raw_error'])
        self.assertIsNone(r['conditional_geometry']['N_X'])
        self.assertNotIn('N_X',r);self.assertNotIn('plan',r)
        self.assertEqual(old['plan']['N_X'],999999999)

    def test_reference_candidate_and_fail_are_not_cache_fallback(self):
        req,old,ref,_=fixture()
        for update in ('binding','pair','status'):
            a,b=deepcopy(old),deepcopy(ref)
            if update=='binding':b['binding']['state']='FOREIGN'
            elif update=='pair':b['evidence']['signal_difference']=1e-8
            else:a['plan']['status']='FAIL'
            with self.assertRaises(ValueError):s0.numeric_diagnostics(req,a,b)

    def test_norm_closure_allowance_and_bias_gates(self):
        req,old,ref,_=fixture()
        for field in ('norm','closure','allowance','bias','raw'):
            bad=deepcopy(old);d=bad['diagnostics']
            if field=='norm':d['signal']['allowance_evidence']['exact_tail_norm_error']=1e-8
            elif field=='closure':d['signal']['allowance_evidence']['raw_corrected_vector_closure']=1e-8
            elif field=='allowance':d['allowance']['u']=0.
            elif field=='bias':d['signal']['total_signed_bias']['real']=.01
            else:d['signal']['raw']['real']+=.01
            with self.assertRaises(ValueError):s0.numeric_diagnostics(req,bad,ref)

    def test_saved_mean_adaptation_and_scope_only_supplied_two(self):
        values,_=args();schedule=A.schedule(**values);requests=[r for r in schedule['rows'] if r['identity']['m'] in s0.IDS and p.arm(r['identity']['candidate'])=='PR']
        means=[]
        for request in requests:
            c=request['identity']['candidate'];z=dict(real=1.,imag=0.)
            record=dict(candidate=c,signal=dict(reference=z,corrected=z,raw=z,exact_tail=z,log_B=.0),
                        **{k:request['identity']['structure'][k] for k in ('signed_weights','allocation','event_count')})
            means.append(A.mean_evidence(request,record,MOCK))
        ev=A.evaluate(schedule,mean_provider=lambda reqs:means)
        selected=[r for r in ev['records'] if r['nominal_N_X'] is not None]
        self.assertEqual([r['identity']['m'] for r in selected],[0,6])
        self.assertTrue(all(r['sufficient_N_X'] is None and not r['rmse_guaranteed'] for r in ev['records']))
        self.assertTrue(all(r['status']=='UNQUALIFIED_MODEL_ASSUMPTIONS' for r in selected))

    def test_consumed_original_sources_remain_pinned(self):
        for path in s0.CONSUMED:
            self.assertEqual((ROOT/path).read_bytes(),s0.blob(ROOT,s0.BASE,path))
        self.assertFalse(any(k in sys.modules for k in ('numpy','scipy','qiskit')))


if __name__=='__main__':unittest.main()
