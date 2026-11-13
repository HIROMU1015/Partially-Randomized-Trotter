"""Artificial laws only: exact scalar history oracle, no MC or saved raw."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
from itertools import product
import math
import unittest
from unittest.mock import patch

from check_rpe_deterministic_risk import case, cost_fixture, D, S, K, B, a
from check_rpe_signal_statistics_caller import MOCK


def refined_case(M=1, vector=(2,2)):
    args=case(M=M,vectors=[vector])
    for mean in args['signal_packet']['means']:
        mean['raw']=dict(real=.25,imag=-.5)
    args['options']['state_refinement']=dict(source=MOCK,
        maximum_frontier_states=10000,maximum_transition_pairs=1000000)
    return args


def risk(args):
    return D.evaluate(**args)[0]['diagnostic']['deterministic_risk']


def scalar_history_oracle(args):
    """Independent comb/Fraction PMF and unchanged SCALAR decoder on tiny laws."""
    support=[]
    for m,(mean,shots) in enumerate(zip(args['signal_packet']['means'],args['allocations'][0]['shots'])):
        axes=[]
        for axis,component in (('X','real'),('Y','imag')):
            n=shots['N_'+axis]; p=(1+a.exact_scalar(mean['raw'][component]))/2
            axes.append([(k,Fraction(math.comb(n,k))*p**k*(1-p)**(n-k)) for k in range(n+1)])
        support.append([(dict(m=m,X=dict(plus=kx,shots=shots['N_X']),Y=dict(plus=ky,shots=shots['N_Y'])),px*py)
                        for (kx,px),(ky,py) in product(*axes) if px*py])
    masses={}; resolved=Fraction(0); total=Fraction(0)
    target=a.exact_scalar(args['evaluation_target']['energy_ha'])
    h=float(a.exact_scalar(args['config']['spec']['h']))
    for history in product(*support):
        probability=math.prod(v[1] for v in history); total+=probability
        trial=S.decode_counts([v[0] for v in history],h=h,M=len(history)-1,prior=[-100,100])
        status=trial['status']; masses[status]=masses.get(status,Fraction(0))+probability
        if trial['energy_estimate_ha'] is not None:
            resolved+=probability*(a.exact_scalar(trial['energy_estimate_ha'])-target)**2
    assert total==1
    unresolved=sum(v for k,v in masses.items() if k!=S.DECODED)
    loss=args['statistical_contract']['summary_options']['unresolved_loss']
    return dict(resolved=resolved,unresolved=unresolved,
        zero=masses.get('UNRESOLVED_ZERO_COORDINATES',Fraction(0)),
        tie=masses.get('UNRESOLVED_BRANCH_TIE',Fraction(0)),
        lower=resolved+unresolved*a.exact_scalar(loss['lower']),
        upper=resolved+unresolved*a.exact_scalar(loss['upper']))


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        for module,name in ((S,'simulate'),(S,'simulate_prepared'),(K,'diagnose'),(D.p,'run_missing')):
            self.stack.enter_context(patch.object(module,name,side_effect=AssertionError('MC/physical provider forbidden')))

    def test_01_complete_DP_equals_exact_scalar_count_histories(self):
        for vector in ((1,1),(2,1),(2,2),(1,3),(2,2,2)):
            args=refined_case(M=len(vector)-1,vector=vector)
            if vector==(1,3): args['allocations'][0]['shots'][0]['N_Y']=2
            result=risk(args); dp=result['state_refinement']; oracle=scalar_history_oracle(args)
            self.assertEqual(dp['status'],'COMPLETE_STATE_LOSS_FAMILY_BOUND')
            for field,key in (('MSE_upper','upper'),('MSE_lower','lower'),('resolved_MSE_contribution','resolved'),
                              ('unresolved_probability','unresolved'),('any_zero','zero'),('actual_global_tie_probability','tie')):
                self.assertEqual(a.exact_scalar(dp[field]),oracle[key],(vector,field))
            self.assertLessEqual(a.exact_scalar(result['MSE_upper']),a.exact_scalar(result['coarse_MSE_upper']))

    def test_02_strict_improvement_when_unsafe_histories_still_decode(self):
        args=refined_case(vector=(1,1))
        for mean in args['signal_packet']['means']:mean['raw']=dict(real=0.,imag=0.)
        r=risk(args); oracle=scalar_history_oracle(args)
        self.assertEqual(oracle['unresolved'],0)
        self.assertLess(a.exact_scalar(r['MSE_upper']),a.exact_scalar(r['coarse_MSE_upper']))
        self.assertEqual(a.exact_scalar(r['MSE_upper']),oracle['upper'])

    def test_03_zero_and_tie_absorption_preserves_later_zero_precedence(self):
        args=refined_case(M=2,vector=(2,2,2))
        for mean in args['signal_packet']['means']:mean['raw']=dict(real=0.,imag=0.)
        oracle=scalar_history_oracle(args); dp=risk(args)['state_refinement']
        self.assertGreater(oracle['zero'],0); self.assertGreater(oracle['tie'],0)
        self.assertEqual(a.exact_scalar(dp['actual_global_tie_probability']),oracle['tie'])
        self.assertEqual(a.exact_scalar(dp['unresolved_probability']),oracle['tie']+oracle['zero'])

    def test_04_partial_frontiers_and_pair_scope_keep_a_sound_residual_bound(self):
        args=refined_case(M=2,vector=(2,2,2)); oracle=scalar_history_oracle(args)
        for states,pairs in ((0,0),(0,1000),(1,1000000),(1000,2),(2,20)):
            args['options']['state_refinement'].update(maximum_frontier_states=states,maximum_transition_pairs=pairs)
            r=risk(args); dp=r['state_refinement']; bounds=dp['unresolved_probability_bounds']
            self.assertEqual(dp['status'],'PARTIAL_STATE_BOUND_WITH_RESIDUAL')
            self.assertGreater(a.exact_scalar(dp['residual_probability']),0)
            self.assertLessEqual(a.exact_scalar(dp['MSE_lower']),oracle['lower'])
            self.assertGreaterEqual(a.exact_scalar(dp['MSE_upper']),oracle['upper'])
            self.assertGreaterEqual(a.exact_scalar(r['MSE_upper']),oracle['upper'])
            self.assertLessEqual(a.exact_scalar(r['MSE_upper']),a.exact_scalar(r['coarse_MSE_upper']))
            self.assertLessEqual(a.exact_scalar(bounds['lower']),oracle['unresolved'])
            self.assertGreaterEqual(a.exact_scalar(bounds['upper']),oracle['unresolved'])
            self.assertLessEqual(dp['transition_pairs'],pairs)

    def test_05_alias_loss_is_not_removed_or_clipped_by_prior(self):
        args=refined_case(vector=(1,1)); args['evaluation_target']['energy_ha']=2*math.pi/10
        r=risk(args); oracle=scalar_history_oracle(args)
        self.assertEqual(a.exact_scalar(r['state_refinement']['MSE_upper']),oracle['upper'])
        self.assertGreater(oracle['resolved'],Fraction(1,10))

    def test_06_unresolved_loss_interval_is_unchanged_and_nonzero_lower_supported(self):
        args=refined_case(); loss=args['statistical_contract']['summary_options']['unresolved_loss']
        loss.update(lower=3,upper=7)
        dp=risk(args)['state_refinement']; oracle=scalar_history_oracle(args)
        self.assertEqual(a.exact_scalar(dp['MSE_lower']),oracle['lower'])
        self.assertEqual(a.exact_scalar(dp['MSE_upper']),oracle['upper'])

    def test_07_default_path_unchanged_and_unknown_contracts_not_bypassed(self):
        args=refined_case(); legacy=deepcopy(args); legacy['options'].pop('state_refinement')
        r=risk(args); old=risk(legacy)
        self.assertEqual(r['coarse_MSE_upper'],old['MSE_upper'])
        self.assertNotIn('state_refinement',old)
        for change in ('loss','closure','domain'):
            bad=deepcopy(args)
            if change=='loss':bad['options']['model_loss_contract']['adopted']=False
            if change=='closure':bad['options']['phase_radii']=[2.6]*2
            if change=='domain':bad['options']['maximum_round_outcomes']=1
            with patch.object(D,'_refine_states',side_effect=AssertionError('invalid baseline refined')):
                self.assertEqual(risk(bad)['status'],'UNKNOWN')

    def test_08_explicit_scopes_and_unprovable_transitions_rejected(self):
        for scope in ({},dict(source=MOCK,maximum_frontier_states=-1,maximum_transition_pairs=10),
                      dict(source=MOCK,maximum_frontier_states=True,maximum_transition_pairs=10)):
            args=refined_case();args['options']['state_refinement']=scope
            with self.assertRaises(ValueError):risk(args)
        args=refined_case()
        original=D._refine_states
        def invalid(tables,**kwargs):
            copied=deepcopy(tables);copied[0]['outcomes'][0]['weight']+=1
            return original(copied,**kwargs)
        with patch.object(D,'_refine_states',side_effect=invalid):
            r=risk(args)
        self.assertEqual(r['state_refinement']['status'],'UNKNOWN')
        self.assertEqual(r['state_refinement']['reason'],'COUNT_MASS_NOT_CONSERVED')
        self.assertEqual(r['status'],'UNKNOWN'); self.assertIsNone(r['MSE_upper'])
        self.assertIsNotNone(r['coarse_MSE_upper'])
        import numpy as np
        def nonfinite(tables,**kwargs):
            def invalid_transition(phases,estimate,q):
                values=np.full(len(phases),np.nan)
                return values,values,values,values,values
            with patch.object(D.F,'_nearest',side_effect=invalid_transition):
                return original(tables,**kwargs)
        with patch.object(D,'_refine_states',side_effect=nonfinite):
            r=risk(args)
        self.assertEqual(r['state_refinement']['reason'],'NONFINITE_ORIGINAL_DECODER_TRANSITION')
        self.assertEqual(r['status'],'UNKNOWN'); self.assertIsNone(r['MSE_upper'])

    def test_09_binary64_boundary_and_signed_zero_state_keys(self):
        # Tiny-support laws include atan2(+0,-1)=pi and principal alias boundary.
        for target in (0.,math.nextafter(math.pi/10,-math.inf),math.nextafter(math.pi/10,math.inf)):
            args=refined_case(M=0,vector=(2,))
            args['signal_packet']['means'][0]['raw']=dict(real=-1.,imag=0.)
            args['evaluation_target']['energy_ha']=target
            self.assertEqual(a.exact_scalar(risk(args)['state_refinement']['MSE_upper']),scalar_history_oracle(args)['upper'])
        table=dict(status='ENUMERATED',m=0,q=1,denominator=2,
            outcomes=[dict(principal=v,zero=False,weight=1) for v in (-0.,0.)])
        dp=D._refine_states([table],h=1.,target=0.,lo=Fraction(0),hi=Fraction(1),
            resolved_limit=Fraction(10),scope=dict(source=MOCK,maximum_frontier_states=2,maximum_transition_pairs=2))
        self.assertEqual(dp['rounds'][0]['grouped_count_outcomes'],2)
        self.assertEqual(dp['rounds'][0]['retained_states'],2)

    def test_10_full_13round_vector_bindings_and_scientific_nulls_preserved(self):
        args=refined_case(M=12,vector=(1,)*13)
        for mean in args['signal_packet']['means']:mean['raw']=dict(real=1.,imag=0.)
        args['allocations'][0]['shots'][5]['N_Y']=2
        before=deepcopy(args)
        with patch.object(D,'binomial_axis',wraps=D.binomial_axis) as pmf:
            record=D.evaluate(**args)[0]
        self.assertEqual(pmf.call_count,26)  # same tables serve coarse bound AND DP
        dp=record['diagnostic']['deterministic_risk']['state_refinement']
        self.assertEqual(len(dp['rounds']),13); self.assertEqual(args,before)
        self.assertEqual(record['allocation']['shots'][5]['N_Y'],2)
        self.assertEqual(record['diagnostic']['deterministic_risk']['scientific_energy_qualification'],'UNKNOWN')
        for key in ('n_packet','sufficient_N','global_RMSE_bound','G'):self.assertIsNone(record[key])

    def test_11_whole_vector_compare_uses_existing_cost_and_source_binding(self):
        args=case(vectors=[(1,5),(8,1)])
        args['options']['state_refinement']=dict(source=MOCK,maximum_frontier_states=1000,maximum_transition_pairs=100000)
        records=D.evaluate(**args); prices,qualifications,options=cost_fixture(args,records)
        out=D.compare(args['config'],records,prices,qualifications=qualifications,
            model_limit=dict(MSE_upper=1,source=MOCK,scope='MODEL_MSE_FILTER_ONLY'),**options)
        self.assertEqual(out['comparison']['point_minimum']['allocation_id'],'v1')
        qualifications['v0']['rmse_evidence']['deterministic_input']['options']['state_refinement']['source']={'fixture':'OTHER_SOURCE'}
        with self.assertRaises(ValueError):
            D.compare(args['config'],records,prices,qualifications=qualifications,
                model_limit=dict(MSE_upper=1,source=MOCK,scope='MODEL_MSE_FILTER_ONLY'),**options)

    def test_12_tiny_nonzero_count_mass_retained_in_integer_DP(self):
        args=refined_case(M=0,vector=(2000,))
        args['allocations'][0]['shots'][0]['N_Y']=1
        args['signal_packet']['means'][0]['raw']=dict(real=0.,imag=1.)
        r=risk(args); dp=r['state_refinement']
        self.assertEqual(dp['status'],'COMPLETE_STATE_LOSS_FAMILY_BOUND')
        self.assertEqual(a.exact_scalar(dp['residual_probability']),0)
        self.assertGreater(Fraction(1,2**2000),0)
        self.assertEqual(float(Fraction(1,2**2000)),0.)
        self.assertEqual(a.exact_scalar(dp['MSE_upper']),scalar_history_oracle(args)['upper'])


if __name__=='__main__':unittest.main()
