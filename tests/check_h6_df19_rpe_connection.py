"""H6 binding delta tests using artificial signals/prices and energy targets.

No molecular fixture, saved numerical signal or scientific runner is read.
The original 15 H4 tests are reused separately, without copying their logic.
"""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h6_df19_rpe_connection as api


def point(later=False):
    identity = '2d2af6ca830dff14a695' if later else '2606efdab20e3b8522e5'
    t, method, prefix, r, k = api.D1_CANDIDATES[identity]
    c = dict(id=identity, system='H6', T=t, delta=t, method=method,
             prefix=prefix, R_outer=r, K=k, q=1, formula='4th(m5_best)')
    compiler = dict(id='TOY_COMPILER')
    prices = [dict(candidate=deepcopy(c), binding=deepcopy(api.BINDING),
        compiler=compiler, scope=api.common.SCOPE,
        classification='DIRECT_COMPILED_FINITE_SAMPLE', cohort=cohort,
        n=n, source='TOY_PRICE_'+cohort,
        mean_metrics={a:{g:cost for g in api.common.existing.METRICS}
                      for a,cost in [('cosine',2.),('sine',3.)]})
        for cohort,n in [('TOY_EXP',9),('TOY_CONFIRM',33)]]
    return dict(candidate=c, binding=deepcopy(api.BINDING), compiler=compiler,
        prices=prices, saved_signal=dict(log_B=math.log(2.), raw=None,
            corrected=dict(real=.1,imag=.2),reference=None))


def conditions(**change):
    # Artificial energy precision/failure/prior. Not the H6 study target.
    values=dict(base_time=.8, energy_ha=.1, alpha_total=.04,
        beta_total=.5, beta_stat=.25, energy_prior=(-1.,1.),
        prior_provenance='TOY external interval, not fitted from E0')
    values.update(change)
    return api.prepare_conditions(**values)


def assess(later=False, cond=None, **change):
    cond=conditions() if cond is None else cond
    m=(4 if later else 3) if cond['base_time']==.1 else int(later)
    row=api.bind_saved_round(point(later),target_id='TOY_TARGET',m=m,
                             base_time=cond['base_time'])
    args=dict(corrected_signal_disk=.01, normalization_error_raw=.001,
        implementation_error_raw=0.,law_error_raw=0.,
        reference_provenance='TOY reference/state disk',
        iid_provenance='TOY independent trials',error_provenance='TOY numerical allowance')
    args.update(change)
    return row,api.assess_supplied_round(row,conditions=cond,
                law_model='ideal_finite_RTE_law',**args)


class H6ConnectionTests(unittest.TestCase):
    def test_passive_import_and_input_inspection_without_numeric_calls(self):
        subprocess.run([sys.executable,'-B','-c',
            'import sys; from trottertracks import h6_df19_rpe_connection; '
            'assert not any(x in sys.modules for x in ["numpy","scipy","qiskit"])'],
            check=True,env=__import__('os').environ|{'PYTHONPATH':str(ROOT/'src')})
        p=point();before=deepcopy(p)
        with patch('math.exp',side_effect=AssertionError('No signal arithmetic')):
            out=api.inspect_saved_point(p)
        self.assertEqual(p,before);self.assertIsNone(out['N_X'])

    def test_h4_default_guard_stays_and_h6_full_binding_is_required(self):
        with self.assertRaises(ValueError):api.common.inspect_saved_point(point())
        for key,value in [('rank',10),('system','H4'),('state','FOREIGN_STATE'),
                          ('hamiltonian','FOREIGN_H'),('snapshot','FOREIGN_INPUT')]:
            p=point();p['binding'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):api.inspect_saved_point(p)

    def test_fixed_four_candidate_scope_and_round_dependent_pr_split(self):
        for later in (False,True):
            out=api.inspect_saved_point(point(later))
            self.assertEqual(out['candidate']['prefix'],7 if later else 5)
            p=point(later);p['candidate']['R_outer']=99
            with self.assertRaises(ValueError):api.inspect_saved_point(p)

    def test_q_pf_and_rpe_mapping_does_not_fill_missing_early_rounds(self):
        for later in (False,True):
            r=api.bind_saved_round(point(later),target_id='TOY',m=4 if later else 3,base_time=.1)
            self.assertEqual(r['q_PF'],1);self.assertEqual(r['q_RPE'],16 if later else 8)
        with self.assertRaises(ValueError):
            api.bind_saved_round(point(),target_id='TOY',m=0,base_time=.1)

    def test_unknown_prior_and_alias_failure_do_not_call_horizon_or_change_h(self):
        with patch.object(api.common.existing,'_legacy',side_effect=AssertionError('No horizon')):
            c=conditions(energy_prior=None)
            self.assertEqual(c['status'],'UNRESOLVED_ENERGY_PRIOR')
            self.assertIsNone(c['horizon'])
            bad=conditions(energy_prior=(-10.,10.))
            self.assertEqual(bad['status'],'ALIAS_WINDOW_EXCEEDED')
            self.assertEqual(bad['base_time'],.8)
        _,out=assess(cond=c)
        self.assertIsNone(out['N_X'])

    def test_uniform_all_axis_allocation_uses_existing_horizon(self):
        c=conditions();m=c['horizon']['maximum_round_index_M']
        self.assertTrue(c['horizon']['minimal_power_of_two_horizon'])
        self.assertLessEqual(math.fsum([c['alpha_axis']]*(2*(m+1))),c['alpha_total'])
        self.assertAlmostEqual(c['alpha_axis'],.04/(2*(m+1)))
        self.assertFalse(c['full_rpe_energy_success'])

    def test_unknown_reference_error_remains_unqualified_and_df_is_not_added(self):
        _,r=assess(corrected_signal_disk=None)
        self.assertEqual(r['status'],'UNRESOLVED_REFERENCE_OR_ERROR')
        self.assertIsNone(r['N_X'])
        _,r=assess()
        self.assertEqual(r['df_accounting'],'own_DF_target_no_DF_energy_shift')
        self.assertFalse(r['full_rpe_energy_success'])

    def test_single_cohort_selection_and_partial_cost_keep_uncertainty_unknown(self):
        c=conditions();plans=[];prices=[]
        for later in (False,True):
            row,plan=assess(later,cond=c);plans.append(plan)
            prices.extend(api.supplied_price_axes(row,cohort='TOY_CONFIRM'))
        self.assertTrue(all(p['n']==33 and p['cohort']=='TOY_CONFIRM' for p in prices))
        out=api.aggregate_partial(plans,prices,conditions=c,compiler=point()['compiler'])
        self.assertEqual(out['status'],'SUPPLIED_AXIS_COST_SUM')
        self.assertTrue(out['partial_only']);self.assertFalse(out['full_horizon_evaluated'])
        self.assertIsNone(out['total_confidence_interval'])
        with self.assertRaises(ValueError):
            api.supplied_price_axes(row,cohort='POOL_EXP_AND_CONFIRM')

    def test_h_point_one_partial_cost_cannot_drop_rounds_zero_to_two(self):
        c=conditions(base_time=.1)
        plans=[];prices=[]
        for later in (False,True):
            row,plan=assess(later,cond=c);plans.append(plan)
            prices.extend(api.supplied_price_axes(row,cohort='TOY_CONFIRM'))
        with self.assertRaises(ValueError):
            api.common.aggregate_supplied_axes(plans,prices,expected_rounds=(3,4),
                alpha_total=c['alpha_total'],compiler=point()['compiler'])
        out=api.aggregate_partial(plans,prices,conditions=c,compiler=point()['compiler'])
        self.assertEqual(out['status'],'SUPPLIED_AXIS_COST_SUM')
        self.assertEqual(out['included_rounds'],(3,4))
        self.assertTrue({0,1,2}.issubset(out['missing_rounds']))
        self.assertFalse(out['full_horizon_evaluated'])

    def test_partial_cost_cannot_mix_pf_pr_or_energy_targets(self):
        c=conditions();_,a=assess(cond=c);_,b=assess(True,cond=c)
        b['candidate']=dict(point(True)['candidate'],id='a4f5ffc0f320644dadfa',
                            method='FULL',prefix=19,R_outer=None,K=None)
        with self.assertRaises(ValueError):
            api.aggregate_partial([a,b],[],conditions=c,compiler=point()['compiler'])
        _,b=assess(True,cond=c);b['energy_ha']=.2
        with self.assertRaises(ValueError):
            api.aggregate_partial([a,b],[],conditions=c,compiler=point()['compiler'])


if __name__=='__main__':unittest.main()
