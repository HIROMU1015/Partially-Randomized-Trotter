"""Synthetic distributions and D metadata only. No H4 decoder/statistical run."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import rpe_decoder_statistics as S
from trottertracks import rpe_d_evidence_intake as D
from check_pr_rpe_rmse_controller import spec

B,p=S.B,S.p
MOCK={'fixture':'SYNTHETIC_ONLY'}
POLICY=dict(kind=S.IID,source=MOCK,target_law='ideal_finite_RTE_law')
UNDEFINED=dict(kind='UNDEFINED',source=MOCK)


def count(x,y,m=0):
    return dict(m=m,X=dict(plus=x[0],shots=x[1]),Y=dict(plus=y[0],shots=y[1]))


def toy(M=2, qualification='UNQUALIFIED'):
    design=spec(M=M);design['rmse_policy']['error_contract'].update(decoder=S.DECODER,energy_prior_ha=[-1,1])
    config=B.generate(design)
    target=dict(quantity='ORIGINAL_DF_H_TARGET_ENERGY',binding=deepcopy(config['spec']['binding']),
        source=MOCK,energy_shift_ha=0,energy_ha=-.2,uncertainty_ha=None)
    means=[];shots=[]
    for r in config['rounds']:
        t=float(B.a.exact_scalar(r['T']));phase=.2*t
        means.append(dict(m=r['m'],identity=deepcopy(r['identity']),law_binding=deepcopy(r['law_binding']),source=MOCK,
            raw=dict(real=.5*math.cos(phase),imag=.5*math.sin(phase)),corrected=dict(real=math.cos(phase),imag=math.sin(phase)),
            normalization=dict(log_B=math.log(2),kind='MOCK_SUPPLIED_FINITE_LOG_B'),target=deepcopy(target),
            qualification=qualification,bias={'finite_K':None},uncertainty={'law_error':None,'numerical_error':None}))
        shots.append(dict(m=r['m'],N_X=3,N_Y=5))
    return config,means,shots,target


class DecoderStatisticsTests(unittest.TestCase):
    def test_01_passive_import(self):
        subprocess.run([sys.executable,'-B','-c','import sys;from trottertracks import rpe_decoder_statistics;assert not any(k in sys.modules for k in ("numpy","scipy","qiskit","pyscf"))'],
            cwd=ROOT,env=__import__('os').environ|{'PYTHONPATH':str(ROOT/'src')},check=True)

    def test_02_counts_outcomes_unequal_shots_and_large_integer_precision(self):
        self.assertEqual(S.outcomes_count([-1,1,-1]),S.count_mean(1,3))
        self.assertEqual(B.a.exact_scalar(S.count_mean(1,3)['exact_mean']),Fraction(-1,3))
        N=2**200+1;result=S.count_mean((N+1)//2,N)
        self.assertEqual(result['shots'],N);self.assertEqual(B.a.exact_scalar(result['exact_mean']),Fraction(1,N))
        self.assertNotEqual(S.count_mean(1,3)['mean'],2.*1/3-1)
        r=S.decode_counts([count((0,1),(1000000,1000001))],h=1,M=0,prior=[-4,4])
        self.assertEqual(r['counts'][0]['Y']['shots'],1000001)
        for bad in ((True,3),(-1,2),(3,2),(1,0),(1,3.)):
            with self.assertRaises(ValueError):S.count_mean(*bad)
        for bad in ([True],[],[0],[1,-1,.5]):
            with self.assertRaises(ValueError):S.outcomes_count(bad)

    def test_03_ideal_phase_uses_original_wrapper_directly(self):
        xy=[(math.cos(.2*(1<<m)),math.sin(.2*(1<<m))) for m in range(4)]
        decoded=S.A.decode_supplied_coordinates(xy,h=1)
        self.assertAlmostEqual(decoded['energy_estimate_ha'],-.2)
        rows=[count((4,5),(3,5)),count((3,5),(4,5),1)]
        expected=S.A.decode_supplied_coordinates([(S.count_mean(**r['X'])['mean'],S.count_mean(**r['Y'])['mean']) for r in rows],h=1)
        actual=S.decode_counts(rows,h=1,M=1,prior=[-1,1])
        for k in ('status','decoded','energy_estimate_ha'):self.assertEqual(actual[k],expected[k])

    def test_04_zero_and_branch_tie_are_unresolved_without_energy(self):
        zero=S.decode_counts([count((1,2),(1,2))],h=1,M=0,prior=[-2,2])
        tie=S.decode_counts([count((2,2),(1,2)),count((0,2),(1,2),1)],h=1,M=1,prior=[-2,2])
        self.assertEqual(zero['status'],'UNRESOLVED_ZERO_COORDINATES')
        self.assertEqual(tie['status'],'UNRESOLVED_BRANCH_TIE')
        for r in (zero,tie):self.assertIsNone(r['energy_estimate_ha'])
        summary=S.summarize([zero,tie],target_energy_ha=0,unresolved_loss=UNDEFINED)
        self.assertIsNone(summary['all_observed_trials_RMSE']);self.assertEqual(summary['unresolved_probability'],1)

    def test_05_alias_boundary_and_prior_endpoints_do_not_change_decoder(self):
        end=S.decode_counts([count((0,2),(1,2))],h=1,M=0,prior=[math.pi-1,math.pi])
        self.assertEqual(end['energy_estimate_ha'],math.pi);self.assertTrue(end['estimate_in_supplied_prior'])
        outside=S.decode_counts([count((0,2),(1,2))],h=1,M=0,prior=[-math.pi,-math.pi+1])
        self.assertEqual(outside['energy_estimate_ha'],end['energy_estimate_ha'])
        self.assertFalse(outside['estimate_in_supplied_prior']);self.assertFalse(outside['prior_applied_to_decoder'])
        trueE=4;xy=[(math.cos(-trueE*(1<<m)),math.sin(-trueE*(1<<m))) for m in range(3)]
        aliased=S.A.decode_supplied_coordinates(xy,h=1)
        self.assertAlmostEqual(aliased['energy_estimate_ha'],trueE-2*math.pi)
        self.assertFalse(aliased['prior_alias_qualified'])
        error=S.summarize([aliased],target_energy_ha=trueE,unresolved_loss=UNDEFINED)
        self.assertAlmostEqual(error['all_observed_trials_RMSE'],2*math.pi)  # No circular loss alias erasure.

    def test_06_missing_statistics_are_not_zero_filled_or_counted_as_decoder_failure(self):
        r=S.decode_counts([dict(m=0,X=dict(plus=1,shots=2))],h=1,M=1,prior=[-1,1])
        self.assertEqual(r['status'],'NOT_EVALUATED_MISSING_COUNTS');self.assertEqual(len(r['missing']),3)
        with self.assertRaises(ValueError):S.summarize([r],target_energy_ha=0,unresolved_loss=UNDEFINED)
        with self.assertRaises(ValueError):S.decode_counts([count((1,1),(1,1)),count((1,1),(1,1))],h=1,M=0,prior=[-1,1])

    def test_07_exact_simple_distribution_retains_unresolved_mass(self):
        rows=[dict(m=0,raw=dict(real=0.,imag=0.),N_X=2,N_Y=2)]
        exact=S.exact_counts(rows,h=1,M=0,prior=[-4,4],policy=POLICY,maximum_states=9)
        out=S.summarize(exact['trials'],target_energy_ha=0,unresolved_loss=UNDEFINED)
        self.assertAlmostEqual(out['unresolved_probability'],.25)
        self.assertAlmostEqual(out['resolved_MSE_contribution'],17*math.pi**2/64)
        self.assertAlmostEqual(out['conditional_on_decoded_RMSE'],math.sqrt(17*math.pi**2/48))
        self.assertIsNone(out['all_observed_trials_RMSE']);self.assertIsNone(out['all_trial_RMSE_interval'])
        interval=S.summarize(exact['trials'],target_energy_ha=0,unresolved_loss=dict(kind='BOUNDED_INTERVAL',lower=0,upper=math.pi**2,source=MOCK))
        self.assertAlmostEqual(interval['all_trial_MSE_interval'][1],33*math.pi**2/64)
        with self.assertRaises(ValueError):S.exact_counts(rows,h=1,M=0,prior=[-4,4],policy=POLICY,maximum_states=8)

    def test_08_Monte_Carlo_matches_analytic_artificial_distribution_with_finite_CI(self):
        halfwidth=math.pi**2/10
        n=S.trial_budget(loss_upper=math.pi**2,mse_half_width=halfwidth,confidence=.999)
        rows=[dict(m=0,raw=dict(real=0.,imag=0.),N_X=1,N_Y=1)]
        simulation=S.simulate(rows,h=1,M=0,prior=[-4,4],policy=POLICY,repetitions=n,seed=777)
        bounded=dict(kind='BOUNDED_INTERVAL',lower=0,upper=math.pi**2,source=MOCK)
        out=S.summarize(simulation['trials'],target_energy_ha=0,unresolved_loss=bounded,iid_repetitions=True,
            confidence=.999,loss_support=dict(upper=math.pi**2,source={'fixture':'Analytic principal range at h=1,target=0'}))
        exact=5*math.pi**2/16
        self.assertLess(abs(out['resolved_MSE_contribution']-exact),halfwidth)
        low,high=out['population_RMSE_CI']['MSE_interval'];self.assertLess(low,exact);self.assertGreater(high,exact)
        self.assertEqual(out['unresolved_count'],0);self.assertGreater(out['unresolved_probability_CI'][1],0)
        self.assertIsNone(out['sufficient_N']);self.assertFalse(out['population_RMSE_CI']['sufficient_quantum_N_qualification'])
        undefined=S.summarize(simulation['trials'],target_energy_ha=0,unresolved_loss=UNDEFINED,iid_repetitions=True,confidence=.95)
        self.assertIsNone(undefined['population_RMSE_CI'])

    def test_09_independence_raw_mean_range_and_joint_law_not_silently_assumed(self):
        rows=[dict(m=0,raw=dict(real=0.,imag=0.),N_X=1,N_Y=1)]
        with self.assertRaises(ValueError):S.simulate(rows,h=1,M=0,prior=[-4,4],policy={},repetitions=1,seed=1)
        wrong=deepcopy(rows);wrong[0]['raw']['real']=1.00000000001
        with self.assertRaises(ValueError):S.simulate(wrong,h=1,M=0,prior=[-4,4],policy=POLICY,repetitions=1,seed=1)
        wrong=deepcopy(rows);wrong[0]['N_X']=2**80
        with self.assertRaisesRegex(ValueError,'not rounded'):S.simulate(wrong,h=1,M=0,prior=[-4,4],policy=POLICY,repetitions=1,seed=1)

    def test_10_all_13_round_adapter_unqualified_or_missing_does_not_auto_execute(self):
        config,means,shots,target=toy(M=12)
        ready=S.prepare(config,means,shots,target=target,shot_policy=dict(meaning='NOMINAL',source=MOCK))
        self.assertTrue(ready['input_complete']);self.assertFalse(ready['physical_qualification_complete'])
        with patch.object(S,'simulate',side_effect=AssertionError('no unqualified simulation')):
            r=S.simulate_prepared(ready,repetitions=2,seed=4,policy=POLICY)
        self.assertIn('UNQUALIFIED',r['status']);self.assertIsNone(r['RMSE'])
        missing=S.prepare(config,means[:3],shots[:3],target=target,shot_policy=dict(meaning='NOMINAL',source=MOCK))
        self.assertEqual(len(missing['rows']),13)
        with patch.object(S,'simulate',side_effect=AssertionError('no missing simulation')):
            self.assertIn('MISSING',S.simulate_prepared(missing,repetitions=1,seed=1,policy=POLICY)['status'])
        with patch.object(S.A,'nominal_axis_shots',side_effect=AssertionError('no N')),patch.object(B,'cost',side_effect=AssertionError('no G')):
            sim=S.simulate_prepared(ready,repetitions=1,seed=4,policy=POLICY,allow_conditional_unqualified=True)
        self.assertEqual(len(sim['trials']),1);self.assertIsNone(sim['physical_error_evidence'][0]['bias']['finite_K'])
        self.assertIsNone(sim['global_RMSE_bound']);self.assertIsNone(sim['G'])

    def test_11_identity_law_SOURCE_target_and_counts_are_not_relabelled(self):
        config,means,shots,target=toy()
        for field in ('identity','law_binding','target','source'):
            bad=deepcopy(means);bad[0][field]=None
            with self.assertRaises(ValueError):S.prepare(config,bad,shots,target=target,shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK))
        shifted=deepcopy(target);shifted['energy_shift_ha']=.1
        with self.assertRaises(ValueError):S.prepare(config,means,shots,target=shifted,shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK))
        shifted=deepcopy(target);shifted['quantity']='EFFECTIVE_PF_ENERGY'
        with self.assertRaises(ValueError):S.prepare(config,means,shots,target=shifted,shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK))

    def test_12_CI_loss_range_missing_failure_policy_and_bad_probability_rejected(self):
        decoded=dict(status=S.DECODED,energy_estimate_ha=1.)
        failed=dict(status='UNRESOLVED_BRANCH_TIE',energy_estimate_ha=None)
        samples=[decoded,failed];bounded=dict(kind='BOUNDED_INTERVAL',lower=4.,upper=9.,source=MOCK)
        out=S.summarize(samples,target_energy_ha=0,unresolved_loss=bounded)
        self.assertEqual(out['all_trial_MSE_interval'],[2.5,5.]);self.assertEqual(out['conditional_on_decoded_RMSE'],1.)
        self.assertIsNone(out['all_observed_trials_RMSE'])
        with self.assertRaises(ValueError):S.summarize(samples,target_energy_ha=0,unresolved_loss=bounded,iid_repetitions=False,confidence=.95)
        with self.assertRaises(ValueError):S.summarize(samples,target_energy_ha=0,unresolved_loss={})
        with self.assertRaises(ValueError):S.summarize([dict(decoded,probability=.6)],target_energy_ha=0,unresolved_loss=UNDEFINED)
        with self.assertRaises(ValueError):S.summarize(samples,target_energy_ha=0,unresolved_loss=bounded,iid_repetitions=True,confidence=.95,loss_support=dict(upper=1,source=MOCK))
        self.assertEqual(S.trial_budget(loss_upper=1e-100,mse_half_width=1e100,confidence=.95),1)
        with self.assertRaises(ValueError):S.trial_budget(loss_upper=1e300,mse_half_width=1e-300,confidence=.95)

    def test_15_existing_A_physical_adapter_copies_normalization_but_samples_only_raw(self):
        from check_pr_rpe_rmse_integration import inputs,I,mock_provider
        args=inputs(M=1);schedule=I.A.schedule(**args)
        config=B.generate(I.a_spec(args,'PR'));ev=I.A.evaluate(schedule,mean_provider=mock_provider)
        record=next(r for r in ev['records'] if r['identity']['candidate']['method']=='PR')
        physical=config['rounds'][record['identity']['m']]
        target=dict(binding=deepcopy(config['spec']['binding']),quantity='ORIGINAL_DF_H_TARGET_ENERGY',energy_shift_ha=0,source=MOCK)
        kw=dict(source_request_id=record['request_id'],law_binding=physical['round_law_definition']['A_declaration'],
            source=MOCK,target=target,qualification='UNQUALIFIED',uncertainty={'numerical':None,'law':None})
        with patch.object(S.A,'existing_finite_mean',side_effect=AssertionError('no new mean')):
            copied=S.mean_from_a(config,record['identity']['m'],record,**kw)
        self.assertEqual(copied['raw'],record['signal']['raw'])
        self.assertEqual(copied['normalization']['log_B'],record['signal']['log_B'])
        bad=deepcopy(record);bad['request_id']='RELABELLED'
        with self.assertRaises(ValueError):S.mean_from_a(config,record['identity']['m'],bad,**kw)
        # Corrected values >1 are not Bernoulli probabilities; raw is used once.
        r=[dict(m=0,raw=dict(real=.6,imag=.8),corrected=dict(real=100.,imag=100.),N_X=3,N_Y=5)]
        out=S.simulate(r,h=1,M=0,prior=[-2,2],repetitions=1,seed=0,policy=POLICY)
        original=deepcopy(r);original[0].pop('corrected')
        same=S.simulate(original,h=1,M=0,prior=[-2,2],repetitions=1,seed=0,policy=POLICY)
        self.assertEqual(out['trials'],same['trials'])

    def test_16_metadata_cli_does_not_send_real_means_to_statistics(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import prepare_rpe_decoder_statistics as CLI
        with patch.object(S,'simulate',side_effect=AssertionError('no H4 simulation')), \
             patch.object(S,'exact_counts',side_effect=AssertionError('no H4 enumeration')), \
             patch.object(S.A,'decode_supplied_coordinates',side_effect=AssertionError('no real decoding')), \
             patch.object(S.A,'evaluate',side_effect=AssertionError('no real N model')), \
             patch.object(B,'cost',side_effect=AssertionError('no G')):
            prepared=CLI.build()
        self.assertFalse(prepared['PREPARATION']['statistics_executed'])
        self.assertFalse(prepared['PREPARATION']['mean_values_imported_for_statistics'])
        self.assertEqual(prepared['PREPARATION']['nominal_rounds'],[0,1,6])
        self.assertEqual(len(prepared['PREPARATION']['rows']),13)
        self.assertIsNone(prepared['PREPARATION']['empirical_H4_RMSE'])


class ReceiptCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store=p.SavedArtifacts(ROOT)
        cls.previous=store.read(dict(commit=D.COVERAGE,path=D.COVERAGE_JSON))
        cls.current=store.read(dict(commit=D.RECEIPT_RELEASE,path=D.RECEIPT_DIR+'COVERAGE_V2.json'))
        cls.receipts=store.read(dict(commit=D.RECEIPT_RELEASE,path=D.RECEIPT_DIR+'RECEIPTS.json'))

    def test_13_latest_73_0_63_preserves_historical_56_17_63_and_no_menu(self):
        out=D.accept_receipt_coverage(self.previous,self.current,self.receipts)
        self.assertEqual(out['totals']['X'],{'EXACT_METADATA_MATCH':73,'MISSING_IN_CHECKED_SCOPE':63})
        self.assertEqual(out['previous_totals']['X']['CONDITIONAL_REUSE'],17)
        self.assertEqual(len(out['promoted_physical_ids']),17)
        self.assertFalse(out['candidates_adopted_as_menu']);self.assertFalse(out['current_RMSE_qualification']);self.assertIsNone(out['G'])

    def test_14_receipt_source_values_axis_and_FORMULA_HISTORY_rejected(self):
        for mutation in ('value','source','axis','formula','numerical','signal_total'):
            now=deepcopy(self.current);receipts=deepcopy(self.receipts)
            row=next(r for r in now['rows'] if r['prices'])
            if mutation=='value':row['prices'][0]['axes']['X']['rz_count']+=1
            elif mutation=='source':receipts['source_checks']['status']='UNKNOWN'
            elif mutation=='axis':now['rows'][0]['X']='MISSING_IN_CHECKED_SCOPE'
            elif mutation=='formula':row['physical_formula_label']='RELABELED'
            elif mutation=='numerical':now['new_compile']=1
            else:now['totals']['signal']={'EXACT_METADATA_MATCH':136}
            with self.assertRaises((ValueError,KeyError)):D.accept_receipt_coverage(self.previous,now,receipts)


if __name__=='__main__':unittest.main(verbosity=2)
