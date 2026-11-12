"""Artificial scalars/matrices/mock evidence only; no molecular N reanalysis."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_rpe_rmse as model
from trottertracks import pr_r_dependent_n as old
from demo_pr_rpe_rmse import artificial_design,mock_provider


def policy(rule='finite_B_diagnostic',error='paper_quadrature',qpe='1'):
    return model.rmse_policy(target_rmse='2',qpe_rmse=qpe,evolution_rmse='1/10',
        error_rule=error,normalization_rule=rule,source=dict(kind='ARTIFICIAL_ONLY'))


def proposal(M=3,kappa='2',design=None,rule='finite_B_diagnostic',lam='1/2'):
    design=design or artificial_design()
    tail=dict(lambda_R=lam,prefix=design['prefix'],binding=design['binding'],source=dict(kind='ARTIFICIAL_ONLY'))
    return model.schedule(M=M,kappa_paper=kappa,design=design,tail_norm=tail,policy=policy(rule))


class RMSETests(unittest.TestCase):
    def test_passive_import_and_real_saved_metadata_never_load_numeric_backend(self):
        code="from pathlib import Path;from trottertracks import pr_rpe_rmse as m,pr_r_dependent_n as a;m.design_from_task(a.saved_task(Path.cwd(),'H6'));import sys;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit'))"
        env=__import__('os').environ.copy();env['PYTHONPATH']='src:scripts'
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)

    def test_exact_kappa_integer_boundary_and_minimum_outer_total(self):
        d=artificial_design();d.update(base_time=1.,delta=1.)
        a=proposal(M=0,kappa='1',design=d,lam='1')['rows'][1]
        b=proposal(M=0,kappa='10000000000000001/10000000000000000',design=d,lam='1')['rows'][1]
        self.assertEqual(a['identity']['candidate']['R_outer'],1)
        self.assertEqual(b['identity']['candidate']['R_outer'],2)
        s=proposal(M=8)
        for r in s['rows'][1::2]:
            i=r['identity'];mapping=r['mapping'];E=i['structure']['event_count']
            self.assertEqual(E,i['q_PF']*i['candidate']['R_outer'])
            self.assertGreaterEqual(E,Fraction(mapping['r_nominal']))
            self.assertFalse(mapping['abstract_rotation_count_is_compiled_RZ'])

    def test_shared_B_draft_mapper_and_packet_required_fields(self):
        payload=json.loads(subprocess.check_output(['git','show',model.B_INTERFACE['commit']+':'+model.B_INTERFACE['path']],cwd=ROOT))
        self.assertEqual(model.MAPPING,payload['kappa']['draft_mapping'])
        mapped=model.paper_event_mapper(kappa_paper='2',lambda_R='1',T='3',q_PF=4,signed_weights=[1.])
        self.assertEqual((mapped['r_target'],mapped['R_outer'],mapped['event_count']),(18,5,20))
        s=proposal();out=model.b_handoff(s,model.evaluate(s,mean_provider=mock_provider),{'kind':'ARTIFICIAL_ONLY'})
        self.assertEqual(len(out['packets']),2)
        self.assertNotEqual(out['packets'][0]['configuration_id'],out['packets'][1]['configuration_id'])
        for packet in out['packets']:
            self.assertTrue(set(payload['configuration_fields'])<=packet['spec'].keys())
            self.assertTrue(set(payload['N_packet_fields'])<=packet['N_packet'].keys())
            self.assertEqual(packet['N_packet']['policy']['family'],'PR_PAPER_RMSE')
            self.assertEqual(packet['N_packet']['status'],'UNQUALIFIED')
            for r in packet['N_packet']['rounds']:
                self.assertIsNone(r['N_X']);self.assertIsNotNone(r['nominal_N_X'])
                self.assertEqual(r['identity']['configuration_id'],packet['configuration_id'])

    def test_no_cross_kappa_schedule_mixing_even_when_candidates_round_same(self):
        a=proposal(kappa='2');b=proposal(kappa='3')
        mixed=deepcopy(a);mixed['rows'][1]=deepcopy(b['rows'][1])
        with self.assertRaises(ValueError):model.evaluate(mixed)
        ev=model.evaluate(a,mean_provider=mock_provider)
        ev['records'][1]['identity']=deepcopy(b['rows'][1]['identity'])
        with self.assertRaises(ValueError):model.b_handoff(a,ev,{'kind':'ARTIFICIAL_ONLY'})

    def test_B_PF_configuration_has_null_kappa_and_PR_schedule_keeps_one_kappa(self):
        packets=[]
        for k in ('2','3'):
            s=proposal(kappa=k);packets.append(model.b_handoff(s,model.evaluate(s,mean_provider=mock_provider),{'kind':'ARTIFICIAL_ONLY'})['packets'])
        self.assertEqual(packets[0][0]['configuration_id'],packets[1][0]['configuration_id'])
        self.assertNotEqual(packets[0][1]['configuration_id'],packets[1][1]['configuration_id'])
        for pf,pr in packets:
            self.assertIsNone(pf['spec']['kappa'])
            for r in pr['N_packet']['rounds']:self.assertEqual(r['identity']['configuration']['kappa'],pr['spec']['kappa'])

    def test_h6_early_rules_and_signed_minimum_use_existing_allocation(self):
        base=old.saved_task(ROOT,'H6');d=model.design_from_task(base);s=proposal(M=5,design=d,kappa='1/100')
        for r,original in zip(s['rows'][:8],base['candidates'][:8]):
            self.assertEqual(r['identity']['candidate'],original['candidate'])
            self.assertFalse(r['mapping']['kappa_applied']);self.assertIsNone(r['mapping']['r_nominal'])
        r=s['rows'][9];self.assertEqual(r['identity']['candidate']['R_outer'],11)
        self.assertEqual(r['identity']['structure']['allocation'],[1]*11)
        self.assertTrue(r['mapping']['negative_signed_pieces'])
        self.assertFalse(r['mapping']['paper_e1_over_kappa_envelope_assumed'])
        d=deepcopy(d);d['early_rounds']=[]
        with self.assertRaises(ValueError):proposal(M=0,design=d)

    def test_M_round_count_qRPE_qPF_and_unknown_future_horizon(self):
        d=model.design_from_task(old.saved_task(ROOT,'H6'));s=proposal(M=13,design=d)
        self.assertEqual(len(s['rows']),28)
        self.assertEqual(s['rows'][-1]['identity']['q_RPE'],8192)
        self.assertEqual(s['rows'][-1]['identity']['q_PF'],512)
        self.assertEqual(s['task']['configuration']['round_count'],14)
        self.assertFalse(s['full_RPE_qualified'])

    def test_invalid_inputs_and_nonintegral_PF_grid_are_rejected(self):
        for k in ('0','-1',True,float('nan')):
            with self.assertRaises((ValueError,TypeError)):proposal(kappa=k)
        for M in (-1,True,1.5):
            with self.assertRaises(ValueError):proposal(M=M)
        d=artificial_design();d['delta']=.3
        with self.assertRaises(ValueError):proposal(M=0,design=d)
        with self.assertRaises(ValueError):proposal(lam='0')
        with self.assertRaises(ValueError):model.rmse_policy(target_rmse=1,qpe_rmse=1,evolution_rmse=0,error_rule='UNKNOWN',normalization_rule='finite_B_diagnostic',source={'kind':'TOY'})

    def test_paper_Nm_per_axis_and_normalization_applied_once(self):
        n=model.nominal_axis_shots(3,1,0.)
        self.assertEqual((n['nominal_N_X'],n['nominal_N_Y']),(19,19))
        self.assertEqual(model.nominal_axis_shots(3,3,0.)['nominal_N_X'],11)
        n=model.nominal_axis_shots(3,1,math.log(2.))
        self.assertIn(n['nominal_N_X'],(76,77)) # Binary64 log input; stable decimal ceiling.
        self.assertEqual(n['nominal_N_X'],n['nominal_N_Y'])
        self.assertFalse(n['rigorous_sufficient_N'])
        huge=model.nominal_axis_shots(0,0,300.)['nominal_N_X']
        self.assertIs(type(huge),int);self.assertGreater(huge,2**53)
        self.assertIsNone(model.nominal_axis_shots(0,0,400.)['nominal_N_X'])

    def test_extreme_positive_kappa_does_not_fabricate_unit_normalization(self):
        s=proposal(M=0,kappa=Fraction(1,10**1000),rule='paper_kappa_envelope')
        r=model.evaluate(s,mean_provider=mock_provider)['records'][1]
        self.assertEqual(r['status'],'NUMERICAL_RANGE_UNRESOLVED');self.assertIsNone(r['nominal_N_X'])
        with self.assertRaises(ValueError):
            model.paper_event_mapper(kappa_paper=10**1000,lambda_R=1,T=1,q_PF=1,signed_weights=[1.])

    def test_envelope_and_finite_B_are_explicit_distinct_policies(self):
        a=proposal(rule='paper_kappa_envelope');b=proposal(rule='finite_B_diagnostic')
        self.assertNotEqual(a['rows'][1]['request_id'],b['rows'][1]['request_id'])
        r=model.evaluate(a,mean_provider=mock_provider)['records'][1]
        self.assertEqual(r['normalization_log_used'],.5)
        self.assertEqual(model.evaluate(b,mean_provider=mock_provider)['records'][1]['normalization_log_used'],.05)
        def over(rows):
            ev=mock_provider(rows)
            for e in ev:
                if model.common.arm(e['identity']['candidate'])=='PR':e['signal']['log_B']=.6
            return ev
        r=model.evaluate(a,mean_provider=over)['records'][1]
        self.assertEqual(r['status'],'INAPPLICABLE_NORMALIZATION_ENVELOPE');self.assertIsNone(r['nominal_N_X'])

    def test_unknown_false_complete_evidence_are_not_scientific_PASS(self):
        s=proposal()
        self.assertTrue(all(r['status']=='NOT_EVALUATED' for r in model.evaluate(s)['records']))
        def supply(value):
            def provider(rows):
                out=mock_provider(rows)
                for e in out:
                    e.update(qualification={k:value for k in model.QUALIFICATIONS},qualification_source={'kind':'ARTIFICIAL_ONLY'})
                return out
            return provider
        for value,status in ((None,'UNQUALIFIED_MODEL_ASSUMPTIONS'),(False,'INAPPLICABLE_MODEL_ASSUMPTIONS'),(True,'EMPIRICAL_MODEL_INPUTS_REPORTED_COMPLETE')):
            out=model.evaluate(s,mean_provider=supply(value))
            for r in out['records']:
                self.assertEqual(r['status'],status);self.assertFalse(r['rmse_guaranteed'])
                self.assertIsNone(r['sufficient_N_X']);self.assertIsNone(r['scientific_PASS_FAIL'])

    def test_old_N_and_foreign_identity_source_and_policy_cannot_be_joined(self):
        s=proposal();request=s['rows'][1]
        with self.assertRaises(ValueError):model.mean_evidence(request,dict(plan={'N_X':999},actual={}),{'kind':'OLD_N'})
        for mutate in (lambda e:e.update(kind='HIGH_PROBABILITY_N'),
            lambda e:e['identity']['binding'].update(state='OTHER'),lambda e:e.update(source=None)):
            def wrong(rows):
                out=mock_provider(rows);mutate(out[0]);return out
            with self.assertRaises(ValueError):model.evaluate(s,mean_provider=wrong)
        bad=deepcopy(s);bad['task']['configuration']['policy']['slope_D']=5
        with self.assertRaises(ValueError):model.evaluate(bad)

    def test_budget_and_empirical_horizon_reject_underresolved_M(self):
        s=proposal(M=0)
        out=model.evaluate(s,mean_provider=mock_provider)
        self.assertTrue(all(r['status']=='INAPPLICABLE_MODEL_ERROR_BUDGET' for r in out['records']))
        s=proposal();p=policy(error='minkowski')
        self.assertEqual(p['error_rule'],'minkowski')
        self.assertNotIn('alpha_axis',s['task']['configuration'])
        self.assertNotIn('beta_stat',s['task']['configuration'])

    def test_existing_finite_mean_leaf_on_artificial_2level_context(self):
        import run_h6_df19_rpe_d2
        from check_h6_df19_rpe_d2 import toy,ready
        from trottertracks import h6_df19_rpe_d2 as action
        s=proposal(M=0);req=s['rows'][1];ctx=toy();ctx.scientific_binding=req['identity']['binding']
        ready(ctx,req['identity']['candidate'])
        ref=dict(binding=ctx.scientific_binding,T=.1,source={'kind':'ARTIFICIAL_ONLY'},evidence=action.reference_pair(ctx,.1))
        with patch.object(old,'analyze',side_effect=AssertionError('No Hoeffding N')):
            ev=model.existing_finite_mean(ctx,req,ref,{'kind':'ARTIFICIAL_ONLY'})
        self.assertEqual(ev['kind'],'FINITE_K_MEAN_ONLY')
        self.assertLess(ev['signal']['allowance_evidence']['raw_corrected_vector_closure'],1e-9)
        self.assertNotIn('plan',ev)

    def test_existing_decoder_sign_and_zero_boundary_without_sampling(self):
        E=-.2;h=.1
        xy=[(math.cos(-E*h*(1<<m)),math.sin(-E*h*(1<<m))) for m in range(4)]
        got=model.decode_supplied_coordinates(xy,h=h)
        self.assertAlmostEqual(got['energy_estimate_ha'],E)
        self.assertFalse(got['rmse_guaranteed']);self.assertFalse(got['prior_alias_qualified'])
        self.assertEqual(model.decode_supplied_coordinates([(0.,0.)],h=h)['status'],'UNRESOLVED_ZERO_COORDINATES')

    def test_public_Hoeffding_P1_sources_unchanged_and_artificial_demo(self):
        for path in ('src/trottertracks/pr_r_dependent_n.py','src/trottertracks/pr_r_detailed_n_p1.py','src/trottertracks/h6_df19_rpe_d2.py','src/trotterlib/rte.py'):
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show','71ba49bb28b295aa28e51101bc3b1bf07d500767:'+path],cwd=ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'demo.json'
            subprocess.run([sys.executable,'-B','scripts/demo_pr_rpe_rmse.py',str(out)],cwd=ROOT,check=True)
            d=json.loads(out.read_text());self.assertEqual(d['evidence_domain'],'ARTIFICIAL_ONLY')
            self.assertEqual(d['real_science_calls'],0)
            self.assertTrue(all(r['actual_N_X'] is None for r in d['result']['records']))
            self.assertTrue(all(p['N_packet']['status']=='SYNTHETIC_ONLY' for p in d['B_handoff']['packets']))


if __name__=='__main__':unittest.main(verbosity=2)
