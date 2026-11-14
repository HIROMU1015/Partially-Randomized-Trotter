"""13-round artificial signals/counts and immutable metadata, no real risk/DP."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_pr_rpe_rmse_integration import inputs, I
from check_rpe_total_binding import T, no_science, MOCK
from check_pr_rpe_rmse_controller import data
from trottertracks import rpe_signal_statistics_caller as K

B = T.B


def artificial():
    args = inputs(M=12)
    args['design']['binding']['rank'] = 10
    args['tail_norm']['binding'] = deepcopy(args['design']['binding'])
    config = B.generate(I.a_spec(args))
    target = dict(quantity='ORIGINAL_DF_H_TARGET_ENERGY', binding=deepcopy(config['spec']['binding']),
        energy_shift_ha=0, source=MOCK, qualification='UNQUALIFIED', ground_state_certified=False)
    api = T.C.interfaces(str(ROOT)).pipeline
    request = api.plan(ROOT, config, target=target, signal_source=MOCK,
        pr_physical={r['m']:r['physical_candidate'] for r in config['rounds']})
    means = [dict(m=r['m'], identity=deepcopy(r['identity']), law_binding=deepcopy(r['law_binding']),
        source=MOCK, implementation_source=MOCK, raw=dict(real=.25,imag=-.5), corrected=dict(real=.5,imag=-1.),
        normalization=dict(log_B=0.,B=1.,attenuation=1.,fixture='ARTIFICIAL_SUPPLIED_NOT_RECOMPUTED'),
        uncertainty=dict(fixture='ARTIFICIAL_UNQUALIFIED'), qualification='UNQUALIFIED', target=deepcopy(target),
        provider_binding=dict(physical_context=api.price_context(request,r['m']), numerical_source=MOCK))
        for r in config['rounds']]
    packet = dict(configuration_id=config['id'],source=MOCK,target=target,means=means,bundle_source=MOCK)
    return config, packet, request


def saved(ld):
    source=dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af',
        path=f'docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD{ld}.json')
    bundle=T._selected(ROOT,source)
    accepted=K.accept_saved(ROOT,bundle_source=source,integration=T.C.interfaces(str(ROOT)).pipeline)
    config=accepted['request']['configuration']
    return T.create(ROOT,config,original_source=bundle['configuration_source']), accepted['signal_packet']


class FixedGridTests(unittest.TestCase):
    def test_13_round_view_preserves_every_physical_and_signal_field(self):
        with no_science():
            old, packet, request = artificial(); before=deepcopy(packet)
            bridge=T.create(ROOT,old,original_source=MOCK); view=T.bind_signal(ROOT,bridge,packet)
            new=view['configuration']; B._validate(new)
            self.assertNotEqual(old['id'],new['id'])
            self.assertEqual(new,B.generate(new['spec']))
            self.assertEqual(new['spec']['A_input'],old['spec']['A_input'])
            self.assertEqual(new['spec']['decoder_origin_spec'],old['spec'])
            self.assertEqual(new['spec']['schedule_adapter'],T.FIXED_GRID_TOTAL)
            self.assertEqual(packet,before)
            for a,b,x,y in zip(old['rounds'],new['rounds'],packet['means'],view['signal_packet']['means']):
                for key in a:
                    if key not in ('candidate','identity','law_binding'):self.assertEqual(a[key],b[key])
                self.assertEqual(a['law_binding']['definition'],b['law_binding']['definition'])
                self.assertEqual(a['physical_candidate']['id'],b['physical_candidate']['id'])
                self.assertNotEqual(a['candidate']['id'],b['candidate']['id'])
                for key in x:
                    if key not in ('identity','law_binding','provider_binding'):self.assertEqual(x[key],y[key])
                self.assertEqual(x['provider_binding'],y['provider_binding'])
            self.assertFalse(bridge['scientific_adoption'])
            for key in ('sufficient_N','global_RMSE_bound','G','winner'):self.assertIsNone(view['signal_packet'][key])

    def test_unknown_grid_mutation_and_external_mapper_rejected(self):
        with no_science():
            old,packet,_=artificial(); bridge=T.create(ROOT,old,original_source=MOCK)
            for name in ('h','kappa','law','A_input','decoder_origin_spec','rmse_policy','schedule_adapter'):
                bad=deepcopy(bridge['total_configuration']);bad['spec'][name]={}
                with self.assertRaises((ValueError,KeyError,TypeError)):B._validate(bad)
            bad=deepcopy(bridge['total_configuration']);bad['rounds'][12]['physical_candidate']['id']='WRONG'
            with self.assertRaises(ValueError):B._validate(bad)
            bad=deepcopy(bridge);bad['mapping'][0]['original_candidate_id']='WRONG'
            with self.assertRaises(ValueError):T.validate(ROOT,bad)
            with self.assertRaises(ValueError):B.generate(bridge['total_configuration']['spec'],event_mapper=lambda *a:None)
            with self.assertRaises(ValueError):T.create(ROOT,bridge['total_configuration'],original_source=MOCK)
            args=inputs();args['design']['delta']=.15
            with self.assertRaisesRegex(ValueError,'Nonintegral PF grid'):I.a_spec(args)

    def test_saved_two_bundles_are_bound_without_real_numerical_calls(self):
        with no_science():
            for ld in (1,2):
                bridge,packet=saved(ld);original=deepcopy(packet)
                view=T.bind_signal(ROOT,bridge,packet)
                self.assertEqual(packet,original);self.assertEqual(len(view['signal_input']['means']),13)
                self.assertEqual(view['request']['provider'],'A_PR')
                for key in ('raw','normalization','target','source','implementation_source'):
                    self.assertEqual([m[key] for m in packet['means']], [m[key] for m in view['signal_input']['means']])
                for key in ('raw','normalization','law_binding','source'):
                    bad=deepcopy(packet);bad['means'][0][key]={}
                    with self.assertRaises(ValueError):T.bind_signal(ROOT,bridge,bad)
                bad=deepcopy(bridge['original_source'])
                bad['path']=f'docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD{3-ld}.json'
                with self.assertRaises(ValueError):T.create(ROOT,bridge['original_configuration'],original_source=bad)
            a,p1=saved(1);b,p2=saved(2)
            with self.assertRaises(ValueError):T.bind_signal(ROOT,a,p2)

    def test_artificial_13_round_counts_keep_zero_tie_and_original_prior_rules(self):
        with no_science():
            config,packet,_=artificial();bridge=T.create(ROOT,config,original_source=MOCK)
            view=T.bind_signal(ROOT,bridge,packet);contract=view['configuration']['spec']['rmse_policy']['error_contract']
            self.assertEqual(contract['decoder'],T.T.ESTIMATOR)
            counts=[dict(m=m,X=dict(plus=1,shots=2),Y=dict(plus=1,shots=2)) for m in range(13)]
            decoded=T.T.decode_counts_total(counts,h=.1,M=12,prior=contract['energy_prior_ha'])
            self.assertEqual(decoded['zero_hold_rounds'],list(range(13)))
            self.assertEqual(decoded['energy_estimate_ha'],0.)
            self.assertFalse(decoded['prior_alias_qualified']);self.assertFalse(decoded['rmse_guaranteed'])
            for row in counts:row['X']['plus']=2
            counts[1]['X']['plus']=0
            decoded=T.T.decode_counts_total(counts,h=.1,M=12,prior=contract['energy_prior_ha'])
            self.assertIn(1,decoded['tie_argmin_rounds'])
            self.assertEqual(decoded['status'],T.T.DECODED_TOTAL)

    def test_artificial_evaluation_delegates_only_to_original_kernel_contract(self):
        old,packet,_=artificial();bridge=T.create(ROOT,old,original_source=MOCK)
        with no_science():view=T.bind_signal(ROOT,bridge,packet)
        allocations=[dict(allocation_id='ART_13',source=MOCK,shot_policy='USER_SUPPLIED',
            shots=[dict(m=m,N_X=2,N_Y=3)for m in range(13)])]
        opts=dict(estimator=T.T.ESTIMATOR)
        with patch.object(T.Q,'evaluate',return_value=[dict(deterministic_input=dict(fixture='KERNEL'))])as call:
            record=T.evaluate(ROOT,bridge,view['signal_packet'],allocations,
                evaluation_target=dict(energy_ha=0.,source=MOCK),statistical_contract=MOCK,source=MOCK,options=opts)[0]
        self.assertEqual(call.call_args.args,(old,packet,allocations))
        self.assertEqual(record['configuration_id'],bridge['total_configuration']['id'])
        self.assertEqual(record['deterministic_input']['kernel_input'],dict(fixture='KERNEL'))
        for key in ('n_packet','sufficient_N','global_RMSE_bound','G','G_qualified'):self.assertIsNone(record[key])

    def test_artificial_price_view_never_reissues_or_changes_physical_ID(self):
        with no_science():
            old,packet,request=artificial();bridge=T.create(ROOT,old,original_source=MOCK)
            _,prices,cohorts=data(old);api=T.C.interfaces(str(ROOT)).pipeline
            sides=[dict(identity=deepcopy(v['identity']),law_binding=deepcopy(v['law_binding']),axis=v['axis'],
                cohort=v['cohort'],source=MOCK,price_source=deepcopy(v['source']),
                physical_context=api.price_context(request,v['identity']['m'])) for v in prices]
            original=deepcopy((prices,sides))
            # Guard behavior itself is covered by the existing real guard's
            # artificial regression tests; this test isolates the ID bridge.
            guard=SimpleNamespace(check=lambda *a,**k:dict(status='ARTIFICIAL_CHECKED'))
            with patch.object(T.R,'guard',return_value=guard):
                bound=T.bind_prices(ROOT,bridge,prices,bindings=sides,cohorts=cohorts,policy=MOCK)
            self.assertFalse(bound['receipt_reissued']);self.assertEqual((prices,sides),original)
            for a,b in zip(sides,bound['bindings']):
                self.assertEqual(a['physical_context'],b['physical_context'])
                self.assertNotEqual(a['identity'],b['identity'])


if __name__=='__main__':unittest.main(verbosity=2)
