"""Canonical metadata and artificial risks/prices only; no real H4 numerics."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import rpe_total_binding as T
from trottertracks import rpe_pauli_signal_intake as A
from check_rpe_deterministic_risk import case
from check_pr_rpe_rmse_controller import data
from check_rpe_price_implementation_guard import fixture, G, CODE, MOCK

B, R, C, Q = T.B, T.R, T.C, T.Q
PAULI_CODE = dict(commit='9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c', path='src/trotterlib/h4_d2_pauli_control.py')


def pauli():
    args = case(M=1, vectors=[(2,2)])
    s = deepcopy(args['config']['spec']); s['binding'].update(system='H4', rank=10)
    payload = dict(prefix=1, binding=deepcopy(s['binding']), lambda_P=1., saved_DF_scalar=0.,
                   mapping='ARTIFICIAL_BIT_ORDER', terms=[])
    dictionary_source = dict(commit='b'*40,path='ARTIFICIAL_DICTIONARY.json')
    descriptor = A._metadata_helpers(str(ROOT)).dictionary_identity(dict(payload=payload,
        dictionary_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        source=dictionary_source))
    s['split'].update(tail_dictionary=descriptor, lambda_source=dict(dictionary_source,selector=['lambda_P']), scalar=0.)
    s['law']['tail_dictionary'] = deepcopy(descriptor)
    config = B.generate(s); args['config'] = config
    packet = args['signal_packet']; packet['configuration_id'] = config['id']
    packet['target']['binding'] = deepcopy(s['binding']); args['evaluation_target']['binding'] = deepcopy(s['binding'])
    packet.update(pauli_dictionary=payload,per_round_source_is_authoritative=True,
        implementation_sources={str(m):dict(commit='a'*40,path='ARTIFICIAL_NUMERICS.py') for m in range(2)})
    api = C.interfaces(str(ROOT)).pipeline
    req = api.plan(ROOT,config,target=packet['target'],signal_source=packet['source'],
        implementation_sources=packet['implementation_sources'],pauli_dictionary=payload)
    for mean,row in zip(packet['means'],config['rounds']):
        mean.update(identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
            raw=dict(real=.25,imag=-.5),target=deepcopy(packet['target']),
            implementation_source=deepcopy(packet['implementation_sources'][str(row['m'])]))
        mean['provider_binding'].update(physical_context=deepcopy(req['pending']['jobs'][row['m']]['physical_context']),
            implementation_source=deepcopy(mean['implementation_source']))
    args['options'].update(estimator=T.T.ESTIMATOR,
        total_state_refinement=dict(source=MOCK,maximum_frontier_states=10000,maximum_transition_pairs=100000))
    return args, req


def prices_for(args, req):
    config=args['config']; _,prices,cohorts=data(config)
    policy=dict(id='ARTIFICIAL_PAULI_D2',source=MOCK,profile=deepcopy(config['spec']['profile']),
        by_method=dict(PR_PAULI_TAIL=dict(native_policy='NATIVE_DF_HEAD_PLUS_PAULI_D2_TAIL',
            rte_rotation_lowering=G.PAULI_D2,scalar_quantity='ARTIFICIAL_ONCE')))
    sides=[];proofs={};api=C.interfaces(str(ROOT)).pipeline
    for price in prices:
        m=price['identity']['m'];axis=price['axis'];row=config['rounds'][m]
        price.update(quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE',normalization_is_price_multiplier=False,
            law_context=dict(N_law=config['spec']['law']['model'],price_law=config['spec']['law']['model'],ideal_sampler_distance_qualified=True))
        facts=dict(policy['by_method']['PR_PAULI_TAIL'],method='PR',route='PR_PAULI_TAIL',profile=policy['profile'],
            measurement=B.p.measurement(row,axis),native_source=CODE,wrapper_source=CODE,scalar_source=CODE,
            scalar_application='ONCE_AT_TOTAL_T',rte_lowering_source=PAULI_CODE)
        key=str(m)+axis
        side=dict(identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),axis=axis,cohort=price['cohort'],
            source=MOCK,
            price_source=deepcopy(price['source']),physical_context=api.price_context(req,m),
            profile_extension=dict(lowering_id=G.PAULI_D2,source=MOCK),implementation=facts,
            implementation_evidence=dict(commit='c'*40,path='ARTIFICIAL_RECEIPT.json',selector=[key]))
        proofs[key]={k:deepcopy(side[k]) for k in ('identity','law_binding','axis','cohort','price_source',
            'physical_context','profile_extension','implementation')}
        sides.append(side)
    return prices,sides,cohorts,policy,proofs


def no_science():
    stack=ExitStack()
    for owner,key in ((Q,'evaluate'),(Q.D,'evaluate'),(Q.D,'binomial_axis'),(K.S,'simulate_prepared'),
                      (B,'cost'),(B,'select'),(B.p,'run_missing')):
        stack.enter_context(patch.object(owner,key,side_effect=AssertionError('Real science prohibited: '+key)))
    return stack


K = T.K


class CanonicalTests(unittest.TestCase):
    def test_decoder_only_identity_change_and_changed_physics_rejected(self):
        args,_=pauli();before=deepcopy(args)
        binding=T.create(ROOT,args['config'],original_source=MOCK);new=binding['total_configuration']
        self.assertNotEqual(new['id'],args['config']['id'])
        self.assertEqual(new['spec']['rmse_policy']['error_contract']['decoder'],T.T.ESTIMATOR)
        self.assertFalse(binding['scientific_adoption'])
        for change in ('h','finite_K','law','profile','L_D'):
            bad=deepcopy(binding)
            if change in ('h','finite_K','L_D'):bad['total_configuration']['spec'][change]=7
            else:bad['total_configuration']['spec'][change]['tampered']=True
            with self.assertRaises(ValueError):T.validate(ROOT,bad)
        self.assertEqual(args,before)
        with self.assertRaises(ValueError):T.create(ROOT,new,original_source=MOCK)

    def test_signal_source_bridge_preserves_original_and_all_coordinates(self):
        args,_=pauli();old=deepcopy(args['signal_packet']);binding=T.create(ROOT,args['config'],original_source=MOCK)
        with no_science():
            view=T.bind_signal(ROOT,binding,old)
            for a,b in zip(old['means'],view['signal_input']['means']):
                self.assertEqual(a['raw'],b['raw']);self.assertEqual(a['source'],b['source'])
                self.assertNotEqual(a['identity'],b['identity']);self.assertNotEqual(a['law_binding'],b['law_binding'])
                self.assertEqual(b['canonical_origin']['provider_binding'],a['provider_binding'])
            bad=deepcopy(old);bad['means'][0]['provider_binding']['implementation_source']={}
            with self.assertRaises(ValueError):T.bind_signal(ROOT,binding,bad)
        self.assertEqual(old,args['signal_packet'])

    def test_artificial_canonical_total_evaluation_matches_unchanged_kernel(self):
        args,_=pauli();binding=T.create(ROOT,args['config'],original_source=MOCK)
        view=T.bind_signal(ROOT,binding,args['signal_packet'])
        opts={k:args[k] for k in ('allocations','evaluation_target','statistical_contract','source','options')}
        direct=Q.evaluate(**dict(args,integration=C.interfaces(str(ROOT)).pipeline))
        out=T.evaluate(ROOT,binding,view['signal_packet'],**opts)
        self.assertEqual(out[0]['diagnostic'],direct[0]['diagnostic'])
        self.assertEqual(out[0]['configuration_id'],binding['total_configuration']['id'])
        self.assertEqual(out[0]['deterministic_input']['kernel_input'],direct[0]['deterministic_input'])
        for k in ('n_packet','sufficient_N','global_RMSE_bound','G','G_qualified'):self.assertIsNone(out[0][k])
        bad=deepcopy(view['signal_packet']);bad['means'][0]['raw']['real']=.5
        with self.assertRaises(ValueError):T.evaluate(ROOT,binding,bad,**opts)
        opts['options']=dict(opts['options'],state_refinement=MOCK)
        with self.assertRaisesRegex(ValueError,'Old partial DP'):T.evaluate(ROOT,binding,view['signal_packet'],**opts)

    def test_pinned_C_Pauli_guard_and_receipt_bridge_before_arithmetic(self):
        args,req=pauli();binding=T.create(ROOT,args['config'],original_source=MOCK)
        prices,sides,cohorts,policy,proofs=prices_for(args,req);before=deepcopy((prices,sides))
        with patch.object(G,'_document',return_value=proofs):
            origin=G.check(ROOT,args['config'],prices,cohorts=cohorts,bindings=sides,policy=policy)
            equivalent=R.check(ROOT,args['config'],prices,cohorts=cohorts,bindings=sides,policy=policy)
            self.assertEqual(origin,equivalent)
            c_path=C.interfaces(str(ROOT)).pipeline.compare(ROOT,req,args['signal_packet'],args['allocations'],[],prices,
                statistical_contract=args['statistical_contract'],cohorts=cohorts,
                price_extension=sides[0]['profile_extension'],price_bindings=sides,comparison_policy=policy)
            self.assertEqual(c_path['price_implementation_check'],equivalent)
            self.assertIsNone(c_path['comparison']['outer_evaluation'])
            view=T.bind_prices(ROOT,binding,prices,bindings=sides,cohorts=cohorts,policy=policy)
            with no_science():
                checked=R.check(ROOT,binding['total_configuration'],view['prices'],cohorts=view['cohorts'],
                    bindings=view['bindings'],policy=policy,canonical_binding=view)
                self.assertFalse(checked['receipt_reissued'])
                bad=deepcopy(view['prices']);bad[0]['mean']['rz_count']=999
                with self.assertRaises(ValueError):R.cost(ROOT,binding['total_configuration'],None,bad,
                    cohorts=view['cohorts'],price_bindings=view['bindings'],comparison_policy=policy,canonical_binding=view)
                with self.assertRaises(ValueError):R.check(ROOT,binding['total_configuration'],view['prices'],
                    cohorts=view['cohorts'],bindings=view['bindings'],policy=policy)
            # Existing controller handles missing N: no nominal promotion or new producer.
            out=R.cost(ROOT,binding['total_configuration'],None,view['prices'],cohorts=view['cohorts'],
                price_bindings=view['bindings'],comparison_policy=policy,canonical_binding=view)
            self.assertIsNone(out['cost'])
            signal=T.bind_signal(ROOT,binding,args['signal_packet'])
            records=T.evaluate(ROOT,binding,signal['signal_packet'],**{k:args[k] for k in (
                'allocations','evaluation_target','statistical_contract','source','options')})
            packet,_,_=data(binding['total_configuration'],shots=2)
            packet['rmse_evidence'].update(statistical_contract=deepcopy(args['statistical_contract']),
                deterministic_input=deepcopy(records[0]['deterministic_input']))
            result=T.compare(ROOT,binding,records,view['prices'],
                model_limit=dict(source=MOCK,scope='MODEL_MSE_FILTER_ONLY',MSE_upper=999),qualifications={'v0':packet},
                cohorts=view['cohorts'],statistical_contract=args['statistical_contract'],
                price_extension=sides[0]['profile_extension'],price_bindings=view['bindings'],
                comparison_policy=policy,canonical_binding=view)
            self.assertEqual(B.a.exact_scalar(result['comparison']['allocations'][0]['G']),8)
            bad=deepcopy(policy);bad['by_method']['PR_PAULI_TAIL']['rte_rotation_lowering']=G.NEW_ZZ
            with self.assertRaises(ValueError):T.bind_prices(ROOT,binding,prices,bindings=sides,cohorts=cohorts,policy=bad)
        self.assertEqual((prices,sides),before)

    def test_C_paper_cost_is_separate_and_never_qualifies_N(self):
        args,req=pauli();prices,sides,cohorts,policy,proofs=prices_for(args,req)
        paper=C.interfaces(str(ROOT)).paper
        with patch.object(G,'_document',return_value=proofs), \
             patch.object(B,'cost',side_effect=AssertionError('paper must not call B.cost')), \
             patch.object(B,'select',side_effect=AssertionError('paper must not call B.select')):
            nominal=paper.nominal_packet(ROOT,args['config'],args['signal_packet'],source=MOCK)
            out=paper.paper_cost(ROOT,args['config'],nominal,prices,cohorts=cohorts,
                price_bindings=sides,comparison_policy=policy)
            self.assertIsNotNone(out['G_paper']);self.assertIsNone(out['G_qualified'])
            self.assertFalse(out['mse_qualified']);self.assertIsNone(out['sufficient_N'])
            binding=T.create(ROOT,args['config'],original_source=MOCK)
            with self.assertRaisesRegex(ValueError,'nominal'):
                T.compare(ROOT,binding,[],[],model_limit={},qualifications={'nominal':nominal})

    def test_N_contract_and_deterministic_input_required_for_qualified_compare(self):
        args,_=pauli();binding=T.create(ROOT,args['config'],original_source=MOCK)
        view=T.bind_signal(ROOT,binding,args['signal_packet'])
        opts={k:args[k] for k in ('allocations','evaluation_target','statistical_contract','source','options')}
        records=T.evaluate(ROOT,binding,view['signal_packet'],**opts)
        packet,_,_=data(binding['total_configuration'])
        packet['rmse_evidence']['error_contract']['decoder']=T.T.ORIGINAL_ESTIMATOR
        with self.assertRaisesRegex(ValueError,'N packet'):
            T.compare(ROOT,binding,records,[],model_limit={},qualifications={'v0':packet})
        with no_science():
            # Missing external N: underlying W.compare produces UNKNOWN, no G.
            out=T.compare(ROOT,binding,records,[],model_limit=dict(source=MOCK,scope='MODEL_MSE_FILTER_ONLY',MSE_upper=999),
                qualifications={},cohorts={},statistical_contract=args['statistical_contract'])
            self.assertIsNone(out['comparison']['outer_evaluation'])


class SavedMetadataTests(unittest.TestCase):
    def test_real_Pauli_canonical_bridge_without_risk_or_N(self):
        with no_science():
            api=C.interfaces(str(ROOT)).pipeline;accepted=A.accept_saved(ROOT,integration=api)
            for index,item in enumerate(accepted['results']):
                self.assertEqual(item['C_plan_price_bridge'],'SOURCE_BOUND_PAULI_PHYSICAL_BRIDGE_METADATA_ONLY')
                binding=T.create(ROOT,item['configuration'],original_source=dict(A.HANDOFF,selector=['packets',index,'configuration']))
                view=T.bind_signal(ROOT,binding,item['signal_packet'])
                self.assertEqual(len(view['signal_input']['means']),13)
                self.assertEqual(view['request']['provider'],'A_PR_PAULI')
                bad=deepcopy(item['signal_packet']);bad['means'][0]['raw']['real']=0.
                with self.assertRaisesRegex(ValueError,'saved SOURCE'):T.bind_signal(ROOT,binding,bad)

    def test_copied_modules_and_old_DP_are_fixed_bytes(self):
        for path in ('src/trottertracks/rpe_total_decoder.py','src/trottertracks/rpe_total_risk.py','tests/check_rpe_total_decoder_risk.py'):
            self.assertEqual((ROOT/path).read_bytes(),R.blob(ROOT,T.DECODER_SOURCE['commit'],path))
        for path in ('src/trottertracks/rpe_deterministic_risk.py','src/trottertracks/rpe_decoder_fast.py',
                     'src/trottertracks/pr_rpe_rmse_controller.py','scripts/run_h4_dp_risk_fixed.py'):
            self.assertEqual((ROOT/path).read_bytes(),R.blob(ROOT,'e7ce23803b5644bd9df0e83c877b390aeb9405e0',path))


if __name__=='__main__':unittest.main(verbosity=2)
