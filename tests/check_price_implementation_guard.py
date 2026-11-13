"""Fictional prices/N only; exercise actual C -> B arithmetic admission."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import check_rpe_optimization_pipeline as T
from trottertracks import price_implementation_guard as G
from trottertracks import rpe_mean_price_model as P

ROOT=T.ROOT;I=T.I;B=T.B
SOURCE=dict(fixture='ARTIFICIAL_IMPLEMENTATION_ONLY')
CODE=dict(commit=I.B_SOURCE,path='src/trottertracks/pf_pr_rpe_pipeline.py')
NEW='CONTROLLED_ZZ_CX_CRZ_CX_V1'
OLD='CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424'
NEW_CODE=dict(commit='01e068af7e06d9f1f5621a5b3800c30eb32ecb06',path='src/trottertracks/efficient_controlled_zz.py')
OLD_CODE=dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe',path='src/trotterlib/df_rte_qiskit.py')


def policy():
    return dict(id='ARTIFICIAL_COMMON_NEW_ZZ_SCENARIO',source=SOURCE,profile=deepcopy(B.p.PROFILE),
        by_method={method:dict(native_policy='ART_NATIVE',rte_rotation_lowering=NEW if method in ('PR','FULL_H_RTE') else G.NO_RTE,
                              scalar_quantity='ART_TOTAL_'+method)
                   for method in ('PR','FULL_H_RTE','FULL_PF','TRUNCATED_PF')})


def bind(values):
    values=deepcopy(values);req=values[0];opts=values[5];pol=policy();method=B.comparison_method(req['configuration']['spec'])
    domain=pol['by_method'][method];lower=domain['native_policy'] if domain['rte_rotation_lowering']==G.NO_RTE else domain['rte_rotation_lowering']
    opts['price_extension']=dict(lowering_id=lower,source=SOURCE);opts['comparison_policy']=pol
    proofs={}
    for price,side in zip(values[4],opts['price_bindings']):
        price['normalization_is_price_multiplier']=False
        row=req['configuration']['rounds'][price['identity']['m']]
        side['profile_extension']=deepcopy(opts['price_extension'])
        facts=dict(domain,method=method,profile=deepcopy(pol['profile']),measurement=B.p.measurement(row,price['axis']),
            native_source=CODE,wrapper_source=CODE,scalar_source=CODE,scalar_application=G.SCALAR_ONCE,
            rte_lowering_source=NEW_CODE if method in ('PR','FULL_H_RTE') else None,
            resolved_scalar={'numerator':1,'denominator':8},expected_scalar={'numerator':1,'denominator':8})
        side['implementation']=facts
        locator=dict(commit='a'*40,path='ARTIFICIAL_FIXED_METADATA.json',selector=[str(row['m'])+price['axis']])
        side['implementation_evidence']=locator
        proof={k:deepcopy(side[k]) for k in ('price_source','identity','law_binding','axis','cohort','physical_context','profile_extension','implementation')}
        proofs[locator['selector'][0]]=proof
    return values,proofs


def compare(values,proofs):
    with patch.object(G,'_document',return_value=proofs):
        return I.compare(ROOT,*values[:5],**values[5])


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures={m:T.fixture(m) for m in ('PR','FULL_H_RTE','FULL','B0I')}

    def fixture(self,method='PR'):return bind(self.fixtures[method])

    def test_same_compiler_old_fullH_in_new_policy_rejected_before_B_cost(self):
        v,proofs=self.fixture('FULL_H_RTE')
        for s in v[5]['price_bindings']:
            s['implementation']['rte_rotation_lowering']=OLD
            proofs[s['implementation_evidence']['selector'][0]]['implementation']['rte_rotation_lowering']=OLD
        with patch.object(B,'cost',side_effect=AssertionError('incompatible price reached B.cost')):
            with self.assertRaisesRegex(ValueError,'rte_rotation_lowering'):compare(v,proofs)

    def test_legacy_fullH_and_new_PR_cannot_share_new_family(self):
        pr,proofs=self.fixture('PR');pr_result=compare(pr,proofs)
        fh,old_proofs=self.fixture('FULL_H_RTE')
        fh[5]['comparison_policy']['id']='ART_OLD_SCENARIO'
        fh[5]['comparison_policy']['by_method']['FULL_H_RTE']['rte_rotation_lowering']=OLD
        fh[5]['comparison_policy']['by_method']['PR']['rte_rotation_lowering']=OLD
        fh[5]['price_extension']['lowering_id']=OLD
        for side in fh[5]['price_bindings']:
            side['implementation']['rte_rotation_lowering']=OLD;side['profile_extension']['lowering_id']=OLD
            side['implementation']['rte_lowering_source']=OLD_CODE
            proof=old_proofs[side['implementation_evidence']['selector'][0]]
            proof['implementation']['rte_rotation_lowering']=OLD;proof['profile_extension']['lowering_id']=OLD
            proof['implementation']['rte_lowering_source']=OLD_CODE
        old_result=compare(fh,old_proofs)
        self.assertIsNotNone(old_result['comparison']['point_minimum']['cost'])
        with self.assertRaisesRegex(ValueError,'common SOURCE-bound'):I.select(ROOT,[pr_result,old_result])

    def test_same_valid_domain_reaches_existing_cost_and_family_select(self):
        results=[]
        for method in ('PR','FULL_H_RTE','FULL','B0I'):
            v,proofs=self.fixture(method);before=deepcopy(v[0]['configuration']);out=compare(v,proofs)
            self.assertEqual([B.a.exact_scalar(r['G']) for r in out['comparison']['allocations']],[1002,216])
            self.assertEqual(out['request']['configuration'],before)
            results.append(out)
        self.assertTrue(I.select(ROOT,results)['menu_complete'])

    def test_one_policy_cannot_declare_old_and_new_RTE_comparable_without_evidence(self):
        v,proofs=self.fixture()
        v[5]['comparison_policy']['by_method']['FULL_H_RTE']['rte_rotation_lowering']=OLD
        with patch.object(B,'cost',side_effect=AssertionError('mixed implementation family')):
            with self.assertRaisesRegex(ValueError,'Mixed RTE lowering'):compare(v,proofs)

    def test_unknown_lowering_and_source_contradiction_rejected(self):
        for change in (lambda v,p:v[5]['price_bindings'][0].pop('implementation'),
                       lambda v,p:v[5]['price_bindings'][0]['implementation'].update(rte_rotation_lowering='UNKNOWN'),
                       lambda v,p:v[4][0].update(source={'fixture':'FOREIGN'}),
                       lambda v,p:p['0X']['implementation'].update(rte_lowering_source=None)):
            v,proofs=self.fixture();change(v,proofs)
            with patch.object(B,'cost',side_effect=AssertionError('unbound SOURCE reached B.cost')):
                with self.assertRaises(ValueError):compare(v,proofs)

    def test_scalar_wrapper_axis_rejected(self):
        for name,value in (('scalar_quantity','TAIL_ONLY'),('scalar_application','TWICE'),
                           ('resolved_scalar',{'numerator':3,'denominator':8}),
                           ('profile',dict(policy()['profile'],wrapper='WITHOUT_MEASUREMENT')),
                           ('measurement',{'axis':'FOREIGN'})):
            v,proofs=self.fixture();side=v[5]['price_bindings'][0]
            side['implementation'][name]=deepcopy(value);proofs['0X']['implementation'][name]=deepcopy(value)
            with patch.object(B,'cost',side_effect=AssertionError('wrong scalar/wrapper/axis reached B.cost')):
                with self.assertRaises(ValueError):compare(v,proofs)

    def test_source_route_cannot_be_relabelled_even_with_matching_sidecar(self):
        v,proofs=self.fixture()
        v[5]['price_bindings'][0]['implementation']['rte_lowering_source']=OLD_CODE
        proofs['0X']['implementation']['rte_lowering_source']=OLD_CODE
        with patch.object(B,'cost',side_effect=AssertionError('old emitter relabelled as new')):
            with self.assertRaisesRegex(ValueError,'emitter SOURCE'):compare(v,proofs)

    def test_PF_and_truncated_no_RTE_not_forced_to_PR_lowering(self):
        for method in ('FULL','B0I'):
            v,proofs=self.fixture(method);side=v[5]['price_bindings'][0]
            self.assertEqual(side['implementation']['rte_rotation_lowering'],G.NO_RTE)
            self.assertIsNone(side['implementation']['rte_lowering_source'])
            self.assertIsNotNone(compare(v,proofs)['comparison']['point_minimum']['cost'])
            side['implementation']['rte_lowering_source']=CODE
            proofs['0X']['implementation']['rte_lowering_source']=CODE
            with self.assertRaisesRegex(ValueError,'no RTE'):compare(v,proofs)

    def test_missing_prices_remain_unknown_and_no_zero_price(self):
        v,proofs=self.fixture();v[4].pop()
        with patch.object(B,'cost',side_effect=AssertionError('missing price reached B.cost')):
            out=compare(v,proofs)
        self.assertIsNone(out['comparison']['point_minimum']['cost'])

    def test_scientific_route_without_policy_never_calls_B_cost(self):
        v=deepcopy(self.fixtures['PR'])
        for evaluation in v[3]:evaluation['n_packet']['evidence_kind']='SCIENTIFIC'
        with patch.object(B,'cost',side_effect=AssertionError('policy-less scientific arithmetic')):
            out=I.compare(ROOT,*v[:5],**v[5])
        self.assertIn('MISSING_PRICE_COMPARISON_IMPLEMENTATION_POLICY',out['blockers'])
        self.assertIsNone(out['comparison']['outer_evaluation'])

    def test_PriceProvider_preserves_sidecar_and_checks_before_return(self):
        v,proofs=self.fixture('PR');req=v[0]
        for r,s in zip(v[4],v[5]['price_bindings']):
            scope=P.context(ROOT,req,r['identity']['m'],r['axis'],profile_extension=v[5]['price_extension'])
            r.update(physical_context=scope['physical_context'],price_profile=scope['price_profile'],
                     normalization_is_price_multiplier=False)
            for key in ('implementation','implementation_evidence'):r[key]=deepcopy(s[key])
        provider=P.PriceProvider(ROOT,req,v[4],profile_extension=v[5]['price_extension'],comparison_policy=v[5]['comparison_policy'])
        with patch.object(G,'_document',return_value=proofs):out=provider()
        self.assertEqual(out['prices'],v[4]);self.assertEqual(out['comparison_policy'],policy())
        provider.records[0]['implementation']['rte_rotation_lowering']=OLD
        with patch.object(G,'_document',return_value=proofs):
            with self.assertRaises(ValueError):provider()

    def test_direct_B_consumer_handoff_guard_before_arithmetic(self):
        v,proofs=self.fixture();s=v[5]
        with patch.object(G,'_document',return_value=proofs):
            result=G.cost(ROOT,B,v[0]['configuration'],v[3][0]['n_packet'],v[4],cohorts=s['cohorts'],
                price_bindings=s['price_bindings'],comparison_policy=s['comparison_policy'])
        self.assertEqual(B.a.exact_scalar(result['cost']),1002)
        s['price_bindings'][0]['implementation']['rte_rotation_lowering']=OLD
        with patch.object(G,'_document',return_value=proofs),patch.object(B,'cost',side_effect=AssertionError('bypass')):
            with self.assertRaises(ValueError):G.cost(ROOT,B,v[0]['configuration'],v[3][0]['n_packet'],v[4],
                cohorts=s['cohorts'],price_bindings=s['price_bindings'],comparison_policy=s['comparison_policy'])

    def test_pinned_metadata_reader_without_price_or_science_calls(self):
        src=dict(commit='3d39f8c0ca68fb1acf86fe65d8ee5e3cc1a15673',
            path='docs/research/d_s1_v1_price_alignment_20261114/SOURCE_CHECKS.json',
            selector=['source_checks','PF_branch_has_no_RTE_builder_and_scalar_once'])
        self.assertIs(G.evidence(ROOT,src),True)
        with self.assertRaises(ValueError):G.evidence(ROOT,dict(src,commit='HEAD'))

    def test_direct_B_handoff_refuses_foreign_physical_context(self):
        v,proofs=self.fixture('FULL_H_RTE');s=v[5]
        side=s['price_bindings'][0];side['physical_context']['physical_candidate']['T']=999
        proofs['0X']['physical_context']['physical_candidate']['T']=999
        with patch.object(G,'_document',return_value=proofs),patch.object(B,'cost',side_effect=AssertionError('wrong physical input')):
            with self.assertRaisesRegex(ValueError,'physical candidate'):
                G.cost(ROOT,B,v[0]['configuration'],v[3][0]['n_packet'],v[4],
                    cohorts=s['cohorts'],price_bindings=s['price_bindings'],comparison_policy=s['comparison_policy'])


if __name__=='__main__':unittest.main(verbosity=2)
