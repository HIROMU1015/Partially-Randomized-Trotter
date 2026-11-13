"""Artificial whole-schedule shot allocations through existing APIs only.

No molecular inputs, new shot solver, or scientific qualification is provided.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_decoder_statistics import toy, POLICY, MOCK
from check_pr_rpe_rmse_controller import data
from trottertracks import rpe_decoder_statistics as S
from trottertracks import rpe_decoder_fast as F

B = S.B


class ShotEvaluationDesignTests(unittest.TestCase):
    def prepared(self, M=2):
        config,means,shots,target = toy(M=M)
        return config,means,shots,target,S.prepare(config,means,shots,target=target,
            shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK))

    def test_01_same_physical_config_distinct_complete_allocations(self):
        config,means,shots,target,_ = self.prepared(M=12)
        original = deepcopy((config,means,shots,target))
        allocations = [deepcopy(shots),[dict(m=r['m'],N_X=2*r['N_X'],N_Y=2*r['N_Y']) for r in shots]]
        prepared = []
        with patch.object(S,'simulate',side_effect=AssertionError('metadata started simulation')):
            for vector in allocations:
                prepared.append(S.prepare(config,means,vector,target=target,
                    shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK)))
        self.assertEqual([v['configuration_id'] for v in prepared],[config['id']]*2)
        self.assertNotEqual(prepared[0]['rows'][0]['shots'],prepared[1]['rows'][0]['shots'])
        self.assertEqual(original,(config,means,shots,target))
        self.assertTrue(all(v['input_complete'] and v['sufficient_N'] is None for v in prepared))

    def test_02_raw_mean_source_identity_law_and_target_are_bound(self):
        config,means,shots,target,_ = self.prepared()
        edits = [lambda v:v.pop('source'),lambda v:v['identity'].update(candidate={'id':'FOREIGN'}),
                 lambda v:v['law_binding'].update(configuration_id='FOREIGN'),
                 lambda v:v['target'].update(energy_shift_ha=1)]
        for edit in edits:
            changed = deepcopy(means);edit(changed[0])
            with self.assertRaises(ValueError):
                S.prepare(config,changed,shots,target=target,shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK))

    def test_03_missing_rounds_never_simulate_and_unqualified_not_promoted(self):
        config,means,shots,target,_ = self.prepared()
        partial=S.prepare(config,means[:-1],shots,target=target,shot_policy=dict(meaning='NOMINAL',source=MOCK))
        with patch.object(S,'simulate',side_effect=AssertionError('missing signal simulation')):
            result=S.simulate_prepared(partial,repetitions=3,seed=1,policy=POLICY,
                batch_decoder=F.decode_sampled_counts,allow_conditional_unqualified=True)
        self.assertEqual(result['status'],'NOT_EVALUATED_MISSING_STATISTICAL_INPUT')
        full=S.prepare(config,means,shots,target=target,shot_policy=dict(meaning='SUFFICIENT_EXTERNALLY_QUALIFIED',source=MOCK))
        self.assertFalse(full['physical_qualification_complete']);self.assertIsNone(full['sufficient_N'])

    def test_04_synthetic_batch_diagnostic_is_not_a_qualified_N_packet(self):
        config,means,shots,target,prepared=self.prepared()
        simulation=S.simulate_prepared(prepared,repetitions=17,seed=919,policy=POLICY,
            allow_conditional_unqualified=True,batch_decoder=F.decode_sampled_counts)
        summary=S.summarize(simulation['trials'],target_energy_ha=target['energy_ha'],
            unresolved_loss=dict(kind='UNDEFINED',source=MOCK),iid_repetitions=True,confidence=.95)
        self.assertIsNone(summary['population_RMSE_CI']);self.assertIsNone(summary['sufficient_N'])
        packet,prices,cohorts=data(config)
        packet.update(status='RPE_SHOT_DIAGNOSTIC_ONLY')
        packet['rmse_evidence'].update(status='UNQUALIFIED',rmse_bound_ha=None,
            covers_state_target_alias_prior_decoder_all_rounds=False)
        packet['diagnostic']=summary
        for row,allocation in zip(packet['rounds'],shots):
            row.update(N_X=None,N_Y=None,evaluated_N_X=allocation['N_X'],evaluated_N_Y=allocation['N_Y'])
        result=B.cost(config,packet,prices,cohorts=cohorts)
        self.assertFalse(result['eligible']);self.assertIsNone(result['cost'])

    def test_05_total_shots_and_resource_cost_have_different_order(self):
        config,_,_,_,_=self.prepared(M=1)
        na,prices,cohorts=data(config);nb=deepcopy(na)
        for row,n in zip(na['rounds'],(1,5)):row.update(N_X=n,N_Y=n)
        for row,n in zip(nb['rounds'],(8,1)):row.update(N_X=n,N_Y=n)
        for price in prices:price['mean']['rz_count']=1 if price['identity']['m']==0 else 100
        ca=B.cost(config,na,prices,cohorts=cohorts);cb=B.cost(config,nb,prices,cohorts=cohorts)
        self.assertEqual(sum(r['N_X']+r['N_Y'] for r in na['rounds']),12)
        self.assertEqual(sum(r['N_X']+r['N_Y'] for r in nb['rounds']),18)
        self.assertEqual(B.a.exact_scalar(ca['cost']),Fraction(1002))
        self.assertEqual(B.a.exact_scalar(cb['cost']),Fraction(216))
        # select() accepts one evaluation per physical configuration. An inner
        # allocation layer must not pretend repeated configurations are unique.
        with self.assertRaisesRegex(ValueError,'unique configuration'):
            B.select([config,config],[ca,cb])

    def test_06_huge_N_remains_exact_and_backend_limit_is_explicit(self):
        config,means,shots,target,_=self.prepared(M=0)
        huge=2**200+1;shots[0].update(N_X=huge,N_Y=huge)
        prepared=S.prepare(config,means,shots,target=target,shot_policy=dict(meaning='USER_SUPPLIED',source=MOCK))
        self.assertEqual(prepared['rows'][0]['shots']['N_X'],huge)
        with self.assertRaisesRegex(ValueError,'int64 domain exceeded'):
            S.simulate_prepared(prepared,repetitions=1,seed=1,policy=POLICY,
                allow_conditional_unqualified=True,batch_decoder=F.decode_sampled_counts)

    def test_07_even_shots_can_introduce_zero_coordinate_mass(self):
        # Artificial model only: N1 has no zero coordinate; N2 does. Thus a
        # bisection premised on monotone decoder resolution needs a proof.
        probability=[]
        for n,states in ((1,4),(2,9)):
            ensemble=S.exact_counts([dict(m=0,raw=dict(real=0.,imag=0.),N_X=n,N_Y=n)],
                h=1,M=0,prior=[-4,4],policy=POLICY,maximum_states=states)
            summary=S.summarize(ensemble['trials'],target_energy_ha=0,
                unresolved_loss=dict(kind='UNDEFINED',source=MOCK))
            probability.append(summary['unresolved_probability'])
        self.assertEqual(probability[0],0.);self.assertGreater(probability[1],0.)


if __name__=='__main__':unittest.main()
