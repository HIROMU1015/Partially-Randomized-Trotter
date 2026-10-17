"""Synthetic-only mathematical, pairing, selection and denial regression tests."""
import copy
from fractions import Fraction
import itertools
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'/'trotterlib'))
import h8_rare_tail_selection as m


class RareLawTests(unittest.TestCase):
    def check_enumeration(self, R, stratum):
        p = (Fraction(1,2), Fraction(1,3), Fraction(1,6))
        reference = {}
        total = Fraction(0)
        # Independent oracle: enumerate ALL words under the unconditional IID law.
        words = []
        for word in itertools.product((0,2,4), repeat=R):
            n2, n4 = word.count(2), word.count(4)
            oracle = {'s0': n2 == 0 and n4 == 0,
                      's2': n2 >= 1 and n4 == 0,
                      's4': n4 >= 1,
                      's2_single': n2 == 1 and n4 == 0,
                      's2_multi': n2 >= 2 and n4 == 0}
            if not oracle[stratum]:
                continue
            weight = math.prod(p[(0,2,4).index(k)] for k in word)
            reference[n2,n4] = reference.get((n2,n4),Fraction(0))+weight
            words.append((word,weight)); total += weight
        if not total:
            with self.assertRaisesRegex(ValueError,'EMPTY'):
                m.conditional_counts(R, tuple(map(float,p)), stratum)
            return
        law = m.conditional_counts(R, tuple(map(float,p)), stratum)
        self.assertAlmostEqual(math.exp(law.log_stratum_mass),float(total),places=13)
        self.assertAlmostEqual(math.fsum(law.probabilities),1.,places=14)
        got = dict(zip(law.counts,law.probabilities))
        for key, weight in reference.items():
            self.assertAlmostEqual(got[key],float(weight/total),places=13)
        # Uniform placement factor must reproduce each conditional word as well.
        for word, weight in words:
            n2,n4 = word.count(2), word.count(4)
            multiplicity = math.comb(R,n4)*math.comb(R-n4,n2)
            self.assertAlmostEqual(got[n2,n4]/multiplicity,float(weight/total),places=13)

    def test_exclusive_partition(self):
        for n2,n4 in itertools.product(range(4),repeat=2):
            self.assertEqual(sum(m.in_stratum(n2,n4,s) for s in ('s0','s2','s4')),1)
            self.assertEqual(m.in_stratum(n2,n4,'s2'),
                             m.in_stratum(n2,n4,'s2_single') or m.in_stratum(n2,n4,'s2_multi'))

    def test_s4_allows_twos_and_multiple_fours(self):
        law=m.conditional_counts(4,(.5,.3,.2),'s4')
        self.assertIn((2,1),law.counts); self.assertIn((0,2),law.counts)
        self.assertIn((0,4),law.counts)

    def test_multi_not_forced_exactly_two(self):
        law=m.conditional_counts(4,(.5,.3,.2),'s2_multi')
        self.assertEqual(set(law.counts),{(2,0),(3,0),(4,0)})

    def test_tiny_s4_no_one_minus_cancellation(self):
        law=m.conditional_counts(4,(1.,1e-220,1e-250),'s4')
        self.assertTrue(math.isfinite(law.log_stratum_mass))
        self.assertAlmostEqual(sum(law.probabilities),1.)
        self.assertEqual(law.counts[law.probabilities.index(max(law.probabilities))],(0,1))

    def test_tiny_multi_no_underflow_normalization(self):
        law=m.conditional_counts(4,(1.,1e-250,0.),'s2_multi')
        self.assertLess(law.log_stratum_mass,-1000)
        self.assertTrue(math.isfinite(law.log_stratum_mass))
        self.assertAlmostEqual(max(law.probabilities),1.)

    def test_lost_tiny_cells_cannot_enter_production_adapter(self):
        law=m.conditional_counts(4,(1.,1e-250,0.),'s2_multi')
        self.assertTrue(law.underflowed_count_cells)
        with self.assertRaisesRegex(ValueError,'UNREPRESENTABLE'):
            m.require_representable(law)

    def test_zero_categories(self):
        self.assertEqual(m.conditional_counts(3,(0.,0.,1.),'s4').counts,((0,3),))
        with self.assertRaises(ValueError):m.conditional_counts(4,(1.,0.,0.),'s4')

    def test_position_uniformity_enumerated_rng(self):
        class PermutationRNG:
            def __init__(self,p):self.p=p
            def sample(self,population,n):return self.p[:n]
        counts={}
        for p in itertools.permutations(range(4)):
            word=m.place_orders(4,1,1,PermutationRNG(p))
            counts[word]=counts.get(word,0)+1
        self.assertEqual(len(counts),12)
        self.assertEqual(set(counts.values()),{2})

    def test_reproducible_synthetic_draws(self):
        law=m.conditional_counts(4,(.5,.3,.2),'s4')
        a,b=random.Random(123),random.Random(123)
        self.assertEqual([m.draw_orders(law,a) for _ in range(40)],
                         [m.draw_orders(law,b) for _ in range(40)])

    def test_q4_q8_separate_R(self):
        for q in (4,8):
            law=m.conditional_counts(q*4,(.5,.3,.2),'s4')
            orders=m.draw_orders(law,random.Random(8))
            m.validate_scope(orders,q=q,r=4,stratum='s4')
            with self.assertRaisesRegex(ValueError,'SCOPE'):
                m.validate_scope(orders,q=12-q,r=4,stratum='s4')

    def test_invalid_membership(self):
        with self.assertRaisesRegex(ValueError,'MEMBERSHIP'):
            m.validate_scope((2,)+(0,)*15,q=4,r=4,stratum='s2_multi')

    def test_nonfinite_probability(self):
        for values in ((float('nan'),0.,0.),(.5,-.1,.6),(.5,.3,.3)):
            with self.assertRaises(ValueError):m.conditional_counts(3,values,'s4')

    def test_invalid_R(self):
        for R in (0,-1,True,1.5):
            with self.assertRaises(ValueError):m.conditional_counts(R,(.5,.3,.2),'s4')

    def test_component_counts_and_provider_sequence(self):
        calls=[]
        def provider(k,indices,components):
            calls.append((k,indices,components))
            return dict(order=k,rotation=indices[0],products=indices[1:],
                        application=indices[1:]+indices[:1],quantum_weight='PINNED')
        out=m.component_events((0,2,4,0),('A','B'),(.25,.75),[random.Random(17)],4,provider)
        self.assertEqual([len(c[1]) for c in calls],[1,3,5,1])
        self.assertEqual([c['order'] for c in out],[0,2,4,0])
        self.assertTrue(all(c['quantum_weight']=='PINNED' for c in out))
        for c in out:self.assertEqual(c['application'],c['products']+(c['rotation'],))

    def test_component_probability_threshold(self):
        class Rng:
            def __init__(self):self.values=iter([.1,.9,.2,.8])
            def random(self):return next(self.values)
        out=m.component_events((2,0),('A','B'),(.25,.75),[Rng()],2,lambda k,i,c:i)
        self.assertEqual(out,((0,1,0),(1,)))

    def test_component_law_invalid(self):
        with self.assertRaises(ValueError):
            m.component_events((2,),('A',),(.5,.5),[random.Random(1)],1,lambda *x:x)

    def test_seed_stage_isolation_and_collision(self):
        a=m.proposed_seed('S4_PILOT','SYN_A',0)
        b=m.proposed_seed('S2_MULTI_PILOT','SYN_A',0)
        self.assertNotEqual(a,b)
        with self.assertRaisesRegex(ValueError,'COLLISION'):
            m.proposed_seed('S4_PILOT','SYN_A',0,[a])


def enum_test(R,stratum):
    def test(self):self.check_enumeration(R,stratum)
    return test
for _R in (1,2,3,4):
    for _s in ('s0','s2','s4','s2_single','s2_multi'):
        setattr(RareLawTests,f'test_exact_unconditional_enumeration_R{_R}_{_s}',enum_test(_R,_s))


def fixtures():
    metadata=dict(input_kind='SYNTHETIC',selected_before_confirmation=True,
                  cross_candidate_design='DISTINCT_STREAMS',within_trajectory_axes_paired=True,
                  exploration_sample_ids=['explore_A'],exploration_seeds=[1000])
    rows=[]
    for j,c in enumerate(('A','B')):
        for i in range(4):
            rows.append(dict(candidate=c,trajectory_id=str(i),sample_id=f'{c}_{i}',
                phase='confirmation',law='unconditional',seed=j*10+i,cosine=2+i+j,
                sine=3+i+j,event_digest_cosine=f'{c}_{i}',event_digest_sine=f'{c}_{i}'))
    return rows,metadata


class SelectionTests(unittest.TestCase):
    def validated(self):
        rows,meta=fixtures()
        return m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_phase_split_valid(self):
        self.assertEqual(set(self.validated()),{'A','B'})

    def test_distinct_estimands_artificial_selection(self):
        by=self.validated()
        result=m.synthetic_estimands(by,{'A':dict(cosine=2,sine=3),
                                         'B':dict(cosine=1,sine=1)},'A')
        self.assertEqual(result['fixed_before_confirmation'],'A')
        self.assertEqual(result['post_confirmation_point_selected'],'B')
        self.assertFalse(result['selected_population_mean_certified'])
        self.assertIsNone(result['unobserved_s4_mean'])

    def test_paired_SE_includes_within_axis_covariance(self):
        by=self.validated();shots={c:dict(cosine=1,sine=1) for c in by}
        result=m.synthetic_estimands(by,shots,'A')
        expected=math.sqrt(5/3) # pair-sum SD/sqrt(4), not independent axis SEs
        self.assertAlmostEqual(result['summaries']['A']['SE'],expected)

    def test_bootstrap_reuses_indices_for_all_endpoints(self):
        by=self.validated();one={c:dict(cosine=1,sine=1) for c in by}
        two={c:dict(cosine=2,sine=2) for c in by}
        out=m.synthetic_cluster_bootstrap(by,{'epsilon1':one,'epsilon2':two,'factor10':one},
                                         replicates=15,seed=17,dependence='DISTINCT_STREAMS')
        for b in out['replicates']:
            for c in by:
                self.assertEqual(b['epsilon2'][c],2*b['epsilon1'][c])
                self.assertEqual(b['epsilon1'][c],b['factor10'][c])
        self.assertFalse(out['tail_support_recovered'])
        self.assertFalse(out['simultaneous_coverage_certified'])

    def test_bootstrap_not_proved_dependence(self):
        with self.assertRaisesRegex(ValueError,'DEPENDENCE'):
            m.synthetic_cluster_bootstrap(self.validated(),{},replicates=2,seed=1,dependence='NOT_VERIFIED')

    def test_bounded_simultaneous_radius(self):
        a=m.bounded_simultaneous_radius(bound=10,n=32,family_size=1,alpha=.05,
                                       support_proof='PROVEN_ALL_SUPPORT_BOUND')
        b=m.bounded_simultaneous_radius(bound=10,n=32,family_size=6,alpha=.05,
                                       support_proof='PROVEN_ALL_SUPPORT_BOUND')
        self.assertGreater(b,a)

    def test_observed_max_not_support_bound(self):
        with self.assertRaisesRegex(ValueError,'ALL_SUPPORT'):
            m.bounded_simultaneous_radius(bound=10,n=32,family_size=6,alpha=.05,
                                         support_proof='OBSERVED_MAXIMUM')

    def test_confirmation_not_retroactively_holdout(self):
        rows,meta=fixtures();meta['selected_before_confirmation']=False
        with self.assertRaisesRegex(ValueError,'TIMING'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_shared_index_not_cross_candidate_pairing(self):
        rows,meta=fixtures();meta['cross_candidate_design']='PAIRED_BY_REPLICA_INDEX'
        with self.assertRaisesRegex(ValueError,'PAIRING_UNPROVED'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_unknown_covariance_remains_unknown(self):
        rows,meta=fixtures();meta['cross_candidate_design']='NOT_VERIFIED'
        self.assertEqual(len(m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)['A']),4)

    def test_seed_reuse_rejects_distinct_design(self):
        rows,meta=fixtures();rows[4]['seed']=rows[0]['seed']
        with self.assertRaisesRegex(ValueError,'SEED_REUSE'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_exploration_id_collision(self):
        rows,meta=fixtures();meta['exploration_sample_ids']=[rows[0]['sample_id']]
        with self.assertRaisesRegex(ValueError,'OVERLAP'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_exploration_seed_collision(self):
        rows,meta=fixtures();meta['exploration_seeds']=[rows[0]['seed']]
        with self.assertRaisesRegex(ValueError,'OVERLAP'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_missing_trajectory_is_not_zero(self):
        rows,meta=fixtures()
        with self.assertRaisesRegex(ValueError,'COUNT'):
            m.validate_selection_data(rows[:-1],('A','B'),expected_n=4,metadata=meta)

    def test_duplicate_endpoint_expansion_rejected(self):
        rows,meta=fixtures();rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_axis_identity_mismatch(self):
        rows,meta=fixtures();rows[0]['event_digest_sine']='other'
        with self.assertRaisesRegex(ValueError,'PAIRING'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)

    def test_shots_schema(self):
        with self.assertRaises(ValueError):m.paired_cost({},dict(cosine=1.2,sine=2))


def reject_pool_test(field,value):
    def test(self):
        rows,meta=fixtures();rows[0][field]=value
        with self.assertRaisesRegex(ValueError,'POOLING'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)
    return test
for _field,_v in (('phase','exploration'),('law','conditional_s2'),('law','s4'),('law','s2_multi')):
    setattr(SelectionTests,'test_no_pool_'+_v,reject_pool_test(_field,_v))

def bad_cost_test(value):
    def test(self):
        rows,meta=fixtures();rows[0]['cosine']=value
        with self.assertRaisesRegex(ValueError,'NOT_ZERO'):
            m.validate_selection_data(rows,('A','B'),expected_n=4,metadata=meta)
    return test
for _name,_v in [('missing',None),('nan',float('nan')),('negative',-1)]:
    setattr(SelectionTests,'test_bad_cost_'+_name,bad_cost_test(_v))


class NativeMockTests(unittest.TestCase):
    def setUp(self):
        self.binding=dict(input_kind='SYNTHETIC',SOURCE='mocksource',input_hash='mockinput',
                          compiler_hash='mockcompiler',profile='mockprofile')
        self.events=[dict(order=4,components=['x']*5)]

    def test_whole_wrapper_paired_compile(self):
        seen=[]
        def compiler(w):
            seen.append(w)
            self.assertEqual(w['scope'],'whole_Hadamard_wrapper')
            self.assertEqual(w['preparation_cost_P'],0)
            self.assertEqual(w['control'],'symmetric_directional')
            return dict(cost_kind='MEASURED_MOCK_WRAPPER',RZ=15,CX=9,depth=22)
        rows=m.mock_wrapper_pair(self.events,self.binding,self.binding.copy(),compiler)
        self.assertEqual([r['axis'] for r in rows],['cosine','sine'])
        self.assertEqual(rows[0]['event_digest'],rows[1]['event_digest'])
        self.assertEqual(len(seen),2)

    def test_model_unit_not_measured(self):
        with self.assertRaisesRegex(ValueError,'NOT_MODEL'):
            m.mock_wrapper_pair(self.events,self.binding,self.binding,
                                lambda w:dict(cost_kind='MODEL_ESTIMATE',RZ=15,CX=9,depth=22))

    def test_event_mutation(self):
        def compiler(w):
            self.events.append(dict(order=0))
            return dict(cost_kind='MEASURED_MOCK_WRAPPER',RZ=15,CX=9,depth=22)
        with self.assertRaisesRegex(ValueError,'MUTATION'):
            m.mock_wrapper_pair(self.events,self.binding,self.binding,compiler)

    def test_no_real_input(self):
        self.binding['input_kind']='MOLECULAR'
        with self.assertRaisesRegex(ValueError,'SYNTHETIC_ONLY'):
            m.mock_wrapper_pair(self.events,self.binding,self.binding,lambda w:None)

    def test_valid_grant_still_no_dispatch(self):
        with self.assertRaisesRegex(ValueError,'UNAVAILABLE'):
            m.authorization_gate(dict(authorization_status='APPROVED'),dict(valid=True))

    def test_invalid_grant(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL_REQUIRED'):
            m.authorization_gate(dict(authorization_status='UNAPPROVED'),dict(valid=False))

    def test_cli_rejects_before_missing_input_open(self):
        script=Path(__file__).resolve().parents[1]/'scripts/run_h8_rare_tail_selection_readiness.py'
        out=subprocess.run([sys.executable,'-B',str(script),'--execute',
                            '--config','/NONEXISTENT/config','--grant','/NONEXISTENT/grant'],capture_output=True,text=True)
        self.assertEqual(out.returncode,2)
        self.assertIn('APPROVAL_REQUIRED',out.stderr)
        self.assertNotIn('FileNotFoundError',out.stderr)

    def test_cli_builtin_fixture(self):
        script=Path(__file__).resolve().parents[1]/'scripts/run_h8_rare_tail_selection_readiness.py'
        out=subprocess.run([sys.executable,'-B',str(script),'--synthetic'],capture_output=True,text=True)
        self.assertEqual(out.returncode,0,out.stderr)
        self.assertIn('"scientific_calls": 0',out.stdout)


def bad_binding_test(key):
    def test(self):
        bad=dict(self.binding);bad[key]='tampered'
        with self.assertRaisesRegex(ValueError,'BINDING'):
            m.mock_wrapper_pair(self.events,bad,self.binding,lambda w:None)
    return test
for _key in ('SOURCE','input_hash','compiler_hash','profile'):
    setattr(NativeMockTests,'test_binding_'+_key,bad_binding_test(_key))


if __name__=='__main__':unittest.main()
