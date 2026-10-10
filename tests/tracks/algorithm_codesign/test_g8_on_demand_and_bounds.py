"""Off-domain G8 semantics/budget/cache tests; native backend is a stub."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from trottertracks.algorithm_codesign.g7_generator import make_generator, ARMS, DeterministicBits
from trottertracks.algorithm_codesign.g7_reference import reference_events
from trottertracks.algorithm_codesign.g8_bounds import (budget, provider_bias, delta_ceiling,
    accepted_cap, acceptance_upper, log_interval, p5_root_formula)
from trottertracks.algorithm_codesign.g8_pipeline import AcquisitionCache, NativePipeline
from trottertracks.algorithm_codesign.g8_reference_audit import oracle_IS_diagnostic

P, X = (F(1,7), F(2,7), F(4,7)), F(3,8)


def row(q):
    seq='WTStH'
    return {'sequence':seq,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),
            'strict_operator_error_upper':'1/100000000','T_count':2,'one_qubit_count':4}


class G8Tests(unittest.TestCase):
    def test_log_outward_not_float_budget(self):
        import mpmath as mp
        mp.mp.dps=150
        for q in (F(1),F(640),F(32000,49),F(8000)):
            iv=log_interval(q);truth=mp.log(mp.mpf(q.numerator)/q.denominator)
            self.assertLessEqual(mp.mpf(iv.lo.numerator)/iv.lo.denominator,truth)
            self.assertGreaterEqual(mp.mpf(iv.hi.numerator)/iv.hi.denominator,truth)

    def test_familywise_estimator_and_resource_failures_separate(self):
        plan=budget(make_generator(P,X,5,'full_return'))
        self.assertEqual(16*plan['alpha_axis']+8*plan['resource_failure_per_row'],F(1,20))
        from math import factorial
        self.assertGreater(sum(F(9)**k/factorial(k) for k in range(41)),8000)
        self.assertGreater(sum(F(7)**k/factorial(k) for k in range(41)),1000)

    def test_finite_provider_margin_and_exact_endpoint(self):
        from trottertracks.algorithm_codesign.g7_generator import budget as old_budget
        gen=make_generator(P,X,5,'full_return')
        self.assertEqual(provider_bias(5,gen.rho,F(1,10**6),0),old_budget(gen)['common_bias_upper'])
        ceiling=delta_ceiling(5,gen.rho,F(1,10**6))
        self.assertEqual(provider_bias(5,gen.rho,F(1,10**6),ceiling),F(1,200))
        with self.assertRaises(ArithmeticError): budget(gen,delta=ceiling)

    def test_acceptance_upper_without_support_table(self):
        for arm in ARMS:
            gen=make_generator(P,X,5,arm)
            z=sum(e['proposal'] for e in reference_events(gen))
            self.assertLessEqual(z,acceptance_upper(gen))

    def test_accepted_cap_satisfies_tail_inequality_and_hard_limit(self):
        for M,z,t in ((1000,F(1,3),9),(2028644,F(108,125),7),(10000,F(7,8),9)):
            cap=accepted_cap(M,z,t)
            excess=cap-M*z
            self.assertGreaterEqual(excess**2,2*t*(M*z+excess/3))
            self.assertLessEqual(cap,M)
        self.assertEqual(accepted_cap(100,F(1)),100)

    def test_p5_root_independent_raw_word_polynomial(self):
        from collections import defaultdict
        from math import factorial
        coefficients=defaultdict(lambda:[F(0),F(0)])
        for n in range(6):
            for word in product(range(3),repeat=n):
                remaining=list(word)
                while True:
                    i=next((i for i in range(len(remaining)-1) if remaining[i]==remaining[i+1]),None)
                    if i is None:break
                    del remaining[i:i+2]
                mass=X**n/factorial(n)
                for i in word:mass*=P[i]
                axis,sign=((0,1),(1,-1),(0,-1),(1,1))[n%4]
                coefficients[tuple(remaining)][axis]+=sign*mass
        a,s=p5_root_formula(P,X)
        self.assertEqual(a,coefficients[()][0])
        self.assertEqual(s,-sum(coefficients[(i,)][1] for i in range(3)))

    def test_production_never_imports_reference_or_saved_cost_inventory(self):
        path=Path(__file__).resolve().parents[3]/'src/trottertracks/algorithm_codesign/g8_pipeline.py'
        code=path.read_text()
        for forbidden in ('reference_events','g8_reference_audit','inventory','B_new','itertools','result_v1.json'):
            self.assertNotIn(forbidden,code)

    def test_zero_draw_requests_no_native_or_provider(self):
        class Zero:
            def sample(self,bits):return None
        cache=AcquisitionCache(lambda q:self.fail('synthesis on zero'))
        pipe=NativePipeline(Zero(),cache,{'hard_attempt_cap_two_axes':100,'accepted_call_cap_two_axes':10},F(1,10**6))
        self.assertIsNone(pipe.step(None));self.assertEqual(cache.requests,0)
        self.assertEqual(pipe.counters()['zeros'],1)

    def test_live_requests_only_and_reuse(self):
        calls=[]
        cache=AcquisitionCache(lambda q:(calls.append(q),row(q))[1])
        self.assertEqual(cache.calls,0)
        cache.get('1/7');cache.get('1/7')
        self.assertEqual(calls,[F(1,7)]);self.assertEqual(cache.hits,1)

    def test_failed_miss_poisoned_no_retry(self):
        calls=[]
        def fail(q):calls.append(q);raise TimeoutError('stub failure')
        cache=AcquisitionCache(fail)
        with self.assertRaises(TimeoutError):cache.get('1/7')
        with self.assertRaises(RuntimeError):cache.get('1/7')
        self.assertEqual(len(calls),1)

    def test_acquisition_capacity_is_hard_not_backend_search(self):
        cache=AcquisitionCache(row,capacity=1)
        cache.get('1/7')
        with self.assertRaises(RuntimeError):cache.get('2/7')
        self.assertEqual(cache.calls,1)

    def test_lru_eviction_does_not_resynthesize(self):
        calls=[];cache=AcquisitionCache(lambda q:(calls.append(q),row(q))[1])
        pipe=NativePipeline(None,cache,{},F(1,10**6),capacity=2)
        for q in ('1/7','2/7','3/7','1/7'):pipe._lookup(q)
        self.assertEqual(len(calls),3);self.assertEqual(pipe.evictions,2)
        self.assertEqual(len(pipe.cache),2)

    def test_cache_byte_cap(self):
        cache=AcquisitionCache(row,byte_cap=1)
        with self.assertRaises(MemoryError):cache.get('1/7')

    def test_attempt_and_accepted_caps_are_distinct(self):
        gen=make_generator(P,X,3,'ordinary');cache=AcquisitionCache(row)
        plan={'hard_attempt_cap_two_axes':100,'accepted_call_cap_two_axes':1}
        pipe=NativePipeline(gen,cache,plan,F(1,10**6))
        bits=DeterministicBits('G8-off-domain')
        pipe.step(bits)
        with self.assertRaises(RuntimeError):pipe.step(bits)
        self.assertEqual(pipe.accepted,1)

    def test_conditional_event_error_uses_target_call_count(self):
        gen=make_generator(P,X,5,'full_return');plan=budget(gen)
        pipe=NativePipeline(gen,AcquisitionCache(row),plan,F(1,10**6))
        bits=DeterministicBits('G8-off-domain')
        output=next(v for v in (pipe.step(bits) for _ in range(100)) if v is not None)
        n=sum(output['event']['provider_calls'].values())
        self.assertEqual(output['conditional_event_error_upper'],F(2,10**8)+n*F(1,10**6))
        bound=2*F(1,10**6)+(gen.m+1)*F(1,10**6)
        self.assertLessEqual(output['conditional_event_error_upper'],bound)

    def test_trace_same_events_as_original_local_generator(self):
        for arm in ARMS:
            direct=make_generator(P,X,5,arm);gen=make_generator(P,X,5,arm)
            a,b=DeterministicBits('G8-off-domain'),DeterministicBits('G8-off-domain')
            pipe=NativePipeline(gen,AcquisitionCache(row),budget(gen),F(1,10**6))
            for _ in range(24):
                actual=pipe.step(a);expected=direct.sample(b)
                self.assertEqual(None if actual is None else actual['event'],expected)

    def test_oracle_IS_is_coefficient_preserving_diagnostic_not_runtime(self):
        for arm in ARMS:
            gen=make_generator(P,X,5,arm);cache={str(e['ratio']):row(e['ratio']) for e in reference_events(gen)}
            report=oracle_IS_diagnostic(gen,cache,budget(gen))
            self.assertTrue(report['all_coefficients_preserved_exactly'])
            self.assertFalse(report['production_law_changed'])
            self.assertEqual(report['quantum_sampling_or_synthesis_calls'],0)
            self.assertGreaterEqual(report['T_Rz_two_axes'],report['fixed_Bernstein_policy_T_Rz_lower_for_all_positive_proposals'])


if __name__=='__main__':unittest.main()
