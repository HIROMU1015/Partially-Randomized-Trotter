"""Artificial checks for the LD2 B_PRICE re-allocation caller; never real H4 risk."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_total_binding import pauli,MOCK,T
spec=importlib.util.spec_from_file_location('_ld2_reallocation',ROOT/'scripts/run_pauli_ld2_reallocation_q3_fixed.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def b_generate():
    """B's adopted generate() from its pinned SOURCE (integer prices only)."""
    import ast
    tree=ast.parse(M.R.blob(*M.B_GENERATOR))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    scope=dict(Fraction=Fraction,exact=lambda v:v)
    def integer(value):
        if type(value) is not int or value<=0:raise ValueError('Positive exact integer required')
        return value
    scope['integer']=integer
    exec(compile(ast.Module(body=[node],type_ignores=[]),'B_generate','exec'),scope)
    return scope['generate']


def artificial_case(label):
    args,_=pauli();bridge=T.create(ROOT,args['config'],original_source=MOCK)
    case=dict(label=label,proposal='ART',original_source=MOCK,binding_id=bridge['id'],allocation=args['allocations'][0],
        evaluation_target=args['evaluation_target'],conditional_cost=M.exact(Fraction(7,3)),budget=M.exact(3),
        cost_m0_m9=M.exact(1),cost_m10_m12=M.exact(Fraction(4,3)))
    inputs=dict(statistical_contract=args['statistical_contract'],options=args['options'],
        acceleration=dict(workers_per_configuration=2,outcome_batch=1,source=MOCK),
        adopted_evaluation=dict(MSE_upper=999,epsilon_E_decimal='ARTIFICIAL_ONLY'),concurrent_configurations=2)
    return args,case,inputs


class Generation(unittest.TestCase):
    def test_matches_B_generator_on_integer_prices(self):
        reference=b_generate();rng=random.Random(17)
        for _ in range(60):
            n=rng.randint(2,8);caps=[rng.randint(1,12) for _ in range(n)];prices=[rng.randint(1,10**6) for _ in range(n)]
            budget=rng.randint(sum(prices)//2,sum(c*p for c,p in zip(caps,prices))+5)
            ours=M.generate(caps,prices,budget,'B_PRICE');theirs=reference(caps,prices,budget,'B_PRICE')
            if theirs is None:self.assertIsNone(ours);continue
            self.assertEqual(ours['shots'],theirs['shots'])
            self.assertEqual(ours['conditional_cost'],theirs['conditional_cost'])
            self.assertEqual(ours['unused_budget'],theirs['unused_budget'])

    def test_rational_prices_bounds_budget_and_maximality(self):
        caps=[48,45,42,9];prices=[Fraction(2226,1),Fraction(41672,10),Fraction(10**8,7),Fraction(3,1)]
        total=sum(c*p for c,p in zip(caps,prices))
        for budget in (sum(prices),total/3,total-1,total):
            g=M.generate(caps,prices,budget,'B_PRICE')
            self.assertTrue(all(1<=n<=c for n,c in zip(g['shots'],caps)))
            self.assertLessEqual(g['conditional_cost'],budget)
            self.assertEqual(g['unused_budget'],budget-g['conditional_cost'])
            if g['next_breakpoint_cost'] is not None:self.assertGreater(g['next_breakpoint_cost'],budget)
        self.assertEqual(M.generate(caps,prices,total,'B_PRICE')['shots'],caps)
        self.assertIsNone(M.generate(caps,prices,sum(prices)-1,'B_PRICE'))
        with self.assertRaises(ValueError):M.generate(caps,[1,0,1,1],10)
        with self.assertRaises(ValueError):M.generate(caps,prices,10,'OTHER')

    def test_count_bounds_match_existing_route_for_equal_axes(self):
        vector=[30,20,10,5,4,3,2,1,1,1,1,1,1]
        shots=[dict(m=m,N_X=n,N_Y=n)for m,n in enumerate(vector)]
        mine=M.count_bounds(shots);old=M.Q.finite_bounds(vector)
        for key in ('maximum_round_outcomes','verification_pairs_upper','transition_pairs_upper'):
            self.assertEqual(mine[key],old[key])


class PauliRoutePersistence(unittest.TestCase):
    def test_worker_persists_api_result_and_resources_on_real_route(self):
        args,case,inputs=artificial_case('ART_LD1')
        with tempfile.TemporaryDirectory()as directory,patch.object(M.R.L.A,'accept_saved',return_value=dict(
                results=[dict(configuration=args['config'],signal_packet=args['signal_packet'])])):
            receipt=M.worker(case,inputs,MOCK,directory)
            out=Path(directory)
            for name in ('API_RETURN.json','RESULT.json','RESOURCE_FINAL.json','PERSISTED.json','START_ART_LD1.json'):
                self.assertTrue((out/name).exists(),name)
            result=json.loads((out/'RESULT.json').read_text())
            self.assertEqual(result['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
            self.assertEqual(result['scientific_evaluation_calls'],1);self.assertTrue(result['original_api_return_saved'])
            self.assertEqual(result['record'],json.loads((out/'API_RETURN.json').read_text())[0])
            self.assertEqual(receipt['operational_status'],'WORKER_ARTIFACTS_SAVED')
            self.assertIsNone(result['sufficient_N']);self.assertIsNone(result['G'])

    def test_supervisor_saves_workers_and_stops_unstarted_calls_on_monitor_error(self):
        args,a,inputs=artificial_case('ART_LD1');_,b,_=artificial_case('BRT_LD1')
        accepted=dict(results=[dict(configuration=args['config'],signal_packet=args['signal_packet'])])
        def broken(*_):raise RuntimeError('artificial monitor failure')
        with tempfile.TemporaryDirectory()as directory,patch.object(M.R.L.A,'accept_saved',return_value=accepted):
            out=Path(directory)
            for c in (a,b):(out/c['label']).mkdir()
            result=M.supervise([a,b],inputs,MOCK,out,monitor=broken,poll_seconds=0.001)
            self.assertTrue((out/'MONITOR_ERROR.json').exists());self.assertEqual(result['errors'],[])
            for c in (a,b):
                persisted=json.loads((out/c['label']/'PERSISTED.json').read_text())
                self.assertLessEqual(persisted['scientific_evaluation_calls'],1)
                if persisted['status'].startswith('UNKNOWN'):self.assertEqual(persisted['scientific_evaluation_calls'],0)
                self.assertTrue((out/c['label']/'SUPERVISOR_RECEIPT.json').exists())

    def test_supervisor_normal_path(self):
        args,a,inputs=artificial_case('ART_LD1');_,b,_=artificial_case('BRT_LD1')
        accepted=dict(results=[dict(configuration=args['config'],signal_packet=args['signal_packet'])])
        calls=[]
        with tempfile.TemporaryDirectory()as directory,patch.object(M.R.L.A,'accept_saved',return_value=accepted):
            out=Path(directory)
            for c in (a,b):(out/c['label']).mkdir()
            result=M.supervise([a,b],inputs,MOCK,out,monitor=lambda d,l,s,p:calls.append(l),poll_seconds=0.001)
            self.assertEqual(result['errors'],[]);self.assertEqual(result['monitor_errors'],[])
            self.assertEqual(sorted(r['status']for r in result['receipts']),['MODEL_BOUND_MEETS_ADOPTED_THRESHOLD']*2)


class FixedReferences(unittest.TestCase):
    def test_pinned_inputs_resolve(self):
        for pair in (M.VP_RESULT,M.V0_RESULT,M.PRICES,M.TREF,M.B_GENERATOR,M.PF_MINIMA):
            self.assertTrue(M.R.blob(*pair))
        fixed=M.document(M.Q3_FIXED,M.Q3_INPUT);ld2=next(c for c in fixed['cases'] if c['label']=='LD2')
        self.assertEqual([r['N_X'] for r in ld2['allocation']['shots']],M.VP)


if __name__=='__main__':unittest.main(verbosity=2)
