"""Synthetic regression for exact context reuse and censored public summaries."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import replay_rpe_candidate_ordering as R


class ReplayIOTests(unittest.TestCase):
    def test_fixed_input_blobs_ignore_live_points_and_materialize_both_registries(self):
        problem=dict(methods={'PR_PAULI':{}},execution_routes={},price_rules={})
        spec=dict(methods=['PR_PAULI'],method_overrides={'PR_PAULI':dict(generation=dict(kappa_points_file='live/points.json'))},
            routes_override={'PR_PAULI':[dict(configurations_file='live/routes.json')]},
            price_rules_overrides=dict(registry_file='live/prices.json'))
        def blob(commit,path):
            self.assertEqual(commit,R.ROUND05_SOURCE)
            if path.endswith('h4_df10_problem.json'):return deepcopy(problem)
            if path.endswith('h4_pr_pauli_schedule_wide_round05.json'):return deepcopy(spec)
            if path.endswith('KAPPA_STATE.json'):return dict(family_constants={})
            if path=='saved/points.json':return [dict(L_D=5,kappa='2')]
            return {'live/routes.json':{'fixed_route':1},'live/prices.json':{'fixed_price':2}}[path]
        def tree(args,**kwargs):
            self.assertEqual(args[:3],['git','ls-tree',R.ROUND05_SOURCE])
            return b'' if args[-1]=='live/points.json' else b'fixed blob'
        with tempfile.TemporaryDirectory()as tmp:
            baseline=Path(tmp)/'baseline';(baseline/'live').mkdir(parents=True)
            for name in ('points','routes','prices'):(baseline/'live'/f'{name}.json').write_text('{"future":999}')
            with patch.object(R.O.Reader,'document',side_effect=blob),patch.object(R.subprocess,'check_output',side_effect=tree):
                pp,rp,constants=R.fixed_metadata_inputs(baseline,Path(tmp)/'own')
            result=json.loads(rp.read_text());g=result['method_overrides']['PR_PAULI']['generation']
            self.assertNotIn('kappa_points_file',g);self.assertEqual(g['kappa_points'],[])
            self.assertEqual(json.loads(Path(result['routes_override']['PR_PAULI'][0]['configurations_file']).read_text()),{'fixed_route':1})
            self.assertEqual(json.loads(Path(result['price_rules_overrides']['registry_file']).read_text()),{'fixed_price':2})
            self.assertEqual(json.loads((baseline/'live/prices.json').read_text()),{'future':999})
            with patch.object(R.O.Reader,'document',side_effect=blob),patch.object(R.subprocess,'check_output',side_effect=tree):
                _,archived,_=R.fixed_metadata_inputs(baseline,Path(tmp)/'declared',prepared_points='saved/points.json')
            declared=json.loads(archived.read_text())['method_overrides']['PR_PAULI']['generation']
            self.assertEqual(declared['kappa_points'],[dict(L_D=5,kappa='2')])

    def data(self):
        return dict(round05_source='d8e806265c3ded8e1184181c37c2d7ec71f08ade',
            candidates=[dict(configuration_id='ARTIFICIAL',prices=['3/7']*26)],
            initial_records=[dict(configuration_id='ARTIFICIAL',vector=[2]*26,
                record=dict(status=R.O.PASS,MSE_lower=F(1,4),MSE_upper=F(1,3),cost=F(100,7),calls=0))],
            family_constants={'5':[dict(m=m,c='1',q=2**m,J=1)for m in range(13)]},tau='1',
            generator=dict(commit='ARTIFICIAL',path='unused',function='unused'),spec={'fixture':True},search={'fixture':True})

    def test_cache_keeps_prices_and_exact_initial_bounds_and_reuses_without_preparation(self):
        with tempfile.TemporaryDirectory()as tmp:
            cache=Path(tmp)/'cache.json';R.save_context(cache,self.data())
            raw=json.loads(cache.read_text());self.assertEqual(raw['candidates'][0]['prices'],['3/7']*26)
            with patch.object(R.O,'pinned_function',return_value='ARTIFICIAL_GENERATOR'), \
                 patch.object(R.T,'round_context',side_effect=AssertionError('cache reran metadata')):
                spec,search,cands,seeds,families,tau,generate=R.context_inputs(Path(tmp),cache,Path(tmp)/'unused')
            self.assertEqual(cands[0]['prices'],[F(3,7)]*26)
            self.assertEqual(seeds[('ARTIFICIAL',tuple([2]*26))]['MSE_upper'],F(1,3))
            self.assertEqual(seeds[('ARTIFICIAL',tuple([2]*26))]['cost'],F(100,7))
            self.assertEqual(generate,'ARTIFICIAL_GENERATOR')
            self.assertFalse((Path(tmp)/'unused').exists())

    def test_missing_price_cache_is_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory()as tmp:
            data=self.data();del data['candidates'][0]['prices'];cache=Path(tmp)/'bad.json'
            R.save_context(cache,data)
            with self.assertRaisesRegex(ValueError,'price vectors'):
                R.context_inputs(Path(tmp),cache,Path(tmp)/'unused')

    def decision(self,chosen):
        return dict(selected=chosen,selection=dict(baseline_id='A',chosen_id=chosen,
            ranks=[dict(id='A',large_diagnostic=F(1,10**100)),dict(id='B'),dict(id='C')]))

    def test_compact_trace_preserves_pool_count_and_shared_first_divergence(self):
        a,b=self.decision('A'),self.decision('B')
        policies={name:dict(decisions=[d],censor=None,seconds=0,curve=[dict(seconds=0,best_all=F(100),launched=0)])
                  for name,d in [('A',a),('B',b)]}
        data=dict(results={'case':dict(policies=policies,paired_observable_horizon=0,first_divergence=dict(A=a,B=b))})
        before=deepcopy(data);out=R.compact_output(data)
        self.assertEqual(data,before)
        x=out['results']['case'];self.assertEqual(x['first_divergence']['A']['selection']['scored_candidates'],3)
        self.assertEqual(len(x['policies']['B']['decisions'][0]['selection']['ranks']),2)
        self.assertEqual(x['policies']['A']['curve'][0]['best_all'],100)

    def test_elapsed_table_never_extrapolates_past_censor(self):
        comparison=dict(paired_observable_horizon=0,policies={
            'A':dict(censor={'seconds':0},curve=[dict(seconds=0,best_all=100,launched=0)]),
            'B':dict(censor={'seconds':3},curve=[dict(seconds=0,best_all=100,launched=0),dict(seconds=2,best_all=90,launched=1)])})
        rows=R.equal_elapsed(comparison,times=(0,2,4))
        self.assertTrue(rows[0]['paired_observable'])
        self.assertFalse(rows[1]['paired_observable'])
        self.assertEqual(rows[1]['policies']['A']['status'],'UNKNOWN_AFTER_CENSOR')
        self.assertIsNone(rows[1]['policies']['A']['best_all'])
        self.assertEqual(rows[1]['policies']['B']['best_all'],90)
        self.assertEqual(rows[2]['policies']['B']['status'],'UNKNOWN_AFTER_CENSOR')


if __name__=='__main__':unittest.main(verbosity=2)
