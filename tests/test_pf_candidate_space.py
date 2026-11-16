"""Canonical PF prefix metadata only; no H4 restore, operators or evaluation."""
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_candidate_space as P


class CandidateSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=P.C.blob(ROOT,P.T2_SOURCE,P.T2_DIR+'CONFIGURATIONS.csv').decode()
        cls.saved_configs=[(r,P.configure(ROOT,'B0I',int(r['prefix']),r['formula'],r['requested_delta_over_h']))
                           for r in csv.DictReader(io.StringIO(raw))]

    def test_twenty_full_canonical_ids_are_unchanged(self):
        old=json.loads(P.C.blob(ROOT,'438d6c9a43351d45c31921a5676359dbc2c80b32',
            'docs/research/h4_pf_optimizer_search_20261116/CANDIDATES_AT_START.json'))
        expected={r['configuration_id']for r in old if r['method']=='FULL_PF'}
        rows=P.candidates(ROOT,p_values=[10])
        self.assertEqual(len(rows),20)
        self.assertEqual({r['configuration_id']for r in rows},expected)

    def test_saved_p7_p8_p9_ids_and_all_42_step_locators(self):
        seen=set()
        for row,cfg in self.saved_configs:
            self.assertEqual(cfg['id'],row['configuration_id'])
            evidence=P.priority_evidence(ROOT,int(row['prefix']),row['formula'],cfg)
            for cell in evidence['T2']:
                source=cell['scope_source'];seen.add((source['commit'],source['path']))
        self.assertEqual(len(seen),42)
        for commit,path in seen:
            scope=json.loads(P.C.blob(ROOT,commit,path))
            self.assertIn(scope['prefix'],(7,8,9))

    def test_p_zero_and_ten_alias_keep_prefix_identity(self):
        configs=[P.configure(ROOT,'B0I',p,'2nd','1')for p in range(11)]
        self.assertEqual(len({c['id']for c in configs}),11)
        for p,c in enumerate(configs[:10]):
            self.assertEqual(c['spec']['L_D'],p)
            self.assertEqual(c['spec']['split']['retained_hamiltonian']['prefix'],p)
            self.assertEqual(c['spec']['split']['original_PF_method'],'B0I')
            self.assertEqual(c['spec']['split']['retained_hamiltonian']['target_hamiltonian'],c['spec']['binding']['hamiltonian'])
            self.assertEqual(c['spec']['binding']['hamiltonian'],configs[10]['spec']['binding']['hamiltonian'])
        self.assertEqual(configs[0]['rounds'][0]['candidate']['prefix'],0)
        self.assertEqual(configs[10],P.configure(ROOT,'FULL_PF',10,'2nd','1'))

    def test_arbitrary_positive_width_signed_pieces_and_compiler(self):
        c=P.configure(ROOT,'B0I',8,'4th','11/3')
        self.assertEqual(P.exact(c['spec']['delta_pf']),P.H*Fraction(11,3))
        self.assertTrue(any(w<0 for w in c['rounds'][0]['structure']['signed_weights']))
        profile=c['spec']['profile']
        expected=dict(P=0,qiskit_version='1.3.0',control='symmetric_directional',
            wrapper='full_Hadamard_with_measurement',basis_gates=['rz','sx','x','cx'],
            optimization_level=1,seed_transpiler=17)
        self.assertTrue(all(profile[k]==v for k,v in expected.items()))

    def test_saved_t1_features_never_veto_a_prefix(self):
        rows=P.candidates(ROOT,formulas=['2nd'],widths=['1'],p_values=range(10))
        self.assertEqual({r['p']for r in rows},set(range(10)))
        for row in rows:
            evidence=row['priority_evidence'];t1=evidence['T1']
            self.assertEqual(evidence['usage'],'PRIORITY_ONLY_NOT_Q3_ACCEPTANCE_OR_REJECTION')
            saved=json.loads(P.C.blob(ROOT,t1['source']['commit'],t1['source']['path']))
            self.assertEqual(t1['delta_E_trunc_B0I_ha'],saved['delta_E_trunc_B0I_ha'])
            self.assertEqual(t1['tail_identity_scalar'],saved['tail_coefficients']['used_c_tail'])
            self.assertEqual(t1['lambda_R'],saved['tail_coefficients']['used_lambda_R'])

    def test_metadata_does_not_open_arrays_or_execute(self):
        original=P.C.blob
        def metadata(root,commit,path):
            if path.endswith(('.npz','.qpy')):raise AssertionError('Scientific array/circuit read')
            return original(root,commit,path)
        with patch.object(P.C,'blob',metadata),patch.object(P.C,'execute',side_effect=AssertionError('Scientific call')):
            rows=P.candidates(ROOT,formulas=['2nd'],widths=['1','1/1',dict(numerator=3,denominator=2)],p_values=[0,7,10])
        self.assertEqual(len(rows),6)
        self.assertEqual(len({r['configuration_id']for r in rows}),6)
        rows[0]['configuration_source']['base']['selector'].append('MUTATED')
        self.assertEqual(P.BASE_SPEC['selector'],['spec'])

    def test_invalid_scope_is_rejected(self):
        for method,p,formula,width in (('FULL_PF',7,'2nd','1'),('B0',7,'2nd','1'),
                ('B0I',-1,'2nd','1'),('B0I',True,'2nd','1'),('B0I',11,'2nd','1'),
                ('B0I',7,'8th','1'),('B0I',7,'2nd','0'),('B0I',7,'2nd',True)):
            with self.assertRaises(ValueError):P.configure(ROOT,method,p,formula,width)
        for values in ([7,7],[-1],[True]):
            with self.assertRaises(ValueError):P.candidates(ROOT,p_values=values)

    def test_scoped_choice_is_independent_of_incumbent_formula_width(self):
        from copy import deepcopy
        from trottertracks import pf_candidate_selection as S,pf_optimizer_connection as PF
        from trottertracks.saved_rpe_cost_catalog import Reader
        reader=Reader(ROOT)
        problem=json.loads((ROOT/'configs/rpe_method_optimizer/h4_df10_problem.json').read_text())
        self.assertEqual(problem['methods']['TRUNCATED_PF']['p_values'],list(range(10)))
        self.assertEqual(problem['methods']['FULL_PF']['p_values'],[10])
        catalog=reader.document('fee61d272c3564ef393fecef5b738dbda825b658',
            'docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json')
        self.assertEqual(S.evaluated_prefixes(dict(catalog=catalog)),[7])
        choices=[]
        for formula,width in (('2nd','2'),('4th','4')):
            scoped=deepcopy(problem)
            for method in ('TRUNCATED_PF','FULL_PF'):
                scoped['methods'][method].update(formulas=[formula],delta_over_h_range=[width,width])
            candidates=[]
            PF.enumerate_prefixes(ROOT,reader,scoped,candidates,methods=['TRUNCATED_PF','FULL_PF'])
            self.assertEqual(len(candidates),11)
            self.assertEqual(len({c['configuration_id']for c in candidates}),11)
            self.assertEqual({c['key']['p']for c in candidates if c['method']=='TRUNCATED_PF'},set(range(10)))
            self.assertEqual([c['key']['p']for c in candidates if c['method']=='FULL_PF'],[10])
            selected=S.select(dict(catalog=catalog,candidates=candidates),['TRUNCATED_PF'])
            self.assertEqual(len(selected['candidates']),10)
            self.assertEqual(selected['selected']['key']['formula'],formula)
            self.assertEqual(Fraction(selected['selected']['key']['delta_over_h']),Fraction(width))
            self.assertEqual(selected['saved_incumbent']['conditional_cost'],dict(numerator=372035410,denominator=1))
            self.assertEqual(selected['priority_scope'],'REQUEST_ORDER_ONLY_NOT_Q3_REJECTION_OR_PRICE_BOUND')
            self.assertEqual(selected['unevaluated_prefixes'],[0,1,2,3,4,5,6,8,9])
            choices.append(selected['selected']['configuration_id'])
        self.assertNotEqual(*choices)
        # The research plan preserves its frozen raw problem and applies the
        # explicit round scope through the same helper as the normal CLI.
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_rpe_method_optimizer as CLI
        directory=ROOT/'docs/research/b0i_prefix_auto_selection_20261117'
        raw_problem=json.loads((directory/'PROBLEM.json').read_text())
        plan_round=json.loads((directory/'SAVED_PLAN_ROUND.json').read_text())
        actual_problem=CLI.scoped(raw_problem,plan_round.get('method_overrides'))
        self.assertEqual(actual_problem['methods']['TRUNCATED_PF']['p_values'],list(range(11)))
        self.assertFalse((plan_round.get('pf_completion')or{}).get('selected'))
        actual=[]
        PF.enumerate_prefixes(ROOT,reader,actual_problem,actual,methods=plan_round['methods'])
        self.assertEqual({c['key']['p']for c in actual},set(range(10)))
        self.assertEqual(len({c['configuration_id']for c in actual}),len(actual))
        decision=S.select(dict(catalog=catalog,candidates=actual),plan_round['methods'])
        self.assertEqual(decision['evaluated_prefixes'],[7])
        self.assertEqual(decision['selected']['key'],dict(p=9,formula='4th',delta_over_h='32/5'))


if __name__=='__main__':unittest.main(verbosity=2)
