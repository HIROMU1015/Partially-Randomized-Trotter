"""B's pinned PR metadata API and invented H/lambda only; no scientific leaf."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_rpe_signal_plan as A
from trottertracks import pf_pr_rpe_pipeline as p

B_SOURCE='3216d27a69ac8532d2a22e78f5f94491d202d238'
B_PATH='src/trottertracks/pr_rpe_rmse_controller.py'


def b_api():
    # Reuse B function bodies; never expose cost/select or scientific providers.
    data=subprocess.check_output(['git','show',B_SOURCE+':'+B_PATH],cwd=ROOT)
    names={'_id','_positive','_ceil','paper_event_mapper','generate','_assemble','_validate'}
    nodes=[n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    exact=dict(Fraction=Fraction,math=math)
    raw=subprocess.check_output(['git','show',B_SOURCE+':src/trottertracks/h4_df10_rpe_connection.py'],cwd=ROOT)
    funcs=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('exact_scalar','exact_value')]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),B_SOURCE+':exact_values','exec'),exact)
    accounting=SimpleNamespace(exact_scalar=exact['exact_scalar'],exact_value=exact['exact_value'])
    scope=dict(p=p,a=accounting,deepcopy=deepcopy,hashlib=hashlib,json=json,
        SCHEMA='PR_RPE_RMSE_V2',METHODS=('PF','PR','FULL_H_RTE'),
        DRAFT_MAPPER='PAPER_TOTAL_UNIFORM_OUTER_DRAFT',EXPLICIT_LAYOUT='FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),B_SOURCE+':'+B_PATH,'exec'),scope)
    return SimpleNamespace(generate=scope['generate'],validate=scope['_validate'],exact_scalar=accounting.exact_scalar)


def fixture(*,prefix=1,K=2,formula='2nd',delta='1/10',kappa='2',M=12):
    source=dict(fixture='SYNTHETIC_ONLY');profile=deepcopy(p.PROFILE);profile['compiler']='MOCK_COMPILER'
    rational=lambda s:dict(numerator=Fraction(s).numerator,denominator=Fraction(s).denominator)
    spec=dict(method='PR',M=M,L_D=prefix,delta_pf=float(Fraction(delta)),formula=formula,h=.1,
        binding=dict(system='MOCK',rank=10,hamiltonian='MOCK_H',state='MOCK_S',snapshot='MOCK_SNAPSHOT',
            decomposition='MOCK_DF',source=source),
        split=dict(L_D=prefix,source=source,randomized_scope='TAIL_AFTER_PREFIX',lambda_R=rational('1/2'),
            lambda_source=source,one_body_policy='DETERMINISTIC',deterministic_scope='PREFIX'),
        law=dict(model='ideal_finite_RTE_law',source=source,sampling='MOCK_IDEAL'),
        kappa=dict(kind='PAPER_TOTAL_ROTATIONS',value=rational(kappa),definition_source=source),
        event_mapping=dict(kind='PAPER_TOTAL_UNIFORM_OUTER_DRAFT',source=source),finite_K=K,
        rmse_policy=dict(family='PR_PAPER_RMSE',id='MOCK_POLICY',source=source,target_rmse_ha=rational('1/1000'),
            error_contract=dict(fixture='MOCK_ONLY'),scientific_status='PROPOSED'),profile=profile)
    b=b_api();c=b.generate(spec);physical={}
    for row in c['rounds']:
        value=dict(system='MOCK',method='PR',formula=formula,prefix=prefix,T=float(b.exact_scalar(row['T'])),
            q=row['q_PF'],delta=float(b.exact_scalar(row['actual_delta_pf'])),R_outer=row['R_outer'],K=K)
        value['id']=p.candidate_id(value);physical[row['m']]=value
    recipe=dict(source=source,deterministic_PF='DIRECT_EXPM',finite_RTE='EXISTING_PAIRED_TAYLOR',sampling=False)
    return b,c,physical,recipe


class SignalPlanTests(unittest.TestCase):
    def test_all13_requests_are_passive_preserve_B_identity_and_nulls(self):
        b,c,physical,recipe=fixture();out=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        self.assertEqual(len(out['jobs']),13);self.assertEqual(out['science_calls'],0)
        self.assertFalse(out['cost_optimizer_implemented']);self.assertIsNone(out['winner'])
        self.assertNotIn('numpy',sys.modules)
        for job,row in zip(out['jobs'],c['rounds']):
            self.assertEqual(job['B_identity'],row['identity']);self.assertEqual(job['B_law_binding'],row['law_binding'])
            self.assertEqual(job['status'],'MISSING_MEAN_REQUIRES_PERMISSION')
            self.assertTrue(all(job[k] is None for k in ('N_X','N_Y','C','G','scientific_PASS_FAIL')))

    def test_split_K_formula_change_physical_context_no_foreign_reuse(self):
        b,c,physical,recipe=fixture();first=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        saved=dict(physical_context=first['jobs'][0]['physical_context'],source={'fixture':'MOCK_SAVED'},
            status='SAVED',signal_fields=['raw','corrected','exact_tail','reference','log_B','allowance_evidence'])
        for kwargs in (dict(prefix=7),dict(K=4),dict(formula='4th(m5_best)')):
            b,d,rows,recipe=fixture(**kwargs)
            out=A.plan(d,rows,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe,saved_inventory=[saved])
            self.assertTrue(all(x['status']=='MISSING_MEAN_REQUIRES_PERMISSION' for x in out['jobs']))

    def test_raw_reuse_does_not_relabel_new_M_schedule_or_N(self):
        b,c,physical,recipe=fixture(M=2)
        old=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)['jobs'][0]
        saved=dict(physical_context=old['physical_context'],source={'fixture':'ORIGINAL'},status='SAVED',
            signal_fields=['raw','corrected','exact_tail','reference','log_B','allowance_evidence'])
        b,new,physical,recipe=fixture(M=12)
        out=A.plan(new,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe,saved_inventory=[saved])
        self.assertEqual(out['jobs'][0]['status'],'SAVED_MEAN_METADATA_NEEDS_ACCEPTANCE')
        self.assertNotEqual(old['B_identity'],out['jobs'][0]['B_identity'])
        self.assertEqual(out['jobs'][0]['saved_source'],saved['source']);self.assertIsNone(out['jobs'][0]['N_X'])

    def test_fail_missing_and_duplicate_evidence_are_not_retried(self):
        b,c,physical,recipe=fixture();out=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        saved=dict(physical_context=out['jobs'][0]['physical_context'],source={'fixture':'MOCK'},status='NUMERICAL_FAILURE')
        x=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe,saved_inventory=[saved])
        self.assertEqual(x['jobs'][0]['status'],'SAVED_FAILURE_NO_AUTOMATIC_RETRY')
        saved['status']='SAVED'
        x=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe,saved_inventory=[saved])
        self.assertEqual(x['jobs'][0]['status'],'SAVED_EVIDENCE_INCOMPLETE_NO_FALLBACK')
        with self.assertRaises(ValueError):A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe,saved_inventory=[saved,saved])
        saved['N_X']=700
        with self.assertRaises(ValueError):A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe,saved_inventory=[saved])

    def test_B_rejects_PR_Yoshida_and_changed_configuration(self):
        with self.assertRaises(ValueError):fixture(formula='4th')
        b,c,physical,recipe=fixture();c['rounds'][-1]['q_PF']+=1
        with self.assertRaises(ValueError):A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)

    def test_terminal_only_foreign_law_and_numerical_recipe_rejected(self):
        b,c,physical,recipe=fixture()
        with self.assertRaises(ValueError):A.plan(c,{12:physical[12]},b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        wrong=deepcopy(physical);wrong[0]['R_outer']+=1;wrong[0]['id']=p.candidate_id(wrong[0])
        with self.assertRaises(ValueError):A.plan(c,wrong,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        recipe['sampling']=True
        with self.assertRaises(ValueError):A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        recipe['sampling']=False;recipe.pop('finite_RTE')
        with self.assertRaises(ValueError):A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)

    def test_q_RPE_and_actual_delta_early_grid_are_explicit(self):
        b,c,physical,recipe=fixture(delta='1/5')
        out=A.plan(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)
        self.assertEqual(out['jobs'][0]['q_RPE'],1)
        self.assertEqual(out['jobs'][1]['q_RPE'],2)
        self.assertEqual(out['jobs'][1]['q_PF'],1)
        self.assertEqual(physical[0]['delta'],.1)
        self.assertEqual(physical[1]['delta'],.2)
        wrong=deepcopy(physical);wrong[0]['delta']=.2;wrong[0]['id']=p.candidate_id(wrong[0])
        with self.assertRaises(ValueError):A.plan(c,wrong,b_validate=b.validate,b_exact_scalar=b.exact_scalar,numerical_recipe=recipe)


if __name__=='__main__':unittest.main()
