"""Plan structure/SHA checks and manual artificial accounting fixtures only.

No production analysis function or execution/grant switch exists here. Reuse
frozen pure accounting and AST-extracted statistics/guard on tiny invented
numbers; real H4 saved endpoints/costs are never sent to these functions.
"""
from pathlib import Path
import ast,csv,hashlib,importlib.util,json,math,subprocess,sys,unittest
from copy import deepcopy
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CASES=['pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR']
EPS=['0.05','0.01','0.005','0.001','0.0005']
AXES=('cosine','sine')

def load_json(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_draft(c):
    assert c['kind']=='H4_CONDITIONAL_CNG_SAVED_REANALYSIS_DRAFT_V1'
    assert all(c[k] is False for k in ('execution_authorized','grant_valid','dispatch_enabled','formal_certificate_accepted','N_G_authorized'))
    assert c['numerical_runner_SOURCE'] is None and c['runtime_manifest'] is None and c['grant_id'] is None
    assert c['cases']==CASES and c['epsilon_sig']==EPS and c['axes']==list(AXES)
    assert c['residual']['default'] is None and c['residual']['automatic_L_or_C_semantic_addition'] is False
    assert c['accounting']['maximum_u_over_headroom']=='0.01'
    assert c['automatic_retry'] is False and c['mandatory_STOP'] is True
    assert all(value==0 for value in c['forbidden_calls'].values())
    assert c['planned_reanalysis_budget']['landmark_fractions']==['0','1/4','1/2','3/4','1']


def extract_function(file,name,namespace):
    tree=ast.parse((ROOT/file).read_bytes())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/file),'exec'),namespace)
    return namespace[name]


def artificial_suite():
    spec=importlib.util.spec_from_file_location('plan_toy_frozen_account',ROOT/'src/trotterlib/fixed_signal_accounting.py')
    acc=importlib.util.module_from_spec(spec);spec.loader.exec_module(acc)
    import numpy as np
    namespace=dict(np=np,math=math,Fraction=Fraction,AXES=AXES,fixed_signal_shots=acc.fixed_signal_shots)
    stats=extract_function('src/trotterlib/fixed_signal_evaluation.py','_statistics',namespace)
    precision=extract_function('src/trotterlib/h2_non_df_pilot.py','precision_accounting',namespace)
    def shots(allowance=.001,kind='EMPIRICAL'):
        return acc.fixed_signal_shots(epsilon_sig=.2,bias=dict(cosine=.01,sine=.02),allowance={a:allowance for a in AXES},log_B_upper=0,evidence_kind=kind,evidence_ref='MANUALLY_INVENTED_ARTIFICIAL_ONLY')
    def toy_result(u):
        return dict(signal=dict(signed_bias=dict(real=.01,imag=.02)),shots=dict(evidence_ref='INVENTED_TOY_ONLY',axes={a:dict(allowance=u) for a in AXES}),normalization=dict(log_B_upper=0),circuit_cost=dict(kind='finite_MC_mean',axes={a:dict(mean_exact='5') for a in AXES},samples=[dict(RZ=dict(cosine=1,sine=9)),dict(RZ=dict(cosine=9,sine=1))]))
    class ArtificialPlanTests(unittest.TestCase):
        def test_unknown_residual_not_zero_filled(self):
            c=load_json(HERE/'execution_contract_DRAFT.json');self.assertIsNone(c['residual']['default'])
            self.assertIsNone(shots(None)['N_total'])
        def test_empirical_accounting_can_be_conditional_without_certificate(self):
            r=shots();self.assertTrue(r['usable']);self.assertFalse(r['certificate_verified'])
        def test_unknown_evidence_keeps_null(self):self.assertIsNone(shots(kind='UNKNOWN')['N_total'])
        def test_original_guard_stays_separate(self):
            plan=dict(evaluation=dict(alpha_axis=.025),numerical=dict(maximum_u_over_headroom=.01))
            r=precision(toy_result(.01),.2,plan)
            self.assertFalse(r['shots']['usable']);self.assertIsNone(r['total_RZ'])
        def test_paired_prices_keep_covariance(self):
            plan=dict(evaluation=dict(alpha_axis=.025),numerical=dict(maximum_u_over_headroom=.01))
            r=precision(toy_result(.0001),.2,plan)
            self.assertTrue(r['shots']['usable'])
            n=r['shots']['axes'];values=[n['cosine']['shots']*1+n['sine']['shots']*9,n['cosine']['shots']*9+n['sine']['shots']*1]
            self.assertEqual(r['total_RZ']['mean_exact'],str(Fraction(sum(values),2)))
            self.assertEqual(stats([10,10])['standard_error'],0)
        def test_guard_boundary_algebra(self):
            g=Fraction(1,100);s=Fraction(1,5);u0=Fraction(1,10000)
            rho=g*s/(1+g)-u0;u=u0+rho;h=s-u
            self.assertEqual(u/h,g);self.assertGreater(h,0)
        def test_unknown_tail_mean_is_not_zero(self):
            c=load_json(HERE/'execution_contract_DRAFT.json');self.assertTrue(c['statistical_plan']['no_population_ranking_without_conditions'])
            self.assertIn('symbolic',c['statistical_plan']['population_uncertainty'])
        def test_active_flags_and_zero_residual_default_rejected(self):
            c=load_json(HERE/'execution_contract_DRAFT.json')
            for key in ('execution_authorized','grant_valid','dispatch_enabled','N_G_authorized'):
                bad=deepcopy(c);bad[key]=True
                with self.assertRaises(AssertionError):validate_draft(bad)
            bad=deepcopy(c);bad['residual']['default']=0
            with self.assertRaises(AssertionError):validate_draft(bad)
    return unittest.defaultTestLoader.loadTestsFromTestCase(ArtificialPlanTests)


def main():
    contract=load_json(HERE/'execution_contract_DRAFT.json');validate_draft(contract)
    manifest=load_json(HERE/'inputs.json');catalog=load_json(HERE/'axis_conditions.json')
    assert sha(HERE/'inputs.json')==contract['inputs_manifest_SHA256']
    assert sha(HERE/'axis_conditions.json')==contract['axis_catalog_SHA256']
    for path,digest in manifest['files'].items():assert sha(ROOT/path)==digest,path
    triples=[(r['case'],r['epsilon_sig'],r['axis']) for r in catalog['axes']]
    assert len(triples)==40 and set(triples)=={(c,e,a) for c in CASES for e in EPS for a in AXES}
    for row in catalog['axes']:
        assert row['residual_default'] is None and row['old_N'] is None and row['old_G'] is None
        data=load_json(ROOT/row['C_result']);item=data['axes'][row['C_axis_pointer'][1]]
        assert (item['case'],str(item['epsilon_sig']),item['axis'])==(row['case'],row['epsilon_sig'],row['axis'])
    # Retain old precision-null state by reading flags only; no accounting call.
    for case in CASES:
        rows=load_json(ROOT/manifest['roles'][case+'_precision']);assert len(rows)==5
        assert all(row['shots']['N_total'] is None and row['total_RZ'] is None for row in rows)
    # Existing joins are preserved and structurally identifiable; scientific
    # event counts, covariance and rare mass aggregations are deliberately deferred.
    for case in CASES:
        d=load_json(ROOT/manifest['roles'][case+'_old']);samples=d['circuit_cost']['samples']
        assert len(samples)==(256 if case.endswith('_PR') else 1)
        assert len({s['trajectory_id'] for s in samples})==len(samples)
        for sample in samples:
            p=ROOT/Path(manifest['roles'][case+'_old']).parent/'trajectories'/sample['trajectory_id']/'price_join.json'
            joined=load_json(p)
            assert joined['compiled_RZ']==sample['RZ'] and joined['case_id']==case
            assert joined['event_record_hash']==sample['event_record_hash'] and joined['trajectory_record_hash']==sample['trajectory_record_hash']
    with (HERE/'artificial_tests.log').open('x') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(artificial_suite())
    report=dict(status='PASS_PLAN_ONLY' if result.wasSuccessful() else 'FAIL',artificial_tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),saved_files_SHA_checked=len(manifest['files']),axis_pointer_rows=40,paired_price_joins_structurally_checked=514,old20_precision_rows_null_unchanged=True,new_real_H4_shot_calls=0,new_real_H4_G_calls=0,new_saved_numeric_aggregations=0,new_sensitivity_evaluations=0,new_science_grants=0,artificial_numbers_only_passed_to_accounting=True,production_analysis_runner_implemented=False,scientific_certificate_accepted=False,mandatory_STOP=True)
    with (HERE/'verification.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report));return 0 if result.wasSuccessful() else 1

if __name__=='__main__':sys.exit(main())
