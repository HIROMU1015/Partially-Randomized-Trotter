"""Metadata-only long-q/R price proposal. No scientific execute entry point.

Reuse B's canonical menu and C's descriptor. Do not load prices, molecular
inputs, the science backend, or call any fitting/prediction/accounting API.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import r_price_phase1 as c1
from trottertracks.r_variable_compiled_price import describe_request

B = '724d313b9c8445763f0fe9c41feb573df28ba5d5'
A_PLAN = 'd33380a8c2fe68d0be1f2c5faaf7fc3b3a956b58'
A_RESULTS = '71ba49bb28b295aa28e51101bc3b1bf07d500767'
C1 = '2e6a0cdf5a647058cc7afeeb5741bfafa24c974b'
FROZEN = '7eeabf0ae482ee610d667e28470692d209f2bb21'
TASKS = '3b742fff26381514e12c07b51701276b7d1b26ec'
OUT = 'docs/research/r_price_longq_largeR_preparation_20261111'
# Finite proposals, not a global menu, execution order, or scientific adoption.
PACKAGES = (
    ('Q', [('H4', 6, 4), ('H6', 8, 16)], 4,
     'A-P1 high-q baseline component of the primary direct-price proposal'),
    ('R', [('H4', 3, 128), ('H6', 4, 128)], 4,
     'R-only transfer at the C1 low-q anchor; allocation/law changes included'),
    ('QR', [('H4', 6, 128), ('H6', 8, 128)], 6,
     'joint q/R transfer; interpret with Q and R and the saved C1 anchor'),
    ('SHORT', [('H4', 0, 4), ('H6', 4, 11)], 4,
     'optional short-q baseline and minimum legal H6 allocation'),
)


def ref(commit, path):
    return dict(commit=commit, path=path)


def make_row(menu, row, request, package, n_paths, anchors):
    desc = describe_request(menu, row, 'X', pipeline=p)
    candidate = row['candidate']
    return dict(system=candidate['system'], m=row['m'], T=candidate['T'],
        q_PF=row['q_PF'], q_RPE=row['q_RPE'], delta=candidate['delta'],
        formula=candidate['formula'], prefix=candidate['prefix'], K=candidate['K'],
        R_outer=candidate['R_outer'], candidate_id=candidate['id'],
        structure=deepcopy(row['structure']), request=request, price_scope=desc,
        package=package, role='INDEPENDENT_FROZEN_C1_TRANSFER_VALIDATION',
        calibration=False, C1_anchor_cell=anchors[candidate['system']],
        proposed_classical_paths=n_paths, proposed_cohorts=1,
        proposed_full_wrapper_compiles=2*n_paths, axes=['X','Y'],
        paired_XY=True, cohort=f'C2_PROPOSED_{package}_{candidate["system"]}',
        seeds=None, seed_policy='Freeze distinct cohort/stream seeds before acquisition; no draws now',
        new_structural_S_required=True, new_direct_compiled_required=True,
        saved_exact_C1_price_available=False, proposed_price_unit='FINITE_COHORT_ONLY',
        existing_model=ref(FROZEN, 'docs/research/r_price_phase1_results_20261111/run01/FROZEN_MODELS.json'),
        model_policy='C1 coefficients unchanged; bounded new-point evaluation requires separate permission and SOURCE',
        compiled_rz=None, structural_rz=None, prediction_rz=None, residual=None,
        coefficient_refit=False, qualified_for_candidate=False,
        all_law_expected_price=None, R_q_systematic=None, joint_covariance=None,
        total_CI=None, G=None, execution_authorized=False,
        workload=dict(events=desc['event_count'], signed_pieces=len(desc['allocation']),
            deterministic_block_visits=2*row['q_PF']*len(desc['allocation'])*(candidate['prefix']+1),
            raw_native_gate_count=None, wall_seconds=None,
            interpretation='recipe loop counts only; no RZ counting or benchmark'))


def prepare(root=ROOT):
    store=p.SavedArtifacts(root)
    # Same common API as B, without copying/replacing B's module/worktree.
    name='src/trottertracks/pf_pr_rpe_pipeline.py'
    if (Path(root)/name).read_bytes()!=c1.blob(root,B,name):
        raise ValueError('Fixed B common metadata API differs; reconcile before generating plan')
    b=c1.fixed_module(root,B,'src/trottertracks/rpe_r_selection.py','trottertracks._longq_fixed_B')
    original=store.read(ref(C1,'docs/research/r_price_phase1_results_20261111/run01/PLAN.json'))
    a=store.read(ref(A_PLAN,'docs/research/pr_r_detailed_n_preparation_20261111/EXECUTION_PLAN.json'))
    published_N=store.read(ref(A_RESULTS,'docs/research/pr_r_detailed_n_p1_results_20261111/run01/N_HANDOFF.json'))
    n_records={v['identity']['candidate']['id']:v for v in published_N['shots']}
    c1.validate_plan(original)
    law=original['menus']['H4']['sampling_law']
    choices={'H4':{0:[2,4,8],3:[4,8,128],4:[4,8],6:[4,128,256],12:[4,8,8192]},
             'H6':{4:[11,16,32,128],5:[16,32],8:[16,128,256],12:[16,32,2048]}}
    menus={}
    for system in choices:
        task=store.read(ref(TASKS,f'docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/{system}_PLAN.json'))['task']
        menus[system]=b.build_menu(task,choices[system],sampling_law=law)
    anchors={'H4':'H4_q8_R4','H6':'H6_q1_R16'}
    packages=[];rows=[]
    for package, specs, n_paths, purpose in PACKAGES:
        ids=[]
        for system,m,R in specs:
            menu=menus[system]
            row=next(v for v in menu['candidates'] if v['m']==m and v['candidate']['method']=='PR' and v['candidate']['R_outer']==R)
            item=make_row(menu,row,b.request_context(menu,row),package,n_paths,anchors)
            rows.append(item);ids.append(item['candidate_id'])
        packages.append(dict(id=package,purpose=purpose,candidate_ids=ids,
            new_full_wrapper_compile_budget=2*n_paths*len(specs),cohorts=len(specs),
            paths_per_cohort=n_paths,execution_authorized=False,automatic_follow_on=False,
            calibration_fit=False,acceptance_threshold=None,
            interpretation='Exploratory finite pilot, not a statistical power or accuracy guarantee',
            prerequisites=['Separate concrete execution permission and fixed runner SOURCE',
                'Freeze C1 coefficients, roles, seeds, metrics and output locations before acquisition'],
            causal_interpretation='R-only points are needed to separate R transfer from joint interaction; not a prerequisite for acquiring the A-P1 point'))
    saved={cell['candidate_id']:cell for cell in original['cells']}
    proposal={r['candidate_id']:r for r in rows};a_coverage=[]
    for ar in a['rows']:
        if ar['package']!='P1_SHORT_MIDDLE':continue
        menu=menus[ar['system']]
        row=next(v for v in menu['candidates'] if v['candidate']['id']==ar['candidate_id'])
        if not p.same(p.identity(menu,row),ar['N_request']['identity']):
            raise ValueError('A P1 full candidate identity mismatch')
        nr=n_records[ar['candidate_id']]
        if (not p.same(nr['identity'],ar['N_request']['identity'])
                or nr['plan']['status']!='CONDITIONAL_ROUND_PASS'
                or nr['N_comparison_policy']!=ar['allowance_policy']):
            raise ValueError('Published A conditional status/identity/policy mismatch')
        source=ref(A_RESULTS,'docs/research/pr_r_detailed_n_p1_results_20261111/run01/N_HANDOFF.json')
        cell=saved.get(ar['candidate_id']);future=proposal.get(ar['candidate_id'])
        a_coverage.append(dict(system=ar['system'],m=ar['m'],q_PF=ar['q_PF'],R_outer=ar['R_outer'],
            candidate_id=ar['candidate_id'],N_status='PUBLISHED_CONDITIONAL_ROUND_PASS',N_source=source,
            N_X=nr['plan']['N_X'],N_Y=nr['plan']['N_Y'],N_values_copied_verbatim=True,
            N_comparison_policy=ar['allowance_policy'],N_values_recomputed=False,
            C1_cell=None if cell is None else cell['id'],
            C1_role=None if cell is None else cell['role'],
            C1_direct_saved=cell is not None,C1_model_prediction_saved=cell is not None,
            C1_independent_validation=False,recipe_eligible=True,
            q_relation_to_C1=('BELOW_OBSERVED' if ar['q_PF']<8 else 'ABOVE_OBSERVED' if ar['q_PF']>16 else 'WITHIN_OBSERVED') if ar['system']=='H4'
                else ('ABOVE_OBSERVED' if ar['q_PF']>2 else 'WITHIN_OBSERVED'),
            R_relation_to_C1=('BELOW_OBSERVED' if ar['R_outer']<4 else 'ABOVE_OBSERVED' if ar['R_outer']>8 else 'WITHIN_OBSERVED') if ar['system']=='H4'
                else ('BELOW_OBSERVED' if ar['R_outer']<16 else 'ABOVE_OBSERVED' if ar['R_outer']>32 else 'WITHIN_OBSERVED'),
            S_status='SAVED_C1_PATH_STRUCTURES' if cell else 'NEW_PATH_STRUCTURAL_COUNT_NEEDED',
            current_C1_evaluator_accepts_point=cell is not None,
            new_price_prediction_evaluated=False,model_qualified_for_candidate=False,
            C_status='SAVED_C1_FINITE_COHORT_ONLY' if cell else 'NOT_ACQUIRED_IN_INSPECTED_CATALOGS',
            C_source=None if cell is None else ref(C1,'docs/research/r_price_phase1_results_20261111/run01/CALIBRATION_EVALUATION.json'),
            proposed_package=None if future is None else future['package'],
            unselected_reason=None if cell or future else 'Not all A P1 candidates selected; retain missing, no interpolation or ranking',
            C=None,G=None,all_law_expected_price=None))
    final=[]
    for system,m,R in [('H4',12,8192),('H6',12,2048)]:
        row=next(v for v in menus[system]['candidates'] if v['m']==m and v['candidate']['method']=='PR' and v['candidate']['R_outer']==R)
        final.append(dict(system=system,m=m,candidate=deepcopy(row['candidate']),q_RPE=row['q_RPE'],
            structure=deepcopy(row['structure']),proposed_direct_compiles=0,price=None,
            status='DEFERRED_METHOD_FEASIBILITY_FIRST_NOT_C2_VALIDATED',execution_authorized=False))
    result=dict(schema_version=1,status='TECHNICAL_LONG_Q_R_PLAN_ONLY_NO_SCIENCE',
        execution_authorized=False,scientific_execution_source=None,
        metadata_generation_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        B_source=B,A_plan_source=A_PLAN,A_public_results_source=A_RESULTS,C1_publication=C1,frozen_model_commit=FROZEN,
        scientific_condition='Saved DF10/DF19, linear 1A STO-3G, H4 S2/delta.1/p1/K2; H6 m5_best/delta1.6/p7/K2',
        compiler=deepcopy(p.PROFILE),packages=packages,rows=rows,A_P1_coverage=a_coverage,
        decision_order=[dict(id='P1_DIRECT',components=['Q','QR'],priority=1,new_full_wrapper_compile_budget=40,
            reason='Four A-P1 N-acquired cells: two same-R high-q baselines and two R128 alternatives; no R256 compile presumed'),
            dict(id='R',components=['R'],priority=2,new_full_wrapper_compile_budget=16,
                reason='Only if attribution of transfer error requires the missing low-q/R128 corners'),
            dict(id='SHORT',components=['SHORT'],priority=2,new_full_wrapper_compile_budget=16,
                reason='Optional short-time baseline/minimum-allocation domain; separate from middle-time priority'),
            dict(id='LONG_METHOD',components=[],priority=3,new_full_wrapper_compile_budget=0,
                reason='Streaming/local-boundary technical feasibility first; no final-round scientific run proposed')],
        deferred_final_rounds=final,existing_C1_cells=[{k:deepcopy(c[k]) for k in ('id','system','candidate_id','role','saved','n','request')} for c in original['cells']],
        plan_scope='Choose finite packages separately; not an instruction to execute all packages',
        rare_dedicated_experiments=0,old_validation_relabelled_as_calibration=False,
        scientific_work_this_task=0,new_price_records=[],N_recomputed=False,G=None,
        uncertainty=dict(coefficient_ref=ref(FROZEN,'docs/research/r_price_phase1_results_20261111/run01/FROZEN_MODELS.json'),
            all_law_expected_price=None,long_q_R_systematic=None,cross_R_round_covariance=None,total_CI=None),
        metrics=['paired C/S per path','bias','MAE','RMSE','path relative RMSE','max absolute relative error',
            'finite cohort mean and nominal variance','paired XY/residual covariance','frozen coefficient nominal covariance separately'],
        precision_policy='No universal percent threshold; future fixed-N sensitivity and simultaneous price intervals must justify adoption; no N/G here',
        output_contract='B supplied price record with exact identity/law/compiler/axis/cohort/SOURCE, finite-cohort classification and false model adoption',
        missing_price_policy='No record; never zero, rescaled fixed-R price, or automatic model evaluation',
        execution_order=['User considers P1_DIRECT (Q+QR, 40 wrapper compiles) first, or chooses an explicit smaller subset',
            'Prepare/fix bounded metadata-compatible runner; preserve old C1 scope and STOP',
            'Acquire only permitted normal finite cohorts; save S before C and paired XY',
            'Evaluate frozen C1 at only the separately permitted points; no refit or validation leakage',
            'Publish observed evidence and STOP; reassess whether R, QR, SHORT or a new model is necessary'],
        stop_conditions=['SOURCE/recipe/identity/law/compiler/allocation mismatch',
            'Missing S/C/axis or numerical finite-value gate failure; no fallback or clamping',
            'Full-wrapper/price unit mismatch; finite cohort cannot become all-law expectation',
            'Budget exceeded, unexpected backend failure or output collision: preserve evidence, no automatic scientific retry',
            'Finite package complete: publish/verify and STOP; no extra paths/q/R/fit/N/G'],
        full_RPE_qualified=False,scientific_R_adopted=False,mandatory_stop=True,saved_metadata_sources=store.sources)
    validate(result)
    return result


def validate(plan):
    """Technical shape/domain checks only; no science values to evaluate."""
    if plan['execution_authorized'] or plan['scientific_execution_source'] is not None or plan['new_price_records']:
        raise ValueError('Preparation only; no execution or acquired price')
    if plan['rare_dedicated_experiments'] or plan['G'] is not None:
        raise ValueError('Rare/G outside preparation scope')
    rows=plan['rows']
    if len({r['candidate_id'] for r in rows})!=len(rows):raise ValueError('Duplicate candidate')
    for r in rows:
        c=r['request']['identity']['candidate'];st=r['structure']
        if r['candidate_id']!=p.candidate_id(c) or st!=p.structure(c):raise ValueError('Candidate/allocation binding')
        if r['system']=='H6' and (len(st['allocation'])!=11 or c['R_outer']<11):raise ValueError('H6 minimum/pieces')
        if st['event_count']!=c['q']*c['R_outer'] or sum(st['allocation'])!=c['R_outer']:raise ValueError('Whole-outer event count')
        if any(r[k] is not None for k in ('compiled_rz','structural_rz','prediction_rz','residual','total_CI','G')):
            raise ValueError('New scientific value forbidden')
        if r['qualified_for_candidate'] or r['calibration'] or r['coefficient_refit']:raise ValueError('No model adoption/refit')
    for package in plan['packages']:
        group=[r for r in rows if r['package']==package['id']]
        if package['new_full_wrapper_compile_budget']!=sum(r['proposed_full_wrapper_compiles'] for r in group):raise ValueError('Compile budget')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    result=prepare()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
