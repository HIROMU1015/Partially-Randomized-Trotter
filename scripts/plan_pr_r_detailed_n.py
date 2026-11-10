"""Metadata-only detailed-N proposal. No scientific execution option.

Copy A0 values verbatim, delegate canonical requests to the pinned B builder,
and count source-level operations. Never import the molecular action backend.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import pr_r_screening as screening

OUT = 'docs/research/pr_r_detailed_n_preparation_20261111'
A0 = '4745bb676568bb11d46552be42ffc6f4fba75385'
B = 'ac3e1ce87b5ed7d9114abc316545366dd097209e'
C = 'c596e88d2a88fb219c2e373fb0d72c6e39cb723a'
H4 = '89c91c692e1d35de94d2e7afbe890aabe6d80b7e'
H6 = 'f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a'
H6_D1 = '45607a5086721ea1d21b1ed2d7edfe870549f0c8'
POLICY = 'R_AWARE_EMPIRICAL_ACTION_COUNT_V1'
SOURCES = {
    'A0': dict(commit=A0, path='docs/research/pr_r_normalization_phase0_results_20261111/run01/result.json'),
    'H4': dict(commit=H4, path='docs/research/h4_df10_rpe_batch_results_20261110/run01/result.json'),
    'H4_N': dict(commit=H4, path='docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json'),
    'H6': dict(commit=H6, path='docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json'),
    'H6_D1': dict(commit=H6_D1, path='docs/research/h6_df19_rpe_d1_results_20261110/result.json'),
    'C_PLAN': dict(commit=C, path='docs/research/r_price_phase1_preparation_20261111/PLAN.json'),
}
# These finite queues are proposals, not scientific adoption or a global R cap.
SELECT = {'H4': {0: [2,4,8], 6: [4,128,256], 12: [4,8,8192]},
          'H6': {4: [11,16,32], 8: [16,128,256], 12: [16,32,2048]}}


def operations(row):
    """Integer structural counts only; no time prediction or error allowance."""
    q = row['q_PF']; st = row['structure']; J = len(st['signed_weights'])
    E = st['event_count']; prefix = row['prefix']
    d = {'H4': 36, 'H6': 400}[row['system']]
    if E != q*row['R_outer'] or sum(st['allocation']) != row['R_outer']:
        raise ValueError('Whole-outer allocation/event mismatch')
    return dict(sector_dimension=d, pieces=J, finite_two_column_updates=E,
        raw_vector_divisions=E, head_three_column_actions=q*2*(prefix+1)*J,
        exact_tail_vector_actions=q*J,
        intermediate_checks=q*2*(prefix+1)*J+E+1,
        finite_dense_dot_terms=2*E*d*d,
        piece_step_matrices=J, piece_exact_tail_exponentials=J,
        matrix_bytes_complex128=16*d*d,
        event_sized_arrays_required=False, wall_seconds_prediction=None,
        interpretation='Source loop counts, not measured FLOPs or a runtime estimate')


def has_pair(ref):
    # Presence/shape only. The future scientific provider must run numeric gates.
    return isinstance(ref, dict) and all(isinstance(ref.get(k), dict)
        and set(('real','imag')) <= ref[k].keys()
        for k in ('reference','independent_reference')) and 'signal_difference' in ref


def dict_literal(node, names):
    if isinstance(node,ast.Name):return names[node.id]
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
        return {k.arg:dict_literal(k.value,names) for k in node.keywords}
    return ast.literal_eval(node)


def prepare(root=ROOT):
    store = p.SavedArtifacts(root)
    data = {k: store.read(v) for k,v in SOURCES.items()}
    a0 = data['A0']
    if (a0['status'] != 'PHASE0_NORMALIZATION_RECORDED' or len(a0['records']) != 24
            or not a0['source_preserved'] or not a0['input_preserved']):
        raise ValueError('Completed immutable A0 evidence required')
    # Read B's LAW literal without importing its science/price integration.
    bpath = 'src/trottertracks/rpe_provider_integration.py'
    btext = screening.subprocess.check_output(['git','show',B+':'+bpath],cwd=root)
    tree=ast.parse(btext)
    names={t.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
           and isinstance(n.value,ast.Constant) for t in n.targets if isinstance(t,ast.Name)}
    law = next(dict_literal(n.value,names) for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LAW' for t in n.targets))
    return _build(root, store, data, law)


def _build(root, store, data, law):
    builder = screening.b_metadata(root); rows=[]; menus={}
    for system in ('H4','H6'):
        base = screening.A.saved_task(root, system)
        Rmap = {m:[r['R_outer'] for r in data['A0']['records'] if r['system']==system and r['m']==m]
                for m in SELECT[system]}
        menu = builder.build_menu(base,Rmap,sampling_law=law); menus[system]=menu
        for a in (r for r in data['A0']['records'] if r['system']==system):
            row = next(r for r in menu['candidates'] if r['candidate']['id']==a['candidate_id'])
            req = builder.request_context(menu,row)
            if not p.same(req['identity'],a['request']['identity']) or not p.same(row['structure'],a['structure']):
                raise ValueError('A0/B identity or allocation differs')
            baseline = a['R_outer'] == (4 if system=='H4' else 16)
            selected = a['R_outer'] in SELECT[system][a['m']]
            package = ('P1_SHORT_MIDDLE' if a['m']!=12 else
                       'P2_H6_FINAL' if system=='H6' else 'P3_H4_FINAL') if selected else None
            saved = data['H4'] if system=='H4' else data['H6']
            records = saved['records'] if system=='H4' else saved['new_records']
            old = next((n for n in records if n['identity']['candidate']['id']==a['candidate_id']),None)
            ref = saved.get('scope_observed',{}).get('references',{}).get(str(a['T']))
            special = system=='H6' and a['m']==4
            if not has_pair(ref) and not special:raise ValueError('Expected bound saved reference pair missing')
            signal = old.get('diagnostics',{}).get('signal',{}) if old else {}
            fields = ('maximum_intermediate_norm','raw_corrected_vector_closure','exact_tail_norm_error')
            reusable = baseline and old is not None and all(k in signal.get('allowance_evidence',{}) for k in fields)
            if baseline and not special and not reusable:raise ValueError('Expected saved baseline evidence missing')
            if old and not p.same(old['identity'],req['identity']):raise ValueError('Saved baseline identity differs')
            source = SOURCES['H4'] if system=='H4' else SOURCES['H6_D1' if special else 'H6']
            signal_path = ('rows[candidate_id='+a['candidate_id']+']' if special else
                ('records' if system=='H4' else 'new_records')+'[identity.candidate.id='+a['candidate_id']+'].diagnostics.signal')
            cell = next((v for v in data['C_PLAN']['cells'] if v['candidate_id']==a['candidate_id']),None)
            cstatus = ('SAVED_DIRECT_FINITE_COHORT_AND_PHASE1_CALIBRATION' if cell and cell['saved'] else
                'PHASE1_PLANNED_NEW_CALIBRATION_NOT_EXECUTED_AT_PIN' if cell else
                'FIXED_R_LONG_Q_UNVALIDATED' if baseline else 'OUTSIDE_INSPECTED_C_PHASE1_DOMAIN')
            rows.append(dict(system=system,m=a['m'],T=a['T'],q_RPE=a['q_RPE'],q_PF=a['q_PF'],
                delta=a['delta'],R_outer=a['R_outer'],prefix=a['prefix'],K=a['K'],formula=a['formula'],
                candidate_id=a['candidate_id'],structure=deepcopy(a['structure']),
                A0_source=deepcopy(SOURCES['A0']),A0_record=deepcopy(a),N_request=req,
                law_binding_note='A0 law retained in A0_record; N law from pinned B, no automatic alias',
                baseline=baseline,package=package,selected_for_proposal=selected,
                selection_reason=('BASELINE_SAME_POLICY' if baseline else
                    'SHORT_TIME_LAW_AND_CROSS_R_CHECK' if a['m'] in (0,4) else
                    'MODEST_R_CONTRAST' if a['R_outer'] in (8,32) else
                    'FIRST_TWO_A0_HIGH_ATTENUATION_BANDS' if a['m']!=12 else
                    'FIRST_A0_HIGH_ATTENUATION_FINAL_BAND'),
                deferred_reason=None if selected else 'Higher R band retained, detailed N and C unmeasured; no dominance claim',
                reference=dict(source=deepcopy(source),field='scope_observed.references.'+str(a['T']) if not special else None,
                    saved_pair_shape_present=has_pair(ref),numeric_validation_performed=False,
                    proposed_new_actions=2 if special else 0,no_implicit_fallback=True),
                signal=dict(saved_source=deepcopy(source) if baseline else None,
                    saved_field=signal_path if baseline else None,baseline_signal_evidence_reusable=reusable,
                    proposed_new_finite_mean=not reusable,exact_tail_control_in_new_mean=not reusable,
                    reason='H6 m4 saved D1 lacks required pair/norm evidence; explicitly budget new propagation' if special and baseline else None),
                allowance_policy=POLICY,scalar_evaluations_proposed=1,axes=['X','Y'],
                baseline_N_source=deepcopy(SOURCES['H4_N' if system=='H4' else 'H6_D1' if special else 'H6']) if baseline else None,
                complexity=operations(a),C_coverage=dict(status=cstatus,phase1_cell=deepcopy(cell),
                    source=deepcopy(SOURCES['C_PLAN']),qualified_for_candidate=False,price=None),
                N=None,C=None,G=None,scientific_PASS_FAIL=None,execution_authorized=False))
    packages=[]
    for name in ('P1_SHORT_MIDDLE','P2_H6_FINAL','P3_H4_FINAL'):
        selected=[r for r in rows if r['package']==name]
        new=sum(r['signal']['proposed_new_finite_mean'] for r in selected)
        refs={(r['system'],r['T']):r['reference']['proposed_new_actions'] for r in selected}
        packages.append(dict(id=name,candidate_ids=[r['candidate_id'] for r in selected],
            execution_order=[r['candidate_id'] for r in sorted(selected,key=lambda r:not r['baseline'])],
            proposed_calls=dict(new_finite_means=new,internal_exact_tail_controls=new,
                saved_baseline_scalar_only=len(selected)-new,scalar_phase_radius_N=len(selected),
                axis_N=2*len(selected),new_reference_actions=sum(refs.values()),
                sampling=0,compile=0,C=0,G=0),
            execution_authorized=False,automatic_follow_on=False,
            readiness='EXISTING_ACTION_KERNEL_PLUS_SMALL_HANDOFF_GLUE_REQUIRED' if name!='P3_H4_FINAL' else
                'DEFER_LARGE_R_UNTIL_BACKEND_FEASIBILITY_AND_PRECISION_PLAN_FIXED'))
    # Retain corrected integers as integers, never recompute or float-convert N.
    baseline_handoff=[deepcopy(n) for n in data['H4_N']['shots'] if
        n['identity']['candidate']['method']=='PR' and n['identity']['m'] in SELECT['H4']]
    return dict(status='TECHNICAL_PLAN_ONLY_NO_SCIENTIFIC_EXECUTION',schema_version=1,
        execution_authorized=False,scientific_execution_source=None,packages=packages,rows=rows,
        menus=menus,B_metadata_source=builder.source,B_integration_source=dict(commit=B,path='src/trottertracks/rpe_provider_integration.py'),
        numerical_components=[dict(commit=A0,path=path,sha256=hashlib.sha256(
            screening.subprocess.check_output(['git','show',A0+':'+path],cwd=root)).hexdigest()) for path in (
                'src/trottertracks/pr_r_dependent_n.py','src/trottertracks/h6_df19_rpe_d2.py',
                'src/trottertracks/h4_df10_rpe_d1.py','src/trotterlib/rte.py',
                'src/trotterlib/h46_m5_plan.py','scripts/run_h6_df19_rpe_d2.py',
                'scripts/verify_h4_df10_rpe_shot_integers.py')],
        C_phase1_cells=deepcopy(data['C_PLAN']['cells']),saved_sources=deepcopy(store.sources),
        H4_corrected_N_inventory=baseline_handoff,old_N_comparable_to_new_policy=False,
        stop_conditions=['SOURCE/H-state-sector-candidate-law-allocation mismatch',
            'Saved pair absent except explicitly proposed H6 m4 two new actions',
            'Missing baseline norm/closure evidence except explicitly proposed H6 m4 new mean',
            'Reference/normalization/closure/norm/phase/radius/integer gate failure',
            'Missing matched allowance policy; unknown actual errors remain unresolved',
            'No automatic retry/R swap/threshold change/reference fallback/C/G/next package'],
        science_calls_this_task=0,R_upper_bound=None,range_complete=False,
        scientific_R_adopted=False,full_RPE_qualified=False,mandatory_stop=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); path=Path(args.output)
    if path.exists():raise FileExistsError('Fresh plan path required')
    result=prepare();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
