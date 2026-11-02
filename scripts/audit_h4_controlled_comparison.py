#!/usr/bin/env python3
"""Static SOURCE/SHA audit and saved-value quotations; no science dispatcher.

Only stdlib is imported. No QPY parsing, circuit construction, compilation,
sampling, shot/cost statistics or grant creation. Selection uses saved event
structure, never compiled prices. --verify rechecks preserved bytes and drafts.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24'
D1=ROOT/'artifacts/non_df_h4_d1_source_execution/2026-10-24'
D2=ROOT/'artifacts/non_df_h4_d2_execution/2026-10-24'
CNG=ROOT/'artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    with (BASE/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


def cohort():
    # No price is used in this rule. This is structural coverage selection from
    # the saved cohort, not a prospective random sample or population estimate.
    original=ROOT/'artifacts/non_df_h4_pilot/2026-10-23/run01/run/pauli_jw_PR'
    samples=read(original/'result.json')['circuit_cost']['samples']
    records=[]
    for sample in samples:
        p=original/'trajectories'/sample['trajectory_id']/'price_join.json'
        j=read(p)
        assert sample['trajectory_id']==j['trajectory_id'] and j['q']==8 and j['r']==4
        records.append(dict(trajectory_index=j['trajectory_index'],trajectory_id=j['trajectory_id'],
            price_join=str(p.relative_to(ROOT)),price_join_sha256=sha(p),
            event_record_hash=j['event_record_hash'],trajectory_record_hash=j['trajectory_record_hash'],
            has_order2=any(e['taylor_order']==2 for e in j['events']),
            has_order4=any(e['taylor_order']==4 for e in j['events'])))
    records.sort(key=lambda r:r['trajectory_index'])
    assert [r['trajectory_index'] for r in records]==list(range(256))
    assert len({r['trajectory_id'] for r in records})==256
    selected=next((r for r in records if r['has_order2']),None)
    assert selected is not None and selected['trajectory_index']!=0
    return dict(scope='METADATA_ONLY_NO_NEW_STATISTICS',records=records,
        selection_rule='minimum saved trajectory_index with any order2 event; independent of all prices',
        selection_kind='POST_COHORT_STRUCTURAL_COVERAGE_NOT_RANDOM_OR_REPRESENTATIVE',selected=selected,
        index0_has_order2=records[0]['has_order2'],index0_has_order4=records[0]['has_order4'])


def record():
    data=cohort();save('cohort_manifest.json',data)
    costs=read(CNG/'cost_summary.json');d2=read(D2/'result_summary.json');d1=read(D1/'result_summary.json')
    rows=[]
    for case,old in costs.items():
        rows.append(dict(case=case,control='generic_whole_body' if case.startswith('pauli') else 'DF_native',
            evidence='SAVED_COHORT_STATISTIC_QUOTE',n=old['n'],axes=old['axes'],
            population_mean_upper=old['population_mean_upper'],source=str((CNG/'cost_summary.json').relative_to(ROOT))))
    for case,metrics in d2['metrics'].items():
        rows.append(dict(case=case,control='central_rotation',evidence='SELECTED_INDEX0_MEASURED_QUOTE',
            n=1,cosine_RZ=metrics['RZ'],sine_RZ=None,new_cohort_mean=None,
            metrics=metrics,source=str((D2/'run01/variants'/case/'metrics.json').relative_to(ROOT))))
    save('comparison_quotes.json',dict(rows=rows,D2_paired_differences=d2['comparisons'],
        D1_interactions=d1['interactions'],new_scientific_arithmetic=False,
        cross_representation_trajectories_are_paired=False,
        generic_central_within_Pauli_same_ID_is_paired=True))
    # AST/source line anchors are technical evidence, not new scientific values.
    requests={
        'src/trotterlib/pauli_signal_adapter.py':['append_pauli_evolution','append_pauli_event','__post_init__'],
        'src/trotterlib/rte_event_audit.py':['trajectory'],
        'src/trotterlib/rte.py':['_make_event'],
        'src/trotterlib/h4_d1_adapter.py':['event_from_record','build','compile_saved'],
        'src/trotterlib/h4_d2_pauli_control.py':['_tokens','_rotation','central_control_body','full_wrapper'],
        'src/trotterlib/h4_d2_control.py':['validate_body','task_run'],
        'src/trotterlib/fixed_signal_evaluation.py':['hadamard_wrapper','default_compiler'],
        'src/trotterlib/h4_conditional_cng.py':['base_axis','shot','paired','cost_summary']}
    anchors=[]
    for path,names in requests.items():
        for n in ast.walk(ast.parse((ROOT/path).read_text())):
            if isinstance(n,ast.FunctionDef) and n.name in names:
                anchors.append(dict(path=path,function=n.name,line=n.lineno,end_line=n.end_lineno,sha256=sha(ROOT/path)))
    runtime=Path('/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/lib/python3.11/site-packages/qiskit')
    runtime_files=['circuit/library/pauli_evolution.py','synthesis/evolution/lie_trotter.py','synthesis/evolution/product_formula.py','synthesis/evolution/suzuki_trotter.py']
    runtime_sha={str(runtime/p):sha(runtime/p) for p in runtime_files}
    reference_runtime=read(ROOT/'artifacts/non_df_h4_d1_execution/2026-10-24/runtime.json')['files']
    assert all(reference_runtime[p]==h for p,h in runtime_sha.items())
    model=read(ROOT/'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json')
    components=model['cases']['PR']['components']
    save('source_mapping.json',dict(anchors=anchors,runtime_source_SHA=runtime_sha,runtime_files_match_fixed_D1_runtime=True,
        nonidentity_component_domain=all(set(c['label'])!={'I'} for c in components),
        full_H4_unitary_rechecked=False,compiled_error_certificate=False,
        saved_D2_validation={case:read(D2/'run01/variants'/case/'validation.json') for case in ('pauli_jw_PF','pauli_jw_PR')},
        proof_layers=dict(mathematical_rule='exact projector-controlled conjugation/product/scalar identities under unitary premise',
            fixed_SOURCE='Saved operands and aggregate phase connected by executed D2 SOURCE gate, retained unchanged',
            artificial='20 small-system checks recorded separately, finite floating tolerance',
            H4_program='no new full-H4 Operator/state/signal verification',
            compiler='rounding and full-program numerical transfer remain unbounded')))


def verify():
    assert not (BASE/'run01').exists() and not (BASE/'grant.json').exists()
    protected=read(BASE/'preservation_before.json')
    bad=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    drafts=read(BASE/'contracts_DRAFT.json')
    assert all(drafts[k] is False for k in ('execution_authorized','dispatch_enabled','grant_valid'))
    assert all(drafts[k] is None for k in ('grant','scientific_SOURCE','runtime_seal','launch_command'))
    assert all(c['execution_authorized'] is False for c in drafts['candidates'])
    tests=[read(BASE/'artificial'/p) for p in ('additional.json','reused_d2.json')]
    assert [(t['tests'],t['failures'],t['errors']) for t in tests]==[(8,0,0),(12,0,0)]
    import sys
    assert not any(k=='qiskit' or k.startswith('qiskit.') for k in sys.modules)
    assert not bad,bad
    return dict(status='STATIC_AUDIT_PASS',preserved_files=len(protected),mismatches=bad,
        artificial_tests=20,no_quantum_import=True,new_real_H4_calls=0,new_scientific_grants=0,
        N_G_reanalysis=0,D3_started=False,mandatory_STOP=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--record',action='store_true');g.add_argument('--verify',action='store_true')
    a=parser.parse_args()
    if a.record:record()
    else:print(json.dumps(verify(),indent=2))
