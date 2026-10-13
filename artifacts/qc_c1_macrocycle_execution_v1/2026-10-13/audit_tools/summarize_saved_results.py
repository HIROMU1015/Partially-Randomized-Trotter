"""Aggregate saved C1 outputs only. No eigensolver, new candidate or optimization."""
import collections
import hashlib
import json
from pathlib import Path
import sys


def summarize(root):
    root=Path(root)
    out=root/'artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1'
    rows=json.loads((out/'comparisons.json').read_text())
    completion=json.loads((out/'completion.json').read_text())
    audit=json.loads((out/'independent_audit.json').read_text())
    assert completion['eigensolver_calls']==940
    assert audit['status']=='SAVED_RESIDUAL_AND_COST_AUDIT_PASS'
    assert len(rows)==32
    table=[]; modes={}
    for score in ('empirical','joint','box'):
        ratios=[]; good=bad=improvements=0
        for row in rows:
            c1=row['frontiers'][score+':C1']
            # Min of already saved frontiers, not a fresh precision or candidate search.
            baselines=[row['frontiers']['oracle_diagnostic:'+k] for k in ('all_single_subdivision','S4','C2')]
            baselines=[x for x in baselines if x is not None]
            if c1 is None or not baselines:
                bad+=1;continue
            good+=1; base=min(baselines,key=lambda x:x['conditional_quantum_cost'])
            ratio=c1['conditional_quantum_cost']/base['conditional_quantum_cost'];ratios.append(ratio)
            improvements+=ratio<.95
            table.append({'input_id':row['input_id'],'state':row['state'],'price':row['price'],
                          'epsilon':row['epsilon'],'selector':score,'selected':c1['method'],
                          'selected_h_index_diagnostic':c1['h_index'],'best_saved_baseline':base['method'],
                          'baseline_h_index_diagnostic':base['h_index'],'cost_ratio':ratio,
                          'interpretation':'posthoc fixed-grid conditional query-cost proxy;not operational QPE cost'})
        modes[score]={'paired_eligible_comparisons':good,'unresolved_or_ineligible':bad,
                      'registered_5_percent_improvements':improvements,
                      'ratio_min':min(ratios),'ratio_max':max(ratios)}
    cases=[]
    for name in ('S1_weak','S2_balanced','S3_signed','S4_strong'):
        case=json.loads((out/(name+'.json')).read_text());probes=case['probes']
        cases.append({'id':name,'true_kernel_d_diagnostic':case['true_kernel_d_diagnostic'],
            'all_three_true_leading_d_positive_observed':all(d>1e-10 for d in case['true_kernel_d_diagnostic']),
            'ground_state_00_overlap_min':min(p['approximate_overlap']['00'] for p in probes),
            'ground_state_11_overlap_max':max(p['approximate_overlap']['11'] for p in probes),
            'branch_valid_probes':sum(p['diagnostic_branch_valid'] for p in probes),
            'probes':len(probes),'wall_seconds':case['wall_seconds'],'max_RSS_KiB':case['max_RSS_KiB'],
            'primitive_cache_entries':case['primitive_exponential_cache_entries']})
    precision=[]
    for row in rows:
        choice=row['coefficient_only_precision_choices']['C1']
        if choice['h_index'] is None:continue
        point=next(p for p in row['rows'] if p['method']==choice['method'] and p['h_index']==choice['h_index'])
        precision.append({'input_id':row['input_id'],'state':row['state'],'price':row['price'],
                          'epsilon':row['epsilon'],**choice,'certified_PF_bound':point['certified_PF_bound'],
                          'saved_numerical_eligibility':point['eligible']})
    files={p.name:{'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
           for p in sorted(out.iterdir()) if p.is_file()}
    return {'status':'FIXED_PILOT_AUDITED_NO_SELECTED_C1_ADVANTAGE_OBSERVED',
        'reference':'fixed4 nonmolecular2-qubit Pauli toys;39unitaries;6h;model targets1e-5/1e-6;prices3/6',
        'geometry_basis_DF_rank_L_D':'not applicable','unique_Hamiltonians':4,'comparison_rows':32,
        'normal_state_comparisons':16,'bad_state_controls':16,'modes':modes,'cases':cases,
        'branch_valid_probes':sum(c['branch_valid_probes'] for c in cases),
        'C2_derived_records':72,'eigensolver_calls':940,'audit_new_eigensolver_calls':0,
        'saved_audit_comparison_rows':audit['comparison_rows_checked'],
        'classifications':dict(collections.Counter(k for r in rows for k in r['classifications'])),
        'coefficient_only_C1_precision_choices_valid':len(precision),
        'coefficient_only_C1_precision_choices_unresolved':32-len(precision),
        'precision_choices':precision,'table':table,'primary_files':files,
        'primary_total_bytes':sum(x['bytes'] for x in files.values()),
        'BCH_information_selection_wall_seconds':completion['BCH_information_selection_wall_seconds'],
        'full_ground_energy_cost':'UNKNOWN','formal_algorithm_adoption':False,
        'known_composition_equivalence':'retained;no independent generator advantage claim',
        'scientific_continuation_decision':'GPT review after separate explicit start approval',
        'mandatory_stop':True,'additional_science_or_retry':0}


if __name__=='__main__':
    root=Path(sys.argv[1]);result=summarize(root)
    dest=root/'artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/saved_results_summary.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','eigensolver_calls','branch_valid_probes',
        'normal_state_comparisons','bad_state_controls','coefficient_only_C1_precision_choices_valid','primary_total_bytes')},ensure_ascii=False))
