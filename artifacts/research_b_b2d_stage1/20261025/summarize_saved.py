"""Summarize saved Stage-1 JSON only; no arrays, backends or new scientific calls."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    def load(path): return json.loads((root/path).read_bytes())
    result=load('technical-r1/result.json')
    status=load('technical-r1/status.json')
    binding=load('technical-r1/binding.json')
    inputs=load('technical-r1/inputs.json')
    calls=status['calls']
    assert status['status']=='COMPLETED_DESCRIPTIVE_MANDATORY_STOP'
    assert load('technical-r1/STOP.json')['status']==status['status']
    assert calls['completed']==binding['call_scope']['boundaries'] and calls['failed']=={}
    assert sum(v for k,v in calls['completed'].items() if k.startswith('dag.'))==941
    assert len(result['rows'])==len(load('technical-r1/partial_results.json')['rows'])==36
    events=[json.loads(line) for line in (root/'technical-r1/events.jsonl').read_text().splitlines()]
    gates=[i for i,e in enumerate(events) if e.get('event')=='complete' and e.get('boundary') in
        ('dag.validate_input','dag.a4_reference_check','published.convention')]
    first_comparison=next(i for i,e in enumerate(events) if e.get('event')=='start' and
        e.get('boundary') in ('dag.operator_error','dag.signals'))
    assert len(gates)==9 and max(gates)<first_comparison
    rows=[]
    for row in result['rows']:
        values=row['outputs'];signals=values['signals']
        record={k:row[k] for k in ('arm','family','delta','q','n_D','T')}
        for name,value in values.items():
            if name not in ('signals','normalization'): record[name]=value
        record.update(values['normalization'])
        record.update({k:v for k,v in signals.items() if not isinstance(v,dict)})
        for target in ('physical','ideal_tail','corrected','raw'):
            assert signals[target]['principal_phase'] is None
            record.update({target+'_'+key:signals[target][key] for key in ('X','Y','radius')})
        rows.append(record)
    groups=[]
    for arm,n in sorted({(r['arm'],r['n_D']) for r in rows},key=lambda p:(p[0],p[1] or 0)):
        group=[r for r in rows if (r['arm'],r['n_D'])==(arm,n)]
        fields=('D_half_operator_error','D_pair_operator_error','outer_operator_error',
            'corrected_physical_error','corrected_physical_signal_error','finite_RTE_error',
            'normalization_consistency')
        ranges={name:None if group[0][name] is None else [min(r[name] for r in group),
            max(r[name] for r in group)] for name in fields}
        groups.append(dict(arm=arm,n_D=n,rows=len(group),ranges=ranges))
    paired=[]
    for baseline in ('A0','A1','A2','A3'):
        for n in (1,2):
            candidate=[r for r in rows if r['arm']=='A4' and r['n_D']==n]
            gains=[];differences=[]
            for row in candidate:
                other=next(r for r in rows if r['arm']==baseline and r['delta']==row['delta'] and
                    r['q']==row['q'] and r['n_D']==(None if baseline=='A0' else n))
                a,b=row['corrected_physical_signal_error'],other['corrected_physical_signal_error']
                gains.append(100*(b-a)/b);differences.append(b-a)
            paired.append(dict(baseline=baseline,n_D=n,signal_error_reduction_percent=[min(gains),max(gains)],
                signal_error_difference=[min(differences),max(differences)]))
    checks=result['correctness'];a4=[v for k,v in checks.items() if k!='input_validation']
    summary=dict(final_status='COMPLETE',scientific_comparison='DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO',
        A4_vs_A3_discrimination='INCONCLUSIVE_AT_COMPLEX128_ROUNDOFF',
        scientific_source_commit=result['source_commit'],
        initial_source_commit='54008c697e686a42ec48941d25219d523bafd454',
        post_save_cli_repair_commit='5bef9c9cbcd7cfd595ec82874531ad3c7982d014',
        conditions=dict(model='synthetic_2mode_onebody_3DF',geometry_basis_molecular_rank='NOT_APPLICABLE',
            N=1,state='mode0_occupation',L_D=2,split_policy='weight_ranked',selected_factors=[0,1],
            residual_factors=[2],delta=['9/100','-9/100'],q=[2,3],n_D=[1,2],r_RTE=2,K=2,
            precision='complex128',correctness_atol=1e-12,phase_diagnostics=False),
        counts=dict(scientific_attempts=2,comparisons=36,dag_dispatch=941,exponential_keys=448,
            input_boundaries=10,published_convention=4,events=len(events),additional_scientific_runs=0),
        correctness=dict(input=checks['input_validation'],A4_cases=4,
            max_A4_math_error=max(c['mathematical_reference_error'] for c in a4),
            max_A4_inverse_error=max(c['actual_inverse_error'] for c in a4),
            max_published_operator_error=max(c['published_convention']['operator_error'] for c in a4),
            all_gates_before_comparison=True),
        environment=binding['environment'],input_provenance='technical-r1/inputs.json',
        group_ranges=groups,paired_signal_error_comparisons=paired,
        normalization=dict(product_range=[min(r['normalization_product'] for r in rows),max(r['normalization_product'] for r in rows)],
            max_rescaling_consistency=max(r['normalization_consistency'] for r in rows)),
        attempts=[dict(id='simple-once',status='TECHNICAL_STOP',source='54008c697e686a42ec48941d25219d523bafd454',
            input_boundaries_completed=9,dag_dispatch=0,comparisons=0,exit_code=2,
            reason='stale contract attributes in input loader',wall_seconds=5.64),
            dict(id='technical-r1',status='COMPLETE',source=result['source_commit'],exit_code=1,
                post_save_cli_error='Fraction JSON stdout serialization after complete persisted results',
                wall_seconds=5.65,no_scientific_rerun_after_completion=True)],
        limitations=['A4/A3 errors near machine precision cannot rank the two published methods.',
            'Small A4/A2 differences are descriptive observations, not certified precision gains.',
            'One fixed synthetic model and delta window; no convergence order or resource superiority conclusion.',
            'Raw signals are attenuated; corrected and raw mean operators retain their distinct normalization.'],
        stop='MANDATORY_STOP_NO_STAGE2_NO_EXTRA_GRID')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'STAGE1_SUMMARY.json').open('x') as file:
        json.dump(summary,file,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);file.write('\n')
    with (out/'COMPARISON.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=summary['final_status'],rows=len(rows),output=str(out))))


if __name__=='__main__': main()
