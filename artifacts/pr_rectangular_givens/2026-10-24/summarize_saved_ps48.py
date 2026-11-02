"""Summarize the one completed PS48 run; stdlib, saved records only.

No circuits/reference/laws are constructed. Means are equal-case descriptive
means over this fixed subset, never full-F2 probability-weighted expectations.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json

METRICS=('rz_count','cx_count','total_depth','rz_depth','cx_depth',
         'quantum_only_depth','circuit_size','measurement_count')


def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    result=dict(wrappers=len(rows),max_errors={key:max(r[key] for r in rows) for key in (
        'raw_operator_error','raw_signal_error','compiled_operator_error',
        'compiled_signal_error','raw_unitarity_error','compiled_unitarity_error')},metrics={})
    for metric in METRICS:
        baseline=sum(r['baseline_'+metric] for r in rows)
        candidate=sum(r['candidate_'+metric] for r in rows)
        deltas=[r['delta_'+metric] for r in rows]
        result['metrics'][metric]=dict(baseline_sum=baseline,candidate_sum=candidate,
            baseline_mean_exact=str(Fraction(baseline,len(rows))),
            candidate_mean_exact=str(Fraction(candidate,len(rows))),
            delta_mean_exact=str(Fraction(candidate-baseline,len(rows))),
            relative_change_percent=100*(candidate-baseline)/baseline if baseline else None,
            improved=sum(x<0 for x in deltas),equal=sum(x==0 for x in deltas),
            worsened=sum(x>0 for x in deltas),delta_min=min(deltas),delta_max=max(deltas))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True);parser.add_argument('--raw',required=True)
    parser.add_argument('--time-report',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();root=Path(args.source_root);raw=Path(args.raw);dest=Path(args.destination)
    plan_path=root/'artifacts/pr_rectangular_givens_preparation/2026-10-24/PLAN.json'
    plan=load(plan_path);summary=load(raw/'summary.json');stop=load(raw/'STOP.json')
    assert summary['completed_wrappers']==48 and stop['status']==summary['status']
    ids={c['id'] for c in plan['cases']};assert len(ids)==48
    comparisons={r['id']:r for r in load(raw/'CASE_COMPARISONS.json')};assert set(comparisons)==ids
    csv_path=root/'artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/CASE_RESOURCES.csv.gz'
    with gzip.open(csv_path,'rt') as f:
        baseline={r['id']:r for r in csv.DictReader(f) if r['id'] in ids}
    assert set(baseline)==ids
    bundle=load(raw/'input_snapshot/handoff/f1_bundle.json');atoms=bundle['arms']['PS']
    rows=[];compiled_records=[]
    for case in plan['cases']:
        cid=case['id'];built=load(raw/'build'/f'{cid}.json')
        compiled=load(raw/'compiled'/f'{cid}.json');comparison=comparisons[cid]
        compiled_records.append(compiled)
        k=case['frame_atom'];atom=atoms[k];support=[i for i,ch in enumerate(atom['word']) if ch!='I']
        row=dict(id=cid,frame_atom=k,word=''.join(atom['word']),support_size=len(support),
            partner=case['partner'],order='frame_first' if case['i']==k else 'partner_first',axis=case['axis'],
            raw_operator_error=built['raw_operator']['max_entry'],raw_signal_error=built['raw_signal_error'],
            compiled_operator_error=compiled['operator']['max_entry'],compiled_signal_error=compiled['signal_error'],
            raw_unitarity_error=built['raw_operator']['unitarity']['max_entry'],
            compiled_unitarity_error=compiled['operator']['unitarity']['max_entry'],scientific_status='PASS',
            compile_key=compiled['compile_key'],cache_alias=compiled['cache_alias'])
        assert built['raw_operator']['rtol']==compiled['operator']['rtol']==0
        assert row['raw_operator_error']<=1e-10 and row['raw_unitarity_error']<=1e-10
        assert row['raw_signal_error']<=2e-10 and row['compiled_operator_error']<=5e-10
        assert row['compiled_signal_error']<=5e-10 and row['compiled_unitarity_error']<=1e-10
        for metric in METRICS:
            old=comparison['baseline'][metric];new=comparison['candidate'][metric]
            assert old==int(baseline[cid][metric]) and new==compiled['metrics'][metric]
            assert comparison['delta'][metric]==new-old
            row['baseline_'+metric]=old;row['candidate_'+metric]=new;row['delta_'+metric]=new-old
            row['raw_candidate_'+metric]=built['raw_metrics'][metric]
        rows.append(row)
    assert Counter(r['support_size'] for r in rows)=={1:24,2:24}
    assert Counter(r['frame_atom'] for r in rows)=={k:8 for k in range(20,26)}
    grouped=dict(all48=summarize(rows),by_support={str(n):summarize([r for r in rows if r['support_size']==n]) for n in (1,2)},
        by_atom={str(k):summarize([r for r in rows if r['frame_atom']==k]) for k in range(20,26)},
        by_partner={str(k):summarize([r for r in rows if r['partner']==k]) for k in (0,2)})
    angles=load(raw/'angle_binding.json');assert set(angles['cases'])==ids
    assert set(angles['cases'].values())=={angles['angle_hex']}
    attempts=[]
    for p in (raw/'attempts').glob('*.jsonl'):
        record=[json.loads(line) for line in p.read_text().splitlines()]
        assert [x['status'] for x in record]==['ATTEMPTED','COMPLETED']
        attempts.append(dict(stage=record[0]['stage'],**record[1]))
    assert Counter(a['stage'] for a in attempts)=={'build':24,'compile':36}
    stages={s:dict(jobs=sum(a['stage']==s for a in attempts),
        sum_job_wall_seconds=sum(a['seconds'] for a in attempts if a['stage']==s),
        sum_job_CPU_seconds=sum(a['cpu_seconds'] for a in attempts if a['stage']==s),
        largest_recorded_process_RSS_KiB=max(a['peak_RSS_KiB'] for a in attempts if a['stage']==s)) for s in ('build','compile')}
    times=Path(args.time_report).read_text().splitlines()
    time_values={line.strip().split(': ',1)[0]:line.strip().split(': ',1)[1] for line in times if ': ' in line}
    whole=dict(user_CPU_seconds=float(time_values['User time (seconds)']),system_CPU_seconds=float(time_values['System time (seconds)']),
        elapsed=time_values['Elapsed (wall clock) time (h:mm:ss or m:ss)'],
        maximum_process_RSS_KiB=int(time_values['Maximum resident set size (kbytes)']),exit_status=int(time_values['Exit status']))
    assert whole['exit_status']==0
    resource=dict(pipeline_wall_seconds=summary['wall_seconds'],parent_CPU_seconds=summary['parent_cpu_seconds'],
        children_CPU_seconds=summary['children_CPU_seconds'],whole_job=whole,stages=stages,
        sum_unique_transpile_seconds=sum(r['compile_seconds'] for r in compiled_records),
        worker_configuration=load(raw/'resources.json'),aggregate_peak_RAM='NOT_MEASURED',
        note='Job wall times overlap in parallel. RSS values are per-process maxima, not aggregate peak memory. No old/new synthesis CPU speedup comparison.')
    result=dict(status=summary['status'],source_commit='770e53addda63a03fa8c6c5d5168f7062067c572',plan_sha256=sha(plan_path),
        scope='Fixed PS48; equal-case descriptive means only; no full-F2 expectation or certified N.',
        context=plan['context'],compiler=plan['compiler'],thresholds=plan['thresholds'],
        counts=dict(candidate_bases=6,ordered_evolutions=24,raw_wrappers=48,compiled_wrappers=48,
            actual_transpiles=36,compile_cache_aliases=12,baseline_builds=0,baseline_compiles=0,
            new_reference_precision_evaluations=0,research_grants_consumed=1,retries=0),
        basis_reports=summary['basis_reports'],angle_hex=angles['angle_hex'],
        groups=grouped,classical_cost=resource,
        unresolved=['No full-F2 probability-weighted comparison','No gamma/interval qualification or certified N','No general advantage or independent novelty certification'])
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'CASE_RESULTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (dest/'RESULT_SUMMARY.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],all48=grouped['all48'],support=grouped['by_support'],classical_cost=resource),indent=2))


if __name__=='__main__':main()
