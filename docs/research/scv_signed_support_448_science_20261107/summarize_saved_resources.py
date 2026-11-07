"""Read-only classical-resource accounting, no statistical/scientific replay."""
from collections import Counter,defaultdict
import json
from pathlib import Path

ROOT=Path('/tmp/scv-signed-support-448-preparation-20261107')
OUT=ROOT/'artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01'
DOC=Path('docs/research/scv_signed_support_448_science_20261107')


def read_lines(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def main():
    final=json.loads((OUT/'RUN_FINALIZATION.json').read_text());starts={};seconds=defaultdict(float);counts=Counter()
    for row in read_lines(OUT/'ADAPTER_CALLS.jsonl'):
        if row['event']=='CALL_RETURN':seconds[row['name']]+=row['seconds'];counts[row['name']]+=1
    cpu_user=cpu_system=worker_wall=0.;completed=0;pss=[];rss=[];missing=[]
    for path in OUT.glob('RESOURCE_worker_*.jsonl'):
        for row in read_lines(path):
            ticket=row['ticket']
            if row['event']=='TRIAL_RESOURCE_START':
                assert ticket not in starts;starts[ticket]=row
            if row['event']=='TRIAL_RESOURCE_RETURN':
                before=starts.pop(ticket);completed+=1
                cpu_user+=row['cpu_user']-before['cpu_user'];cpu_system+=row['cpu_system']-before['cpu_system']
                worker_wall+=row['elapsed_wall_inclusive']
            for key,values in (('Pss_bytes',pss),('Rss_bytes',rss)):
                value=row.get('process_memory',{}).get(key)
                if value is not None:values.append(value)
    missing=list(starts)
    cache=Counter();peaks=[]
    for path in OUT.glob('worker_*.jsonl'):
        for row in read_lines(path):
            if row['event']=='PRICE_SAVED':cache['hits' if row['cache_hit'] else 'misses']+=1
            if 'worker_peak_RSS_bytes' in row:peaks.append(row['worker_peak_RSS_bytes'])
    record=dict(schema='signed448_classical_resource_summary_v1',parent_finalization=final,
        worker_completed_resource_intervals=completed,incomplete_resource_tickets=missing,
        measured_worker_user_CPU_seconds=cpu_user,measured_worker_system_CPU_seconds=cpu_system,
        worker_wall_interval_sum_seconds=worker_wall,worker_wall_is_not_parent_wall_or_CPU=True,
        call_RETURN_counts=dict(counts),call_RETURN_inclusive_wall_sums_seconds=dict(seconds),
        inclusive_timers_are_nested_not_additive=True,cache=cache,cache_hits_charged_full_quantum_RZ=True,
        max_worker_RSS_highwater_bytes=max(peaks) if peaks else 'UNKNOWN',
        max_observed_worker_PSS_bytes=max(pss) if pss else 'UNKNOWN',max_observed_worker_RSS_bytes=max(rss) if rss else 'UNKNOWN',
        shared_pages_not_summed=True,unmeasured_pure_cache_lookup_seconds='UNKNOWN',
        unmeasured_time_not_zero=True,science_calls=0)
    (DOC/'CLASSICAL_RESOURCE_SUMMARY.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('parent_finalization','call_RETURN_counts','call_RETURN_inclusive_wall_sums_seconds')}))


if __name__=='__main__':main()
