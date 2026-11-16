"""Read completed-job operational logs. Never write the observed run or rank Q3 outcomes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time


def timestamp(s):
    return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()


def summary(values):
    if not values:return None
    v=sorted(values)
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),
                p90=v[min(len(v)-1,int(.9*len(v)))],minimum=v[0],maximum=v[-1])


def read(path):
    return json.loads(path.read_bytes())


def lines(data):
    out=[]
    for line in data.splitlines():
        try:out.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):pass  # an incomplete live tail is not evidence
    return out


def analyze(run):
    started=time.time();completed=list(run.glob('jobs/*/*/PROCESS_RETURN.json'))
    # Read only terminal job files. No RESULT / API return / checkpoint is read.
    jobs=[];worker_specs=[]
    for i,end in enumerate(completed):
        job=end.parent;launch=read(job/'LAUNCH.json');ret=read(end)
        stages=lines((job/'JOB_STAGES.jsonl').read_bytes())
        nested=job/job.name;worker=lines((nested/'STAGES.jsonl').read_bytes()) if (nested/'STAGES.jsonl').exists() else []
        phase={x['stage']:timestamp(x['UTC'])for x in stages};wp={x['stage']:timestamp(x['UTC'])for x in worker}
        resource=read(nested/'RESOURCE_FINAL.json')if(nested/'RESOURCE_FINAL.json').exists()else{}
        supervision=read(job/'SUPERVISION.json')if(job/'SUPERVISION.json').exists()else{}
        if len(worker_specs)<5 and (job/'CASE_INPUTS.json').exists():worker_specs.append(read(job/'CASE_INPUTS.json').get('acceleration'))
        jobs.append(dict(job=job.name,method=job.parent.name,start=launch['started'],wall=ret['wall_seconds'],
            end=launch['started']+ret['wall_seconds'],returncode=ret['returncode'],signal=ret['signal'],
            launch_to_call=None if 'SCIENTIFIC_CALL_START'not in wp else wp['SCIENTIFIC_CALL_START']-launch['started'],
            metadata_seconds=None if 'METADATA_PASS'not in phase or 'CASE_BUILT'not in phase else phase['METADATA_PASS']-phase['CASE_BUILT'],
            DP_seconds=None if 'TOTAL_DP_RETURNED'not in wp or 'TOTAL_DP_START'not in wp else wp['TOTAL_DP_RETURNED']-wp['TOTAL_DP_START'],
            tree_CPU_seconds=resource.get('process_tree_CPU_seconds'),
            sampled_tree_peak_RSS_KiB=max(supervision.get('sampled_process_tree_peak_RSS_KiB',{}).values(),default=None)))
        if (i+1)%1000==0:print('read terminal jobs',i+1,flush=True)
    progress_bytes=(run/'PROGRESS.jsonl').read_bytes();progress=lines(progress_bytes)
    cutoff=max(j['end']for j in jobs);begin=min(j['start']for j in jobs)
    # Complete progress lines include launches whose exit is right-censored.
    # Times here have the existing logger's one-second resolution.
    events=[e for e in progress if timestamp(e['UTC'])<=cutoff]
    active=set();hist=Counter();previous=timestamp(events[0]['UTC']);peak=0
    for e in events:
        now=timestamp(e['UTC']);hist[len(active)]+=max(0,now-previous);previous=now
        if e['event']=='LAUNCH':active.add(e['job'])
        elif e['event']=='EXIT':active.discard(e['job'])
        peak=max(peak,len(active))
    hist[len(active)]+=max(0,cutoff-previous)
    total=sum(hist.values());slots=read(run/'ROUND.json')['executor']['max_concurrent_jobs']
    occupied=sum(n*t for n,t in hist.items())
    def values(k):return [j[k]for j in jobs if j.get(k)is not None]
    return dict(schema='RPE_COMPLETED_OPERATIONAL_LOG_ANALYSIS_V1',observed_run=str(run.resolve()),
        snapshot_UTC=datetime.now(timezone.utc).isoformat(),base_execution_source='1e88cd85cc8d2739a4dce43d138b94b153309333',
        completed_jobs=len(jobs),read_seconds=time.time()-started,
        progress_snapshot_sha256=hashlib.sha256(progress_bytes).hexdigest(),progress_complete_lines=len(progress),
        window=dict(start=begin,end=cutoff,seconds=total,resolution='progress UTC seconds; terminal launch timestamps retain subsecond precision'),
        concurrency=dict(configured_slots=slots,observed_peak=peak,time_weighted_jobs=occupied/total,
            slot_utilization=occupied/(slots*total),idle_slot_seconds=slots*total-occupied,
            seconds_by_active_jobs=dict(sorted(hist.items())),right_censored_jobs=len(active)),
        job_wall_seconds=summary(values('wall')),launch_to_scientific_call_seconds=summary(values('launch_to_call')),
        metadata_seconds=summary(values('metadata_seconds')),DP_seconds=summary(values('DP_seconds')),
        sampled_job_tree_peak_RSS_KiB=summary(values('sampled_tree_peak_RSS_KiB')),
        observed_CPU_per_job_wall=summary([j['tree_CPU_seconds']/j['wall']for j in jobs if j['tree_CPU_seconds']is not None]),
        configured_DP_worker_samples=worker_specs,
        candidate_wait_idle_seconds=None,candidate_wait_reason='No enqueue/ready/wait intervals in legacy progress log',
        adopted_lookahead_fraction=None,adoption_reason='Legacy run has LAUNCH/EXIT only; no durable ADOPT trace yet. No running checkpoint inspected.',
        real_Q3_speedup_measured=False,new_real_Q3_calls=0,
        limits=['No simultaneous memory peak inferred by adding individual peaks.',
                'CPU service ratio does not establish absence/presence of host saturation.',
                'Right-censored progress intervals and one-second timestamps limit utilization precision.'],
        terminal_job_projection=jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(a.run.resolve()):raise ValueError('Observed run is read-only')
    result=analyze(a.run);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k in ('completed_jobs','concurrency','launch_to_scientific_call_seconds','job_wall_seconds')},indent=2))


if __name__=='__main__':main()
