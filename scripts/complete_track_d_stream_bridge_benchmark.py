#!/usr/bin/env python3
"""Complete only the eight unstarted old optimizations after a keyword bug.

Keep the frozen runner, successful fast outputs and original failure unchanged.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
import time
from unittest.mock import patch

import run_track_d_stream_bridge_fast as campaign
from run_track_d_h4_native_bridge import ROOT,read,write,sha,git

ORIGINAL_BASELINE=campaign.baseline


def compatible_baseline(operations,nq,nc,*,global_phase):
    return ORIGINAL_BASELINE(operations,nq,nc,global_phase)


def old_worker(row):
    with patch.object(campaign,'baseline',compatible_baseline):
        return campaign.benchmark_worker((row,'old'))


def main():
    out=ROOT/campaign.OUT;input_commit='b5b1ad594c9261ebf00e0fcb5700d885a98d11ae'
    data=(out/'INPUTS.json').read_bytes()
    if data!=git('show',input_commit+':'+campaign.OUT+'/INPUTS.json'):raise ValueError('fixed input commit')
    failure=read(out/'FAILED_benchmark.json')
    if failure['error']!="baseline() got an unexpected keyword argument 'global_phase'":raise ValueError('specific pre-call error required')
    rows=read(out/'INPUTS.json')['records'];fast=[];preserved={}
    for row in rows:
        path=out/(row['case']+'_fast.json');record=read(path)
        if record['status']!='PASS' or not all(record['checks'].values()):raise ValueError('fast output not qualified')
        if (out/(row['case']+'_old.json')).exists():raise ValueError('old optimization already completed; no rerun')
        fast.append(record);preserved[path.name]=sha(path.read_bytes())
    original_source=read(out/'SOURCE.json')
    for p,h in original_source['files'].items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('frozen original code changed')
    source=git('rev-parse','HEAD').decode().strip();relative='scripts/complete_track_d_stream_bridge_benchmark.py'
    code=(ROOT/relative).read_bytes()
    if code!=git('show',source+':'+relative):raise ValueError('unfixed adapter source')
    write(out/'COMPLETION_SOURCE.json',dict(execution_source=source,adapter_sha256=sha(code),
        original_source=original_source['execution_source'],input_commit=input_commit,
        failure_sha256=sha((out/'FAILED_benchmark.json').read_bytes()),preserved_fast_files=preserved,
        change='keyword-to-positional adapter only; frozen old/fast algorithms unchanged',
        first_fast_wave_requested_processes=16,old_completion_tasks=8,rerun_completed_fast=0))
    start=time.perf_counter();old=[];workers=min(len(rows),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(old_worker,r) for r in rows]):
            row=f.result();old.append(row);print('OLD_COMPLETED',row['case'],'PASS',flush=True)
    for name,h in preserved.items():
        if sha((out/name).read_bytes())!=h:raise ValueError('retained fast result changed')
    write(out/'RESULT.json',dict(status='ALL_EXACT_PASS',input_commit=input_commit,
        completion_source=source,first_fast_wave_requested_processes=16,old_completion_workers=workers,
        old_completion_wall_seconds=time.perf_counter()-start,paired_simultaneous=False,
        retained_fast_records=8,newly_completed_old_records=8,rerun_completed_fast=0,
        records=sorted(fast+old,key=lambda r:(r['case'],r['variant'])),new_P0=0,new_sampling=0,new_simulation=0,q16_or_above=False))
    write(out/'STOP.json',dict(mandatory_stop=True,status='FIXED_Q4_Q8_COMPLETE_AFTER_KEYWORD_ADAPTER',
        original_failure_preserved=True,next_owner='user/GPT'))


if __name__=='__main__':main()
