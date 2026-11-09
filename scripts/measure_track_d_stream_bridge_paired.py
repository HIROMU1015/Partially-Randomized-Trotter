#!/usr/bin/env python3
"""One controlled simultaneous old/new measurement of the same eight inputs.

Keep the first measurement/failure unchanged. No new fixture or rewrite rule.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
import time
from unittest.mock import patch

import run_track_d_stream_bridge_fast as campaign
from complete_track_d_stream_bridge_benchmark import compatible_baseline
from run_track_d_h4_native_bridge import ROOT,read,write,sha,git

OUT='artifacts/track_d_stream_bridge_fast/2026-11-10/run02_paired'


def worker(task):
    row,variant=task
    expected=ROOT/campaign.OUT/(row['case']+'_'+variant+'.json')
    original_write=campaign.write
    def route(path,value):
        if path!=expected:raise ValueError('unexpected result write path')
        original_write(ROOT/OUT/path.name,value)
    with (patch.object(campaign,'baseline',compatible_baseline),patch.object(campaign,'write',route)):
        return campaign.benchmark_worker(task)


def main():
    original=ROOT/campaign.OUT;inputs=(original/'INPUTS.json').read_bytes()
    input_commit='b5b1ad594c9261ebf00e0fcb5700d885a98d11ae'
    if inputs!=git('show',input_commit+':'+campaign.OUT+'/INPUTS.json'):raise ValueError('fixed input bytes')
    rows=read(original/'INPUTS.json')['records']
    if len(rows)!=8 or {r['q'] for r in rows}!={4,8}:raise ValueError('fixed eight wrappers only')
    pins=read(original/'SOURCE.json')['files']
    for p,h in pins.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('algorithm SOURCE changed')
    for p in ['scripts/measure_track_d_stream_bridge_paired.py','scripts/complete_track_d_stream_bridge_benchmark.py']:
        data=(ROOT/p).read_bytes()
        if data!=git('show','HEAD:'+p):raise ValueError('unfixed measurement adapter')
        pins[p]=sha(data)
    source=git('rev-parse','HEAD').decode().strip();out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    write(out/'SOURCE.json',dict(execution_source=source,input_commit=input_commit,files=pins,
        input_sha256=sha(inputs),purpose='same simultaneous pool after keyword fix',
        previous_results_unchanged=True,new_conditions=False))
    tasks=[(r,v) for r in rows for v in ('old','fast')];start=time.perf_counter();records=[]
    workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            row=future.result();records.append(row);print('PAIRED',row['case'],row['variant'],'PASS',flush=True)
    write(out/'RESULT.json',dict(status='ALL_EXACT_PASS',execution_source=source,input_commit=input_commit,
        workers=workers,wall_seconds=time.perf_counter()-start,paired_simultaneous=True,
        records=sorted(records,key=lambda r:(r['case'],r['variant'])),new_P0=0,new_sampling=0,new_simulation=0,q16_or_above=False))
    write(out/'STOP.json',dict(status='CONTROLLED_PAIRED_COMPLETE',mandatory_stop=True,next_owner='user/GPT'))


if __name__=='__main__':main()
