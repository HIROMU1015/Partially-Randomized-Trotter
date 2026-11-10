#!/usr/bin/env python3
"""Fixed saved q4/q8 native differential and one common-pool benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import time
from unittest.mock import patch

import run_track_d_stream_bridge_fast as campaign
from run_track_d_h4_native_bridge import ROOT, read, write, sha, git
from trottertracks import track_d_stream_bridge_fast as frozen
from trottertracks.track_d_bridge_run_reuse import optimize_native_runs

BASE = 'e6bd6c5dedf27601b0c3b50877653ad43d04cd17'
OUT = 'artifacts/track_d_bridge_run_reuse/2026-11-10'
DOC = 'docs/research/track_d_bridge_run_reuse_20261110'
FILES = tuple(campaign.PINNED) + (
    'src/trottertracks/track_d_stream_bridge_fast.py',
    'src/trottertracks/track_d_bridge_run_reuse.py',
    'scripts/run_track_d_stream_bridge_fast.py',
    'scripts/run_track_d_bridge_run_reuse.py',
    'tests/check_track_d_bridge_run_reuse.py', DOC+'/PLAN.md')


def worker(task):
    row, variant, mode = task
    trace = []
    function = frozen.optimize_native_stream if variant == 'previous' else optimize_native_runs
    def traced_function(*a, **kw):
        if variant == 'reuse':
            return function(*a, **kw, trace=lambda i, ops, p: trace.append(campaign.word_hash(ops)))
        class TraceCX(frozen.ObservedCX):
            def operations(self):
                result = list(super().operations())
                trace.append(campaign.word_hash(result))
                return iter(result)
        with patch.object(frozen, 'ObservedCX', TraceCX):
            return function(*a, **kw)
    record = None
    def save(path, value):
        nonlocal record
        if path != ROOT/campaign.OUT/(row['case']+'_fast.json'):
            raise ValueError('unexpected old writer path')
        value['variant'] = variant
        if mode == 'correctness':
            value['round_word_sha256'] = trace
        reference = read(ROOT/campaign.OUT/(row['case']+'_fast.json'))
        value['checks']['round_phase_bits'] = (
            [v.hex() for v in value['phase_additions']] ==
            [v.hex() for v in reference['phase_additions']])
        if not all(value['checks'].values()):
            raise ValueError('round phase mismatch')
        write(ROOT/OUT/mode/(row['case']+'_'+variant+'.json'), value)
        record = value
    with (patch.object(campaign, 'optimize_native_stream',
                       traced_function if mode == 'correctness' else function),
          patch.object(campaign, 'write', save)):
        campaign.benchmark_worker((row, 'fast'))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['correctness','performance'])
    mode = parser.parse_args().mode
    old = ROOT/campaign.OUT
    data = (old/'INPUTS.json').read_bytes()
    if data != git('show', BASE+':'+campaign.OUT+'/INPUTS.json'):
        raise ValueError('frozen inputs index changed')
    rows = read(old/'INPUTS.json')['records']
    if len(rows) != 8 or {r['q'] for r in rows} != {4,8}:
        raise ValueError('fixed eight wrappers')
    source = git('rev-parse','HEAD').decode().strip()
    pins = {}
    for p in FILES:
        raw = (ROOT/p).read_bytes()
        if raw != git('show',source+':'+p):
            raise ValueError('unfixed execution SOURCE: '+p)
        if p not in ('src/trottertracks/track_d_bridge_run_reuse.py',
                     'scripts/run_track_d_bridge_run_reuse.py',
                     'tests/check_track_d_bridge_run_reuse.py',DOC+'/PLAN.md'):
            if raw != git('show',BASE+':'+p):
                raise ValueError('protected SOURCE changed: '+p)
        pins[p] = sha(raw)
    if mode == 'performance' and read(ROOT/OUT/'correctness/RESULT.json')['status'] != 'ALL_EXACT_PASS':
        raise ValueError('correctness must pass before measurement')
    out = ROOT/OUT/mode
    out.mkdir(parents=True, exist_ok=False)
    write(out/'SOURCE.json', dict(execution_source=source, base=BASE, files=pins,
        fixed_input_index_sha256=sha(data), raw_public=False,
        scope='saved q4/q8 eight native inputs; no new P0/sampling/simulation'))
    tasks = [(r,v,mode) for r in rows for v in ('previous','reuse')]
    workers = min(len(tasks),len(os.sched_getaffinity(0)))
    records = []
    start = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(worker,t) for t in tasks]):
                r = future.result()
                records.append(r)
                print(mode,r['case'],r['variant'],'PASS',flush=True)
        pairs = []
        for case in sorted({r['case'] for r in records}):
            a = next(r for r in records if r['case']==case and r['variant']=='previous')
            b = next(r for r in records if r['case']==case and r['variant']=='reuse')
            checks = dict(word=a['output_word_sha256']==b['output_word_sha256'],
                phase=a['global_phase'].hex()==b['global_phase'].hex(),
                precise_phase=a['precise_phase']==b['precise_phase'], rounds=a['rounds']==b['rounds'],
                metrics=a['metrics']==b['metrics'],
                round_phase_bits=[v.hex() for v in a['phase_additions']]==[v.hex() for v in b['phase_additions']])
            if mode == 'correctness':
                checks['all_round_words'] = a['round_word_sha256']==b['round_word_sha256']
            if not all(checks.values()):
                raise ValueError('paired differential mismatch: '+case)
            pairs.append(dict(case=case,checks=checks,
                speedup=a['performance']['total_seconds']/b['performance']['total_seconds'],
                rss_reduction_fraction=1-b['performance']['peak_RSS_MiB']/a['performance']['peak_RSS_MiB']))
        write(out/'RESULT.json',dict(status='ALL_EXACT_PASS',mode=mode,execution_source=source,
            workers=workers,wall_seconds=time.perf_counter()-start,paired_same_pool=True,
            records=sorted(records,key=lambda r:(r['case'],r['variant'])),pairs=pairs,
            new_P0=0,new_sampling=0,new_raw=0,new_simulation=0,q16_or_above=False))
    except Exception as error:
        write(out/'FAILED.json',dict(status='STOP_PARTIAL',error_type=type(error).__name__,
            error=str(error).replace(str(ROOT),'<task>')))
        raise


if __name__ == '__main__':
    main()
