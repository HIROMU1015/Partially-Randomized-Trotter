#!/usr/bin/env python3
"""Read-only audit of known artificial trial files; no science module imports."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT/'.pilot-runs'
DOC = ROOT/'docs/research/track_b_g6_w2c_b3_performance_scaling_pilot_20261019'


def inspect_file(item):
    path, row = item
    h, previous = sha256(), '0'*64
    counters, indices = Counter(), {'X': 0, 'Y': 0}
    seen = set()
    for line in Path(path).open('rb'):
        h.update(line)
        envelope = json.loads(line)
        expected = sha256((previous+envelope['payload']).encode()).hexdigest()
        assert envelope['previous'] == previous and envelope['sha256'] == expected
        previous = expected
        payload = json.loads(envelope['payload'])
        if 'trial_id' not in payload:
            continue
        trial_id = payload['trial_id']
        assert trial_id not in seen
        seen.add(trial_id)
        axis = payload['axis']
        index = int(trial_id.rsplit('/', 1)[-1])
        assert index == indices[axis]
        indices[axis] += 1
        counters[payload['kind']] += 1
    assert len(seen) == row['completed']
    assert h.hexdigest() == row['file_sha256'] and previous == row['chain_sha256']
    assert Path(path).stat().st_size == row['bytes']
    if row['status'] == 'OK':
        assert indices == dict.fromkeys(['X', 'Y'], row['task']['n'])
        assert counters == row['result']['counts']
        for axis in row['result']['axes']:
            assert axis['H_bits_supplied'] == 160*indices[axis['axis']]
    else:
        assert counters == row['failure']['completed_prefix']
    return len(seen), len(h.digest()), row['bytes']


def main():
    started = time.monotonic()
    items, total_started, metadata_pairs = [], 0, 0
    for suite in ['scaling', 'optimizations', 'stop']:
        for line in (RAW/suite/'results.jsonl').read_text().splitlines():
            row = json.loads(line)
            if 'workers_results' not in row:
                continue
            repeat, workers, case = row['repeat'], row['workers'], row['case']
            if suite == 'scaling':
                folder = RAW/suite/f'{case}_r{repeat}_w{workers}'
            elif suite == 'stop':
                folder = RAW/suite/f'fail_r{repeat}_w{workers}'
            elif case in ('mixed', 'heavy'):
                folder = RAW/suite/f"{case}_{row['variant']}_b{row['buffer_bytes']}_r{repeat}"
            elif case == 'startup64_tiny':
                folder = RAW/suite/f"reuse{row['reuse']}_r{repeat}"
            else:
                folder = RAW/suite/f'pass_overlap_r{repeat}_w{workers}'
            known = set()
            by_stream = {}
            for worker in row['workers_results']:
                key = worker['task']['key']
                assert key not in known
                known.add(key)
                items.append((str(folder/(key+'.jsonl')), worker))
                if worker['status'] == 'OK':
                    task = worker['task']
                    stream_key = task['replica'], task['arm']
                    event_metadata = [(a['event_trace_sha256'], a['first_correctness_prefix_sha256'], a['H_bits_supplied'])
                                      for a in worker['result']['axes']]
                    if stream_key in by_stream:
                        assert by_stream[stream_key] == event_metadata
                        metadata_pairs += 1
                    by_stream[stream_key] = event_metadata
            if row.get('worker_start_logs_complete'):
                starts = [(w['task']['key'], s['trial_id']) for w in row['workers_results'] for s in w['started']]
                assert len(starts) == len(set(starts))
                total_started += len(starts)
    with ProcessPoolExecutor(max_workers=16) as pool:
        checked = list(pool.map(inspect_file, items, chunksize=8))
    result = {'artificial_only': True, 'files_checked': len(items), 'completed_trials_checked': sum(x[0] for x in checked),
              'bytes_read': sum(x[2] for x in checked), 'duplicate_trial_ids_within_file': 0,
              'global_identity': '(run directory, task key, local R1-shaped trial_id)',
              'duplicate_task_keys_within_run': 0, 'failure_start_records_checked': total_started,
              'cross_layer_pass_stream_metadata_pairs_checked': metadata_pairs,
              'chain_file_sha_bytes_and_counts': 'PASS', 'axis_trial_indices_and_bit_counts': 'PASS',
              'wall_s': time.monotonic()-started,
              'child_cpu_s': resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
              'science_sampling': 0, 'not_an_independent_reviewer': True}
    (DOC/'all_trial_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
