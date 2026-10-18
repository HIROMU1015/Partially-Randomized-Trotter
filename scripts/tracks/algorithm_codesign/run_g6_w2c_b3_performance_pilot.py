#!/usr/bin/env python3
"""Artificial benchmark runner. All timing output is technical, not G7 data."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from fractions import Fraction as F
from hashlib import sha256
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_performance_pilot as P


def environment():
    result = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'python': sys.version, 'platform': platform.platform(),
              'affinity': sorted(os.sched_getaffinity(0)), 'cpu_count': os.cpu_count(),
              'load': os.getloadavg(), 'packages': {p: importlib.metadata.version(p) for p in ['gmpy2', 'orjson']}}
    for name in ('/proc/meminfo', '/proc/stat', '/proc/pressure/cpu', '/proc/pressure/memory',
                 '/proc/pressure/io', '/proc/self/cgroup'):
        try:
            result[name] = Path(name).read_text()
        except OSError as exc:
            result[name] = str(exc)
    for command in (['lscpu'], ['df', '-B1', str(ROOT)],
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader']):
        try:
            q = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[' '.join(command)] = q.stdout or q.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[' '.join(command)] = str(exc)
    return result


def inventory():
    files = ['src/trottertracks/algorithm_codesign/'+f for f in
             ('g6_w2c_b3_driver.py', 'g6_w2c_cost_preparation.py', 'g6_w2c_b3_performance_pilot.py')]
    files += [str(Path(__file__).relative_to(ROOT)),
              'tests/tracks/algorithm_codesign/test_g6_w2c_b3_performance_pilot.py']
    return {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'files': {p: {'sha256': sha256((ROOT/p).read_bytes()).hexdigest(), 'bytes': (ROOT/p).stat().st_size}
                      for p in files}, 'science_sampling': 0, 'science_compile': 0,
            'real_authorization_or_claim_operations': 0}


def signature(row):
    return {r['task']['key']: {'trace': r['result']['trace_sha256'], 'axes': r['result']['axes'],
                             'normalized_file': r['normalized_file_sha256'],
                             'normalized_chain': r['normalized_chain_sha256']}
            for r in row['workers_results']}


def micro_kernel(data):
    return sha256(data).digest()


def _micro(spec, output, send):
    import gmpy2, orjson  # Warm import explicitly outside timed inner loops.
    kind, variant = spec['kind'], spec['variant']
    bits = spec.get('bits', 64)
    z = F((1 << bits)+123, (1 << (bits+2))+57)
    payload = {'ARTIFICIAL': True, 'word': tuple(range(16)), 'ratio': z,
               'proof': [F(i+1, i+19) for i in range(32)], 'bytes': 'x'*spec.get('size', 512)}
    text = P.D.audit.canonical(P.D.exact(payload))
    data = text.encode()
    previous = 'a'*64
    meter = P.D.Meter(P.D.clock)
    stream = None
    if kind == 'rational':
        operation = {'reference': P.atan_reference, 'recurrence': P.atan_recurrence, 'gmp': P.atan_gmp}[variant]
        def call():
            return operation(z, 32)
    elif kind == 'serialization':
        def call():
            return (P.D.audit.canonical(P.D.exact(payload)) if variant == 'reference' else
                    P.canonical_fast(payload, 'orjson' if variant == 'orjson' else 'python'))
    elif kind == 'hash':
        if variant == 'concat':
            def call():
                return sha256((previous+text).encode()).digest()
        else:
            prefix_bytes = previous.encode()
            def call():
                h = sha256(prefix_bytes)
                h.update(data)
                return h.digest()
    elif kind == 'io':
        path = Path(output)/(spec['name']+'.jsonl')
        stream = (P.D.Stream(path, meter, byte_limit=1 << 32, flush_rows=4096) if variant == 'reference' else
                  P.PrototypeStream(path, meter, byte_limit=1 << 32, flush_rows=4096,
                                    backend='orjson' if 'orjson' in variant else 'python',
                                    buffer_bytes=65536 if 'buffer' in variant else 0))
        def call():
            stream.emit(payload)
            return stream.previous
    elif kind == 'observer':
        def call():
            if 'stage' in variant:
                with meter.span('synthetic_kernel'):
                    return micro_kernel(data)
            return micro_kernel(data)
    else:
        raise ValueError(kind)
    for _ in range(3):
        call()
    begin = time.perf_counter()
    for _ in range(3):
        call()
    estimate = (time.perf_counter()-begin)/3
    loops = max(3, int(.20/max(estimate, 1e-9)))
    observer = P.D.FunctionObserver([__file__]) if kind == 'observer' and 'function' in variant else None
    begin = P.D.clock()
    last = None
    with observer if observer else nullcontext():
        for _ in range(loops):
            last = call()
    if stream:
        stream.close()
    elapsed = P.D.delta(begin, P.D.clock())
    checksum = sha256(last if isinstance(last, bytes) else P.canonical_fast(last).encode()).hexdigest()
    result = dict(spec, loops=loops, cpu_s=elapsed[0]/1e9, wall_s=elapsed[1]/1e9,
                  cpu_per_op_s=elapsed[0]/1e9/loops, wall_per_op_s=elapsed[1]/1e9/loops,
                  memory=P.D.memory_snapshot(), bytes=stream.bytes if stream else 0,
                  checksum=checksum, stage_cost=meter.rows)
    if stream:
        result['readback'] = P.readback(path)
        assert result['readback']['file_sha256'] == stream.file_hash.hexdigest()
    P.assert_no_science()
    send.send(result)
    send.close()


def run_micro(spec, output):
    ctx = mp.get_context('fork')
    recv, send = ctx.Pipe(False)
    process = ctx.Process(target=_micro, args=(spec, output, send))
    process.start()
    send.close()
    try:
        result = recv.recv()
    finally:
        recv.close()
        process.join()
    if process.exitcode:
        raise RuntimeError('MICRO_FAILED')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['scaling', 'optimizations', 'hotpaths', 'stop'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'environment_start.json').write_text(json.dumps(environment(), indent=2))
    (args.output/'source_inventory.json').write_text(json.dumps(inventory(), indent=2))
    log = (args.output/'results.jsonl').open('x')
    def emit(row):
        log.write(json.dumps(row, default=P.json_default)+'\n')
        log.flush()
        print(json.dumps({k: row[k] for k in ('suite', 'case', 'repeat', 'workers', 'wall_s', 'total_cpu_s',
                                             'throughput', 'starts_after_failure', 'completed_after_failure') if k in row}), flush=True)
    rng = random.Random(20261019)
    if args.suite == 'scaling':
        warm = P.scheduler(P.tasks_for(8, 16), 8, args.output/'warmup')
        (args.output/'warmup.json').write_text(json.dumps(warm, default=P.json_default))
        cases = [('pair8_light', 8, 1024, 'light', [1, 2, 4, 8]),
                 ('pair8_mixed', 8, 512, 'mixed', [1, 2, 4, 8]),
                 ('pair8_heavy', 8, 64, 'heavy', [1, 2, 4, 8]),
                 ('synthetic64_mixed', 64, 256, 'mixed', [1, 2, 4, 8, 16, 32, 64])]
        for case, count, n, workload, settings in cases:
            expected = None
            for repeat in range(args.repeats):
                order = settings.copy()
                rng.shuffle(order)
                for workers in order:
                    row = P.scheduler(P.tasks_for(count, n, workload), workers,
                                      args.output/f'{case}_r{repeat}_w{workers}')
                    if any(r['status'] != 'OK' for r in row['workers_results']):
                        raise AssertionError('ARTIFICIAL_WORKER_FAILED')
                    current = signature(row)
                    if expected is None:
                        expected = current
                    if current != expected:
                        raise AssertionError('SCALING_CONTENT_TRACE_BITS_MISMATCH')
                    emit(dict(row, suite=args.suite, case=case, repeat=repeat, equivalence=True))
    elif args.suite == 'optimizations':
        for workload, n in [('mixed', 512), ('heavy', 64)]:
            expected = None
            variants = [('reference', 0), ('recurrence', 0), ('gmp', 0), ('orjson', 0), ('orjson', 65536)]
            for repeat in range(args.repeats):
                order = variants.copy()
                rng.shuffle(order)
                for variant, buffer in order:
                    tasks = [replace(t, buffer_bytes=buffer) for t in P.tasks_for(8, n, workload, variant)]
                    row = P.scheduler(tasks, 8, args.output/f'{workload}_{variant}_b{buffer}_r{repeat}')
                    current = signature(row)
                    if expected is None:
                        expected = current
                    if current != expected:
                        raise AssertionError('OPTIMIZATION_CONTENT_TRACE_BITS_MISMATCH')
                    emit(dict(row, suite=args.suite, case=workload, variant=variant, buffer_bytes=buffer,
                              repeat=repeat, equivalence=True))
        for repeat in range(args.repeats):
            for reuse in (False, True):
                row = P.scheduler(P.tasks_for(64, 1), 8, args.output/f'reuse{reuse}_r{repeat}', reuse=reuse)
                emit(dict(row, suite=args.suite, case='startup64_tiny', repeat=repeat))
            for workers in (1, 8, 16):
                row = P.scheduler(P.tasks_for(16, 64, pass_overlap=True), workers,
                                  args.output/f'pass_overlap_r{repeat}_w{workers}')
                emit(dict(row, suite=args.suite, case='pass_overlap16', repeat=repeat))
    elif args.suite == 'hotpaths':
        specs = []
        for bits in (32, 96, 256):
            specs += [dict(kind='rational', variant=v, bits=bits) for v in ('reference', 'recurrence', 'gmp')]
        specs += [dict(kind='serialization', variant=v) for v in ('reference', 'fused', 'orjson')]
        for size in (512, 16384, 262144):
            specs += [dict(kind='hash', variant=v, size=size) for v in ('concat', 'incremental')]
        specs += [dict(kind='io', variant=v) for v in ('reference', 'fused', 'orjson', 'orjson_buffer')]
        specs += [dict(kind='observer', variant=v) for v in ('none', 'stage', 'function', 'stage_function')]
        for repeat in range(args.repeats):
            order = specs.copy()
            rng.shuffle(order)
            for i, spec in enumerate(order):
                spec = dict(spec, name=f'ARTIFICIAL_micro_r{repeat}_{i}')
                row = run_micro(spec, args.output)
                emit(dict(row, suite=args.suite, case=spec['kind'], repeat=repeat))
    else:
        for buffer in (0, 65536):
            row = P.buffer_crash_probe(args.output/f'ARTIFICIAL_crash_b{buffer}.jsonl', buffer)
            emit(dict(row, suite=args.suite, case='buffer_crash', repeat=0))
        for repeat in range(args.repeats):
            for workers in (1, 2, 4, 8):
                tasks = [replace(t, track_starts=True) for t in P.tasks_for(16, 64)]
                tasks[0] = replace(tasks[0], failure_trial=3)
                row = P.scheduler(tasks, workers, args.output/f'fail_r{repeat}_w{workers}')
                assert row['starts_after_failure'] == 0
                assert any(r['status'] == 'FAILED' for r in row['workers_results'])
                emit(dict(row, suite=args.suite, case='injected_sample_failure', repeat=repeat))
    log.close()
    P.assert_no_science()
    (args.output/'environment_end.json').write_text(json.dumps(environment(), indent=2))
    (args.output/'ARTIFICIAL_SUITE_STOP.json').write_text(json.dumps({'technical_suite_complete': True,
        'real_G7_sampling': 0, 'science_compile': 0, 'real_authorization_or_claim': 0,
        'forbidden_imports': P.BLOCKED_IMPORTS, 'retry': 0, 'resume': False}))


if __name__ == '__main__':
    main()
