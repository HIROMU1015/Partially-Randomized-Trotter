#!/usr/bin/env python3
"""Synthetic C1 pipeline comparison in isolated, interleaved processes.

No molecular input, RNG sampling, saved price reanalysis or scientific launcher.
One child at a time avoids benchmark self-contention; this is not a resource cap.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT / 'scripts')]


def worker(send, q, R, mode):
    try:
        from compiled_price_artificial import fixture, backend, compile_local, E, dfq
        import numpy as np
        import qiskit
        from trotterlib import rte
        def forbidden(*a, **kw):
            raise AssertionError('Artificial benchmark forbids real input and sampling')
        np.load = rte.sample_rte_events = rte.iter_sample_rte_events = forbidden
        science, ctx, c, cell, plan, requests = fixture(q, R)
        stages = {}
        def timed(name, fn):
            def call(*args, **kw):
                start = time.perf_counter()
                result = fn(*args, **kw)
                stages[name] = stages.get(name, 0.) + time.perf_counter() - start
                return result
            return call
        if mode == 'baseline':
            science.structural = timed('structural', science.structural)
            ctx.h['compose'] = timed('compose', ctx.h['compose'])
            original_wrapper = ctx.native.build_native_hadamard_wrapper
            def wrapper(*args, **kwargs):
                return timed(kwargs['axis'] + '_wrapper', original_wrapper)(*args, **kwargs)
            ctx.native.build_native_hadamard_wrapper = wrapper
            original_context = ctx.fingerprint_context
            @contextmanager
            def fingerprint_context():
                with original_context() as fp:
                    original_hash = fp.fingerprint
                    class Measured:
                        def fingerprint(self, qc):
                            # Compose calls this for its final evolution too.
                            axis = 'cosine' if qc.num_clbits and not any(
                                x.operation.name == 'sdg' for x in qc.data[-4:]) else 'sine'
                            key = axis + ('_fingerprint' if calls.get(axis, 0) == 0 else '_mutation_check')
                            if not qc.num_clbits:
                                return original_hash(qc)
                            calls[axis] = calls.get(axis, 0) + 1
                            return timed(key, original_hash)(qc)
                    calls = {}
                    yield Measured()
            ctx.fingerprint_context = fingerprint_context
            original_compile = ctx.compile
            def compile_measured(qc, *args, **kwargs):
                axis = 'sine' if any(x.operation.name == 'sdg' for x in qc.data[-4:]) else 'cosine'
                return timed(axis + '_compile', original_compile)(qc, *args, **kwargs)
            ctx.compile = compile_measured
        with tempfile.TemporaryDirectory(prefix='d-price-artificial-') as directory:
            directory = Path(directory)
            plan['runtime']['old_cache'] = str(directory / 'empty_cache')
            b = backend(science, ctx, plan, mode == 'accelerated')
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            start = time.perf_counter()
            with compile_local(), E.efficient_controlled_zz(dfq):
                result = b.price_requests(cell, 0, directory / 'price', requests, [], [], conditional=False)
            seconds = time.perf_counter() - start
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            axes = {a: json.loads((directory / 'price' / (a + '.json')).read_text()) for a in science.m.AXES}
            if mode == 'accelerated':
                stages = json.loads((directory / 'price/TECHNICAL_TIMING.json').read_text())['stages_seconds']
            comparable = {k: result[k] for k in ('C', 'S', 'event_digest', 'fingerprints', 'compiler', 'cache_hits', 'new_compiles')}
            comparable['axes'] = axes
        send.send(dict(ok=True, mode=mode, q=q, R=R, event_positions=q * R, wall_seconds=seconds,
            peak_rss_bytes=peak, before_pipeline_highwater_bytes=rss_before, stages_seconds=stages,
            comparable=comparable, runtime=dict(numpy=np.__version__, qiskit=qiskit.__version__, python=sys.version.split()[0])))
    except BaseException:
        send.send(dict(ok=False, error=traceback.format_exc()))
    finally:
        send.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.output.exists():
        parser.error('Use an unused output path')
    rows = []
    for q, R in ((2, 4), (8, 32)):
        for repeat in range(args.repeats):
            for mode in (('baseline', 'accelerated') if repeat % 2 == 0 else ('accelerated', 'baseline')):
                ctx = mp.get_context('spawn')
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=worker, args=(send, q, R, mode))
                proc.start()
                send.close()
                value = recv.recv()
                proc.join()
                if not value['ok'] or proc.exitcode:
                    raise RuntimeError(value)
                value['repeat'] = repeat
                rows.append(value)
                print(q, R, repeat, mode, round(value['wall_seconds'], 3), value['peak_rss_bytes'], flush=True)
            pair = rows[-2:]
            if pair[0]['comparable'] != pair[1]['comparable']:
                raise ValueError('Baseline/accelerated identity or metrics differ')
    payload = dict(status='PASS_ARTIFICIAL_ONLY', rows=rows, real_H4_runs=0, saved_scientific_reanalysis=0,
        protocol='Fresh spawned child per observation; alternating order; wall excludes import/fixture; '
                 'RSS is child lifetime high-water including import/fixture. No tracemalloc.',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            'src/trottertracks/compiled_price_fast.py', 'scripts/h46_b3_fast_fingerprint.py',
            'tests/compiled_price_artificial.py', 'scripts/benchmark_compiled_price_artificial.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
