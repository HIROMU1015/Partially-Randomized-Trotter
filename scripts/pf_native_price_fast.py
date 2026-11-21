"""Opt-in PF compose acceleration, exact same native builder and canonical SHA.

Process-local, one compose only. No persistent cache, emitter or compiler fork.
"""
from contextlib import contextmanager
import importlib
import threading
import time


def compose_reusing_parts(original, row, prep, requests, scalar, fingerprint, *, stats=None):
    if row['method'] not in ('FULL', 'B0I') or requests:
        raise ValueError('Deterministic PF compose only')
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ValueError('PF compose reuse requires the single worker main thread')
    from trottertracks.resource_applicability import ax2b_native_df_v5 as api
    builder = api.build_deterministic_native
    cache = {}; references = []
    counts = dict(builder_calls=0, reused_parts=0, distinct_parts=0)

    def build(blocks, **kw):
        # Every builder argument participates; block references stay alive and
        # immutable throughout the unchanged compose loop. compose(copy=True)
        # leaves the cached template unchanged, checked again below.
        key = (tuple(id(b) for b in blocks), tuple(sorted((k, type(v).__name__, v.hex() if isinstance(v, float) else v) for k, v in kw.items())))
        counts['builder_calls'] += 1
        if key in cache:
            counts['reused_parts'] += 1
            return cache[key]
        references.append(tuple(blocks))
        part = builder(blocks, **kw); cache[key] = part
        counts['distinct_parts'] += 1
        return part

    api.build_deterministic_native = build
    try:
        result = original(row, prep, requests, scalar, fingerprint)
        for part in cache.values():
            if fingerprint(part.circuit) != part.circuit_fingerprint:
                raise ValueError('PF compose modified a cached native template')
        return result
    finally:
        api.build_deterministic_native = builder
        if stats is not None: stats.update(counts)


@contextmanager
def accelerated_native(native):
    """Use existing fast writer in the original cost_trial, restore on all exits."""
    import h46_b3_fast_fingerprint as fast
    reuse = importlib.import_module('trottertracks.resource_applicability.h6_hash_reuse_v1')
    old_context, old_compose, old_trial = reuse.use_fingerprint_reuse, native.compose, native.cost_trial
    timings = []
    reuse.use_fingerprint_reuse = fast.use_fast_fingerprint

    def compose(row, prep, requests, scalar, fingerprint):
        stats = {}; started = time.monotonic()
        try:
            return compose_reusing_parts(old_compose, row, prep, requests, scalar, fingerprint, stats=stats)
        finally:
            timings.append(dict(compose_seconds=time.monotonic()-started, **stats))

    def trial(job, **options):
        start = len(timings)
        result = old_trial(job, **options)
        # Return-only technical annotation; original private raw result stays unchanged.
        return dict(result, technical_acceleration=timings[start:])

    native.compose = compose
    native.cost_trial = trial
    try:
        yield
    finally:
        native.compose = old_compose
        native.cost_trial = old_trial
        reuse.use_fingerprint_reuse = old_context
