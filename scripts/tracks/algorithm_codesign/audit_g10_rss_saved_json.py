"""Saved JSON / I/O-only memory diagnostic. Never imports or calls G10 science.

Three predefined, isolated modes concern the already-published technical JSON.
They cannot reconstruct the original G10 heap, retry G10, or promote its prefix.
Only the pure serial() AST is extracted from the fixed source in materialized
mode; no source module is imported. Results are small JSON lines on stdout.
"""
import argparse
import ast
import gc
import hashlib
import inspect
import json
import os
import platform
import resource
import signal
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT/'artifacts/track_b_g10_degree_result/2026-10-10/v1/result_v1.json'
SERIAL_SOURCE = ROOT/'src/trottertracks/algorithm_codesign/g10_saved.py'
EXPECTED_RESULT = 'b62695c19964a5a121b965c39142427bfa8048efad9bce05f3221494bb14dfe1'
RSS_CAP_KIB = 512*1024
phase = 'startup'
started = time.monotonic()
samples = []


def usage():
    u = resource.getrusage(resource.RUSAGE_SELF)
    current = None
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                current = int(line.split()[1])
                break
    return {'wall_seconds': time.monotonic()-started,
            'CPU_seconds': u.ru_utime+u.ru_stime,
            'peak_RSS_KiB': u.ru_maxrss, 'current_RSS_KiB': current}


def emit(kind, **fields):
    print(json.dumps({'kind': kind, 'phase': phase, **usage(), **fields}), flush=True)


def checkpoint(name, **fields):
    global phase
    phase = name
    emit('phase_snapshot', **fields)
    guard()


def guard(*_):
    u = resource.getrusage(resource.RUSAGE_SELF)
    if u.ru_maxrss > RSS_CAP_KIB:
        raise MemoryError('saved-only diagnostic RSS cap512 MiB; no G10 invocation')
    if time.monotonic()-started >= 60 or u.ru_utime+u.ru_stime >= 60:
        raise TimeoutError('saved-only diagnostic 60s wall/CPU cap')


def serial_function():
    tree = ast.parse(SERIAL_SOURCE.read_bytes())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'serial')
    namespace = {'F': Fraction}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SERIAL_SOURCE), 'exec'), namespace)
    return namespace['serial']


def census(tree):
    # Structural lengths and allocation sizes only; never evaluates cost/mean.
    seen, stack, counts = set(), [tree], Counter()
    allocation, string_chars, dict_keys = 0, 0, 0
    while stack:
        v = stack.pop()
        token = id(v)
        if token in seen:
            continue
        seen.add(token)
        allocation += sys.getsizeof(v)
        counts[type(v).__name__] += 1
        if isinstance(v, dict):
            assert all(type(k) is str for k in v), 'saved JSON key schema'
            dict_keys += len(v)
            stack.extend(v.keys())
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, str):
            string_chars += len(v)
    return {'unique_objects': len(seen), 'object_counts': dict(counts),
            'deep_getsizeof_bytes_excluding_census_workspace': allocation,
            'unique_string_characters': string_chars, 'dict_key_occurrences': dict_keys}


def encoded_shape(value):
    chunks = bytes_total = shallow_chunk_bytes = 0
    for part in json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False).iterencode(value):
        chunks += 1
        bytes_total += len(part.encode('utf-8'))
        shallow_chunk_bytes += sys.getsizeof(part)
    return {'encoder_chunks': chunks, 'UTF8_bytes_without_newline': bytes_total,
            'sum_chunk_getsizeof_occurrences_bytes': shallow_chunk_bytes,
            'sum_chunk_getsizeof_is_not_unique_RSS_or_exact_live_memory': True,
            'pointer_array_lower_bytes_64bit': 8*chunks}


def run(mode):
    emit('diagnostic_identity', mode=mode, python=platform.python_version(),
         executable=os.path.realpath(sys.executable), science_runner_invocations=0,
         saved_payload_is_technical_prefix=True, original_heap_recreated=False,
         diagnostic_limits={'RSS_MiB':512, 'AS_MiB':1536, 'wall_seconds':60, 'CPU_seconds':60},
         stdlib_json_encoder_path=inspect.getfile(json.encoder),
         stdlib_json_encoder_sha256=hashlib.sha256(Path(inspect.getfile(json.encoder)).read_bytes()).hexdigest())
    resource.setrlimit(resource.RLIMIT_AS, (1536*1024**2, 1536*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, 0.01, 0.01)
    global phase
    phase = 'read_saved_bytes'
    b = RESULT.read_bytes()
    assert hashlib.sha256(b).hexdigest() == EXPECTED_RESULT
    checkpoint('saved_bytes_loaded', bytes=len(b))
    phase = 'decode_saved_JSON'
    tree = json.loads(b)
    assert tree['status'] == 'G10_TECHNICAL_INCONCLUSIVE'
    assert tree['prefix_rows_usable_for_final_research_decision'] is False
    checkpoint('JSON_decoded_with_bytes_live')
    del b
    gc.collect()
    checkpoint('decoded_tree_only')
    if mode == 'census':
        phase = 'structural_census'
        info = census(tree)
        checkpoint('census_finished', census=info)
        phase = 'count_encoder_chunks_without_retaining'
        shape = encoded_shape(tree)
        by_section = {k: encoded_shape(tree[k]) for k in ('rows', 'synthesis_cache', 'inventory', 'CTS_certificates')}
        ir_gates = sum(len(b['native_ir']) for r in tree['rows'] for b in r['events'])
        static_keys = {tuple(g) for r in tree['rows'] for b in r['events']
                       for g in b['native_ir'] if g[0] != 'R'}
        checkpoint('shape_finished', shape=shape, sections=by_section,
                   native_IR_gate_records=ir_gates, distinct_fixed_matrix_cache_keys=len(static_keys),
                   fixed_matrix_ndarray_data_bytes_if_n4=4096*len(static_keys),
                   this_is_static_size_bound_not_matrix_evaluation=True)
    elif mode == 'materialized':
        serial = serial_function()
        phase = 'recursive_serial_container_copy'
        converted = serial(tree)
        checkpoint('serial_copy_retained')
        phase = 'JSON_dumps_chunk_list_and_join'
        payload = json.dumps(converted, indent=2, ensure_ascii=False, allow_nan=False)
        checkpoint('dumps_string_retained', characters=len(payload))
        phase = 'newline_string_copy'
        payload = payload+'\n'
        checkpoint('newline_copy_retained')
        phase = 'UTF8_encode_byte_cap_copy'
        encoded = payload.encode()
        checkpoint('UTF8_bytes_retained', bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
        assert hashlib.sha256(encoded).hexdigest() == EXPECTED_RESULT
    elif mode == 'stream':
        phase = 'stream_saved_tree_no_serial_copy_no_chunk_list'
        digest, count, pieces = hashlib.sha256(), 0, 0
        for part in json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False).iterencode(tree):
            b = part.encode('utf-8')
            count += len(b)
            pieces += 1
            digest.update(b)
        digest.update(b'\n')
        count += 1
        assert digest.hexdigest() == EXPECTED_RESULT
        checkpoint('stream_byte_identity_verified', bytes=count, sha256=digest.hexdigest(), encoder_chunks=pieces)
    else:
        raise ValueError('unknown predefined diagnostic mode')
    emit('diagnostic_complete', mode=mode, scientific_result_generated=False,
         outcome_reclassified=False, mandatory_STOP=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('census', 'materialized', 'stream'), required=True)
    mode = parser.parse_args().mode
    try:
        run(mode)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        emit('diagnostic_stopped', mode=mode, reason=type(exc).__name__+': '+str(exc),
             scientific_result_generated=False, original_G10_failure_location_proven=False)
        sys.exit(2)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
