"""Exact identity encoding, independent of scientific packages and filesystems."""
import hashlib
import json
import math
import re
import time
from .streaming import LazyList


class Stop(RuntimeError):
    """Mandatory return to review; never retry or repair scientific inputs."""


def require(condition, reason):
    if not condition:
        raise Stop(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_id(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'missing actual hash')
    return value


def exact(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        require(math.isfinite(value), 'nonfinite number')
        return {'real64_hex': value.hex()}
    if type(value) is complex:
        require(math.isfinite(value.real) and math.isfinite(value.imag), 'nonfinite complex')
        return {'complex128_hex': [value.real.hex(), value.imag.hex()]}
    if isinstance(value, (list, tuple, LazyList)):
        return [exact(v) for v in value]
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), 'nonstring key')
        return {k: exact(v) for k, v in sorted(value.items())}
    raise Stop('unsupported/symbolic identity value')


def canonical(value):
    return json.dumps(exact(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def exact_json_parts(value):
    """Yield canonical JSON without allocating a second exact container tree."""
    if value is None:
        yield 'null'
    elif type(value) is bool:
        yield 'true' if value else 'false'
    elif type(value) is int:
        yield str(value)
    elif type(value) is str:
        yield json.encoder.encode_basestring(value)
    elif type(value) in (float,complex):
        yield from exact_json_parts(exact(value))
    elif isinstance(value,(list,tuple,LazyList)):
        yield '['
        for index,item in enumerate(value):
            if index:yield ','
            yield from exact_json_parts(item)
        yield ']'
    elif isinstance(value,dict):
        require(all(type(k) is str for k in value),'nonstring key')
        yield '{'
        for index,(key,item) in enumerate(sorted(value.items())):
            if index:yield ','
            yield json.encoder.encode_basestring(key)
            yield ':'
            yield from exact_json_parts(item)
        yield '}'
    else:
        raise Stop('unsupported/symbolic identity value')


def canonical_chunks(value, chunk_size=65536):
    """Bound retained encoding bytes, including a single long JSON string."""
    require(type(chunk_size) is int and 0 < chunk_size <= 65536, 'encoding chunk budget')
    pending = bytearray()
    for part in exact_json_parts(value):
        for start in range(0, len(part), chunk_size // 4 or 1):
            encoded = part[start:start + (chunk_size // 4 or 1)].encode()
            while encoded:
                count = min(chunk_size - len(pending), len(encoded))
                pending.extend(encoded[:count]); encoded = encoded[count:]
                if len(pending) == chunk_size:
                    yield bytes(pending)
                    pending.clear()
                    time.sleep(0)
    if pending:
        yield bytes(pending)


def fingerprint(domain, payload):
    digest = hashlib.sha256()
    for chunk in canonical_chunks({'domain': domain, **payload}):
        digest.update(chunk)
    return digest.hexdigest()


def uint_seed(domain, payload):
    return int.from_bytes(bytes.fromhex(fingerprint(domain, payload))[:8], 'big')


INPUT_FIELDS = ('geometry', 'H', 'DF', 'state', 'input', 'template', 'source', 'compiler', 'environment', 'wrapper_semantics')


def trajectory_seed(identity, index, *, actual_inputs_frozen, signal_launch):
    require(actual_inputs_frozen and signal_launch, 'seeds require input freeze and signal launch')
    require(type(index) is int and 0 <= index < 32, 'trajectory index')
    require(set(identity) == set(INPUT_FIELDS), 'seed identity closure')
    for k in ('H', 'DF', 'state', 'input', 'template', 'compiler', 'environment'):
        hash_id(identity[k])
    require(isinstance(identity['source'],str) and re.fullmatch('[0-9a-f]{40}',identity['source']), 'actual source commit')
    return uint_seed('h4-trajectory-v1', {'master_seed': 20261006, **identity, 'index': index})


def step_seed(parent, outer_step, short_step, occurrence, draw_kind):
    require(type(parent) is int and 0 <= parent < 2**64, 'parent seed')
    require(all(type(n) is int and n >= 0 for n in (outer_step, short_step, occurrence)), 'step indices')
    require(draw_kind in ('order', 'component'), 'draw kind')
    return uint_seed('h4-step-occurrence-v1', dict(parent_trajectory_seed=parent,
                    outer_step=outer_step, short_step=short_step, occurrence=occurrence, draw_kind=draw_kind))


def wrapper_key(identity, axis, seed, index):
    require(axis in ('cosine', 'sine'), 'axis')
    require((seed is None) == (index is None), 'baseline seed/index pair')
    return fingerprint('h4-wrapper-v1', {**identity, 'axis': axis, 'trajectory_seed': seed, 'index': index})
