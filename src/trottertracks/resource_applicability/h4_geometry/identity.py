"""Exact identity encoding, independent of scientific packages and filesystems."""
import hashlib
import json
import math
import re
import time


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
    if isinstance(value, (list, tuple)):
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
    elif isinstance(value,(list,tuple)):
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


def fingerprint(domain, payload):
    # Both normalization and JSON encoding are lazy: the monitor must also
    # run while traversing a large numerical circuit's container graph.
    digest = hashlib.sha256()
    pending = bytearray()
    for part in exact_json_parts({'domain': domain, **payload}):
        pending.extend(part.encode())
        if len(pending) >= 65536:
            digest.update(pending)
            pending.clear()
            time.sleep(0)  # process-local cooperative scheduling only
    if pending:
        digest.update(pending)
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
