"""Unadopted fused driver exact + audit packing. Standard JSON format unchanged.

Reference semantics are audit.canonical(driver.exact(value)), not F3's separate
$rat/$tuple evidence codec. No timers, source monkeypatches, caches, or authority.
"""
from fractions import Fraction
import json
from . import g6_w2c_cost_preparation as audit


class _ReferenceRequired(Exception):
    pass


def _exact_reference(value):
    if isinstance(value, dict):
        return {str(k): _exact_reference(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_exact_reference(v) for v in value)
    return value


def _packed(value):
    t = type(value)
    if t is Fraction:
        return {'rat': [hex(value.numerator), hex(value.denominator)]}
    if t is int:
        return {'int': hex(value)}
    if value is None or t in (str, bool):
        return value
    if t in (list, tuple):
        return [_packed(v) for v in value]
    if t is dict:
        if all(type(k) is str for k in value):
            return {k: _packed(v) for k, v in value.items()}
        # Normalize key collisions before validating values, like exact then pack.
        # Unusual keys/container subclasses take the complete reference path,
        # preserving evaluation order and unsupported-overwritten-value behavior.
        keys = {}
        for k, v in value.items():
            if type(k) not in (str, int, bool, Fraction) and k is not None:
                raise _ReferenceRequired()
            key = str(k)
            if key in keys:
                raise _ReferenceRequired()
            keys[key] = v
        return {k: _packed(v) for k, v in keys.items()}
    raise _ReferenceRequired()


def canonical(value):
    try:
        packed = _packed(value)
    except _ReferenceRequired:
        return audit.canonical(_exact_reference(value))
    # Python's original encoder handles non-ASCII, surrogate, DEL/control chars.
    return json.dumps(packed, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
