"""Metadata-only legacy bridge and strict primitive/native error connection.

No synthesis, worker, provider construction, or scientific forecast. Saved
cache rows are eligible metadata, never revalidated operators on conversion.
Only explicitly requested off-domain tests may call the interval guard here.
"""
from fractions import Fraction as F
import hashlib
import json

from .parent_allocation import rational, integer, check_cache_metadata, native_plan


def operator_target(event, *, identity, multiply, involution, rotation, scale):
    """Provider-neutral diagnostic target, including endpoints and time sign.

    Unlike the old G9 reference, never takes atan(None) and never discards the
    sign. Caller supplies independent matrix/algebra primitives and must obtain
    authorization before using registered providers. No operators at import.
    """
    native_plan(event, lambda i: [], lambda i,r,s: [], control=0)
    out = identity()
    for i in event['word']: out = multiply(out, involution(i))
    kind = event.get('event_kind', 'rotation')
    if kind == 'rotation':
        out = multiply(rotation(event['child'], rational(event['ratio']),
                                event.get('rotation_sign',1)), out)
    elif kind == 'pure_child': out = multiply(involution(event['child']), out)
    return scale(out, 1j**event['phase_i_power'])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest_value(value, size=64):
    if not isinstance(value, str) or len(value) != size or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('malformed digest')
    return value


def bound_json(raw, *, expected_bytes, expected_sha256):
    """Bind raw saved documents; expected identities come from frozen evidence."""
    integer(expected_bytes, 'expected bytes')
    if not isinstance(raw, bytes) or len(raw) != expected_bytes or sha(raw) != digest_value(expected_sha256):
        raise ValueError('saved document byte identity mismatch')
    return json.loads(raw)


def tool_binding(tool_raw, *, expected_tool_sha256, runtime_executable_sha256,
                 options, interval_dps):
    """Expanded key metadata. Does not adopt a tool or validate its runtime."""
    tool = bound_json(tool_raw, expected_bytes=len(tool_raw), expected_sha256=expected_tool_sha256)
    digest_value(runtime_executable_sha256)
    integer(interval_dps, 'interval dps', 1)
    if (options.get('up_to_phase') is not False or 'seed' not in options
            or isinstance(options['seed'], bool) or not isinstance(options['seed'], (int, str))):
        raise ValueError('explicit seed and strict-phase options required')
    source = digest_value(tool['source_py_tree_sha256']['pygridsynth'])
    return dict(name='pygridsynth', version=tool['packages']['pygridsynth'],
                source_sha256=source, seed=options['seed'],
                tool_identity_sha256=expected_tool_sha256,
                runtime_executable_sha256=runtime_executable_sha256,
                package_lock_sha256=sha(canonical(tool['packages'])),
                python=tool['python'], synthesizer_options=json.loads(canonical(options)),
                interval_dps=interval_dps)


def legacy_spec(expected):
    """Exact D1/D3 inventory spec, without consulting a scientific inventory."""
    spec = expected['spec']; tool = spec['tool']
    required = ('tool_identity_sha256', 'runtime_executable_sha256',
                'package_lock_sha256', 'synthesizer_options', 'interval_dps')
    if any(k not in tool for k in required):
        raise ValueError('expanded acquisition identity required; seed alone is insufficient')
    for k in required[:3]: digest_value(tool[k])
    if tool['synthesizer_options'].get('up_to_phase') is not False or tool['synthesizer_options'].get('seed') != tool['seed']:
        raise ValueError('tool options/seed identity mismatch')
    integer(tool['interval_dps'], 'interval dps', 1)
    if (spec['operation'] != 'RZ' or spec['angle_kind'] != 'atan' or spec['scale'] != '1'
            or spec['angle_units'] != 'radians' or spec['up_to_phase'] is not False
            or spec['signed_inverse'] != 'actual_sequence_adjoint'):
        raise ValueError('incompatible primitive/control convention')
    return dict(angle_kind='atan', ratio=str(rational(spec['ratio'])), scale='1',
                epsilon=str(rational(spec['precision'])), options=tool['synthesizer_options'],
                tool_identity_sha256=tool['tool_identity_sha256'],
                relative_controlled_phase='strict; signed inverse from actual adjoint')


def bridge_legacy_row(saved, expected, *, provenance):
    """Normalize metadata and counts; NEVER run a saved operator validation.

    Caller must bind the raw cache/contract/inventory/guard audit to frozen Git
    evidence. A row does not itself establish its originating tool/runtime.
    No historical-source phase guard is silently replaced or relaxed.
    """
    spec = legacy_spec(expected)
    for field in ('source_commit', 'result_commit'): digest_value(provenance[field], 40)
    for field in ('cache_sha256', 'saved_row_sha256', 'contract_sha256',
                  'inventory_sha256', 'guard_audit_sha256', 'strict_guard_source_sha256'):
        digest_value(provenance[field])
    if (provenance.get('saved_row_sha256') != sha(canonical(saved))
            or provenance.get('tool_identity_sha256') != spec['tool_identity_sha256']
            or provenance.get('runtime_executable_sha256') != expected['spec']['tool']['runtime_executable_sha256']
            or provenance.get('inventory_spec') != spec
            or provenance.get('inventory_key') != sha(canonical(spec))
            or provenance.get('guard_policy') != 'STRICT_RTE_RZ_FROBENIUS_NO_PHASE_MINIMIZATION'):
        raise ValueError('legacy provenance/spec/guard policy mismatch')
    ratio, eps = rational(spec['ratio']), rational(spec['epsilon'])
    if ratio <= 0 or eps <= 0: raise ValueError('positive interior primitive required')
    angle_key = 'atan:'+str(ratio)+':scale:1'
    sequence = saved['sequence']
    if (not isinstance(sequence, str) or any(c not in 'HTtSXW' for c in sequence)
            or saved['angle_key'] != angle_key or saved['key'] != angle_key+':epsilon:'+str(eps)
            or saved['epsilon'] != str(eps) or saved['error_pass'] is not True
            or saved['sequence_sha256'] != sha(sequence.encode())):
        raise ValueError('legacy sequence/key/strict receipt mismatch')
    counts = dict(T_count=sequence.count('T')+sequence.count('t'),
                  Tdagger_count=sequence.count('t'), global_W_count=sequence.count('W'),
                  one_qubit_count=len(sequence)-sequence.count('W'))
    if any(type(saved[k]) is not int or saved[k] != v for k,v in counts.items()):
        raise ValueError('legacy stored sequence/count mismatch')
    error = rational(saved['strict_operator_error_upper'])
    if not 0 <= error <= eps: raise ValueError('legacy saved error receipt failed')
    result = dict(key=expected['key'], spec=expected['spec'], sequence=sequence,
                  sequence_sha256=saved['sequence_sha256'], up_to_phase=False,
                  operator_error_upper=str(error), complete=True, **counts,
                  legacy_provenance=dict(provenance), metadata_only=True,
                  strict_guard_rechecked=False, eligible_for_guarded_reuse=False)
    check_cache_metadata(result, expected)
    return result


def interval_angle(ratio, *, dps):
    """Exact rational interval endpoints for atan(ratio), no point synthesis."""
    q = rational(ratio); integer(dps, 'dps', 1)
    if q <= 0: raise ValueError('positive ratio required; endpoints are exact gates')
    import mpmath as mp
    from .rte_reallocation.numeric import enclosure, ivf
    old = mp.iv.dps
    try:
        mp.iv.dps = dps
        lo, hi = enclosure(mp.iv.atan2(ivf(q), mp.iv.mpf(1)))
        return dict(lo=lo, hi=hi, radians=True, interval_dps=dps)
    finally:
        mp.iv.dps = old


def strict_primitive_guard(sequence, ratio, *, dps):
    """Use the existing strict guard, retaining W and restoring precision.

    This is an operator evaluation. Registered/saved scientific sequences must
    not be passed here without separate execution authorization.
    """
    q = rational(ratio); integer(dps, 'dps', 1)
    if q <= 0: raise ValueError('positive ratio required')
    import mpmath as mp
    from .rte_reallocation.native import Angle
    from .rte_reallocation.numeric import configure, strict_guard
    old = mp.mp.dps, mp.iv.dps
    try:
        configure(dps)
        return strict_guard(sequence, Angle('atan', q))
    finally:
        mp.mp.dps, mp.iv.dps = old


def recheck_entry(entry, expected, *, guard=strict_primitive_guard):
    """Explicit separate guard call, no synthesis. Off-domain mocks allowed."""
    check_cache_metadata(entry, expected)
    legacy_spec(expected)
    actual = rational(guard(entry['sequence'], rational(expected['spec']['ratio']),
                            dps=expected['spec']['tool']['interval_dps']))
    if not 0 <= actual <= rational(expected['spec']['precision']):
        raise ValueError('independent strict operator guard failed')
    binding = dict(key=entry['key'], sequence_sha256=entry['sequence_sha256'],
                   operator_error_upper=str(actual), interval_dps=expected['spec']['tool']['interval_dps'])
    return dict(entry, operator_error_upper=str(actual), strict_guard_rechecked=True,
                guard_binding_sha256=sha(canonical(binding)),
                eligible_for_guarded_reuse=True, metadata_only=False)


def checked_entry(entry):
    expected = dict(key=entry['key'], spec=entry['spec'])
    check_cache_metadata(entry, expected)
    legacy_spec(expected)
    s = entry['sequence']
    binding = dict(key=entry['key'], sequence_sha256=entry['sequence_sha256'],
                   operator_error_upper=entry['operator_error_upper'],
                   interval_dps=entry['spec']['tool']['interval_dps'])
    if (entry.get('strict_guard_rechecked') is not True or entry.get('eligible_for_guarded_reuse') is not True
            or entry.get('guard_binding_sha256') != sha(canonical(binding))
            or any(type(entry[k]) is not int or entry[k] != v for k,v in dict(
                T_count=s.count('T')+s.count('t'), Tdagger_count=s.count('t'),
                global_W_count=s.count('W'), one_qubit_count=len(s)-s.count('W')).items())):
        raise ValueError('metadata-only, incomplete or unbound native cache')
    return entry


def native_accounting(gates, cache):
    """Additive logical cost and telescoping primitive error, no forecast.

    cache uses canonical ratio strings, normalized metadata AND rechecked guard.
    Error targets the ideal atan primitive, so backend angle rounding is already
    included. Additional independent angle approximations need a separate bound.
    """
    out = {'T':0, 'CX':0, '1Q':0}; error = F(0); rotations = 0
    for raw in gates:
        g = tuple(raw)
        if not g: raise ValueError('empty gate')
        name = g[0]
        if name == 'R':
            if len(g) != 4 or type(g[3]) is not int or g[3] not in (-1,1):
                raise ValueError('invalid signed R primitive')
            integer(g[1], 'wire')
            ratio = rational(g[2]); row = checked_entry(cache[str(ratio)])
            if rational(row['spec']['ratio']) != ratio: raise ValueError('cache ratio differs from R token')
            out['T'] += row['T_count']; out['1Q'] += row['one_qubit_count']
            error += rational(row['operator_error_upper']); rotations += 1
        elif name in ('CX','CZ'):
            if len(g) != 3 or g[1] == g[2]: raise ValueError('invalid controlled gate')
            integer(g[1], 'wire'); integer(g[2], 'wire')
            out['CX'] += 1; out['1Q'] += 2*(name=='CZ')
        elif name in ('H','X','Y','Z','S','s','T','t','W','w'):
            if len(g) != 2: raise ValueError('invalid single-qubit descriptor')
            integer(g[1], 'wire')
            out['T'] += name in ('T','t'); out['1Q'] += name not in ('W','w')
        else: raise ValueError('unsupported gate; no missing-price zero')
    return dict(**out, strict_native_error_upper=error, R_occurrences=rotations,
                up_to_phase=False, whole_circuit_optimized=False, forecast=False)


def realized_time_gates(gates, cache):
    """Expand saved written-product strings into circuit TIME order.

    Positive ABC product applies C,B,A; its adjoint applies A†,B†,C†.
    W/w are retained as scalar gates on the addressed primitive wire.
    """
    inverse = {'H':'H','X':'X','S':'s','T':'t','t':'T','W':'w'}
    result = []
    for g in gates:
        if g[0] != 'R': result.append(tuple(g)); continue
        if len(g) != 4 or type(g[3]) is not int or g[3] not in (-1,1): raise ValueError('invalid sign')
        integer(g[1], 'wire')
        row = checked_entry(cache[str(rational(g[2]))])
        if rational(row['spec']['ratio']) != rational(g[2]): raise ValueError('cache ratio mismatch')
        s = row['sequence']
        letters = reversed(s) if g[3] == 1 else (inverse[c] for c in s)
        result.extend((letter,g[1]) for letter in letters)
    return result
