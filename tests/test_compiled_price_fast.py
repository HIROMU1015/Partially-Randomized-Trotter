"""Synthetic-only equivalence of the actual pinned C1 price_requests paths."""
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest
from compiled_price_artificial import fixture, backend, compile_local, F, E, dfq, costs, native
import h46_b3_fast_fingerprint as fast_hash


@pytest.fixture(autouse=True)
def no_scientific_io_or_sampling(monkeypatch):
    import numpy as np
    from trotterlib import rte
    from trotterlib.df_rte_circuit import DFRTEEventPreparation
    def forbidden(*args, **kwargs):
        raise AssertionError('No scientific input or sampler in technical tests')
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(rte, 'sample_rte_events', forbidden)
    monkeypatch.setattr(rte, 'iter_sample_rte_events', forbidden)
    monkeypatch.setattr(DFRTEEventPreparation, 'sample_occurrence_request', forbidden)


@pytest.mark.parametrize('q,R,scalar,negative,mixed', [(1, 1, 0., False, False),
    (3, 7, .173, False, True), (2, 4, -.21, True, True)])
def test_structural_outputs_equal_original_including_suffix_and_folding(q, R, scalar, negative, mixed):
    s, ctx, c, _, _, requests = fixture(q, R, scalar=scalar, negative=negative, mixed=mixed)
    events = [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    expected = [s.structural(ctx.Accounting, ctx.blocks, c, scalar, a, events) for a in s.m.AXES]
    assert F.structural_pair(ctx.Accounting, ctx.blocks, c, scalar, events) == expected


def test_projection_preserves_native_instruction_validation():
    from trottertracks.track_d_native_resources import NativeInstruction
    counter = F._RZCounter(9, 1)
    with pytest.raises(TypeError):
        counter.feed(object())
    with pytest.raises(ValueError, match='outside'):
        counter.feed(NativeInstruction('rz', (9,), params=(.1,)))


def run(science, ctx, cell, plan, requests, path, accelerated):
    records = []
    original = ctx.compile
    def compile_record(qc, settings, **kw):
        # Capture the recursive identity on both sides of the unchanged compiler.
        before = fast_hash.legacy.canonical_qiskit_circuit_fingerprint(qc)
        cost = original(qc, settings, **kw)
        records.append(dict(input=before, compiled=fast_hash.legacy.canonical_qiskit_circuit_fingerprint(
            cost.transpiled_circuit), settings=cost.compiler_settings_hash,
            measurement=dict(cost.transpiled_circuit.count_ops()).get('measure', 0)))
        return cost
    ctx.compile = compile_record
    try:
        with compile_local(), E.efficient_controlled_zz(dfq):
            out = backend(science, ctx, plan, accelerated).price_requests(
                cell, 0, path, requests, [], [], conditional=False)
    finally:
        ctx.compile = original
    return out, records


def test_pinned_backend_and_fast_path_have_identical_compiled_circuits_and_records(tmp_path):
    s, ctx, _, cell, plan, requests = fixture()
    plan['runtime']['old_cache'] = str(tmp_path / 'empty_cache')
    old, old_compile = run(s, ctx, cell, plan, requests, tmp_path / 'old', False)
    new, new_compile = run(s, ctx, cell, plan, requests, tmp_path / 'new', True)
    assert old_compile == new_compile
    assert len(new_compile) == 2  # each axis actually compiled
    assert all(v['measurement'] == 1 for v in new_compile)
    for k in old:
        if k != 'source':
            assert old[k] == new[k], k
    for axis in s.m.AXES:
        assert json.loads((tmp_path / 'old' / (axis + '.json')).read_text()) == json.loads(
            (tmp_path / 'new' / (axis + '.json')).read_text())
    assert (tmp_path / 'old/events_PRIVATE.json').read_bytes() == (tmp_path / 'new/events_PRIVATE.json').read_bytes()
    timing = json.loads((tmp_path / 'new/TECHNICAL_TIMING.json').read_text())
    assert all(a + '_compile' in timing['stages_seconds'] for a in s.m.AXES)


def test_wrapper_canonical_bytes_mutations_and_restore():
    s, ctx, c, _, _, requests = fixture(1, 3)
    original = native.canonical_qiskit_circuit_fingerprint
    with E.efficient_controlled_zz(dfq), fast_hash.use_fast_fingerprint() as fp:
        evolution = ctx.h['compose'](c, ctx.prep, requests, ctx.scalar, fp.fingerprint)
        for axis in s.m.AXES:
            wrapper = native.build_native_hadamard_wrapper(evolution, axis=axis, max_instructions=100000)
            stream = io.BytesIO()
            fast_hash.FastWriter(stream).circuit(wrapper)
            expected = bytearray()
            fast_hash.legacy.write_canonical_bytes(wrapper, expected.extend)
            assert stream.getvalue() == expected
            before = fp.fingerprint(wrapper)
            wrapper.global_phase += .013
            assert fp.fingerprint(wrapper) != before
            assert fp.fingerprint(wrapper) == original(wrapper)
        evolution.circuit.x(0)
        with pytest.raises(ValueError, match='mutated'):
            native.build_native_hadamard_wrapper(evolution, axis='cosine', max_instructions=100000)
    assert native.canonical_qiskit_circuit_fingerprint is original


def test_cache_identity_and_compiler_binding_are_preserved(tmp_path):
    s, ctx, _, cell, plan, requests = fixture(1, 2)
    cache = tmp_path / 'cache'
    cache.mkdir()
    plan['runtime']['old_cache'] = str(cache)
    first, _ = run(s, ctx, cell, plan, requests, tmp_path / 'first', True)
    for axis in s.m.AXES:
        value = json.loads((tmp_path / 'first' / (axis + '.json')).read_text())
        (cache / (value['fingerprint'] + '.json')).write_text(json.dumps(value))
    def forbidden(*args, **kwargs):
        raise AssertionError('Cache must serve both axes')
    ctx.compile = forbidden
    with E.efficient_controlled_zz(dfq):
        result = backend(s, ctx, plan, True).price_requests(cell, 0, tmp_path / 'hit', requests, [], [], conditional=False)
        assert result['cache_hits'] == 2 and result['new_compiles'] == 0
        target = cache / (first['fingerprints'][0] + '.json')
        bad = json.loads(target.read_text())
        bad['compiler'] = 'DIFFERENT'
        target.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='CACHE_BINDING_NO_FALLBACK'):
            backend(s, ctx, plan, True).price_requests(cell, 0, tmp_path / 'bad', requests, [], [], conditional=False)


def test_compiler_input_mutation_still_fails(tmp_path):
    s, ctx, _, cell, plan, requests = fixture(1, 2)
    plan['runtime']['old_cache'] = str(tmp_path / 'cache')
    original = ctx.compile
    def mutating(qc, *args, **kwargs):
        cost = original(qc, *args, **kwargs)
        qc.x(0)
        return cost
    ctx.compile = mutating
    with compile_local(), E.efficient_controlled_zz(dfq), pytest.raises(ValueError, match='MUTATED_INPUT'):
        backend(s, ctx, plan, True).price_requests(cell, 0, tmp_path / 'bad', requests, [], [], conditional=False)


def test_lowering_and_allocation_checks_remain(tmp_path):
    s, ctx, _, cell, plan, requests = fixture(1, 2)
    plan['runtime']['old_cache'] = str(tmp_path / 'cache')
    b = backend(s, ctx, plan, True)
    with pytest.raises(ValueError, match='Efficient controlled ZZ'):
        b.price_requests(cell, 0, tmp_path / 'no_lowering', requests, [], [], conditional=False)
    with E.efficient_controlled_zz(dfq), pytest.raises(ValueError, match='ALLOCATION'):
        b.price_requests(cell, 0, tmp_path / 'wrong', [], [], [], conditional=False)
