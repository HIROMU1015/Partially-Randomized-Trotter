"""Artificial saved QPY only: process isolation and preservation of old prices."""
import gzip
import io
import json
from pathlib import Path
import sys

import pytest
from qiskit import QuantumCircuit, qpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import recover_h4_p3_stage3_process as recovery


def fixture_bytes():
    circuit = QuantumCircuit(2, 1)
    circuit.h(1)
    circuit.cp(.23, 1, 0)
    circuit.sdg(1)
    circuit.h(1)
    circuit.measure(1, 0)
    stream = io.BytesIO()
    qpy.dump(circuit, stream)
    decoded = stream.getvalue()
    raw = gzip.compress(decoded, mtime=0)
    row = dict(path='toy.qpy.gz', sha256=recovery.sha(raw), stored_bytes=len(raw),
               decoded_QPY_sha256=recovery.sha(decoded), decoded_bytes=len(decoded))
    return raw, row


def test_saved_qpy_checks_both_byte_hashes():
    raw, row = fixture_bytes()
    assert recovery.checked_qpy_bytes(raw, row) == gzip.decompress(raw)
    with pytest.raises(ValueError, match='COMPRESSED_BYTES'):
        recovery.checked_qpy_bytes(raw + b'x', row)
    with pytest.raises(ValueError, match='QPY_BYTES'):
        recovery.checked_qpy_bytes(raw, dict(row, decoded_QPY_sha256='0' * 64))


def test_spawn_process_is_one_target_and_has_same_compiler_result(tmp_path):
    kernel = recovery.original.modules()
    raw, row = fixture_bytes()
    circuit = qpy.load(io.BytesIO(gzip.decompress(raw)))[0]
    _, expected = kernel.price(circuit)
    expected = json.loads(json.dumps(expected))
    # More tasks than workers verifies actual interpreter retirement, not just spawn.
    with recovery.process_pool(2) as pool:
        futures = [pool.submit(recovery.compile_saved_target, 'toy' + str(i), raw, row,
                   str(tmp_path), expected['compiler'], expected['compiler_hash'], 2) for i in range(3)]
        result = [f.result() for f in futures]
    assert len({worker['pid'] for _, _, worker in result}) == 3
    for key, metrics, worker in result:
        assert worker['compile_calls'] == 1
        assert {k: metrics[k] for k in expected} == expected
        saved = json.loads((tmp_path / ('prices/' + key + '/metrics.json')).read_text())
        assert saved == metrics
        output = tmp_path / metrics['output_QPY']['path']
        recovery.checked_qpy_bytes(output.read_bytes(), metrics['output_QPY'])


def test_merge_does_not_modify_saved_metrics_and_rejects_overlap_or_missing():
    old = {'old': {'RZ': 7}}
    new = {'new': {'RZ': 9}}
    assert recovery.merge_prices(old, new, {'old': {}, 'new': {}}, {'new': {}})['old'] is old['old']
    assert old == {'old': {'RZ': 7}}
    with pytest.raises(ValueError, match='TARGET_SET'):
        recovery.merge_prices(old, old, old, old)
    with pytest.raises(ValueError, match='INCOMPLETE'):
        recovery.merge_prices(old, new, {'extra': {}}, {'new': {}})


def test_planned_interaction_uses_matching_axis_and_never_adds_depth():
    prices = {}
    for axis in ('cosine', 'sine'):
        prices['R/' + axis] = dict(RZ=10, CX=20, size=30, depth=300)
        prices['Empty/' + axis] = dict(RZ=1, CX=2, size=3, depth=3)
        for arm in ('P', 'DF'):
            for i in range(4):
                prices[f'{arm}_{i}/F/' + axis] = dict(RZ=9, CX=19, size=29, depth=29)
                prices[f'{arm}_{i}/E/' + axis] = dict(RZ=4, CX=5, size=6, depth=7)
    values = recovery.interactions(prices)['values']
    assert len(values) == 16
    assert all(row == dict(RZ=-4, CX=-4, size=-4) for row in values.values())
