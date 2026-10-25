"""Artificial cache recovery: keep the actual compiled-action check enabled."""
import gzip
import io
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_b3_recover as r
r.b.setup()


def test_recovered_cache_hit_supplies_compiled_circuit_for_action_check(tmp_path,monkeypatch):
    from qiskit import QuantumCircuit,qpy
    from qiskit.quantum_info import Statevector
    qc=QuantumCircuit(1);qc.h(0);qc.rz(.3,0)
    stream=io.BytesIO();qpy.dump(qc,stream)
    cache=tmp_path/'compiled_cache';cache.mkdir()
    (cache/'artificial_compiled.qpy.gz').write_bytes(gzip.compress(stream.getvalue()))
    monkeypatch.setattr(r.b,'PRIVATE',tmp_path)
    monkeypatch.setattr(r,'ORIGINAL_COMPILE',lambda *a:({'metrics':{}},None,True))
    _,restored,reused=r.compiled_with_representative_action(qc,'artificial',True)
    assert restored is not None and reused
    assert np.linalg.norm(Statevector.from_instruction(restored).data-Statevector.from_instruction(qc).data)<1e-15


def test_event_replay_comparison_preserves_raw_bytes_under_compression(tmp_path):
    raw=tmp_path/'events_PRIVATE.json';raw.write_bytes(b'{"synthetic_events": [1, 2, 3]}\n')
    expected=r.event_digest(raw)
    raw.with_suffix('.json.gz').write_bytes(gzip.compress(raw.read_bytes()))
    raw.unlink()
    assert r.event_digest(raw)==expected
