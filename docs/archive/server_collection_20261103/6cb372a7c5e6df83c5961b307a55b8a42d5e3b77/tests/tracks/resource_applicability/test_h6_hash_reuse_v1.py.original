"""Synthetic canonical bytes/mutation/bounds; no science arrays/transpile."""
import builtins
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.library import RZGate

from trotterlib.rte_compiled_cost import _circuit_payload, canonical_qiskit_circuit_fingerprint as legacy
from trottertracks.resource_applicability import h6_hash_reuse_v1 as f
from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as old


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    original, path_open = builtins.open, Path.open
    def check(p):
        if isinstance(p,(str,Path)) and str(p).endswith('.npz'):raise AssertionError('NO_SCIENTIFIC_ARRAYS')
    def guarded(p,*a,**k):check(p);return original(p,*a,**k)
    def guarded_path(p,*a,**k):check(p);return path_open(p,*a,**k)
    def forbidden(*a,**k):raise AssertionError('NO_SAMPLING_OR_TRANSPILE')
    monkeypatch.setattr(builtins,'open',guarded);monkeypatch.setattr(Path,'open',guarded_path)
    monkeypatch.setattr(np,'load',forbidden)
    import qiskit
    import trotterlib.rte_compiled_cost as cost
    monkeypatch.setattr(qiskit,'transpile',forbidden);monkeypatch.setattr(cost,'transpile',forbidden)


def fixture(kind='nested'):
    c=QuantumCircuit(13,2);c.global_phase=-.17
    d=QuantumCircuit(1);d.rz(.23,0);d.p(-.19,0);d.global_phase=.31
    g=Gate('shared',1,[]);g.definition=d
    if kind=='empty':return c
    if kind=='matrix':
        # Byte-only artificial 256x256 complex parameter, no unitary synthesis.
        a=np.arange(256*256,dtype=np.float64).reshape(256,256).astype(complex)*.001
        a[0,0]=complex(-.0,.0);c.append(Instruction('artificial_dense',8,0,[a]),range(8))
    elif kind=='conditional':
        c.rz(.27,1).c_if(c.clbits[1],1);c.x(2).c_if(c.clbits[0],1)
    elif kind=='leaves':
        c.append(Instruction('literal',1,0,[True,2,'数値',None,-.0,complex(.2,-.3),[.2,1]]),[0])
    elif kind=='control':
        c.append(RZGate(.17).control(ctrl_state=0),[0,1])
        c.append(g.control(),[1,2])
    else:
        for _ in range(8):c.append(g,[0],copy=False)
        c.cx(0,1);c.h(12);c.measure(12,1)
    return c


@pytest.mark.parametrize('kind',['empty','nested','matrix','conditional','leaves','control'])
def test_exact_old_bytes_and_digest_and_second_call_reuse(kind):
    c=fixture(kind);expected=json.dumps(_circuit_payload(c),sort_keys=True,separators=(',',':')).encode()
    session=f.FingerprintReuse();actual=bytearray()
    class Sink:
        def write(self,b):actual.extend(b)
    f._ReuseWriter(Sink(),session,f._Guard(session.stats)).circuit(c)
    assert bytes(actual)==expected
    assert session.fingerprint(c)==old(c)==legacy(c)==hashlib.sha256(expected).hexdigest()
    assert session.fingerprint(c)==legacy(c)
    assert session.stats['final_digest_hits']==1
    assert session.snapshot()['payload_peak_reserved_bytes']<=f.CACHE_BYTES


@pytest.mark.parametrize('change',['phase','definition','nested_param','control','array','dtype','shape','signed_zero','qubits','clbits','condition'])
def test_current_graph_guard_detects_cross_call_mutations(change):
    c=fixture('matrix' if change in ('array','dtype','shape','signed_zero') else 'nested')
    if change=='control':c=fixture('control')
    if change=='condition':c=fixture('conditional')
    session=f.FingerprintReuse();before=session.fingerprint(c)
    if change=='phase':c.global_phase+=.01
    elif change=='definition':c.data[0].operation.definition.x(0)
    elif change=='nested_param':
        d=c.data[0].operation.definition;item=d.data[0];op=item.operation.copy();op.params=[.251];d.data[0]=item.replace(operation=op)
    elif change=='control':c.data[0].operation.ctrl_state=1
    elif change=='array':c.data[0].operation.params[0][1,1]+=.001j
    elif change=='dtype':c.data[0].operation.params=[c.data[0].operation.params[0].astype(np.complex64)]
    elif change=='shape':c.data[0].operation.params=[c.data[0].operation.params[0].reshape(128,512)]
    elif change=='signed_zero':c.data[0].operation.params[0][0,0]=complex(.0,.0)
    elif change=='qubits':c.data[0]=c.data[0].replace(qubits=(c.qubits[1],))
    elif change=='clbits':c.data[-1]=c.data[-1].replace(clbits=(c.clbits[0],))
    else:c.data[0].operation.condition=(c.clbits[0],1)
    after=session.fingerprint(c)
    assert after==old(c)==legacy(c) and after!=before
    assert session.stats['final_digest_hits']==0


@pytest.mark.parametrize('kind',['symbolic','nan','array_nan','object','layout','calibration','recursive'])
def test_cache_hit_never_hides_invalid_changed_graph(kind):
    c=fixture();session=f.FingerprintReuse();session.fingerprint(c)
    if kind=='symbolic':c.rz(Parameter('x'),0)
    elif kind=='nan':c.append(Instruction('bad',1,0,[float('nan')]),[0])
    elif kind=='array_nan':c.append(Instruction('bad',1,0,[np.array([complex(float('nan'),0)])]),[0])
    elif kind=='object':c.append(Instruction('bad',1,0,[np.array(['x'],dtype=object)]),[0])
    elif kind=='layout':c._layout=object()
    elif kind=='calibration':c._calibrations={'x':{'synthetic':True}}
    else:c.data[0].operation.definition.append(c.data[0].operation,[0],copy=False)
    with pytest.raises((ValueError,TypeError)):session.fingerprint(c)


def test_content_keys_reuse_distinct_objects_and_changed_wrapper_retains_operations():
    session=f.FingerprintReuse();a=fixture();b=fixture()
    assert a is not b and session.fingerprint(a)==session.fingerprint(b)
    assert session.stats['final_digest_hits']==1
    hits=session.stats['operation_byte_hits'];b.sdg(12)
    assert session.fingerprint(b)==old(b)
    assert session.stats['operation_byte_hits']>hits
    assert all(type(k) is str and len(k)==64 for k in session.entries)


def test_condition_context_and_instruction_order_are_content_keys():
    session=f.FingerprintReuse();a=fixture('conditional');b=fixture('conditional')
    b.data=list(reversed(b.data))
    assert session.fingerprint(a)==old(a)
    assert session.fingerprint(b)==old(b)!=old(a)
    b.data[0].operation.condition=(b.clbits[1],1)
    assert session.fingerprint(b)==old(b)


@pytest.mark.parametrize('cap',[0,2**20,32*2**20])
def test_bounded_cache_fallback_is_byte_identical(cap,monkeypatch):
    monkeypatch.setattr(f,'GUARD_NODES',2)
    session=f.FingerprintReuse(byte_cap=cap);c=fixture('control')
    for i in range(6):
        c.rz(.013*i,3);assert session.fingerprint(c)==old(c)
        assert session.cached_bytes==sum(map(len,session.entries.values()))<=cap
        assert session.snapshot()['payload_peak_reserved_bytes']<=cap and session.reserved_bytes==0
    session.clear();assert not session.entries and not session.digests and session.cached_bytes==0


def test_private_process_aliases_restore_even_after_failure():
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native, h6_matched_memory_v3 as memory
    with pytest.raises(RuntimeError):
        with f.use_fingerprint_reuse() as session:
            assert native.canonical_qiskit_circuit_fingerprint.__self__ is session
            assert memory.canonical_qiskit_circuit_fingerprint.__self__ is session
            assert native.canonical_qiskit_circuit_fingerprint(fixture())==old(fixture())
            with pytest.raises(ValueError,match='ALIAS_ALREADY_CHANGED'):
                with f.use_fingerprint_reuse():pass
            raise RuntimeError('synthetic failure')
    assert native.canonical_qiskit_circuit_fingerprint is old and memory.canonical_qiskit_circuit_fingerprint is old
    assert session.cached_bytes==0 and not session.digests


def test_worker_wraps_only_frozen_cost_port_and_emits_reuse_statistics():
    from trottertracks.resource_applicability import h6_confirmation_fast_execution_v1 as execution
    text=inspect.getsource(execution.worker)
    assert 'from .h6_roundoff_cost_port_v1 import cost_task' in text
    assert 'use_preparations(preparations), use_fingerprint_reuse()' in text
    assert "writer.write('hash_reuse_statistics.json',hash_cache.snapshot())" in text


def test_lazy_standard_definition_memo_is_call_local_and_skips_explicit_override():
    session=f.FingerprintReuse();guard=f._Guard(session.stats);c=QuantumCircuit(1)
    a,b=RZGate(.17),RZGate(.17)
    assert guard.operation(a,c)==guard.operation(b,c)
    assert session.stats['guard_standard_definition_hits']>0
    altered=RZGate(.17);d=QuantumCircuit(1);d.x(0);altered.definition=d
    assert guard.operation(altered,c)!=guard.operation(a,c)
    # A new call never reuses the prior call's definition guard.
    fresh=f._Guard(session.stats);before=session.stats['guard_standard_definition_hits']
    fresh.operation(RZGate(.17),c)
    assert session.stats['guard_standard_definition_hits']==before


def test_overridden_lazy_standard_method_is_walked(monkeypatch):
    session=f.FingerprintReuse();guard=f._Guard(session.stats);c=QuantumCircuit(1)
    before=guard.operation(RZGate(.17),c)
    def alternate(op):
        d=QuantumCircuit(1);d.x(0);op.definition=d
    monkeypatch.setattr(RZGate,'_define',alternate)
    assert guard.operation(RZGate(.17),c)!=before
    assert f._Guard(session.stats).operation(RZGate(.17),c)!=before
