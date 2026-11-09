"""Fixed-cache input interface and compiler/reset consumer for V6-Q.

No canonical event producer is installed here. A supplier declaration is not
proof of its sampling law. Real cache loading, supplier calls, circuit builds
and compilation require their separately authorized scientific scope.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from . import pr_reference_v6q_schedule as caller

CACHE_SHA256 = '68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2'
CANONICAL_LAW = 'canonical_floor_cdf_fresh_independent_bits'


@dataclass(frozen=True)
class FixedInputs:
    cache_path: Path
    cache_sha256: str
    common_data: object
    prep: object


@dataclass(frozen=True)
class DeclaredCanonicalEvents:
    """Scientific provenance supplied at the interface; no qualification flag."""
    metadata: caller.SavedRound
    prep: object
    cfg: object
    dist: object
    events: object
    law: str
    producer_source: str


def _runtime():
    from qiskit import QuantumCircuit
    from .track_c_compiler_a import compile_reference
    from .track_c_h4 import common
    return SimpleNamespace(circuit=QuantumCircuit, compile_reference=compile_reference, common=common)


def load_fixed_inputs(cache_path):
    """Load the existing fixed pickle once in a fresh scientific process.

    Reuse the returned context for subsequent requests. Never clear another
    caller's loader cache or relabel an already-loaded preparation as this one.
    """
    path = Path(cache_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CACHE_SHA256:
        raise ValueError('The cache bytes do not match the fixed science input')
    runtime = _runtime()
    if runtime.common.cache_info().currsize:
        raise ValueError('Load the fixed context before another preparation enters common()')
    key = 'TRACK_C_H4_COMMON_CACHE'
    previous = os.environ.get(key)
    try:
        os.environ[key] = str(path)
        data = runtime.common()  # Existing read-only loader; no prepare/solve.
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    prep = data['preps'][3]
    if prep.ld != 3 or prep.num_system_qubits != 8 or prep.product_formula != '2nd':
        raise ValueError('The loaded preparation does not match fixed H4/Partial S2')
    return FixedInputs(path, digest, data, prep)


def _validate_batch(inputs, batch, metadata):
    if not isinstance(batch, DeclaredCanonicalEvents) or batch.metadata != metadata:
        raise ValueError('Supplier metadata does not match the saved schedule')
    if batch.prep is not inputs.prep:
        raise ValueError('Supplier did not use the bound fixed preparation')
    if batch.law != CANONICAL_LAW or not isinstance(batch.producer_source, str) or not batch.producer_source.strip():
        raise ValueError('An explicit canonical producer SOURCE is required; cohort samplers are not substitutes')
    tail = inputs.prep.rte_preparation.symbolic_tail
    if (batch.cfg.tail_id != tail.tail_id or batch.cfg.tail_hash != tail.tail_hash
            or batch.cfg.lambda_r != inputs.prep.exact_rte_lambda_r):
        raise ValueError('Supplier tail/lambda does not match the fixed preparation')
    # The preserved caller checks r/h/K/dist consistency and q*r length before
    # assembly. Its builders retain event basis ID/hash and ordering checks.


def _compile_and_reset(axis_circuit, runtime):
    native = runtime.compile_reference(axis_circuit)
    if native.num_qubits != 17 or native.num_clbits != 1:
        raise ValueError('Reference native axes must retain 17 qubits and 1 clbit')
    # Exact old transfer.sample convention: insert reset AFTER compilation,
    # but BEFORE the compiled interrogation in physical execution order.
    full = runtime.circuit(17, 1)
    full.reset(range(8, 16))
    full.compose(native, inplace=True)
    return full


def consume_saved_round(inputs, m, *, q, r, bits, canonical_source=None, mode='legacy'):
    """Connect a declared supplier to the old caller/compiler/reset pipeline.

    canonical_source(metadata, prep) must return DeclaredCanonicalEvents. There
    is deliberately no default producer, NumPy cohort fallback or RNG policy.
    This interface never certifies a supplier merely from its law/source tags.
    """
    metadata, _ = caller.bind_round(m, q=q, r=r, bits=bits, mode=mode)
    if canonical_source is None:
        raise NotImplementedError('A qualified fixed-law event producer is not installed')
    if not callable(canonical_source):
        raise TypeError('canonical_source must be an explicit supplier')
    if (not isinstance(inputs, FixedInputs) or inputs.cache_sha256 != CACHE_SHA256
            or inputs.common_data['preps'][3] is not inputs.prep):
        raise ValueError('A bound fixed-cache context is required')
    runtime = _runtime()
    if runtime.common() is not inputs.common_data:
        raise ValueError('The assembly common loader no longer refers to the fixed context')
    batch = canonical_source(metadata, inputs.prep)
    _validate_batch(inputs, batch, metadata)
    result = caller.reference_axes_for_saved_round(m, q=q, r=r, bits=bits,
        cfg=batch.cfg, dist=batch.dist, events=batch.events, mode=mode)
    result['native_axes'] = {axis:_compile_and_reset(result['axes'][axis], runtime) for axis in ('X', 'Y')}
    result['input_provenance'] = dict(cache_sha256=inputs.cache_sha256, law=batch.law,
        producer_source=batch.producer_source, producer_qualification='NOT_ESTABLISHED_BY_THIS_INTERFACE')
    return result
