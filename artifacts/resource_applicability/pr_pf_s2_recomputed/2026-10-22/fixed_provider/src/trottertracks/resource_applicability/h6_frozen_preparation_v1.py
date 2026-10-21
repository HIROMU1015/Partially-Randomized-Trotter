"""Portable, exact DF preparation archive; no pickle, resampling or compiler change.

Imports of numerical libraries are deferred until an authorized diagnostic or
worker calls the codec. This module is also delivered beside the old-host
exporter, which uses the unchanged scientific source in that host's repository.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'h6_frozen_df_preparations_v1'
SCIENCE_SOURCE = '5a903aea2492c2d211c1e15e7737df0d50694c12'
OLD_MANIFEST = 'artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/sealed_manifest_v3.json'
OLD_RUN = 'artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1'
MAX_BYTES = 16 * 2**20
MAX_NODES = 50000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def types():
    from trotterlib import df_partial_s2 as partial, df_rte_tail as tail
    from trotterlib import df_rte_circuit as circuit, rte
    classes = [partial.DFPartialS2Preparation, partial.DFDeterministicOneBodySpec,
               partial.DFDeterministicFragmentSpec, tail.DFTailExtraction,
               tail.DFDiagonalPauliComponent, tail.DFTailExtractionMetadata,
               tail.SymbolicRTETail, tail.BasisChangeMetadata,
               circuit.DFRTEComponentCircuitSpec, circuit.DFRTEIdentityCircuitSpec,
               circuit.DFRTEEventPreparation, rte.BasisChangeOperation,
               rte.RTEComponent, rte.TailNormalizationMetadata]
    return {cls.__name__: cls for cls in classes}


def encode_preparations(preparations):
    """Content-addressed nodes deduplicate repeated bases and preserve float bits."""
    from trotterlib.df_rte_tail import DFBasisRegistry
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    from qiskit.quantum_info import Operator
    import numpy as np
    allowed = types(); nodes = {}; memo = {}

    def encode(value):
        if type(value) in (str, int, bool) or value is None:
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError('FROZEN_NONFINITE')
            return {'float_hex': float(value).hex()}
        identity = id(value)
        if identity in memo:
            return {'ref': memo[identity]}
        if isinstance(value, DFBasisRegistry):
            record = dict(type='DFBasisRegistry', definitions=[
                dict(basis_id=m.basis_id, operations=encode(value.operations(m.basis_id)),
                     num_system_qubits=m.num_system_qubits)
                for m in value.metadata()])
        elif isinstance(value, (RZGate, XXPlusYYGate)):
            matrix = np.ascontiguousarray(Operator(value).data, dtype=np.complex128)
            record = dict(type='gate', name=value.name,
                          parameters=[encode(float(x)) for x in value.params],
                          label=value.label, matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest())
        elif type(value) is tuple:
            record = dict(type='tuple', items=[encode(x) for x in value])
        elif is_dataclass(value) and type(value) in allowed.values():
            record = dict(type=type(value).__name__, fields={
                f.name: encode(getattr(value, f.name)) for f in fields(value)})
        else:
            raise ValueError('FROZEN_UNSUPPORTED_TYPE:' + type(value).__name__)
        sha = digest(record); nodes[sha] = record; memo[identity] = sha
        if len(nodes) > MAX_NODES:
            raise ValueError('FROZEN_NODE_CAP')
        return {'ref': sha}

    roots = {key: encode(value) for key, value in sorted(preparations.items())}
    return dict(roots=roots, nodes=nodes)


def decode_preparations(codec):
    """Reject unknown types, cycles, field changes, nonfinite values and gate drift."""
    from trotterlib.df_rte_tail import DFBasisRegistry
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    from qiskit.quantum_info import Operator
    import numpy as np
    if (type(codec) is not dict or set(codec) != {'roots', 'nodes'}
            or type(codec['nodes']) is not dict or not 0 < len(codec['nodes']) <= MAX_NODES
            or type(codec['roots']) is not dict or not 0 < len(codec['roots']) <= 8):
        raise ValueError('FROZEN_CODEC')
    nodes = codec['nodes']; allowed = types(); memo = {}; visiting = set()
    for sha, record in nodes.items():
        if sha != digest(record):
            raise ValueError('FROZEN_NODE_HASH')

    def decode(value, depth=0):
        if depth > 64:
            raise ValueError('FROZEN_DEPTH')
        if type(value) in (str, int, bool) or value is None:
            return value
        if type(value) is not dict:
            raise ValueError('FROZEN_VALUE')
        if set(value) == {'float_hex'}:
            if type(value['float_hex']) is not str:
                raise ValueError('FROZEN_FLOAT')
            f = float.fromhex(value['float_hex'])
            if not math.isfinite(f) or f.hex() != value['float_hex']:
                raise ValueError('FROZEN_FLOAT')
            return f
        if set(value) != {'ref'} or type(value['ref']) is not str or value['ref'] not in nodes:
            raise ValueError('FROZEN_REFERENCE')
        sha = value['ref']
        if sha in visiting:
            raise ValueError('FROZEN_CYCLE')
        if sha in memo:
            return memo[sha]
        visiting.add(sha); record = nodes[sha]; kind = record.get('type')
        child = lambda v: decode(v, depth + 1)
        if kind == 'tuple' and set(record) == {'type', 'items'} and type(record['items']) is list:
            result = tuple(child(x) for x in record['items'])
        elif kind == 'gate' and set(record) == {'type', 'name', 'parameters', 'label', 'matrix_sha256'}:
            factories = {'rz': (RZGate, 1), 'xx_plus_yy': (XXPlusYYGate, 2)}
            if record['name'] not in factories or type(record['parameters']) is not list:
                raise ValueError('FROZEN_GATE')
            factory, count = factories[record['name']]
            parameters = [child(x) for x in record['parameters']]
            if len(parameters) != count or any(type(x) is not float for x in parameters):
                raise ValueError('FROZEN_GATE_PARAMETERS')
            result = factory(*parameters, label=record['label'])
            matrix = np.ascontiguousarray(Operator(result).data, dtype=np.complex128)
            if hashlib.sha256(matrix.tobytes()).hexdigest() != record['matrix_sha256']:
                raise ValueError('FROZEN_LOCAL_GATE_MATRIX_CHANGED')
        elif kind == 'DFBasisRegistry' and set(record) == {'type', 'definitions'} and type(record['definitions']) is list:
            result = DFBasisRegistry(); seen = set()
            for definition in record['definitions']:
                if (type(definition) is not dict
                        or set(definition) != {'basis_id', 'operations', 'num_system_qubits'}
                        or definition['basis_id'] in seen):
                    raise ValueError('FROZEN_REGISTRY')
                seen.add(definition['basis_id'])
                result.register(child(definition['operations']),
                                basis_id=definition['basis_id'],
                                num_system_qubits=definition['num_system_qubits'])
        elif (kind in allowed and set(record) == {'type', 'fields'}
              and type(record['fields']) is dict
              and set(record['fields']) == {f.name for f in fields(allowed[kind])}):
            result = allowed[kind](**{key: child(v) for key, v in record['fields'].items()})
        else:
            raise ValueError('FROZEN_RECORD_TYPE_OR_FIELDS')
        visiting.remove(sha); memo[sha] = result
        return result

    preparations = {key: decode(value) for key, value in codec['roots'].items()}
    # Require canonical lossless encoding, including signed zero and registries.
    if encode_preparations(preparations) != codec:
        raise ValueError('FROZEN_NONCANONICAL_ROUNDTRIP')
    return preparations


def key(discard, prefix):
    if type(discard) is not bool or type(prefix) is not int or prefix < 0:
        raise ValueError('FROZEN_PREPARATION_KEY')
    return ('discard' if discard else 'faithful') + ':p' + str(prefix)


def expected_preparations(root):
    rows = json.loads((Path(root) / OLD_RUN / 'signal/prepared_representation.json').read_bytes())['preparations']
    result = {}
    for row in rows:
        parts = row['cell_id'].split('_'); name = key(parts[0] == 'B0', int(parts[1][1:]))
        expected = {k: row[k] for k in ('preparation_hash', 'partition_hash', 'lambda_r', 'extracted_identity', 'constant')}
        if name in result and result[name] != expected:
            raise ValueError('FROZEN_OLD_PREPARATION_CONFLICT')
        result[name] = expected
    return result


def verify_preparations(preparations, expected, hamiltonian_hash):
    from trotterlib.df_partial_s2 import DFPartialS2Preparation, _block_hash_payload
    from trotterlib.df_rte_tail import DFBasisRegistry, extraction_to_symbolic_rte_tail
    if set(preparations) != set(expected):
        raise ValueError('FROZEN_PREPARATION_SET')
    for name, p in preparations.items():
        policy, prefix_text = name.split(':p'); prefix = int(prefix_text)
        # B0 is deliberately prepared from select_blocks(prefix), so its hash
        # is a TRUNCATED Hamiltonian hash, bound by the old preparation digest.
        if (not isinstance(p, DFPartialS2Preparation) or policy not in ('faithful', 'discard')
                or (policy == 'faithful' and p.hamiltonian_hash != hamiltonian_hash)
                or p.ld != prefix or p.deterministic_fragment_indices != tuple(range(prefix))
                or (policy == 'discard' and p.randomized_block_indices)):
            raise ValueError('FROZEN_HAMILTONIAN_BINDING')
        payload = dict(hamiltonian_hash=p.hamiltonian_hash, partition_hash=p.partition_hash,
                       deterministic_blocks=[_block_hash_payload(b) for b in p.deterministic_blocks],
                       tail_hash=p.tail_extraction.tail_hash, identity_policy=p.identity_policy,
                       coefficient_atol=p.coefficient_atol,
                       basis_construction_policy=p.basis_construction_policy,
                       diagonal_sort=p.diagonal_sort, product_formula='2nd',
                       preparation_hash_policy='df_partial_s2_preparation_v1')
        if digest(payload) != p.preparation_hash:
            raise ValueError('FROZEN_PREPARATION_HASH')
        actual = dict(preparation_hash=p.preparation_hash, partition_hash=p.partition_hash,
                      lambda_r=p.exact_rte_lambda_r, extracted_identity=p.extracted_identity_coefficient,
                      constant=p.constant_coefficient)
        if actual != expected[name]:
            raise ValueError('FROZEN_OLD_PREPARATION_MISMATCH:' + name)
        if (p.identity_policy != 'extract_identity_phase' or p.coefficient_atol != 0.
                or p.threshold_dropped_component_count != 0
                or p.diagonal_sort != 'descending_abs'
                or p.basis_construction_policy != 'fermionic_gaussian_jw'):
            raise ValueError('FROZEN_SCIENTIFIC_POLICY')
        if p.rte_preparation.symbolic_tail != extraction_to_symbolic_rte_tail(p.tail_extraction):
            raise ValueError('FROZEN_TAIL_COMPONENTS')
        for b in p.deterministic_blocks:
            registry = DFBasisRegistry()
            metadata = registry.register(b.runtime_basis_operations,
                                         num_system_qubits=p.num_system_qubits, basis_id=b.basis_id).metadata
            if metadata.basis_hash != b.basis_hash or metadata.operations != b.basis_change_operations:
                raise ValueError('FROZEN_DETERMINISTIC_BASIS')


def read_archive(path, *, sha256):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES
            or file_hash(path) != sha256):
        raise ValueError('FROZEN_ARCHIVE_BYTES')
    def pairs(rows):
        result = {}
        for k, v in rows:
            if k in result:
                raise ValueError('FROZEN_DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    archive = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('FROZEN_NONFINITE_JSON')))
    if (type(archive) is not dict or set(archive) != {'schema', 'binding', 'codec', 'codec_digest', 'export_receipt'}
            or archive['schema'] != SCHEMA or archive['codec_digest'] != digest(archive['codec'])):
        raise ValueError('FROZEN_ARCHIVE_SCHEMA')
    return archive


def verify_archive(root, path, *, sha256):
    """Verify old-source/input/representation receipt before any cost dispatch."""
    root = Path(root); archive = read_archive(path, sha256=sha256)
    from trottertracks.resource_applicability.h6_matched_contract_v1 import environment
    old = json.loads((root / OLD_MANIFEST).read_bytes())
    expected = dict(science_source=SCIENCE_SOURCE, science_source_hashes=old['source_hashes'],
                    input_identity=old['input_identity'], compiler=old['plan']['compiler_proposed'],
                    old_prepared_sha256=file_hash(root / OLD_RUN / 'signal/prepared_representation.json'),
                    old_partial_450_sha256=file_hash(root / OLD_RUN / 'cost_exploration_0450/trajectory.json'),
                    environment=old['environment'])
    if archive['binding'] != expected or environment() != old['environment']:
        raise ValueError('FROZEN_ARCHIVE_BINDING')
    preparations = decode_preparations(archive['codec'])
    verify_preparations(preparations, expected_preparations(root), old['input_identity']['hamiltonian_hash'])
    return preparations, archive


def verify_partial_450(root, preparations):
    """One bounded trajectory-regeneration diagnostic; no build or transpile."""
    from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import MolecularPort
    from trottertracks.resource_applicability.ax2b_limits import CallBudget
    root = Path(root); old = json.loads((root / OLD_RUN / 'cost_exploration_0450/trajectory.json').read_bytes())
    task = old['task']; cell = task['cell']
    # The frozen sampling method needs only these fields. Never call setup().
    port = object.__new__(MolecularPort)
    port.preps = {cell['id']: preparations[key(False, cell['prefix'])]}
    port.plan = {'T': .8}; port.ham = type('Register', (), {'n_qubits': 12})()
    port.calls = CallBudget(trajectory=1, occurrence=8)
    steps = MolecularPort.sampled_steps(port, cell, task['seed'])
    events = [[e.to_dict() for e in s.rte_occurrence.events] for s in steps]
    actual = digest(events)
    if actual != old['event_digest'] or events != old['events']:
        raise ValueError('FROZEN_PARTIAL_450_REPRODUCTION_MISMATCH')
    return dict(index=450, task=task, event_digest=actual, calls=port.calls.used,
                compile=0, signal=0, input_generation=0, exact_events_matched=True)


def verify_phase_a_wrappers(preparations, base):
    """Compare existing 447 wrappers with the old frozen preparation, no compile.

    A different basis must not silently mix the server's earlier 447 costs into
    the old cost series. Reuse requires BOTH actual full-wrapper fingerprints.
    """
    from trottertracks.resource_applicability.ax2b_native_df_v5 import (
        build_deterministic_native, build_native_hadamard_wrapper)
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
    p = preparations[key(False, 19)]; rows = {}
    run = Path(base['exclusive_output']['absolute_path']) / 'cost_exploration_0447'
    task = base['phase_scope']['task']; cell = task['cell']
    evolution = build_deterministic_native(p.deterministic_blocks, num_system_qubits=p.num_system_qubits,
        T=base['plan']['T'], q=cell['q'], formula=cell['formula'], scalar=p.constant_coefficient,
        control_policy='symmetric_directional', max_instructions=base['plan']['caps_proposed']['untranspiled_instructions'])
    for axis in ('cosine', 'sine'):
        old = json.loads((run / ('wrapper_symmetric_directional_' + axis + '.json')).read_bytes())
        wrapper = build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True,
            max_instructions=base['plan']['caps_proposed']['untranspiled_instructions'])
        actual = canonical_qiskit_circuit_fingerprint(wrapper)
        rows[axis] = dict(expected=old['fingerprint'], actual=actual, matched=actual == old['fingerprint'])
        del wrapper
    return dict(index=447, fingerprints=rows, exact=all(row['matched'] for row in rows.values()),
                compile_calls=0, signal_calls=0, trajectory_calls=0)


@contextmanager
def use_frozen_preparations(preparations):
    """Scope the old port's preparation inputs to this one-thread worker only."""
    from trottertracks.resource_applicability import h6_matched_cost_port_v3 as port
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    original = port._prepare, port._prepare_discard

    def resolve(ham, prefix, *, discard):
        p = preparations[key(discard, prefix)]
        prepared_ham = ham.select_blocks(tuple(range(prefix))) if discard else ham
        if df_hamiltonian_hash(prepared_ham) != p.hamiltonian_hash:
            raise ValueError('FROZEN_WORKER_HAMILTONIAN')
        return p

    port._prepare = lambda ham, prefix: resolve(ham, prefix, discard=False)
    port._prepare_discard = lambda ham, prefix: resolve(ham, prefix, discard=True)
    try:
        yield
    finally:
        port._prepare, port._prepare_discard = original
