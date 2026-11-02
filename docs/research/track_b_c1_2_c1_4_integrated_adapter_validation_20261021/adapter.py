"""G6–DF sidecar v1. Direct Z/ZZ conjugation; unchanged G6/F3 sampler.

This is an ideal logical provider, not a native synthesis/cost certificate.
No RTEEvent or RPE evolution result is impersonated.
"""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RZZGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from support import R1, MODEL, canonical, digest, isolate

isolate()
from trotterlib.df_rte_tail import DFBasisRegistry, describe_basis_change_operations, canonical_basis_hash
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.rpe_hadamard_interrogation import QiskitRPEHadamardInterrogationBuilder
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, F3Failure, fingerprint

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def matrix_hash(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.complex128).view(np.uint8).tobytes()).hexdigest()

def restore_r1():
    """Exact metadata and original gate constructors from saved R1, no diagonalization."""
    core = json.loads((R1 / 'result/exact_model_core.json').read_text())
    require(hashlib.sha256(canonical(core)).hexdigest() == MODEL, 'R1_MODEL_MISMATCH')
    rows = []
    for chunk in core['label_chunks_in_order']:
        path = R1 / 'result' / chunk['path']
        require(digest(path) == chunk['sha256'], 'LABEL_BYTES_MISMATCH')
        rows.extend(json.loads(path.read_text()))
    require([r['label'] for r in rows] == list(range(324)), 'LABEL_ORDER_MISMATCH')
    lam = F(core['lambda_exact'])
    p = tuple(F(r['probability_exact']) for r in rows)
    require(all(v > 0 for v in p) and sum(p) == 1, 'EXACT_P_MISMATCH')
    require(all(F(r['signed_coefficient_exact']) == r['sign'] * lam * p[i]
                and F.from_float(float.fromhex(r['coefficient_hex'])) == F(r['signed_coefficient_exact'])
                and r['sign'] in (-1, 1) and not r['identity'] for i, r in enumerate(rows)),
            'SIGNED_COEFFICIENT_MISMATCH')
    registry, saved = DFBasisRegistry(), {}
    index = json.loads((R1 / 'result/basis_registry_index.json').read_text())
    for item in index['basis_definitions']:
        path = R1 / 'result' / item['path']
        require(digest(path) == item['sha256'], 'BASIS_BYTES_MISMATCH')
        doc = json.loads(path.read_text())
        operations = []
        for record in doc['operations_in_time_order']:
            descriptor = record['descriptor']
            params = list(map(float, descriptor['parameters']))
            if descriptor['name'] == 'rz':
                gate = RZGate(*params)
            elif descriptor['name'] == 'xx_plus_yy':
                gate = XXPlusYYGate(*params)
            else:
                raise ValueError('UNSUPPORTED_SAVED_GATE')
            require(type(gate).__module__ + '.' + type(gate).__name__ == record['gate_class'],
                    'SAVED_GATE_CLASS_MISMATCH')
            require(matrix_hash(Operator(gate).data) == descriptor['matrix_sha256'] == record['actual_matrix_sha256'],
                    'RESTORED_LOCAL_MATRIX_MISMATCH')
            operations.append((gate, tuple(descriptor['qubits'])))
        definition = registry.register(operations, num_system_qubits=8, basis_id=doc['basis_id'])
        require(definition.basis_hash == doc['basis_hash'] == item['basis_hash'], 'BASIS_ID_MISMATCH')
        require([asdict(d) for d in definition.metadata.operations] ==
                [{**r['descriptor'], 'qubits': tuple(r['descriptor']['qubits']),
                  'parameters': tuple(r['descriptor']['parameters'])} for r in doc['operations_in_time_order']],
                'BASIS_DESCRIPTOR_MISMATCH')
        saved[definition.basis_id] = doc
    require(len(saved) == 13 and all(registry.definition(r['basis_id']).basis_hash == r['basis_hash'] for r in rows),
            'LABEL_BASIS_MISMATCH')
    return core, rows, p, registry, saved

class DFAdapter:
    def __init__(self, rows, p, registry, n, model_id):
        keys = ('label', 'component_id', 'sign', 'support', 'basis_id', 'basis_hash')
        self.rows = tuple({k: r[k] for k in keys} for r in rows)
        self.p = tuple(p)
        self.registry, self.n, self.model_id = registry, n, model_id
        self.labels_id = fingerprint(self.rows)

    def bind(self, session, record):
        require(session.request.p == self.p and session.request.m == 5, 'G6_MODEL_P_OR_M_MISMATCH')
        data = record.read()
        require(data['kind'] == 'EVENT', 'ACCEPTED_EVENT_REQUIRED')
        angle = session.angle_for_event(record)
        return Evidence.create({'schema': 'g6_df_sidecar_v1', 'model_id': self.model_id,
                'labels_id': self.labels_id, 'event_id': record.digest, 'event_record': data,
                'angle_id': angle.digest, 'angle': angle.read(),
                'provider': 'direct_DF_Z_ZZ_ideal_v1', 'native_certified': False})

    def _basis(self, circuit, label, inverse):
        row = self.rows[label]
        definition = self.registry.definition(row['basis_id'])
        require(definition.basis_hash == row['basis_hash'] ==
                canonical_basis_hash(self.n, describe_basis_change_operations(definition.runtime_operations)),
                'RUNTIME_BASIS_CHANGED')
        QiskitDFRTEEventCircuitBuilder._append_basis(circuit, definition, inverse=inverse)

    def append_q(self, circuit, label, rotation=None):
        """B† -> controlled central signed P/rotation -> B; identity on inactive branch."""
        row, control = self.rows[label], self.n
        support = tuple(row['support'])
        require(row['sign'] in (-1, 1) and len(support) in (1, 2) and len(set(support)) == len(support),
                'INVALID_SIGNED_Z_SUPPORT')
        self._basis(circuit, label, True)
        if rotation is None:
            for qubit in support:
                circuit.cz(control, qubit)
            if row['sign'] == -1:
                circuit.z(control)
        else:
            angle = 2 * row['sign'] * rotation
            gate = RZGate(angle) if len(support) == 1 else RZZGate(angle)
            circuit.append(gate.control(1), [control, *support])
        self._basis(circuit, label, False)

    def build(self, sidecar, scalar_time=F(0), scalar=F(0)):
        data = sidecar.read()
        require(data['schema'] == 'g6_df_sidecar_v1' and data['model_id'] == self.model_id and
                data['labels_id'] == self.labels_id == fingerprint(self.rows), 'SIDECAR_MODEL_MISMATCH')
        record, angle = data['event_record'], data['angle']
        require(fingerprint(record) == data['event_id'] and fingerprint(angle) == data['angle_id'] and
                angle['context']['event_id'] == data['event_id'], 'SIDECAR_EVIDENCE_MISMATCH')
        event, sigma = record['event'], record['sigma']
        word, child = event['word'], event['child']
        require(len(word) % 2 == 0 and len(word) <= 4 and all(i != j for i, j in zip(word, word[1:]))
                and (not word or child != word[0]) and sigma in (-1, 1) and
                event['phase_i_power'] == len(word) % 4 and event['circuit_time_word'] == tuple(reversed(word)),
                'EVENT_PHASE_WORD_MISMATCH')
        require(event['proposal'] > 0 and event['proposal'] * event['weight'] == event['coefficient'],
                'FINITE_PROPOSAL_MISMATCH')
        circuit = QuantumCircuit(self.n + 1, name='g6_df_controlled_v1')
        if len(word) % 4 == 2:
            circuit.z(self.n)  # (-i sigma)^even length = -1 for length 2.
        for label in reversed(word):
            self.append_q(circuit, label)
        self.append_q(circuit, child, rotation=float(angle['signed_representative']))
        require(type(scalar_time) is F and type(scalar) is F, 'EXACT_SCALAR_TIME_REQUIRED')
        if scalar_time * scalar:
            circuit.p(-float(scalar_time * scalar), self.n)
        return circuit

def hadamard(controlled, n, axis):
    require(axis in ('X', 'Y'), 'AXIS_REQUIRED')
    circuit = QiskitRPEHadamardInterrogationBuilder._new_wrapper_circuit(controlled)
    circuit.name = 'g6_df_hadamard_' + axis
    circuit.h(n)
    circuit.compose(controlled, inplace=True)
    if axis == 'Y':
        circuit.sdg(n)
    circuit.h(n)
    return circuit

class AllTrials:
    """Estimator bookkeeping only; acquires no shots and generates no trajectories."""
    def __init__(self, session):
        status = session.status()
        self.input_id = session.request.input_id
        self.context = {k: status[k] for k in ('input_id', 'source_id', 'preflight_id')}
        self.context.update(policy_id=session.request.policy_id, sigma=session.request.sigma)
        self.seen, self.total, self.accepted, self.zeros = set(), F(0), 0, 0
        self.failure = None

    def add(self, record, observation=None):
        require(self.failure is None, 'ACCUMULATOR_STOPPED')
        try:
            data = record.read()
            require(all(data.get(k) == v for k, v in self.context.items()) and
                    data['trial_id'] not in self.seen, 'TRIAL_IDENTITY_MISMATCH')
            require(data.get('selection_scope') is None, 'FORMAL_QUERY_NOT_SAMPLED_TRIAL')
            if data['kind'] == 'ZERO':
                require(data['is_valid_zero_trial'] and data['reason'] in ('RAW_NONREDUCED', 'COIN_REJECT')
                        and observation is None, 'INVALID_ZERO')
                self.zeros += 1
            elif data['kind'] == 'EVENT':
                require(type(observation) is int and observation in (-1, 1), 'OBSERVATION_REQUIRED')
                event = data['event']
                require(event['proposal'] > 0 and event['proposal'] * event['weight'] == event['coefficient'],
                        'WEIGHT_MISMATCH')
                self.total += observation * event['weight']
                self.accepted += 1
            else:
                raise ValueError('TECHNICAL_OR_UNSUPPORTED_RECORD_IS_NOT_ZERO')
            self.seen.add(data['trial_id'])
        except Exception as exc:
            self.failure = type(exc).__name__ + ':' + str(exc)
            raise

    def evidence(self):
        return Evidence.create({'schema': 'g6_df_all_trials_v1', 'input_id': self.input_id,
            'trials': len(self.seen), 'accepted': self.accepted, 'valid_zeros': self.zeros,
            'weighted_sum': self.total, 'mean': self.total / len(self.seen) if self.seen else None,
            'technical_failure': self.failure, 'completion_certified': False,
            'accepted_only_normalization': False})
