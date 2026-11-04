"""Operator-derived global-phase accounting for this fixed H8 compiler.

There is no statevector argument or phase fitting. Every nonbasis gate in the
actual wrapper must pass a small-operator test before its phase is accounted.
This supplement does not change the shared compiler installation.
"""
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing

import h8_long_time_adapter as a

BASIS = frozenset(('rz', 'sx', 'x', 'cx', 'measure', 'barrier'))
NONBASIS = frozenset(('xx_plus_yy', 'rzz', 'crzz', 'crz', 'h', 'p'))
POLICY = 'XX_PLUS_YY_EXTRA_PLUS_I_IN_THIS_FIXED_COMPILER'


def gate_key(op):
    return a.digest(dict(name=op.name, gate_class=op.base_class.__module__ + '.' + op.base_class.__name__,
        qubits=op.num_qubits, controls=getattr(op, 'ctrl_state', None),
        parameters=[float(p).hex() for p in op.params]))


def nonbasis_gates(wrapper):
    gates = {}
    for item in wrapper.data:
        op = item.operation
        a.require(op.name in BASIS | NONBASIS, 'UNCERTIFIED_GATE_FAMILY:' + op.name)
        if op.name in NONBASIS:
            a.require(op.num_qubits <= 3, 'SMALL_OPERATOR_ONLY')
            gates[gate_key(op)] = op
    return gates


def check_gate(item):
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    key, op = item
    circuit = QuantumCircuit(op.num_qubits)
    circuit.append(op, range(op.num_qubits))
    compiled = transpile(circuit, basis_gates=['rz', 'sx', 'x', 'cx'],
                         optimization_level=1, seed_transpiler=17, num_processes=1)
    expected = Operator(op).data
    actual = Operator(compiled).data
    quarters = 1 if op.name == 'xx_plus_yy' else 0
    error = float(np.linalg.norm(actual - (1j ** quarters) * expected))
    return key, dict(name=op.name, extra_phase_quarters=quarters, error=error)


def certify(wrapper, cache, workers=1):
    """Test each new actual gate/parameter tuple, reuse certified local tests."""
    gates = nonbasis_gates(wrapper)
    missing = [(key, op) for key, op in gates.items() if key not in cache]
    if workers > 1 and missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing)),
                mp_context=multiprocessing.get_context('fork')) as pool:
            tested = list(pool.map(check_gate, missing, chunksize=16))
    else:
        tested = [check_gate(item) for item in missing]
    cache.update(tested)
    records = {key: cache[key] for key in gates}
    a.require(all(math.isfinite(v['error']) and 0 <= v['error'] < 1e-10
                  and v['name'] == gates[key].name
                  and v['extra_phase_quarters'] == (1 if v['name'] == 'xx_plus_yy' else 0)
                  for key, v in records.items()), 'LOCAL_OPERATOR_PHASE_MISMATCH')
    return dict(compiler=a.COMPILER, policy=POLICY, records=records,
                maximum_error=max((v['error'] for v in records.values()), default=0.))


def correct_phase(wrapper, compiled, certificate):
    """Copy the full compiled wrapper and add only the certified global phase."""
    gates = nonbasis_gates(wrapper)
    a.require(certificate['compiler'] == a.COMPILER and certificate['policy'] == POLICY,
              'FIXED_COMPILER_PHASE_POLICY')
    records = certificate['records']
    a.require(set(records) == set(gates), 'ACTUAL_GATE_PARAMETERS_CERTIFIED')
    for key, op in gates.items():
        value = records[key]
        a.require(value['name'] == op.name and math.isfinite(value['error'])
                  and 0 <= value['error'] < 1e-10
                  and value['extra_phase_quarters'] == (1 if op.name == 'xx_plus_yy' else 0),
                  'CERTIFIED_GATE_OPERATOR_PHASE')
    a.require(all(i.operation.name in BASIS for i in compiled.data), 'COMPILED_FIXED_BASIS')
    number = sum(i.operation.name == 'xx_plus_yy' for i in wrapper.data)
    quarters = (-number) % 4
    corrected = compiled.copy()
    corrected.global_phase = float(compiled.global_phase) + quarters * math.pi / 2
    a.require(corrected.count_ops() == compiled.count_ops(), 'UNCHANGED_COMPILED_GATE_COUNTS')
    return corrected, dict(policy=POLICY, derived_from='CERTIFIED_SMALL_GATE_OPERATORS',
        xx_plus_yy_count=number, delta_phase_quarters=quarters,
        certified_unique_gates=len(records), maximum_local_operator_error=certificate['maximum_error'],
        certificate_sha256=a.digest(certificate), statevector_phase_alignment=False,
        gate_counts_unchanged=True)
