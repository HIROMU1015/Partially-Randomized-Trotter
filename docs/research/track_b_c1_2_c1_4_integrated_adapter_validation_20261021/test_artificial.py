"""C1-2 only: small artificial tests; never opens H4 or R1 model/basis."""
from fractions import Fraction as F
from itertools import product
from math import factorial
from pathlib import Path
import json
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from support import HERE, isolate, save
isolate()
from adapter import DFAdapter, AllTrials
from reference import q_reference, event_reference, compare, TOLERANCE
from trotterlib.df_rte_tail import DFBasisRegistry
from trottertracks.algorithm_codesign.g6_f3_generator import prepare
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, F3Failure
from trottertracks.algorithm_codesign.return_aggregation import ReturnKernel

POLICY = dict(root_bits=256, probability_bits=160, eta=F(1, 10**12), rho=F(1, 10**12),
              angle_tau=F(1, 10**12), angle_root_bits=256, angle_terms=64)

class Bits:
    def __init__(self, values): self.values = iter(values)
    def bits(self, n): return next(self.values)

def main():
    n, p, x = 2, (F(1, 5), F(3, 10), F(1, 2)), F(1, 2)
    registry = DFBasisRegistry()
    sequences = [((RYGate(.37), (0,)), (RZGate(.29), (0,))),
                 ((XXPlusYYGate(.61, .23), (0, 1)), (RZGate(-.43), (1,))),
                 ((RYGate(-.81), (1,)), (XXPlusYYGate(.42, -.31), (0, 1)))]
    rows, bases = [], {}
    for i, ops in enumerate(sequences):
        definition = registry.register(ops, num_system_qubits=n)
        circuit = QuantumCircuit(n)
        for gate, qubits in ops: circuit.append(gate, qubits)
        bases[definition.basis_id] = Operator(circuit).data
        rows.append(dict(label=i, component_id='artificial:' + str(i), sign=(-1 if i == 1 else 1),
                support=([0, 1] if i == 2 else [i % 2]), basis_id=definition.basis_id, basis_hash=definition.basis_hash))
    adapter = DFAdapter(rows, p, registry, n, 'artificial-v1')
    q = q_reference(rows, bases, n)
    assert all(np.linalg.norm(q(i) @ q(i) - np.eye(4), 2) <= TOLERANCE for i in range(3))
    assert all(np.linalg.norm(q(i) - q(i).conj().T, 2) <= TOLERANCE for i in range(3))
    state = np.array([1, 2j, -3, .7j], dtype=complex)
    state /= np.linalg.norm(state)
    checks = []
    for sigma in (1, -1):
        session = prepare(p, x, 5, sigma=sigma, **POLICY)
        for index, (word, child) in enumerate((((), 0), ((), 1), ((0, 1), 2), ((1, 2), 0), ((0, 1, 2, 0), 1))):
            record = session.event(word, child, trial_id=f'artificial-{sigma}-{index}')
            sidecar = adapter.bind(session, record)
            circuit = adapter.build(sidecar, F(2, 7), F(-3, 11))
            active = event_reference(record.read()['event'], q, n, sigma, float(F(-6, 77)))
            result = compare(circuit, active, state, n)
            assert result['operator_numerical_gate_passed']
            assert result['actual_adjoint_Frobenius_error'] <= TOLERANCE
            assert all(v['absolute_error_diagnostic'] <= TOLERANCE for v in result['XY'].values())
            checks.append({'sigma': sigma, 'word': list(word), 'child': child, **result})
            data = sidecar.read(); data['model_id'] = 'corrupted'
            try: adapter.build(Evidence.create(data))
            except ValueError: pass
            else: raise AssertionError('altered model accepted')
    # All artificial parents only, with independently constructed finite polynomial.
    kernel, mean, digital_mean = ReturnKernel(p, x, 5), np.zeros((4, 4), complex), np.zeros((4, 4), complex)
    session = prepare(p, x, 5, **POLICY)
    count = 0
    for length in (0, 2, 4):
        for word in product(range(3), repeat=length):
            if any(a == b for a, b in zip(word, word[1:])): continue
            parent = kernel.parent(word)
            d = math.sqrt(float(parent.d2))
            for child, mass in zip(parent.children, parent.masses):
                record = session.event(word, child, trial_id='mean-' + str(count))
                event = record.read()['event']
                assert event['proposal'] * event['weight'] == event['coefficient']
                unitary = event_reference(event, q, n)
                mean += d * float(mass / parent.s) * unitary
                digital_mean += float(event['proposal'] * event['weight']) * unitary
                count += 1
    a = sum(float(pi) * q(i) for i, pi in enumerate(p))
    target = sum((-1j * float(x)) ** j / factorial(j) * np.linalg.matrix_power(a, j) for j in range(6))
    ideal_error = float(np.linalg.norm(mean - target, 2))
    digital_error = float(np.linalg.norm(digital_mean - target, 2))
    assert ideal_error <= TOLERANCE and digital_error <= TOLERANCE
    # Construct deterministic bit-map fixtures, never assume these are IID samples.
    session = prepare(p, x, 5, **POLICY)
    generator = session._generator  # fixture introspection only; no mutation
    width = 1 << POLICY['probability_bits']
    order_two = int(generator.order[0] * width)
    raw_zero = session.sample(Bits([order_two, 0, 0]), trial_id='raw-zero')
    coin_zero = session.sample(Bits([0, width - 1]), trial_id='coin-zero')
    accepted = session.sample(Bits([0, 0, 0]), trial_id='accepted')
    assert raw_zero.read()['reason'] == 'RAW_NONREDUCED'
    assert coin_zero.read()['reason'] == 'COIN_REJECT' and accepted.read()['kind'] == 'EVENT'
    accumulator = AllTrials(session)
    accumulator.add(raw_zero); accumulator.add(coin_zero); accumulator.add(accepted, -1)
    result = accumulator.evidence().read()
    assert result['trials'] == 3 and result['mean'] == -accepted.read()['event']['weight'] / 3
    broken = prepare(p, x, 5, **POLICY)
    try: broken.sample(Bits([-1]), trial_id='technical-failure')
    except F3Failure as exc: failure = exc.evidence
    else: raise AssertionError('technical failure mapped to zero')
    before = len(accumulator.seen)
    try: accumulator.add(failure)
    except Exception: pass
    else: raise AssertionError('technical failure counted as zero')
    assert len(accumulator.seen) == before and accumulator.failure
    output = {'status': 'PASS', 'scope': 'C1-2 artificial inputs only; primary-agent checks',
              'event_comparisons': checks, 'first_moment_artificial_events': count,
              'ideal_finite_Taylor_spectral_error': ideal_error,
              'digital_first_moment_spectral_error': digital_error,
              'digital_mean_is_approximate': True, 'all_trials': result | {'mean': str(result['mean']), 'weighted_sum': str(result['weighted_sum'])},
              'technical_failure_terminal_not_zero': True,
              'no_H4_model_or_basis_opened': True, 'no_native_compile_or_shots': True}
    save(HERE / 'artificial_checks.json', output)
    print(json.dumps({'status': 'PASS', 'event_comparisons': len(checks), 'mean_events': count,
                      'ideal_mean_error': ideal_error, 'digital_mean_error': digital_error}), flush=True)

if __name__ == '__main__': main()
