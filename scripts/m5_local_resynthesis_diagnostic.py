"""Synthetic-only separation of product, Euler, scalar and angle errors.

No molecular loader, QPY, NPZ, grant or resource-accounting entry point exists.
The source mirror is a diagnostic, never a replacement compiler policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import mpmath
import numpy as np
from qiskit._accelerate import euler_one_qubit_decomposer as engine

POLICY = 'M5_LOCAL_RESYNTHESIS_ARTIFICIAL_V1'
MODES = {'default': (True, None), 'zero_atol': (True, 0.0), 'no_simplify': (False, None)}
GATE_NAMES = {name: name for name in ('rz', 'sx', 'x')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def context(dps=80):
    c = mpmath.mp.clone()
    c.dps = dps
    return c


def scalar(c, value):
    return c.mpf(float(value))  # Exact value of the supplied binary64 input.


def matrix(c, a):
    return c.matrix([[c.mpc(scalar(c, complex(x).real), scalar(c, complex(x).imag)) for x in row] for row in a])


def norm(c, a):
    return float(c.sqrt(sum(abs(x)**2 for x in a)))


def rz(c, angle):
    z = c.exp(c.j*angle/2)
    return c.matrix([[1/z, 0], [0, z]])


def ry(c, angle):
    co, si = c.cos(angle/2), c.sin(angle/2)
    return c.matrix([[co, -si], [si, co]])


def gate(c, name, params):
    if name == 'rz':
        return rz(c, scalar(c, params[0]))
    if name == 'sx':
        return c.matrix([[1+c.j, 1-c.j], [1-c.j, 1+c.j]])/2
    if name == 'x':
        return c.matrix([[0, 1], [1, 0]])
    raise ValueError('UNREGISTERED_SYNTHETIC_GATE:'+name)


def sequence(sequence):
    gates = [(GATE_NAMES[g.name], list(p)) for g, p in sequence]
    return gates, float(sequence.global_phase)


def action(c, gates, phase=0.0):
    out = c.eye(2)
    for name, p in gates:
        out = gate(c, name, p)*out
    return c.exp(c.j*scalar(c, phase))*out


def generate(angles, mode):
    simplify, atol = MODES[mode]
    return sequence(engine.generate_circuit(engine.EulerBasis('ZSXX'), *angles, simplify, atol))


def wrap(angle, atol):
    result = (angle+math.pi) % (2*math.pi)-math.pi
    return -math.pi if abs(result-math.pi) < atol else result


def trace_zsxx(angles, mode):
    """Trace fixed Rust branches; exact SUM ONLY keeps all f64 angle work.

    Must match the installed generate_circuit bit for bit before use. The
    exact-sum counterfactual retains every original signed phase increment.
    """
    theta, phi, lam, initial = angles
    simplify, atol = MODES[mode]
    atol = 1e-12 if atol is None else atol
    if not simplify:
        atol = -1.0
    phase, exact, gates, updates, branches, wrapped = initial, Fraction.from_float(initial), [], [], [], []

    def add(value):
        nonlocal phase, exact
        phase += value
        exact += Fraction.from_float(value)
        updates.append(value.hex())

    def pfun(value):
        out = wrap(value, atol)
        emit = abs(out) > atol
        wrapped.append(dict(input_hex=value.hex(), output_hex=out.hex(), emitted=emit))
        if emit:
            gates.append(('rz', [out]))
            add(out/2)

    if abs(theta) < atol:
        branches.append('ZERO_SX')
        pfun(lam+phi)
    elif abs(theta-math.pi/2) < atol:
        branches.append('SINGLE_SX')
        pfun(lam-math.pi/2)
        gates.append(('sx', []))
        pfun(phi+math.pi/2)
    else:
        branches.append('GENERAL')
        if abs(theta-math.pi) < atol:
            branches.append('THETA_PI')
            add(lam)
            phi -= lam
            lam = 0.0
        if abs(wrap(lam+math.pi, atol)) < atol or abs(wrap(phi, atol)) < atol:
            branches.append('AXIS_FLIP')
            lam += math.pi
            theta = -theta
            phi += math.pi
            add(-theta)
        theta += math.pi
        phi += math.pi
        add(-math.pi/2)
        pfun(lam)
        if abs(wrap(theta, atol)) < atol:
            branches.append('USE_X')
            gates.append(('x', []))
        else:
            gates.append(('sx', []))
            pfun(theta)
            gates.append(('sx', []))
        pfun(phi)
    return dict(gates=gates, phase=phase, exact=exact, branches=branches,
                phase_updates_hex=updates, wrapped=wrapped)


def target_u1x(c, angles):
    theta, phi, lam, phase = [scalar(c, a) for a in angles]
    return c.exp(c.j*(phase+(theta+phi+lam)/2))*rz(c, phi)*ry(c, theta)*rz(c, lam)


def direct_cases():
    # Fixed dyadic probes straddle 1e-12; chosen without molecular artifacts.
    cases = [('generic', (.6, .37, -.29, -.21)), ('zero', (0., 0., 0., 0.))]
    for eps in (2.**-55, 2.**-42, 2.**-40, 2.**-39):
        for sign in (-1, 1):
            e = sign*eps
            cases.append((f'rz_{e.hex()}', (0., 0., e, -e/2)))
            cases.append((f'phi_{e.hex()}', (.6, e, -.29, -.21)))
        cases.append((f'theta_{eps.hex()}', (eps, .37, -.29, -.21)))
        for sign in (-1, 1):
            cases.append((f'single_sx_{sign}_{eps.hex()}', (math.pi/2+sign*eps, .37, -.29, -.21)))
        cases.append((f'near_pi_{eps.hex()}', (math.pi-eps, .37, -.29, -.21)))
        cases.append((f'wrap_endpoint_{eps.hex()}', (.6, math.pi-eps, -.29, -.21)))
        cases.append((f'axis_flip_{eps.hex()}', (.6, .37, -math.pi+eps, -.21)))
    cases.extend([('theta_pi', (math.pi, .37, -.29, -.21)),
                  ('single_sx', (math.pi/2, .37, -.29, -.21)),
                  ('axis_flip_zero', (.6, 0., -.29, -.21))])
    return cases


def direct_probe():
    rows = []
    for label, angles in direct_cases():
        for mode in MODES:
            gates, phase = generate(angles, mode)
            trace = trace_zsxx(angles, mode)
            require(gates == trace['gates'] and phase.hex() == trace['phase'].hex(), 'SOURCE_MIRROR_BIT_IDENTITY:'+label+':'+mode)
            c = context()
            target = target_u1x(c, angles)
            actual = action(c, gates, phase)
            gate_only = action(c, gates)
            rational = trace['exact']
            exact_phase = c.mpf(rational.numerator)/rational.denominator
            corrected_sum = c.exp(c.j*exact_phase)*gate_only
            rows.append(dict(case=label, mode=mode, angles_hex=[a.hex() for a in angles],
                gates=[[n, [a.hex() for a in p]] for n,p in gates],
                phase_hex=phase.hex(), raw_frobenius=norm(c, actual-target),
                exact_scalar_sum_only_raw_frobenius=norm(c, corrected_sum-target),
                scalar_addition_roundoff=float(scalar(c, phase)-exact_phase),
                sum_only_operator_change=norm(c, corrected_sum-actual),
                source_mirror_bit_identity=True, branches=trace['branches'],
                phase_updates_hex=trace['phase_updates_hex'], wrapped=trace['wrapped']))
    return rows


def zyz_action(c, angles):
    theta, phi, lam, phase = angles
    return c.exp(c.j*phase)*rz(c, phi)*ry(c, theta)*rz(c, lam)


def precise_zyz(c, unitary):
    det = unitary[0,0]*unitary[1,1]-unitary[0,1]*unitary[1,0]
    da = c.arg(det)
    theta = 2*c.atan2(abs(unitary[1,0]), abs(unitary[0,0]))
    a1, a2 = c.arg(unitary[1,1]), c.arg(unitary[1,0])
    return theta, a1+a2-da, a1-a2, da/2


def euler_probe():
    rows = []
    for label, angles in direct_cases():
        c = context()
        target = target_u1x(c, angles)
        unitary = np.array(target.tolist(), dtype=complex)
        f64 = matrix(c, unitary)
        extracted = list(engine.params_zyz(unitary))
        generated = list(engine.params_u1x(unitary))
        precise = zyz_action(c, precise_zyz(c, f64))
        double = zyz_action(c, [scalar(c, x) for x in extracted])
        exact_conversion = Fraction.from_float(extracted[3])-sum((Fraction.from_float(x) for x in extracted[:3]), Fraction())/2
        conversion_loss = Fraction.from_float(generated[3])-exact_conversion
        row = dict(case=label, input_rounding_raw=norm(c, f64-target),
            f64_Euler_vs_same_input_raw=norm(c, double-f64),
            high_precision_Euler_vs_same_input_raw=norm(c, precise-f64),
            f64_vs_high_precision_Euler_raw=norm(c, double-precise),
            u1x_scalar_conversion_roundoff=float(conversion_loss),
            extracted_zyz_hex=[x.hex() for x in extracted],
            generated_u1x_hex=[x.hex() for x in generated])
        row['modes'] = {}
        for mode in MODES:
            gates, phase = generate(generated, mode)
            output = action(c, gates, phase)
            row['modes'][mode] = dict(raw_vs_same_input=norm(c, output-f64),
                raw_vs_float_Euler=norm(c, output-double), gates=len(gates))
        rows.append(row)
    return rows


def native_pass(gates):
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    from m5_phase_batch_compiler import PhaseBatchOptimize1qGatesDecomposition
    circuit = QuantumCircuit(1)
    for name, params in gates:
        getattr(circuit, name)(*params, 0)
    records = []
    old = Optimize1qGatesDecomposition(basis=['rz', 'sx', 'x', 'cx'])
    out = dag_to_circuit(PhaseBatchOptimize1qGatesDecomposition(old, records.append).run(circuit_to_dag(circuit)))
    return [(i.operation.name, [float(p) for p in i.operation.params]) for i in out.data], float(out.global_phase), records


def product_probe():
    from qiskit.circuit.library import RZGate, SXGate, XGate
    patterns = {
        'generic': [('rz', [.21]), ('sx', []), ('rz', [-.37]), ('sx', []), ('rz', [.17]), ('x', [])],
        'cancellation': [('rz', [.21]), ('rz', [-.21]), ('sx', []), ('sx', []), ('x', [])],
        'small_turn': [('rz', [2.**-40]), ('sx', []), ('sx', []), ('x', [])]}
    constructors = {'rz': RZGate, 'sx': SXGate, 'x': XGate}
    rows = []
    for name, pattern in patterns.items():
        for count in (16, 256, 4096):
            gates = [pattern[i % len(pattern)] for i in range(count)]
            # Each product uses the same gate order. Matrix entries vs angle
            # reference and product arithmetic are kept as separate quantities.
            c = context()
            floating = np.eye(2, dtype=complex)
            exact_entries, exact_angles = c.eye(2), c.eye(2)
            cache = {}
            for gate_name, params in gates:
                key = (gate_name, tuple(params))
                if key not in cache:
                    arr = np.asarray(constructors[gate_name](*params).to_matrix())
                    cache[key] = (arr, matrix(c, arr), gate(c, gate_name, params))
                arr, mp_arr, analytic = cache[key]
                # Explicit two-term f64 product mirrors the Rust source form;
                # no claim of binary equivalence to the compiled Rust library.
                floating = np.array([[arr[r,0]*floating[0,k]+arr[r,1]*floating[1,k] for k in range(2)] for r in range(2)])
                exact_entries = mp_arr*exact_entries
                exact_angles = analytic*exact_angles
            output_gates, phase, receipts = native_pass(gates)
            out = action(c, output_gates, phase)
            d = context(120)
            block_entries, block_angles = d.eye(2), d.eye(2)
            for gate_name, params in pattern:
                block_entries = matrix(d, cache[(gate_name,tuple(params))][0])*block_entries
                block_angles = gate(d, gate_name, params)*block_angles
            quotient, remainder = divmod(count, len(pattern))
            check_entries, check_angles = block_entries**quotient, block_angles**quotient
            for gate_name, params in pattern[:remainder]:
                check_entries = matrix(d, cache[(gate_name,tuple(params))][0])*check_entries
                check_angles = gate(d, gate_name, params)*check_angles
            def precision_error(a, b):
                return float(max(abs(d.mpc(str(x.real),str(x.imag))-y) for x,y in zip(a,b)))
            stability = max(precision_error(exact_entries,check_entries), precision_error(exact_angles,check_angles))
            require(stability < 1e-72, 'PRODUCT_REFERENCE_PRECISION_STABILITY')
            rows.append(dict(case=name, gate_count=count,
                gate_matrix_rounding_raw=norm(c, exact_entries-exact_angles),
                explicit_f64_product_vs_exact_entries_raw=norm(c, matrix(c, floating)-exact_entries),
                exact_batch_actual_pass_vs_exact_angles_raw=norm(c, out-exact_angles),
                exact_batch_actual_pass_vs_exact_entries_raw=norm(c, out-exact_entries),
                output_gates=len(output_gates), pass_executions=len(receipts),
                phase_increment_count=sum(x['phase_increment_count'] for x in receipts),
                reference_80_120_max_element_difference=stability,
                Rust_product_bit_identity_claimed=False))
    return rows


def coherent_probe():
    # Separate local runs with barriers, preventing fusion before simplification.
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    from m5_phase_batch_compiler import PhaseBatchOptimize1qGatesDecomposition
    c = context()
    rows = []
    for count in (1, 16, 256, 4096):
        angle = 2.**-40
        qc = QuantumCircuit(1)
        for _ in range(count):
            qc.rz(angle, 0)
            qc.barrier(0)
        receipts = []
        old = Optimize1qGatesDecomposition(basis=['rz', 'sx', 'x', 'cx'])
        out = dag_to_circuit(PhaseBatchOptimize1qGatesDecomposition(old, receipts.append).run(circuit_to_dag(qc)))
        gates = [(i.operation.name, [float(v) for v in i.operation.params]) for i in out.data if i.operation.name != 'barrier']
        target = rz(c, count*scalar(c, angle))
        generated = action(c, gates, float(out.global_phase))
        # Analytic unsimplified control, not a modified production compiler.
        rows.append(dict(independent_runs=count, angle_hex=angle.hex(),
            raw_frobenius=norm(c, generated-target),
            raw_basis0=norm(c, (generated-target)[:,0]), raw_basis1=norm(c, (generated-target)[:,1]),
            output_nonbarrier_gates=len(gates),
            phase_increment_count=sum(r['phase_increment_count'] for r in receipts),
            exact_outer_phase_batch_used=True, H6_causal_proof=False))
    return rows


def reference_checks():
    c, d = context(80), context(120)
    errors = []
    for label, angles in direct_cases():
        a, b = target_u1x(c, angles), target_u1x(d, angles)
        err = max(abs(d.mpc(str(x.real), str(x.imag))-y) for x,y in zip(a,b))
        require(err < d.mpf('1e-75'), 'REFERENCE_PRECISION_STABILITY:'+label)
        errors.append(float(err))
    return dict(precision_digits=[80,120], checked_cases=len(errors), max_element_difference=max(errors),
                status='INTERNAL_PRECISION_CHECK_NOT_RIGOROUS_CERTIFICATE')


def worker(kind):
    started = time.monotonic()
    functions = {'generation': direct_probe, 'Euler': euler_probe, 'products': product_probe, 'coherent': coherent_probe}
    return dict(section=kind, rows=functions[kind](), worker_pid=os.getpid(), wall_seconds=time.monotonic()-started)


def run_artificial():
    require(__import__('qiskit').__version__ == '1.3.0', 'PINNED_QISKIT_1_3_0')
    started = time.monotonic()
    # Four independent diagnostic units; processes avoid shared MP contexts.
    sections = ['generation', 'Euler', 'products', 'coherent']
    with ProcessPoolExecutor(max_workers=len(sections), mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(worker, sections))
    return dict(policy=POLICY, status='ARTIFICIAL_DIAGNOSTICS_COMPLETE_NOT_H6_PASS',
        sections={r.pop('section'): r for r in results}, reference_checks=reference_checks(),
        wall_seconds=time.monotonic()-started, productive_work_units=len(sections),
        workers_joined=True, raw_global_phase_retained=True, phase_alignment=False,
        molecular_QPY_NPZ_loads=0, molecular_compile=0, molecular_replay=0,
        sampling=0, accounting=0, accepted_cost_rows=0, production_policy_changed=False,
        H6_causal_proof=False, H6_threshold_changed=False, mandatory_STOP=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-artificial', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.run_artificial:
        print(json.dumps(dict(policy=POLICY, status='ARTIFICIAL_SCOPE_ONLY', molecular_compile=0)))
        return
    require(args.output is not None and not args.output.exists(), 'NEW_EXCLUSIVE_OUTPUT_REQUIRED')
    result = run_artificial()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], sections=len(result['sections']), wall_seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
