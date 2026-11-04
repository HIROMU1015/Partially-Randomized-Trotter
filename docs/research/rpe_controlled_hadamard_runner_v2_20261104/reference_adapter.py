"""96-bit analytic occupation-energy reference; no input loading at import."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys


def modules():
    root = Path(__file__).resolve().parents[3]
    path = root/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module('ball_adapter'), importlib.import_module('mean_adapter')


def verify_runtime_basis(preparation, export, indices):
    """Parameter/class/descriptor checks, without local Operator evaluations."""
    from dataclasses import asdict
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    components = preparation.rte_preparation.symbolic_tail.components
    used = {components[i].basis_id for i in indices}
    used.update(block.basis_id for block in preparation.deterministic_blocks)
    registry = preparation.rte_preparation.basis_registry
    # Both registries use the same deterministic and tail definitions.
    definitions = {block.basis_id: block for block in preparation.deterministic_blocks}
    for key in sorted(used):
        saved = export['bases'][key]
        if key in definitions:
            block = definitions[key]
            fingerprint, desc, ops = block.basis_hash, block.basis_change_operations, block.runtime_basis_operations
        else:
            block = registry.definition(key)
            fingerprint, desc, ops = block.basis_hash, block.metadata.operations, block.runtime_operations
        if saved['basis_hash'] != fingerprint or len(ops) != len(saved['operations']) or len(desc) != len(ops):
            raise ValueError('UNKNOWN/STOP: reference basis identity')
        for description, (gate, wires), old in zip(desc, ops, saved['operations'], strict=True):
            if (gate.base_class not in (RZGate, XXPlusYYGate) or gate.name != old['name']
                    or list(wires) != old['qubits'] or [float(p).hex() for p in gate.params] != old['params']
                    or json.dumps(asdict(description), sort_keys=True) != json.dumps(old['legacy_descriptor'], sort_keys=True)):
                raise ValueError('UNKNOWN/STOP: reference gate class/parameters/descriptor')
    return sorted(used)


def build_reference(preparation, export, indices, delta):
    """One F/R pair + two selected signed components. No primitive generator."""
    from flint import acb, acb_mat
    b, mean = modules()
    ordered = export['ordered_sector']
    dimension = len(ordered)
    used = verify_runtime_basis(preparation, export, indices)
    bases = {key: mean.basis_matrix(export['bases'][key], ordered, export['n_qubits'], export['n_electrons'])
             for key in used}
    blocks = []
    for block in preparation.deterministic_blocks:
        values = []
        for state in ordered:
            if block.block_kind == 'one_body':
                energy = sum((b.real(e)*((state >> j) & 1) for j, e in enumerate(block.diagonal_eigenvalues)), b.real(0))
            elif block.block_kind == 'df_fragment':
                occupied = sum((b.real(e)*((state >> j) & 1) for j, e in enumerate(block.diagonal_eta)), b.real(0))
                energy = b.real(block.lam)*occupied**2
            else:
                raise ValueError('UNKNOWN/STOP: unsupported deterministic block')
            values.append(acb(0, -b.real(delta)*energy/2).exp())
        u = bases[block.basis_id]
        blocks.append(b.product(b.product(u, mean.diagonal(values, 1)), b.adjoint(u)))
    # Match the stored per-step phase parameters, not reassociated optimized scalar arithmetic.
    phase = acb(1)
    expected = [-delta*preparation.constant_coefficient, -delta*preparation.extracted_identity_coefficient]
    if [float(x).hex() for x in expected] != export['step_phase_angles']:
        raise ValueError('UNKNOWN/STOP: saved constant/identity phase parameters')
    for angle in export['step_phase_angles']:
        phase *= acb(0, b.real(angle)).exp()
    f, r = b.unit(phase*b.identity(dimension)), b.unit(b.identity(dimension))
    for block in blocks:
        f = b.product(block, f)
    for block in reversed(blocks):
        r = b.product(block, r)
    components = preparation.rte_preparation.symbolic_tail.components
    operators = {}
    for index in indices:
        c = components[index]
        signs = [c.coefficient_sign*(-1)**sum((state >> q) & 1 for q in c.diagonal_pauli_support) for state in ordered]
        u = bases[c.basis_id]
        operators[c.component_id] = b.product(b.product(u, mean.diagonal(signs, 1)), b.adjoint(u))
    psi = acb_mat([[b.zball(pair)] for pair in export['state']])
    norm2 = (b.dagger(psi)*psi)[0, 0].real
    if not norm2 > 0:
        raise ValueError('UNKNOWN/STOP: state norm²')
    state = b.recenter(psi/norm2.sqrt(), norm=1)
    return {'forward': f, 'reverse': r, 'operators': operators, 'state': state,
            'ordered': ordered, 'n_qubits': export['n_qubits'], 'event_cache': {},
            'norm2': b.endpoints(norm2), 'basis_count': len(bases)}


def event_operator(event, reference):
    from flint import acb
    b, _ = modules()
    key = (event.rotation_component_id, event.product_component_ids, event.rotation_angle.hex(), event.phase)
    cache = reference['event_cache']
    if key not in cache:
        operators = reference['operators']
        eye = b.unit(b.identity(len(reference['ordered'])))
        theta = b.real(event.rotation_angle)
        # Norm=1 follows from a Hermitian involution, not from observed float norm.
        rotation = b.weighted_sum([(theta.cos(), eye), (-acb(0, 1)*theta.sin(), operators[event.rotation_component_id])], norm=1)
        product = eye
        for component in event.product_component_ids:
            product = b.product(operators[component], product)
        cache[key] = b.weighted_sum([(event.phase.real, b.product(rotation, product))], norm=1)
    return cache[key]


def to_float_vector(vector, reference):
    import numpy as np
    from flint import acb
    b, _ = modules()
    full = np.zeros(1 << reference['n_qubits'], dtype=np.complex128)
    bound = b.real(0)
    for row, position in enumerate(reference['ordered']):
        value = vector.center[row, 0]
        z = complex(float(value.real.mid()), float(value.imag.mid()))
        full[position] = z
        error = b.upper(abs(value-acb(b.real(z.real), b.real(z.imag))))
        if error > bound:
            bound = error
    return full, str(b.dyadic(b.upper(bound+vector.radius)))


def trajectory_reference(reference, events, q, r):
    b, _ = modules()
    if q not in (1, 2) or len(events) != q*r:
        raise ValueError('UNKNOWN/STOP: reference probe count')
    vector = reference['state']
    for j in range(q):
        vector = b.product(reference['forward'], vector)
        for event in events[j*r:(j+1)*r]:
            vector = b.product(event_operator(event, reference), vector)
        vector = b.product(reference['reverse'], vector)
    scalar = b.product(b.adjoint(reference['state']), vector)
    target, error = to_float_vector(vector, reference)
    signal, signal_error = {}, {}
    for name, coordinate in (('real', scalar.center[0, 0].real), ('imag', scalar.center[0, 0].imag)):
        number = float(coordinate.mid())
        signal[name] = number
        signal_error[name] = str(b.dyadic(b.upper(abs(coordinate-b.real(number))+scalar.radius)))
    return {'target': target, 'target_error': error, 'signal': signal, 'signal_error': signal_error}


def scalar_residual(actual, expected, reference_error):
    import math
    b, _ = modules()
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError('UNKNOWN/STOP: nonfinite mean')
    bound = b.upper(abs(b.real(float(actual))-b.real(float(expected)))+b.real(reference_error))
    return {'residual_upper': str(b.dyadic(bound)), 'pass': bool(bound <= b.real('1e-10').lower())}


def vector_residual(actual, expected, reference_error):
    import numpy as np
    from flint import acb
    b, _ = modules()
    if np.shape(actual) != np.shape(expected) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('UNKNOWN/STOP: nonfinite/mismatched vector')
    largest = b.real(0)
    for a, e in zip(actual, expected, strict=True):
        difference = acb(b.real(float(a.real))-b.real(float(e.real)), b.real(float(a.imag))-b.real(float(e.imag)))
        bound = b.upper(abs(difference))
        if bound > largest:
            largest = bound
    bound = b.upper(largest+b.real(reference_error))
    return {'residual_upper': str(b.dyadic(bound)), 'pass': bool(bound <= b.real('1e-10').lower())}
