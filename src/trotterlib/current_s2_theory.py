"""Fixed native S2 cost integration; never generate molecular or compiled data.

The caller supplies original, hash-validated prepared bases/primitive metadata,
component probabilities and native binary64 finite-RTE order probabilities.
Missing scientific metadata must be reported, never reconstructed from counts.
"""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'artifacts/resource_applicability/track_a_current_s2_theory_compiled/2026-10-22/run01/fixed_source'


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, FROZEN / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = _load('_current_s2_fixed_native', 'native_df_gate_model.py')
grouped = _load('_current_s2_fixed_grouped', 'native_df_grouped_expectation.py')
MODEL = 'NATIVE_MIXED / explicit_rz_sx_x_cx_v1'


def scalar_phase(method, *, constant, T, extracted=None, saved_B0I_scalar=None):
    """Use native final P(-scalar*T), once per trajectory, with original grouping.

    B0I uses the scalar already recorded in its original compile receipt. Do not
    derive the discarded-tail identity from a different preparation or replace
    the saved binary64 sum by a newly reassociated mathematical expression.
    """
    if method == 'B0I':
        if saved_B0I_scalar is None:
            raise ValueError('MISSING_ORIGINAL_B0I_SCALAR_RECEIPT')
        scalar = float(saved_B0I_scalar)
    elif method == 'B2':
        if extracted is None:
            raise ValueError('MISSING_PR_EXTRACTED_IDENTITY')
        scalar = float(constant) + float(extracted)
    elif method in ('B0', 'B1'):
        scalar = float(constant)
    else:
        raise ValueError('UNREGISTERED_METHOD')
    return -scalar * T


def normalized_native_law(components, order_probabilities):
    """Same Fraction normalization of saved binary64 probabilities as H46."""
    z = sum((F(c.probability) for c in components), F())
    w = sum((F(p) for p in order_probabilities.values()), F())
    if z <= 0 or w <= 0:
        raise ValueError('MISSING_COMPLETE_NATIVE_LAW')
    cs = tuple(m.Component(c.component_id, F(c.probability) / z, c.basis, c.support, c.sign)
               for c in components)
    orders = {k: F(p) / w for k, p in order_probabilities.items()}
    return cs, orders


def native_units(*, candidate, bases, primitive_factory, ancilla, constant, T=.8,
                 extracted=None, saved_B0I_scalar=None, components=None,
                 native_order_probabilities=None, tau=None):
    if candidate['formula'] != '2nd':
        raise ValueError('S2_ONLY')
    method = candidate['method']
    phase = scalar_phase(method, constant=constant, T=T, extracted=extracted,
                         saved_B0I_scalar=saved_B0I_scalar)
    if method == 'B2':
        if components is None or native_order_probabilities is None or tau is None:
            raise ValueError('MISSING_COMPLETE_PR_LAW')
        cs, orders = normalized_native_law(components, native_order_probabilities)
        prefix = m.partial_prefix_circuit(bases, primitive_factory, T, candidate['q'], ancilla,
                                         'symmetric_directional', phase, 'A')
        tail = grouped.expected_rte_grouped(m, cs, orders, candidate['r'], tau, ancilla, 'B')
        out = {}
        for axis, quad in [('cosine', 'X'), ('sine', 'Y')]:
            value = m.expected_partial_trajectory(tail, candidate['q'], prefix, ancilla, quad)
            out[axis] = dict(lower=str(value.lower.rz), upper=str(value.upper.rz))
        return out
    if components is not None:
        raise ValueError('DETERMINISTIC_METHOD_CANNOT_HAVE_RTE_TAIL')
    evolution = m.deterministic_circuit(bases, primitive_factory, T, candidate['q'], '2nd',
                                       'symmetric_directional', ancilla, phase, 'A')
    return {axis: dict(lower=str(m.count(m.hadamard_wrapper(evolution, ancilla, quad)).rz),
                       upper=str(m.count(m.hadamard_wrapper(evolution, ancilla, quad)).rz))
            for axis, quad in [('cosine', 'X'), ('sine', 'Y')]}


def total_from_saved_shots(units, shots):
    if set(shots) != {'cosine', 'sine'} or any(type(n) is not int or n <= 0 for n in shots.values()):
        raise ValueError('ORIGINAL_AXIS_SHOTS_REQUIRED')
    if set(units) != set(shots):
        raise ValueError('AXIS_SCOPE_MISMATCH')
    return {bound: str(sum((F(units[a][bound]) * shots[a] for a in shots), F()))
            for bound in ('lower', 'upper')}


def fixed_pair_display(PR, PF):
    result = {'rho_compiled': str(F(PR['compiled']) / F(PF['compiled']))}
    if PR.get('theory') is None or PF.get('theory') is None:
        result.update(rho_theory_lower=None, rho_theory_upper=None, PR_change=None, PF_change=None)
        return result
    result.update(rho_theory_lower=str(F(PR['theory']['lower']) / F(PF['theory']['upper'])),
                  rho_theory_upper=str(F(PR['theory']['upper']) / F(PF['theory']['lower'])))
    for role, row in [('PR', PR), ('PF', PF)]:
        result[role + '_change'] = {b: str(100 * (1 - F(row['compiled']) / F(row['theory'][b])))
                                    for b in ['lower', 'upper']}
    return result
