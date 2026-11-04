"""Parameter/one-particle transfer tools; import performs no scientific work.

No H4 defaults, chemistry loader, RNG, trajectory builder or execution entrypoint.
Proofs and prerequisites are in CONNECTION_ANALYSIS.md. Missing evidence is None.
"""
from __future__ import annotations

from fractions import Fraction as Q
import importlib
from pathlib import Path
import sys


def modules():
    path = Path(__file__).resolve().parents[3] / 'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module('ball_adapter'), importlib.import_module('mean_adapter')


def exact(value):
    b, _ = modules()
    return b.rational(value)


def positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError('Positive integer required')
    return value


def scalar_gap(q, raw_angles, optimized_angle):
    """Exact rational gap between *generated* parameters, not real-arithmetic c's."""
    positive_integer(q)
    return abs(exact(optimized_angle) - q * sum(map(exact, raw_angles), Q(0)))


def _primitive_map(primitives):
    rz, rzz = {}, {}
    for wire, angle in primitives.rz:
        rz[wire] = rz.get(wire, Q(0)) + exact(angle)
    for a, b, angle in primitives.rzz:
        if a == b:
            raise ValueError('RZZ needs distinct wires')
        key = tuple(sorted((a, b)))
        rzz[key] = rzz.get(key, Q(0)) + exact(angle)
    return exact(primitives.global_phase), rz, rzz


def block_gap(half_primitives, full_primitives):
    """||D_full-D_half**2|| <= min(2, result), for common diagonal Pauli axes."""
    h, f = _primitive_map(half_primitives), _primitive_map(full_primitives)
    gap = abs(f[0] - 2*h[0])
    for hm, fm in zip(h[1:], f[1:], strict=True):
        gap += sum((abs(fm.get(k, Q(0))-2*hm.get(k, Q(0))) / 2
                    for k in hm.keys() | fm.keys()), Q(0))
    return gap


def boundary_bound(q, scalar_parameter_gap, first_block_gap):
    positive_integer(q)
    a, b = exact(scalar_parameter_gap), exact(first_block_gap)
    if a < 0 or b < 0:
        raise ValueError('Nonnegative bounds required')
    return min(Q(2), a + (q-1)*b)


def half_angle_exact(angle):
    """Definition-level CRZ prerequisite; catches subnormal halving loss."""
    angle = float(angle)
    return exact(angle / 2) == exact(angle) / 2


def gaussian_descriptor_check(operations, n):
    """Only adjacent JW matchgates qualify for the CAR column certificate."""
    positive_integer(n)
    for op in operations:
        wires, params, name = op['qubits'], op['params'], op['name']
        if (len(set(wires)) != len(wires)
                or any(type(i) is not int or not 0 <= i < n for i in wires)):
            raise ValueError('Invalid wires')
        if name == 'rz' and len(wires) == len(params) == 1:
            pass
        elif (name == 'xx_plus_yy' and len(wires) == len(params) == 2
              and abs(wires[0]-wires[1]) == 1):
            pass
        else:
            raise ValueError('Unqualified Gaussian gate/topology')
        for p in params:
            exact(p)  # finite exact binary64/hex parameters


def one_particle(operations, n):
    """Certified n x n ideal Gaussian action relative to its vacuum phase.

    Reuses analytic local gate formulas and MatrixBall arithmetic. No Fock/sector
    evolution is built. RZ's vacuum phase is removed only inside this basis
    conjugation certificate, never from controlled evolution.
    """
    from flint import acb
    b, mean = modules()
    gaussian_descriptor_check(operations, n)
    out = b.unit(b.identity(n))
    for op in operations:
        local = b.identity(n)
        wires = op['qubits']
        if op['name'] == 'rz':
            local[wires[0], wires[0]] = acb(0, b.real(op['params'][0])).exp()
        else:
            gate = mean.gate_matrix(op)
            for i in range(2):
                for j in range(2):
                    local[wires[i], wires[j]] = gate[i+1, j+1]
        out = b.product(b.unit(local), out)
    return out


def component_gap(full, restricted, support):
    """CAR: ||B Z_S B† - B' Z_S B'†|| <= min(2,4 sum column gaps).

    Both MatrixBalls must enclose the qualified *unitary* one-particle actions.
    Radii are retained; a float zero residual is never used as a certificate.
    """
    from flint import arb
    b, _ = modules()
    n = full.center.nrows()
    if (full.center.ncols() != n or restricted.center.nrows() != n
            or restricted.center.ncols() != n or len(support) not in (1, 2)
            or len(set(support)) != len(support)
            or any(type(i) is not int or not 0 <= i < n for i in support)):
        raise ValueError('Invalid component support/dimension')
    total = Q(0)
    for i in support:
        square = sum((b.upper(abs(full.center[j,i]-restricted.center[j,i]))**2
                      for j in range(n)), arb(0))
        column = b.upper(square.sqrt() + full.radius + restricted.radius)
        total += b.dyadic(column)
    return min(Q(2), 4*total)


def total_transfer(q, r, *, boundary, component, lowering):
    """K2, <=3qr component applications; lowering is a whole-trajectory bound.

    The existing event pi64 bound is ALREADY in I_cat; do not pass it here again.
    None means unresolved. Setting lowering=0 needs a separate structural proof.
    """
    positive_integer(q)
    positive_integer(r)
    values = (boundary, component, lowering)
    if any(x is None for x in values):
        return None
    a, d, g = map(exact, values)
    if min(a, d, g) < 0:
        raise ValueError('Nonnegative transfer bounds required')
    return min(Q(2), a + 3*q*r*d + g)
