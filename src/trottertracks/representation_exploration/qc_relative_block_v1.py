"""Structural relative-block plan and small synthetic matrix semantics.

No molecular input, eigensolver, compiler, or resource-performance oracle.
Scalar angle functions are classical reference functions, not reversible gates.
"""
import cmath
from fractions import Fraction
import math


def validate(n, create, annihilate):
    if type(n) is not int or n < 2 or len(create) != len(annihilate) or len(create) not in (1, 2):
        raise ValueError('EXCITATION_DOMAIN')
    support = tuple(create) + tuple(annihilate)
    if any(type(i) is not int or not 0 <= i < n for i in support) or len(set(support)) != len(support):
        raise ValueError('DISJOINT_ORDERED_SUPPORT_REQUIRED')
    return tuple(sorted(support))


def detuning_plan(eps, density, create, annihilate):
    n = len(eps); support = validate(n, create, annihilate)
    if len(density) != n or any(len(row) != n for row in density):
        raise ValueError('DENSITY_SHAPE')
    if any(density[i][i] != 0 or any(density[i][j] != density[j][i] for j in range(n)) for i in range(n)):
        raise ValueError('SYMMETRIC_ZERO_DIAGONAL_J')
    if any(not isinstance(x, (int, Fraction)) or isinstance(x, bool) for x in list(eps) + [x for row in density for x in row]):
        raise ValueError('EXACT_SYNTHETIC_COEFFICIENTS_REQUIRED')
    c = {i: int(i in create) for i in support}; a = {i: int(i in annihilate) for i in support}
    constant = sum((eps[i] * (c[i] - a[i]) for i in support), Fraction())
    constant += sum((density[i][j] * (c[i] * c[j] - a[i] * a[j]) for i in support for j in support if i < j), Fraction())
    weights = {k: sum((density[k][i] * (c[i] - a[i]) for i in support), Fraction()) for k in range(n) if k not in support}
    return {'constant': constant, 'spectator_weights': weights}


def affine_value(plan, occupation):
    return plan['constant'] + sum((v * ((occupation >> i) & 1) for i, v in plan['spectator_weights'].items()), Fraction())


def parity_plan(n, create, annihilate):
    support = validate(n, create, annihilate)
    order = tuple(reversed(annihilate)) + tuple(reversed(create))
    source = sum(1 << i for i in annihilate); current = source; constant = 0
    for i in order:
        constant ^= (current & ((1 << i) - 1)).bit_count() & 1
        current ^= 1 << i
    mask = [k for k in range(n) if k not in support and sum(i > k for i in order) % 2]
    return constant, tuple(mask)


def parity_sign(plan, occupation):
    constant, mask = plan
    return (-1) ** (constant + sum((occupation >> k) & 1 for k in mask))


def encode_pair(n, create, annihilate, occupation, inverse=False):
    support = validate(n, create, annihilate); pivot = support[0]
    flip = pivot in create
    x = occupation
    if inverse and flip:
        x ^= 1 << pivot
    for k in support[1:]:
        if (x >> pivot) & 1:
            x ^= 1 << k
    if not inverse and flip:
        x ^= 1 << pivot
    return x


def valid_encoded(n, create, annihilate, occupation):
    support = validate(n, create, annihilate); pivot = support[0]
    pattern = {i: int(i in create) ^ int(pivot in create) for i in support[1:]}
    return all(((occupation >> i) & 1) == bit for i, bit in pattern.items())


def relative_matrix(delta, amplitude, time):
    if any(not math.isfinite(x) for x in (delta, amplitude, time)):
        raise ValueError('FINITE_SCALARS')
    if amplitude == 0 or time == 0:
        return ((1+0j, 0j), (0j, 1+0j))
    d = delta / 2; rho = math.hypot(d, amplitude)
    f = math.sin(time * rho) / rho; cosine = math.cos(time * rho)
    phase = cmath.exp(1j * time * d)
    return ((phase * (cosine - 1j * f * d), -1j * phase * f * amplitude),
            (-1j * phase.conjugate() * f * amplitude, phase.conjugate() * (cosine + 1j * f * d)))


def angle_recipe(delta, amplitude, time):
    if amplitude == 0 or time == 0:
        return {'kind': 'EXACT_IDENTITY_BRANCH', 'chronological_rotations': []}
    theta = math.atan2(amplitude, delta / 2)
    return {'kind': 'CLASSICAL_REFERENCE_ANGLES_NOT_REVERSIBLE_IMPLEMENTATION',
            'chronological_rotations': [('RY', -theta), ('RZ', 2 * time * math.hypot(delta / 2, amplitude)),
                                        ('RY', theta), ('RZ', -time * delta)]}


def precision_bound(time, amplitude_max, eta_delta, eta_amplitude, epsilon_impl=0):
    if any(not math.isfinite(x) for x in (time, amplitude_max, eta_delta, eta_amplitude, epsilon_impl)):
        raise ValueError('FINITE_SCALARS')
    if any(x < 0 for x in (amplitude_max, eta_delta, eta_amplitude, epsilon_impl)):
        raise ValueError('NONNEGATIVE_ERROR_BUDGETS')
    return abs(amplitude_max) * eta_delta * time * time / 2 + abs(time) * eta_amplitude + epsilon_impl


def quantize(x, fractional_bits):
    if not isinstance(x, (int, Fraction)) or isinstance(x, bool) or type(fractional_bits) is not int or not 0 <= fractional_bits <= 128:
        raise ValueError('FIXED_POINT_DOMAIN')
    scale = 1 << fractional_bits; y = Fraction(x) * scale
    # Nearest dyadic; exact half ties toward positive infinity.
    integer = (2 * y.numerator + y.denominator) // (2 * y.denominator)
    return Fraction(integer, scale)


def structural_ir(n, create, annihilate):
    support = validate(n, create, annihilate)
    return {'kind': 'STRUCTURAL_COHERENT_CONSTRUCTION_PLAN_NOT_COMPILED_CIRCUIT', 'modes': n,
            'support': support, 'pivot': support[0], 'spectator_coefficient_count': n - len(support),
            'stages': ['CNOT_pair_encode', 'compute_valid_invariant_flag', 'compute_JW_parity',
                       'compute_affine_detuning_fixed_point', 'compute_amplitude_if_grouped',
                       'compute_angles_by_reversible_arithmetic', 'conjoin_valid_and_QPE_control',
                       'parity_Z_conjugation', 'four_variable_rotations_with_exact_zero_branch',
                       'parity_Z_conjugation_inverse', 'uncompute_angles_amplitude_detuning_parity_flags',
                       'CNOT_pair_decode'],
            'exponential_spectator_table_used': False, 'reversible_function_evaluator_implemented': False,
            'compiled_T_count': None, 'circuit_depth': None, 'science_execution_authorized': False}
