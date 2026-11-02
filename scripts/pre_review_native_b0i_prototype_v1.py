"""Independent scalar-overlay prototype; no native builder, runner or input IO.

The proposal follows H4's full-Fock Pauli identity convention for
lambda * (dGamma(G))**2. It does not adopt a new scientific baseline.
"""
from copy import deepcopy
import cmath
import hashlib
import json
import math
import re


NAMESPACE = 'NATIVE_B0_PLUS_I_PROTOTYPE_V1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('FINITE_REAL_REQUIRED')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('NONFINITE') from error
    if not math.isfinite(result):
        raise ValueError('NONFINITE')
    return result


def finite_complex(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ValueError('FINITE_COMPLEX_REQUIRED')
    try:
        z = complex(value)
    except OverflowError as error:
        raise ValueError('NONFINITE') from error
    if not math.isfinite(z.real) or not math.isfinite(z.imag):
        raise ValueError('NONFINITE')
    return z


def fragment_identity(matrix, lam):
    """lambda/4 [(Tr G)^2 + Tr(G^2)], not a fixed-sector trace.

    Exact Hermiticity is required by this prototype. Production needs a
    separately fixed receipt/tolerance if its stored representation differs.
    Lower-triangle summation agrees with the H4 UPLO=L convention on Hermitian G.
    """
    lam = finite(lam)
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError('SQUARE_HERMITIAN_REQUIRED')
    g = tuple(tuple(finite_complex(z) for z in row) for row in matrix)
    if any(g[i][j] != g[j][i].conjugate() for i in range(n) for j in range(n)):
        raise ValueError('NONHERMITIAN')
    diag = tuple(g[i][i].real for i in range(n))
    try:
        trace = math.fsum(diag)
        trace_square = math.fsum([x*x for x in diag] +
            [2.0*abs(g[i][j])**2 for i in range(n) for j in range(i)])
        return finite(lam*.25*(trace*trace+trace_square))
    except (OverflowError, ArithmeticError) as error:
        raise ValueError('IDENTITY_OVERFLOW') from error


def validate_binding(binding):
    for field, length in [('source', 40), ('hamiltonian', 64), ('df', 64),
            ('state', 64), ('input', 64), ('compiler', 64), ('b0_fingerprint', 64)]:
        if not re.fullmatch('[0-9a-f]{'+str(length)+'}', binding.get(field, '')):
            raise ValueError('BINDING:'+field)
    order = binding.get('fragment_order')
    if (not isinstance(order, list) or not order or
            any(not isinstance(x, str) or not x for x in order) or len(set(order)) != len(order)):
        raise ValueError('FRAGMENT_ORDER_BINDING')


def overlay_b0(base, fragments, *, prefix, total_time, binding):
    """Return a new immutable-by-convention proposal, never mutate B0.

    Keep all retained blocks and their internal global phases. Replace only
    the final native scalar argument (constant + discarded identity once).
    Appending an uncontrolled global phase would lose the Hadamard signal.
    """
    validate_binding(binding)
    if type(prefix) is not int or not 0 <= prefix <= len(fragments):
        raise ValueError('PREFIX')
    if ([f['id'] for f in fragments] != binding['fragment_order'] or
            base.get('retained_fragment_ids') != binding['fragment_order'][:prefix]):
        raise ValueError('FRAGMENT_ORDER_BINDING')
    if len({len(f['G']) for f in fragments}) != 1:
        raise ValueError('INCONSISTENT_MODE_DIMENSION')
    if (base.get('method') != 'B0' or base.get('tail_identity_added') is not False or
            base.get('extracted_tail_identity') != 0 or
            base.get('scalar_semantics') != 'ORIGINAL_H_CONSTANT_ONLY'):
        raise ValueError('B0_SCALAR_ALREADY_OVERLAID_OR_AMBIGUOUS')
    if base.get('binding') != binding:
        raise ValueError('B0_SOURCE_INPUT_BINDING')
    if base.get('control') not in ('ordinary', 'symmetric_directional'):
        raise ValueError('CONTROL')
    original_digest = digest(base)
    identities = [fragment_identity(f['G'], f['lambda']) for f in fragments]
    tail = finite(math.fsum(identities[prefix:]))
    time = finite(total_time)
    scalar = finite(finite(base['scalar']) + tail)
    phase = finite(-time*tail)
    result = deepcopy(base)
    result.update(method='B0_PLUS_I_PROPOSAL', namespace=NAMESPACE,
        tail_identity_added=True, extracted_tail_identity=tail, scalar=scalar,
        total_time=time, discarded_fragment_ids=binding['fragment_order'][prefix:],
        b0_original_digest=original_digest,
        scalar_application='REPLACE_NATIVE_FINAL_SCALAR_ONCE',
        added_relative_phase=phase, identity_convention='FULL_FOCK_PAULI_IDENTITY',
        scientific_adoption=False, runtime_authorization=False)
    if digest(base) != original_digest:
        raise ValueError('B0_MUTATED')
    return result


def rotate_saved_signal(signal, *, total_time, tail_identity):
    z = finite_complex(signal)
    angle = finite(-finite(total_time)*finite(tail_identity))
    return finite_complex(cmath.exp(1j*angle)*z)


def wrapper_descriptor(proposal, axis):
    if proposal.get('namespace') != NAMESPACE or axis not in ('cosine', 'sine'):
        raise ValueError('PROPOSAL_OR_AXIS')
    return dict(namespace=NAMESPACE, binding=deepcopy(proposal['binding']),
        proposal_digest=digest(proposal), axis=axis, control=proposal['control'],
        scalar_phase_target='ANCILLA_RELATIVE_PHASE',
        native_final_phase_angle=finite(-proposal['total_time']*proposal['scalar']),
        axis_expectation='real' if axis == 'cosine' else 'imag',
        after_evolution=[] if axis == 'cosine' else ['Sdg(ancilla)'],
        runtime_authorization=False)


def rotated_allowance(u_real, u_imag, *, angle, signal_abs_upper,
        phase_error_upper, rotation_roundoff_upper):
    """Propagate axis errors; supplied bounds retain their original evidence class.

    This does not estimate scalar uncertainty or upgrade empirical u to a
    certificate. Using the original per-axis u unchanged is not justified.
    """
    values = [finite(x) for x in (u_real, u_imag, signal_abs_upper,
        phase_error_upper, rotation_roundoff_upper)]
    if min(values) < 0:
        raise ValueError('NEGATIVE_ALLOWANCE')
    ur, ui, magnitude, phase_error, rounding = values
    angle = finite(angle)
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    extra = finite(magnitude*min(2.0, phase_error)+rounding)
    return dict(real=finite(c*ur+s*ui+extra), imag=finite(s*ur+c*ui+extra),
        certificate_created=False, scalar_error_estimation='CALLER_MUST_SUPPLY')
