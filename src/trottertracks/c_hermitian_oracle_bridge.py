"""Opt-in exact matrix semantics and oracle bridges; no scientific runner.

Constructing an eigh problem identifies a producer's eigenproblem. It does not
adopt that matrix as the research Hamiltonian. No files or arrays are loaded.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from . import c_direct_m0_oracle as oracle
from . import c_fresh_iid_calibration as fresh
from . import c_pr_aware_df_zsxx_enclosure as z


def pair(value):
    if isinstance(value, complex):
        return oracle.exact_real(value.real), oracle.exact_real(value.imag)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return tuple(oracle.exact_real(x) for x in value)
    return oracle.exact_real(value), F(0)


def matrix(values):
    a = tuple(tuple(pair(x) for x in row) for row in values)
    if len(a) < 2 or any(len(row) != len(a) for row in a):
        raise ValueError('SQUARE_MATRIX_REQUIRED')
    return a


def matrix_id(a):
    return fresh.digest([[[str(r), str(i)] for r, i in row] for row in a])


def conjugate(x):
    return x[0], -x[1]


def hermitian(a):
    if any(a[i][j] != conjugate(a[j][i])
           for i in range(len(a)) for j in range(len(a))):
        raise ValueError('EXACT_HERMITIAN_REQUIRED_NO_SYMMETRIZATION')
    return a


def sqrt_upper(value):
    from flint import ctx
    value = oracle.exact_real(value)
    if value < 0:
        raise ValueError('NONNEGATIVE_BOUND_REQUIRED')
    with z._BALL_LOCK, ctx.workprec(192):
        return z._fraction(z._real(value).sqrt().upper())


@dataclass(frozen=True)
class Problem:
    values: tuple
    source_sha256: str
    input_sha256: str
    kind: str
    saved_matrix_id: str
    uplo: str
    dtype: str

    @property
    def id(self):
        return fresh.digest(dict(matrix_id=matrix_id(self.values),
            source=self.source_sha256, input=self.input_sha256, kind=self.kind,
            saved=self.saved_matrix_id, uplo=self.uplo, dtype=self.dtype))


def _binding(source, input_id):
    for value in (source, input_id):
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('SOURCE_INPUT_SHA256_REQUIRED')


def eigh_problem(values, *, uplo, dtype, source_sha256, input_sha256):
    """Exact binary64 component semantics, explicitly L/U; never an average.

    The caller supplies already decoded values. dtype declares the producer's
    binary64 path; values are exact scalars, not rounded again by this adapter.
    """
    _binding(source_sha256, input_sha256)
    if uplo not in ('L', 'U'):
        raise ValueError('EXPLICIT_UPLO_L_OR_U_REQUIRED')
    if dtype not in ('<f8', '<c16'):
        raise ValueError('BINARY64_PRODUCER_DTYPE_REQUIRED')
    a = matrix(values)
    if dtype == '<f8' and any(x[1] for row in a for x in row):
        raise ValueError('REAL_DTYPE_COMPLEX_INPUT')
    b = tuple(tuple((a[i][i][0], F(0)) if i == j else
        (a[i][j] if ((i > j) == (uplo == 'L')) else conjugate(a[j][i]))
        for j in range(len(a))) for i in range(len(a)))
    return Problem(hermitian(b), source_sha256, input_sha256,
        'eigh_semantics_only_NOT_research_target', matrix_id(a), uplo, dtype)


def candidate_problem(values, *, kind, source_sha256, input_sha256):
    """An explicitly supplied Hermitian candidate; does not adopt it."""
    _binding(source_sha256, input_sha256)
    if kind not in ('candidate_original_model', 'candidate_exact_coefficients'):
        raise ValueError('EXPLICIT_CANDIDATE_DEFINITION_REQUIRED')
    a = hermitian(matrix(values))
    return Problem(a, source_sha256, input_sha256, kind, matrix_id(a), '', 'exact')


def validate(problem, *, expected_id, source_sha256, input_sha256):
    if (problem.id != expected_id or problem.source_sha256 != source_sha256
            or problem.input_sha256 != input_sha256):
        raise ValueError('SOURCE_INPUT_OR_TARGET_IDENTITY_MISMATCH')
    hermitian(problem.values)


def difference_bound(left, right):
    """Exact squared Frobenius difference, valid also for non-Hermitian Hsaved."""
    a, b = matrix(left), matrix(right)
    if len(a) != len(b):
        raise ValueError('MATRIX_DIMENSION_MISMATCH')
    norm2 = sum(((a[i][j][k]-b[i][j][k])**2
        for i in range(len(a)) for j in range(len(a)) for k in range(2)), F())
    return dict(left_matrix_id=matrix_id(a), right_matrix_id=matrix_id(b),
        identical=a == b, frobenius_squared_exact=str(norm2),
        operator_norm_upper_exact=str(sqrt_upper(norm2)), precision_bits=192,
        method='operator norm <= exact Frobenius norm; Arb outward sqrt')


def qualify(problem, amplitudes, shift, *, expected_id):
    """Exact inertia/residual certificate for THIS problem, with a supplied shift.

    The real case reuses the frozen oracle unchanged. A complex Hermitian H is
    realified as [[Re H,-Im H],[Im H,Re H]]: each eigenvalue occurs twice.
    This reuses the same exact LDL primitive, requiring exactly two negatives.
    Zero pivots STOP even for nonsingular matrices; no automatic shift retry.
    """
    validate(problem, expected_id=expected_id,
        source_sha256=problem.source_sha256, input_sha256=problem.input_sha256)
    a = problem.values
    v = [pair(x) for x in amplitudes]
    if len(v) != len(a):
        raise ValueError('VECTOR_DIMENSION_MISMATCH')
    sigma = oracle.exact_real(shift)
    if all(not x[1] for row in a for x in row):
        real = [[x[0] for x in row] for row in a]
        result = oracle.qualify_ground(real, v,
            dict(H_id=oracle.matrix_id(real), shift_exact=str(sigma)))
        result['inertia_representation'] = 'original real matrix'
    else:
        n = len(a)
        real = [[a[i][j][0] for j in range(n)] + [-a[i][j][1] for j in range(n)]
                for i in range(n)] + [
                [a[i][j][1] for j in range(n)] + [a[i][j][0] for j in range(n)]
                for i in range(n)]
        w = [x[0] for x in v] + [x[1] for x in v]
        norm2 = sum((x*x for x in w), F())
        if norm2 <= 0:
            raise ValueError('NONZERO_ORACLE_CENTRE')
        hw = [sum((x*y for x, y in zip(row, w)), F()) for row in real]
        mu = sum((x*y for x, y in zip(w, hw)), F()) / norm2
        inertia = oracle.inertia_below(real, sigma)
        if inertia['negative'] != 2 or inertia['positive'] != 2*n-2 or mu >= sigma:
            raise ValueError('UNIQUE_GROUND_ISOLATION_NOT_CERTIFIED')
        residual2 = sum(((x-mu*y)**2 for x, y in zip(hw, w)), F()) / norm2
        sin2 = residual2 / (sigma-mu)**2
        if sin2 >= 1:
            raise ValueError('ORACLE_OVERLAP_BOUND_UNRESOLVED')
        from flint import ctx
        with z._BALL_LOCK, ctx.workprec(192):
            norm = z._real(norm2).sqrt()
            lo, hi = z._fraction(norm.lower()), z._fraction(norm.upper())
        normalization = max(abs(1-lo), abs(1-hi))
        result = dict(status='VALIDATED_DISTANCE_TO_EXACT_HERMITIAN_INPUT_GROUND',
            centre_norm_squared_exact=str(norm2), centre_norm_interval=[str(lo), str(hi)],
            rayleigh_exact=str(mu), normalized_residual_squared_exact=str(residual2),
            excited_weight_upper_exact=str(sin2),
            normalization_distance_upper_exact=str(normalization),
            vector2_distance_upper_exact=str(normalization+sqrt_upper(2*sin2)),
            inertia=inertia, inertia_representation='realification; multiplicity doubled',
            input_normalized_or_projected=False, precision_bits=192, rounding_enclosed=True,
            ground_phase='aligned to original centre', statistical_confidence=None)
    result.update(problem_id=problem.id, exact_hermitian_matrix_id=matrix_id(a),
        vector_id=fresh.digest([[str(x) for x in q] for q in v]),
        ground_gap_lower_exact=str(sigma-F(result['rayleigh_exact'])),
        research_target_adopted=False, existing_own_DF_target_identity_established=False)
    return result


def transport_ground(reference, target, amplitudes, shift, *, expected_reference_id,
                     expected_target_id):
    """Distance from ORIGINAL centre to candidate target ground; no adoption.

    If A has gap >=g and ||B-A||<=eps, g>2eps implies B unique with gap
    >=g-2eps. Excited weight of g_A in B is <=(eps/(g-eps))^2.
    Phase-aligned unit ground distance <=sqrt(2)*eps/(g-eps).
    """
    validate(target, expected_id=expected_target_id,
        source_sha256=target.source_sha256, input_sha256=target.input_sha256)
    cert = qualify(reference, amplitudes, shift, expected_id=expected_reference_id)
    bridge = difference_bound(reference.values, target.values)
    eps = F(bridge['operator_norm_upper_exact'])
    gap = F(cert['ground_gap_lower_exact'])
    if gap <= 2*eps:
        raise ValueError('PERTURBATION_GAP_NOT_CERTIFIED_NO_TARGET_ADOPTION')
    distance = sqrt_upper(2*eps**2/(gap-eps)**2)
    return dict(reference_problem_id=reference.id, target_problem_id=target.id,
        reference_certificate=cert, matrix_bridge=bridge,
        target_ground_gap_lower_exact=str(gap-2*eps),
        ground_transport_distance_upper_exact=str(distance),
        original_centre_to_target_distance_upper_exact=str(
            F(cert['vector2_distance_upper_exact'])+distance),
        research_target_adopted=False, precision_bits=192)


def state_signal_bounds(eta, *, map_norm_upper):
    """Two UNIT initial states, phase-aligned distance eta, same linear map M.

    Each X/Y observable has norm 1; error <=2||M||^2 eta per axis.
    For unnormalized saved centres use direct.observable_z's radius formula.
    Do not add this again if eta is already the direct evaluator's radius.
    """
    eta, b = oracle.exact_real(eta), oracle.exact_real(map_norm_upper)
    if min(eta, b) < 0:
        raise ValueError('NONNEGATIVE_STATE_AND_MAP_BOUNDS')
    axis = 2*b*b*eta
    return dict(X_upper_exact=str(axis), Y_upper_exact=str(axis),
        complex_upper_exact=str(sqrt_upper(2*axis*axis)),
        assumes_two_unit_initial_states=True, covers_native_synthesis_error=False)
