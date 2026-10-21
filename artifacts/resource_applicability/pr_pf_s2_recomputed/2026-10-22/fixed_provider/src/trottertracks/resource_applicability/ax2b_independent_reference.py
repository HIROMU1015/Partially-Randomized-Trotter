"""Small occupation-basis oracle, independent of DF excitation tables.

No molecule builder, eigensolver, sector matvec, or circuit library is called.
Big-endian OpenFermion occupation integers are explicit. G^2 is applied in
the full occupation space before any projection. Molecular use is pending.
"""
from __future__ import annotations

import math


def _ladder(index, mode, n, create):
    bit = 1 << (n - 1 - mode)
    occupied = bool(index & bit)
    if occupied == create:
        return None
    parity = sum(bool(index & (1 << (n - 1 - k))) for k in range(mode))
    return index ^ bit, -1 if parity % 2 else 1


def _one_body(matrix, vector):
    n = len(matrix)
    result = {}
    for index, amplitude in vector.items():
        for q in range(n):
            removed = _ladder(index, q, n, False)
            if removed is None:
                continue
            for p in range(n):
                created = _ladder(removed[0], p, n, True)
                coefficient = matrix[p][q]
                if created is not None and coefficient != 0:
                    target, sign = created
                    result[target] = result.get(target, 0) + coefficient * amplitude * removed[1] * sign
    return result


def occupation_df_matrix(constant, one_body, lambdas, g_matrices, basis_indices,
                         *, convert=complex, max_dimension=400):
    """Caller-supplied coefficients, without implicit Hermitization/cutoff.

    The converter can use arbitrary precision; no scientific input is read.
    Escaping a declared sector is rejected, not projected away silently.
    """
    n = len(one_body)
    basis = tuple(int(i) for i in basis_indices)
    if not n or n > 12 or not basis or len(basis) > max_dimension:
        raise ValueError("SMALL_REFERENCE_DIMENSION")
    if len(set(basis)) != len(basis) or any(i < 0 or i >= 2**n for i in basis):
        raise ValueError("BASIS_INDICES")
    if len(lambdas) != len(g_matrices):
        raise ValueError("DF_RANK")
    def matrix(value):
        if len(value) != n or any(len(row) != n for row in value):
            raise ValueError("COEFFICIENT_SHAPE")
        return [[convert(x) for x in row] for row in value]
    one, fragments = matrix(one_body), [matrix(g) for g in g_matrices]
    positions = {index: i for i, index in enumerate(basis)}
    result = [[convert(0) for _ in basis] for _ in basis]
    for column, index in enumerate(basis):
        value = _one_body(one, {index: convert(1)})
        value[index] = value.get(index, 0) + convert(constant)
        for weight, g in zip(lambdas, fragments, strict=True):
            squared = _one_body(g, _one_body(g, {index: convert(1)}))
            for target, amplitude in squared.items():
                value[target] = value.get(target, 0) + convert(weight) * amplitude
        for target, amplitude in value.items():
            if target not in positions and amplitude != 0:
                raise ValueError("INDEPENDENT_REFERENCE_LEAVES_SECTOR")
            if target in positions:
                result[positions[target]][column] = amplitude
    return result


def reference_signal_mp(constant, one_body, lambdas, g_matrices, basis_indices,
                        state, T, *, dps=80):
    """Precision comparison backend, not an interval certificate.

    Binary64 inputs are lifted via integer ratios, never decimal str(float).
    The target state is the saved vector normalized in this precision.
    Returned decimal strings preserve the result; normalization change is
    diagnostic. Use 80/120 digits on the *same* saved target in a later run.
    """
    if type(dps) is not int or not 30 <= dps <= 160:
        raise ValueError("PRECISION_LEVEL")
    import mpmath as mp
    with mp.workdps(dps):
        def exact(value):
            z = complex(value)
            if not math.isfinite(z.real) or not math.isfinite(z.imag):
                raise ValueError("NONFINITE_COEFFICIENT")
            def real(x):
                a, b = x.as_integer_ratio()
                return mp.mpf(a) / b
            return mp.mpc(real(z.real), real(z.imag))
        h = mp.matrix(occupation_df_matrix(constant, one_body, lambdas, g_matrices,
                                           basis_indices, convert=exact))
        if len(state) != h.rows:
            raise ValueError("STATE_DIMENSION")
        psi = mp.matrix([exact(x) for x in state])
        norm = mp.sqrt(sum(abs(x)**2 for x in psi))
        if norm == 0 or abs(norm - 1) > mp.mpf("1e-12"):
            raise ValueError("SAVED_STATE_NORMALIZATION")
        hermitian_difference = mp.norm(h - h.transpose_conj())
        if hermitian_difference > mp.mpf("1e-25"):
            raise ValueError("SAVED_TARGET_NOT_HERMITIAN_AT_REQUESTED_PRECISION")
        psi /= norm
        evolved = mp.expm(-mp.j * exact(T) * h) * psi
        signal = sum(mp.conj(psi[i]) * evolved[i] for i in range(h.rows))
        return {"real": mp.nstr(signal.real, dps), "imag": mp.nstr(signal.imag, dps),
                "state_norm_before": mp.nstr(norm, dps),
                "hermitian_difference": mp.nstr(hermitian_difference, dps),
                "dps": dps, "evidence_kind": "EMPIRICAL", "certified": False}
