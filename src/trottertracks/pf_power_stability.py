"""Diagnostics for supplied matrices; no molecular inputs or production dispatch.

The original screen and its frozen gates remain unchanged. Alternative LAPACK
drivers and direct expm are explicit diagnostic choices, never fallbacks.
"""
import math

POWER_TOLERANCE = 1e-9


def spectral_metrics(matrix, spec):
    import numpy as np
    from .algorithm_codesign.numerics import Spectrum
    values, vectors, adjoint = spec
    n = len(matrix)
    # Reuse the existing observer for the SAME supplied eigensystem. Its
    # conditional roundoff guard is not a new global RMSE certificate.
    observer = Spectrum(matrix, eigensystem=(values, vectors))
    return dict(
        reconstruction_fro=float(np.linalg.norm((vectors*values)@adjoint-matrix, 'fro')),
        eigen_residual_fro=float(np.linalg.norm(matrix@vectors-vectors*values, 'fro')),
        gram_defect_fro=float(np.linalg.norm(adjoint@vectors-np.eye(n), 'fro')),
        zero_time_identity_defect_fro=float(np.linalg.norm(vectors@adjoint-np.eye(n), 'fro')),
        existing_conditional_gram_guard=observer.gram_error,
        existing_conditional_matrix_guard=observer.matrix_error)


def explicit_spectrum(matrix, driver):
    """Same Hermitian problem, explicit driver; no projection or retries."""
    import numpy as np
    from scipy.linalg import eigh
    a = np.asarray(matrix, dtype=np.complex128)
    if driver not in ('evr', 'evd', 'ev'):
        raise ValueError('Explicit LAPACK driver required')
    if not np.isfinite(a).all() or np.linalg.norm(a-a.conj().T, 'fro') > POWER_TOLERANCE:
        raise ValueError('Finite Hermitian diagnostic matrix required')
    values, vectors = eigh(a, check_finite=True, driver=driver)
    return values, vectors, vectors.conj().T


def expm_step(matrices, weights, delta, cache=None):
    """Independent same-order PF construction, using existing scipy expm."""
    import numpy as np
    from scipy.linalg import expm
    cache = {} if cache is None else cache
    result = np.eye(len(matrices[0]), dtype=np.complex128)
    order = (*range(len(matrices)), *reversed(range(len(matrices))))
    for weight in weights:
        time = .5*delta*weight
        for i in order:
            key = i, time
            if key not in cache:
                cache[key] = expm(-1j*time*matrices[i])
            result = cache[key]@result
    return result


def power_diagnostics(step, q, state):
    """Return gate values BEFORE a caller decides to STOP; no repair/raise on FAIL.

    q and the normalized input state are checked, never clipped or normalized.
    These are floating-point diagnostics, not rigorous error bounds.
    """
    import numpy as np
    if type(q) is not int or q < 1:
        raise ValueError('Positive integer q required')
    step = np.asarray(step, dtype=np.complex128)
    state = np.asarray(state, dtype=np.complex128)
    if (step.shape != (len(state), len(state)) or not np.isfinite(step).all()
            or not np.isfinite(state).all() or abs(np.linalg.norm(state)-1) > 1e-12):
        raise ValueError('Finite square step and normalized supplied state required')
    with np.errstate(over='raise', invalid='raise'):
        powered = np.linalg.matrix_power(step, q)
        propagated = powered@state
        norm_error = abs(float(np.linalg.norm(propagated))-1.)
        identity = np.eye(len(state))
        unitarity = float(np.linalg.norm(powered.conj().T@powered-identity, 'fro'))
        local = float(np.linalg.norm(step.conj().T@step-identity, 'fro'))
    if not all(math.isfinite(v) for v in (norm_error, unitarity, local)):
        raise ValueError('Nonfinite diagnostic')
    return powered, dict(q=q,step_unitarity_error=local,norm_error=norm_error,
        power_unitarity_error=unitarity,threshold=POWER_TOLERANCE,
        norm_gate_pass=norm_error<=POWER_TOLERANCE,unitarity_gate_pass=unitarity<=POWER_TOLERANCE,
        gate_pass=norm_error<=POWER_TOLERANCE and unitarity<=POWER_TOLERANCE,
        repaired=False,certified=False)


def artificial_blocks(kind, seed, n=36, count=11):
    """Invented matrices/state only; no saved values, Hamiltonian or spectrum."""
    import numpy as np
    if kind not in ('regular', 'clustered_zero'):
        raise ValueError('Artificial fixture kind required')
    rng = np.random.default_rng(seed)
    values = (np.linspace(-1,1,n) if kind=='regular' else
              np.r_[np.zeros(n*2//3),np.full(n//6,1e-12),np.ones(n-n*2//3-n//6)])
    matrices = []
    for j in range(count):
        z = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        basis, _ = np.linalg.qr(z)
        a = (basis*(values*(j+1)/(5*count)))@basis.conj().T
        matrices.append(.5*(a+a.conj().T))
    state = rng.normal(size=n)+1j*rng.normal(size=n)
    state /= np.linalg.norm(state)  # Fixture creation only, never a propagated state.
    return matrices, state
