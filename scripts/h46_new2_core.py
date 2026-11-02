"""Small-sector, exact-mean diagnostic; never samples or compiles a circuit."""
import math
import numpy as np
from scipy.linalg import eigh


def weights(label):
    from trotterlib.product_formula import _get_w_list
    half = tuple(_get_w_list(label))
    result = tuple(reversed(half[1:])) + half[:1] + half[1:]
    if abs(math.fsum(result)-1) > 1e-12 or result != tuple(reversed(result)):
        raise ValueError('COEFFICIENT_NORMALIZATION_OR_SYMMETRY')
    return result


def allocation(coefficients, total):
    """Same total RTE steps per physical outer step for every PF family."""
    if type(total) is not int or total < len(coefficients):
        raise ValueError('INSUFFICIENT_OCCURRENCE_STEPS')
    magnitude = [abs(x) for x in coefficients]
    quota = [(total-len(coefficients))*x/math.fsum(magnitude) for x in magnitude]
    result = [1+math.floor(x) for x in quota]
    order = sorted(range(len(result)), key=lambda i: (-(quota[i]-math.floor(quota[i])), i))
    for i in order[:total-sum(result)]:
        result[i] += 1
    if sum(result) != total or min(result) < 1:
        raise ValueError('ALLOCATION')
    return tuple(result)


def spectrum(matrix, tolerance=1e-9):
    a = np.asarray(matrix, dtype=complex)
    if not np.isfinite(a).all() or np.linalg.norm(a-a.conj().T, 'fro') > tolerance:
        raise ValueError('NONFINITE_OR_NONHERMITIAN')
    values, vectors = eigh(a, check_finite=True)
    if np.linalg.norm((vectors*values)@vectors.conj().T-a, 'fro') > tolerance:
        raise ValueError('SPECTRUM_RECONSTRUCTION')
    return values, vectors, vectors.conj().T


def spectral_apply(spec, values, vectors):
    result = spec[1] @ (values[:, None]*(spec[2] @ vectors))
    if not np.isfinite(result).all():
        raise ValueError('NONFINITE_PROPAGATION')
    return result


def polynomial(values, time, cutoff):
    """Source-defined paired Taylor numerator, degrees zero through K+1."""
    if type(cutoff) is not int or cutoff < 0 or cutoff % 2:
        raise ValueError('INVALID_FINITE_TAYLOR_ORDER')
    x = np.asarray(values, dtype=complex)
    term = np.ones(x.shape, dtype=complex)
    result = term.copy()
    for k in range(1, cutoff+2):
        term = term*(-1j*time*x/k)
        result += term
    return result


def compare_prefix(state, deterministic_spectra, tail_spectrum, scalar, lambda_r,
                   label, q, *, T=.8, total_steps=32, cutoffs=(2,4)):
    from trotterlib.rte import finite_rte_distribution
    coefficients = weights(label)
    counts = allocation(coefficients, total_steps)
    # Exact-tail PF, corrected means, independently attenuated raw means.
    columns = 1+2*len(cutoffs)
    vectors = np.repeat(np.asarray(state,complex)[:,None], columns, axis=1)
    logs = [0.]*len(cutoffs)
    phases = []
    for c, r in zip(coefficients, counts):
        t = T*c/q
        factor = np.empty((len(state), columns),complex)
        factor[:,0] = np.exp(-1j*t*tail_spectrum[0])
        for k, cutoff in enumerate(cutoffs):
            b = finite_rte_distribution(lambda_r*t/r, cutoff).exact_finite_distribution
            value = polynomial(tail_spectrum[0], t/r, cutoff)**r
            factor[:,1+k] = value
            factor[:,1+len(cutoffs)+k] = value/(b**r)
            logs[k] += q*r*math.log(b)
        phases.append((t, factor))
    for _ in range(q):
        for t, tail_factor in phases:
            for spec in deterministic_spectra:
                vectors = spectral_apply(spec, np.exp(-.5j*t*spec[0]), vectors)
            vectors = tail_spectrum[1] @ (tail_factor*(tail_spectrum[2] @ vectors))
            if not np.isfinite(vectors).all():
                raise ValueError('NONFINITE_TAIL')
            for spec in reversed(deterministic_spectra):
                vectors = spectral_apply(spec, np.exp(-.5j*t*spec[0]), vectors)
    vectors *= np.exp(-1j*scalar*T)
    if abs(np.linalg.norm(vectors[:,0])-1) > 1e-9:
        raise ValueError('EXACT_TAIL_PF_NORM')
    rows = []
    for k, cutoff in enumerate(cutoffs):
        corrected = vectors[:,1+k]
        raw = vectors[:,1+len(cutoffs)+k]
        attenuation = math.exp(-logs[k])
        discrepancy = float(np.linalg.norm(raw-attenuation*corrected))
        if discrepancy > 1e-9:
            raise ValueError('RAW_CORRECTED_NORMALIZATION')
        rows.append(dict(cutoff=cutoff, log_B=logs[k], attenuation=attenuation,
            normalization_only_shot_multiplier=(math.exp(2*logs[k]) if 2*logs[k]<700 else None),
            shot_multiplier_overflow=2*logs[k]>=700,
            corrected_signal=complex(np.vdot(state,corrected)),
            raw_signal=complex(np.vdot(state,raw)),
            corrected_norm=float(np.linalg.norm(corrected)),
            raw_corrected_vector_discrepancy=discrepancy,
            finite_RTE_vector_error=float(np.linalg.norm(corrected-vectors[:,0]))))
    return dict(exact_tail_vector=vectors[:,0], finite_rows=rows,
        coefficients=coefficients, allocation=counts,
        deterministic_primitive_actions=q*2*len(deterministic_spectra)*len(coefficients),
        RTE_short_steps=q*total_steps,
        tail_absolute_time=T*math.fsum(abs(c) for c in coefficients))


def occupation_one_body(matrix, sector):
    from trottertracks.resource_applicability.ax2b_independent_reference import _one_body
    basis = tuple(int(x) for x in sector.basis_indices)
    positions = {b:i for i,b in enumerate(basis)}
    result = np.zeros((len(basis),len(basis)),complex)
    for j,b in enumerate(basis):
        for target,value in _one_body(matrix, {b:1.+0j}).items():
            if target not in positions and value != 0:
                raise ValueError('ONE_BODY_SECTOR_LEAKAGE')
            if target in positions:
                result[positions[target],j] = value
    return result


def matrix_only_tail_coefficients(hamiltonian, prefix):
    """Reuse canonical full-Fock I/Z/ZZ coefficients without building a circuit."""
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_rte_tail import exact_df_diagonal_coefficients
    if type(prefix) is not int or not 0 <= prefix < hamiltonian.n_blocks:
        raise ValueError('INVALID_GENERATION_PREFIX')
    identity = []
    magnitudes = []
    for i in range(prefix, hamiltonian.n_blocks):
        _, eta = diag_hermitian(hamiltonian.g_matrices[i],sort='descending_abs',assume_hermitian=True)
        for support,value in exact_df_diagonal_coefficients(eta,hamiltonian.lambdas[i]):
            if support:
                magnitudes.append(abs(value))
            else:
                identity.append(value)
    return math.fsum(identity), math.fsum(magnitudes)


def complex_record(value):
    return dict(real=float(value.real),imag=float(value.imag))
