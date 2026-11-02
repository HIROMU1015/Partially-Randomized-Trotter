"""P1 numerical signal diagnostics; no circuit, sampler, law or shot calls."""
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from .pauli_signal_adapter import paired_polynomial_action

CONSISTENCY_ATOL = 1e-10
# A predeclared engineering resolution, not a proved floating-point bound.
DIAGNOSTIC_AXIS_RESOLUTION = 1e-10


def split_operators(ham, prefix):
    """Assemble partial operators from existing coefficients; no new H input."""
    if not 0 <= prefix <= len(ham.terms):
        raise ValueError("PREFIX")
    blocks = tuple(ham.sparse_operator([term], constant=0.)
                   for term in ham.terms[:prefix])
    shape = (1 << ham.num_qubits,) * 2
    hd = sum(blocks, csr_matrix(shape, dtype=complex))
    tail = ham.sparse_operator(ham.terms[prefix:], constant=0.)
    return SimpleNamespace(blocks=blocks, hd=hd, tail=tail,
                           scalar=ham.constant, prefix=prefix)


def propagate(ops, state, norm_exact, *, mode, T, q, r, K,
              tail_subdivisions=1):
    """One full-state signal propagation, preserving chronological half sweeps.

    Explicit CSR and traceA=0 avoid random trace estimation. Every constituent
    Pauli term is nonidentity. No state renormalization or sector projection.
    """
    if mode not in ("B", "C", "Dpoly") or tail_subdivisions not in (1, 2):
        raise ValueError("DIAGNOSTIC_MODE")
    if mode == "Dpoly" and tail_subdivisions != 1:
        raise ValueError("POLYNOMIAL_SUBDIVISION_CHANGES_TARGET")
    raw = np.asarray(state, dtype=np.complex128)
    out = raw.copy()
    dt = T / q
    exact_calls = 0
    polynomial_matvecs = 0

    def exact(op, t, value):
        nonlocal exact_calls
        exact_calls += 1
        return expm_multiply((-1j*t)*op, value, traceA=0.)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for _ in range(q):
            if mode == "B":
                out = exact(ops.hd, dt/2, out)
            else:
                for op in ops.blocks:
                    out = exact(op, dt/2, out)
            if mode == "Dpoly":
                out = paired_polynomial_action(ops.tail, out, dt, r, K)
                polynomial_matvecs += r*(K+1)
            else:
                for _ in range(tail_subdivisions):
                    out = exact(ops.tail, dt/tail_subdivisions, out)
            if mode == "B":
                out = exact(ops.hd, dt/2, out)
            else:
                for op in reversed(ops.blocks):
                    out = exact(op, dt/2, out)
            # Same per-step scalar placement as sparse_mean_action.
            out *= np.exp(-1j*ops.scalar*dt)
    z = complex(np.vdot(raw, out)/float(Fraction(norm_exact)))
    if not np.isfinite([z.real, z.imag]).all():
        raise ValueError("NONFINITE_SIGNAL")
    return dict(real=z.real, imag=z.imag,
                axis_resolution=DIAGNOSTIC_AXIS_RESOLUTION,
                error_status="EMPIRICAL_RESOLUTION_NOT_CERTIFIED",
                exponential_action_calls=exact_calls,
                polynomial_matvecs=polynomial_matvecs,
                T=T, q=q, r=r, K=K, mode=mode,
                tail_subdivisions=tail_subdivisions,
                dimension=len(raw), state_renormalized=False,
                norm_divisions=1, B_multiplications=0)


def enclosure(rect):
    """Read an existing physical signal enclosure, never evaluate its target."""
    centers, radii = {}, {}
    for axis in ("real", "imag"):
        lo, hi = Fraction(rect[axis]["lower"]), Fraction(rect[axis]["upper"])
        if lo > hi:
            raise ValueError("REVERSED_SAVED_INTERVAL")
        center = (lo+hi)/2
        centers[axis] = float(center)
        # Include conversion to the displayed binary64 center.
        radii[axis] = float((hi-lo)/2 + abs(Fraction(float(center))-center))
    return dict(**centers, axis_resolution=radii,
                error_status="REUSED_SAVED_ENCLOSURE")


def value(row):
    return complex(row["real"], row["imag"])


def difference(left, right):
    z = value(left)-value(right)
    def radius(row, axis):
        u = row["axis_resolution"]
        return u[axis] if isinstance(u, dict) else u
    u = {a: radius(left, a)+radius(right, a) for a in ("real", "imag")}
    radial = float(np.hypot(u["real"], u["imag"]))
    return dict(real=z.real, imag=z.imag, absolute=abs(z),
                axis_resolution=u, absolute_resolution=radial,
                status="RESOLVED_EMPIRICAL" if abs(z)>radial else "UNRESOLVED",
                uncertainty_kind="SAVED_ENCLOSURE_PLUS_EMPIRICAL_DIAGNOSTIC_FLOOR")
