"""Bounded occupation oracle and spin-factorized primitive actions.

The oracle enumerates fermionic ladders independently of the production DF
excitation tables. Exact cross-spin zeros are required before squaring a
projected primitive. No 4900-square dense matrix or full unitary is allocated.
"""
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, expm_multiply
from .ax2b_independent_reference import _ladder
from .ax2b_h6_controller import finite_scale_guard
from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_stage_validation_v2 import canonical_cell
from .ax2a_state_action import ActionBudget


class OccupationCSR:
    def __init__(self, basis, n):
        self.basis = tuple(int(i) for i in basis); self.n = n
        if not 1 <= n <= 16 or not 1 <= len(self.basis) <= 4900 or len(set(self.basis)) != len(self.basis):
            raise ValueError('SPARSE_SECTOR_LAYOUT')
        positions = {v: i for i, v in enumerate(self.basis)}
        rows = []; cols = []; ps = []; qs = []; signs = []
        for col, index in enumerate(self.basis):
            for q in range(n):
                a = _ladder(index, q, n, False)
                if a is None:
                    continue
                for p in range(n):
                    if p % 2 != q % 2:
                        continue
                    b = _ladder(a[0], p, n, True)
                    if b is None:
                        continue
                    if b[0] not in positions:
                        raise ValueError('SPARSE_ORACLE_SECTOR_ESCAPE')
                    rows.append(positions[b[0]]); cols.append(col)
                    ps.append(p); qs.append(q); signs.append(a[1]*b[1])
        self.rows, self.cols, self.ps, self.qs, self.signs = map(np.asarray, (rows, cols, ps, qs, signs))

    def one(self, matrix):
        a = np.asarray(matrix, dtype=complex)
        cross = np.arange(self.n)[:, None] % 2 != np.arange(self.n)[None, :] % 2
        if a.shape != (self.n, self.n) or not np.isfinite(a).all() or np.any(a[cross] != 0):
            raise ValueError('SPARSE_ORACLE_COEFFICIENT_STRUCTURE')
        if np.linalg.norm(a-a.conj().T) > 1e-12:
            raise ValueError('SPARSE_ORACLE_NOT_HERMITIAN')
        values = a[self.ps, self.qs]*self.signs
        nonzero = values != 0  # Exact zeros only; no coefficient cutoff.
        return csr_matrix((values[nonzero], (self.rows[nonzero], self.cols[nonzero])),
                          shape=(len(self.basis), len(self.basis)))


def counted_operator(dimension, action, trace, budget):
    if not math.isfinite(float(trace)):
        raise ValueError('REFERENCE_TRACE_NONFINITE')
    def mv(value):
        budget.take('reference_matvec')
        x = np.asarray(value, dtype=complex).reshape(dimension)
        y = np.asarray(action(x), dtype=complex)
        if y.shape != (dimension,) or not np.isfinite(y).all():
            raise ValueError('REFERENCE_MATVEC_NONFINITE')
        return y
    return LinearOperator((dimension, dimension), matvec=mv, rmatvec=mv, dtype=complex), float(trace)


def df_operator(constant, one, weights, blocks, budget):
    n = one.shape[0]
    def action(x):
        y = constant*x + one@x
        for w, g in zip(weights, blocks, strict=True):
            y += w*(g@(g@x))
        return y
    trace = constant*n + float(np.real(one.diagonal().sum()))
    trace += sum(float(w)*float(np.sum(np.abs(g.data)**2)) for w, g in zip(weights, blocks, strict=True))
    return counted_operator(n, action, trace, budget)


def exponential_action(operator, trace, vector, t):
    x = np.asarray(vector, dtype=complex)
    if x.ndim != 1 or not np.isfinite(x).all() or not math.isfinite(t):
        raise ValueError('EXP_ACTION_INPUT')
    if t == 0:
        return x.copy()
    return np.asarray(expm_multiply((-1j*t)*operator, x, traceA=(-1j*t)*trace), dtype=complex)


def reference_pair(operator, trace, state, T):
    a = exponential_action(operator, trace, state, T)
    b = exponential_action(operator, trace, exponential_action(operator, trace, state, T/2), T/2)
    error = float(np.linalg.norm(a-b))
    if not np.isfinite(a).all() or error > 1e-10:
        raise ValueError('REFERENCE_SPLIT_TIME_DISCREPANCY')
    energy = float(np.real(np.vdot(state, operator@state)))
    residual = float(np.linalg.norm(operator@state-energy*state))
    eigen = np.exp(-1j*T*energy)*state
    observed = float(np.linalg.norm(a-eigen))
    if observed > abs(T)*residual+1e-10:
        raise ValueError('REFERENCE_EIGEN_RESIDUAL_DISCREPANCY')
    return complex(np.vdot(state, a)), dict(state_difference=error,
        signal_difference=float(abs(np.vdot(state, a-b))), rayleigh_energy=energy,
        residual=residual, eigen_phase_difference=observed,
        method='CSR expm action versus split-time and eigen residual', certified=False)


def small_one_body(matrix, basis):
    """Independent one-spin Fock matrix (at most 70 square)."""
    a = np.asarray(matrix); n = len(a); positions = {v: i for i, v in enumerate(basis)}
    out = np.zeros((len(basis), len(basis)), dtype=complex)
    for col, index in enumerate(basis):
        for q in range(n):
            removed = _ladder(index, q, n, False)
            if removed is None:
                continue
            for p in range(n):
                created = _ladder(removed[0], p, n, True)
                if created is not None:
                    out[positions[created[0]], col] += a[p, q]*removed[1]*created[1]
    return out


class SpinPrimitive:
    def __init__(self, matrix, sector, *, weight=None):
        a = np.asarray(matrix); n = sector.n_qubits; m = n//2
        if n % 2 or m > 8 or sector.nelec_alpha != sector.nelec_beta:
            raise ValueError('SPIN_FACTORIZED_SCOPE')
        cross = np.arange(n)[:, None] % 2 != np.arange(n)[None, :] % 2
        if (a.shape != (n, n) or not np.isfinite(a).all() or np.any(a[cross] != 0)
            or not np.array_equal(a[::2, ::2], a[1::2, 1::2])):
            raise ValueError('SPIN_FACTORIZED_STRUCTURE')
        basis = tuple(v for v in range(1 << m) if v.bit_count() == sector.nelec_alpha)
        small = small_one_body(a[::2, ::2], basis)
        if np.linalg.norm(small-small.conj().T) > 1e-12:
            raise ValueError('SPIN_PRIMITIVE_HERMITICITY')
        self.eigenvalues, self.vectors = np.linalg.eigh(small)
        pos = {v: i for i, v in enumerate(basis)}; flat = []; signs = []
        for index in sector.basis_indices:
            bits = [(int(index) >> (n-1-i)) & 1 for i in range(n)]
            alpha = sum(bits[2*i] << (m-1-i) for i in range(m))
            beta = sum(bits[2*i+1] << (m-1-i) for i in range(m))
            flat.append(pos[alpha]*len(basis)+pos[beta])
            swaps = sum(bits[2*i+1]*bits[2*j] for i in range(m) for j in range(i+1, m))
            signs.append(-1 if swaps % 2 else 1)
        self.flat = np.asarray(flat); self.signs = np.asarray(signs); self.side = len(basis)
        if len(set(flat)) != self.side**2 or len(flat) != sector.dimension:
            raise ValueError('SPIN_PRIMITIVE_BASIS_BIJECTION')
        values = self.eigenvalues[:, None]+self.eigenvalues[None, :]
        self.energies = values if weight is None else float(weight)*values**2
        self.phases = {}

    def __call__(self, vector, time):
        x = np.asarray(vector, dtype=complex)
        if x.shape != self.flat.shape or not np.isfinite(x).all() or not math.isfinite(time):
            raise ValueError('SPIN_PRIMITIVE_INPUT')
        v = np.empty(self.side**2, dtype=complex); v[self.flat] = self.signs*x
        v = v.reshape(self.side, self.side)
        u = self.vectors
        if time not in self.phases:
            if len(self.phases) >= 32:
                raise ValueError('SPIN_PRIMITIVE_TIME_CACHE_CAP')
            self.phases[time] = np.exp(-1j*time*self.energies)
        out = u@((u.conj().T@v@u.conj())*self.phases[time])@u.T
        return self.signs*out.reshape(-1)[self.flat]


def recurrence_paths(state, actions, tail, *, cell, T, scalar, lambda_r, maximum_tail):
    """Independent forward Taylor/exact-tail action; no Horner or renormalization."""
    c = canonical_cell(cell); psi = np.asarray(state, dtype=complex).copy()
    if psi.ndim != 1 or len(psi) > 4900 or abs(np.linalg.norm(psi)-1) > 1e-12 or not np.isfinite(psi).all():
        raise ValueError('H8_ORACLE_STATE')
    random = bool(c['R']); tau = lambda_r*T/c['q']/c['r'] if random else 0.
    b = sum(abs(tau)**k/math.factorial(k)*math.hypot(1., tau/(k+1))
            for k in range(0, c['K']+1, 2)) if random else 1.
    log_B = c['R']*math.log(b) if random else 0.
    budget = ActionBudget(maximum_tail, 100000); maximum = 1.; signals = {}; vectors = {}
    def checked(v):
        nonlocal maximum
        if v.shape != psi.shape or not np.isfinite(v).all():
            raise ValueError('H8_ORACLE_NONFINITE')
        maximum = max(maximum, float(np.linalg.norm(v)))
        finite_scale_guard(log_B=log_B, intermediate_norm=maximum, absolute_discrepancy=0.)
        return v
    for path in ('corrected', 'raw', 'exact_tail') if random else ('corrected',):
        v = psi.copy()
        for _ in range(c['q']):
            seq = primitive_time_schedule(c, T=T)['ordinary_one_outer_step']
            if random:
                for i, t in seq[:len(actions)]:
                    budget.deterministic(); v = checked(actions[i](v, t))
                for _ in range(c['r']):
                    if path == 'exact_tail':
                        v = checked(exponential_action(tail[0], tail[1], v, tau))
                    else:
                        power = v.copy(); total = v.copy()
                        for degree in range(1, c['K']+2):
                            budget.tail(); power = checked((-1j*tau/degree)*(tail[0]@power))
                            total = checked(total+power)
                        v = checked(total/b if path == 'raw' else total)
                seq = seq[len(actions):]
            for i, t in seq:
                budget.deterministic(); v = checked(actions[i](v, t))
            v = checked(np.exp(-1j*scalar*T/c['q'])*v)
        signals[path] = complex(np.vdot(psi, v)); vectors[path] = v
    return dict(signals=signals, vectors=vectors, log_B=log_B, b=b, intermediate_norm_max=maximum,
                counts=dict(tail=budget.tail_matvecs, deterministic=budget.deterministic_actions))
