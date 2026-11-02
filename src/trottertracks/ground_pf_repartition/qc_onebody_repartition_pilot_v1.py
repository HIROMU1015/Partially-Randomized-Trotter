"""Bounded first-pair pilot kernels; the registered entry is separately gated.

No molecule builder, optimizer, compiler, or full ground-state generator exists
here. Tests supply small, unregistered hand-written arrays. Registered arrays
must be loaded by the authorization wrapper, never by a test/preflight.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.linalg import schur


class PilotError(ValueError):
    pass


@dataclass
class Budget:
    limits: dict
    counts: dict = field(default_factory=dict)

    def tick(self, name, number=1):
        count = self.counts.get(name, 0) + number
        if name not in self.limits or count > self.limits[name]:
            raise PilotError('CALL_LIMIT_' + name)
        self.counts[name] = count


def hermitian(a, tolerance=1e-10):
    a = np.asarray(a, complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise PilotError('INVALID_MATRIX')
    if np.linalg.norm(a - a.conj().T, 'fro') > tolerance:
        raise PilotError('NON_HERMITIAN')
    return a


def occupations(n, labels):
    return ((np.asarray(labels)[:, None] >> np.arange(n - 1, -1, -1)) & 1)


def sector_ladders(n, labels):
    """CAR transitions with mode0 the most significant bit, not Qiskit order."""
    labels = [int(x) for x in labels]
    if len(set(labels)) != len(labels) or any(x < 0 or x >= 1 << n for x in labels):
        raise PilotError('SECTOR_LABELS')
    index = {x: i for i, x in enumerate(labels)}
    out = np.zeros((n, n, len(labels), len(labels)), complex)
    for col, label in enumerate(labels):
        for q in range(n):
            qb = 1 << (n - 1 - q)
            if not label & qb:
                continue
            first_sign = (-1) ** ((label >> (n - q)).bit_count())
            removed = label ^ qb
            for p in range(n):
                pb = 1 << (n - 1 - p)
                if removed & pb:
                    continue
                row = index.get(removed | pb)
                if row is not None:
                    sign = first_sign * (-1) ** ((removed >> (n - p)).bit_count())
                    out[p, q, row, col] = sign
    return out


def lift(a, ladders):
    return np.einsum('pq,pqij->ij', a, ladders, optimize=True)


def rdm(state, ladders, budget):
    budget.tick('RDM_acquisitions')
    return np.einsum('i,pqij,j->pq', state.conj(), ladders, state, optimize=True).T


def response(a, gamma):
    a, gamma = hermitian(a), hermitian(gamma)
    if a.shape != gamma.shape:
        raise PilotError('RDM_SHAPE')
    square = a @ a
    g = np.real(np.diag(gamma @ square + square @ gamma - 2 * a @ gamma @ a)) / 12
    Q = np.zeros(a.shape, float)
    for r in range(len(a)):
        for s in range(r + 1, len(a)):
            weight = -float(np.real(a[r, s] * gamma[s, r])) / 12
            Q[r, r] += weight
            Q[s, s] += weight
            Q[r, s] -= weight
            Q[s, r] -= weight
    return g, Q


def bounded_candidates(d0, linear, quadratic, bound, tolerance=1e-12):
    if not np.isfinite([d0, linear, quadratic, bound]).all() or bound < 0:
        raise PilotError('INVALID_SELECTOR_INPUT')
    points = [0., -bound, bound]
    if abs(quadratic) > tolerance:
        points.append(-linear / (2 * quadratic))
        disc = linear * linear - 4 * quadratic * d0
        if disc >= 0:
            # Stable pair of quadratic roots; no iterative optimization.
            term = -.5 * (linear + math.copysign(math.sqrt(disc), linear))
            if term:
                points.extend([term / quadratic, d0 / term])
            else:
                points.append(-linear / (2 * quadratic))
    elif abs(linear) > tolerance:
        points.append(-d0 / linear)
    points = sorted(set(float(t) for t in points if -bound <= t <= bound))
    values = [d0 + linear * t + quadratic * t * t for t in points]
    best = min(abs(v) for v in values)
    eligible = [i for i, v in enumerate(values) if abs(v) <= best + tolerance]
    winner = min(eligible, key=lambda i: (abs(points[i]), points[i]))
    return points[winner], [{'t': t, 'leading': v} for t, v in zip(points, values)]


def select_response(a_frame, gamma_frame, d0, bound):
    g, Q = response(a_frame, gamma_frame)
    direction = g[::2] + g[1::2]
    direction -= direction.mean()
    scale = float(np.max(np.abs(direction)))
    if scale <= 1e-12:
        return np.zeros(len(g)), {'status': 'ZERO_PROJECTED_GRADIENT', 't': 0., 'candidates': []}
    y = np.repeat(direction / scale, 2)
    linear, quadratic = float(g @ y), float(y @ Q @ y)
    t, candidates = bounded_candidates(float(d0), linear, quadratic, float(bound))
    return t * y, {'status': 'SEALED_LEADING_ONLY', 't': t, 'direction': y.tolist(),
                    'linear': linear, 'quadratic': quadratic, 'bound': float(bound),
                    'candidates': candidates, 'predicted_leading': float(d0 + linear*t + quadratic*t*t),
                    'E0_accessed': False, 'finite_step_truth_accessed': False}


def known_linear(eta, weight, bound):
    x = float(weight) * np.asarray(eta, float)**2
    x -= x.mean()
    maximum = float(np.max(np.abs(x)))
    return np.repeat(x * min(1., bound / maximum) if maximum else x, 2)


def comm(a, b):
    return a @ b - b @ a


def leading_operator(fragments, budget):
    budget.tick('direct_leading_evaluations')
    out = np.zeros_like(fragments[0]); tail = fragments[-1].copy()
    for a in reversed(fragments[:-1]):
        ab = comm(a, tail)
        out += comm(a, ab) / 24 + comm(tail, ab) / 12
        tail += a
    return hermitian(out)


def spin_frame(spatial):
    n = len(spatial) * 2
    out = np.zeros((n, n), complex)
    out[::2, ::2] = spatial
    out[1::2, 1::2] = spatial
    return out


def paired_eigh(a, budget):
    a = hermitian(a); n = len(a)
    if n % 2 or np.max(np.abs(a[::2, 1::2])) > 1e-12 or np.max(np.abs(a[1::2, ::2])) > 1e-12:
        raise PilotError('SPIN_MIXING_NOT_SUPPORTED')
    if np.max(np.abs(a[::2, ::2] - a[1::2, 1::2])) > 1e-12:
        raise PilotError('SPIN_BLOCKS_DIFFER')
    budget.tick('spatial_eigh_calls')
    eta, frame = np.linalg.eigh(a[::2, ::2])
    for j in range(len(frame)):
        pivot = int(np.argmax(np.abs(frame[:, j])))
        frame[:, j] *= np.exp(-1j * np.angle(frame[pivot, j]))
    return np.repeat(eta, 2), spin_frame(frame)


def exterior_frame(u, labels):
    occ = [np.flatnonzero(row).tolist() for row in occupations(len(u), labels)]
    if len({len(row) for row in occ}) != 1:
        raise PilotError('NON_FIXED_PARTICLE_SECTOR')
    idx = np.asarray(occ)
    minors = u[idx[:, None, :, None], idx[None, :, None, :]]
    return np.linalg.det(minors)


def build_design(one_body, weights, gs, constant, labels, state, budget):
    """Acquisition/design only; does not compute E0 or any PF eigenphase."""
    a = hermitian(one_body); n = len(a); state = np.asarray(state, complex)
    if state.shape != (len(labels),) or not np.isfinite(state).all() or abs(np.vdot(state, state).real - 1) > 1e-10:
        raise PilotError('STATE_NORMALIZATION_OR_SHAPE')
    if len(gs) != len(weights) or len(gs) < 2:
        raise PilotError('DENSITY_FRAGMENT_COUNT')
    ladders = sector_ladders(n, labels); A = lift(a, ladders)
    Bs = [float(w) * (lift(hermitian(g), ladders) @ lift(g, ladders)) for w, g in zip(weights, gs)]
    H = float(constant) * np.eye(len(labels)) + A + sum(Bs)
    gamma = rdm(state, ladders, budget)
    spectra = [paired_eigh(g, budget) for g in gs]
    eta, u = spectra[0]
    D0 = leading_operator([A, *Bs], budget)
    d0 = float(np.vdot(state, D0 @ state).real)
    bound = .25 * max(1e-12, float(np.max(np.abs(a))))
    x, selection = select_response(u.conj().T @ a @ u, u.conj().T @ gamma @ u, d0, bound)
    known = known_linear(eta[::2], weights[0], bound)
    configurations = []
    for name, shift in [('original', np.zeros(n)), ('response_selected', x), ('known_linear_extraction', known)]:
        Xorb = (u * shift) @ u.conj().T; X = lift(Xorb, ladders)
        fragments = [A + X, Bs[0] - X, *Bs[1:]]
        if np.linalg.norm(sum(fragments) + float(constant)*np.eye(len(H)) - H, 'fro') > 1e-10:
            raise PilotError('HAMILTONIAN_CHANGED')
        if name == 'original':
            D, d = D0, d0
        else:
            D = leading_operator(fragments, budget)
            d = float(np.vdot(state, D @ state).real)
            delta = np.trace(gamma @ (comm(a, comm(a, Xorb))/12 + comm(Xorb, comm(a, Xorb))/24)).real
            if abs(d - d0 - delta) > 1e-9:
                raise PilotError('RESPONSE_DIRECT_MISMATCH')
        configurations.append({'name': name, 'shift': shift, 'one_body': a+Xorb,
                               'fragments': fragments, 'D': D, 'd': d})
    # Check that input actually has noncommuting density fragments, not a label.
    noncommuting = any(np.linalg.norm(comm(Bs[i], Bs[j]), 'fro') > 1e-12
                       for i in range(len(Bs)) for j in range(i+1, len(Bs)))
    if not noncommuting:
        raise PilotError('NO_NONCOMMUTING_DENSITY_FRAGMENTS')
    return {'a': a, 'gamma': gamma, 'ladders': ladders, 'H': H, 'state': state,
            'constant': float(constant), 'labels': np.asarray(labels), 'weights': np.asarray(weights),
            'g_matrices': np.asarray(gs), 'spectra': spectra, 'configurations': configurations,
            'selector': selection, 'bound': bound}


def prepare_actors(design, budget):
    labels = design['labels']; occ = occupations(len(design['a']), labels)
    shared = [(eta, u, exterior_frame(u, labels)) for eta, u in design['spectra']]
    bound = 0.
    for config in design['configurations']:
        eta, u = paired_eigh(config['one_body'], budget)
        actors = [{'frame': exterior_frame(u, labels), 'orbital_frame': u, 'energies': occ@eta, 'linear': eta, 'pairs': np.zeros((len(eta), len(eta))),
                   'kind':'onebody','eta':eta,'particles':int(occ[0].sum())}]
        local_bound = abs(design['constant']) + float(np.sum(np.abs(eta)))
        for j, (values, frame, lifted) in enumerate(shared):
            linear = design['weights'][j]*values**2 - (config['shift'] if j == 0 else 0)
            pairs = 2*design['weights'][j]*np.outer(values, values)
            energies = design['weights'][j]*(occ@values)**2 - (occ@config['shift'] if j == 0 else 0)
            actors.append({'frame': lifted, 'orbital_frame': frame, 'energies': energies, 'linear': linear, 'pairs': pairs,
                           'kind':'density','eta':values,'weight':float(design['weights'][j]),
                           'shift':config['shift'] if j==0 else np.zeros(len(eta)), 'particles':int(occ[0].sum())})
            local_bound += abs(float(design['weights'][j]))*float(np.sum(np.abs(values)))**2
            if j == 0:
                local_bound += float(np.sum(np.abs(config['shift'])))
        config['actors'] = actors
        bound = max(bound, local_bound)
    return max(bound, 1e-12)


def word(order, h, length):
    def s2(t):
        return [(j, t/2) for j in range(length-1)] + [(length-1, t)] + [(j, t/2) for j in reversed(range(length-1))]
    if order == 2:
        raw = s2(h)
    elif order == 4:
        p = 1 / (4 - 4**(1/3))
        raw = sum((s2(h*t) for t in [p,p,1-4*p,p,p]), [])
    else:
        raise PilotError('UNREGISTERED_PF_ORDER')
    fused = []
    for j, t in raw:
        if fused and fused[-1][0] == j:
            fused[-1] = (j, fused[-1][1] + t)
        else:
            fused.append((j, t))
    return [(j,t) for j,t in fused if t != 0]


def seed(actors, nodes, constant, h, budget):
    budget.tick('PF_seed_evaluations')
    out = np.eye(len(actors[0]['frame']), dtype=complex); cache = {}
    for j, t in nodes:
        key = (j, float(t).hex())
        if key not in cache:
            actor = actors[j]; u = actor['frame']
            cache[key] = (u * np.exp(-1j*t*actor['energies'])) @ u.conj().T
        out = cache[key] @ out
    return np.exp(-1j*constant*h)*out


def phase_record(U, state, h, E0, contract, budget):
    budget.tick('PF_Schur_calls')
    triangular, vectors = schur(U, output='complex')
    values = np.diag(triangular); energies = -np.angle(values)/h
    residual = float(np.linalg.norm(U@vectors - vectors*values, 'fro'))
    defect = float(np.linalg.norm(U.conj().T@U - np.eye(len(U)), 'fro'))
    if max(residual, defect) > 1e-9:
        raise PilotError('UNITARY_SPECTRAL_DEFECT')
    weights = np.abs(vectors.conj().T@state)**2
    order = np.argsort(energies); clusters = []
    tol = contract['evaluation']['energy_cluster_absolute_tolerance_Ha']
    for k in order:
        if not clusters or abs(energies[k] - energies[clusters[-1][-1]]) > tol:
            clusters.append([int(k)])
        else:
            clusters[-1].append(int(k))
    scores = [float(sum(weights[c])) for c in clusters]
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    chosen = ranked[0]
    ambiguous = len(ranked)>1 and scores[chosen]-scores[ranked[1]] <= contract['evaluation']['overlap_tie_tolerance']
    energy = float(np.mean(energies[clusters[chosen]]))
    allowance = max(contract['evaluation']['numeric_absolute_tolerance'], (residual+defect)/h)
    spread = float(np.max(np.abs(energies[clusters[chosen]] - energy)))
    return {'energy': energy, 'overlap': scores[chosen], 'bias': energy-float(E0),
            'numerical_allowance': allowance+spread, 'ambiguous': ambiguous,
            'spectral_residual': residual, 'unitarity_defect': defect,
            'cluster_indices': clusters[chosen], 'vectors': vectors, 'eigenvalues': values}
