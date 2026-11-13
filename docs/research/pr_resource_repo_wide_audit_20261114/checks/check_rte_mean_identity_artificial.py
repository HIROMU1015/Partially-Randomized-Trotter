"""Audit check (artificial): finite-K RTE event law vs paired-Taylor polynomial, both time signs.

Uses the repository functions unchanged with fictional diagonal +/-1 involutions (no chemistry).
Claim under test (carrier/provider signal code relies on it):
    b_K(tau) * sum_events q(event) U(event) == sum_{j=0}^{K+1} (-i tau H/lambda)^j / j!
"""
import itertools, math, sys
import numpy as np
from trotterlib.rte import (RTEComponent, finite_rte_distribution, enumerate_rte_events,
                            exact_enumerated_event_mean_operator, finite_taylor_operator)

rng = np.random.default_rng(20261114)
n = 3; dim = 2 ** n
paulis = {}
for support in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
    diag = np.ones(dim)
    for b in range(dim):
        for q in support:
            if (b >> q) & 1: diag[b] *= -1
    paulis[support] = np.diag(diag).astype(complex)
coeffs = {s: float(rng.normal()) for s in paulis}
lam = sum(abs(c) for c in coeffs.values())
components = [RTEComponent(component_id='c%d' % i, probability=abs(c) / lam, coefficient_abs=abs(c),
                           coefficient_sign=1 if c > 0 else -1, diagonal_pauli_support=s)
              for i, (s, c) in enumerate(coeffs.items())]
operators = {comp.component_id: comp.coefficient_sign * paulis[comp.diagonal_pauli_support] for comp in components}
normalized = sum(comp.probability * operators[comp.component_id] for comp in components)
worst = 0.0
for K in (0, 2, 4):
    for tau in (0.37, -0.37, 1e-3, -1.9):
        d = finite_rte_distribution(tau, K)
        events = enumerate_rte_events(components, d, max_events=10 ** 7)
        mean = exact_enumerated_event_mean_operator(events, operators)
        poly = finite_taylor_operator(normalized, tau, K)
        err = float(np.abs(d.exact_finite_distribution * mean - poly).max())
        # independent closed form of the normalizer
        b = sum(abs(tau) ** k / math.factorial(k) * math.hypot(1, tau / (k + 1)) for k in range(0, K + 1, 2))
        worst = max(worst, err, abs(b - d.exact_finite_distribution))
        print('K=%d tau=%+.4g events=%d |b*E[U]-poly|=%.2e |b-closed|=%.1e attenuation=%.6f' % (K, tau, len(events), err, abs(b - d.exact_finite_distribution), 1 / b))
print('PASS' if worst < 1e-12 else 'FAIL', worst)
sys.exit(0 if worst < 1e-12 else 1)
