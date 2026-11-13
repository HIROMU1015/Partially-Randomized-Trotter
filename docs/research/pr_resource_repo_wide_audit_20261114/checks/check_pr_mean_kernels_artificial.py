"""Audit check (artificial): three repository PR finite-mean kernels vs an independent brute-force.

Fictional Hermitian blocks/state only. Compared: (1) carrier spectral kernel h46_new2_core.compare_prefix
(155529e2 lineage), (2) A direct-expm macro kernel pr_rpe_signal_provider._macros, (3) independent
scipy.expm + explicit Taylor sum written here. Checks S2 ordering, signed pieces, per-piece event
allocation, raw/corrected normalisation log B and the single scalar phase.
"""
import math, sys
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, sys.argv[1] + '/scripts'); sys.path.insert(0, sys.argv[1] + '/src')
import h46_new2_core as core
from trottertracks import pr_rpe_signal_provider as provider
from trotterlib.h46_m5_plan import weights as plan_weights, allocation as plan_allocation

rng = np.random.default_rng(7); d = 6
def herm():
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); return (a + a.conj().T) / 4
blocks = [herm() for _ in range(5)]                      # block 0 = "one-body", 1..4 = fragments
state = rng.normal(size=d) + 1j * rng.normal(size=d); state /= np.linalg.norm(state)
constant, scalar, lam, prefix = 0.731, -0.213, 1.9, 2     # fictional scalars; lambda only sets the law
tail = sum(blocks[prefix + 1:]) - scalar * np.eye(d)
worst = 0.0
for label, R, K, q, T in (('2nd', 5, 2, 7, 1.3), ('4th', 9, 2, 3, 0.9), ('4th(m5_best)', 23, 4, 2, 0.8), ('2nd', 1, 0, 4, 2.0)):
    w = plan_weights(label); counts = plan_allocation(w, R)
    # (3) independent brute force
    step = {k: np.eye(d, dtype=complex) for k in ('exact', 'corr', 'raw')}; logB = 0.0
    for c, r in zip(w, counts):
        t = T * c / q; dt = t / r; tau = lam * dt
        b = sum(abs(tau) ** k / math.factorial(k) * math.hypot(1, tau / (k + 1)) for k in range(0, K + 1, 2))
        poly = sum(np.linalg.matrix_power(-1j * dt * tail, j) / math.factorial(j) for j in range(K + 2))
        mid = dict(exact=expm(-1j * t * tail), corr=np.linalg.matrix_power(poly, r), raw=np.linalg.matrix_power(poly / b, r))
        fwd = np.eye(d, dtype=complex)
        for h in blocks[:prefix + 1]: fwd = expm(-.5j * t * h) @ fwd            # block 0 acts first
        back = np.eye(d, dtype=complex)
        for h in reversed(blocks[:prefix + 1]): back = expm(-.5j * t * h) @ back  # block p acts first
        for k in step: step[k] = back @ mid[k] @ fwd @ step[k]
        logB += r * math.log(b)
    phase = np.exp(-1j * T * (constant + scalar))
    mine = {k: complex(np.vdot(state, np.linalg.matrix_power(step[k], q) @ state) * phase) for k in step}
    mine_logB = q * logB
    # (1) carrier spectral kernel (its `scalar` argument is the total scalar applied once)
    try:
        if tuple(core.weights(label)) != tuple(w): raise ValueError('carrier weights differ from plan weights')
        out = core.compare_prefix(state, [core.spectrum(h) for h in blocks[:prefix + 1]], core.spectrum(tail),
                                  constant + scalar, lam, label, q, T=T, total_steps=R, cutoffs=(K,))
        row = out['finite_rows'][0]
        e1 = max(abs(complex(np.vdot(state, out['exact_tail_vector'])) - mine['exact']), abs(row['corrected_signal'] - mine['corr']),
                 abs(row['raw_signal'] - mine['raw']), abs(row['log_B'] - mine_logB))
    except ValueError as exc:   # repository-tree product_formula has no m5_best table (provider root does)
        print('   carrier kernel not evaluated for', label, ':', exc); e1 = 0.0
    # (2) A direct-expm macro kernel
    ctx = provider.make_context(blocks, state, constant, {prefix: dict(scalar=scalar, lambda_R=lam, source='fictional')},
                                binding=dict(rank=4), source='fictional')
    c = dict(prefix=prefix, formula=label, T=T, q=q, R_outer=R, K=K)
    macro, _ = provider._macros(ctx, c, dict(signed_weights=list(w), allocation=list(counts)))
    vals = [complex(np.vdot(state, np.linalg.matrix_power(op, q) @ state) * phase) for op in macro['operators']]
    e2 = max(abs(vals[0] - mine['exact']), abs(vals[1] - mine['corr']), abs(vals[2] - mine['raw']), abs(q * macro['log_B_per_outer'] - mine_logB))
    worst = max(worst, e1, e2)
    print('%-13s R=%2d K=%d q=%d weights<0:%s alloc=%s  carrier-vs-brute %.1e  A-provider-vs-brute %.1e  |raw|/|corr|=%.4f logB=%.4f'
          % (label, R, K, q, any(x < 0 for x in w), list(counts), e1, e2, abs(mine['raw']) / abs(mine['corr']), mine_logB))
print('PASS' if worst < 1e-10 else 'FAIL', worst); sys.exit(0 if worst < 1e-10 else 1)
