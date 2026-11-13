"""Run from the root of a B 3216d27a-based tree: python -B <this file>.
Artificial benchmark: existing vectorized RPE branch reconstruction vs an O(1)-per-round closed form.

Random synthetic phases only (no physical signal, shots or science). Measures time and the
largest temporary array of the existing implementation, and agreement of selected branches.
"""
import math, sys, time
import numpy as np
sys.path.insert(0, 'src')
from trotterlib.rpe_four_round_phase_validation import _vectorized_reconstruction, _wrap_phase

def closed_form(phases, q_values):
    est = np.asarray(_wrap_phase(phases[0]), dtype=np.float64); sel = [np.zeros(est.shape, np.int64)]
    for phase, q in zip(phases[1:], q_values[1:]):
        p = np.asarray(_wrap_phase(phase), dtype=np.float64)
        k0 = np.floor((q * est - p) / (2 * math.pi)).astype(np.int64)
        best = None
        for dk in (-1, 0, 1, 2):            # nearest of equally spaced candidates lies in this window
            k = np.mod(k0 + dk, q)
            cand = _wrap_phase((p + 2 * math.pi * k) / q)
            dist = np.abs(np.angle(np.exp(1j * (cand - est))))
            if best is None:
                best = (dist, k, cand)
            else:
                better = (dist < best[0]) | ((dist == best[0]) & (k < best[1]))
                best = (np.where(better, dist, best[0]), np.where(better, k, best[1]), np.where(better, cand, best[2]))
        est = best[2]; sel.append(best[1])
    return est, sel

rng = np.random.default_rng(12345)
for M, reps in ((8, 2000), (12, 2000)):
    q = [1 << m for m in range(M + 1)]
    truth = rng.uniform(-math.pi, math.pi)
    phases = [_wrap_phase(qq * truth + rng.normal(0, 0.05, reps)) for qq in q]
    t = time.perf_counter(); e1, s1, amb = _vectorized_reconstruction(phases, q); t1 = time.perf_counter() - t
    t = time.perf_counter(); e2, s2 = closed_form(phases, q); t2 = time.perf_counter() - t
    agree = all(np.array_equal(a, b) for a, b in zip(s1, s2)) and np.array_equal(e1, e2)
    temp_bytes = reps * q[-1] * 16
    print(dict(M=M, reps=reps, existing_s=round(t1, 4), closed_form_s=round(t2, 4), speedup=round(t1 / t2, 1),
               branches_and_estimates_identical=bool(agree), ambiguous=int(amb.sum()),
               existing_largest_complex_temp_MB=round(temp_bytes / 2**20, 1)))
