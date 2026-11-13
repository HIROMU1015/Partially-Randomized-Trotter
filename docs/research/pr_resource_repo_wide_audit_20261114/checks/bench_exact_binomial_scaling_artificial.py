"""Audit microbenchmark (artificial raw mean): scaling of the exact-rational binomial table and of the
per-round count enumeration in rpe_deterministic_risk with the shot number N (X and Y equal)."""
import sys, time, tracemalloc
sys.path.insert(0, sys.argv[1] + '/src')
from trottertracks import rpe_deterministic_risk as R
raw = 0.8565499611146139          # fictional binary64 coordinate
print('N  binomial_axis: seconds, peak MB, bits of denominator | round_distribution(N,N): seconds, cells, peak MB')
for n in (12, 64, 256, 1024, 4096):
    tracemalloc.start(); t = time.perf_counter()
    axis = R.binomial_axis(raw, n, probability_rule=R.EXACT_AFFINE); dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    line = '%5d  %8.3f s %9.1f MB %9d bits' % (n, dt, peak, axis['denominator'].bit_length())
    if n <= 256:
        tracemalloc.start(); t = time.perf_counter()
        table = R.round_distribution(dict(m=3, raw=dict(real=raw, imag=-0.31), N_X=n, N_Y=n), h=.1, target_energy=-2.1,
                                     radius=1.0, probability_rule=R.EXACT_AFFINE, maximum_outcomes=10 ** 9)
        dt = time.perf_counter() - t; peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
        line += ' | %8.3f s %8d cells %9.1f MB' % (dt, table['cells'], peak)
    print(line)
