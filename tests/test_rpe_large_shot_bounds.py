from fractions import Fraction
import importlib.util
from itertools import product
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'src/trottertracks/rpe_large_shot_bounds.py'
FIXED = 'cbc471c913072de45c25a0ed22d51dff8a9b7034'
FIXTURE = Path(os.environ.get('RPE_LARGE_SHOT_FIXED_VIEW',
               ROOT.parent/'rpe-distributed-q3-local'/'.local-q3'/'views'/FIXED))


@unittest.skipUnless((FIXTURE/'src/trottertracks/rpe_total_risk.py').is_file(),
                     'requires the fixed Z1+T1 source view')
class LargeShotTests(unittest.TestCase):
    def test_exact_and_large_artificial_cases_in_fixed_runtime(self):
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--fixed-fixture'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def exercise_fixed_runtime():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FIXTURE, text=True).strip() == FIXED
    assert subprocess.run(['git','diff','--quiet','HEAD','--','src','scripts'],cwd=FIXTURE).returncode == 0
    sys.path.insert(0, str(FIXTURE/'src'))
    from trottertracks import rpe_deterministic_risk as risk, rpe_total_risk as total

    spec = importlib.util.spec_from_file_location('_large_shot_test', MODULE)
    bounds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bounds)
    probability_rule = risk.EXACT_AFFINE
    threshold = Fraction('0.00159360010199040')**2

    def outcome(rows, target=0.0, **limits):
        return bounds.evaluate(rows, h=0.1, target=target, probability_rule=probability_rule,
                               threshold=threshold, risk=risk, total=total, **limits)

    def exhaustive(rows, target):
        axes = []
        for row in rows:
            for axis, component in (('X', 'real'), ('Y', 'imag')):
                table = risk.binomial_axis(row['raw'][component], row['N_'+axis],
                                           probability_rule=probability_rule)
                axes.append([(risk.S.count_mean(k, row['N_'+axis])['mean'], Fraction(w, table['denominator']))
                             for k, w in zip(table['counts'], table['numerators'])])
        value = Fraction(0)
        for history in product(*axes):
            xy = [(history[m][0], history[m+1][0]) for m in range(0, len(history), 2)]
            result = total.T.decode_supplied_coordinates_total(xy, h=0.1)['energy_estimate_ha']
            value += math.prod(weight for _, weight in history)*(Fraction(result)-Fraction(target))**2
        return value

    # Near the origin, near a branch boundary, phase wrapping and zero-hold.
    for raw, target in [((0.,0.), 0.), ((-0.94,0.), -0.1), ((0.6,0.35), 0.2)]:
        rows = [dict(m=m,N_X=2,N_Y=2,raw=dict(real=raw[0],imag=raw[1])) for m in range(3)]
        reference = exhaustive(rows, target)
        for cells in (1, 8, 32):
            result = outcome(rows,target,maximum_round_cells=cells)
            lower, upper = (bounds._decode(result[k]) for k in ('MSE_lower','MSE_upper'))
            assert lower <= reference <= upper, (raw, cells, float(reference), float(lower), float(upper))
            assert (result['status'] != 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' or reference <= threshold)
            assert (result['status'] != 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' or reference > threshold)

    shots = [16*(11+4*(12-m)) for m in range(13)]
    rows = [dict(m=m,N_X=n,N_Y=n,raw=dict(real=1.0,imag=0.0)) for m,n in enumerate(shots)]
    assert shots[0] == 944
    good = outcome(rows)
    bad = outcome(rows, target=0.05)
    assert good['status'] == 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', good['status']
    assert bad['status'] == 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD', bad['status']
    assert bounds._decode(good['residual_probability']) < Fraction(1, 10**10)


if __name__ == '__main__' and sys.argv[1:] == ['--fixed-fixture']:
    exercise_fixed_runtime()
