#!/usr/bin/env python3
"""Limited fixed-U diagnosis, separate from the six-trajectory experiment.

Uses saved objective definitions verbatim. No physical/PF/native callbacks.
SciPy's private subproblem hook is used only to observe its existing algorithm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(chart, iterations):
    import numpy as np
    from scipy.optimize._trustregion import _minimize_trust_region
    from scipy.optimize._trustregion_exact import IterativeSubproblem
    zero = np.zeros(chart.dimension)
    _, grad = chart.vg(zero)
    H = chart.hess(zero)
    directions = {'gradient': -grad / np.linalg.norm(grad)}
    rng = np.random.default_rng(710)
    d = rng.normal(size=chart.dimension)
    directions['random'] = d / np.linalg.norm(d)
    fd = []
    for name, d in directions.items():
        # Measure local curvature with a step that keeps the penalty's narrow
        # smoothing transitions resolved; larger steps reveal nonlocal change.
        for h in (1e-4, 1e-6, 1e-8, 1e-10):
            a, ga = chart.centered(h*d)
            b, gb = chart.centered(-h*d)
            observed = (ga-gb)/(2*h)
            expected = H@d
            fd.append(dict(direction=name, step=h,
                first_derivative=(a-b)/(2*h), predicted_first=float(grad@d),
                Hessian_relative_l2=float(np.linalg.norm(observed-expected)/max(1., np.linalg.norm(expected))),
                curvature=float(d@expected)))
    trace = []
    class Observed(IterativeSubproblem):
        def solve(self, radius):
            step, boundary = super().solve(radius)
            predicted = -float(self.jac@step + .5*step@self.hess@step)
            actual = float(self.fun-chart.centered(self._x+step)[0])
            trace.append(dict(radius=float(radius), step_l2=float(np.linalg.norm(step)),
                boundary=bool(boundary), predicted=predicted, actual=actual,
                rho=actual/predicted if predicted else None,
                coordinate_gradient_l2=float(np.linalg.norm(self.jac))))
            return step, boundary
    tick = time.monotonic()
    result = _minimize_trust_region(lambda z: chart.centered(z)[0], zero,
        jac=lambda z: chart.centered(z)[1], hess=chart.hess,
        subproblem=Observed, maxiter=iterations, gtol=1e-13, inexact=False)
    value, physical, pullback = chart.independent(result.x)
    return dict(finite_differences=fd, trace=trace, nit=int(result.nit),
        status=int(result.status), message=str(result.message),
        objective=value, physical_gradient_infinity=float(np.max(abs(physical))),
        coordinate_gradient_l2=float(np.linalg.norm(pullback)),
        wall_seconds=time.monotonic()-tick), result.x


def main():
    import numpy as np
    from trottertracks import c_h6_fit_conditioning as fit
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=['PR', 'ENTRY'], required=True)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = args.parent/'artifacts/research_c_h6_independent_validation/2026-11-06-stablemetric/run01'
    taskfile = old/f'optimization/H6_100_C0_{args.arm}_g01/task.json'
    task = json.loads(taskfile.read_text())
    inputs = {name: args.parent/task[name] for name in ('input_path', 'cold_path')}
    source = dict(scope='fixed U, strength .1, C0 and saved final point, PR/ENTRY; diagnosis only',
        parent=str(args.parent), arm=args.arm, iterations=args.iterations,
        files={str(p): sha(p) for p in [Path(__file__), ROOT/'src/trottertracks/c_h6_fit_conditioning.py', taskfile, *inputs.values()]})
    (args.out/'SOURCE.json').write_text(json.dumps(source, indent=2)+'\n')
    for name, path in inputs.items():
        if sha(path) != task[name.replace('_path', '_sha256')]:
            raise ValueError('INPUT_CHANGED')
    with np.load(inputs['input_path']) as packet: g=packet['g']
    with np.load(inputs['cold_path']) as packet: J,U=packet['J'],packet['U']
    cls = fit.near.near.pr.Objective if args.arm=='PR' else fit.near.near.Objective
    spec = cls(**task['objective_spec'])
    p = fit.backend(spec).Coefficients(g,U,spec)
    records = {}
    for label, j in [('C0', J), ('saved_final', np.load(old/f'optimization/H6_100_C0_{args.arm}_g01/raw/000008.npz')['J'])]:
        chart = fit.Chart(p,j,fit.inner.DEFAULT)
        row,z = diagnose(chart,args.iterations if label=='C0' else 1)
        records[label]=row
        np.savez(args.out/(label+'.npz'), z=z, J=chart.point(z)[1])
    (args.out/'diagnosis.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (args.out/'STOP.json').write_text(json.dumps(dict(completed=True,retry=0,resume=0,scientific_trajectory=False))+'\n')
    print(json.dumps({label:{k:v for k,v in row.items() if k not in ('finite_differences','trace')} for label,row in records.items()}),flush=True)


if __name__=='__main__':
    main()
