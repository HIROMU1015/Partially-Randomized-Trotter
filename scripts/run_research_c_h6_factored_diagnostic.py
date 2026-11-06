#!/usr/bin/env python3
"""One fixed-U diagnostic block per arm; no outer trajectory or downstream."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    import numpy as np
    import scipy
    from trottertracks import c_h6_fit_conditioning as fit
    from trottertracks import c_h6_factored_newton as newton
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=['PR', 'ENTRY'], required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = args.parent/'artifacts/research_c_h6_independent_validation/2026-11-06-stablemetric/run01'
    taskfile = old/f'optimization/H6_100_C0_{args.arm}_g01/task.json'
    task = json.loads(taskfile.read_text())
    paths = {name:args.parent/task[name] for name in ('input_path', 'cold_path')}
    parent_source = args.parent/'artifacts/research_c_h6_stablemetric/2026-11-06/execution_preflight/SOURCE.json'
    protected = [old/'STOP.json', parent_source]
    code = [Path(__file__), ROOT/'src/trottertracks/c_h6_factored_newton.py', ROOT/'src/trottertracks/c_h6_fit_conditioning.py']
    identities = {str(p):sha(p) for p in [*code, taskfile, *paths.values(), *protected]}
    scope = dict(system=task['system'], objective=task['objective_spec'],
        arm=args.arm, fixed_U='saved C0 exactly', blocks=1, maxiter=1000,
        outer_updates=0, PF=0, native=0, downstream_eligibility=False,
        retry=0, resume=0, purpose='limited convergence and arithmetic diagnosis',
        inherited_scientific_contract='research_c_h6_stablemetric_contract_20261106.json',
        change='factored Newton with Armijo instead of trust-exact; identical F/epsilon/gates',
        original_full_run_not_resumed=True)
    save(args.out/'scope.json', scope)
    save(args.out/'SOURCE.json', dict(parent_SOURCE_sha256=sha(parent_source),
        files=identities, numerical_backend=newton.VERSION,
        numpy=np.__version__, scipy=scipy.__version__, python=sys.version))
    for name,p in paths.items():
        if sha(p)!=task[name.replace('_path','_sha256')]: raise ValueError('INPUT_IDENTITY')
    tick=time.monotonic()
    try:
        with np.load(paths['input_path']) as packet:g=packet['g']
        with np.load(paths['cold_path']) as packet:J,U=packet['J'],packet['U']
        cls=fit.near.near.pr.Objective if args.arm=='PR' else fit.near.near.Objective
        spec=cls(**task['objective_spec'])
        p=fit.backend(spec).Coefficients(g,U,spec)
        chart=fit.Chart(p,J,fit.inner.DEFAULT)
        with (args.out/'iterations.jsonl').open('x') as trace:
            def record(row,z):
                trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
            result=newton.solve(chart,maxiter=1000,record=record)
        value,gx,pullback=chart.independent(result.x)
        endpoint=chart.point(result.x)[1]
        binary=(fit.near.parts_gradient_J if args.arm=='ENTRY' else fit.inner.stable.precision.parts_gradient_J)(g,endpoint,U,spec,extended=False)
        policy=fit.inner.DEFAULT;compare=fit.inner.point.compare
        checks=dict(chart_gradient=compare(result.jac,pullback,atol=policy.gradient_atol,rtol=policy.gradient_rtol),
            binary64_value=compare(value,binary[0]['objective'],atol=policy.value_atol,rtol=policy.value_rtol),
            centered_value=compare(result.fun+float(chart.origin_primal[-1]),value,atol=policy.value_atol,rtol=policy.value_rtol))
        stationarity=fit.inner.stable.old.gradient_gate(max(float(np.max(abs(gx))),float(np.max(abs(binary[1])))),policy.j_gradient_infinity,policy)
        row=dict(arm=args.arm,method=newton.VERSION,status=int(result.status),
            message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),
            objective=value,J_gradient_infinity=float(np.max(abs(gx))),
            binary64_J_gradient_infinity=float(np.max(abs(binary[1]))),
            J_norm=float(np.linalg.norm(endpoint)),J_max=float(np.max(abs(endpoint))),
            consistency=checks,stationarity=stationarity,
            objective_unchanged=True,thresholds_changed=False,
            directions_dropped=0,downstream_eligible=False,
            U_stability_not_evaluated=True,wall_seconds=time.monotonic()-tick)
        np.savez(args.out/'endpoint.npz',J=endpoint,U=U,z=result.x,solver_jac=result.jac)
        save(args.out/'result.json',row)
        for path,expected in identities.items():
            if sha(Path(path))!=expected:raise ValueError('DIAGNOSTIC_SOURCE_OR_PROTECTED_INPUT_CHANGED')
        save(args.out/'STOP.json',dict(completed=True,failed=False,retry=0,resume=0,downstream_started=False))
        print(json.dumps(row,allow_nan=False),flush=True)
    except Exception as exc:
        save(args.out/'STOP.json',dict(completed=False,failed=True,error=repr(exc),retry=0,resume=0))
        raise


if __name__=='__main__':main()
