#!/usr/bin/env python3
"""Fixed-U .1 diagnosis only, original same-array state checks, no downstream."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.diagnostic-packages')]
from run_research_c_h6_factored_diagnostic import save,sha


def main():
    import numpy as np
    from trottertracks import c_h6_conic_diagnostic as conic
    from trottertracks import c_h6_fit_conditioning as fit
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--arm',choices=['PR','ENTRY'],required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    old=args.parent/'artifacts/research_c_h6_independent_validation/2026-11-06-stablemetric/run01'
    taskfile=old/f'optimization/H6_100_C0_{args.arm}_g01/task.json'
    task=json.loads(taskfile.read_text())
    paths={k:args.parent/task[k] for k in ('input_path','cold_path')}
    parent_source=args.parent/'artifacts/research_c_h6_stablemetric/2026-11-06/execution_preflight/SOURCE.json'
    code=[Path(__file__),ROOT/'scripts/run_research_c_h6_factored_diagnostic.py',
        ROOT/'src/trottertracks/c_h6_conic_diagnostic.py',ROOT/'src/trottertracks/c_h6_factored_newton.py',
        ROOT/'src/trottertracks/c_h6_fit_conditioning.py']
    package=ROOT/'artifacts/research_c_h6_trustmodel/2026-11-06/technical/conic_package.json'
    identities={str(p):sha(p) for p in [*code,*paths.values(),taskfile,old/'STOP.json',parent_source,package]}
    save(args.out/'SOURCE.json',dict(files=identities,parent_SOURCE_sha256=sha(parent_source),backend=conic.VERSION))
    save(args.out/'scope.json',dict(system=task['system'],objective=task['objective_spec'],
        fixed_U='saved C0 exactly',arm=args.arm,maximum_blocks=3,maximum_iterations_per_block=1000,
        outer_updates=0,DF_norm=0,PF=0,native=0,retry=0,resume=0,downstream_eligible=False,
        purpose='limited conditioning/convergence diagnosis; cone global stage and identical Newton polish',
        original_objective_and_thresholds_unchanged=True))
    for name,p in paths.items():
        if sha(p)!=task[name.replace('_path','_sha256')]:raise ValueError('INPUT_IDENTITY')
    tick=time.monotonic();rows=[];state_gradients=[];failed=None
    try:
        with np.load(paths['input_path']) as d:g=d['g']
        with np.load(paths['cold_path']) as d:J,U=d['J'],d['U']
        cls=fit.near.near.pr.Objective if args.arm=='PR' else fit.near.near.Objective
        spec=cls(**task['objective_spec']);backend=fit.backend(spec)
        p=backend.Coefficients(g,U,spec)
        for block in range(3):
            chart=fit.Chart(p,J,fit.inner.DEFAULT)
            with (args.out/f'iterations_{block}.jsonl').open('x') as trace:
                def record(row,z):trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
                def endpoint(info,arrays):np.savez(args.out/f'raw_{block}.npz',**arrays)
                J,info=conic.solve_and_polish(chart,maxiter=1000,record=record,record_endpoint=endpoint)
            row=dict(block=block,solver=info,J_norm=float(np.linalg.norm(J)),J_max=float(np.max(abs(J))))
            rows.append(row)
            if info['polish'] is None or not all(x['PASS'] for x in info['polish']['consistency'].values()):
                raise ValueError('CONIC_POLISH_POINT_CONSISTENCY_MISMATCH')
            # Reuse the unchanged exact-array physical and binary64 checks,
            # including the independent AD/forward-pullback paths.
            state=backend.state(g,J,U,spec)
            row['state']=state[0];state_gradients.append(state[2])
            save(args.out/f'block_{block}.json',row)
        shift=float(np.linalg.norm(state_gradients[-1]-state_gradients[-2]))
        save(args.out/'result.json',dict(arm=args.arm,blocks=rows,latest_U_shift_l2=shift,
            U_stability_threshold=1e-6,latest_J_MET=rows[-1]['state']['J_stationarity']['status']=='MET',
            head=fit.head_report(p),downstream_eligible=False,wall_seconds=time.monotonic()-tick))
    except Exception as exc:
        failed=repr(exc)
        save(args.out/'failure.json',dict(error=failed,blocks=rows,wall_seconds=time.monotonic()-tick))
    for name,expected in identities.items():
        if sha(Path(name))!=expected:raise ValueError('SOURCE_OR_PROTECTED_INPUT_CHANGED')
    save(args.out/'STOP.json',dict(failed=failed is not None,error=failed,retry=0,resume=0,downstream_started=False))
    print(json.dumps(dict(arm=args.arm,failed=failed,blocks=len(rows),wall_seconds=time.monotonic()-tick)),flush=True)


if __name__=='__main__':main()
