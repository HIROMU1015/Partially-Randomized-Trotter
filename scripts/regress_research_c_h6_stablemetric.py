"""Arithmetic regression at saved failed points; no optimizer or circuit call."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parent=args.parent_root.resolve()
    import numpy as np
    from trottertracks import c_h6_fit_conditioning as fix
    from trottertracks import c_near_stable as near
    run=parent/'artifacts/research_c_h6_independent_validation/2026-11-06-conditionfix/run01'
    with np.load(parent/'artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/input.npz') as stream:g=stream['g']
    with np.load(parent/'artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01/fit/final.npz') as stream:U=stream['U']
    rows=[]
    for folder in sorted((run/'optimization').iterdir()):
        task=json.loads((run/'execution_jobs'/(folder.name+'.json')).read_text())
        spec=(near.near.Objective if task['arm']=='ENTRY' else near.near.pr.Objective)(**task['objective_spec'])
        with (folder/'trajectory.jsonl').open() as stream:records=[json.loads(v) for v in stream]
        for old in records:
            if old.get('kind')!='inner_raw' or not old.get('array_path') or 'consistency' not in old:continue
            path=parent/old['array_path']
            with np.load(path) as stream:J,x=stream['J'],stream['x']
            p=fix.backend(spec).Coefficients(g,U,spec);chart=fix.Chart(p,J,fix.inner.DEFAULT)
            # Bind to the exact recorded x, rather than repacking J (one ulp).
            chart.x0=x.copy();chart._primal_key=None
            zero=np.zeros(chart.dimension)
            chart.origin_primal=chart._primal(zero);chart.origin_J=chart.origin_primal[1].astype(chart.ld)
            np.testing.assert_array_equal(chart.point(zero)[1],J)
            value,gradient=chart.vg(zero);independent,_,pullback=chart.independent(zero)
            grad_check=fix.inner.point.compare(gradient,pullback,atol=1e-8,rtol=1e-7)
            evaluator=near.parts_gradient_J if task['arm']=='ENTRY' else fix.inner.stable.precision.parts_gradient_J
            binary=evaluator(g,J,U,spec,extended=False)[0]['objective']
            value_check=fix.inner.point.compare(value,binary,atol=1e-10,rtol=1e-8)
            centered=chart.centered(zero)[0]
            assert grad_check['PASS'] and centered==0.
            rows.append(dict(id=folder.name,label=old['label'],raw_path=old['array_path'],
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),exact_saved_J_bound=True,
                old_gradient=old['consistency']['chart_vs_physical_gradient'],gradient=grad_check,
                value_check_retained=value_check,centered_origin=centered,
                metric_floor=p._h6_metric.scaling_floor,limited_directions=p._h6_metric.limited_directions))
    result=dict(scope='saved exact x/J arithmetic regression only; no new scientific optimization or condition',
        backend=fix.VERSION,points=len(rows),gradient_PASS=sum(v['gradient']['PASS'] for v in rows),
        all_acceptance_checks_PASS=all(v['gradient']['PASS'] and v['value_check_retained']['PASS'] for v in rows),
        caveat='Value-inconsistent old candidates remain rejected; gradient repair is not endpoint admission or convergence evidence',rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
