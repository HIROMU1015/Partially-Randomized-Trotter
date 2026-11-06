"""Report-only six-metric aggregation of an already completed fixed C0 pilot.

No operator, circuit, simulator, sampler, shot calculator or new CI is called.
"""
import json
import hashlib
import math
from pathlib import Path

METRICS=('cx_count','rz_count','cx_depth','rz_depth','total_depth','circuit_size')
ARMS=('P','DF','strong_DF_PF')


def summarize(payload):
    empty=dict(status='UNASSESSED',total_mean_cost=None,
               price_confidence_interval=None,guarantee_status='not_certified')
    if payload.get('status')!='C0_CONDITIONAL_PILOT_COMPLETE':
        return empty|dict(reason='Scientific pilot has not completed all qualifications')
    rows=payload['priced_rounds']
    wanted={(arm,m) for arm in ARMS for m in range(7)}
    actual=[(row['arm'],row['round_index']) for row in rows]
    if len(actual)!=21 or set(actual)!=wanted:
        return empty|dict(reason='Missing or duplicate arm/round price')
    if any(not row['account']['feasible'] for row in rows):
        return empty|dict(reason='Infeasible round')
    if any(row['arm']!='strong_DF_PF' and max(row['CX_relative_sample_SE'].values())>.1 for row in rows):
        return empty|dict(reason='C0 price precision not qualified')
    totals={}
    for arm in ARMS:
        totals[arm]={}
        for metric in METRICS:
            round_costs=[]
            for row in rows:
                if row['arm']!=arm: continue
                values=[]
                for axis in ('cosine','sine'):
                    N=row['account']['shots'][axis]
                    C=row['cost'][axis+'_expected_cost'][metric]
                    if type(N) is not int or N<=0 or not math.isfinite(C) or C<0:
                        raise ValueError('Valid recorded N and mean price required')
                    values.append(N*C)
                round_costs.append(math.fsum(values))
            totals[arm][metric]=math.fsum(round_costs)
    return dict(status='CONDITIONAL_MEAN_ACCOUNTING_ONLY',total_mean_cost=totals,
        guarantee_status='not_certified',price_confidence_interval=None,
        same_recorded_N_for_all_metrics=True,classical_MC_count_is_not_a_multiplier=True,
        depth_metrics='sum of shot-weighted per-shot depths; not hardware wall-clock time',
        sample_SE='recorded per arm/round/axis/metric in original result; no total CI',
        scientific_run_provenance=payload['provenance'],state_preparation_included=False)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=summarize(json.loads(args.result.read_text()))
    value['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


if __name__=='__main__': main()
