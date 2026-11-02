"""Recover secondary metrics from accepted saved costs, without a new compile."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import h46_long_time_signal as study

METRICS=('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')


def run():
    saved=study.read(study.PRIVATE/'saved_prices.json')
    selected={r['id']:r for r in study.candidates() if r['id'] in saved}
    output=study.PUBLIC/'saved_secondary_metrics.json'
    result=study.read(output) if output.exists() else {}
    def accept(identity,metrics,origin,n=1):
        old=saved[identity]
        for axis in ('cosine','sine'):
            expected=old['unit_statistics'][axis]['mean'] if 'unit_statistics' in old else old[axis]
            if abs(metrics[axis]['rz_count']-expected)>1e-10:
                raise ValueError('SAVED_SECONDARY_METRICS_RZ_COHORT_MISMATCH')
        result[identity]=dict(mean_metrics=metrics,origin=origin,n=n,new_compile=0)
    for identity in selected:
        if identity in result:
            continue
        axes={}
        for proof in saved[identity].get('proof',[]):
            if proof.get('common_native_identity'):
                for axis in proof.get('axes',[]):
                    axes[axis['axis']]=axis['metrics']
        if set(axes)=={'cosine','sine'}:
            accept(identity,axes,'SAVED_COMMON_NATIVE_ACCEPTED_PAIRED_PROOF')
    for system in study.PREFIX:
        folder=study.ROOT/f'artifacts/resource_applicability/track_a_pf_common_{system}_results/2026-10-22'
        for record in study.read(folder/'native_results.json.gz'):
            row=dict(record['candidate'],T=.8,R_outer=None,K=None);identity=study.cid(row)
            if identity in selected and identity not in result:
                accept(identity,{a:record['prices'][a]['metrics'] for a in ('cosine','sine')},
                       str((folder/'native_results.json.gz').relative_to(study.ROOT)))
    path=study.ROOT/'artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/measured_gate_costs_96.csv'
    pairs={}
    for record in csv.DictReader(path.open()):
        row=dict(system=record['system'],method=record['method'],formula=record['formula'],prefix=int(record['prefix']),
                 q=int(record['q']),T=.8,R_outer=None,K=None);identity=study.cid(row)
        if identity in selected and identity not in result:
            pairs.setdefault(identity,{})[record['axis']]={m:int(record[m]) for m in METRICS}
    for identity,metrics in pairs.items():
        if set(metrics)=={'cosine','sine'}:
            accept(identity,metrics,str(path.relative_to(study.ROOT)))
    for system in study.PREFIX:
        cfg=study.read(study.ORIGIN/(system+'_PR_EXPLORATION/config_APPROVED_PRIVATE.json'))
        for identity,row in selected.items():
            if identity in result or row['method']!='PR' or row['system']!=system:
                continue
            original_id=saved[identity]['origin'].rsplit('#',1)[-1]
            tasks=[t for t in cfg['tasks'] if t['candidate']['id']==original_id]
            expected=saved[identity]['unit_statistics']['cosine']['n']
            if len(tasks)!=expected:
                raise ValueError('SAVED_PR_SECONDARY_METRICS_N_MISMATCH')
            total={a:{m:0 for m in METRICS} for a in ('cosine','sine')}
            for task in tasks:
                value=study.read(Path(cfg['output'])/task['id']/'result.json')
                if not value['native_scientific_PASS'] or value['conditional_sampling'] or value['task']!=task:
                    raise ValueError('UNACCEPTED_HISTORICAL_PR_METRICS')
                for axis in value['axis_costs']:
                    for metric in METRICS:
                        total[axis['axis']][metric]+=axis['metrics'][metric]
            metrics={a:{m:float(Fraction(v,expected)) for m,v in metrics.items()} for a,metrics in total.items()}
            accept(identity,metrics,'ACCEPTED_HISTORICAL_EXPLORATION_METADATA_SAME_COHORT',expected)
    study.save(study.PUBLIC/'saved_secondary_metrics.json',result)
    return dict(saved_selected=len(selected),secondary_metrics_recovered=len(result),new_compile=0,
                remaining_missing=sorted(set(selected)-set(result)))


if __name__=='__main__':
    print(json.dumps(run()))
