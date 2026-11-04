"""Default: bytes/schema/literals ONLY. --run needs separate scientific permission."""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE),str(ROOT/'docs/research/rpe_controlled_connection_runner_20261104')]
import run_connection as base
from integration_audit import require, classify_axis
from connection_pipeline import STATUS, write_json
import sharpness_adapter as sharp


def static_inputs():
    """Read archived data, never decode basis balls or evaluate new H4 bounds."""
    spec = json.loads((HERE/'FIXED_BINDINGS.json').read_text())
    require(spec['precision_bits']==96 and spec['passes']==1
            and spec['bound_scope']=='IID_CATEGORICAL_MEAN_ONLY_NOT_POINTWISE','fixed mean/precision scope')
    for entry in spec['dependencies']:
        base.verify_bound(entry,current=True)
    original = base.static_inputs()
    validation = json.loads(base.verify_bound(spec['validation'],current=True))
    for item in validation['artifacts'].values():
        raw = (ROOT/item['path']).read_bytes()
        require(len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],
                'archived raw changed: '+item['path'])
    result = json.loads((ROOT/spec['result_path']).read_text())
    descriptors = json.loads((ROOT/spec['descriptors_path']).read_text())
    require(result['source']['commit']=='5605ea6fdb90fb7de2dac8768d392c81435217a3'
            and result['status']=='COMPLETED_MANDATORY_STOP'
            and result['connection_decision']=='UNDETERMINED'
            and result['lowering']['bound']=='0'
            and not result['pi64_added_again'],'old certificate scope/status')
    require(descriptors['support_policy']==original['support_policy']=='support_for_source_basis_run_le_1'
            and descriptors['training_fingerprint']==original['training_fingerprint'],
            'run_le_1 policy/training mismatch')
    old_ids = [x['id'] for x in result['components']]
    mapping_ids = [x['id'] for x in descriptors['components']]
    export_ids = [x['id'] for x in original['export']['components']]
    require(old_ids==mapping_ids==export_ids and len(set(old_ids))==218,'component/law order identity')
    require(len(descriptors['bases'])==229,'basis descriptor coverage')
    for row,old in zip(original['rows'],result['rounds'],strict=True):
        require(all(row[k]==old[k] for k in ('m','q','r','K')),'archived round join')
    component_law = json.loads(original['raw']['LAWS'])['component']
    require(component_law['grid_bits']==53 and len(component_law['grid_masses'])==218,
            'exact uniform53 component-law schema')
    parent = json.loads((ROOT/'docs/research/rpe_categorical_circuit_connection_20261104/FIXED_BINDINGS.json').read_text())
    order_masses = {g['r']:g['order_law']['grid_masses'] for g in parent['six_r_literal_joins']}
    require(set(order_masses)==set(original['theta_by_r']) and all(len(w)==2 for w in order_masses.values()),
            'K2 order-law coverage')
    colors = []
    for c in descriptors['components']:
        require(len(c['support']) in (1,2) and c['full'] in descriptors['bases']
                and c['selected'] in descriptors['bases'],'identity/unknown application')
        colors.append((c['full'],descriptors['bases'][c['full']]['basis_hash']))
    return dict(mode='FIXED_H4_WEIGHTED_BOUND_SCIENCE',original=original,result=result,colors=colors,
                probabilities=component_law['grid_masses'],
                errors=[c['upper'] for c in result['components']],order_masses=order_masses)


def describe():
    bundle = static_inputs()
    return {**STATUS,'status':'PREPARED_NOT_SCIENTIFICALLY_EXECUTED','new_H4_scientific_calls':0,
            'bound_scope':'IID_CATEGORICAL_MEAN_ONLY_NOT_POINTWISE','rounds':18,'axes':36,
            'archived_components':len(bundle['errors']),'method':'singleton_run_expected_telescoping',
            'old_sufficient_rounds':[0,1,2,3,4,5,7,8,9],
            'old_undetermined_rounds':[6,10,11,12,13,14,15,16,17]}


def evaluate(output, bundle, source=None):
    """Shared end-to-end path for synthetic input and future authorized H4 pass."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    record = {**STATUS,'status':'STARTED','decision':'UNDETERMINED','groups':[],'rounds':[],
              'counts':{},'source':source,'bound_scope':'IID_CATEGORICAL_MEAN_ONLY_NOT_POINTWISE',
              'mode':bundle.get('mode','ARTIFICIAL'),'resources':bundle.get('resources',{}),
              'pi64_added_again':False,'scientific_passes':1,'precision_bits':96}
    def checkpoint():
        write_json(output/'RESULT.json',record)
    checkpoint()
    started = time.monotonic()
    try:
        record['stage']='weighted_moments'
        record['counts']['moment_passes_started']=1
        moments = sharp.grouped_moments(bundle['probabilities'],bundle['errors'],bundle['colors'])
        record['moments']=[str(x) for x in moments]
        tail_bounds = {}
        from flint import ctx
        with ctx.workprec(96):
            for r,theta in sorted(bundle['original']['theta_by_r'].items()):
                record['counts']['sine_bounds_started']=record['counts'].get('sine_bounds_started',0)+2
                sine = [sharp.sine_upper(t) for t in theta]
                record['counts']['tail_bounds_started']=record['counts'].get('tail_bounds_started',0)+1
                tail = sharp.singleton_mean_tail(r,bundle['order_masses'][r],sine,moments)
                tail_bounds[r]=tail
                record['groups'].append(dict(r=r,sine_upper=[str(s) for s in sine],
                    order_masses=[str(w) for w in bundle['order_masses'][r]],mean_tail_upper=str(tail)))
                checkpoint()
        record['stage']='mean_transfers'
        original=bundle['original']
        require(len(original['rows'])==len(bundle['result']['rounds'])==18,'all18 required')
        for row,old in zip(original['rows'],bundle['result']['rounds'],strict=True):
            m,q,r = row['m'],row['q'],row['r']
            require(all(row[k]==old[k] for k in ('m','q','r','K')) and q==1<<m and row['K']==2,'tuple mismatch')
            record['counts']['round_transfers_started']=record['counts'].get('round_transfers_started',0)+1
            extra=sharp.mean_extra(q,tail_bounds[r],old['boundary_upper'],old['lowering_upper'],old['extra_upper'])
            require(extra is not None,'missing transfer evidence')
            axes={}
            for axis in ('cosine','sine'):
                record['counts']['strict_comparisons_started']=record['counts'].get('strict_comparisons_started',0)+1
                axes[axis]=classify_axis(original['saved_axes'][m][axis],row['axes'][axis]['epsilon'],extra)
            record['rounds'].append(dict(m=m,q=q,r=r,K=2,mean_extra_upper=str(extra),
                old_uniform_extra_upper=old['extra_upper'],boundary_upper=old['boundary_upper'],axes=axes))
            checkpoint()
        if all(a['decision']=='SUFFICIENT_CONDITION_CONFIRMED' for row in record['rounds'] for a in row['axes'].values()):
            record['decision']='CONDITIONAL_IDEAL_CONTROLLED_IID_MEAN_LOCAL_CONDITIONS_CONFIRMED'
        record['stage']='complete'
        record['status']='COMPLETED_MANDATORY_STOP'
    except BaseException as exc:
        record['status']='UNKNOWN_MANDATORY_STOP'
        record['error']=type(exc).__name__+': '+str(exc)
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        checkpoint()
    return record


def frozen_source():
    source=base.frozen_source()
    for path in [*HERE.glob('*.py'),HERE/'FIXED_BINDINGS.json',HERE/'SCIENTIFIC_EXECUTION_PLAN.md']:
        require(path.read_bytes()==base.git('show',source['commit']+':'+path.relative_to(ROOT).as_posix()),
                'freeze weighted SOURCE before --run')
    return source


def run_science(output):
    bundle=static_inputs()
    source=frozen_source()
    output=Path(output).resolve()
    namespace=ROOT/'artifacts/rpe_support_weighted_transfer'
    require(output.is_relative_to(namespace) and output!=namespace,'fresh weighted namespace required')
    # Scalar Fraction work shares a single O(218) aggregation; no dense/GPU workload.
    bundle['resources']=dict(affinity_cpus=len(os.sched_getaffinity(0)),
        available_memory=[line for line in Path('/proc/meminfo').read_text().splitlines()
                          if line.startswith('MemAvailable:')],
        parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        backend='one shared exact-rational aggregate and twelve scalar Arb sine bounds; no spawn/GPU')
    return evaluate(output,bundle,source)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    require(args.run==bool(args.output),'--output only with explicit --run')
    result=run_science(args.output) if args.run else describe()
    print(json.dumps(result if not args.run else {k:result[k] for k in
        (*STATUS,'status','decision','counts')},ensure_ascii=False,indent=2))
    return 0 if not args.run or result['decision'].startswith('CONDITIONAL_') else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as exc:
        print('UNKNOWN/STOP: '+str(exc),file=sys.stderr)
        raise SystemExit(2)
