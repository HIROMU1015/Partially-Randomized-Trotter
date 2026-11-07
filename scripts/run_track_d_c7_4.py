#!/usr/bin/env python3
"""C7-4 saved-only model fits and fixed artificial stability comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
OUTPUT='artifacts/track_d_c7_4_models/2026-11-07/run01'
EXEC=('scripts/run_track_d_c7_4.py','src/trottertracks/track_d_c7_4_models.py',
      'src/trottertracks/track_d_c7_4_stability.py','tests/check_track_d_c7_4.py',
      'docs/research/track_d_c7_4_model_stability_20261107/PLAN.md')


def initialize(origin):
    import run_track_d_h4_reference as boot
    boot.bootstrap(origin)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--origin',type=Path,required=True)
    parser.add_argument('--evidence-root',type=Path,default=ROOT.parent)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise ValueError('EXPLICIT_C7_4_ONLY')
    initialize(args.origin)
    from threadpoolctl import threadpool_limits
    from trottertracks import track_d_c7_4_models as m
    from trottertracks import track_d_c7_4_stability as stability
    import numpy as np
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source={}
    for path in EXEC:
        if (ROOT/path).read_bytes()!=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT):
            raise ValueError('EXECUTION_SOURCE_NOT_FIXED:'+path)
        source[path]=m.sha(ROOT/path)
    for name in ['src/trottertracks/c1_h4_native_fair.py','src/trottertracks/c1_h6_compiler_precision.py',
                 'src/trottertracks/c1_h6_native_precision.py']:
        if (args.origin/name).read_bytes()!=subprocess.check_output(['git','show','ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7:'+name],cwd=ROOT):
            raise ValueError('ORIGINAL_NUMERICAL_SOURCE:'+name)
    if importlib.metadata.version('qiskit')!='1.3.0':raise ValueError('P0_VERSION')
    out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=False)
    local=out/'local';local.mkdir();(local/'.gitignore').write_text('*\n')
    source_record=dict(execution_source=commit,base='acea43987edae0d5eb0c186a89aa8cd7139fbd7b',
        source_files=source,scope='saved88 reanalysis and fixed artificial only',
        new_molecular_sampling=0,new_molecular_build=0,new_molecular_compile=0,
        packages={p:importlib.metadata.version(p) for p in ('numpy','qiskit','qiskit-aer')})
    m.write(out/'SOURCE.json',source_record)
    state='STOP_FAILURE';phase='ingestion';complete=[]
    try:
        rows,features,checks,bindings,private=m.ingest(ROOT,args.evidence_root)
        m.write(out/'input_bindings.json',bindings)
        m.write(out/'saved_prices_scalar.json',rows);m.write(out/'event_features.json',features)
        with threadpool_limits(limits=1):
            y=m.matrix(rows,features);integration=m.integrate(rows,features,y)
            m.write(out/'integrated_statistics.json',integration)
        complete.append('88_saved_paths_integrated');print(complete[-1],flush=True)
        phase='exploratory_models'
        with threadpool_limits(limits=1):models=m.analyze_models(ROOT,features,y)
        for name,model in models.items():m.write(out/('model_'+name+'.json'),model)
        frozen=m.read(ROOT/'artifacts/track_d_c6a_paired_price/2026-11-06/run01/C6b_planning_proposal.json')
        c=frozen['conditional_parameter_covariance']
        # Reuse saved C6a coefficient covariance, including method/axis/metric
        # off-diagonal entries. Full original labels retained, not diagonalized.
        m.write(out/'frozen_C6a_future.json',dict(q=list(m.FUTURE),quantity_labels=m.LABELS,
            predictions=m.old_predictions(ROOT,[dict(q=q) for q in m.FUTURE]).tolist(),
            saved_parameter_covariance=c,shared_across_rounds=True,model_refit=False,
            covariance_rule='v(q,g)^T Cov(beta) v(q2,g2), with saved intercept/slope labels; no independent-round assumption',
            bias_bound=None,law_error_bound=None,qualified_coverage=False))
        complete.append('four_models_fitted_and_five_fold_CV');print(complete[-1],flush=True)
        phase='saved_numerical_recheck'
        trends=m.numerical_trends(checks);m.write(out/'numerical_residual_trends.json',trends)
        selected=trends['selected_worst'];directory=private[selected['sample_id']]
        selected_hashes={p.name:m.sha(p) for p in [directory/'references.npz',*directory.glob('*_observable.npz')]}
        # Verify selected originals against the pre-existing local-only index.
        old=ROOT/'artifacts'/m.RUNS[-1][0]/'payload_indices'/(selected['sample_id']+'.json')
        expected={r['path']:r['sha256'] for r in m.read(old)['members']}
        if any(expected[n]!=h for n,h in selected_hashes.items()):raise ValueError('SELECTED_SAVED_NUMERICAL_PAYLOAD_CHANGED')
        recheck=m.saved_recheck(directory,selected);recheck['original_payload_hashes']=selected_hashes
        m.write(out/'saved_numerical_recheck.json',recheck)
        if recheck['status']!='PASS':raise ValueError('SAVED_NUMERICAL_DEPENDENCY_STOP')
        complete.append('one_saved_worst_path_array_recheck');print(complete[-1],flush=True)
        phase='artificial_comparison'
        tasks=[(width,n) for width in (3,5) for n in (1024,8192,32768)]
        workers=min(len(tasks),len(os.sched_getaffinity(0)))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(args.origin,)) as pool:
            toys=list(pool.map(stability.toy_case,tasks))
        # Only these two short, wholly artificial P0 calls. No H4 QPY is loaded.
        compiled=[stability.short_compile(width) for width in (3,5)]
        m.write(out/'artificial_stability.json',dict(cases=toys,short_P0=compiled,
            long_cases=6,artificial_compile_calls=2,molecular_compile_calls=0,
            all_full_matrices=True,full17_operator=False,
            status='RECORDED_FIXED_COMPARISONS',existing_tolerances_unchanged=True,
            reference_is_exact_interval_oracle=False))
        complete.append('six_toy_length_comparisons_two_short_P0');print(complete[-1],flush=True)
        phase='source_preservation'
        if source!={p:m.sha(ROOT/p) for p in EXEC}:raise ValueError('C7_4_SOURCE_CHANGED')
        for p,d in bindings.items():
            if m.sha(ROOT/p)!=d['sha256']:raise ValueError('OLD_SCIENTIFIC_EVIDENCE_CHANGED:'+p)
        if selected_hashes!={p.name:m.sha(p) for p in [directory/'references.npz',*directory.glob('*_observable.npz')]}:
            raise ValueError('OLD_NUMERICAL_PAYLOAD_CHANGED')
        for r in rows:
            if r['method']==m.METHODS[0] and r['axis']=='X':
                event=private[r['sample_id']].parent/'events'/(r['sample_id']+'.json')
                if m.sha(event)!=r['event_sha256']:raise ValueError('OLD_EVENT_CHANGED:'+r['sample_id'])
        state='C7_4_ANALYSIS_COMPLETE_MANDATORY_STOP'
    except BaseException as error:
        # No automatic run retry, phase fitting, tolerance change or overwrite.
        message=str(error).replace(str(ROOT),'<task>').replace(str(args.evidence_root),'<evidence-root>')
        m.write(out/'failure.json',dict(phase=phase,type=type(error).__name__,error=message))
        raise
    finally:
        m.write(out/'result.json',dict(status=state,source_commit=commit,completed=complete,
            saved_whole_paths=88,new_H4_sampling=0,new_H4_compile=0,new_ground_solve=0,new_DF_fit=0,
            N_evaluated=False,RPE_total_evaluated=False,old_results_modified=False,
            statistical_qualification='EXPLORATORY_MODELS_UNQUALIFIED',next_stage_authorized=False,mandatory_stop=True))
        m.write(out/'STOP.json',dict(status=state,mandatory_stop=True,next_stage_authorized=False,
            next_actor='user/GPT',no_automatic_q64=True))


if __name__=='__main__':main()
