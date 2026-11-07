#!/usr/bin/env python3
"""B6b metadata preparation and at most16 artificial wrapper P0 calls.

No command in this runner aggregates H4 N*cost, samples H4 or recomputes N.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
OUTPUT='artifacts/track_d_b6b_bc/2026-11-08/run01'
EXEC=('scripts/run_track_d_b6b.py','src/trottertracks/track_d_b6b_bc.py','src/trottertracks/track_d_b6b_toy.py',
      'tests/check_track_d_b6b.py','docs/research/track_d_b6b_bc_integration_20261108/PLAN.md',
      'docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md')


def initialize(origin):
    import run_track_d_h4_reference as boot
    boot.bootstrap(origin)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def optional_c74(bc,bindings):
    base='artifacts/track_d_c7_4_models/2026-11-07/run02/'
    packets={n:json.loads(bc.blob(ROOT,bc.C74,base+n,bindings)) for n in ('SOURCE.json','result.json','STOP.json','input_bindings.json')}
    if packets['result.json']['status']!='C7_4_ANALYSIS_COMPLETE_MANDATORY_STOP' or not packets['STOP.json']['mandatory_stop']:
        raise ValueError('C7_4_NOT_PUBLISHED_COMPLETE')
    for q,n,commit,directory in bc.PRICE_SETS:
        path='artifacts/'+directory+'/prices.json'
        used=packets['input_bindings.json'][path]
        if used['commit']!=commit or used['sha256']!=bindings[commit+':'+path]['sha256']:
            raise ValueError('C7_4_PRICE_INPUT_BINDING')
    models=[]
    for name in ('affine','sqrt_lane','basis_runs','lane_runs'):
        p=base+'model_'+name+'.json';d=json.loads(bc.blob(ROOT,bc.C74,p,bindings))
        models.append(dict(name=name,parameters=d['parameters'],source_path=p,
            future_numeric_available=d['future'].get('predictions') is not None,
            independent_validation=d['independent_validation'],qualified_coverage=d['qualified_coverage']))
    return dict(commit=bc.C74,execution_source=packets['SOURCE.json']['execution_source'],models=models,
        classification='EXPLORATORY_PROXY_ONLY',adopted_for_H4_total=False,
        scope='saved q2..32 fit; q64..4096 model extrapolation only; no direct long-q price validation',
        coefficients_read_only=True,models_refit=False,C7_4_worktree_accessed=False)


def compile_task(task):
    name,method,axis,eligible,local=task;local=Path(local)
    from threadpoolctl import threadpool_limits
    from trottertracks import track_d_b6b_toy as toy
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    import numpy as np
    base=toy.old.c3.base;stem=name+'_'+method+'_'+axis
    os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/stem)
    raw=base.load_qpy(local/(stem+'_raw.qpy'))
    with np.load(local/(name+'_target.npz'),allow_pickle=False) as z:v=z['V']
    with threadpool_limits(limits=1):
        compiled=compile_checked(raw,CompilerProfile())
        base.save_qpy(local/(stem+'_compiled.qpy'),compiled)
        report=toy.postcheck(compiled,raw,v,axis,method,eligible)
    report.update(case=name,method=method,axis=axis,full_wrapper_compile_calls=1,
        compiler=asdict(CompilerProfile()),artificial_only=True,H4_cost_comparison=False)
    write(local.parent/'postchecks'/(stem+'.json'),report)
    return stem


def main():
    p=argparse.ArgumentParser();p.add_argument('--origin',type=Path,required=True)
    p.add_argument('--include-published-c74',action='store_true');p.add_argument('--execute-toy',action='store_true')
    args=p.parse_args();initialize(args.origin)
    from trottertracks import track_d_b6b_bc as bc
    from trottertracks import track_d_b6b_toy as toy
    from threadpoolctl import threadpool_limits
    import run_track_d_h4_reference as boot
    import numpy as np
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={n:digest(ROOT/n) for n in EXEC}
    for n in EXEC:
        if (ROOT/n).read_bytes()!=subprocess.check_output(['git','show',source+':'+n],cwd=ROOT):raise ValueError('UNFIXED_B6B_SOURCE:'+n)
    if importlib.metadata.version('qiskit')!='1.3.0':raise ValueError('P0_VERSION')
    reused=('src/trottertracks/track_d_c7_2_toy.py','src/trottertracks/track_d_c3_whole_path.py',
        'src/trottertracks/track_d_h4_reference.py','src/trottertracks/c1_stage3_accuracy.py',
        'src/trotterlib/rpe_resource_accounting.py','scripts/native_df_qiskit_validation_support.py',
        'docs/research/track_d_c7_2_cx_structure_20261107/CASES.json')
    reused_hashes={n:digest(ROOT/n) for n in reused}
    for n in reused:
        if (ROOT/n).read_bytes()!=bc.blob(ROOT,source,n):raise ValueError('UNFIXED_REUSED_SOURCE:'+n)
    origin_files=('src/trottertracks/c1_h4_native_fair.py','src/trottertracks/c1_h6_compiler_precision.py',
        'src/trottertracks/c1_h6_native_precision.py','src/trottertracks/c1_h4_strong_baseline.py')
    origin_hashes={n:digest(args.origin/n) for n in origin_files}
    for n in origin_files:
        if (args.origin/n).read_bytes()!=bc.blob(ROOT,'ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7',n):raise ValueError('ORIGIN_SOURCE_BYTES:'+n)
    # Actual selected helper, not just a similarly named file in another tree.
    original=bc.blob(ROOT,bc.B4,'src/trottertracks/c1_stage3_accuracy.py').decode()
    extract=lambda s:ast.dump(next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='aggregate_totals'),include_attributes=False)
    if extract(original)!=extract(inspect.getsource(bc.accounting.aggregate_totals)):raise ValueError('ACCOUNTING_REUSE_MISMATCH')
    out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=False);local=out/'local';local.mkdir();(local/'.gitignore').write_text('*\n')
    (out/'postchecks').mkdir();bindings={};finished=[];status='STOP_FAILURE';compile_names=[]
    write(out/'SOURCE.json',dict(execution_source=source,base='acea43987edae0d5eb0c186a89aa8cd7139fbd7b',files=hashes,
        origin_science_source='ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7',B4=bc.B4,B5=bc.B5,
        reused_files=reused_hashes,origin_files=origin_hashes,
        reused_aggregate_function_AST_equals_B4=True,selected_accounting_source_sha256=hashlib.sha256(inspect.getsource(bc.accounting.aggregate_totals).encode()).hexdigest(),
        scope='metadata/derivation, fixed artificial full-wrapper P0, synthetic accounting only',
        packages={p:importlib.metadata.version(p) for p in ('numpy','qiskit','qiskit-aer')}))
    try:
        handoff,summary=bc.read_b4(ROOT,bindings);catalog=bc.price_catalog(ROOT,bindings)
        bc.blob(ROOT,'e9d4e2f602d5f9e014507ca7e391719bab017b88',
            'artifacts/track_d_c6a_paired_price/2026-11-06/run01/exploratory_affine.json',bindings)
        bc.blob(ROOT,bc.B5,'docs/research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md',bindings)
        for n in ('src/trottertracks/track_d_original_h_accuracy.py','src/trottertracks/c1_stage3_accuracy.py',
                  'src/trotterlib/rpe_resource_accounting.py'):
            bc.blob(ROOT,bc.B4,n,bindings)
        c74=optional_c74(bc,bindings) if args.include_published_c74 else None
        write(out/'input_bindings.json',bindings);write(out/'B4_handoff_metadata.json',summary)
        write(out/'price_catalog_metadata.json',catalog);write(out/'coverage13.json',bc.coverage(handoff,catalog,c74_metadata=c74))
        if c74:write(out/'C7_4_version_metadata.json',c74)
        premises={k:True for k in ('same_V','actual_adjoint','vacuum_eigenray_with_chi','same_initial_system_state',
            'same_whole_path_law','same_XY_convention','fresh_independent_axis_trials','same_target_decoder','law_is_B4_ideal_product')}
        assessment=dict(premises=premises,premises_are_declared_ideal_conditions_not_actual_evidence=True,
            ideal=bc.common_N(premises),actual=bc.common_N(premises,lane='actual'),
            stored_B4_N_common_field_unchanged=handoff['Reference_symmetric_N_common_qualified'],
            B4_N_recomputed=False,actual_N=None)
        try:bc.aggregate_registered(handoff,[],premises)
        except ValueError as e:assessment['empty_H4_price_table_rejected']=str(e)
        else:raise ValueError('MISSING_PRICE_GUARD_FAILED')
        write(out/'common_N_conditions.json',assessment);finished.append('B_C_metadata_and_coverage_only')
        print(finished[-1],flush=True)
        if not args.execute_toy:raise ValueError('EXPLICIT_TOY_EXECUTION_REQUIRED')
        spec=json.loads((ROOT/'docs/research/track_d_c7_2_cx_structure_20261107/CASES.json').read_text())
        support=boot.load_file('b6b_exterior_fixture',ROOT/'scripts/native_df_qiskit_validation_support.py')
        with threadpool_limits(limits=1):cases,initial=toy.build_cases(spec,support.exterior_fock_lift)
        raw=[];tasks=[]
        with threadpool_limits(limits=1):
            for case in cases:
                wrappers,report=toy.check_case(case);raw.append(report)
                np.savez_compressed(local/(case['name']+'_target.npz'),V=case['V'])
                for (method,axis),wrapper in wrappers.items():
                    stem=case['name']+'_'+method+'_'+axis
                    toy.old.c3.base.save_qpy(local/(stem+'_raw.qpy'),wrapper)
                    tasks.append((case['name'],method,axis,case['DF_eligible'],str(local)))
            negative=toy.negative_controls(cases[0])
        if len(tasks)!=16:raise ValueError('FIXED16_ARTIFICIAL_WRAPPERS')
        write(out/'raw_theorem_checks.json',dict(status='PASS',initial_basis_checks=initial,cases=raw,
            negative_controls=negative,arbitrary_AB_unconditional_claim=False,H4_input_loaded=False))
        finished.append('four_artificial_cases_and_five_negative_controls');print(finished[-1],flush=True)
        workers=min(len(tasks),len(os.sched_getaffinity(0)))
        write(local/'resources.json',dict(affinity=len(os.sched_getaffinity(0)),workers=workers,
            MemAvailable_observation=[s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')]))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.origin,)) as pool:
            compile_names=list(pool.map(compile_task,tasks))
        finished.append('sixteen_artificial_P0_postchecks_PASS');print(finished[-1],flush=True)
        members=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in local.iterdir() if p.suffix in ('.qpy','.npz')]
        write(out/'payload_index.json',dict(members=members,raw_payload_uploaded=False,artificial_only=True))
        if hashes!={n:digest(ROOT/n) for n in EXEC}:raise ValueError('B6B_SOURCE_CHANGED')
        if reused_hashes!={n:digest(ROOT/n) for n in reused}:raise ValueError('REUSED_SOURCE_CHANGED')
        if origin_hashes!={n:digest(args.origin/n) for n in origin_files}:raise ValueError('ORIGINAL_SOURCE_CHANGED')
        status='B6B_TECHNICAL_PREPARATION_COMPLETE_MANDATORY_STOP'
    except BaseException as e:
        message=str(e).replace(str(ROOT),'<task>').replace(str(args.origin),'<original-science>')
        write(out/'failure.json',dict(type=type(e).__name__,error=message));raise
    finally:
        write(out/'result.json',dict(status=status,source_commit=source,completed=finished,
            artificial_full_wrapper_P0_completed=len(compile_names),H4_sampling=0,H4_build=0,H4_compile=0,
            H4_price_reanalysis=False,N_recomputed=False,H4_N_times_price_evaluated=False,
            actual_success_guaranteed=False,old_results_modified=False,mandatory_stop=True,next_stage_authorized=False))
        write(out/'STOP.json',dict(status=status,mandatory_stop=True,next_stage_authorized=False,next_actor='user/GPT'))


if __name__=='__main__':main()
