"""New one-shot PR q16/n33 and PF q32 campaign; no fits, N, or G."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h4_df10_price_v2_independent as model

PRIVATE = Path('/home/AbeHiromu/projects/h4-df10-price-v2-independent-q/20261110/run01')
PUBLIC = ROOT/'docs/research/h4_df10_price_v2_independent_q_20261110'
ORIGINAL = '155529e20a8fac39e00786c40b57830b71203c4f'
OLD_CACHE = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/compiled_cache')
SOURCE_FILES = ['scripts/run_h4_df10_price_v2_independent.py',
                'src/trottertracks/h4_df10_price_v2_independent.py',
                'src/trottertracks/h4_df10_price_v1.py',
                'src/trottertracks/h4_df10_price_v2.py',
                'src/trottertracks/track_d_native_resources.py',
                'scripts/run_h4_df10_price_v1.py']
ORIGINAL_FILES = ['scripts/h46_long_time_native.py','scripts/h46_long_time_signal.py',
    'scripts/h46_new2_native.py','scripts/h46_new2_core.py','scripts/h46_m5_tol_native.py',
    'scripts/h46_m5_tol_saved_input.py','scripts/h46_m5_contract.py',
    'scripts/h46_m5_tol_contract.py','scripts/h46_pr_pilot_repair.py',
    'src/trotterlib/h46_m5_plan.py']
SAMPLED = {}
EVENT_HASHES = {}
EVENTS = {}
DESCRIPTORS = None
NATIVE = None
ORIGINAL_COMPILE = None


def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sampled(row, cohort, index):
    return SAMPLED[row['id'], index]


def predict_trial(job):
    row, _, index = job
    blocks, constant = DESCRIPTORS
    pr = row['method']=='PR'
    scalar = constant+(NATIVE.CONTEXT['H4'].preparations[1].extracted_identity_coefficient if pr else 0.)
    output=[]
    for axis in ('cosine','sine'):
        p=model.predict(blocks[:2] if pr else blocks,row['q'],scalar,row['T'],axis,
                        EVENTS[row['id'],index] if pr else None)
        output.append(dict(method='PR' if pr else 'PF',candidate_id=row['id'],q=row['q'],T=row['T'],
            path_index=index,axis=axis,events_sha256=EVENT_HASHES[row['id'],index],
            structural_rz=p['structural_rz'],Mq=p['structural_rz']+model.B['PR' if pr else 'PF']*row['q'],
            features=p['features']))
    return output


def cached(wrapper, fingerprint, representative):
    old = OLD_CACHE/(fingerprint+'.json')
    if old.exists():
        return model.validate_cache(json.loads(old.read_bytes()), fingerprint), None, True
    result, qc, reused = ORIGINAL_COMPILE(wrapper, fingerprint, representative)
    return model.validate_cache(result, fingerprint), qc, reused


def trial(job):
    result = NATIVE.cost_trial(job)
    directory = PRIVATE/'cost'/job[1]/job[0]['id']/str(job[2])
    saved = json.loads((directory/'events_PRIVATE.json').read_bytes())
    if digest(json.dumps(saved['events'],sort_keys=True,separators=(',',':')).encode()) != EVENT_HASHES[job[0]['id'],job[2]]:
        raise ValueError('PREDICTION_COMPILE_EVENT_BINDING')
    bindings = json.loads((directory/'bindings.json').read_text())
    result['wrapper_bindings'] = bindings
    NATIVE.save(directory/'result.json', result)
    return result


def integers(value):
    if type(value) is int:
        yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from integers(item)
    elif isinstance(value, list):
        for item in value: yield from integers(item)


def main():
    global NATIVE, ORIGINAL_COMPILE, DESCRIPTORS
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    for path in SOURCE_FILES:
        if (ROOT/path).read_bytes() != blob(source, path):
            raise ValueError('UNCOMMITTED_SOURCE:'+path)
    for path in ORIGINAL_FILES:
        if (ROOT/path).read_bytes() != blob(ORIGINAL, path):
            raise ValueError('ORIGINAL_SOURCE_CHANGED:'+path)
    for path in SOURCE_FILES[2:]:
        if (ROOT/path).read_bytes() != blob(model.S1, path):
            raise ValueError('FROZEN_ENGINE_CHANGED:'+path)
    s1 = json.loads(blob(model.S1, 'docs/research/h4_df10_price_v2_s1_20261110/RESULT.json'))
    for c in s1['calibrations']:
        if c['model'] == 'Mq' and c['coefficients'] != [[model.B[c['method']]]*2]:
            raise ValueError('FROZEN_S1_COEFFICIENT')
    if any((PUBLIC/name).exists() for name in ('PREDICTIONS.json','PROVENANCE.json','RESULT.json')):
        raise FileExistsError('preserve previous independent-q outputs')
    PRIVATE.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    import h46_long_time_native as native
    import h46_long_time_signal as study
    cfg = study.install()
    provider_files = ['src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
        'src/trottertracks/resource_applicability/h6_hash_reuse_v1.py',
        'src/trotterlib/df_rte_qiskit.py','src/trotterlib/df_partial_s2.py',
        'src/trotterlib/rte_compiled_cost.py','src/trotterlib/rte.py']
    for path in provider_files:
        if (Path(cfg['provider_root'])/path).read_bytes() != blob('c31ef424175c02379eafebe8fc6c7548c1d9aabe',path):
            raise ValueError('PROVIDER_SOURCE_CHANGED:'+path)
    import trottertracks
    trottertracks.__path__.insert(0, str(Path(cfg['provider_root'])/'src/trottertracks'))
    NATIVE = native
    native.PRIVATE = PRIVATE
    native.sample_requests, original_sample = sampled, native.sample_requests
    ORIGINAL_COMPILE = native.compile_cached
    # Record the exact wrapper identity even when a cache provides its price.
    def compile_recorded(wrapper, fingerprint, representative):
        result, qc, hit = cached(wrapper, fingerprint, representative)
        directory = Path(CURRENT_DIRECTORY[0])
        values = json.loads((directory/'bindings.json').read_text()) if (directory/'bindings.json').exists() else []
        values.append(dict(axis=('cosine','sine')[len(values)], fingerprint=fingerprint, compiler=model.COMPILER,
                           cache_hit=hit, old_cache_hit=(OLD_CACHE/(fingerprint+'.json')).exists()))
        native.save(directory/'bindings.json', values)
        return result, qc, hit
    # This closure is inherited by fork workers; each worker has private state.
    original_trial = native.cost_trial
    def local_trial(job):
        CURRENT_DIRECTORY[0] = str(PRIVATE/'cost'/job[1]/job[0]['id']/str(job[2]))
        return original_trial(job)
    CURRENT_DIRECTORY = ['']
    native.compile_cached = compile_recorded
    native.cost_trial = local_trial
    try:
        ctx = native.native_context('H4', [1, 7])
        native.CONTEXT['H4'] = ctx
        spec = importlib.util.spec_from_file_location('v1_descriptors', ROOT/'scripts/run_h4_df10_price_v1.py')
        v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
        blocks, constant, binding, input_hashes = v1.inputs_and_blocks()
        DESCRIPTORS = (blocks,constant)
        if ctx.preparations[1].preparation_hash != v1.PREPARATION:
            raise ValueError('SAVED_PR_PREPARATION_BINDING')
        for name, meta in cfg['systems']['H4']['inputs'].items():
            if digest(Path(meta['path']).read_bytes()) != input_hashes[name]:
                raise ValueError('SAVED_INPUT_CHANGED')
        rows = []
        for method, q, T, prefix in [('B0',32,3.2,7),('PR',16,1.6,1)]:
            row = dict(system='H4', method=method, formula='2nd', prefix=prefix,
                       q=q, T=T, delta=.1, R_outer=4 if method=='PR' else None,
                       K=2 if method=='PR' else None)
            row['id'] = study.cid(row); rows.append(row)
        known = set()
        # Explicit historical registries only, never other worktrees/processes.
        for path in [cfg.get('excluded_seeds_path'), cfg.get('registry_path')]:
            if path: known.update(integers(study.read(path)))
        h4cfg = study.read(study.ORIGIN/'H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json')
        for key in ('excluded_seeds_path','registry_path'):
            known.update(integers(study.read(h4cfg[key])))
        old_private = OLD_CACHE.parent
        for name in ('EXPLORATION','CONFIRMATION','EXTENSION_EXPLORATION','FIXED_DELTA_EXPLORATION'):
            path = old_private/(name+'_seeds.json.gz')
            if path.exists(): known.update(integers(study.read(path)))
        occupied = set(known); predictions = []; raw_cohort = []; jobs = []
        for row in rows:
            for index in range(33 if row['method']=='PR' else 1):
                prep = ctx.preparations[row['prefix']]
                reqs, seeds, laws = original_sample(row, prep, model.COHORT, index) if row['method']=='PR' else ([],[],[])
                if any(seed in occupied for seed in seeds) or len(set(seeds)) != len(seeds):
                    raise ValueError('KNOWN_OR_NEW_SEED_COLLISION')
                occupied.update(seeds)
                SAMPLED[row['id'], index] = (reqs, seeds, laws)
                events = [[e.to_dict() for e in r.rte_occurrence.events] for r in reqs]
                event_bytes = json.dumps(events, sort_keys=True, separators=(',',':')).encode()
                event_hash = digest(event_bytes)
                EVENT_HASHES[row['id'],index] = event_hash
                EVENTS[row['id'],index] = events
                raw_cohort.append(dict(candidate=row,index=index,seeds=seeds,events=events,laws=laws))
                jobs.append((row, model.COHORT, index))
        native.save(PRIVATE/'COHORT_PRIVATE.json.gz', raw_cohort)
        workers = min(len(jobs), len(os.sched_getaffinity(0)))
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
            for values in pool.map(predict_trial,jobs):
                predictions.extend(values)
        print(json.dumps(dict(predictions_frozen=len(predictions),compiled_prices_seen=0)),flush=True)
        native.save(PUBLIC/'PREDICTIONS.json', dict(source=source, s1=model.S1, b=model.B,
                    candidates=rows, predictions=predictions, price_observations_used=False))
        native.save(PUBLIC/'PROVENANCE.json', dict(source=source,original_carrier=ORIGINAL,
            provider_source=v1.PROVIDER,scientific_binding=binding,input_sha256=input_hashes,
            compiler=model.COMPILER,cohort=model.COHORT,new_PR_trajectories=33,
            new_seeds=len(occupied)-len(known),known_seed_collisions=0,
            complete_historical_seed_coverage_claimed=False,PRNG_independence_proved=False,
            source_sha256={p:digest((ROOT/p).read_bytes()) for p in SOURCE_FILES+ORIGINAL_FILES},
            frozen_predictions_sha256=digest((PUBLIC/'PREDICTIONS.json').read_bytes())))
        native.save(PRIVATE/'resources.json',dict(workers=workers,affinity_cpus=len(os.sched_getaffinity(0)),
                                               nested_compiler_processes=1))
        results, errors = [], []
        # Original representative correctness gates release the other PR paths.
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
            pending = {pool.submit(trial,j): j for j in jobs if j[2]==0}
            while pending:
                future = next(as_completed(pending)); job = pending.pop(future)
                try:
                    value = future.result(); results.append(value)
                    print(json.dumps(dict(complete=len(results),method=job[0]['method'],index=job[2])),flush=True)
                    if job[2]==0:
                        for other in jobs:
                            if other[0]['id']==job[0]['id'] and other[2]>0:
                                pending[pool.submit(trial,other)] = other
                except BaseException as error:
                    errors.append(dict(candidate=job[0],index=job[2],reason=str(error)))
                native.save(PRIVATE/'progress.json',dict(complete=len(results),total=len(jobs),errors=errors))
        if errors:
            raise RuntimeError('CAMPAIGN_STOP:'+json.dumps(errors))
        residuals = model.residual_rows(predictions,results)
        summaries = model.summarize(residuals)
        import numpy as np
        for summary in summaries:
            group = [r for r in residuals if all(r[k]==summary[k] for k in ('method','model','axis'))]
            metrics = summary['metrics']; n = summary['n_paths']; sd = metrics['residual_SD']
            metrics['residual_variance'] = sd**2 if sd is not None else None
            metrics['descriptive_mean_error_SE_under_IID_given_fit'] = sd/math.sqrt(n) if sd is not None else None
            prices = [r['compiled_rz'] for r in group]
            metrics['observed_price_SD'] = float(np.std(prices,ddof=1)) if n>1 else None
            metrics['observed_mean_SE_under_IID'] = metrics['observed_price_SD']/math.sqrt(n) if n>1 else None
        bindings = [dict(candidate_id=t['candidate']['id'],index=t['index'],
            preparation_hash=t['preparation_hash'],events_sha256=digest((PRIVATE/'cost'/model.COHORT/t['candidate']['id']/str(t['index'])/'events_PRIVATE.json').read_bytes()),
            wrapper_bindings=t['wrapper_bindings'],checks=t['checks'],orders=t['orders']) for t in results]
        new_compiles = sum(not b['cache_hit'] for t in results for b in t['wrapper_bindings'])
        assert new_compiles <= 68
        result = dict(status='INDEPENDENT_Q_COMPLETE',source=source,b=model.B,s1=model.S1,
            candidates=rows,cohort=model.COHORT,PR_trajectories=33,PF_circuits=1,
            summaries=summaries,trial_bindings=bindings,new_wrapper_compiles=new_compiles,
            cache_hits=68-new_compiles,fit_calls=0,new_N_calls=0,N_times_C_calls=0,
            long_q_qualified=False,all_law_expectation=None,total_CI=None,
            S1_reference_summaries=s1['summaries'],main_claim_changed=False,mandatory_STOP=True)
        native.save(PUBLIC/'RESULT.json',result)
        with (PUBLIC/'PATH_RESIDUALS.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(residuals[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(residuals)
        native.save(PRIVATE/'COMPLETE.json',dict(source=source,complete=len(results),mandatory_STOP=True))
        print(json.dumps(dict(status=result['status'],new_compiles=new_compiles,mandatory_STOP=True)),flush=True)
    except BaseException as error:
        native.save(PRIVATE/'STOP.json',dict(source=source,reason=str(error),traceback=traceback.format_exc(),mandatory_STOP=True))
        raise


if __name__ == '__main__':
    main()
