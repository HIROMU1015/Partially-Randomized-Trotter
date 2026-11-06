"""Run only the fixed six-cell S2 finite-table forecast; no sampling or build."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter, process_time
import hashlib
import json
import os
import platform
import resource
import subprocess

from trottertracks.algorithm_codesign.s2_multi_event import (
    pair_table, positive_t_prices, evaluate_law, exact_json, METRICS)
from trottertracks.algorithm_codesign.parent_experiment.design import integer_io_limit
from trottertracks.algorithm_codesign.rte_development.accounting import error_budget

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT/'artifacts/o2_s2_multi_event_validation/2026-11-06/v2'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def write(name, value):
    # Exact decimal rational serialization needs more than CPython's default.
    with integer_io_limit(1_000_000):
        data = json.dumps(exact_json(value), ensure_ascii=False, indent=2)+'\n'
    with (OUTPUT/name).open('x', encoding='utf-8') as out:
        out.write(data)
    return len(data.encode())


def read_fixed(spec):
    data = git('show', spec['commit']+':'+spec['path'])
    if hashlib.sha256(data).hexdigest() != spec['sha256']:
        raise ValueError('fixed input bytes mismatch')
    return json.loads(data)


def main():
    if (OUTPUT/'RESULT.json').exists() or (OUTPUT/'pairs.json').exists():
        raise FileExistsError('existing scientific output; no automatic retry')
    wall, cpu = perf_counter(), process_time()
    source = json.loads((OUTPUT/'SOURCE.json').read_text())
    for spec in source['reused_code']:
        if git('show', spec['commit']+':'+spec['path']) != (ROOT/spec['path']).read_bytes():
            raise ValueError('reused SOURCE differs from fixed version: '+spec['path'])
    head = git('rev-parse', 'HEAD').decode().strip()
    for path, digest in source['new_code_sha256'].items():
        data=(ROOT/path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest or data!=git('show',head+':'+path):
            raise ValueError('new scientific SOURCE is not the committed version')
    result = dict(status='STARTED', source_commit=head, design_commit=source['design_commit'],
                  source_path=str((OUTPUT/'SOURCE.json').relative_to(ROOT)),
                  quantum_shots_executed=0, sampling_calls=0, matrix_calls=0,
                  build_compile_calls=0, synthesis_calls=0, mandatory_STOP=True,
                  online_classical_cost='NOT_MEASURED', laws={}, comparisons={})
    try:
        start=perf_counter()
        fixed={k:read_fixed(v) for k,v in source['inputs'].items()}
        bindings=fixed['events']; cache=fixed['cache']; row=fixed['d1_map']['rows'][10]
        expected=dict(steps=2,x='5/14',degree=5,arm='closed_P5_full',state='000')
        if any(row.get(k)!=v for k,v in expected.items()) or len(bindings)!=63:
            raise ValueError('fixed input/target mismatch')
        if hashlib.sha256(git('show',source['inputs']['events']['commit']+':'+source['inputs']['events']['path'])).hexdigest()!=row['events']['sha256']:
            raise ValueError('D1 event binding identity mismatch')
        keys={g[2] for b in bindings for g in b['native_ir'] if g[0]=='R'}
        for key in keys:
            entry=cache[key]; seq=entry['sequence']
            if (any(g not in 'HTtSXW' for g in seq)
                or hashlib.sha256(seq.encode()).hexdigest()!=entry['sequence_sha256']
                or entry['T_count']!=seq.count('T')+seq.count('t')
                or entry['Tdagger_count']!=seq.count('t')
                or entry['one_qubit_count']!=len(seq)-seq.count('W')
                or entry['global_W_count']!=seq.count('W')
                or entry['epsilon']!='1/1000000' or entry['error_pass'] is not True
                or not 0<=F(entry['strict_operator_error_upper'])<=F(1,10**6)):
                raise ValueError('saved sequence/strict error identity mismatch')
        result['input_loading_and_identity_wall_seconds']=perf_counter()-start
        result['saved_cache_keys_used']=len(keys)
        err=error_budget(F(5,14),2,source['conditions'])
        if any(F(row['budget'][k])!=v for k,v in err.items() if isinstance(v,F)):
            raise ValueError('saved D2 common bias/remaining mismatch')
        result['common_error_budget']=err
        result['historical_D1_production_N_per_axis']=row['budget']['N_per_axis']
        start=perf_counter()
        pairs=pair_table(bindings,cache)
        if len(pairs)!=3969: raise ValueError('ordered pair support mismatch')
        result['pair_table_wall_seconds']=perf_counter()-start
        result['pair_count']=len(pairs)
        result['pair_table_bytes']=write('pairs.json',pairs)
        result['zero_T_pairs']=[dict(i=r['i'],j=r['j'],recipe=k) for r in pairs for k in ('A','B') if r[k]['T']==0]
        positive_t_prices(pairs)
        print('Fixed pair table saved; positive T prices verified.',flush=True)
        start=perf_counter()
        # Three independent finite-law tasks are the dominant rational workload.
        with integer_io_limit(1_000_000), ProcessPoolExecutor(max_workers=3, mp_context=get_context('fork')) as pool:
            jobs={pool.submit(evaluate_law,name,pairs,err['remaining'],F(49,24000)):name for name in ('P0','qA','qB')}
            for job in as_completed(jobs):
                name=jobs[job]; law=job.result()
                events=law.pop('events')
                law['law_table_bytes']=write('law_'+name+'.json',events)
                result['laws'][name]=law
                print(name+' complete; N_per_axis='+str(law['N_per_axis']),flush=True)
        result['parallel_laws_wall_seconds']=perf_counter()-start
        # Existing D2 aggregate is an input-connection check, not a seventh cell.
        anchor=next(r for r in fixed['d2_map']['rows'] if r['arm']=='closed_P5_full')
        p0=result['laws']['P0']
        for k in METRICS:
            if p0['cells']['A']['body_expected'][k]!=F(anchor['additive_per_whole_trial'][k]) or p0['cells']['B']['body_expected'][k]!=F(anchor['after_per_whole_trial'][k]):
                raise ValueError('D2 original-law aggregate anchor mismatch')
        result['D2_original_law_price_anchor']='EXACT_MATCH'
        for baseline in ('P0','qA'):
            cand=result['laws']['qB']['cells']['B']; base=result['laws'][baseline]['cells']['B']
            diff={k:cand['known_two_axis_cost'][k]-base['known_two_axis_cost'][k] for k in METRICS}
            slope=cand['unknown_common_h_slope']-base['unknown_common_h_slope']
            result['comparisons']['qB_B_minus_'+baseline+'_B']=dict(
                known_cost_difference=diff, relative_saving={k:-diff[k]/base['known_two_axis_cost'][k] for k in METRICS},
                common_h_difference_slope=slope,
                equal_cost_h={k:-diff[k]/slope if slope else None for k in METRICS})
        result['status']='COMPLETE_FIXED_SYNTHETIC_SIX_CELL_FORECAST'
    except Exception as exc:
        result['status']='STOPPED'
        result['stop_reason']=type(exc).__name__+': '+str(exc)
        print(result['stop_reason'],flush=True)
    finally:
        result['setup_runtime']=dict(wall_seconds=perf_counter()-wall,parent_CPU_seconds=process_time()-cpu,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            completed_children_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            completed_children_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            available_CPU_affinity=len(os.sched_getaffinity(0)),law_workers=3,python=platform.python_version(),
            GPU='driver unavailable; CPU rational workload; no GPU use',online='NOT_MEASURED')
        write('RESULT.json',result)
        print(result['status'],flush=True)


if __name__=='__main__': main()
