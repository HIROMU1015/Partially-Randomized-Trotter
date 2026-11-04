"""Shared synthetic/scientific finite-certificate pipeline; no H4 loader."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as Q
import json
import multiprocessing
from pathlib import Path
import time

import integration_audit as audit
from integration_audit import require, transfer

STATUS = dict(RPE_tolerance_certified=False,scientific_status='UNDETERMINED',
              U_virtual_IID_upper=None,U_exec_upper=None,mandatory_STOP=True)
_STOP = None


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temp.replace(path)


def initialize_basis_worker(stop):
    global _STOP
    _STOP = stop


def basis_worker(task):
    """Return serializable balls; retain the started/error state on failure."""
    key,n,operations = task
    item = dict(key=key,started=False)
    if _STOP is not None and _STOP.is_set():
        return {**item,'skipped_after_stop':True}
    try:
        import ctypes
        from flint import ctx
        require(ctypes.CDLL(None).fegetround()==0, 'worker requires round-to-nearest')
        ctx.threads = 1  # Independent tiny matrices are the parallel dimension.
        with ctx.workprec(96):
            item['started'] = True
            b,_ = transfer.modules()
            ball = transfer.one_particle(operations,n)
            item['ball'] = b.encode_matrix_ball(ball)
    except Exception as exc:
        if _STOP is not None:
            _STOP.set()
        item['error'] = type(exc).__name__+': '+str(exc)
    return item


def evaluate_bases(tasks, workers, output, record):
    require(type(workers) is int and workers>=1,'positive worker count required')
    results = {}
    def collect(item):
        with (output/'basis_enclosures.jsonl').open('a') as stream:
            stream.write(json.dumps(item,allow_nan=False)+'\n')
            stream.flush()
        if item['started']:
            record['counts']['basis_evaluations_started'] = record['counts'].get('basis_evaluations_started',0)+1
        if 'ball' in item:
            results[item['key']] = item['ball']
        if 'error' in item:
            record.setdefault('basis_errors',[]).append(item)
    if workers == 1:
        for task in tasks:
            item = basis_worker(task)
            collect(item)
            if 'error' in item:
                break
    else:
        context = multiprocessing.get_context('spawn')
        stop = context.Event()
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),mp_context=context,
                initializer=initialize_basis_worker,initargs=(stop,)) as pool:
            futures = [pool.submit(basis_worker,task) for task in tasks]
            for future in as_completed(futures):
                try:
                    collect(future.result())
                except BaseException:
                    stop.set()
                    record['basis_started_counts_unresolved'] = True
                    raise
    require(not record.get('basis_errors') and len(results)==len(tasks),'basis evaluation UNKNOWN/STOP')
    return results


def execute(output, bundle, context_loader, *, workers=1, source=None):
    """Fresh output only. context_loader is H4 ONLY behind the explicit --run."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)  # Never catch then overwrite an old run.
    record = {**STATUS,'status':'STARTED','stage':'context','counts':{},'components':[],
              'boundary':[],'rounds':[],'connection_decision':'UNDETERMINED',
              'mode':bundle.get('mode','SYNTHETIC'),'source':source,'precision_bits':96,
              'resources':bundle.get('resources',{}),'workers':workers,
              'pi64_already_in_saved_I_cat':True,'pi64_added_again':False,
              'forbidden_calls':{k:0 for k in ('state_load_or_solve','new_H_or_DF_factorization',
                  'law_CDF_atan_mean','trajectory','sampling_shots','many_body_evolution',
                  'long_q_build','transpile','cost_D_review')}}
    started = time.monotonic()
    def checkpoint():
        write_json(output/'RESULT.json',record)
    checkpoint()
    try:
        data = context_loader(bundle,record)
        prep,support,saved = data['preparation'],data['support'],data['export']
        counts = record['counts']
        record['stage'] = 'descriptors'
        bases,components = audit.export_context(prep,support,saved,bundle['support_proofs'],counts)
        require(len(bases)<=229 and len(components)<=218 and len(support)<=218,'descriptor scope exceeded')
        write_json(output/'DESCRIPTORS.json',dict(bases=bases,components=components,
                   support_policy=bundle['support_policy'],training_fingerprint=bundle['training_fingerprint']))
        record['stage'] = 'boundary_primitives'
        delta = float(bundle['rows'][0]['delta'])
        require(all(float(row['delta'])==delta for row in bundle['rows']),'step-time mismatch')
        halves,full,raw_angles,db = audit.boundary_context(prep,saved,delta,counts)
        record['primitive_parameters'] = dict(half=[audit.primitive_payload(p) for p in halves],
            first_full=audit.primitive_payload(full),raw_scalar=[x.hex() for x in raw_angles],block_gap=str(db))
        record['stage'] = 'lowering'
        record['lowering'] = audit.audit_lowering(halves,full,bundle['theta_by_r'],counts,data['library_evidence'])
        checkpoint()
        record['stage'] = 'basis_enclosures'
        unique,aliases = {},{}
        for name,spec in sorted(bases.items()):
            key = spec['basis_hash']
            task = (key,spec['n_qubits'],spec['operations'])
            if key in unique:
                require(unique[key]==task,'basis cache identity collision')
            unique[key] = task
            aliases[name] = key
        record['basis_cache'] = dict(definitions=len(bases),evaluations=len(unique),hits=len(bases)-len(unique))
        checkpoint()
        encoded = evaluate_bases(list(unique.values()),workers,output,record)
        from flint import ctx
        # decode reuses the old enclosure helper and retains all decoding uncertainty.
        import sys
        sys.path.insert(0,str(audit.ROOT/'docs/research/rpe_18round_ideal_enclosure_preparation_20261104'))
        from round_adapter import decode
        with ctx.workprec(96):
            decoded = {key:decode(value) for key,value in encoded.items()}
            record['stage'] = 'component_bounds'
            for c in components:
                counts['component_bounds'] = counts.get('component_bounds',0)+1
                gap = transfer.component_gap(decoded[aliases[c['full']]],
                    decoded[aliases[c['selected']]],c['support'])
                record['components'].append(dict(id=c['id'],upper=str(gap)))
                checkpoint()
        require(len(record['components'])==len(saved['components']),'component bound coverage')
        maximum = max(Q(c['upper']) for c in record['components'])
        record['component_max_upper'] = str(maximum)
        record['stage'] = 'round_transfers'
        require(len(bundle['rows'])==18 and {r['m'] for r in bundle['rows']}==set(range(18)),
                'all18 round coverage required')
        for row in bundle['rows']:
            m,q,r = row['m'],row['q'],row['r']
            record['active_round'] = m
            require(q==1<<m and r in bundle['theta_by_r'] and row['K']==2,'round/r/K mismatch')
            counts['scalar_boundary_evaluations'] = counts.get('scalar_boundary_evaluations',0)+1
            g = -delta*q*(prep.constant_coefficient+prep.extracted_identity_coefficient)
            ds = transfer.scalar_gap(q,raw_angles,g)
            boundary = transfer.boundary_bound(q,ds,db)
            extra = transfer.total_transfer(q,r,boundary=boundary,component=maximum,
                                             lowering=record['lowering']['bound'])
            record['boundary'].append(dict(m=m,q=q,r=r,optimized_scalar=audit.finite_hex(g),
                                           scalar_gap=str(ds),upper=str(boundary)))
            counts['round_transfers'] = counts.get('round_transfers',0)+1
            axes = {}
            for axis in ('cosine','sine'):
                counts['axis_comparisons'] = counts.get('axis_comparisons',0)+1
                axes[axis] = audit.classify_axis(bundle['saved_axes'][m][axis],
                                                 row['axes'][axis]['epsilon'],extra)
            record['rounds'].append(dict(m=m,q=q,r=r,K=2,boundary_upper=str(boundary),
                support_upper=str(min(Q(2),3*q*r*maximum)),lowering_upper=record['lowering']['bound'],
                extra_upper=str(extra),axes=axes))
            checkpoint()
        decisions = [a['decision'] for r in record['rounds'] for a in r['axes'].values()]
        if len(decisions)==36 and all(x=='SUFFICIENT_CONDITION_CONFIRMED' for x in decisions):
            record['connection_decision'] = 'CONDITIONAL_IDEAL_CONTROLLED_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED'
        record['status'] = 'COMPLETED_MANDATORY_STOP'
        record['stage'] = 'complete'
        record.pop('active_round',None)
    except BaseException as exc:
        record['status'] = 'UNKNOWN_MANDATORY_STOP'
        record['error'] = type(exc).__name__+': '+str(exc)
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        checkpoint()
    return record
