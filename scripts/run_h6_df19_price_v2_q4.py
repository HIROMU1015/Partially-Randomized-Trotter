"""q4 execution: reuse one whole-path evolution, fork independent X/Y pricing.

No change to scientific word, sampler, fingerprint, compiler, or 36-price budget.
The q2 source is preserved. Fork happens before any compiler invocation.
"""
import multiprocessing as mp
import os,time,resource,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import run_h6_df19_price_v2_pilot as p
AXIS_CONTEXT=None


def price_axis(axis,directory,evolution,fp):
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
    fingerprint=fp.fingerprint(wrapper);old=p.OLD_CACHE/(fingerprint+'.json');hit=old.exists()
    if hit:
        data=p.read(old);p.require(data['fingerprint']==fingerprint and data['compiler']==p.COMPILER,'Cache binding');metrics=data['metrics']
        p.write(directory/(axis+'_cache_hit.json'),dict(path=str(old),sha256=p.sha(old.read_bytes()),fingerprint=fingerprint,compiler=p.COMPILER))
    else:
        p.write(directory/(axis+'_compile_started.json'),dict(fingerprint=fingerprint,compiler=p.COMPILER))
        compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
        cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
        p.require(cost.compiler_settings_hash==p.COMPILER,'Compiler hash');metrics={k:getattr(cost,k) for k in p.METRICS}
        del cost
    p.require(fp.fingerprint(wrapper)==fingerprint,'Compiler mutated input')
    result=dict(metrics=metrics,fingerprint=fingerprint,compiler=p.COMPILER,cache_hit=hit,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    p.write(directory/(axis+'_price.json'),result);return result


def child(axis,directory,pipe):
    try:
        evolution,fp=AXIS_CONTEXT;pipe.send(dict(result=price_axis(axis,directory,evolution,fp)))
    except BaseException as error:pipe.send(dict(error=str(error),traceback=traceback.format_exc()))
    finally:pipe.close()


def parallel_axes(directory,evolution,fp):
    global AXIS_CONTEXT
    AXIS_CONTEXT=(evolution,fp);pending=[];ctx=mp.get_context('fork')
    for axis in p.AXES:
        parent,remote=ctx.Pipe(duplex=False);proc=ctx.Process(target=child,args=(axis,directory,remote));proc.start();remote.close();pending.append((axis,proc,parent))
    results={};errors=[]
    for axis,proc,pipe in pending:
        try:message=pipe.recv()
        except EOFError:message=dict(error='AXIS_WORKER_EOF')
        proc.join();pipe.close()
        if proc.exitcode!=0 or 'error' in message:errors.append(dict(axis=axis,exitcode=proc.exitcode,message=message))
        else:results[axis]=message['result']
    AXIS_CONTEXT=None
    if errors:raise ValueError(errors)
    return results


def trial(job):
    group,index=job;row=group['candidate'];started=time.monotonic();ctx=p.context();prep=ctx.preps[row['prefix']]
    directory=p.PRIVATE/'q4'/row['id']/str(index);directory.mkdir(parents=True,exist_ok=False)
    try:
        pr=row['method']=='PR';requests,seeds,laws=ctx.h['sample_requests'](row,prep,group['cohort'],index) if pr else ([],[],[])
        if pr:p.require(seeds==group['seed_streams'][index],'Frozen seeds')
        events=[[e.to_dict() for e in req.rte_occurrence.events] for req in requests]
        p.write(directory/'events_PRIVATE.json',dict(events=events,seeds=seeds,laws=laws,cohort=group['cohort'],conditional=False))
        scalar=ctx.ham.constant+(prep.extracted_identity_coefficient if pr else 0.)
        blocks=p.piece_blocks(prep,ctx);predictions={a:p.predict(blocks,4,scalar,6.4,a,events if pr else None) for a in p.AXES}
        p.write(directory/'predictions.json',predictions)
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        with use_fingerprint_reuse() as fp:
            evolution=ctx.h['compose'](row,prep,requests,scalar,fp.fingerprint)
            p.write(directory/'native.json',dict(instructions=len(evolution.circuit.data),instruction_bound=evolution.instruction_upper_bound,fingerprint=evolution.circuit_fingerprint))
            axes=parallel_axes(directory,evolution,fp)
        result=dict(status='PASS_PRICE_ACQUIRED',candidate=row,cohort=group['cohort'],index=index,preparation_hash=prep.preparation_hash,scalar=scalar,prices={a:axes[a]['metrics'] for a in p.AXES},fingerprints={a:axes[a]['fingerprint'] for a in p.AXES},predictions=predictions,cache_hits=sum(int(axes[a]['cache_hit']) for a in p.AXES),orders={str(k):sum(e['taylor_order']==k for batch in events for e in batch) for k in (0,2)},wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,axis_peak_RSS_bytes={a:axes[a]['peak_RSS_bytes'] for a in p.AXES},conditional_sampling=False,signal_N_G_computed=False,full_wrapper_statevector_validation=False)
        p.write(directory/'result.json',result);return result
    except BaseException as error:
        p.write(directory/'STOP.json',dict(reason=str(error),traceback=traceback.format_exc()));raise


def main():
    source=p.source();name='scripts/run_h6_df19_price_v2_q4.py';p.require(Path(__file__).read_bytes()==p.blob(source,name),'Commit q4 source')
    plan=p.read(p.PRIVATE/'INPUTS_PRIVATE.json');frozen=p.read(p.PUBLIC/'Q4_FROZEN_MODELS.json')
    p.require(frozen['q4_execution_script_sha256']==p.sha(Path(__file__).read_bytes()),'Frozen q4 executor')
    p.require(not (p.PRIVATE/'q4').exists(),'No implicit q4 rerun')
    p.context();groups=[g for g in plan['groups'] if g['candidate']['q']==4];jobs=[(g,i) for g in groups for i in range(g['paths'])];workers=min(len(jobs),len(os.sched_getaffinity(0)))
    print('START q4',workers,'whole-path workers, 2 independent axis children each',flush=True);rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        pending={pool.submit(trial,j):j for j in jobs}
        for future in as_completed(pending):
            r=future.result();rows.append(r);print('DONE 4',r['candidate']['method'],r['index'],r['prices']['cosine']['rz_count'],round(r['wall_seconds'],2),flush=True)
    rows.sort(key=lambda r:(r['candidate']['method'],r['index']))
    p.write(p.PUBLIC/'Q4_DIRECT.json',dict(scientific_source=source,q2_acquisition_source=plan['source'],q4_frozen_models_sha256=p.sha((p.PUBLIC/'Q4_FROZEN_MODELS.json').read_bytes()),compiler_hash=p.COMPILER,rows=rows,whole_path_workers=workers,axis_children_per_path=2,nested_compiler_processes=1,new_full_wrapper_prices=sum(2-r['cache_hits'] for r in rows),cache_hits=sum(r['cache_hits'] for r in rows),full_RPE_qualified=False))


if __name__=='__main__':main()
