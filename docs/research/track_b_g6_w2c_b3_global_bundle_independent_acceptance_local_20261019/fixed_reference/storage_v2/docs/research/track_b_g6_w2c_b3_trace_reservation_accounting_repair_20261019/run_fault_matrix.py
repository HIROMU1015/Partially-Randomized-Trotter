"""Parallel isolated storage faults, bounded by available CPU and 8-pair workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def specifications():
    shapes=[dict(name='normal'),dict(name='original_reserve_writer',writer='fail'),
        dict(name='partial_complete_writer',writer='partial',writer_op='COMPLETE')]
    for op in('RESERVE','ENTER','COMPLETE','STOP','PHASE_BEGIN','PHASE_END'):
        row=dict(name='trace_write_'+op,point='trace.write',match=dict(op=op))
        if op=='STOP':row['settings']=dict(faults=[[0,0,'sample',0]])
        shapes.append(row)
    for op in('RESERVE','COMPLETE'):
        for point in('trace.flush','trace.hash'):
            shapes.append(dict(name=point.replace('.','_')+'_'+op,point=point,match=dict(op=op)))
    for op in('WRITE_RESERVE','WRITE_ACK'):
        shapes.append(dict(name='trace_'+op,point='trace.write',match=dict(op=op,channel='result')))
    for state in('START_RESERVED','STARTED','IN_FLIGHT','COMPLETED','FAILED','CANCELLED_BEFORE_START','TERMINATED_WITH_UNKNOWN_RESULT'):
        row=dict(name='state_write_'+state,point='state.write',match=dict(state=state))
        if state=='FAILED':row['settings']=dict(faults=[[0,0,'sample',0]])
        if state=='CANCELLED_BEFORE_START':continue # covered deterministically by transaction tests
        if state=='TERMINATED_WITH_UNKNOWN_RESULT':row['settings']=dict(faults=[[0,0,'exit',0]])
        shapes.append(row)
    shapes.append(dict(name='state_after_reserve',point='state.after_write',match=dict(state='START_RESERVED')))
    for point in('trace.open','state.open','state.truncate','journal.open'):
        shapes.append(dict(name=point.replace('.','_'),point=point,early=True))
    for event in('WORKER_END','FAILURE_OBSERVED','PILOT_COMPLETION_RECEIPT'):
        row=dict(name='journal_'+event,point='journal.write',match=dict(event=event))
        if event=='FAILURE_OBSERVED':row['settings']=dict(faults=[[0,0,'sample',0]])
        shapes.append(row)
    for point in('summary.write','summary.after_write','summary.flush','receipt.write','receipt.after_write',
                  'receipt.flush','final_report.write','final_report.after_write','final_report.flush',
                  'trace.close','state.close','journal.close'):
        shapes.append(dict(name=point.replace('.','_'),point=point))
    for name,settings in dict(worker_exit=dict(faults=[[0,0,'exit',0]]),reserved_exit=dict(faults=[[0,0,'reserved_exit',0]]),
        unacked_exit=dict(faults=[[0,0,'ack_exit',0]]),partial_pair_write=dict(faults=[[0,0,'write',0]]),
        pair_flush=dict(faults=[[0,0,'flush',0]],flush_rows=1),call_cap=dict(call_limit=2),
        result_cap=dict(result_bytes=100),log_cap=dict(log_bytes=100),
        cooperative_drain=dict(sync_failure='one'),simultaneous=dict(sync_failure='all')).items():
        shapes.append(dict(name=name,settings=settings,early=name=='log_cap'))
    shapes.append(dict(name='parent_crash',settings=dict(parent_fault='after_reserved'),parent_crash=True))
    for point in('failure_record_save','stop_record_save'):
        shapes.append(dict(name='admission_'+point,admission_point=point))
    return [dict(shape,mode=mode,case_id=mode+'_'+shape['name'])for mode in('S','P1')for shape in shapes]


def main():
    out=HERE/('fault_cases'+('_'+sys.argv[1]if len(sys.argv)>1 else''));out.mkdir(exist_ok=False)
    specs=specifications();(out/'matrix_spec.json').write_text(json.dumps(specs,indent=2)+'\n')
    def one(spec):
        dest=out/spec['case_id']
        p=subprocess.run([sys.executable,str(HERE/'run_storage_case.py'),json.dumps(spec),str(dest)],
            cwd=ROOT,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (out/(spec['case_id']+'.log')).write_bytes(p.stdout+p.stderr)
        if p.returncode:
            print(spec['case_id'],'ERROR',flush=True)
            return dict(case=spec['case_id'],successful=False,exit_code=p.returncode)
        r=json.loads((dest/'case_result.json').read_text());r['exit_code']=p.returncode
        print(spec['case_id'],r['status'],flush=True);return r
    concurrency=max(1,len(os.sched_getaffinity(0))//8)
    with ThreadPoolExecutor(max_workers=concurrency)as pool:results=list(pool.map(one,specs))
    (out/'results.json').write_text(json.dumps(dict(concurrency=concurrency,cases=results),indent=2)+'\n')
    assert all(r['successful']for r in results)


if __name__=='__main__':main()
