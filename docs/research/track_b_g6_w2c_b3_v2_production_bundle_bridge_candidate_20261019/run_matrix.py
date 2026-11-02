"""Fresh artificial cases, computational subprocess parallelism only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,json,subprocess,sys,os,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def specs():
    rows=[dict(name='normal_S',manifest=dict(modes=['S'])),dict(name='normal_P1',manifest=dict(modes=['P1'])),dict(name='normal_bundle_small'),dict(name='normal_bundle_n256',manifest=dict(profile_n=256))]
    def failure(name,mode,stage,phase=0,extra=None):
        mf=dict(condition_faults={mode:[[0,phase+(3 if mode=='P1'else 0),stage,0]]});mf.update(extra or {});rows.append(dict(name=name,manifest=mf))
    failure('S_failure_blocks_P1','S','sample');failure('P1_failure_preserves_S','P1','sample');failure('F01_after_event','S','after_event')
    failure('worker_crash','P1','exit');failure('reserved_worker_crash','P1','reserved_exit');failure('missing_ACK_unknown_credit','P1','ack_exit');failure('partial_worker_write','P1','write');failure('worker_flush','P1','flush')
    rows.append(dict(name='P1_inflight_drain',manifest=dict(sync_failure='one')))
    rows.append(dict(name='parent_dispatch_failure',manifest=dict(parent_fault='dispatch')))
    rows.append(dict(name='parent_crash',manifest=dict(parent_fault='after_reserved')))
    rows += [dict(name='call_cap',manifest=dict(call_limit=255)),dict(name='result_cap',manifest=dict(result_bytes=5*1024*1024)),dict(name='log_cap',manifest=dict(log_bytes=8192))]
    for mode in ('S','P1'):
        for op in ('RESERVE','ENTER','COMPLETE'):
            rows.append(dict(name=mode+'_'+op.lower()+'_persistence',fault=dict(point='trace.write',condition=mode,match=dict(op=op))))
    rows += [dict(name='admission_commit_ack_unknown',fault=dict(point='admission.commit_ack')),dict(name='summary_save',fault=dict(point='summary.json.write')),dict(name='state_reserve_write',fault=dict(point='state.write',condition='S',match=dict(state='START_RESERVED'))),dict(name='state_entry_write',fault=dict(point='state.write',condition='S',match=dict(state='IN_FLIGHT'))),dict(name='trace_partial',fault=dict(point='trace.write',condition='P1',match=dict(op='RESERVE'),action='partial')),dict(name='trace_complete_flush',fault=dict(point='trace.flush',condition='P1',match=dict(op='COMPLETE')))]
    for name,point in [('diagnostic_save','diagnostic.json.write'),('receipt_save','receipt.json.write'),('receipt_flush','receipt.json.flush'),('receipt_close','receipt.json.close'),('marker_save','marker.json.write'),('trace_close','trace.close'),('state_close','state.close'),('journal_close','journal.close'),('fake_STOP_save','admission.stop'),('final_close','final.json.close'),('delivery_close','delivery.json.close')]:
        rows.append(dict(name=name,fault=dict(point=point)))
    for name,point,match in [('trace_STOP_save','trace.write',dict(op='STOP')),('journal_STOP_save','journal.write',dict(event='FAILURE_OBSERVED'))]:
        rows.append(dict(name=name,manifest=dict(condition_faults={'S':[[0,0,'sample',0]]}),fault=dict(point=point,match=match)))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--only');a=parser.parse_args();dest=HERE/('runs_'+a.label);dest.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT/'src'));from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_contract_v1 as K
    c=K.candidate(ROOT);(dest/'trusted_candidate.json').write_bytes(K.encoded(c));cases=[s for s in specs()if not a.only or s['name']in a.only.split(',')]
    def one(spec):
        name=spec['name'];sp=dest/(name+'_spec.json');sp.write_text(json.dumps(spec,indent=2)+'\n');rp=dest/(name+'_record.json')
        p=subprocess.run([sys.executable,str(HERE/'run_case.py'),'--spec',str(sp),'--record',str(rp)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (dest/(name+'.log')).write_bytes(p.stdout)
        rec=json.loads(rp.read_text())if rp.exists()else dict(case=name,spec=spec)
        rec['process_exit']=p.returncode
        if rec.get('root'):
            code="import sys,json;from pathlib import Path;from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_offline_v1 as O;r=O.verify(sys.argv[1],json.loads(Path(sys.argv[2]).read_text()));r['consumer_forbidden_imports']=[x for x in sys.modules if x.rsplit('.',1)[-1]in ('g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_admission','g7_generator','g6_f3_generator')];Path(sys.argv[3]).write_text(json.dumps(r,indent=2,default=str)+'\\n')"
            vp=dest/(name+'_verification.json');v=subprocess.run([sys.executable,'-c',code,rec['root'],str(dest/'trusted_candidate.json'),str(vp)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if vp.exists():
                result=json.loads(vp.read_text());rec.update(verification_status=result['status'],verification_issues=result['issues'],evidence_gaps=result['evidence_gaps'],consumer_forbidden_imports=result['consumer_forbidden_imports'],calls=result.get('calls'),conditions=result.get('conditions'))
            else:rec.update(verification_status='HARNESS_ERROR',consumer_error=v.stdout.decode())
        (dest/(name+'_result.json')).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec.get(k)for k in ('case','process_exit','verification_status','verification_issues')}),flush=True);return rec
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.jobs)as pool:rows=list(pool.map(one,cases))
    (dest/'matrix.json').write_text(json.dumps(dict(cases=rows,elapsed_seconds=time.monotonic()-start,computational_process_parallelism=a.jobs,subagents_used=False),indent=2)+'\n')
if __name__=='__main__':main()
