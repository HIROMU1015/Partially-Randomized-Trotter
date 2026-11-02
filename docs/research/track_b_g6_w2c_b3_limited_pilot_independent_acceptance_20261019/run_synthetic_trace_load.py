"""Synthetic storage load only; never scientific samples or producer execution."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json,sys,tempfile,time,os
ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G
from trottertracks.algorithm_codesign.g6_w2c_b3_pilot_independent_trace import Decisions

def one(n):
 tasks,total=G.plan(n);combined=0;conditions=[]
 with tempfile.TemporaryDirectory(prefix='pilot-synthetic-log-') as tmp:
  for condition in ('S','P1'):
   path=Path(tmp)/(condition+'.jsonl');prev=V.ZERO;seq=0;run_id=sha256(('INDEPENDENT_SYNTHETIC_'+condition).encode()).hexdigest()
   with path.open('wb',buffering=262144) as f:
    def emit(op,**kw):
     nonlocal seq,prev
     r=dict(seq=seq,run_id=run_id,op=op,**kw);raw=json.dumps(r,separators=(',',':'));prev=sha256((prev+raw).encode()).hexdigest();f.write((json.dumps(dict(payload=raw,previous=V.ZERO if seq==0 else emit.last,sha256=prev),separators=(',',':'))+'\n').encode());emit.last=prev;seq+=1
    emit.last=V.ZERO
    for phase in range(3):
     emit('PHASE_BEGIN',phase=phase)
     for task in [t for t in tasks if t['phase_index']==phase]:
      for axis in ('X','Y'):
       for i in range(task['n']):
        tid=V.trial(dict(task,N=task['n']),axis,i);fields=dict(task_key=task['key'],trial_id=tid)
        emit('RESERVE',**fields);emit('ENTER',**fields)
        # Two synthetic result writes/trial, deliberately more than the normal
        # single-line write, without claiming a bound on real G7 payloads.
        for _ in range(2):emit('WRITE_RESERVE',channel='result',amount=2048);emit('WRITE_ACK',channel='result',amount=2048)
        emit('COMPLETE',**fields)
     emit('PHASE_END',phase=phase)
   reader=Decisions(path,tasks,run_id);h=G.check_operations(reader,tasks,run_id)
   assert h['counts']['completed']==total and not h['outstanding']
   conditions.append(dict(condition=condition,calls=total,trace_bytes=path.stat().st_size,rows=reader.stats['rows'],trace_sha256=reader.stats['sha256'],synthetic_result_bytes=h['reserved_bytes']['result']));combined+=path.stat().st_size
 return dict(profile_n=n,total_static_calls=2*total,conditions=conditions,combined_trace_bytes=combined,log_limit=268435456,trace_only_fits=combined<=268435456,real_G7_payload_bound_established=False,science_calls=0)

if __name__=='__main__':
 with ProcessPoolExecutor() as pool:rows=list(pool.map(one,(64,256,1024)))
 out=Path(sys.argv[1]);assert not out.exists();out.write_text(json.dumps(dict(schema='pilot_independent_synthetic_trace_load_v1',rows=rows,global_budget_covers_journal_state_trace=True,actual_pilot_log_fit='PENDING_REAL_PAYLOAD_AND_FINAL_BRIDGE'),indent=2)+'\n')
 print(json.dumps(rows))
