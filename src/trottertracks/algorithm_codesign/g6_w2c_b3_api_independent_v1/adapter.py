"""Independent offline port adapter. Imports only existing independent consumers.

No producer port/exercise/authority/self-checker, scheduler or science import.
Saved caller evidence and failure-sidecar attempts are checked separately from
completed F01 trials. Global Bundle acceptance is a separate result.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sqlite3
from . import contract as K, bits as B
from .. import g6_w2c_b3_offline_result as V
from .. import g6_w2c_b3_parallel_offline_adapter as A
from .. import g6_w2c_b3_offline_cost as C
from ..g6_w2c_b3_bundle_independent_identity_v1 import plan

def read(p):return A.safe_values(K.read(p))
def integer(v,why,positive=False):V.require(type(v)is int and (v>0 if positive else v>=0),why)

def verify(root,c,source_root,*,trusted,execution_root=None):
    root=Path(root);logical=Path(execution_root) if execution_root else root
    result=dict(schema='b3_API_independent_port_verification_v1',status='INCOMPLETE_EVIDENCE',issues=[],
                scientific_GO=False,mandatory_STOP=True,global_bundle_equivalence=False)
    try:
        result['identity']=K.candidate(c,trusted,source_root)
        r=read(root/'port_report.json');K.equal(r['schema'],'b3_ARTIFICIAL_port_exercise_v1','REPORT_SCHEMA')
        K.equal(r['candidate_digest'],K.digest(c),'REPORT_CANDIDATE_HASH');K.equal(r['run_root'],str(logical),'LOGICAL_ROOT')
        K.equal(r['measured_SOURCE'],K.MEASURED,'MEASURED_SOURCE')
        for k,v in dict(scientific_GO=False,mandatory_STOP=True,global_bundle_equivalence=False,forbidden_imports=[],fixed_bundle_injection='UNIMPLEMENTED_IN_IMMUTABLE_BUNDLE').items():K.equal(r[k],v,'REPORT_CLOSED:'+k)
        result['authority']=K.authority(root,c,logical)
        tasks,total=plan(r['n']);K.equal(r['planned'],total,'CALL_PLAN')
        V.require(type(r['complete'])is bool and type(r['rows'])is list and len(r['rows'])<=48,'REPORT_COVERAGE_TYPE')
        pairs=[];calls=0;inflight=0;sum_cpu=0;global_ids=set();prefixes={};domains=Counter();kinds=Counter();failed_phases=set()
        for i,row in enumerate(r['rows']):
            t=tasks[i];K.equal(row['task'],t,'TASK_ORDER_BINDING')
            V.require(not failed_phases or t['phase_index'] in failed_phases,'PHASE_AFTER_FAILURE')
            base=root/'ports'/t['key'];K.equal(read(str(base)+'.worker.json'),row,'SAVED_WORKER_REPORT')
            K.equal(row['forbidden_imports'],[],'WORKER_SCIENCE_IMPORT');K.equal(row['guard_installed'],True,'WORKER_GUARD')
            V.require(row['status'] in ('OK','FAILED'),'WORKER_STATUS');integer(row['pid'],'WORKER_PID',True)
            pc=A.verify_worker(Path(str(base)+'.jsonl'),t,row)
            V.require(pc['status'] in ('VERIFIED_COMPLETE','TECHNICAL_FAILURE'),'F01_PAIR:'+str(pc['issues']))
            if row['status']=='OK':
                V.require(row['failure'] is None and row['outcome'] is not None and pc['status']=='VERIFIED_COMPLETE','SUCCESS_F01_COMPLETE')
            else:
                failed_phases.add(t['phase_index']);f=row['failure'];f01=f['trial_failure_provenance'];cur=f01['current_attempt']
                V.require(row['outcome'] is None and f['type']=='Rejected' and f01['type']==f['type'] and f01['reason']==f['reason'],'TYPED_FAILURE_BINDING')
                for k,v in dict(mandatory_STOP=True,is_random_zero=False,retry=0,resume=False).items():K.equal(f01[k],v,'F01_CLOSED:'+k)
                allowed={'ARTIFICIAL_GENERATOR_FAILURE':('SAMPLING',False),'ARTIFICIAL_AUDIT_FAILURE':('SAMPLING',False),'ARTIFICIAL_ANGLE_REQUERY_FAILURE':('ANGLE_AND_REQUERY',True)}
                V.require(f['reason'] in allowed,'UNKNOWN_ARTIFICIAL_FAILURE');stage,acquired=allowed[f['reason']]
                K.equal(cur['stage'],stage,'FAILURE_STAGE');K.equal(cur['result_acquired'],acquired,'FAILURE_ACQUIRED')
                K.equal(cur['trial_completed'],False,'FAILED_NOT_COMPLETED')
            for name in ('cpu_ns','wall_ns','output_bytes','output_rows'):integer(row[name],'WORKER_INT:'+name)
            K.equal(C.partition(row['coarse'],row['reconciliation'])['outer_total_ns'],[row['cpu_ns'],row['wall_ns']],'COARSE_CPU_PARTITION')
            if t['pass_name']=='instrumented':
                V.require(type(row['fine'])is dict,'FINE_PRESENT')
                K.equal(C.partition(row['fine'],row['fine_reconciliation'])['outer_total_ns'],[row['cpu_ns'],row['wall_ns']],'FINE_CPU_ALTERNATIVE')
            else:K.equal(row['fine'],None,'BASELINE_NO_FINE')
            sum_cpu+=row['cpu_ns'];side=read(str(base)+'.port.json');K.equal(side['task'],t,'SIDECAR_TASK')
            records=side['records'];V.require(type(records)is list,'SIDECAR_RECORD_LIST');lookup={x['local_trial_id']:x for x in records}
            V.require(len(records)==len(lookup),'DUPLICATE_PORT_LOCAL_ID');visible=set();counters={'X':0,'Y':0};indices={'X':0,'Y':0}
            reader=V.Reader(Path(str(base)+'.jsonl'))
            for envelope in reader:
                p=envelope.payload
                if p.get('kind') not in V.KINDS or 'trial_id' not in p:continue
                record=lookup.get(p['trial_id']);V.require(record is not None,'MISSING_PORT_RECORD')
                kind=B.trial(p,record,t,logical.name,counters,indices);visible.add(p['trial_id']);calls+=1;kinds[kind]+=1
                gid=record['global_trial_id'];V.require(gid not in global_ids,'DUPLICATE_GLOBAL_TRIAL');global_ids.add(gid)
                axis=p['axis'];key=(t['arm'],axis,indices[axis]-1);value=(record['event_digest'],K.digest(record['consumption']))
                if key in prefixes:K.equal(value,prefixes[key],'S_P1_AB_PHASE_PASS_PREFIX')
                prefixes[key]=value;domains[record['domain']]+=1
            extras=set(lookup)-visible
            if extras:
                V.require(row['status']=='FAILED' and len(extras)==1,'UNCOMMITTED_SIDECAR_COUNT')
                f01=row['failure']['trial_failure_provenance'];cur=f01['current_attempt']
                K.equal(extras,{cur['trial_id']},'UNCOMMITTED_TRIAL_ID');K.equal(cur['stage'],'ANGLE_AND_REQUERY','UNCOMMITTED_STAGE')
                record=lookup[cur['trial_id']];axis=cur['axis'];kind,event,tr,end=B.replay(t['arm'],axis,counters[axis])
                K.equal(f01['inflight_event'],event,'INFLIGHT_EVENT_BINDING')
                angle=dict(artificial=True,not_a_mathematical_enclosure=True,event_requery_surrogate=sha256(V.canonical(event).encode()).hexdigest(),actual_angle=None)
                B.trial(dict(trial_id=cur['trial_id'],axis=axis,kind=kind,event=event,evidence=dict(artificial=True,consumption=tr),angle=angle),record,t,logical.name,counters,indices)
                inflight+=1
            # Engine bit summaries are completed only; failure inflight consumption is separate.
            if row['status']=='OK':
                for ar in row['outcome']['axes']:K.equal(ar['H_bits_supplied'],160*counters[ar['axis']],'ENGINE_BIT_COUNT')
            pairs.append(dict(key=t['key'],status=pc['status'],committed_trials=pc.get('committed_trials'),uncommitted_sidecars=len(extras)))
        outer=r['outer']
        for k in ('parent_cpu_ns','all_worker_cpu_ns','wall_ns'):integer(outer[k],'OUTER_CPU:'+k,k=='wall_ns')
        V.require(sum_cpu<=outer['all_worker_cpu_ns'],'ALL_WORKERS_COVER_PAIR_CPU')
        # Every produced exercise has a closed report/caller record, including failures.
        caller=read(root/'port_caller.json')
        K.equal(caller,dict(schema='b3_ARTIFICIAL_port_caller_v1',report_sha256=V.file_digest(root/'port_report.json'),report_write_flush_close_observed=True,forbidden_imports=[],candidate_digest=K.digest(c)),'CALLER_POST_CLOSE')
        result.update(calls=calls,uncommitted_sidecars=inflight,pairs=pairs,domains=len(domains),kinds=dict(kinds),
            cpu=dict(outer,total_cpu_ns=outer['parent_cpu_ns']+outer['all_worker_cpu_ns'],pair_cpu_ns=sum_cpu),
            physical_port_tree_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
            snapshot_is_complete_transition_proof=False,caller_witness_own_close_proven=False)
        if r['complete']:
            V.require(len(r['rows'])==48 and not failed_phases and calls==total and len(domains)==8,'GRID_COMPLETE')
            result['status']='VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE'
        elif failed_phases:
            V.require(len(r['rows'])%8==0 and len(failed_phases)==1,'FAILURE_PHASE_PREFIX')
            result['status']='VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE'
        else:result['issues'].append('NO_TYPED_FAILURE_OR_COMPLETE_GRID')
    except FileNotFoundError as e:result['issues'].append('MISSING:'+str(e.filename))
    except (V.Invalid,KeyError,ValueError,TypeError,IndexError,OverflowError,sqlite3.Error) as e:
        # Malformed evidence must yield a result; interrupts/process termination propagate.
        result['status']='INTEGRITY_MISMATCH';result['issues'].append(type(e).__name__+':'+str(e))
    return result
