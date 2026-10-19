"""Read-only self-check consumer. Does not import ports/exercise/static/authority.

Uses fixed offline raw/F01/codec primitives; replicates identity, bit-consumption
and coverage judgments without calling the new Producer implementation.
"""
from pathlib import Path
from hashlib import sha256
import json
import sqlite3
from .. import g6_w2c_b3_offline_result as V
from .. import g6_w2c_b3_parallel_offline_adapter as A
from .. import g6_w2c_b3_offline_cost as C

def ident(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def read(p):return A.safe_values(V.plain(Path(p).read_bytes()))
def shafile(p):return sha256(Path(p).read_bytes()).hexdigest()

def verify(root, trusted, source_root, *, execution_root=None):
    root=Path(root)
    logical=Path(execution_root) if execution_root is not None else root
    result=dict(schema='b3_API_offline_acceptance_v1',status='INCOMPLETE_EVIDENCE',issues=[],
                scientific_GO=False,mandatory_STOP=True,global_bundle_equivalence=False)
    try:
        for p,h in trusted['source_pins'].items():
            V.require(shafile(V.safe_path(source_root,p))==h,'TRUSTED_SOURCE_BYTES:'+p)
        for p,h in trusted['scientific_reference']['input_pins'].items():
            V.require(shafile(V.safe_path(source_root,p))==h,'TRUSTED_INPUT_BYTES:'+p)
        for row in trusted['python']['libraries'].values():
            V.require(shafile(row['path'])==row['sha256'],'RUNTIME_LIBRARY_BYTES')
        V.require(shafile(trusted['python']['executable'])==trusted['python']['executable_sha256'],'RUNTIME_EXECUTABLE_BYTES')
        V.require(trusted['schema']=='b3_closed_api_binding_candidate_v1','CANDIDATE_SCHEMA')
        for k,v in dict(run_ready=False,content_adopted=False,science_release=None,
                        scientific_adoption=None,contract_grants_execution=False,retry=0,resume=False,
                        mandatory_STOP=True,artificial_only=True).items():
            V.require(type(trusted[k]) is type(v) and trusted[k]==v,'CLOSED:'+k)
        r=read(root/'port_report.json')
        V.require(r['schema']=='b3_ARTIFICIAL_port_exercise_v1' and r['candidate_digest']==ident(trusted),'REPORT_BINDING')
        V.require(r['run_root']==str(logical), 'EXTERNAL_EXECUTION_ROOT_BINDING')
        b=read(root/'authority_binding.json')
        scope='ARTIFICIAL_ONLY/API_PORT/'+logical.name
        expected_binding=dict(schema='b3_FAKE_API_authority_binding_v1',formal_source_release=None,
            source_release_candidate=trusted['source_release_candidate'],scientific_input_hash=trusted['scientific_reference_sha256'],
            runtime_hash=ident(trusted['python']),contract_hash=ident(trusted),global_scope=scope,
            global_claim=ident(['FAKE_API_CLAIM',scope,ident(trusted)]),call_budget=16896,result_budget=1<<36,log_budget=1<<28,
            seed_namespace='ARTIFICIAL_API_BINDING_V1',retry=0,resume=False,
            worker_policy='fresh spawn, owned workers; S sequential / P1 fixed candidate eight',storage_destination=str(logical),
            monitoring_STOP_hash=ident(trusted['monitoring']),artificial_only=True,real_grant=None,real_authority=False)
        V.require(ident(b)==ident(expected_binding), 'AUTHORITY_BINDING')
        with sqlite3.connect((root/'FAKE_ONLY.sqlite3').resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            claims=db.execute('SELECT scope,claim,binding,status,spent FROM claims').fetchall()
            history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
        V.require(claims==[(b['global_scope'],b['global_claim'],ident(b),'STOPPED',1)],'FAKE_ONE_SHOT_SNAPSHOT')
        V.require(history==[(1,scope,'CLAIM_COMMITTED',ident(b)),(2,scope,'STOP_COMMITTED',ident(b))],'FAKE_SAVED_HISTORY')
        V.require(read(root/'fake_snapshot.json')==dict(schema='b3_FAKE_ONLY_snapshot_v1',claim=list(claims[0]),
            history=[list(x) for x in history],real_authority=False),'FAKE_SNAPSHOT_BINDING')
        V.require(type(r['n']) is int and 16<=r['n']<=256,'ARTIFICIAL_PROFILE_RANGE')
        tasks=[];offset=0
        for j,mode in enumerate(('S','P1')):
            for phase,(name,pa) in enumerate((('correctness','baseline'),('profile','baseline'),('profile','instrumented'))):
                for i in range(8):
                    arm=V.ARMS[i//2];layer=('A','B')[i%2];n=16 if phase==0 else r['n']
                    tasks.append(dict(key=f'{mode}__{name}_{pa}_{arm}_{layer}',condition=mode,phase_index=3*j+phase,
                                      phase=name,pass_name=pa,arm=arm,layer=layer,pair_index=i,n=n,offset=offset))
                    offset+=2*n
        V.require(r['planned']==offset and len(r['rows'])<=48,'STATIC_CALL_PLAN')
        comparisons={};domain_count={};seen=set();pairs=[];calls=0;sum_pair_cpu=0
        for index,row in enumerate(r['rows']):
            t=row['task'];V.require(t==tasks[index],'TASK_ORDER_IDENTITY')
            V.require(read(root/'ports'/(t['key']+'.worker.json'))==row,'WORKER_REPORT_BYTES')
            V.require(not row['forbidden_imports'] and row['guard_installed'] is True,'WORKER_IMPORT_GUARD')
            path=root/'ports'/(t['key']+'.jsonl')
            pc=A.verify_worker(path,t,row)
            V.require(pc['status'] not in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_F01:'+str(pc['issues']))
            pairs.append(dict(key=t['key'],status=pc['status']))
            V.require(C.partition(row['coarse'],row['reconciliation'])['outer_total_ns']==[row['cpu_ns'],row['wall_ns']], 'COARSE_OUTER')
            if row['fine'] is not None:
                V.require(C.partition(row['fine'],row['fine_reconciliation'])['outer_total_ns']==[row['cpu_ns'],row['wall_ns']], 'FINE_OUTER')
            sum_pair_cpu+=row['cpu_ns']
            side=read(root/'ports'/(t['key']+'.port.json'))
            V.require(side['task']==t,'PORT_TASK_IDENTITY')
            bylocal={x['local_trial_id']:x for x in side['records']}
            V.require(len(bylocal)==len(side['records']),'DUPLICATE_PORT_ID')
            counters={'X':0,'Y':0}
            raw_visible=0;visible_ids=set()
            for envelope in V.Reader(path):
                p=envelope.payload
                if 'trial_id' not in p or p.get('kind') not in V.KINDS:continue
                raw_visible+=1
                local=p['trial_id'];record=bylocal.get(local)
                V.require(record is not None,'MISSING_PORT_RECORD')
                visible_ids.add(local)
                fields=local.split('/');V.require(len(fields)==6,'LOCAL_ID_SCHEMA')
                phase,pa,arm,layer,axis,trial=fields
                V.require((phase,pa,arm,layer)==(t['phase'],t['pass_name'],t['arm'],t['layer']) and axis in ('X','Y'),'PORT_CELL_IDENTITY')
                gid=logical.name
                expected='/'.join((gid,t['condition'],local))
                V.require(record['global_trial_id']==expected and expected not in seen,'GLOBAL_TRIAL_IDENTITY')
                seen.add(expected)
                tr=record['consumption'];domain='ARTIFICIAL_API_BINDING_V1/'+arm+'/'+axis
                V.require(tr['schema']=='b3_ARTIFICIAL_consumption_v1' and record['domain']==tr['domain']==domain,'SEED_DOMAIN')
                expected_draws=({'ZERO_R':1,'ZERO_C':2,'EVENT':3}[p['kind']] if arm=='full_return' else 2)
                V.require(len(tr['draws'])==expected_draws and (arm=='full_return' or p['kind']=='EVENT'),'CONSUMPTION_COUNT')
                for draw in tr['draws']:
                    V.require(all(type(draw.get(k)) is int for k in ('width','counter_before','counter_after','discarded_bits')),'CONSUMPTION_INTEGER_TYPE')
                    counter=counters[axis]
                    raw=sha256(domain.encode()+counter.to_bytes(8,'big')).digest()
                    value=int.from_bytes(raw,'big')>>96
                    V.require(draw==dict(width=160,counter_before=counter,counter_after=counter+1,
                        value_hex=hex(value),discarded_bits=96),'ARTIFICIAL_CONSUMPTION_REPLAY')
                    counters[axis]+=1
                V.require(type(tr['counter_after']) is int and type(tr['bits_consumed']) is int and
                          tr['counter_after']==counters[axis] and tr['bits_consumed']==160*counters[axis], 'CONSUMPTION_TOTAL')
                evhash=V.event_hash(p)
                V.require(record['event_digest']==evhash,'PORT_EVENT_IDENTITY')
                key=(arm,axis,int(trial))
                parity=(evhash,ident(tr))
                V.require(key not in comparisons or comparisons[key]==parity,'S_P1_A_B_PASS_PREFIX')
                comparisons[key]=parity;domain_count[domain]=domain_count.get(domain,0)+1
                calls+=1
            extra=set(bylocal)-visible_ids
            if extra:
                f=(row.get('failure') or {}).get('trial_failure_provenance') or {}
                current=f.get('current_attempt') or {}
                V.require(len(extra)==1 and extra=={current.get('trial_id')} and current.get('result_acquired') is True
                          and current.get('trial_completed') is False and current.get('stage')=='ANGLE_AND_REQUERY', 'UNCOMMITTED_PORT_RECORD')
            V.require(raw_visible+len(extra)==len(side['records']),'PORT_RECORD_VISIBLE_PREFIX')
        outer=r['outer']
        V.require(sum_pair_cpu<=outer['all_worker_cpu_ns'] and outer['parent_cpu_ns']>=0 and outer['wall_ns']>0,'CPU_OUTER_NO_DOUBLE_ADD')
        V.require(not r['forbidden_imports'] and r['scientific_GO'] is False and r['global_bundle_equivalence'] is False,'SCIENCE_CLOSED')
        result.update(calls=calls,pairs=pairs,domains=len(domain_count),cpu=dict(outer,
            total_cpu_ns=outer['parent_cpu_ns']+outer['all_worker_cpu_ns'],pair_cpu_ns=sum_pair_cpu),
            snapshot_is_complete_transition_proof=False)
        if not r['complete']:
            V.require(len(r['rows'])<48 or any(x['status']!='OK' for x in r['rows']),'FALSE_FAILURE_FLAG')
            result['status']='VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE' if any(x['status']!='OK' for x in r['rows']) else 'INCOMPLETE_EVIDENCE'
            return result
        V.require(len(r['rows'])==48 and calls==offset and all(x['status']=='OK' for x in r['rows'])
                  and len(domain_count)==8,'NORMAL_ALL_16_CELLS_6_PHASES')
        caller=read(root/'port_caller.json')
        V.require(caller['schema']=='b3_ARTIFICIAL_port_caller_v1' and
                  caller['candidate_digest']==ident(trusted) and caller['report_sha256']==shafile(root/'port_report.json')
                  and caller['report_write_flush_close_observed'] is True and not caller['forbidden_imports'],'CALLER_POST_CLOSE')
        result['status']='VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE'
    except FileNotFoundError as exc:
        result['issues'].append('MISSING:'+str(exc.filename))
    except (V.Invalid,KeyError,ValueError,TypeError,sqlite3.Error) as exc:
        result['status']='INTEGRITY_MISMATCH';result['issues'].append(str(exc))
    return result
