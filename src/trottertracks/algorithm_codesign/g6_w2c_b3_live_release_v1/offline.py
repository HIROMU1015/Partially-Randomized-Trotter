"""Read-only same-author exact-document, trace, counter and receipt checker.

Does not construct a generator, sample, import a scientific module or open a
store. Shared Codec/Result validation is explicitly not independent review.
"""
from pathlib import Path
from fractions import Fraction as F
from hashlib import sha256
from collections import Counter
import json
from .guard import require
from . import contract as K
from .bits import validate_trace
from ..g6_w2c_b3_live_preflight_v1 import spec as S
from ..g6_w2c_b3_real_adapter_v1 import codec as C,result as R
from .. import g6_w2c_b3_offline_result as V

ARTIFICIAL_SOURCE='ARTIFICIAL_API_SHAPED_RUNTIME_RELEASE_V1'


def read(path):return json.loads(Path(path).read_text())
def driver_hash(value):return sha256(C.driver_canonical(value).encode()).hexdigest()


def state(task,m):
    return dict(artificial=True,p=tuple(map(F,S.INPUT['p'])),x=F(S.INPUT['x']),m=5,arm=task['arm'],
        H=160,K=256,eta=F(S.PRECISION['eta']),rho=F(S.PRECISION['rho']),order=(F(1,2),F(1,2)),
        labels=(F(1,2),F(1,4),F(1,4)),artificial_acceptance=F(0) if m['scenario']=='coverage'
            and task['arm']=='full_return' else F(3,4))


def preflight_id(task,m):
    if task['layer']=='A':return 'G7_UNMONITORED_LAYER_A'
    if task['arm']=='full_return':return C.Document.create(dict(artificial=True,ready=True)).digest
    return driver_hash(dict(artificial_snapshot=True,held_state=state(task,m),scientific_canonical_snapshot_verified=False))


def context(c,m,task,tid):
    full_B=task['arm']=='full_return' and task['layer']=='B'
    return dict(trial_id=tid,input_id='ARTIFICIAL_INPUT_BINDING' if full_B else S.digest(S.INPUT),
        policy_id='ARTIFICIAL_PRECISION_BINDING' if full_B else S.digest([S.PRECISION,S.ANGLE]),
        source_id=ARTIFICIAL_SOURCE if full_B else c['source_release_candidate'],
        preflight_id=preflight_id(task,m),sigma=1)


def payload(p,c,m,task):
    ev=p['evidence'];a=ev['adapted'];tid=p['trial_id'];phase,pa,arm,layer,axis,index=tid.split('/')
    index=int(index);require((phase,pa,arm,layer)==('correctness','baseline',task['arm'],task['layer'])
        and axis in ('X','Y') and 0<=index<16,'TRIAL_ID')
    expected=context(c,m,task,tid);source=expected['source_id']
    original=C.restore(a['codec'],source_identity=source)
    if arm=='full_return' and layer=='B':fresh=R.adapt_f3(original,expected=expected)
    else:
        observation=C.restore(a['g7_observation'],source_identity=source) if arm=='full_return' else original
        data=observation.read();fresh=R.adapt_g7(data['event'],observation=observation,expected=expected,arm=arm)
    require(C.driver_canonical(a)==C.driver_canonical(fresh),'ADAPTED_ORIGINAL_PROJECTION')
    require(p['kind']==fresh['driver_kind'] and C.driver_canonical(p['event'])==C.driver_canonical(fresh['event']),
            'RESULT_EVENT_PROJECTION')
    consumption=ev['consumption'];validate_trace(consumption,arm,axis)
    trace=original.read()['trace']
    require([(d['width'],d['value']) for d in trace['draws']]==
        [(d['width'],int(d['value_hex'],16)) for d in consumption['draws']],'ORIGINAL_TRACE_CONSUMPTION')
    require(len(consumption['draws'])<=(7 if arm=='full_return' else 6),'STATIC_DRAW_BOUND')
    connection=ev['connection'];offset=task['offset']+(0 if axis=='X' else 16)+index
    expected_session=driver_hash(dict(source=c['source_release_candidate'],manifest=S.digest(m),arm=arm,layer=layer,
        state=state(task,m),input=S.INPUT,precision=S.PRECISION,angle=S.ANGLE,preflight=preflight_id(task,m)))
    require(connection['global_trial_id']==m['run_id']+'/'+task['condition']+'/'+tid and
        connection['global_offset']==offset and connection['task_key']==task['key'] and
        connection['source_binding']==c['source_release_candidate'] and connection['manifest_binding']==S.digest(m)
        and connection['runtime_binding']==m['runtime_binding'] and connection['route']==S.route(arm,layer)
        and connection['session']==expected_session,'GLOBAL_CONNECTION_BINDING')
    require(ev['artificial_only'] is True and ev['live_API_compatibility']=='NOT_ESTABLISHED','NO_SCIENCE_CERTIFICATE')
    require(not ev['guard']['core_loaded'] and not ev['guard']['core_import_attempts'],'CORE_LOADED_OR_ATTEMPTED')
    require(connection['worker_pid']==ev['guard']['pid'],'WORKER_GUARD_BINDING')
    require(ev['boundary_counts']['scheduled_sampling']==index+(16 if axis=='Y' else 0)+1
            and ev['boundary_counts']['primary_constructions']==1,'BOUNDARY_COUNTERS')
    monitor=ev['monitoring']
    for kind in ('cpu','wall'):
        require(monitor['outer_'+kind+'_ns']==sum(monitor['exclusive_'+kind+'_ns'].values())+
                monitor['residual_'+kind+'_ns'],'EXCLUSIVE_COST_RECONCILIATION')
    local=index+(16 if axis=='Y' else 0)
    if layer=='B':
        ident=driver_hash(dict(input=S.INPUT,policy=S.PRECISION,angle=S.ANGLE));sid=driver_hash(state(task,m))
        before=local if arm=='full_return' else 2*local
        seal=C.DriverDocument.create(dict(arm=arm,identity=ident,state_id=sid,payload=original.document,
                                         completed_calls_before=before,confidence_certified=False))
        expected_audit=R.canonical_audit(seal,arm=arm,identity=ident,state_id=sid,completed_calls_before=before)
        require(C.driver_canonical(ev['audit'])==C.driver_canonical(expected_audit),'AUDIT_ORIGINAL_SEAL')
    else:require(ev['audit'] is None,'A_AUDIT_FORBIDDEN')
    if layer=='B' and fresh['classification']=='ACCEPTED':
        angle=p['angle'];require(type(angle) is dict,'ANGLE_REQUIRED')
        angle_doc=C.restore(angle['angle'],source_identity=source)
        data=angle_doc.read()
        require(data['phi_interval']==(F(1,8),F(1,8)+F(1,10**8)),'ARTIFICIAL_ANGLE_FIXTURE')
        e=fresh['event'];query=dict(word=e['word'],child=e['child'])
        if arm=='full_return':
            expected_angle=R.angle_binding(fresh,requery_input=query,requery_event=e,angle_record=angle_doc,synthetic=True)
        else:
            query.update(group_index=original.read()['group_index'],raw_word=e['raw_word'])
            expected_angle=R.canonical_angle_binding(fresh,requery_input=query,requery_event=e,angle_record=angle_doc,
                input_payload=S.INPUT,policy_payload=S.PRECISION,synthetic=True)
            seal=C.DriverDocument.create(dict(arm=arm,identity=ident,state_id=sid,payload=angle_doc.document,
                completed_calls_before=before+1,confidence_certified=False))
            extra=R.canonical_audit(seal,arm=arm,identity=ident,state_id=sid,completed_calls_before=before+1)
            require(C.driver_canonical(ev['angle_audit'])==C.driver_canonical(extra),'ANGLE_AUDIT_SEAL')
        require(C.driver_canonical(angle)==C.driver_canonical(expected_angle),'ANGLE_REQUERY_EVIDENCE')
    else:require(p['angle'] is None,'UNEXPECTED_ANGLE')
    return dict(offset=offset,gid=connection['global_trial_id'],event=fresh['event'],
                classification=fresh['classification'],consumption=consumption,worker=connection['worker_pid'])


def inspect(root,c,m):
    root=Path(root);out=root/'output'
    missing=[n for n in ('final.json','delivery.json','caller.json','host_outer.json') if not (out/n).is_file()]
    if missing:return dict(status='INCOMPLETE_EVIDENCE',missing=missing)
    report=read(out/'final.json');delivery=read(out/'delivery.json');caller=read(out/'caller.json');outer=read(out/'host_outer.json')
    require(delivery['final_sha256']==sha256((out/'final.json').read_bytes()).hexdigest(),'DELIVERY_FINAL')
    require(caller['observed'] is True and caller['delivery_sha256']==sha256((out/'delivery.json').read_bytes()).hexdigest(),'CALLER')
    require(report['candidate_binding']==S.digest(c) and report['manifest_binding']==S.digest(m),'REPORT_BINDING')
    require(outer['total_cpu_ns']==outer['parent_cpu_ns']+outer['all_worker_cpu_ns'],'CPU_COUNT_ONCE')
    require(not outer['coarse_and_fine_added'],'COST_PARTITIONS_NOT_ADDED')
    for partition in outer['exclusive_partitions_reconciled_separately'].values():
        require(partition['outer_residual_cpu_ns']>=0 and partition['worker_exclusive_cpu_ns']+
                partition['outer_residual_cpu_ns']==outer['total_cpu_ns'],'GLOBAL_EXCLUSIVE_RECONCILIATION')
    trace=[];previous='0'*64;stopped=False
    for seq,line in enumerate((out/'operations.jsonl').read_bytes().splitlines()):
        row=json.loads(line);require(row['previous']==previous,'TRACE_CHAIN')
        previous=sha256(previous.encode()+row['payload'].encode()).hexdigest();require(row['sha256']==previous,'TRACE_DIGEST')
        value=json.loads(row['payload']);require(value['seq']==seq and value['run_id']==m['run_id'],'TRACE_ID')
        if value['op']=='STOP':stopped=True
        if value['op']=='ENTER':require(not stopped,'ENTER_AFTER_STOP')
        trace.append(value)
    marker=read(out/'STOP_marker.json');require(marker['mandatory_STOP'] is True,'MANDATORY_STOP')
    require(sha256((out/'STOP_marker.json').read_bytes()).hexdigest()==report['STOP_marker_sha256'],'STOP_MARKER_HASH')
    claim=marker['store']['claim'];require(claim[:3]==[m['global_scope'],m['global_claim'],S.digest(m)]
        and claim[3] in ('STOPPED','UNKNOWN_STOPPED') and claim[4]==1,'CLAIM_TERMINAL_SPENT')
    if not report['complete']:
        require(stopped and not (out/'receipt.json').exists(),'STOP_NO_SUCCESS_RECEIPT')
        return dict(status=report['classification'],completed=report['states']['COMPLETED'],conditions=report['conditions'],
            states=report['states'],failures=report['failures'],ENTER_after_STOP=0,unknown_credit=report['output']['unknown_credit'])
    receipt=read(out/'receipt.json');require(report['receipt_sha256']==sha256((out/'receipt.json').read_bytes()).hexdigest(),'RECEIPT_HASH')
    require(receipt['candidate_binding']==S.digest(c) and receipt['manifest_binding']==S.digest(m) and receipt['completed']==512,'RECEIPT_BINDING')
    require(receipt['state_sha256']==sha256((out/'trial_states.bin').read_bytes()).hexdigest()
        and receipt['trace_sha256']==sha256((out/'operations.jsonl').read_bytes()).hexdigest(),'RECEIPT_STATE_TRACE')
    require((out/'trial_states.bin').read_bytes()==bytes([4])*512,'ALL_512_COMPLETED')
    require(claim[5]==report['receipt_sha256'],'STORE_RECEIPT_LINK')
    complete={r['global_offset']:r['global_trial_id'] for r in trace if r['op']=='COMPLETE'}
    require(set(complete)==set(range(512)),'COMPLETE_TRACE_COVERAGE')
    rows={r['task']['key']:r for r in report['pairs']};prefix={};cells=[];global_counts=Counter();guard_pids=set();full=0
    for task in m['tasks']:
        path=out/(task['key']+'.jsonl');require(sha256(path.read_bytes()).hexdigest()==rows[task['key']]['file_sha'],'PAIR_FILE_HASH')
        previous='0'*64;expected_counter={'X':0,'Y':0};counts={x:Counter() for x in ('X','Y')};last={}
        for i,line in enumerate(path.read_bytes().splitlines()):
            envelope=json.loads(line);require(envelope['previous']==previous,'PAIR_CHAIN')
            previous=sha256((previous+envelope['payload']).encode()).hexdigest();require(envelope['sha256']==previous,'PAIR_DIGEST')
            p=V.decode(envelope['payload']);joined=payload(p,c,m,task);axis=p['axis'];index=int(p['trial_id'].rsplit('/',1)[-1])
            require((axis,index)==(('X' if i<16 else 'Y'),i%16),'PAIR_TRIAL_ORDER')
            require(complete[joined['offset']]==joined['gid'],'EVIDENCE_COMPLETE_JOIN')
            require(joined['worker']==rows[task['key']]['pid'],'WORKER_PID_JOIN');guard_pids.add(joined['worker'])
            cons=joined['consumption'];require(cons['counter_before']==expected_counter[axis],'BIT_PREFIX_POSITION')
            expected_counter[axis]=cons['counter_after']
            key=(task['arm'],axis,index);fp=C.fingerprint(dict(event=joined['event'],consumption=cons))
            require(key not in prefix or prefix[key]==fp,'AB_SP1_PREFIX');prefix[key]=fp
            counts[axis][joined['classification']]+=1;last=p['evidence']['boundary_counts'];full+=1
        for axis,values in counts.items():
            require(sum(values.values())==16,'FIXED_TRIAL_COUNT')
            cells.append(dict(mode=task['condition'],arm=task['arm'],layer=task['layer'],axis=axis,
                              completed=16,accepted=values['ACCEPTED'],classifications=dict(values)))
        global_counts.update(last)
    for name in ('scheduled_sampling','primary_constructions','G7_make_generator','F3_prepare','canonical_snapshot',
                 'AuditParity_construction','B_sample_audit','B_angle','B_event_requery','canonical_angle','Full_angle',
                 'Full_sample_packet','Full_angle_packet'):
        require(global_counts[name]<=m['call_budgets'][name],'GLOBAL_BOUNDARY_BUDGET:'+name)
    require(global_counts['bit_draw_requests']<=3200,'GLOBAL_DRAW_BUDGET')
    classification='INCONCLUSIVE_COVERAGE' if any(x['accepted']==0 for x in cells) else 'ARTIFICIAL_TECHNICAL_COMPLETE'
    require(report['classification']==classification==receipt['classification'],'COVERAGE_CLASSIFICATION')
    return dict(status=classification,full_documents=full,trials=512,cells=cells,boundary_counts=dict(global_counts),
        worker_pids=sorted(guard_pids),core_loaded=[],core_import_attempts=[],receipt_caller_join=True,
        host_outer=outer,science_correctness_certified=False,live_compatibility='NOT_ESTABLISHED')
