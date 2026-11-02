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
from . import contract as K,metadata as M
from .bits import validate_trace,validate_consumption
from . import spec as S
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


def payload(p,c,m,task,*,setup=None,artificial=True):
    ev=p['evidence'];a=ev['adapted'];tid=p['trial_id'];phase,pa,arm,layer,axis,index=tid.split('/')
    index=int(index);require((phase,pa,arm,layer)==(task['phase'],task['pass_name'],task['arm'],task['layer'])
        and axis in ('X','Y') and 0<=index<task['n'],'TRIAL_ID')
    expected=context(c,m,task,tid) if artificial else dict(setup['context'],trial_id=tid)
    source=expected['source_id']
    original=C.restore(a['codec'],source_identity=source)
    if arm=='full_return' and layer=='B':fresh=R.adapt_f3(original,expected=expected)
    else:
        observation=C.restore(a['g7_observation'],source_identity=source) if arm=='full_return' else original
        data=observation.read();fresh=R.adapt_g7(data['event'],observation=observation,expected=expected,arm=arm)
    require(C.driver_canonical(a)==C.driver_canonical(fresh),'ADAPTED_ORIGINAL_PROJECTION')
    require(p['kind']==fresh['driver_kind'] and C.driver_canonical(p['event'])==C.driver_canonical(fresh['event']),
            'RESULT_EVENT_PROJECTION')
    consumption=ev['consumption']
    if artificial:validate_trace(consumption,arm,axis)
    else:validate_consumption(consumption,arm,axis,namespace=m['seed_namespace'],artificial=False)
    trace=original.read()['trace']
    require([(d['width'],d['value']) for d in trace['draws']]==
        [(d['width'],int(d['value_hex'],16)) for d in consumption['draws']],'ORIGINAL_TRACE_CONSUMPTION')
    require(len(consumption['draws'])<=(7 if arm=='full_return' else 6),'STATIC_DRAW_BOUND')
    connection=ev['connection'];offset=task['offset']+(0 if axis=='X' else task['n'])+index
    expected_session=driver_hash(dict(source=c['source_release_candidate'],manifest=S.digest(m),arm=arm,layer=layer,
        state=state(task,m) if artificial else setup['state'],input=S.INPUT,precision=S.PRECISION,angle=S.ANGLE,
        preflight=preflight_id(task,m) if artificial else setup['context']['preflight_id']))
    require(connection['global_trial_id']==m['run_id']+'/'+task['condition']+'/'+tid and
        connection['global_offset']==offset and connection['task_key']==task['key'] and
        connection['source_binding']==c['source_release_candidate'] and connection['manifest_binding']==S.digest(m)
        and connection['runtime_binding']==m['runtime_binding'] and connection['route']==S.route(arm,layer)
        and connection['session']==expected_session,'GLOBAL_CONNECTION_BINDING')
    require(ev['artificial_only'] is artificial and ev['live_API_compatibility']=='NOT_ESTABLISHED','NO_SCIENCE_CERTIFICATE')
    if artificial:require(not ev['guard']['core_loaded'] and not ev['guard']['core_import_attempts'],'CORE_LOADED_OR_ATTEMPTED')
    else:
        from .guard import CORE
        require(set(n.rsplit('.',1)[-1] for n in ev['guard']['core_loaded'])==set(CORE)
            and ev['guard']['source_binding']==c['source_release_candidate']
            and ev['guard']['admission_binding']==S.digest(m['approval_record']),'LIVE_WORKER_SOURCE_ADMISSION')
    require(connection['worker_pid']==ev['guard']['pid'],'WORKER_GUARD_BINDING')
    require(ev['boundary_counts']['scheduled_sampling']==index+(task['n'] if axis=='Y' else 0)+1
            and ev['boundary_counts']['primary_constructions']==1,'BOUNDARY_COUNTERS')
    monitor=ev['monitoring']
    for kind in ('cpu','wall'):
        require(monitor['outer_'+kind+'_ns']==sum(monitor['exclusive_'+kind+'_ns'].values())+
                monitor['residual_'+kind+'_ns'],'EXCLUSIVE_COST_RECONCILIATION')
    local=index+(task['n'] if axis=='Y' else 0)
    if layer=='B':
        ident=driver_hash(dict(input=S.INPUT,policy=S.PRECISION,angle=S.ANGLE));sid=driver_hash(state(task,m) if artificial else setup['state'])
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
        if artificial:require(data['phi_interval']==(F(1,8),F(1,8)+F(1,10**8)),'ARTIFICIAL_ANGLE_FIXTURE')
        e=fresh['event'];query=dict(word=e['word'],child=e['child'])
        if arm=='full_return':
            expected_angle=R.angle_binding(fresh,requery_input=query,requery_event=e,angle_record=angle_doc,synthetic=artificial)
        else:
            query.update(group_index=original.read()['group_index'],raw_word=e['raw_word'])
            expected_angle=R.canonical_angle_binding(fresh,requery_input=query,requery_event=e,angle_record=angle_doc,
                input_payload=S.INPUT,policy_payload=S.PRECISION,synthetic=artificial)
            seal=C.DriverDocument.create(dict(arm=arm,identity=ident,state_id=sid,payload=angle_doc.document,
                completed_calls_before=before+1,confidence_certified=False))
            extra=R.canonical_audit(seal,arm=arm,identity=ident,state_id=sid,completed_calls_before=before+1)
            require(C.driver_canonical(ev['angle_audit'])==C.driver_canonical(extra),'ANGLE_AUDIT_SEAL')
        require(C.driver_canonical(angle)==C.driver_canonical(expected_angle),'ANGLE_REQUERY_EVIDENCE')
    else:require(p['angle'] is None,'UNEXPECTED_ANGLE')
    return dict(offset=offset,gid=connection['global_trial_id'],event=fresh['event'],
                classification=fresh['classification'],consumption=consumption,worker=connection['worker_pid'])


def validate_setup(transport,c,m,task,*,artificial):
    doc=C.Document(transport['document'],transport['digest']);v=doc.read()
    require(v['input']==S.INPUT and v['precision']==S.PRECISION and v['angle']==S.ANGLE
        and v['arm']==task['arm'] and v['layer']==task['layer']
        and v['source_binding']==c['source_release_candidate'] and v['manifest_binding']==S.digest(m),'SETUP_FIXED_BINDING')
    if artificial:
        require(C.canonical(v['state'])==C.canonical(state(task,m))
            and v['context']=={k:x for k,x in context(c,m,task,'').items() if k!='trial_id'},'ARTIFICIAL_SETUP')
        return doc,v
    full=task['arm']=='full_return';full_B=full and task['layer']=='B'
    held=v['state']['parameters'] if full else v['state']
    require(tuple(held['p'])==tuple(map(F,S.INPUT['p'])) and held['x']==F(S.INPUT['x'])
        and held['m']==5 and held['arm']==task['arm'] and held['eta']==F(S.PRECISION['eta'])
        and held['rho']==F(S.PRECISION['rho']),'SETUP_HELD_INPUT_PRECISION')
    widths=(held['root_bits'],held['probability_bits']) if full else (held['K'],held['H'])
    require(widths==(256,160),'SETUP_HELD_WIDTHS')
    if full_B:
        constants=c['final_runtime_binding']['f3_constants']
        req=dict(p=tuple(map(F,S.INPUT['p'])),x=F(S.INPUT['x']),m=5,arm='full_return',sigma=1,
            eta=F(S.PRECISION['eta']),rho=F(S.PRECISION['rho']),root_bits=256,probability_bits=160,
            angle_tau=F(S.ANGLE['tau']),angle_root_bits=32,angle_terms=32)
        input_id=C.fingerprint({k:req[k] for k in ('p','x','m','arm','sigma')})
        policy_id=C.fingerprint(dict(policy=constants['POLICY'],spec_commit=constants['SPEC_COMMIT'],
            **{k:v for k,v in req.items() if k not in ('p','x','m','arm','sigma')}))
        source=C.Document.create(dict(schema='g6_f3_source_identity_v1',old_SHA256=constants['OLD_PINS'],
            F3_SHA256={n:c['source_pins'][S.SOURCE+n] for n in constants['F3_FILES']},
            spec_commit=constants['SPEC_COMMIT'],proof_commit=constants['PROOF_COMMIT'],G7_commit=constants['G7_COMMIT'],
            query_model='conditional exact controlled-Q',physical_provider_certified=False))
        raw=v['preflight_record'];cert=C.Document(raw['document'],raw['digest']);data=cert.read()
        require(C.canonical(data['request'])==C.canonical(req) and data['ready'] is True
            and data['actual_law_state_id']==C.fingerprint(v['state'])
            and data['input_id']==input_id and data['policy_id']==policy_id and data['source_id']==source.digest,
            'SETUP_F3_CERTIFICATE_BINDING')
        expected=dict(input_id=input_id,policy_id=policy_id,source_id=source.digest,preflight_id=cert.digest,sigma=1)
    else:
        pf='G7_UNMONITORED_LAYER_A'
        if task['layer']=='B':
            snap=v['canonical_preflight_snapshot']
            require({k:x for k,x in snap.items() if k!='conditional_witnesses'}==v['state'],'SETUP_CANONICAL_STATE')
            pf=driver_hash(snap)
        expected=dict(input_id=S.digest(S.INPUT),policy_id=S.digest([S.PRECISION,S.ANGLE]),
            source_id=c['source_release_candidate'],preflight_id=pf,sigma=1)
    require(v['context']==expected,'SETUP_CONTEXT_BINDING')
    return doc,v



def check_pairs(out,c,m,reports,*,artificial,require_all=False):
    out=Path(out);rows={r['task']['key']:r for r in reports};prefix={};counts=Counter();cells=[];full=compact=0;correct_witness=0
    for task in m['tasks']:
        if task['key'] not in rows:continue
        report=rows[task['key']];require(report['status']=='OK','PAIR_STATUS')
        path=out/(task['key']+'.jsonl');require(sha256(path.read_bytes()).hexdigest()==report['file_sha'],'PAIR_FILE_HASH')
        previous='0'*64;draws=0;position={'X':0,'Y':0};classes={a:Counter() for a in ('X','Y')};setup=None;setup_doc=None;last=None;chain_rows=path.read_bytes().splitlines()
        require(len(chain_rows)==2*task['n'],'PAIR_FIXED_ROWS')
        for i,line in enumerate(chain_rows):
            envelope=json.loads(line);require(envelope['previous']==previous,'PAIR_CHAIN')
            previous=sha256((previous+envelope['payload']).encode()).hexdigest();require(envelope['sha256']==previous,'PAIR_DIGEST')
            p=V.decode(envelope['payload']);axis='X' if i<task['n'] else 'Y';index=i%task['n']
            tid=f"{task['phase']}/{task['pass_name']}/{task['arm']}/{task['layer']}/{axis}/{index}"
            require(p['axis']==axis and p['trial_id']==tid,'PAIR_TRIAL_ORDER')
            if index<16 or task['phase']=='correctness':
                ev=p['evidence']
                if i==0:setup_doc,setup=validate_setup(ev['setup_witness'],c,m,task,artificial=artificial)
                require(ev['setup_digest']==setup_doc.digest and (i==0 or ev['setup_witness'] is None),'SETUP_REFERENCE')
                joined=payload(p,c,m,task,setup=setup,artificial=artificial);cons=joined['consumption'];event=joined['event'];classification=joined['classification'];last=ev['boundary_counts'];full+=1
                connection=ev['connection']
            else:
                require(p['schema']=='b3_pilot_compact_trial_v1' and p['retention']=='FULL_CONSTRUCTED_HASHED_NOT_RECONSTRUCTABLE','COMPACT_RETENTION')
                require(p['artificial_only'] is artificial and p['general_correctness_certified'] is False,'COMPACT_MODE')
                cons=p['consumption'];event=p['event'];classification=p['classification'];connection=p['connection'];last=p['boundary_counts']
                require(p['setup_digest']==setup_doc.digest,'COMPACT_SETUP_REFERENCE')
                validate_trace(cons,task['arm'],axis) if artificial else validate_consumption(cons,task['arm'],axis,namespace=m['seed_namespace'],artificial=False)
                if event is not None:
                    raw_event=C.Document(p['exact_event']['document'],p['exact_event']['digest']).read()
                    require(C.driver_canonical(raw_event)==C.driver_canonical(event),'COMPACT_EVENT_EXACT_PROJECTION')
                    R.validate_event(raw_event,full=task['arm']=='full_return');require(p['kind']=='EVENT' and classification=='ACCEPTED','COMPACT_ACCEPTED')
                else:
                    require(C.Document(p['exact_event']['document'],p['exact_event']['digest']).read() is None,'COMPACT_ZERO_EXACT')
                    require(task['arm']=='full_return' and (p['kind'],classification) in [('ZERO_R','RAW_NONREDUCED'),('ZERO_C','COIN_REJECT')],'COMPACT_ZERO')
                require(p['angle_present']==(task['layer']=='B' and event is not None),'COMPACT_ANGLE')
                for name in ('original_digest','exact_codec_sha256','full_evidence_sha256','angle_sha256'):
                    require(type(p[name]) is str and len(p[name])==64 and all(x in '0123456789abcdef' for x in p[name]),'COMPACT_EXACT_DIGEST:'+name)
                compact+=1
            require(connection['global_trial_id']==m['run_id']+'/'+task['condition']+'/'+tid
                and connection['global_offset']==task['offset']+i and connection['task_key']==task['key']
                and connection['worker_pid']==report['pid'] and connection['manifest_binding']==S.digest(m)
                and connection['source_binding']==c['source_release_candidate'] and connection['runtime_binding']==m['runtime_binding'],'GLOBAL_TRIAL_BINDING')
            require(cons['counter_before']==position[axis],'BIT_POSITION_PREFIX');position[axis]=cons['counter_after']
            key=task['arm'],axis,index;fingerprint=C.fingerprint(dict(event=event,consumption=cons))
            require(key not in prefix or prefix[key]==fingerprint,'AB_SP1_PASS_CORRECTNESS_PREFIX');prefix[key]=fingerprint
            classes[axis][classification]+=1;draws+=len(cons['draws'])
            accepted=sum(v['ACCEPTED'] for v in classes.values());rawzero=sum(v['RAW_NONREDUCED'] for v in classes.values())
            is_b=task['layer']=='B';is_full=task['arm']=='full_return'
            expected_counts=dict(primary_constructions=1,scheduled_sampling=i+1,bit_draw_requests=draws,
                G7_make_generator=int(not(is_full and is_b)),F3_prepare=int(is_full and is_b),
                canonical_snapshot=int(is_b and not is_full),AuditParity_construction=int(is_b),
                B_sample_audit=(i+1 if is_b else 0),nested_B_event_witness=(accepted if is_b else 0),
                B_angle=(accepted if is_b else 0),B_event_requery=(accepted if is_b else 0),
                canonical_angle=(accepted if is_b and not is_full else 0),Full_angle=(accepted if is_b and is_full else 0),
                Full_sample_packet=(i+1-rawzero if is_full else 0),Full_angle_packet=(accepted if is_b and is_full else 0))
            require(all(type(v) is int and v>=0 for v in last.values()) and all(last.get(k,0)==v for k,v in expected_counts.items()),'EXACT_ROUTING_COUNTERS')
            require(p['trial_cost']['observer_enabled']==(task['pass_name']=='instrumented'),'OBSERVER_PASS_BINDING')
            require(last['scheduled_sampling']==i+1 and last['primary_constructions']==1,'PAIR_CALL_COUNTER')
            if task['phase']=='correctness' and event is not None:correct_witness+=1
        require(previous==report['chain'] and report['output_rows']==2*task['n'],'PAIR_SUMMARY_CHAIN_ROWS')
        counts.update(last)
        for axis,v in classes.items():cells.append(dict(condition=task['condition'],phase=task['phase'],pass_name=task['pass_name'],arm=task['arm'],layer=task['layer'],axis=axis,N=task['n'],classifications=dict(v),accepted=v['ACCEPTED']))
        require((report['fine'] is not None)==(task['pass_name']=='instrumented'),'WORKER_OBSERVER_MODE')
    for name,limit in m['call_budgets'].items():
        if name in counts and type(limit) is int:require(counts[name]<=limit,'GLOBAL_BOUNDARY:'+name)
    require(counts['bit_draw_requests']<=105600 and correct_witness<=512,'DRAW_OR_WITNESS_BUDGET')
    if require_all:require(len(rows)==48 and full==1536 and compact==15360 and counts['scheduled_sampling']==16896,'ALL_PLAN_COMPLETED')
    return dict(full_documents=full,compact_documents=compact,boundary_counts=dict(counts),driver_correctness_event_witness=correct_witness,
        cells=cells,prefix_valid=True,full_original_reconstructable_for_compact=False,mathematical_independent_certification=False,extra_sampling=0)


def inspect(root,c,m):
    root=Path(root);out=root/'output';required=['final.json','delivery.json','caller.json','simple_run_state.json','STOP_marker.json','host_outer.json']
    missing=[n for n in required if not(out/n).is_file()]
    if missing:
        emergency=read(out/'STOP_failure_marker.json') if (out/'STOP_failure_marker.json').exists() else None
        if emergency:require(emergency['mandatory_STOP'] is True and emergency['claim_spent'] is True and emergency['manifest_binding']==S.digest(m),'EMERGENCY_STOP_BINDING')
        return dict(status='INCOMPLETE_EVIDENCE',missing=missing,emergency_STOP=emergency)
    report=read(out/'final.json');delivery=read(out/'delivery.json');caller=read(out/'caller.json');simple=read(out/'simple_run_state.json');marker=read(out/'STOP_marker.json')
    require(delivery['final_sha256']==sha256((out/'final.json').read_bytes()).hexdigest() and caller['observed'] is True
        and caller['delivery_sha256']==sha256((out/'delivery.json').read_bytes()).hexdigest(),'CALLER_DELIVERY_JOIN')
    require(report['manifest_binding']==S.digest(m) and report['candidate_binding']==S.digest(c),'FINAL_CONTEXT')
    from .metadata import validate_bundle
    docs=dict(delivery=delivery,caller=caller,marker=marker,simple=simple)
    if (out/'receipt.json').exists():docs['receipt']=read(out/'receipt.json')
    validate_bundle(report,m,docs)
    require(marker['mandatory_STOP'] is True and sha256((out/'STOP_marker.json').read_bytes()).hexdigest()==report['STOP_marker_sha256'],'MANDATORY_STOP')
    require(simple['claim'][4]==1 and simple['claim'][3] in ('STOPPED','UNKNOWN_STOPPED'),'CLAIM_SPENT_TERMINAL')
    previous='0'*64;stopped=False;complete={};trace=[]
    for i,line in enumerate((out/'operations.jsonl').read_bytes().splitlines()):
        v=json.loads(line);require(v['previous']==previous,'TRACE_CHAIN');previous=sha256(previous.encode()+v['payload'].encode()).hexdigest();require(v['sha256']==previous,'TRACE_DIGEST')
        value=json.loads(v['payload']);require(value['seq']==i and value['run_id']==m['run_id'],'TRACE_ID')
        if value['op']=='STOP':stopped=True
        if value['op']=='ENTER':require(not stopped,'ENTER_AFTER_STOP')
        if value['op']=='COMPLETE':require(value['global_offset'] not in complete,'DUPLICATE_COMPLETE');complete[value['global_offset']]=value['global_trial_id']
        trace.append(value)
    if not report['complete']:
        require(stopped and not(out/'receipt.json').exists(),'STOP_NO_SUCCESS_RECEIPT')
        return dict(status=report['classification'],completed=report['states']['COMPLETED'],states=report['states'],failures=report['failures'],claim_spent=True,retry=0,resume=False)
    receipt=docs['receipt'];require(sha256((out/'receipt.json').read_bytes()).hexdigest()==report['receipt_sha256']==simple['claim'][5],'RECEIPT_HASH')
    require(receipt['state_sha256']==sha256((out/'trial_states.bin').read_bytes()).hexdigest() and receipt['trace_sha256']==sha256((out/'operations.jsonl').read_bytes()).hexdigest(),'RECEIPT_STATE_TRACE')
    require(simple['execution_state']=='COMPLETED' and (out/'trial_states.bin').read_bytes()==bytes([4])*16896 and set(complete)==set(range(16896)),'ALL_TERMINAL_COMPLETED')
    for task in m['tasks']:
        for axis in ('X','Y'):
            for index in range(task['n']):
                offset=task['offset']+(0 if axis=='X' else task['n'])+index
                require(complete[offset]==f"{m['run_id']}/{task['condition']}/{task['phase']}/{task['pass_name']}/{task['arm']}/{task['layer']}/{axis}/{index}",'COMPLETE_DOCUMENT_JOIN')
    checked=check_pairs(out,c,m,report['pairs'],artificial=m['artificial_only'],require_all=True)
    outer=read(out/'host_outer.json');require(outer['total_cpu_ns']==outer['parent_cpu_ns']+outer['reaped_child_hierarchy_cpu_ns'] and not outer['coarse_and_fine_added'],'CPU_COUNT_ONCE')
    return dict(status=report['classification'],completed=16896,receipt_caller_join=True,claim_spent=True,mandatory_STOP=True,**checked)
