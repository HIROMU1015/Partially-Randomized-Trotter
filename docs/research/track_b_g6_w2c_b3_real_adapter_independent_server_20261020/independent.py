"""Consumer-owned exact algebra, artificial oracle, and Storage V2 integration.

No imports from the candidate's codec/result/fixture/offline implementation.
Only the previously published independent Storage V2 consumer is reused.
"""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import importlib.abc
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/tmp/b3-independent-server-r2-20261020')
SCIENCE = {'g7_generator','g7_provider','return_aggregation','g6_f3_contract',
           'g6_f3_precision','g6_f3_generator','g6_w2c_b3_driver','g6_w2c_cost_preparation'}

class NeverLoadScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] in SCIENCE:
            raise RuntimeError('INDEPENDENT_PRELOAD_DENIAL:' + fullname)

sys.meta_path.insert(0, NeverLoadScience())
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / 'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_adapter_v1 as STORAGE
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_identity_v1 import plan, CLOSED
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as OLD

class Invalid(ValueError):
    pass

def check(condition, message):
    if not condition:
        raise Invalid(message)

def dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def digest(data):
    return sha256(data.encode() if isinstance(data, str) else data).hexdigest()

def plain(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, 'JSON_DUPLICATE:' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda _: check(False, 'JSON_NONFINITE'))

def encode(value, driver=False):
    """Exact type dispatch. Ordinary driver maps deliberately stringify keys."""
    typ = type(value)
    prefix = '' if driver else '$'
    if typ is int:
        return {prefix + 'int': hex(value)}
    if typ is Fraction:
        return {prefix + 'rat': [hex(value.numerator), hex(value.denominator)]}
    if value is None or typ in (str, bool):
        return value
    if typ in (tuple, list):
        sequence = [encode(x, driver) for x in value]
        return {'$tuple': sequence} if typ is tuple and not driver else sequence
    check(typ is dict, 'UNSUPPORTED_TYPE:' + typ.__name__)
    if driver:
        keys = [str(k) for k in value]
        check(len(keys) == len(set(keys)), 'PROJECTED_KEY_COLLISION')
        return {str(k): encode(v, True) for k, v in value.items()}
    check(not any(type(k) is str and k.startswith('$') for k in value), 'RESERVED_TAG')
    if all(type(k) is str for k in value):
        return {k: encode(v) for k,v in value.items()}
    rows = [[encode(k), encode(v)] for k,v in value.items()]
    return {'$map': sorted(rows, key=lambda pair: json.dumps(pair[0], sort_keys=True))}

def canonical(value, driver=False):
    return dump(encode(value, driver))

def fp(value):
    return digest(canonical(value))

def exact_hex(value):
    check(type(value) is str, 'HEX_TYPE')
    integer = int(value, 16)
    check(hex(integer) == value, 'HEX_NONCANONICAL')
    return integer

def decode(value, driver=False):
    typ = type(value)
    if value is None or typ in (str, bool):
        return value
    if typ is list:
        return [decode(v, driver) for v in value]
    check(typ is dict, 'BARE_NUMBER')
    tag = '' if driver else '$'
    if set(value) == {tag + 'int'}:
        return exact_hex(value[tag + 'int'])
    if set(value) == {tag + 'rat'}:
        pair = value[tag + 'rat']
        check(type(pair) is list and len(pair) == 2, 'RAT_PAIR')
        n,d = [exact_hex(x) for x in pair]
        check(d > 0, 'RAT_DENOMINATOR')
        answer = Fraction(n,d)
        check((n,d) == (answer.numerator,answer.denominator), 'RAT_REDUCTION')
        return answer
    if not driver and set(value) == {'$tuple'}:
        check(type(value['$tuple']) is list, 'TUPLE_SHAPE')
        return tuple(decode(v) for v in value['$tuple'])
    if not driver and set(value) == {'$map'}:
        check(type(value['$map']) is list, 'MAP_SHAPE')
        result = {}
        for pair in value['$map']:
            check(type(pair) is list and len(pair) == 2, 'MAP_PAIR')
            k,v = [decode(x) for x in pair]
            check(k not in result, 'MAP_EQUAL_KEY')
            result[k] = v
        return result
    check(driver or not any(k.startswith('$') for k in value), 'UNKNOWN_TAG')
    return {k:decode(v,driver) for k,v in value.items()}

def document(value):
    text = canonical(value)
    return dict(document=text,digest=digest(text))

def read_document(record):
    check(type(record) is dict and set(record) == {'document','digest'}, 'DOCUMENT_FIELDS')
    text = record['document']
    check(type(text) is str and type(record['digest']) is str and digest(text) == record['digest'], 'DOCUMENT_HASH')
    value = decode(plain(text))
    check(canonical(value) == text, 'DOCUMENT_CANONICAL')
    return value

def sidecar(value, source):
    original = document(value)
    projection = canonical(value, True)
    return dict(schema='b3_exact_codec_sidecar_v1', source_identity=source,
                f3=original, driver_document=projection, driver_digest=digest(projection),
                preservation='F3_SOURCE_DOCUMENT_IS_LOSSLESS; DRIVER_PROJECTION_IS_NOT',
                original_fingerprint=original['digest'])

def read_sidecar(value, source):
    original = read_document(value['f3'])
    same(value, sidecar(original, source), 'SIDECAR_BINDING')
    return original

def same(actual, expected, name):
    # Serialization compares exact types, including bool/int and tuple/list.
    check(canonical(actual) == canonical(expected), name)

def driver_same(actual, expected, name):
    check(canonical(actual, True) == canonical(expected, True), name)

ARMS = ('ordinary','partial_return_tail','closed_P3_tail','full_return')
SOURCE_ID = 'ARTIFICIAL_SOURCE_G7_F3_SHAPED_V1'

def bit_draw(seed, start, width):
    check(type(width) is int and width > 0, 'WIDTH')
    check(type(start) is int and start >= 0, 'COUNTER')
    blocks = (width + 255) // 256
    check(start + blocks <= 2**64, 'COUNTER_OVERFLOW')
    bits = ''.join(format(int.from_bytes(sha256(seed.encode('utf-8') + n.to_bytes(8,'big')).digest(),'big'),'0256b')
                   for n in range(start, start + blocks))
    value = int(bits[:width], 2)
    return dict(width=width,counter_before=start,counter_after=start+blocks,
                value_hex=hex(value),discarded_bits=len(bits)-width)

def validate_bits(trace, arm, axis):
    seed = 'ARTIFICIAL_REAL_ADAPTER_V1/' + arm + '/' + axis
    check(arm in ARMS and axis in ('X','Y'), 'ARTIFICIAL_DOMAIN')
    check(trace['domain'] == seed and trace['artificial_only'] is True and
          trace['real_consumption_equivalence'] is False, 'BITS_DOMAIN')
    previous = None
    for row in trace['draws']:
        if previous is not None:
            check(row['counter_before'] == previous, 'COUNTER_CONTIGUITY')
        same(row, bit_draw(seed,row['counter_before'],row['width']), 'DRAW_BYTES')
        previous = row['counter_after']
    check(previous == trace['counter_after'], 'FINAL_COUNTER')
    check(trace['bits_consumed_before'] + sum(r['width'] for r in trace['draws']) == trace['bits_consumed'], 'CONSUMED')

def oracle(arm, axis, start, tid, layer):
    """Synthetic formula from the fixture contract; no candidate calls."""
    seed = 'ARTIFICIAL_REAL_ADAPTER_V1/' + arm + '/' + axis
    first = bit_draw(seed,start,160)
    first_value = int(first['value_hex'],16)
    branch = 'ACCEPTED'
    if arm == 'full_return':
        branch = {0:'RAW_NONREDUCED',1:'COIN_REJECT',2:'ACCEPTED',3:'ACCEPTED'}[first_value & 3]
    count = 2 if arm != 'full_return' else {'RAW_NONREDUCED':3,'COIN_REJECT':4,'ACCEPTED':5}[branch]
    draws = [bit_draw(seed,start+i,160) for i in range(count)]
    end = start + count
    token = 0
    for draw in draws:
        token ^= int(draw['value_hex'],16)
    raw = (0,1) if arm == 'full_return' and branch != 'RAW_NONREDUCED' else (0,0)
    event = None
    if branch == 'ACCEPTED':
        ratio = Fraction(16777217+(token % 8192),67108921)
        weight = ratio * Fraction(7,11)
        w = raw if arm == 'full_return' else ()
        event = dict(word=w,raw_word=raw,child=2,ratio=ratio,weight=weight,
                     coefficient=weight*Fraction(1,2),proposal=Fraction(1,2),phase_i_power=2,
                     provider_calls={0:1,1:1,2:2} if w else {2:2},helper=1,
                     event_unitary='i^phase exp(-i*sigma*atan(ratio)*Q_child) Q(word)',
                     circuit_time_word=w[::-1])
    consumption = dict(schema='b3_ARTIFICIAL_adapter_consumption_v1',domain=seed,draws=draws,
        counter_after=end,bits_consumed=160*end,bits_consumed_before=160*start,
        artificial_only=True,real_consumption_equivalence=False)
    context = dict(input_id='ARTIFICIAL_INPUT_V1',policy_id='ARTIFICIAL_PRECISION_V1',source_id=SOURCE_ID,
                   preflight_id='ARTIFICIAL_PREFLIGHT_V1',trial_id=tid,sigma=1)
    stages = ['order','label','label'] + ([] if branch == 'RAW_NONREDUCED' else ['acceptance']) + (['child'] if branch == 'ACCEPTED' else [])
    trace = dict(draws=[dict(stage=s,width=160,value=int(d['value_hex'],16)) for s,d in zip(stages,draws)],
        raw_word=raw,degree=2,bits_consumed=160*count,draw_count=count,
        completed_bit_map_calls_by_stage={s:stages.count(s) for s in ('order','label','acceptance','child')},
        extra_random_draws_for_audit=0,fresh_IID_verified=False)
    g7 = None
    if arm != 'full_return':
        original = dict(schema='g7_observation_candidate_v1',**context,arm=arm,branch=branch,
                        trace=consumption,packet_calls=0,group_index=0,event=event)
    else:
        original = dict(schema='g6_f3_trial_v1',**context,run_completion_or_confidence_certified=False)
        if event is None:
            original.update(kind='ZERO',reason=branch,trace=trace,is_valid_zero_trial=True)
        else:
            original.update(kind='EVENT',event=event,trace=trace)
        if layer == 'A':
            g7 = dict(schema='g7_observation_candidate_v1',**context,arm=arm,branch=branch,trace=trace,
                      packet_calls=int(branch != 'RAW_NONREDUCED'),group_index=None,event=event)
    kind = {'ACCEPTED':'EVENT','RAW_NONREDUCED':'ZERO_R','COIN_REJECT':'ZERO_C'}[branch]
    adapted = dict(schema='b3_adapted_result_v1',original_kind='EVENT' if event else 'ZERO',classification=branch,
        driver_kind=kind,event=event,event_fingerprint=fp(event) if event else None,
        event_identity=fp(original),source_identity=SOURCE_ID,trial_id=tid,codec=sidecar(original,SOURCE_ID),
        is_random_zero=event is None,mandatory_STOP=False)
    if g7 is not None:
        adapted['g7_observation'] = sidecar(g7,SOURCE_ID)
    angle = angle_oracle(adapted,context,arm) if event and layer == 'B' else None
    return adapted,consumption,angle,end

def angle_oracle(adapted, context, arm):
    event = adapted['event']
    lo,hi = Fraction(1,8),Fraction(1,8)+Fraction(1,10**8)
    mid,radius = (lo+hi)/2,(hi-lo)/2
    ctx = {k:context[k] for k in ('input_id','policy_id','source_id','preflight_id','sigma')}
    ctx.update(event_id=adapted['event_identity'],parent=event['word'],child=event['child'])
    query = dict(word=event['word'],child=event['child'])
    legacy_id = digest(canonical(event, True))
    if arm != 'full_return':
        ctx = dict(event_id=legacy_id,input={'id':'ARTIFICIAL_INPUT_V1'},
                   policy={'id':'ARTIFICIAL_PRECISION_V1'},preflight=context['preflight_id'])
        query.update(group_index=0,raw_word=event['raw_word'])
    angle = dict(schema='g6_f3_angle_enclosure_v1',ratio=event['ratio'],sigma=1,root_bits=32,terms=32,
        tau=Fraction(1,10**6),context=ctx,sqrt_interval=None,half_argument_interval=None,endpoint_series_evidence=None,
        phi_interval=(lo,hi),representative=mid,representative_error_upper=radius,
        signed_phi_interval=(lo,hi),signed_representative=mid,
        precision_check=dict(name='representative_radius',left=radius,operator='<=',right=Fraction(1,10**6),passed=True),
        native_rotation_acquired=False,physical_provider_certified=False,
        basis={'scope':'ARTIFICIAL_SHAPE_ONLY_NOT_AN_ATAN_ENCLOSURE'})
    result = dict(schema='b3_angle_requery_binding_v1',event_identity=adapted['event_identity'],
        event_fingerprint=adapted['event_fingerprint'],requery_input=query,requery_output=document(event),
        angle=sidecar(angle,SOURCE_ID),synthetic=True,mathematical_precision_verified=False,terminal_STOP=False)
    if arm != 'full_return':
        result.update(angle_context_kind='CANONICAL_LEGACY',angle_original_event_id=legacy_id,
                      canonical_input_binding=document({'id':'ARTIFICIAL_INPUT_V1'}),
                      canonical_policy_binding=document({'id':'ARTIFICIAL_PRECISION_V1'}),
                      requery_callable_arguments=(0,event['raw_word'],event['child']))
    return result

class Reference:
    """Caller input is anchored by the independently checked Git archive manifest."""
    def __init__(self, trusted):
        self.trusted = trusted

    def identity(self, c, m):
        # Convert our errors to the established independent consumer's error class.
        try:
            same(c,self.trusted,'TRUSTED_CALLER_CANDIDATE')
            check(c['schema']=='b3_closed_real_adapter_candidate_v1' and m['schema']=='b3_closed_real_adapter_manifest_v1','SCHEMAS')
            for obj in (c,m):
                for key,value in CLOSED.items():
                    same(obj[key],value,'CLOSED:'+key)
            check(c['source_release_candidate']==digest(dump(c['source_pins']))=='024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb','SOURCE_IDENTITY')
            check(m['candidate_binding']==digest(dump(c)),'CANDIDATE_BINDING')
            check(m['source_release_candidate']==c['source_release_candidate'],'SOURCE_JOIN')
            check(m['endpoint_binding']==digest(dump(c['endpoint_binding'])),'ENDPOINT_BINDING')
            for k,v in dict(seed_namespace='ARTIFICIAL_REAL_ADAPTER_V1',workers=8,start_method='spawn',codec='fused',observer='cached_all_call_return').items():
                same(m[k],v,'MODE_POLICY:'+k)
            check(c['scientific_API_dispatch']=='NOT_IMPLEMENTED' and c['endpoint_binding']['real_dispatch'] is False,'NO_SCIENCE')
            check(c['fixed_bundle_commit']=='d24cccbecdf0928fe977080dd7b28c591df0089e' and c['fixed_port_commit']=='82896d53fa97235fa848d6c0833eb193af8b99d6','PRIOR_REFERENCES')
            check(m['global_scope'].startswith('ARTIFICIAL_ONLY/'),'FAKE_SCOPE')
            check(m['global_claim']==digest(dump(['ARTIFICIAL_ONLY',m['global_scope'],digest(dump(c))])),'CLAIM')
            check(m['run_id']==digest(dump(['RUN',m['global_scope'],digest(dump(c))])),'RUN_ID')
            tasks,total = plan(m['profile_n'])
            same(m['tasks'],tasks,'PLAN');check(m['planned']==total,'PLANNED')
            check(m['static_scientific_candidate_calls']==16896 and type(m['call_limit']) is int and 0<=m['call_limit']<=total,'CALL_BUDGET')
            for channel in ('result','log'):
                check(type(m['byte_limits'][channel]) is int and 0<m['byte_limits'][channel]<=c['proposal'][channel+'_bytes'],'BYTE_BUDGET')
            same(c['worker_runtime']['source_pins'],c['source_pins'],'WORKER_PINS')
            return dict(tasks=tasks,planned=total,candidate_digest=digest(dump(c)),manifest_digest=digest(dump(m)),
                        run_id=m['run_id'],historical_runtime_bytes_proven=False,local_runtime_bytes_proven=False,
                        runtime_projection=False,real_authority_verified=False)
        except Invalid as exc:
            raise OLD.Invalid(str(exc)) from exc

def payloads(path):
    previous = '0'*64
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            envelope = plain(line)
            text = envelope['payload']
            check(envelope['previous']==previous and envelope['sha256']==digest(previous+text),'RAW_CHAIN')
            value = decode(plain(text),True)
            check(canonical(value,True)==text,'RAW_CANONICAL')
            previous = envelope['sha256']
            yield value

def dispatch(root, storage):
    root = Path(root);out = root/'output'
    if not (out/'final.json').exists() or not (out/'final.json').read_bytes().strip():
        return dict(available=False)
    c,m,r = [plain((out/n).read_bytes()) for n in ('candidate.json','manifest.json','final.json')]
    # Only trusted prefix operations count as evidence.
    prefix = r['evidence_v2']['trace_prefix']
    operations = []
    def visit(row,size):
        operations.append(row)
    from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_trace_v1 as TRACE
    TRACE.scan(out/'operations.jsonl',prefix,m['run_id'],complete=r['evidence_v2']['trace_complete'],visit=visit)
    trace = {op:{x['global_trial_id']:x for x in operations if x['op']==op} for op in ('RESERVE','ENTER','COMPLETE')}
    state = (out/'trial_states.bin').read_bytes()
    reports = {row['task']['key']:row for row in r['pairs']}
    full=hashed=joined=0;counts=Counter();prefixes={};gaps=[];failures=[]
    for task in m['tasks']:
        key = task['key'];path=out/(key+'.jsonl');report=reports.get(key)
        if report is None or not path.exists():
            continue
        counters = dict(X=0,Y=0);indices=dict(X=0,Y=0);last_axis='X'
        for p in payloads(path):
            if p.get('kind') not in ('EVENT','ZERO_R','ZERO_C') or 'trial_id' not in p:
                continue
            axis=p['axis'];idx=indices[axis];tid='/'.join([task['phase'],task['pass_name'],task['arm'],task['layer'],axis,str(idx)])
            check(not(last_axis=='Y' and axis=='X'),'AXIS_ORDER');last_axis=axis
            check(p['trial_id']==tid,'TRIAL_SEQUENCE');indices[axis]+=1
            adapted,bits,angle,counters[axis] = oracle(task['arm'],axis,counters[axis],tid,task['layer'])
            kind,event = adapted['driver_kind'],adapted['event']
            event_hash = digest(canonical(dict(kind=kind,event=event),True))
            gid = m['run_id']+'/'+task['condition']+'/'+tid
            offset = task['offset']+(task['n'] if axis=='Y' else 0)+idx
            d = dict(schema='b3_ARTIFICIAL_adapter_dispatch_v1',artificial_only=True,
                endpoint_binding=m['endpoint_binding'],candidate_binding=digest(dump(c)),manifest_binding=digest(dump(m)),
                runtime_binding=digest(dump(c['worker_runtime'])),factory=c['endpoint_binding']['adapter'],run_id=m['run_id'],
                global_trial_id=gid,global_offset=offset,task_key=key,worker_pid=report['pid'],
                event_digest=event_hash,adapted_digest=fp(adapted),consumption_digest=fp(bits))
            evidence=dict(artificial=True,origin='SYNTHETIC_DOCUMENTS_NOT_REAL_G7_F3',consumption=bits,adapted=adapted,dispatch=d)
            if task['layer']=='B':
                seal=dict(arm=task['arm'],identity=fp(dict(arm=task['arm'],source=SOURCE_ID)),
                    state_id=fp(dict(artificial=True,arm=task['arm'],layer='B')),payload=adapted['codec']['f3']['document'],
                    completed_calls_before=idx+(task['n'] if axis=='Y' else 0),confidence_certified=False)
                text=canonical(seal,True)
                normalized=decode(plain(text),True)
                evidence['audit']=dict(schema='b3_driver_audit_sidecar_v1',original_driver_seal=dict(text=text,digest=digest(text)),
                    normalized_document=document(normalized),earlier_tuple_or_key_types='NOT_RECOVERABLE_FROM_DRIVER_BYTES_ALONE',
                    actual_scientific_audit_verified=False)
            check(p['kind']==kind,'RESULT_FROM_ARTIFICIAL_BITS')
            if 'evidence' in p:
                driver_same(p['event'],event,'EVENT_VALUE')
                driver_same(p['evidence'],evidence,'EVIDENCE_DISPATCH_CODEC')
                driver_same(p['angle'],angle,'ANGLE_REQUERY_CONTEXT');full+=1
            else:
                check(p['retention']=='HASHED_FULL_DOCUMENTS_NOT_RETAINED','HASH_RETENTION')
                check(p['event_sha256']==event_hash,'EVENT_HASH')
                check(p['evidence_sha256']==digest(canonical(evidence,True)),'EVIDENCE_HASH')
                check(p['angle_sha256']==digest(canonical(angle,True)),'ANGLE_HASH');hashed+=1
            validate_bits(bits,task['arm'],axis)
            check(type(report['pid']) is int and report['pid']>0,'WORKER_PID')
            if gid in trace['COMPLETE']:
                for op in ('RESERVE','ENTER','COMPLETE'):
                    check(gid in trace[op] and trace[op][gid]['global_offset']==offset and trace[op][gid]['task_key']==key,'GLOBAL_'+op)
                if r['evidence_v2']['statefile_complete']:
                    check(state[offset]==4,'PHYSICAL_COMPLETE')
                joined+=1
            else:
                gaps.append(gid)
            pref=(task['arm'],axis,idx)
            pair=(event_hash,fp(bits))
            if pref in prefixes:
                check(prefixes[pref]==pair,'S_P1_AB_PHASE_PASS_PREFIX')
            prefixes[pref]=pair;counts[kind]+=1
        failure=(report.get('failure') or {}).get('trial_failure_provenance')
        if failure:
            check(failure['mandatory_STOP'] is True and failure['is_random_zero'] is False and
                  failure['retry']==0 and failure['resume'] is False and failure['current_attempt']['trial_completed'] is False,'F01_TERMINAL')
            frozen=failure.get('frozen_failure_evidence')
            if frozen:
                original=read_document(frozen['original'])
                for field in ('category','reason','stage','mandatory_STOP','is_random_zero'):
                    same(frozen[field],original[field],'FAILURE_LOSSLESS:'+field)
                check(original['run_complete'] is False and original['mandatory_STOP'] is True,'FAILURE_STOP')
            failures.append(dict(task=key,stage=failure['current_attempt']['stage'],type=failure['type'],reason=failure['reason'],original_evidence_preserved=bool(frozen)))
    complete=storage['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'
    if complete:
        check(full+hashed==joined==m['planned'] and not gaps,'FULL_COVERAGE')
    return dict(available=True,complete=complete,trials=full+hashed,full_documents=full,hash_only=hashed,
                joined=joined,counts=dict(counts),unjoined=gaps,typed_failures=failures,
                hash_only_original_documents_acquired=False,artificial_replay=True,live_science_compatibility=False)

def verify(root):
    trusted=plain((Path(root)/'candidate_input.json').read_bytes())
    storage=STORAGE.verify(root,Reference(trusted))
    try:
        storage['independent_adapter']=dispatch(root,storage)
    except (Invalid,KeyError,TypeError,ValueError,OSError,OLD.Invalid) as exc:
        if storage['status']=='INCOMPLETE_EVIDENCE' and isinstance(exc,(FileNotFoundError,json.JSONDecodeError)):
            storage['independent_adapter']=dict(available=False,gap=str(exc))
        else:
            storage.update(status='INTEGRITY_MISMATCH',issues=storage['issues']+[str(exc)])
    return storage
