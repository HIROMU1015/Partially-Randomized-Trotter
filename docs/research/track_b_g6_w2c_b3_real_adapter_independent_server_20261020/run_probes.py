"""Independent expected values and negatives, with an isolated candidate subject."""
from copy import deepcopy
from fractions import Fraction as F
import json
import subprocess
import sys
import independent as I

def main():
    requests=[];expected={};groups={}
    def add(label,op,group='codec',want=True,value=None,**kwargs):
        requests.append(dict(id=label,op=op,**kwargs));expected[label]=(want,value);groups[label]=group
    examples=[None,True,False,0,-1,2**4096,F(-11,17),(),[],(1,[F(2,3),False]),
              {3:'three',(1,2):{'unicode':'日本語'}},{'int':'0x1'}, {'rat':['0x1','0x2']}]
    add('encoder_golden_algebra','encode_examples',value=[I.document(v) for v in examples])
    for index,value in enumerate(examples):
        add('codec_roundtrip_'+str(index),'read',record=I.document(value),
            value=dict(record=I.document(value),sidecar=I.sidecar(value,'TEST_SOURCE')))
    bad_documents={
        'bool_int':'{"$int":true}', 'bare_int':'1', 'float':'0.125', 'nan':'NaN',
        'unreduced':'{"$rat":["0x3","0x9"]}', 'negative_denominator':'{"$rat":["0x1","-0x2"]}',
        'zero_denominator':'{"$rat":["0x0","0x0"]}', 'bad_signed':'{"$int":"0x-1"}',
        'plus_hex':'{"$int":"+0x1"}', 'leading_hex':'{"$int":"0x01"}',
        'negative_zero':'{"$int":"-0x0"}', 'upper_hex':'{"$int":"0xAB"}',
        'duplicate_json':'{"a":true,"a":false}', 'unknown_tag':'{"$other":[]}',
        'extra_tag':'{"$int":"0x1","extra":false}', 'tuple_shape':'{"$tuple":null}',
        'map_duplicate':'{"$map":[[{"$int":"0x1"},"a"],[true,"b"]]}',
        'unhashable':'{"$map":[[[],true]]}', 'map_pair':'{"$map":[[true]]}',
        'noncanonical_order':'{"b":true,"a":false}', 'whitespace':'{ "a":true}',
        'projection_collision':I.canonical({1:'a','1':'b'}),
    }
    for name,text in bad_documents.items():
        add('codec_reject_'+name,'read',want=False,record=dict(document=text,digest=I.digest(text)))
    add('codec_wrong_digest','read',want=False,record=dict(document='{}',digest='0'*64))
    add('unsupported_float','plain_unsupported',want=False,input_value=0.5)
    # Sidecar changes are rejected even when their ordinary JSON is well formed.
    sc=I.sidecar({1:(F(-2,7),[False,None,2**1024])},'TEST_SOURCE')
    add('sidecar_restore_golden','restore',sidecar=sc,source='TEST_SOURCE',value=sc['f3'])
    for field in sc:
        changed=deepcopy(sc)
        changed[field]='MUTATED' if field!='f3' else dict(document='{}',digest=I.digest('{}'))
        add('sidecar_changed_'+field,'restore',want=False,sidecar=changed,source='TEST_SOURCE')
    add('sidecar_wrong_source','restore',want=False,sidecar=sc,source='OTHER_SOURCE')
    for variant in ('missing','extra'):
        changed=deepcopy(sc)
        if variant=='missing':changed.pop('preservation')
        else:changed['extra']=None
        add('sidecar_'+variant,'restore',want=False,sidecar=changed,source='TEST_SOURCE')
    result_golden=[];full_event=None;canonical_event=None
    for arm in I.ARMS:
        for layer in ('A','B'):
            counter=0
            for index in range(8):
                tid=f'correctness/baseline/{arm}/{layer}/X/{index}'
                adapted,bits,angle,counter=I.oracle(arm,'X',counter,tid,layer)
                original=I.read_document(adapted['codec']['f3'])
                ctx={k:original[k] for k in ('input_id','policy_id','source_id','preflight_id','trial_id','sigma')}
                q=dict(record=adapted['codec']['f3'],expected=ctx)
                if arm!='full_return':
                    q.update(arm=arm,event=I.document(adapted['event']))
                    op='g7'
                elif layer=='A':
                    obs=I.read_document(adapted['g7_observation']['f3'])
                    q.update(record=adapted['g7_observation']['f3'],arm=arm,event=I.document(adapted['event']))
                    op='g7'
                else:op='result'
                label=f'result_{arm}_{layer}_{index}'
                add(label,op,group='result',value=adapted,**q)
                result_golden.append(dict(id=label,classification=adapted['classification']))
                if arm=='full_return' and layer=='B' and adapted['event'] and full_event is None:
                    full_event=(adapted,bits,angle,ctx,q)
                if arm=='ordinary' and layer=='B' and canonical_event is None:
                    canonical_event=(adapted,bits,angle,ctx,q)
                if angle:
                    add('angle_'+label,'angle',group='result',value=angle,arm=arm,record=angle['angle']['f3'],
                        query=I.document(angle['requery_input']),event=I.document(adapted['event']),adapted=I.document(adapted))
    a,b,angle,ctx,q=full_event
    for field in ctx:
        bad=deepcopy(q);bad['expected'][field]=False if field=='sigma' else 'WRONG'
        add('context_wrong_'+field,'result',group='result',want=False,**bad)
    original=I.read_document(q['record'])
    for field,value in dict(child=True,proposal=F(1,3),weight=F(-1),word=(0,0),raw_word=(0,0),
                            phase_i_power=1,circuit_time_word=(0,1),provider_calls={2:2},ratio=1,helper=True).items():
        bad=deepcopy(original);bad['event'][field]=value
        add('event_invalid_'+field,'result',group='result',want=False,record=I.document(bad),expected=ctx)
    for field,value in dict(bits_consumed=1,draw_count=True,raw_word=(0,0),fresh_IID_verified=True).items():
        bad=deepcopy(original);bad['trace'][field]=value
        add('trace_invalid_'+field,'result',group='result',want=False,record=I.document(bad),expected=ctx)
    for variant in ('identity','query'):
        data={k:original[k] for k in ('schema','trial_id','input_id','policy_id','source_id','preflight_id','sigma','run_completion_or_confidence_certified')}
        if variant=='identity':data.update(kind='IDENTITY',deterministic_operator='I',bits_consumed=0)
        else:data.update(kind='EVENT',event=original['event'],trace=None,selection_scope='deterministic formal event query; not a sampled trial')
        wanted=deepcopy(a);wanted.update(original_kind=data['kind'],classification='IDENTITY' if variant=='identity' else 'EVENT_QUERY',
            driver_kind=None,event_identity=I.fp(data),codec=I.sidecar(data,I.SOURCE_ID))
        if variant=='identity':wanted.update(event=None,event_fingerprint=None,is_random_zero=False)
        add('result_'+variant,'result',group='result',value=wanted,record=I.document(data),expected=ctx)
    for arm_data in (full_event,canonical_event):
        adapted,_,ang,context,query=arm_data
        arm='full_return' if adapted is a else 'ordinary'
        aq=dict(arm=arm,record=ang['angle']['f3'],query=I.document(ang['requery_input']),event=I.document(adapted['event']),adapted=I.document(adapted))
        for field,value in dict(ratio=F(1),sigma=-1,root_bits=True,terms=0,tau=F(-1),representative=F(1),representative_error_upper=F(1),
                                signed_phi_interval=(F(0),F(1)),native_rotation_acquired=True).items():
            bad=deepcopy(aq);doc=I.read_document(bad['record']);doc[field]=value;bad['record']=I.document(doc)
            add('angle_invalid_'+arm+'_'+field,'angle',group='result',want=False,**bad)
        bad=deepcopy(aq);query_value=I.read_document(bad['query']);query_value['child']=1;bad['query']=I.document(query_value)
        add('requery_child_'+arm,'angle',group='result',want=False,**bad)
    ca,_,_,cctx,cq=canonical_event
    bad=deepcopy(cq);data=I.read_document(bad['record']);data['event']=None;data['branch']='COIN_REJECT';bad['record']=I.document(data);bad['event']=I.document(None)
    add('canonical_none_not_zero','g7',group='result',want=False,**bad)
    data=dict(schema='g7_observation_candidate_v1',**ctx,arm='full_return',branch='UNKNOWN',trace=None,packet_calls=0,group_index=None,event=None)
    add('full_none_without_provenance','g7',group='result',want=False,record=I.document(data),event=I.document(None),expected=ctx,arm='full_return')
    failure=dict(schema='g6_f3_failure_v1',category='PRECISION_FAILURE',reason='TEST_REASON',stage='angle',input_id='TEST_INPUT',
                 details={'signed':F(-5,9),'tuple':(1,False)},mandatory_STOP=True,run_complete=False,is_random_zero=False)
    want=dict(schema='b3_adapter_terminal_failure_v1',original=I.document(failure),category='PRECISION_FAILURE',reason='TEST_REASON',stage='angle',mandatory_STOP=True,is_random_zero=False)
    add('failure_preserves_original','failure',group='result',value=want,record=I.document(failure))
    add('typed_failure_not_success','result',group='result',want=False,record=I.document(failure),expected=ctx)
    for field in ('mandatory_STOP','run_complete','is_random_zero'):
        bad=deepcopy(failure);bad[field]=not bad[field]
        add('failure_invalid_'+field,'failure',group='result',want=False,record=I.document(bad))
    seal_data=dict(arm='ordinary',identity='TEST_IDENTITY',state_id='TEST_STATE',payload='ORIGINAL_BYTES',completed_calls_before=7,confidence_certified=False)
    seal_text=I.canonical(seal_data,True)
    seal=dict(text=seal_text,digest=I.digest(seal_text))
    audit_expected={k:seal_data[k] for k in ('arm','identity','state_id','completed_calls_before')}
    add('canonical_audit_original_seal','audit',group='result',seal=seal,expected=audit_expected,
        value=dict(schema='b3_driver_audit_sidecar_v1',original_driver_seal=seal,
                   normalized_document=I.document(seal_data),earlier_tuple_or_key_types='NOT_RECOVERABLE_FROM_DRIVER_BYTES_ALONE',
                   actual_scientific_audit_verified=False))
    for field in audit_expected:
        expected_bad=deepcopy(audit_expected);expected_bad[field]=True if field=='completed_calls_before' else 'WRONG'
        add('canonical_audit_wrong_'+field,'audit',group='result',want=False,seal=seal,expected=expected_bad)
    for label,changed_text in [('digest',seal_text),('noncanonical',' '+seal_text),('unreduced','{"rat":["0x2","0x4"]}'),('tag_type','{"int":true}')]:
        add('canonical_audit_invalid_'+label,'audit',group='result',want=False,
            seal=dict(text=changed_text,digest='0'*64 if label=='digest' else I.digest(changed_text)),expected=audit_expected)
    for arm in I.ARMS:
        for axis in ('X','Y'):
            widths=[1,160,256,257,512,513];counter=0;draws=[]
            for width in widths:
                row=I.bit_draw('ARTIFICIAL_REAL_ADAPTER_V1/'+arm+'/'+axis,counter,width);counter=row['counter_after'];draws.append(row)
            add('bits_boundaries_'+arm+'_'+axis,'bits',group='bits',arm=arm,axis=axis,widths=widths,
                value=dict(draws=draws,counter=counter,consumed_bits=sum(widths)))
    for label,width in [('bool',True),('zero',0),('negative',-1),('float',1.5)]:
        add('bits_invalid_'+label,'bits',group='bits',arm='ordinary',axis='X',widths=[160,width],value='TERMINAL_BITS')
    for namespace in ('B3_FULL_N_NEW_20261019_V1','PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1','REAL'):
        add('reject_namespace_'+namespace,'bits',group='bits',want=False,arm='ordinary',axis='X',widths=[1],namespace=namespace)
    add('mutated_bits_domain','bits',group='bits',arm='ordinary',axis='X',widths=[1],domain='REAL',value='DOMAIN_STOP')
    add('closed_parent','closed',group='gate',value='ALL_DENIED')
    blocked=[dict(id=q['id'],group=groups[q['id']],status='NOT_EXECUTED_STATIC_ONLY_MODULE_DEPENDENCY',
                  dependency='g6_w2c_cost_preparation',reason='User section 3E permits only static reading of this module')
             for q in requests if groups[q['id']] in ('bits','gate')]
    requests=[q for q in requests if groups[q['id']] not in ('bits','gate')]
    (I.HERE/'blocked_dynamic_checks.json').write_text(json.dumps(blocked,indent=2)+'\n')
    (I.HERE/'probe_inputs.json').write_text(json.dumps(requests,indent=2)+'\n')
    proc=subprocess.run([sys.executable,str(I.HERE/'candidate_probe.py')],input=json.dumps(requests),capture_output=True,text=True)
    (I.HERE/'probe_stderr.log').write_text(proc.stderr)
    check=I.check;check(proc.returncode==0,'SUBJECT_PROCESS_FAILED:'+proc.stderr)
    response=I.plain(proc.stdout)
    check(response['science_modules_loaded']==[],'SCIENCE_LOADED')
    results=[]
    for row in response['rows']:
        want,value=expected[row['id']];passed=row['ok']==want;reason=row.get('reason')
        if passed and want:
            actual=I.decode(I.plain(row['result']))
            try:
                if value=='TERMINAL_BITS':
                    check(actual['rejected'] is True and actual['stopped'] is True and actual['failure_position']==dict(draw_index=1,counter=1,consumed_bits=160),'BITS_FAILURE_POSITION')
                elif value=='DOMAIN_STOP':check(actual['rejected'] is True and actual['counter']==0,'DOMAIN_BEFORE_GENERATION')
                elif value=='ALL_DENIED':check(all(x['rejected'] for x in actual) and len(actual)==12,'GATE_REJECTION')
                else:I.same(actual,value,'INDEPENDENT_GOLDEN_VALUE')
            except (I.Invalid,TypeError) as exc:passed=False;reason=str(exc)
        results.append(dict(id=row['id'],group=groups[row['id']],pass_=passed,expected_accept=want,observed_accept=row['ok'],reason=reason))
    report=dict(tests=len(results),PASS=sum(x['pass_'] for x in results),FAIL=sum(not x['pass_'] for x in results),
                source='b7ad344401b10395c49e4cfa806909282859bab5',science_modules_loaded=[],not_executed=len(blocked),results=results)
    (I.HERE/'independent_probe_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (I.HERE/'probe_outputs.json').write_text(json.dumps(response,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    for row in results:
        if not row['pass_']:print(row)

if __name__=='__main__':main()
