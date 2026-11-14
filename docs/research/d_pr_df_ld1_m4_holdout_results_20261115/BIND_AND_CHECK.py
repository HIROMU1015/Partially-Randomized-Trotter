"""Bind only the published m4 raw receipts; metadata/price reads, no circuit calls or fit."""
from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parent
ROOT=B.parents[2]
REL=str(B.relative_to(ROOT));OUT=REL+'/run01'
RAW='3042e6e66ad693ed514274334adecd3d1b42dc02'
SOURCE='7bdfc6f629aa74a5c1347abf84ad6486c669c9d8'

def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def read(path,commit=RAW):return json.loads(blob(commit,path))
def loc(path,selector=(),commit=RAW):return dict(commit=commit,path=path,selector=list(selector))
def save(name,value):
    with (B/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

plan=read(REL+'/INPUTS.json',SOURCE);cell=plan['cells'][0];request=plan['menus']['H4']['candidates'][0]
assert json.loads((B/'INPUTS.json').read_text())==plan
result=read(OUT+'/RESULT.json');archive=read(OUT+'/ARCHIVED_RECEIPTS.json');gate=read(OUT+'/RECEIPT_CHECK.json')
assert result['execution_source']==archive['execution_source']==SOURCE
assert result['status']=='M4_SC_PAIRS_ACQUIRED' and result['new_compiles']==8 and result['compile_attempts']==8
assert gate['valid_axis_rows']==8 and len(gate['checks'])==4 and gate['missing']==gate['withheld']==0
assert cell['path_indices']==[0,1,2,3] and cell['n']==4 and cell['role']=='HOLDOUT'
rows=[]
for i in range(4):
    envelope=read(OUT+f'/p{i}.json');assert envelope['ok'] and envelope['index']==i and envelope['exitcode']==0
    p=envelope['value'];a=archive['paths'][i];assert a['index']==i
    assert (p['m'],p['index'],p['candidate_id'],p['cohort'],p['seeds'],p['role'])==(4,i,cell['candidate_id'],cell['cohort'],cell['seeds'][i],'HOLDOUT')
    assert p['request']==request['candidate'] and p['law_binding']==request['law_binding']
    assert p['source']['execution_source']==SOURCE and p['lowering']==plan['lowering']['id']
    assert p['scalar_hex']==plan['scalar_saved']['hex'] and p['scalar_application']=='ONCE_AT_TOTAL_T'
    assert p['numerical_action_gate']=='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH' and p['full_wrapper_statevector_validation'] is False
    assert p['gate_status']=='INPUT_PREPARATION_REQUEST_LOWERING_COMPILER_FINGERPRINT_CHECKED'
    assert p['conditional'] is False and p['cache_hits']==0 and p['new_compiles']==2
    assert [r['axis'] for r in p['compile_log']]==['cosine','sine'] and all(r['status']=='COMPLETE' for r in p['compile_log'])
    for v in a['copies'].values():assert hashlib.sha256(blob(RAW,v['path'])).hexdigest()==v['sha256']
    original=read(a['copies']['result.json']['path']);struct=read(a['copies']['STRUCTURAL.json']['path'])
    assert (original['index'],original['cohort'],original['candidate_id'])==(i,cell['cohort'],cell['candidate_id'])
    assert original['source']['execution_source']==struct['scientific_source']==SOURCE
    assert original['event_digest']==struct['event_digest']==a['event_original']['event_digest']==p['event_digest']
    assert original['S']==struct['S']==p['S'] and original['C']==p['C'] and original['fingerprints']==p['fingerprints']
    for ai,(axis,native) in enumerate((('X','cosine'),('Y','sine'))):
        receipt=a['copies'][native+'.json'];r=read(receipt['path'])
        assert r['fingerprint']==p['fingerprints'][ai] and r['metrics']['rz_count']==p['C'][ai]
        assert r['compiler']==p['compiler']==plan['compiler']['settings_hash'] and r['cache_hit'] is False
        assert p['fingerprints'][0]!=p['fingerprints'][1]
        assert all(type(v) is int and v>=0 for v in (p['S'][ai],p['C'][ai]))
        rows.append(dict(m=4,physical_id=p['candidate_id'],role='HOLDOUT',cohort=p['cohort'],index=i,seeds=p['seeds'],axis=axis,native_axis=native,
            S=p['S'][ai],C=p['C'][ai],event_digest=p['event_digest'],wrapper_fingerprint=p['fingerprints'][ai],compiler=p['compiler'],
            lowering=p['lowering'],scalar=p['scalar'],scalar_hex=p['scalar_hex'],scalar_application=p['scalar_application'],
            execution_source=SOURCE,result_publication_commit=RAW,original_price_configuration_id=plan['configuration_id'],
            original_price_canonical_id=request['V2_candidate_id'],frozen_prediction_identity=plan['predictions'][axis]['identity'],
            request=loc(REL+'/INPUTS.json',['menus','H4','candidates',0],SOURCE),seed_source=loc(REL+'/INPUTS.json',['cells',0,'seeds',i],SOURCE),
            path_result=loc(OUT+f'/p{i}.json',['value']),
            original_receipt=dict(loc(a['copies']['result.json']['path']),sha256=a['copies']['result.json']['sha256'],original_private_path=a['copies']['result.json']['original_private_path']),
            axis_receipt=dict(loc(receipt['path']),sha256=receipt['sha256'],original_private_path=receipt['original_private_path']),
            structural_receipt=dict(loc(a['copies']['STRUCTURAL.json']['path']),sha256=a['copies']['STRUCTURAL.json']['sha256']),
            event_original=a['event_original'],gate=loc(OUT+'/RECEIPT_CHECK.json',['checks',i]),
            valid=True,reason=None,status='D2_RECEIPT_BINDING_CHECKED',
            numerical_action_gate=p['numerical_action_gate'],full_wrapper_statevector_validation=False,expectation='FINITE_DECLARED_NORMAL_COHORT_Q_ONLY'))
save('ADMITTED_RECEIPTS.json',dict(schema='PR_DF_LD1_M4_ADMITTED_SC_ROWS_V1',execution_source=SOURCE,raw_publication_commit=RAW,
    source_inputs=loc(REL+'/INPUTS.json',commit=SOURCE),counts=dict(planned_compiles=8,reused=0,new_compiles=8,compile_attempts=8,valid=8,withheld_or_missing=0,fresh_paths=4,retry=0),
    rows=rows,model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',C_consumer_admission='PENDING',fit=False,mandatory_stop=True))
save('RECEIPT_BINDING_CHECK.json',dict(status='EIGHT_PUBLISHED_AXIS_RECEIPTS_MATCHED',checked_rows=8,failed_rows=0,
    execution_source=SOURCE,raw_publication_commit=RAW,metadata_checked=['m/cell','physical request','original IDs separately from frozen prediction IDs',
    'axis/cosine-sine','index/cohort/16 literal seeds','law','SOURCE','event digest','S/C','metrics/compiler/fingerprint','lowering/scalar','receipt SHA256'],
    numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',full_wrapper_statevector_validation=False,
    model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',C_consumer_admission='PENDING'))
print(json.dumps(dict(raw=RAW,source=SOURCE,checked_rows=8,failed=0)))
