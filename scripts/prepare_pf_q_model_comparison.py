"""Literal-only PF model/direct joins. No prediction, fit, residual or ratio API."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import prepare_pf_direct_compile_inventory as inventory

ROOT=Path(__file__).resolve().parents[1]
BASE='2619688b26c1146ae8c32d31d272282bfd1de407'
OUT=ROOT/'artifacts/resource_applicability/track_a_pf_q_model_comparison_preparation/2026-10-20'
PLAN='artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19'
P0='artifacts/resource_applicability/track_a_pf_direct_compile_interrupted_audit/2026-10-20'
P1='artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20'
P2='artifacts/resource_applicability/track_a_pf_direct_compile_retry106/2026-10-20'
COMPILER=inventory.COMPILER
COHORTS={'220b9644df6e114116737fd041973973fcf86cf5':'INITIAL_572_GLOBAL_GATE_MISSING',
 '0c5fcc83ef55045c2c64746aadfaf3128667b058':'RETRY216_ACCEPTED_110',
 '9c65fb312891d8334037f30ef953336c287bc262':'RETRY106_ACCEPTED_106'}


def fixed(label,path,commit=BASE,fmt='json'):
    return inventory.source(label,path,commit,fmt)


def key(row):return (row['system'],row['candidate_id'],row['axis'])


def classify(has_model,has_prediction,hard_mismatch,reasons):
    if not has_model:return 'NOT_COMPARABLE','NO_REGISTERED_Q_MODEL_IN_SAME_INPUT_SERIES'
    if not has_prediction:return 'NOT_COMPARABLE','NO_SAVED_PREDICTION;NEW_MODEL_EVALUATION_NOT_AUTHORIZED'
    if hard_mismatch:return 'NOT_COMPARABLE','MODEL_IDENTITY_OR_COMPILER_MISMATCH'
    if reasons:return 'CONDITIONAL_COMPARISON',';'.join(reasons)
    return 'COMPARABLE','MATCHED_SAVED_POINT_AND_COMPLETE_IDENTITIES'


def model_digest_kind(value,detail):
    if not value:return 'UNDECLARED'
    if value==detail['source_sha256']:return 'FILE_SHA256'
    if value==detail['canonical_json_sha256']:return 'CANONICAL_JSON_SHA256'
    return 'MISMATCH'


def canonical_json_digest(raw):
    # Match h8_phase_b2_contract.digest on the original JSON numeric types.
    # The literal-only data reader intentionally parses decimals as strings.
    document=json.loads(raw)
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def locate(document,locator):
    if locator.startswith('JSONL_RECORD:'):
        head,_,tail=locator.partition('/')
        document=document[int(head.split(':')[1])-1];locator='/'+tail
    for token in locator.strip('/').split('/'):
        if token:document=document[int(token)] if isinstance(document,list) else document[token.replace('~1','/').replace('~0','~')]
    return document


def audit_publication():
    scope=fixed('APPROVED_SCOPE',P2+'/publication_scope_final.json')
    names=scope['cumulative_unapproved_paths'];assert len(names)==80
    delta=set(inventory.git('diff','--name-only','-z',scope['last_approved_publication_commit'],BASE).decode().split('\0'))-{''}
    assert delta==set(names),'PUBLICATION_SCOPE_MISMATCH'
    patterns={
      'PRIVATE_KEY':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
      'GITHUB_TOKEN':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
      'AWS_ACCESS_KEY':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
      'OPENAI_KEY':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b',
      'JWT':r'\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b',
      'BEARER':r'(?i)Bearer\s+[A-Za-z0-9._~+/-]{15,}',
      'URL_CREDENTIALS':r'https?://[^\s/:]+:[^\s/@]+@'}
    patterns={name:re.compile(value) for name,value in patterns.items()}
    secret_keys={'password','passwd','secret','client_secret','access_token','refresh_token','api_key','authorization','credentials'}
    findings=[];records=[]
    for name in names:
        # Review the published Git object. Current indexes may acquire a new entry.
        raw=inventory.git('show',BASE+':'+name)
        content=gzip.decompress(raw) if name.endswith('.gz') else raw
        text=content.decode('utf-8');counts=Counter();fields=[]
        for kind,pattern in patterns.items():
            if pattern.search(text):findings.append(dict(file=name,kind=kind,location='TEXT_PATTERN',value_not_copied=True))
        def walk(value,pointer=''):
            if isinstance(value,dict):
                for k,v in value.items():
                    pointer2=pointer+'/'+k.replace('~','~0').replace('/','~1')
                    if k.startswith('/home/'):
                        counts['home_path_hash_keys' if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) else 'home_path_other_keys']+=1
                    if k.lower() in secret_keys and isinstance(v,str) and v.strip() and v not in ('REDACTED','<redacted>'):
                        findings.append(dict(file=name,kind='NONEMPTY_CREDENTIAL_FIELD',location=pointer2,value_not_copied=True))
                    if k in ('runtime_authorization','approved_by_user'):
                        fields.append(dict(pointer=pointer2,type=type(v).__name__,classification='SCOPED_SCIENTIFIC_APPROVAL_RECEIPT'))
                    walk(v,pointer2)
            elif isinstance(value,list):
                for i,v in enumerate(value):walk(v,pointer+'/'+str(i))
            elif isinstance(value,str) and value.startswith('/home/'):counts['home_path_values']+=1
        if name.endswith('.json') or name.endswith('.json.gz'):walk(json.loads(text))
        elif name.endswith('.jsonl.gz'):
            for i,line in enumerate(text.splitlines()):walk(json.loads(line),'/line/'+str(i+1))
        records.append(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),compressed=name.endswith('.gz'),
            decoded_bytes=len(content),metadata_counts=dict(counts),reviewed_approval_fields=fields))
    return dict(status='STOP_PUBLICATION_ISSUE' if findings else 'PASS_SCOPED_PUBLICATION_REVIEW',
        base_commit=BASE,files_checked=80,decoded_gzip_included=True,findings=findings,files=records,
        authorization=dict(source='explicit user message in this conversation',verbatim='この80ファイルの公開を承認',
            commit=BASE,branch='codex/pf-direct-compile-retry106-20261020',scope_paths_match=True),
        runtime_paths_and_hashes_are_disclosed_metadata=True,
        scientific_grants_are_not_GitHub_credentials=True,absolute_secret_absence_guaranteed=False,
        history_changes=0,deletions=0,force_pushes=0,new_remote_writes=0)


def collect():
    inventory.SOURCES.clear()
    audit=audit_publication()
    if audit['findings']:return dict(publication_review=audit,STOP=True)
    axes=fixed('AXES',PLAN+'/axis_inventory.csv',fmt='csv')
    evidence=fixed('PRIOR_EVIDENCE',PLAN+'/axis_evidence.csv',fmt='csv')
    registry=fixed('REGISTRY',PLAN+'/model_registry.csv',fmt='csv')
    old_sources=fixed('PRIOR_SOURCE_INVENTORY',PLAN+'/source_inventory.json')
    bindings=fixed('PUBLIC_SYSTEM_BINDINGS',PLAN+'/system_bindings.json')
    excluded=fixed('EXCLUDED_48',PLAN+'/conditional_reacquisition_axes.csv',fmt='csv')
    observations=fixed('OBSERVED_788',P2+'/results01/combined_observation_index.json')['rows']
    provenance={v['id']:v for v in old_sources}
    models={};canonical_digests={}
    for label in ('H6_MODEL','H8_MODEL'):
        s=provenance[label];model=fixed(label,s['path'],s['commit'])
        assert inventory.SOURCES[label]['sha256']==s['sha256'];models[label]=model
        canonical_digests[label]=canonical_json_digest(inventory.git('show',s['commit']+':'+s['path']))
    fixed('H8_DIGEST_DEFINITION','scripts/h8_phase_b2_contract.py',fmt='bytes')
    # Verify transcription directly against the original saved predictions.
    prediction_documents={}
    observation_keys={key(row) for row in observations}
    for label in sorted({e['source_id'] for e in evidence if e['kind']=='MODEL' and key(e) in observation_keys}):
        s=provenance[label]
        prediction_documents[label]=fixed(label,s['path'],s['commit'],fmt='jsonl' if '.jsonl' in s['path'] else 'json')
        assert inventory.SOURCES[label]['sha256']==s['sha256']
    # Read the original metadata seal, checked against its already-public file hash.
    preserve=fixed('ORIGINAL_SEAL_HASHES',P0+'/original_run_preservation.json')['files']
    original_root=ROOT.parent/'pf-direct-compile-execution-20261020'
    original_specs={}
    for group in ('current_4series_722','legacy_h4_df12_66'):
        path=original_root/'.pf_direct_execution_private/sealed01'/(group+'_spec.json')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert preserve[str(path)]==digest
        original_specs[group]=json.loads(raw,parse_float=str)
        inventory.SOURCES['ORIGINAL_'+group]=dict(id='ORIGINAL_'+group,sha256=digest,
            metadata_only_local_read=True,public_hash_evidence='ORIGINAL_SEAL_HASHES',private_content_republished=False)
    sealed1=fixed('RETRY216_SEAL',P1+'/sealed_contract_and_grant.json.gz')['spec']
    sealed2=fixed('RETRY106_SEAL',P2+'/sealed_contract_and_grant.json.gz')['spec']
    systems={}
    for group,spec in original_specs.items():systems.update(spec['systems'])
    sanitized={}
    for system,v in systems.items():
        sanitized[system]=dict(binding=v['config']['binding'],profile=v['profile'],provider_SOURCE=v['provider_SOURCE'],
            compiler_options=v['config']['compiler_options'],compiler_hash=COMPILER,
            original_seal_source=original_specs['legacy_h4_df12_66' if system=='H4_LEGACY_DF12' else 'current_4series_722']['SOURCE'])
    workers={}
    bundles=[('INITIAL',P0+'/worker_evidence.json.gz'),('RETRY216',P1+'/results01/worker_evidence.json.gz'),
             ('RETRY106',P2+'/results01/worker_evidence.json.gz')]
    for label,path in bundles:
        d=fixed(label+'_WORKERS',path)
        pairs=[p for group in d['contracts'] for p in group['pairs']] if 'contracts' in d else d['pairs']
        for i,p in enumerate(pairs):
            files=p['files'];exitdoc=files.get('owned_worker_exit.json')
            if not exitdoc or exitdoc.get('status')!='PASS':continue
            for row in files['accepted_measurements.json']['rows']:
                workers[(row['SOURCE'],row['task_id'])]=dict(result_sha256=exitdoc['result_sha256'],
                    worker_evidence_source=label+'_WORKERS',worker_pair_locator=i,
                    fingerprint=row['compiled_input_fingerprint'],accepted_row=row,binding=files.get('preparation_binding.json',{}))
    fit=[];used=defaultdict(set);model_detail={}
    for model in registry:
        document=models[model['source_id']];index=int(model['source_locator'].rsplit('/',1)[1]);data=document['models'][index]
        stratum=data.get('stratum',data)
        model_detail[model['model_id']]=dict(model,profile_hash=stratum.get('profile_hash',document.get('profile_sha256')),
            compiler_hash=stratum.get('compiler_hash',document.get('compiler_hash')),
            source_sha256=provenance[model['source_id']]['sha256'],
            model_definition_SOURCE=document.get('SOURCE'),source_commit=provenance[model['source_id']]['commit'],
            source_path=provenance[model['source_id']]['path'],
            canonical_json_sha256=canonical_digests[model['source_id']],
            canonical_digest_definition='SHA256(json.dumps(json.loads(original_bytes),sort_keys=True,separators=(comma,colon),allow_nan=False).encode())',
            document_scope=document.get('scope','H8 DF28 named-series frozen model'),
            transfer_equivalence_certified=document.get('transfer_equivalence_certified'),
            S2_application_prefixes=document.get('S2_application_prefixes'),S4_application_prefixes=document.get('S4_application_prefixes'),
            q_domain=document.get('q_domain'),runtime_binary_identity_verified=False)
        points=[]
        if 'anchor_ids' in data:
            points += [(v,'TRAIN_ANCHOR') for v in data['anchor_ids']]
            points += [(v['cell']['id']+':'+v['axis'],'DECLARED_VALIDATION') for v in document['validation_axes']
                if v['cell']['method']==model['method'] and v['axis']==model['axis']]
        else:
            points += [(v['id'].rsplit('_',1)[0]+':'+model['axis'],v['role']) for v in data['checks']]
        for point,role in points:
            cid,axis=point.rsplit(':',1);tid=model['system']+':'+point;used[(model['model_id'],tid)].add(role)
            fit.append(dict(model_id=model['model_id'],system=model['system'],candidate_id=cid,axis=axis,
                role=role,task_id=tid,source_id=model['source_id'],source_locator=model['source_locator'],
                source_sha256=provenance[model['source_id']]['sha256']))
    # Explicit post-freeze direct checks in the existing inventory are validation history too.
    validation_sources={'H6_S4_VALIDATION','H6_P7_COSINE','H6_P7_SINE','H6_D_LATEST','H8_SIX_DIRECT',
        'H8_D_B0_p15_q8_2nd','H8_D_B0I_p16_q4_4th'}
    for e in evidence:
        if e['kind']!='DIRECT' or e['source_id'] not in validation_sources:continue
        for m in registry:
            if (m['system'],m['method'],m['axis'])==(e['system'],e['candidate_id'].split('_')[0],e['axis']):
                tid=':'.join(key(e));used[(m['model_id'],tid)].add('PRIOR_DIRECT_VALIDATION')
                fit.append(dict(model_id=m['model_id'],system=e['system'],candidate_id=e['candidate_id'],axis=e['axis'],
                    role='PRIOR_DIRECT_VALIDATION',task_id=tid,source_id=e['source_id'],source_locator=e['source_locator'],
                    source_sha256=provenance[e['source_id']]['sha256']))
    axis_by_key={key(v):v for v in axes};model_by_key={(m['system'],m['method'],m['axis']):m for m in registry}
    estimates=defaultdict(list)
    for e in evidence:
        if e['kind']=='MODEL':estimates[key(e)].append(e)
    correspondence=[];prediction_records=[]
    for number,row in enumerate(observations):
        k=key(row);axis=axis_by_key[k];method=row['candidate_id'].split('_')[0]
        m=model_by_key.get((row['system'],method,row['axis']));pred=estimates[k]
        witness=workers[(row['SOURCE'],row['task_id'])]
        assert witness['fingerprint']==row['compiled_input_fingerprint'] and row['native_exit_code']==0 and row['reaped']
        assert all(witness['accepted_row'][f]==row[f] for f in ('RZ','CX','depth','size','compiler_hash','task_id','axis','system','candidate_id'))
        native_binding=witness['binding'];public=bindings[row['system']].get('binding',bindings[row['system']])
        assert native_binding['hamiltonian_hash']==public['hamiltonian'] and native_binding['state_hash']==public['state']
        detail=model_detail[m['model_id']] if m else {};reasons=[];mismatch=False
        if m:
            mismatch=(row['compiler_hash']!=detail['compiler_hash'])
            # Compiler options match, but historical anchor/runtime binary transfer remains unproved.
            reasons.append('MODEL_ANCHOR_TO_CURRENT_RUNTIME_BINARY_EQUIVALENCE_NOT_ESTABLISHED')
            if systems[row['system']]['profile'].get('sha256')!=detail['profile_hash']:
                reasons.append('NOMINAL_PROFILE_HASH_DIFFERS_FROM_FROZEN_MODEL')
            if row['SOURCE']==next(iter(COHORTS)):reasons.append('INITIAL_RUN_GLOBAL_FINAL_GATE_MISSING')
            if axis['PF']=='4th':reasons.append('S4_SCHEDULE_MODEL_IS_FROZEN_COMPOSITION_HYPOTHESIS')
            if used[(m['model_id'],row['task_id'])]:reasons.append('POINT_PREVIOUSLY_USED_FOR_MODEL_TRAINING_OR_VALIDATION')
        for e in pred:
            source=provenance[e['source_id']]
            original=locate(prediction_documents[e['source_id']],e['source_locator'])
            literal=str(original.get('measured_RZ',original.get('RZ',''))) if isinstance(original,dict) else str(original)
            assert literal==e['RZ_literal'],'SAVED_PREDICTION_TRANSCRIPTION_MISMATCH'
            digest_kind=model_digest_kind(e['model_sha256'],detail)
            if digest_kind=='MISMATCH':mismatch=True
            prediction_records.append(dict(task_id=row['task_id'],model_id=m['model_id'] if m else None,
                saved_prediction_literal=e['RZ_literal'],evidence_id=e['evidence_id'],model_sha256_declared=e['model_sha256'],
                model_digest_kind=digest_kind,
                source_commit=source['commit'],source_path=source['path'],source_sha256=source['sha256'],
                source_locator=e['source_locator'],original_literal_verified=True,evaluated_this_task=False))
        category,reason=classify(bool(m),bool(pred),mismatch,reasons)
        cohort=COHORTS[row['SOURCE']]
        actual_system=sealed2['systems'][row['system']] if cohort=='RETRY106_ACCEPTED_106' else sealed1['systems'][row['system']] if cohort=='RETRY216_ACCEPTED_110' else systems[row['system']]
        nominal_profile=actual_system['profile'].get('sha256')
        record=dict(task_id=row['task_id'],candidate_id=row['candidate_id'],system=row['system'],DF_rank={'H2_DF3':3,'H4_DF10':10,'H6_DF19':19,'H8_DF28':28,'H4_LEGACY_DF12':12}[row['system']],
            geometry='linear_1Angstrom',basis='STO-3G',T='0.8',P=0,method=method,PF=axis['PF'],prefix=axis['prefix'],q=axis['q'],axis=row['axis'],
            control='symmetric_directional',metric='FULL_HADAMARD_WRAPPER_COMPILED_RZ_EXCLUDES_STATE_PREPARATION',
            direct_RZ_literal=str(row['RZ']),direct_SOURCE=row['SOURCE'],direct_cohort=cohort,
            direct_combined_source='OBSERVED_788',direct_row_locator=f'/rows/{number}',direct_result_sha256=witness['result_sha256'],
            worker_evidence_source=witness['worker_evidence_source'],worker_pair_locator=witness['worker_pair_locator'],
            circuit_fingerprint=row['compiled_input_fingerprint'],compiler_hash=row['compiler_hash'],
            direct_profile_sha256=nominal_profile,model_profile_sha256=detail.get('profile_hash'),
            model_compiler_hash=detail.get('compiler_hash'),compiler_settings_equal=row['compiler_hash']==detail.get('compiler_hash') if m else None,
            nominal_profile_hash_equal=nominal_profile==detail.get('profile_hash') if m else None,
            runtime_binary_equivalence='NOT_VERIFIED_FOR_MODEL_ANCHORS' if m else 'NO_MODEL',
            input_hamiltonian_sha256=native_binding['hamiltonian_hash'],state_sha256=native_binding['state_hash'],
            input_binding_match='MATCHED_SAVED_NAMED_SERIES_BINDING',provider_SOURCE=actual_system['provider_SOURCE'],
            model_id=m['model_id'] if m else None,model_source_sha256=detail.get('source_sha256'),
            model_definition_SOURCE=detail.get('model_definition_SOURCE'),model_source_commit=detail.get('source_commit'),
            model_source_path=detail.get('source_path'),model_source_locator=m['source_locator'] if m else None,
            model_canonical_json_sha256=detail.get('canonical_json_sha256'),
            model_formula=m['S2_formula' if axis['PF']=='2nd' else 'S4_formula'] if m else None,
            coefficients_literal=m['coefficients_literal'] if m else None,
            saved_prediction_literals=sorted({e['RZ_literal'] for e in pred}),saved_prediction_evidence_ids=[e['evidence_id'] for e in pred],
            prior_model_point_roles=sorted(used[(m['model_id'],row['task_id'])]) if m else [],
            original_global_final_gate_missing=cohort=='INITIAL_572_GLOBAL_GATE_MISSING',
            requires_saved_classical_signal_reconciliation=row.get('requires_saved_classical_signal_reconciliation',False),
            excluded_48_intersection=False,comparison_class=category,comparison_reason=reason,
            model_evaluated_this_task=False,residuals_computed=False,cost_reaccounted=False)
        if m and detail['q_domain'] is not None:
            prefixes=detail['S2_application_prefixes' if axis['PF']=='2nd' else 'S4_application_prefixes']
            record['declared_model_domain_status']='INSIDE' if int(axis['q']) in detail['q_domain'] and int(axis['prefix']) in prefixes else 'OUTSIDE'
        else:record['declared_model_domain_status']='SAVED_PRICE_POINT_ONLY_DOMAIN_NOT_DECLARED' if pred else 'NO_SAVED_PRICE_POINT'
        correspondence.append(record)
    ids={v['task_id'] for v in correspondence}
    assert len(ids)==len(correspondence)==788
    assert not ids&{':'.join(key(e)) for e in excluded}
    for v in fit:v['point_in_788_axis_set']=v['task_id'] in ids
    return dict(STOP=False,publication_review=audit,correspondence=correspondence,predictions=prediction_records,
        model_registry=list(model_detail.values()),model_used_points=fit,excluded48=excluded,direct_system_bindings=sanitized,
        counts={k:sum(v['comparison_class']==k for v in correspondence) for k in ('COMPARABLE','CONDITIONAL_COMPARISON','NOT_COMPARABLE')},
        source_inventory=list(inventory.SOURCES.values()),prior_source_inventory=old_sources)


def csv_text(rows):
    stream=io.StringIO();fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else '' if v is None else str(v) for k,v in row.items()})
    return stream.getvalue()


def outputs(data):
    result={'publication_review.json':data['publication_review']}
    if data['STOP']:
        result['MANDATORY_STOP.json']=dict(status='STOP_PUBLICATION_ISSUE',comparison_not_started=True,new_scientific_calls=0)
        return result
    result.update({'axis_correspondence.json':dict(rows=data['correspondence']),
        'axis_correspondence.csv':csv_text(data['correspondence']),
        'saved_prediction_records.json':dict(rows=data['predictions']),
        'saved_prediction_records.csv':csv_text(data['predictions']),
        'model_registry.json':data['model_registry'],'model_used_points.csv':csv_text(data['model_used_points']),
        'model_used_points.json':dict(rows=data['model_used_points']),
        'excluded_48_axes.csv':csv_text(data['excluded48']),
        'direct_system_bindings.json':data['direct_system_bindings'],
        'source_inventory.json':data['source_inventory'],'prior_source_inventory.json':data['prior_source_inventory'],
        'classification_summary.json':dict(axis_counts=data['counts'],total_direct_axes=788,
            systems={s:dict(Counter(v['comparison_class'] for v in data['correspondence'] if v['system']==s)) for s in sorted({v['system'] for v in data['correspondence']})},
            saved_prediction_axis_count=sum(bool(v['saved_prediction_literals']) for v in data['correspondence']),
            prior_used_axis_count=sum(bool(v['prior_model_point_roles']) for v in data['correspondence']),
            initial_global_final_gate_missing_axes=572,old_classical_signal_qualified_axes=20,excluded_additional_axes=48,
            no_error_ratio_prediction_fit_cost_ranking_calculated=True),
        'MANDATORY_STOP.json':dict(status='COMPARISON_PREPARATION_COMPLETE_MANDATORY_STOP',
            quantitative_analysis_authorized=False,new_scientific_calls=0,model_predictions_evaluated=0,
            residual_ratio_analysis=0,fit=0,cost_reaccounting=0,ranking=0,detailed_scientific_review=False,final_evidence_freeze=False)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    data=collect();files=outputs(data);OUT.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
        if args.check:assert (OUT/name).read_text()==text,name
        else:(OUT/name).write_text(text)
    print(json.dumps(dict(status='STOP_PUBLICATION_ISSUE' if data['STOP'] else 'PASS_PREPARATION_ONLY',counts=data.get('counts'),files=len(files))))
    if data['STOP']:raise SystemExit(2)


if __name__=='__main__':main()
