#!/usr/bin/env python3
"""Metadata/byte preparation ONLY. No analysis import, ratios, means or statistics."""
import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

BASE='0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce'
COST='fc6bd039dc7171d5674ac04f0b3ad427f1b6495b'
H8='e6eea5387a6e84d1e39a9adb3b61975acc74e200'
PAPER='artifacts/resource_applicability/track_a_paper_preparation/2026-10-19/'
HP='artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/'
ROOT=Path(__file__).resolve().parents[1]

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read_bytes(b):
    if b[:2]==b'\x1f\x8b':b=gzip.decompress(b)
    return json.loads(b,parse_float=str)
def put(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def csv_data(raw):return list(csv.DictReader(io.StringIO(raw.decode())))
def rows(d):return d['signals'] if isinstance(d,dict) and isinstance(d.get('signals'),list) else d if isinstance(d,list) else [d] if isinstance(d,dict) and 'cell' in d else []
def number_text(value):return None if value in ('',None) else str(value)

def prepare(previous,destination):
    destination=destination.resolve();destination.mkdir(mode=0o700,exist_ok=True)
    store=destination/'inputs';store.mkdir(exist_ok=True)
    records=[];cache={}
    def snapshot(alias,raw,origin,expected=None,commit=None,path=None):
        digest=sha_bytes(raw)
        if expected is not None:assert digest==expected,('SOURCE_BYTES_MISMATCH',alias)
        target=store/digest
        if target.exists():assert target.read_bytes()==raw
        else:target.write_bytes(raw)
        records.append(dict(alias=alias,path=str(target),sha256=digest,bytes=len(raw),origin=origin,commit=commit,repository_path=path))
        return read_bytes(raw) if not alias.endswith('.csv') else csv_data(raw)
    def public(commit,path,alias):
        raw=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'show',commit+':'+path])
        return snapshot(alias,raw,'Git object',commit=commit,path=path)
    def private(alias,path,expected=None):return snapshot(alias,Path(path).read_bytes(),str(path),expected)
    prior=private('PREVIOUS_PRIVATE_LOCK',previous/'private_input_lock.json')
    old={(r['system'],r['role']):r for r in prior}
    def old_input(system,role):
        key=(system,role)
        if key not in cache:
            r=old[key];cache[key]=private(system+'/'+role,r['snapshot'],r['sha256'])
        return cache[key]
    targets=public(BASE,PAPER+'mechanism_targets_DRAFT.csv','targets.csv')
    main=[t for t in targets if t['bias_mode']=='USUAL_SIGNED' and t['factor']=='1.0' and t['deterministic_scope'] in ('old_plus_direct_subset','strengthened_S2_S4','strengthened184')]
    confirm=[t for t in targets if t['PR_role']=='CONFIRMATION_N48' and t['bias_mode']=='USUAL_SIGNED']
    fixed=main+confirm;assert len(main)==30 and len(confirm)==2 and len(fixed)==32
    secondary=[t for t in targets if t not in fixed];assert len(secondary)==212
    catalog=public(COST,'artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/a_candidates.json','REGISTRATION')
    cmps=public('64c85f7a833b5950ef2ccae623708b73081edcd6','artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv','comparisons.csv')
    hcmp=public(H8,HP+'comparisons.csv','h8_comparisons.csv');haxes=public(H8,HP+'paired_metrics.csv','h8_axes.csv')
    signal_sets={s:[] for s in ['H4','H6','H8']};signal_proof=[]
    for s in signal_sets:
        config=old_input(s,'native_config_metadata')
        native=config['binding'];full=catalog[s]['binding']
        assert all(native[k]==full[k] for k in ['hamiltonian','state','compiler']),('NATIVE_BINDING_MISMATCH',s)
        assert native.get('snapshot',native.get('input'))==full['snapshot'],('SNAPSHOT_BINDING_MISMATCH',s)
        assert old_input(s,'NEW_A_SEAL')['binding']==full and old_input(s,'D_spec')['binding']==full,('FULL_SCIENCE_BINDING_MISMATCH',s)
        signal_sets[s].append(('NEW_A_SAVED_SIGNAL',rows(old_input(s,'A_signal'))))
        roles={'H4':['old_signals'],'H6':['A_origin_signal','D_saved_signal'],'H8':['old_signal_q4','old_signal_q8']}[s]
        for item in config['inputs']:
            if item['role'] in roles:
                data=private(s+'/'+item['role'],item['path'],item.get('sha256'));signal_sets[s].append((item['role'],rows(data)))
                signal_proof.append(dict(system=s,role=item['role'],parent_binding_matches=True,source=config.get('provider_source'),origin=item['path']))
    # Paths below are explicit references in fixed prepare_pr_cost_batch.py.
    h4origin=Path('/home/AbeHiromu/projects/h4-native-validation/20261018/private')
    signal_sets['H4'].append(('H4_ORIGINAL_D_SIGNALS',rows(private('H4/D_SIGNALS',h4origin/'D_run01/output/signals.json'))))
    old_d_spec=old_input('H4','D_spec');assert old_d_spec['binding']==catalog['H4']['binding']
    h6config_path=Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261016/track-a-v1-reaccount-run01/config_ready.json')
    h6config=private('H6/V1_CONFIG',h6config_path)
    for role in ('additional','old_signals','b0i_signals'):
        signal_sets['H6'].append(('H6_V1_'+role,rows(private('H6/V1_'+role,h6config['input_mapping'][role]))))
    signal_sets['H8'].append(('H8_FIXED_PHASE_A_SIGNALS',rows(public(COST,'artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/H8_signal_complete_v1.json.gz','H8/PHASE_A_SIGNALS'))))
    gap=[];normalized=[]
    def matching(records,cid,t):
        out=[(i,r) for i,r in enumerate(records) if r['cell_id']==cid and str(r['epsilon'])==t['epsilon'] and r['bias_mode']==t['bias_mode'] and str(r['factor']) in (t['factor'],t['factor']+'.0')]
        assert len(out)==1,('EXACT_ENDPOINT_ROW',cid,t['contrast_id']);return out[0]
    def explanation(system,cid,epsilon,shots,accounting=None):
        choices=[];partial=[]
        for label,signals in signal_sets[system]:
            for i,signal in enumerate(signals):
                if signal['cell']['id']!=cid:continue
                signed_partial=signal.get('error_decomposition',{}).get('total_signed')
                partial.append(dict(B=signal.get('b'),log_B=signal.get('log_B'),signed_bias=None if signed_partial is None else {a:signed_partial.get(ch) for a,ch in [('cosine','real'),('sine','imag')]},allowance=None,headroom=None,alpha_axis=None,log_shot_bound=None,integer_status=None,source_signal=label,source_signal_record=i,join_evidence='CANDIDATE_BINDING_ONLY;ACCOUNTING_ENDPOINT_NOT_SAVED_OR_NOT_BOUND;NO_INVERSE_FILL',candidate_cell=signal['cell']))
                ac=signal.get('precision_rows',{}).get(epsilon,{}).get('1.0')
                if not isinstance(ac,dict) or not isinstance(ac.get('axes'),dict):continue
                saved={a:ac['axes'][ch].get('shots') for a,ch in [('cosine','real'),('sine','imag')]}
                if saved!=shots:continue
                signed=signal.get('error_decomposition',{}).get('total_signed')
                desc=dict(B=signal.get('b'),log_B=signal.get('log_B'),signed_bias=None if signed is None else {a:signed.get(ch) for a,ch in [('cosine','real'),('sine','imag')]},
                    allowance={a:ac['axes'][ch].get('allowance') for a,ch in [('cosine','real'),('sine','imag')]},
                    headroom={a:ac['axes'][ch].get('headroom') for a,ch in [('cosine','real'),('sine','imag')]},
                    alpha_axis=ac.get('alpha_axis'),log_shot_bound={a:ac['axes'][ch].get('log_shot_bound') for a,ch in [('cosine','real'),('sine','imag')]},
                    integer_status={a:ac['axes'][ch].get('integer_status') for a,ch in [('cosine','real'),('sine','imag')]},source_signal=label,source_signal_record=i,
                    join_evidence='EXACT_CANDIDATE_ENDPOINT_AND_SAVED_SHOT_COPY;FIXED_PARENT_INPUT_CHAIN;NO_NUMERICAL_RECONSTRUCTION',candidate_cell=signal['cell'])
                choices.append(desc)
        # No choice depends on scientific values: fixed source order only.
        return choices[0] if choices else partial[0] if partial else None
    for t in fixed:
        s=t['system'];binding=catalog[s]['binding'];D=old_input(s,'D_result');DP=old_input(s,'D_payload');A=old_input(s,'A_result');NR=old_input(s,'NEW_A_RESULT')
        j,pfrow=matching(D['rows'],t['PF_id'],t)
        raw_axes=pfrow['axis_costs'];axis_map=raw_axes if isinstance(raw_axes,dict) else {x['axis']:x for x in raw_axes} if raw_axes is not None else {}
        pf=dict(candidate_id=t['PF_id'],binding=copy.deepcopy(binding),axis_costs={a:axis_map[a]['RZ'] for a in ('cosine','sine')} if axis_map else None,
                shots=pfrow.get('shots'),eligible=pfrow['eligible'],saved_G_exact=pfrow.get('G_exact'),cost_class='VALIDATED_MODEL_ESTIMATE' if 'MODEL' in pfrow['cost_status'] else 'DIRECT_MEASURED',source_evidence=dict(alias=s+'/D_result',record=j,axis_evidence=axis_map),explanation=None)
        cid=t['PR_id'];role=t['PR_role'];cohort='FIXED_ANCHOR_SELECTION_AND_ESTIMATION' if s=='H4' and role=='OLD_PR' else 'CONFIRMATION_UNION_POST_SELECTION' if role=='OLD_PR' else 'EXPLORATION_N8' if role=='NEW_PR' else 'INDEPENDENT_FIXED_CONFIRMATION_N48'
        if role=='CONFIRMATION_N48':
            raw=hcmp[int(t['source_record'])-1];pairs=[]
            grouped={}
            for record in haxes:
                if record['scope']==raw['scope'] and record['candidate']==cid:grouped.setdefault(record['replica'],{})[record['axis']]=record
            for replica,axes in grouped.items():
                assert set(axes)=={'cosine','sine'}
                pair_id=raw['scope']+'|'+cid+'|'+replica
                pairs.append(dict(replica=int(replica),pair_id=pair_id,candidate_id=cid,cohort=cohort,reused_complete_trajectory=axes['cosine']['reused_complete_trajectory'],axes={a:dict(RZ=r['RZ'],pair_id=pair_id) for a,r in axes.items()},pairing_evidence='PUBLIC_SCOPE_CANDIDATE_REPLICA_AXES;PRIVATE_SEED_NOT_REAUDITED'))
            shots={a:int(raw['shots_'+a]) for a in ('cosine','sine')};eligible=True;G=raw['mean_exact'];se=number_text(raw['paired_SE']);reduction=number_text(raw['PR_reduction_percent_exact']);n=48
            assert raw['fixed_PF_G_exact']==pf['saved_G_exact']
        else:
            raw=cmps[int(t['source_record'])-1]
            if role=='NEW_PR':
                i,saved=matching(NR['new_rows'],cid,t);ac=saved['accounting'];shots={a:ac['axes'][ch]['shots'] for a,ch in [('cosine','real'),('sine','imag')]};eligible=ac['cost_acquisition_eligible'];originals=[p for p in A['paired_costs'] if p['cell_id']==cid];G=raw['new_n8_G_exact'];se=number_text(raw['new_n8_G_SE']);reduction=number_text(raw['new_vs_PF_reduction_percent_exact']);n=8
            else:
                i,saved=matching(D['rows'],cid,t);shots=saved['shots'];eligible=saved['eligible'];originals=DP['paired_PR'][cid];G=raw['old_PR_G_exact'];se=number_text(raw['old_PR_G_SE']);reduction=number_text(raw['old_vs_PF_reduction_percent_exact']);n=32
            pairs=[]
            for original in originals:
                assert original['binding']==binding,('PAIR_SOURCE_BINDING',s,cid)
                pair_id=cid+'|'+str(original['replica']);axes=original['axes']
                pairs.append(dict(replica=original['replica'],pair_id=pair_id,candidate_id=cid,cohort=cohort,seed=original['seed'],event_digest=original['event_digest'],original_cohort=original['cohort'],
                    axes={a:dict(RZ=axes[a]['RZ'],pair_id=pair_id,seed=axes[a]['seed'],event_digest=axes[a]['event_digest']) for a in ('cosine','sine')}))
        pr=dict(candidate_id=cid,binding=copy.deepcopy(binding),cohort=cohort,n=n,trajectories=pairs,shots=shots,eligible=eligible,saved_G_exact=G,source_evidence=dict(result_commit=t['source_commit'],source_file=t['source_file'],source_record=t['source_record']),explanation=None)
        assert len(pairs)==n and len({p['replica'] for p in pairs})==n
        for method,side in [('PR',pr),('PF',pf)]:
            side['explanation']=explanation(s,side['candidate_id'],t['epsilon'],side['shots'])
            expl=side['explanation'];missing=[k for k in ['log_B','signed_bias','allowance','headroom','alpha_axis'] if expl is None or expl.get(k) is None]
            gap.append(dict(contrast_id=t['contrast_id'],system=s,method=method,candidate_id=side['candidate_id'],core_field_presence='PRESENT_NOT_NUMERICALLY_TESTED',explanation_field_presence='PRESENT_NOT_NUMERICALLY_TESTED' if not missing else 'EXPLANATORY_INPUT_MISSING',missing_fields=missing,source_signal=None if expl is None else expl['source_signal']))
        normalized.append(dict(artificial=False,contrast_id=t['contrast_id'],system=s,H4_stage='H4_PRE_D_P_VALIDATION' if s=='H4' else 'NOT_APPLICABLE',epsilon=t['epsilon'],factor='1',bias_mode=t['bias_mode'],deterministic_scope=t['deterministic_scope'],PR_role=role,PR_id=cid,PF_id=t['PF_id'],binding=copy.deepcopy(binding),PR=pr,PF=pf,cohorts_pooled=False,saved_SE=se,saved_reduction_percent_exact=reduction,source_preparation_commit=BASE))
    payload=dict(schema='TRACK_A_MECHANISM_V1',kind='SAVED_REAL_VALUES',target_ids=[t['contrast_id'] for t in fixed],comparisons=normalized)
    payload_path=destination/'payload_PRIMARY32_SAVED_UNEXECUTED.json';put(payload_path,payload)
    records.append(dict(alias='NORMALIZED_PAYLOAD',path=str(payload_path),sha256=sha_bytes(payload_path.read_bytes()),bytes=payload_path.stat().st_size,origin='metadata-only copies; no numerical analysis'))
    binding=dict(schema='TRACK_A_MECHANISM_V1',kind='SAVED_REAL_VALUES',status='READY_FOR_USER_AUTHORIZATION',target_ids=payload['target_ids'],target_count=32,secondary_target_count=212,payload=str(payload_path),payload_sha256=sha_bytes(payload_path.read_bytes()),input_files=records,protected_paths=[str(previous)],gaps=gap,signal_join_proof=signal_proof,scientific_numerical_analysis_executed=0,new_statistics=0)
    put(destination/'mechanism_input_binding_final_DRAFT.json',binding)
    put(destination/'secondary_212_METADATA_ONLY.json',secondary)
    print(json.dumps(dict(targets=32,secondary=212,explanation_methods_present=len([x for x in gap if not x['missing_fields']]),explanation_methods_missing=len([x for x in gap if x['missing_fields']]),science_analysis=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--previous-private',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();prepare(args.previous_private,args.directory)
