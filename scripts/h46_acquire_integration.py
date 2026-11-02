"""Fail-closed input assembly for a future, separately approved comparison.

inspect() joins metadata/cost-record identities only. assemble_rows() does not
compute weighted costs, statistics or rankings; the comparison kernel is lazy.
"""
import json
from pathlib import Path

from h46_higher_pf_adapter import load_rows, kernel, INPUT_HASH


def inspect(config, campaign_output):
    c=config;k=kernel();root=Path(campaign_output)
    witness=json.loads((root/'completion_witness_PRIVATE.json').read_text())
    if witness['status']!='COMPLETE_ACQUISITION_ONLY' or not witness['all_pool_workers_reaped']:
        raise ValueError('ACQUISITION_NOT_COMPLETE')
    groups=[]
    for group in c['plan']['groups']:
        records=[]
        for sample in range(8):
            result=json.loads((root/group['id']/('sample_%02d'%sample)/'result.json').read_text())
            if result['group']!=group or result['sample']!=sample or result['SOURCE']!=c['SOURCE']:
                raise ValueError('SOURCE_GROUP_SAMPLE')
            if result['plan_sha256']!=c['plan_sha256'] or result['seed_registry_sha256']!=c['seed_registry_sha256']:
                raise ValueError('PLAN_SEED_IDENTITY')
            if result['status']!='ACQUIRED_VALIDATED_NO_ACCOUNTING' or result['forced_events_enter_cost_sample'] is not False:
                raise ValueError('STATUS_FORCED_EVENT')
            if result['input_sha256']!=INPUT_HASH[group['system']] or result['preparation_hash']!=c['group_binding'][group['id']]['preparation_hash']:
                raise ValueError('INPUT_PREPARATION')
            axes={r['axis']:r for r in result['records']}
            if set(axes)!={'cosine','sine'} or len(result['records'])!=2:
                raise ValueError('PAIRED_AXES')
            if axes['cosine']['event_digest']!=axes['sine']['event_digest']:
                raise ValueError('UNPAIRED_EVENT')
            if any(r['compiler_settings_hash']!=k.PROFILE for r in axes.values()):
                raise ValueError('COMPILER')
            records.extend(result['records'])
        groups.append({'candidate_id':group['id'],'sample_count':8,'cost_axes':len(records),'SOURCE':c['SOURCE'],
                       'compiler_settings_hash':k.PROFILE,'preparation_hash':c['group_binding'][group['id']]['preparation_hash'],
                       'input_sha256':INPUT_HASH[group['system']],'seed_namespace':c['plan']['seed_namespace']})
    return {'status':'METADATA_BOUND_NO_ACCOUNTING','new_groups':groups,'new_cost_axes':192,
            'saved_groups_reused':6,'eligible_groups_ready':18,'ranking_executed':False,
            'future_approval_required':True}


def assemble_rows(config,campaign_output):
    inspect(config,campaign_output)
    rows,metadata=load_rows();k=kernel();group_map={g['id']:g for g in config['plan']['groups']}
    for row in rows:
        group=group_map.get(row['candidate_id'])
        if group is None:continue
        row['identity']['SOURCE']=config['SOURCE'];row['records']=[];row['cost_kind']='MEASURED'
        for sample in range(8):
            result=json.loads((Path(campaign_output)/group['id']/('sample_%02d'%sample)/'result.json').read_text())
            for record in result['records']:
                row['records'].append({**row['identity'],'replica_id':str(sample),'axis':record['axis'],
                    'event_digest':record['event_digest'],'seed_namespace':config['plan']['seed_namespace'],
                    'forced_event':False,'cost_kind':'MEASURED',**record['metrics']})
    return rows,metadata


def compare_after_separate_approval(config,campaign_output,approval):
    if approval.get('phase')!='B_INTEGRATED_COMPARE' or approval.get('valid') is not True:
        raise ValueError('SEPARATE_INTEGRATED_COMPARISON_APPROVAL_REQUIRED')
    raise ValueError('NEW_ONE_SHOT_COMPARISON_CONTRACT_MUST_BE_SEALED_BEFORE_DISPATCH')
