#!/usr/bin/env python3
"""Static byte/lineage/coverage audit of supplement; never numerical re-evaluation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v3 import (
    validate_manifest,validate_grant,verify_sources,file_hash,verify_parent,environment)
from trottertracks.resource_applicability.h6_matched_reuse_v3 import (
    read,verify_evidence,admit_group,complete_exploration)
from trottertracks.resource_applicability.h6_matched_accounting_v1 import tasks,confirmation_ids
from trottertracks.resource_applicability.ax2a_preparation import digest


def audit(root,output):
    root=Path(root);output=Path(output);m=read(output/'frozen_manifest.json')
    validate_manifest(m);validate_grant(m,read(output/'authorization.json'),output.resolve())
    verify_sources(root,m['source_commit'],m['source_hashes']);verify_evidence(root,m['reuse_registry'])
    if verify_parent(root)!=m['input_identity'] or environment()!=m['environment']:raise ValueError('AUDIT_INPUT_ENVIRONMENT')
    g=read(output/'authorization.json')
    if read(output/'authorization_source.json')!=g or read(output/'launch_binding.json')!={
        'manifest_digest':digest(m),'authorization_digest':digest(g),
        'authorization_source_sha256':file_hash(output/'authorization_source.json')}:raise ValueError('AUDIT_GRANT_BYTES')
    inv=read(output/'execution_inventory.json');term=read(output/'execution_terminal.json')
    if inv['source_commit']!=m['source_commit'] or term['source_commit']!=m['source_commit']:raise ValueError('AUDIT_SOURCE')
    paths={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and '.runtime_cache' not in p.parts
        and p.name not in ('execution_inventory.json','execution_terminal.json','STOP_REQUEST')}
    if set(inv['file_hashes'])!=paths:raise ValueError('AUDIT_RAW_CLOSURE')
    for p,sha in inv['file_hashes'].items():
        path=(output/p).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path)!=sha:raise ValueError('AUDIT_RAW_BYTES:'+p)
    if (read(output/'coordinator_limits.json')['rlimit_as']!=[2*2**30,8*2**30] or
        term['mandatory_stop'] is not True or term['next_stage_authorized'] is not False or
        term['H6_status']!='H6_NOT_AUTHORIZED' or term['contract_status']!='DRAFT_NOT_AUTHORIZATION'):
        raise ValueError('AUDIT_LIMITS_STOP')
    r=m['reuse_registry'];registered=r['original_tasks'];groups={};new_count=0;new_by_phase={}
    for x in r['accepted']:groups[x['index']]=admit_group(root/x['directory'],registered[x['index']])
    for path in sorted(output.glob('cost_*_[0-9][0-9][0-9][0-9]')):
        terminal=path/'worker_terminal.json'
        if not terminal.exists() or read(terminal)['status']!='TASK_COMPLETE':continue
        phase=path.name.split('_')[1];i=int(path.name.rsplit('_',1)[1])
        if phase=='exploration':
            if i not in {x['index'] for x in r['missing']} or i in groups:raise ValueError('AUDIT_RECOMPUTATION')
            task=registered[i]
        elif phase=='confirmation':task=read(output/'confirmation_tasks.json')['tasks'][i]
        else:raise ValueError('AUDIT_COST_PHASE')
        result=admit_group(path,task,source=m['source_commit']);new_count+=len(result['rows'])
        new_by_phase.setdefault(phase,{})[i]=result
        if phase=='exploration':groups[i]=result
    full=term['status']=='H6_MATCHED_SUPPLEMENT_COMPLETE_MANDATORY_STOP'
    if full:
        rows=complete_exploration(registered,groups)
        lineage=read(output/'integrated_exploration.json')
        if (lineage['group_count']!=592 or lineage['wrapper_count']!=1188 or
            lineage['complete_before_selection'] is not True or len(lineage['groups'])!=592):
            raise ValueError('AUDIT_LINEAGE_COVERAGE')
        old={x['index']:root/x['directory'] for x in r['accepted']}
        for i,x in enumerate(lineage['groups']):
            path=(old.get(i) or output/f'cost_exploration_{i:04d}')/'cost_result.json'
            if x!={'index':i,'task_digest':digest(registered[i]),'cost_result_path':str(path.relative_to(root)),
                'cost_result_sha256':file_hash(path),'source_commit':r['source_commit'] if i in old else m['source_commit']}:
                raise ValueError('AUDIT_LINEAGE_IDENTITY')
        signals=read(root/r['original_raw_path']/'signal/signal_summary.json')['signals']
        selected=confirmation_ids(signals,rows,m['plan']);confirmation=read(output/'confirmation_tasks.json')
        if (confirmation['selected']!=selected or confirmation['tasks']!=tasks(signals,m['plan'],'confirmation',selected) or
            confirmation['exploration_rows_digest']!=digest(rows) or
            confirmation['exploration_binding_sha256']!=file_hash(output/'integrated_exploration.json')):
            raise ValueError('AUDIT_SELECTION')
        complete_exploration(confirmation['tasks'],new_by_phase.get('confirmation',{}))
        if new_count>798 or 902+new_count>1700:raise ValueError('AUDIT_WRAPPER_CAP')
    elif term['status']!='H6_MATCHED_SUPPLEMENT_STOP':raise ValueError('AUDIT_STATUS')
    return dict(status='STATIC_BYTES_LINEAGE_COVERAGE_PASS',execution_status=term['status'],
        reused_groups=449,new_complete_exploration_groups=len(groups)-449,new_saved_complete_wrappers=new_count,
        full_registered_coverage_verified=full,numerical_certificate_verified=False,scientific_GO_issued=False,
        H6_status='H6_NOT_AUTHORIZED',contract_status='DRAFT_NOT_AUTHORIZATION',mandatory_stop=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(audit(ROOT,p.parse_args().output),sort_keys=True))
