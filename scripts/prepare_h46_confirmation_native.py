"""Prepare an INVALID proposal only. No event sampler, chemistry or compiler."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

from h46_confirmation_contract import (ROOT, PHASE, GROUPS, PROPOSED_N, PAIRS, PROFILE,
    INPUT_SHA, digest, sha, task_id, scope, registry_audit, write)

SOURCE_PATHS = ('scripts/h46_confirmation_contract.py','scripts/h46_confirmation_native.py',
    'scripts/launch_h46_confirmation.py','scripts/h46_confirmation_collect.py',
    'scripts/prepare_h46_confirmation_native.py','scripts/h46_acquire_native.py',
    'scripts/h46_new2_native.py','scripts/h46_new2_core.py','scripts/h46_higher_pf_adapter.py',
    'src/trotterlib/higher_pf_readiness.py')


def metadata_seeds(value, key=''):
    if isinstance(value,dict):
        for key,item in value.items():yield from metadata_seeds(item,key)
    elif isinstance(value,list):
        for item in value:yield from metadata_seeds(item,key)
    elif 'seed' in key.lower():
        if type(value) is int:yield value
        elif isinstance(value,str) and value.isdigit():yield int(value)


def dictionaries(value):
    if isinstance(value,dict):
        yield value
        for item in value.values():yield from dictionaries(item)
    elif isinstance(value,list):
        for item in value:yield from dictionaries(item)


def previous_seeds(root, old_registry):
    """Same bounded legacy rules used by the published B acquisition audit."""
    import hashlib
    legacy=set();sources=('687679a4ae20e4661bde1187dd8ab92ae176bd44',
        '8b0f6cf2263a7c6e0af973c9bc742178fa0bd514','f0dd2a9aa4e011b204c4ba6c51a050dff09bc471')
    for source in sources:
        for system in ('H4','H6'):
            for short,q,pieces in [('S2',4,1),('Y4',2,3),('N2',1,5)]:
                for sample in range(8):
                    for outer in range(q):
                        for piece in range(pieces):
                            legacy.add(int.from_bytes(hashlib.sha256(f'{source}|{system}_{short}|{sample}|{outer}|{piece}'.encode()).digest()[:8],'big'))
    descriptors={};evidence=[]
    for ref in sorted((root/'references').glob('*.json')):
        content=json.loads(ref.read_text());legacy.update(metadata_seeds(content))
        evidence.append({'path':str(ref.relative_to(root)),'sha256':sha(ref)})
        for row in dictionaries(content):
            path=row.get('path') or row.get('original_path');commit=row.get('commit') or row.get('original_commit')
            if isinstance(path,str) and isinstance(commit,str) and ('provenance' in path or 'seed' in path) and path.endswith(('.csv','.json')):
                descriptors[(commit,path)]=row
    for commit,path in sorted(descriptors):
        try:data=subprocess.check_output(['git','show',commit+':'+path],cwd=root,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:continue
        content=list(csv.DictReader(io.StringIO(data.decode()))) if path.endswith('.csv') else json.loads(data)
        legacy.update(metadata_seeds(content))
        evidence.append({'commit':commit,'path':path,'sha256':hashlib.sha256(data).hexdigest()})
    acquired={v for seeds in old_registry['tasks'].values() for v in seeds.values()}
    if len(legacy)!=1505 or len(acquired)!=1824 or legacy & acquired:raise ValueError('PUBLISHED_EXCLUSION_INVENTORY_CHANGED')
    return legacy|acquired, {'legacy_unique':1505,'B_acquired_unique':1824,'total_unique':3329,
                            'legacy_SOURCEs':sources,'bounded_reference_evidence':evidence,
                            'scope':'KNOWN_FIXED_HIGHER_PF_EXPLORATION_AND_ACCESSIBLE_SEED_METADATA_NOT_UNKNOWN_JOBS'}


def prepare(old_config_path, destination, source):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    old=json.loads(Path(old_config_path).read_text())
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=source:raise ValueError('SOURCE_HEAD')
    for path in SOURCE_PATHS:
        data=subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)
        if data!=(ROOT/path).read_bytes():raise ValueError('SOURCE_TRACKED_BYTES')
    for item in old['sealed_files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('PREVIOUS_IMMUTABLE_INPUT_CHANGED')
    old_registry=json.loads(Path(old['seed_registry_path']).read_text())
    excluded,exclusion_audit=previous_seeds(ROOT,old_registry)
    integrated=ROOT/'artifacts/resource_applicability/track_a_h46_higher_pf_integrated_run01/2026-10-18/result.json'
    if sha(integrated)!='ef6adede7b64a91b672df0f94e7a183da71421a0eeac2b7633b97aaa9bcd612e':raise ValueError('FIXED_EXPLORATION_EVIDENCE')
    rows=json.loads(integrated.read_text())['comparison']['rows'];bindings={}
    # All H6 groups share the saved p5 preparation. q-specific shots/log_B are
    # copied from frozen outputs, never recalculated in this readiness step.
    signal_path=ROOT/'artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/H6_run02_results.json'
    signal=json.loads(signal_path.read_text())
    for group in GROUPS:
        row=next(r for r in rows if r['candidate_id']==group['id'])
        finite=next(r for r in signal['records'] if (r['label'],r['prefix'],r['q'])==(group['label'],5,group['q']))
        mean=next(r for r in finite['finite_rows'] if r['cutoff']==4)
        saved=old['group_binding']['H6_S2_q8' if group['label']=='2nd' else 'H6_Y4_q4']
        bindings[group['id']]={**saved,'shots':row['shots'],'log_B':mean['log_B']}
    inputs={k:v for k,v in old['inputs'].items() if k.startswith('H6_')}
    payload={'study_id':'HIGHER_PF_PURPOSE_A_REPLICATION','stage':'CONFIRMATION','SOURCE':source,
             'groups':list(GROUPS),'n_per_candidate':PROPOSED_N,'input_sha256':INPUT_SHA,'selection_evidence_sha256':sha(integrated)}
    seed_scope=digest(payload);namespace='H6_HIGHER_PF_C_CONFIRMATION:'+source+':'+seed_scope
    plan={'phase':PHASE,'study_id':payload['study_id'],'status':'DRAFT_UNAPPROVED','SOURCE':source,
          'provider_SOURCE':old['provider_SOURCE'],'sample_size_formally_adopted':False,
          'groups':list(GROUPS),'n_per_candidate':PROPOSED_N,'comparison_pairs':[list(p) for p in PAIRS],
          'trajectories':103,'sample_axes':206,'extra_scientific_compiles':0,'native_validation_checks_expected':536,
          'compiler_settings_hash':PROFILE,'numeric_tolerance':1e-9,
          'fixed_science':{'T':.8,'K':4,'RTE_steps_per_outer':32,'epsilon':.001,'alpha_axis':.025,'allowance':1e-9,'log_B_margin':1e-12,'P':0,'control':'ordinary'},
          'purpose':'INDEPENDENT_POINT_ORDER_REPLICATION_NOT_POPULATION_GUARANTEE',
          'analysis_policy':{'primary':'EXACT_MEAN_DIFFERENCE_SIGN','secondary':'NORMAL_BONFERRONI_APPROX_DIAGNOSTIC','MC_family_alpha':.05,'family_size':2,'population_guarantee':False,'pool_exploration':False},
          'seed_scope_payload':payload,'seed_scope_digest':seed_scope,'seed_namespace':namespace,
          'input_sha256':INPUT_SHA,'selection_evidence_sha256':sha(integrated),'retry':False,'mandatory_STOP':True}
    plan['tasks']=[task_id(plan,g,i) for g in GROUPS for i in range(PROPOSED_N[g['id']])]
    from h46_higher_pf_adapter import kernel
    k=kernel();registry={'stage':'CONFIRMATION','namespace':namespace,'seed_scope_digest':seed_scope,'tasks':{}}
    occupied=set(excluded)
    for g in GROUPS:
        for sample in range(PROPOSED_N[g['id']]):
            values={}
            for outer in range(g['q']):
                for piece in range(1 if g['label']=='2nd' else 3):
                    value=k.seed_for(namespace,'CONFIRMATION',g['id'],sample,outer,piece,INPUT_SHA,occupied)
                    occupied.add(value);values[f'{outer}:{piece}']=value
            registry['tasks'][task_id(plan,g,sample)]=values
    registry_path=destination/'seed_registry_PROVISIONAL_PRIVATE.json';write(registry_path,registry)
    exclusion_path=destination/'excluded_seeds_PRIVATE.json';write(exclusion_path,{'values':sorted(excluded),'audit':exclusion_audit})
    plan['seed_registry_sha256']=sha(registry_path)
    sealed=[e for e in old['sealed_files'] if (str(e['path']).startswith(old['provider_root']+'/') or '/site-packages/' in e['path'] or e['path']=='/usr/bin/python3.12')]
    paths=[ROOT/p for p in SOURCE_PATHS]+[Path(v['path']) for v in inputs.values()]+[integrated,signal_path,registry_path,exclusion_path]
    sealed += [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
    plan.update(closure_digest=digest(sealed),input_bindings_digest=digest(inputs),group_binding_digest=digest(bindings))
    scope(plan);audit=registry_audit(plan,registry,excluded)
    plan_path=destination/'plan_DRAFT_PRIVATE.json';write(plan_path,plan)
    grant={'valid':False,'phase':PHASE,'SOURCE':source,'plan_sha256':sha(plan_path),'grant_id':None,'approval_reference':None,
           'maximum_trajectories':103,'maximum_cost_axes':206,'maximum_extra_compiles':0,'retry':False,
           'output':str(destination/'future_run01'),'call_ledger':str(destination/'future_ledger_run01'),
           'grant_marker':str(destination/'future_grant_CONSUMED.json')}
    config={'authorized':False,'SOURCE':source,'provider_SOURCE':old['provider_SOURCE'],'source_root':str(ROOT),
            'provider_root':old['provider_root'],'python':old['python'],'plan':plan,'plan_path':str(plan_path),'plan_sha256':sha(plan_path),
            'seed_registry_path':str(registry_path),'seed_registry_sha256':sha(registry_path),'excluded_seed_path':str(exclusion_path),
            'inputs':inputs,'sealed_files':sealed,'group_binding':bindings,'grant':grant,
            **{key:grant[key] for key in ('output','call_ledger','grant_marker')},
            'resources':{'group_estimated_RSS':{'H6_S2_q8':1579675648,'H6_Y4_q2':1319124992,'H6_Y4_q4':1723486208}}}
    write(destination/'config_DRAFT_PRIVATE.json',config);write(destination/'grant_INVALID.json',grant)
    write(destination/'seed_audit.json',{'status':'PROVISIONAL_METADATA_ONLY_NO_EVENT_SAMPLING',**audit,**exclusion_audit,'registry_sha256':sha(registry_path)})
    return {'status':'DRAFT_PREPARED_NO_SCIENTIFIC_EXECUTION','SOURCE':source,'proposed_trajectories':103,'proposed_axes':206,
            'fresh_seed_values':914,'excluded_unique':3329,'sealed_files':len(sealed),'valid_grants':0,'sample_size_adopted':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',required=True);p.add_argument('--destination',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.old_config,a.destination,a.source)))


if __name__=='__main__':main()
