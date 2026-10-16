"""B1 receipt binding and grant gate. Importing or default CLI never fits data."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import h8_phase_b1_model as model
import h8_b0_anchor_contract as acquisition
import h8_strengthened_runtime as io

REGISTRY='artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16/anchor_registry_v1.json'
REGISTRY_SHA='b58b236d6b767a06142a87d6e482aa356a334f81b15ba506b98bdf035ca82f83'
EVIDENCE='artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16/source_evidence_v1.json'
INPUTS=('B0_config','B0_grant','B0_terminal','B0_launcher_terminal','B0_final_gate','B0_fresh_gate','B0I_registry')
SOURCES=('src/trotterlib/h8_phase_b1_model.py','src/trotterlib/h8_phase_b1_contract.py',
    'scripts/h8_phase_b1_validation.py','scripts/run_h8_phase_b1_validation.py',
    'scripts/launch_h8_phase_b1_validation.py','scripts/h8_b0_anchor_contract.py',
    'src/trotterlib/h8_b0_anchor_plan.py','scripts/h8_b0_anchor_runtime.py',
    'scripts/h8_strengthened_runtime.py','scripts/h8_strengthened_contract.py',
    'scripts/launch_h8_strengthened_phase_a.py','scripts/track_a_phase_a_parallel.py',
    'scripts/launch_h8_b0_anchors.py',
    acquisition.PLAN_PATH,acquisition.TARGET_PATH,REGISTRY,EVIDENCE)
require=model.require


def write(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(data)<=model.RESOURCES['output_bytes'],'OUTPUT_CAP')
    with Path(path).open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def verify_receipts(items):
    require(type(items) is dict and set(items)==set(INPUTS),'EXACT_RECEIPT_INVENTORY')
    for row in items.values():
        path=Path(row['path'])
        require(path.is_absolute() and path.resolve().is_relative_to('/home/AbeHiromu')
            and not path.is_symlink() and path.is_file(),'HOME_REGULAR_INPUT_REQUIRED')
        require(type(row['bytes']) is int and 0<row['bytes']<=32*2**20
            and model.hex_digest(row['sha256']),'RECEIPT_LAYOUT')
        require(path.stat().st_size==row['bytes'] and io.sha(path)==row['sha256'],'INPUT_BYTES_MISMATCH')
    require(items['B0I_registry']['sha256']==REGISTRY_SHA,'PINNED_B0I_REGISTRY_REQUIRED')


def completed_acquisition(source,t,launcher,final):
    require(t.get('status')=='COMPLETE_MANDATORY_STOP' and t.get('SOURCE')==source
        and t.get('next_stage_authorized') is False,'ACQUISITION_NOT_COMPLETE')
    require(launcher.get('status')=='COMPLETE_MANDATORY_STOP' and launcher.get('native_exit_code')==0
        and launcher.get('first_stop') is None and launcher.get('cleanup_errors')==[]
        and launcher.get('all_reaped') is True and type(launcher.get('max_monitor_gap_seconds')) in (int,float)
        and 0<=launcher['max_monitor_gap_seconds']<=5,'ACQUISITION_STOP_OR_CLEANUP_FAILURE')
    require(final==dict(status='PASS_MANDATORY_STOP',SOURCE=source,profile_sha256=model.PROFILE,
                        next_stage_authorized=False),'ACQUISITION_FINAL_GATE')


def normalize(config):
    """Receipt-only validation/normalization: no design rank, fit or residuals."""
    items=config['inputs'];verify_receipts(items)
    values={k:io.read(row['path']) for k,row in items.items()}
    old,g,t,launcher,final,fresh=(values[k] for k in INPUTS[:6])
    source=config['B0_acquisition_SOURCE'];require(model.hex_digest(source,40),'B0_SOURCE_REQUIRED')
    completed_acquisition(source,t,launcher,final)
    require(old['SOURCE']==t['SOURCE']==source and old['execution_sealed'] is True,'ACQUISITION_SOURCE_BINDING')
    require(old['provider_source']==acquisition.PROVIDER and old['expected_profile_digest']==model.PROFILE,
            'ACQUISITION_PROVIDER_PROFILE')
    require(len(old['runtime_file_hashes'])==34273 and model.digest(old['runtime_file_hashes'])==acquisition.RUNTIME_MAP_SHA
        and len(old['provider_hashes'])==265 and model.digest(old['provider_hashes'])==acquisition.PROVIDER_MAP_SHA,
        'ACQUISITION_PROFILE_CLOSURE')
    p=acquisition.plan();binding=p['binding']
    files={r['logical_name']:r for r in old['files']}
    require(len(files)==len(old['files']),'ACQUISITION_DUPLICATE_INPUT')
    for expected in binding['inputs']:
        require(all(files[expected['logical_name']][k]==expected[k] for k in ('bytes','sha256')),
                'ACQUISITION_SCIENTIFIC_INPUT_BINDING')
    require(g.get('schema')=='H8_B0_ANCHOR_NEW_SCOPE_GRANT_V1'
        and g.get('approved_by_user') is True and g.get('runtime_authorization') is True
        and type(g.get('approval_text')) is str and bool(g['approval_text'].strip())
        and g.get('SOURCE')==source and g.get('scope')==acquisition.SCOPE
        and g.get('config_sha256')==items['B0_config']['sha256']
        and g.get('plan_sha256')==old.get('plan_sha256')==model.digest(p)
        and g.get('resources')==old.get('resources')==p['resources']
        and g.get('call_caps')==p['call_caps']
        and all(g.get(k) is v for k,v in [('one_shot',True),('retry',False),('resume',False),('next_stage_authorized',False)]),
        'ACQUISITION_GRANTED_BINDING')
    require(fresh.get('source')==source and fresh.get('config_sha256')==items['B0_config']['sha256']
        and fresh.get('grant_sha256')==model.digest(g)
        and all(fresh.get(k) is True for k in ('profile_verified','source_input_bytes_verified',
            'physical_cores_verified','resources_idle_verified','protected_jobs_unchanged','quota_verified',
            'external_RSS_monitor_ready','driver_bindings_verified')),'ACQUISITION_FRESH_PROOF')
    acquisition.original.validate_terminal(acquisition.original.proposal(binding),t)
    input_binding=model.digest(binding['inputs']);rows=[]
    for r in t['axes']:
        case=next(c for c in acquisition.original.cases() if c['id']==r['case_id'])
        rows.append(dict(method='B0',prefix=case['prefix'],q=case['q'],axis=r['axis'],formula='2nd',
            saved_RZ=r['measured_RZ'],role=r['role'],origin_SOURCE=source,
            runtime_profile_sha256=r['profile_sha256'],compiler_hash=r['compiler_hash'],
            control=r['control'],scalar_policy=r['scalar_policy'],
            cost_preparation_hash=r['preparation_binding']['saved_preparation_hash'],
            fingerprint=r['fingerprint_before'],input_binding_sha256=input_binding,
            evidence={k:True for k in model.REQUIRED}))
    # Reuse only the fixed audit's 16 B0I rows. Its unresolved old B0 rows are ignored.
    registry=values['B0I_registry'];require(registry['schema']=='H8_B1_ANCHOR_METADATA_V1','B0I_REGISTRY_SCHEMA')
    for r in registry['anchors']:
        if r['method']!='B0I':continue
        row=deepcopy(r);row.update(role='TRAIN',input_binding_sha256=input_binding);rows.append(row)
    model.validate_rows(rows,source,input_binding)
    return rows,input_binding


def verify_source(config):
    require(model.hex_digest(config['SOURCE'],40) and config['execution_sealed'] is True,'SEALED_B1_SOURCE_REQUIRED')
    require(set(config['source_hashes'])==set(SOURCES),'B1_SOURCE_CLOSURE')
    for name,expected in config['source_hashes'].items():
        require(io.sha(ROOT/name)==expected,'B1_SOURCE_DISK_CHANGED')
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',config['SOURCE']+':'+name])
        require(hashlib.sha256(blob).hexdigest()==expected,'B1_SOURCE_GIT_CHANGED')
    require(io.sha(config['python'])==config['python_sha256'],'B1_PYTHON_BYTES_CHANGED')
    require(config['plan_sha256']==model.digest(model.plan()),'FIXED_B1_PLAN_CHANGED')


def authorize(config,grant,*,config_sha,affinity):
    require(grant.get('schema')=='H8_B1_NEW_FIT_GRANT_V1' and grant.get('valid') is True
        and grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True
        and type(grant.get('approval_text')) is str and bool(grant['approval_text'].strip())
        and grant.get('scope')==model.SCOPE and grant.get('SOURCE')==config['SOURCE']
        and grant.get('config_sha256')==config_sha
        and grant.get('plan_sha256')==config.get('plan_sha256')==model.digest(model.plan())
        and grant.get('resources')==model.RESOURCES and grant.get('call_caps')==model.CAPS
        and all(grant.get(k) is v for k,v in [('one_shot',True),('retry',False),('resume',False),
            ('refit',False),('additional_measurement',False),('B2_authorized',False),
            ('C_authorized',False),('next_stage_authorized',False)]),'NEW_EXPLICIT_B1_GRANT_REQUIRED')
    cpus=[grant.get('driver_cpu'),grant.get('observer_cpu')]
    require(all(type(c) is int and c>=0 for c in cpus) and len(set(cpus))==2
        and grant.get('allowed_cpus')==cpus and set(cpus)<=set(affinity),'TWO_EXPLICIT_B1_CPUS')
    paths=[io.private_path(config[k]) for k in ('output','control','lock')]
    require(len(set(paths))==3 and len({p.parent for p in paths})==1,'EXCLUSIVE_SIBLING_B1_PATHS')
    return cpus


def prepare(receipt_manifest,directory,source,python):
    require(model.hex_digest(source,40),'SOURCE_REQUIRED')
    require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==source,'SOURCE_HEAD')
    directory=io.private_path(directory);require(not directory.exists(),'NEW_DRAFT_DIRECTORY_REQUIRED')
    spec=io.read(receipt_manifest)
    config=dict(schema='H8_B1_CONFIG_V1',SOURCE=source,execution_sealed=True,
        inputs=spec['inputs'],B0_acquisition_SOURCE=spec['B0_acquisition_SOURCE'],
        source_hashes={name:io.sha(ROOT/name) for name in SOURCES},
        python=str(Path(python).resolve()),python_sha256=io.sha(python),
        plan_sha256=model.digest(model.plan()),output=str(directory/'output01'),
        control=str(directory/'control01'),lock=str(directory/'one_shot.lock'))
    verify_source(config);rows,binding=normalize(config)  # Metadata only, no fit/rank/residual.
    directory.mkdir(mode=0o700);write(directory/'config_UNAPPROVED.json',config)
    grant=dict(schema='H8_B1_NEW_FIT_GRANT_V1',valid=False,approved_by_user=False,runtime_authorization=False,
        approval_text=None,scope=model.SCOPE,SOURCE=source,config_sha256=io.sha(directory/'config_UNAPPROVED.json'),
        plan_sha256=config['plan_sha256'],resources=model.RESOURCES,call_caps=model.CAPS,
        one_shot=True,retry=False,resume=False,refit=False,additional_measurement=False,
        driver_cpu=None,observer_cpu=None,allowed_cpus=[],command=None,
        B2_authorized=False,C_authorized=False,next_stage_authorized=False)
    write(directory/'grant_INVALID_DRAFT.json',grant)
    return dict(status='PREPARED_NOT_AUTHORIZED',SOURCE=source,axes=len(rows),
                input_binding_sha256=binding,actual_fit=0,actual_residual=0,valid_grants=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-receipts',type=Path)
    p.add_argument('--directory',type=Path);p.add_argument('--source');p.add_argument('--python',type=Path)
    a=p.parse_args()
    if a.prepare_receipts:
        if not all((a.directory,a.source,a.python)):p.error('directory/source/python are required')
        print(json.dumps(prepare(a.prepare_receipts,a.directory,a.source,a.python),indent=2))
    else:
        if any((a.directory,a.source,a.python)):p.error('real paths require explicit preparation mode')
        print(json.dumps(model.plan(),indent=2))


if __name__=='__main__':main()
