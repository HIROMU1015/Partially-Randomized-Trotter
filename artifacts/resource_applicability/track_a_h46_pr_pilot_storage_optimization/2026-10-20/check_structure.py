"""Storage metadata arithmetic and immutable reference checks only."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('storage',ROOT/'scripts/h46_pilot_lossless_storage.py')
storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)

def read(name):return json.loads((HERE/name).read_bytes())

def check():
    checks=0
    def need(value,message):
        nonlocal checks
        if not value:raise ValueError(message)
        checks+=1
    summary=read('summary.json');source=summary['SOURCE']
    for r in read('source_closure.json')['files']:
        b=(ROOT/r['path']).read_bytes()
        need(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],'SOURCE_SHA')
        need(b==subprocess.check_output(['git','show',source+':'+r['path']],cwd=ROOT),'SOURCE_GIT_BYTES')
    with (HERE/'compression_evidence_276.csv').open() as f:rows=list(csv.DictReader(f))
    need(len(rows)==276,'276_COMPRESSION_OBSERVATIONS')
    need(len({(r['candidate_id'],r['axis'],r['component']) for r in rows})==276,'UNIQUE_BYTE_OBJECTS')
    oldpath='artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/private_raw_inventory.json'
    oldbytes=subprocess.check_output(['git','show',summary['base_commit']+':'+oldpath],cwd=ROOT)
    need(oldbytes==(ROOT/oldpath).read_bytes(),'SAVED_INVENTORY_UNCHANGED')
    old={(r['sha256'],r['bytes']) for r in json.loads(oldbytes)}
    pair={}
    for r in rows:
        need((r['original_sha256'],int(r['original_bytes'])) in old,'ORIGINAL_SHA_IN_FIXED_INVENTORY')
        need(r['raw_SHA_restoration_verified']=='True' and r['codec']=='zlib_deflate' and r['level']=='1','LOSSLESS_VERIFICATION')
        need(len(r['stored_sha256'])==64 and int(r['stored_bytes'])>0,'ARCHIVE_SHA_SIZE')
        key=(r['system'],r['candidate_id']);pair[key]=pair.get(key,0)+int(r['stored_bytes'])
    measure=read('compression_measurement.json')
    need(sum(int(r['original_bytes']) for r in rows)==measure['input_bytes']==10356702848,'RAW_BYTES_SUM')
    need(sum(int(r['stored_bytes']) for r in rows)==measure['stored_bytes']==635361139,'ARCHIVE_BYTES_SUM')
    need(measure['originals_modified_or_deleted']==0 and measure['scientific_execution']==0,'NO_SCIENTIFIC_RUN_OR_RAW_CHANGE')
    cases=read('capacity_cases.json')
    for system,p in cases['planning_basis'].items():
        need(p['max_measured_compressed_pair_bytes']==max(n for (s,c),n in pair.items() if s==system),'MAX_PAIR')
    for r in cases['cases']:
        expected=storage.planning_case(cases['planning_basis'],n_each=r['n_each'],raw_buffers=r['raw_worker_allocation_scenario'],
          compressed_factor=r['compressed_factor_proposed'],metadata_reserve_per_trajectory=r['metadata_reserve_per_trajectory_proposed'])
        need(all(r[k]==v for k,v in expected.items()),'CAPACITY_ARITHMETIC')
        need(r['PR_or_rare_output_upper_bound'] is None and r['chosen'] is False,'PROXY_NOT_BOUND_OR_ADOPTED')
    need(len(cases['cases'])==18,'CASE_COUNT')
    policy=read('storage_policy_DRAFT.json')
    need(policy['adopted'] is False and policy['scientific_execution_authorized'] is False and policy['execution_grant'] is None,'UNAPPROVED')
    need(policy['scientific_hook_installed'] is False and policy['metadata_and_call_ledger_omission'] is False,'OLD_RUNNER_UNCHANGED_NO_OMISSION')
    need(policy['n_E'] is None and policy['AD03']=='DEFERRED' and policy['formal_shot_policy_adopted'] is False,'SCIENTIFIC_PENDING')
    for row in read('inventory.json')['files']:
        b=(HERE/row['file']).read_bytes()
        need(hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes'],'BUNDLE_SHA')
    for p in HERE.iterdir():
        if p.is_file():need(not any(s in p.read_bytes() for s in (b'/' + b'home/',b'/' + b'tmp/',b'gh'+b'p_',b'github'+b'_pat_')),'NO_PRIVATE_ABSOLUTE_PATH_OR_TOKEN')
    for p in [HERE/'README.md',ROOT/'docs/research/track_a_h46_pr_pilot_storage_optimization_20261020.md']:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=target.split('#')[0]
            if target and '://' not in target:need((p.parent/target).exists(),'LOCAL_LINK')
    return dict(status='PASS',metadata_checks=checks,scientific_execution=0,independent_review=False)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
