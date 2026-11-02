"""Validate new H8 documentation only; stdlib, no scientific modules or arithmetic."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '1923fce8f7461513d1bc47b5061e87cd70917e87'
RESULT = 'e6eea5387a6e84d1e39a9adb3b61975acc74e200'
SOURCE = '745a674546f2cb6258526fbfde1d9a00601689f8'
B = 'artifacts/resource_applicability/track_a_h8_evidence_recovery/2026-10-20/'
R = 'artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT), *args])

def load(path):
    return json.loads((ROOT/path).read_text())

def csv_rows(path):
    return list(csv.DictReader(io.StringIO((ROOT/path).read_text())))

def check(local=False):
    manifest=load(B+'public/manifest.json')
    require(manifest['base_commit']==BASE and manifest['result_commit']==RESULT and manifest['science_SOURCE']==SOURCE,'fixed anchors')
    require(manifest['content_origin']=='EXISTING_PUBLIC_FILES_ONLY' and manifest['private_derived_content_in_this_public_packet'] is False,'public boundary')
    manifests=[manifest]
    if local:
        lm=load(B+'local_review/manifest.json')
        require(lm['publication_status']=='LOCAL_ONLY_REQUIRES_EXPLICIT_SHARING_APPROVAL','local publication boundary')
        require(lm['contains_seed_values'] is False and lm['contains_private_raw'] is False,'raw exclusion')
        manifests.append(lm)
    for m in manifests:
        for row in m['files']:
            raw=(ROOT/row['path']).read_bytes()
            require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'packet file hash')
    remote=load(B+'public/public_read_receipt.json')['files']
    require(len(remote)==18,'public file inventory')
    for row in remote:
        raw=git('show',row['commit']+':'+row['path'])
        blob=git('rev-parse',row['commit']+':'+row['path']).decode().strip()
        require(hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes'] and blob==row['git_blob_sha'],'public fixed bytes')
        require(row['local_fixed_blob_matches_remote'] is True and row['read_extent']=='FULL_FILE','full public read')
    original=list(csv.DictReader(io.StringIO(git('show',RESULT+':'+R+'comparisons.csv').decode())))
    rows=csv_rows(B+'public/condition_map.csv')
    require(len(rows)==2,'two independent conditions')
    for row,saved in zip(rows,[r for r in original if r['mode']=='primary']):
        require(all(row[k]==v for k,v in saved.items()),'saved result token changed')
        require(row['result_commit']==RESULT and row['science_SOURCE']==SOURCE and row['public_execution_audit']=='PUBLIC_REPORT_ONLY_DETAILED_PR_RECEIPTS_NOT_IN_FIXED_PUBLIC_SET','public audit scope')
    stop=load(B+'public/MANDATORY_STOP.json')
    require(stop['status']=='MANDATORY_STOP' and stop['new_scientific_calls']==0 and stop['final_evidence_freeze'] is False and stop['detailed_scientific_review_started'] is False and stop['next_stage_authorized'] is False,'STOP boundary')
    if local:
        receipt=load(B+'local_review/recovery_receipt_REVIEW_REQUIRED.json')
        require(receipt['status']=='PASS_EVIDENCE_STRUCTURE_ONLY' and receipt['new_scientific_calls']==0 and receipt['new_statistics']==0 and receipt['raw_cost_reaggregation'] is False,'metadata-only recovery')
        require(len(receipt['scopes'])==2 and all(s['n']==48 and s['axes']==96 and s['driver_exit_code']==0 and s['driver_reaped'] is True for s in receipt['scopes']),'native coverage')
        evidence=csv_rows(B+'local_review/evidence_status.csv')
        require({r['status'] for r in evidence}=={'原本で確認済み','公開資料のみ確認','未取得','確認不能'},'four status categories')
        for row in evidence:
            if row['item'] in ['ORIGINAL_PROFILE_BYTES','SAVED_PR_CIRCUIT_BYTES']:
                require(row['status']=='未取得','missing original incorrectly upgraded')
            if row['item'] in ['MATHEMATICAL_INDEPENDENCE','ORIGINAL_CIRCUIT_FINGERPRINT_REDERIVATION']:
                require(row['status']=='確認不能','unsupported proof')
        localrows=csv_rows(B+'local_review/condition_evidence_map.csv')
        for row,scope in zip(localrows,receipt['scopes']):
            require(row['scope']==scope['scope'] and row['result_sha256']==scope['result_sha256'] and row['native_receipt_sha256']==scope['native_receipt_sha256'],'condition audit link')
        require(receipt['publication_status']=='LOCAL_ONLY_REQUIRES_EXPLICIT_SHARING_APPROVAL','receipt publication boundary')
    # User forbids changing existing artifacts. This is an additive documentation branch.
    changes=git('diff','--name-status',BASE,'--').decode().splitlines()
    require(all(line.startswith('A\t') for line in changes),'existing tracked file changed')
    return dict(status='PASS_DOCUMENT_STRUCTURE_ONLY',public_files_matched=18,conditions=2,local_recovery_checked=local,new_scientific_calls=0,old_tracked_files_changed=0,final_evidence_freeze=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-recovery',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check(args.local_recovery),ensure_ascii=False,indent=2))
