"""Fixed input hashes/old evidence only. No QPY decode or state evaluation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
PREP=ROOT/'artifacts/non_df_h4_central_rho_preparation/2026-10-24'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    files=dict(read(PREP/'inputs.json')['files'])
    for p,h in files.items():assert sha(ROOT/p)==h,p
    for name in ('targets.json','guard_inputs_quoted.json','error_ownership.json','contracts_DRAFT.json','verification.json'):
        p=PREP/name;files[str(p.relative_to(ROOT))]=sha(p)
    prior_seal=ROOT/'artifacts/non_df_h4_a_axes/2026-10-24/seal.json'
    prior=read(prior_seal);runtime=ROOT/prior['runtime_manifest']
    assert sha(runtime)==prior['runtime_manifest_sha256']
    files[str(runtime.relative_to(ROOT))]=sha(runtime)
    files[str(prior_seal.relative_to(ROOT))]=sha(prior_seal)
    save(BASE/'inputs.json',dict(kind='R0_FIXED_INPUTS_V1',files=files,selected=['PF0_cosine','PR0_cosine'],state_actions_performed=0))
    preservation=dict(read(PREP/'preservation_before.json'))
    for p,h in preservation.items():assert sha(ROOT/p)==h,p
    paths=subprocess.check_output(['git','ls-files','--',str(PREP.relative_to(ROOT))],cwd=ROOT,text=True).splitlines()
    preservation.update({p:sha(ROOT/p) for p in paths});save(BASE/'preservation_before.json',preservation)
    manifest=read(ROOT/'artifacts/validation_manifest.json')
    save(BASE/'prior_manifest_rows.json',dict(row_count=len(manifest),row_sha256={k:digest(v) for k,v in manifest.items()},base_commit='2734b97c2dd5f017e93cd81619ff52a6a0c6f16b'))
    shutil.copyfile('/home/AbeHiromu/.codex/attachments/ab7aebea-b37d-4be5-b8bd-089becf283ef/貼り付けたテキスト.txt',BASE/'approval.txt')
    shutil.copyfile('/tmp/h4_r0_artificial01.log',BASE/'artificial10_PASS.log')
    print(json.dumps(dict(inputs=len(files),preserved=len(preservation),prior_manifest_rows=len(manifest),state_actions=0)))
