"""After audited success, index/preserve/commit authorized results and STOP."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FINAL='artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22'
META='artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22'
H6='artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    with Path(p).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def cmd(args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def run(common,metadata):
    common=Path(common);metadata=Path(metadata)
    save(common/'LOCAL_COMMIT_DRIVER_ENTERED.json',dict(status='WAITING_FOR_FINAL_AUDIT',scientific_calls=0,GitHub_push=False))
    while not (common/'FINAL_RESULT_DRIVER_COMPLETE.json').exists():
        if any((common/n).exists() for n in ('SUCCESSOR_DRIVER_STOP.json','FINAL_RESULT_DRIVER_STOP.json')):
            raise RuntimeError('PRIOR_FAILURE_NO_SUCCESS_COMMIT_OR_RETRY')
        time.sleep(2)
    out=ROOT/FINAL
    assert read(out/'summary.json')['complete']==588
    assert read(out/'independent_saved_result_checks.json')['status']=='PASS'
    # Preflight explicit document targets before adding latest-result indexes.
    for rel in ('docs/research/track_a_h246_pf_common_final_20261022.md',META+'/README.md',H6+'/README.md',FINAL+'/README.md'):
        p=ROOT/rel
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):assert (p.parent/target).exists(),target
    subprocess.run([sys.executable,'-B','scripts/index_h246_pf_common_final.py'],cwd=ROOT,check=True)
    # All old numerical results and failed-stage receipts stay immutable.
    for entry in read(out/'prior_results_preserved.json')['files']:
        assert sha(ROOT/entry['path'])==entry['sha256']
    old_private=metadata.parent/'METADATA63_run02'
    old_inventory=read(ROOT/'artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/original_control_inventory.json')
    for rel,digest in old_inventory.items():assert sha(old_private/rel)==digest
    base=json.loads(cmd(['git','show','0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c:artifacts/validation_manifest.json']))
    manifest=read(ROOT/'artifacts/validation_manifest.json');assert all(manifest[k]==v for k,v in base.items())
    save(out/'ending_preservation_checks.json',dict(status='PASS',old_results_byte_identical=True,old_failed_grant_ledger_marker_STOP_unchanged=True,
        old_manifest_entries_unchanged=True,all_result_links_exist=True,H2_H4_recomputed=False,new_PR_sampling=0,GitHub_push=False))
    save(out/'MANDATORY_STOP.json',dict(status='MANDATORY_STOP',reason='Approved H6 completion and H246 fixed common-grid comparison finished',new_scientific_stage_started=False,
        result_summary_sha256=sha(out/'summary.json'),fixed_grid_checks_sha256=sha(out/'independent_saved_result_checks.json'),GitHub_push=False))
    # Metadata bundle has its own earlier immutable inventory and STOP.
    for directory in (ROOT/H6,out):
        save(directory/'final_file_inventory.json',dict(excludes_self=True,files=[dict(path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(directory.rglob('*')) if p.is_file()]))
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    paths=[META,H6,FINAL,'PROJECT_MAP.md','VALIDATION_STATUS.md','artifacts/validation_manifest.json','docs/README.md',
        'docs/research/研究概要・現状.md','docs/research/研究ノート/2026-10-22_h6_pf_metadata_STOP.md',
        'docs/research/track_a_h246_pf_common_final_20261022.md','scripts/README.md','src/trotterlib/README.md']
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Complete H6 common PF acquisition and audited H246 PR/PF comparison'],cwd=ROOT,check=True)
    commit=cmd(['git','rev-parse','HEAD'])
    receipt=dict(status='LOCAL_COMMIT_PASS_MANDATORY_STOP',local_result_commit=commit,branch=cmd(['git','branch','--show-current']),
                 new_scientific_stage_started=False,GitHub_push=False,scientific_calls_in_this_driver=0)
    save(common/'MANDATORY_STOP.json',receipt)
    save(common/'LOCAL_COMMIT_DRIVER_COMPLETE.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--common',required=True);parser.add_argument('--metadata',required=True);a=parser.parse_args()
    try:run(a.common,a.metadata)
    except BaseException as exc:
        p=Path(a.common)/'LOCAL_COMMIT_DRIVER_STOP.json'
        if not p.exists():save(p,dict(status='STOP',reason=str(exc),scientific_retry=False,GitHub_push=False))
        raise
