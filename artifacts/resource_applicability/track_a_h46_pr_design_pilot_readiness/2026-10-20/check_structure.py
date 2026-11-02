"""Metadata/Git-byte checks only. No private originals or scientific imports."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import h46_pr_design_pilot as d

def read(name):return json.loads((HERE/name).read_bytes())
def check():
    count=0
    def need(value,message):
        nonlocal count
        if not value:raise ValueError(message)
        count+=1
    summary=read('summary.json');source=summary['SOURCE']
    need(summary['base_commit']==d.BASE,'BASE')
    p=json.loads((ROOT/'references/h46_pr_design_pilot_proposal_v1.json').read_bytes())
    need(p==d.proposal(),'PROPOSAL_REPRODUCED_FROM_METADATA')
    with (HERE/'pilot_groups_24.csv').open() as f:rows=list(csv.DictReader(f))
    need(len(rows)==24 and len({r['candidate_id'] for r in rows})==24,'EXACT_24')
    need({r['candidate_id'] for r in rows}=={r['id'] for r in d.groups()},'FIXED_IDS')
    need(all(r['n_P_adopted']==r['n_E']==r['unit_cost_variance']==r['population_tail_bound']=='' for r in rows),'UNADOPTED_NULL_NOT_ZERO')
    for item in read('source_closure.json')['files']:
        b=(ROOT/item['path']).read_bytes()
        need(len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],'SOURCE_SHA')
        need(b==subprocess.check_output(['git','show',source+':'+item['path']],cwd=ROOT),'SOURCE_COMMIT_BYTES')
    for stage in d.STAGES:
        cfg=read(stage+'_INVALID_DRAFT.json')
        need(cfg['authorized'] is False and cfg['grant']['valid'] is False,'DISABLED_GRANT')
        need(cfg['cohort']==d.COHORT and cfg['SOURCE']==source and cfg['n_P_adopted'] is None,'COHORT_SOURCE_N')
        need(cfg['scientific_call_scope'] is None and cfg['resources']['full_output_capacity_adopted'] is False,'PENDING_CALL_AND_SCRATCH')
    need(all(v==0 for v in summary['scientific_execution'].values()),'NO_REAL_SCIENCE')
    need(summary['formal_shot_policy_adopted'] is False and summary['AD03']=='DEFERRED','AD03_SHOTS')
    need(read('reuse_provenance.json')['native_file_fork_exact_documented_transform_verified'] is True,'SAME_LAW_FORK')
    for item in read('inventory.json')['files']:
        b=(HERE/item['file']).read_bytes()
        need(hashlib.sha256(b).hexdigest()==item['sha256'] and len(b)==item['bytes'],'BUNDLE_SHA')
    for path in HERE.iterdir():
        if not path.is_file():continue
        b=path.read_bytes()
        need(not any(t in b for t in (b'/' + b'home/',b'/' + b'tmp/',b'file'+b'://',b'github'+b'_pat_',b'gh'+b'p_')),'NO_INTERNAL_PATH_OR_TOKEN')
    for path in [HERE/'README.md',ROOT/'docs/research/track_a_h46_pr_design_pilot_readiness_20261020.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            target=target.split('#')[0]
            if target and '://' not in target:need((path.parent/target).exists(),'DOCUMENT_LINK')
    return {'status':'PASS','metadata_checks':count,'SOURCE':source,'real_scientific_execution':0,'independent_review':False}

if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
