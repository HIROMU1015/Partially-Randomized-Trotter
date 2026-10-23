"""Technical metadata closeout only. No science, broad search or old mutations."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE='65085ff7f818fd92af3424b1980f6bd9dbf1c742'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def verify():
    source=read(HERE/'SOURCE.json')
    for item in source['files']:
        assert sha(ROOT/item['path'])==item['sha256']
        assert git('show',source['audit_SOURCE']+':'+item['path'])==(ROOT/item['path']).read_bytes()
    baseline=json.loads(git('show',BASE+':artifacts/validation_manifest.json'))
    current=read(ROOT/'artifacts/validation_manifest.json')
    assert all(current[k]==v for k,v in baseline.items())
    assert set(current)-set(baseline)=={'track_a_h2_runtime_cost_reuse_20261023'}
    changed=git('diff','--name-only','-z',BASE).decode().strip('\0').split('\0')
    allowed={'PROJECT_MAP.md','docs/research/研究概要・現状.md','scripts/README.md',
             'docs/README.md','src/trotterlib/README.md','tests/README.md','artifacts/validation_manifest.json'}
    assert all(p in allowed or p.startswith(('scripts/audit_h2_runtime_cost_reuse.py',
        'scripts/h2_runtime_cost_reuse.py','tests/test_h2_runtime_cost_reuse.py',
        'artifacts/resource_applicability/track_a_h2_runtime_cost_reuse/',
        'docs/research/track_a_h2_runtime_cost_reuse_20261023.md','docs/research/研究ノート/2026-10-23.md')) for p in changed)
    links=[]
    for p in (HERE/'REPORT.md',ROOT/'docs/research/track_a_h2_runtime_cost_reuse_20261023.md'):
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if raw.startswith(('https://','http://','#')):continue
            target=(p.parent/raw.split('#')[0]).resolve()
            assert target.is_file(), (p.name,raw)
            links.append(dict(document=str(p.relative_to(ROOT)),target=raw))
    inventory=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ('file_inventory.json','final_verification.json'):
            raw=p.read_bytes()
            assert not re.search(rb'(?:/home/|/usr/|/tmp/)[A-Za-z0-9_.-]',raw),p.name
            if p.suffix=='.json':json.loads(raw)
            inventory.append(dict(path=p.name,sha256=sha(p),bytes=len(raw)))
    pf=read(HERE/'saved_common_PF_cost_objects_4.json')['rows']
    pr=read(HERE/'saved_PR_cost_objects_134.json')['rows']
    assert len(pr)==134 and len(pf)==4
    assert {(x['requested_candidate'],x['RZ']) for x in pf}=={
        ('B1_p3_q4_2nd',1655),('B1_p3_q2_4th',2463)}
    coverage=read(HERE/'same_runtime_comparability.json')
    assert coverage['target_native_binding']['same_preparation_hash'] is True
    assert coverage['COMPARE']=='BLOCKED' and coverage['original_B_cross_runtime_condition_unchanged']=='UNRESOLVED'
    stop=read(HERE/'STOP.json')
    assert all(x==0 for x in stop['actual_scientific_calls'].values()) and stop['live_grants']==0
    tests=read(HERE/'test_results.json')
    assert tests['new_artificial_tests']==34 and tests['existing_alternative_artificial_tests']==48
    preservation=read(HERE/'preservation_audit.json')
    assert preservation['changes']==0 and preservation['protected_files']==16824
    assert preservation['before_sha256']==preservation['after_sha256']
    probe=read(HERE/'preserved_four_axis_probe_DRAFT.json')
    assert len(probe['axes'])==4 and probe['proof_of_134_axes'] is False
    assert probe['grant'] is None and probe['production_backend_connected'] is False
    result=dict(status='PASS_TECHNICAL_CLOSEOUT_MANDATORY_STOP',audit_SOURCE=source['audit_SOURCE'],
        files_checked=len(inventory),local_links_checked=len(links),
        old_manifest_entries_preserved=len(baseline),old_scientific_namespaces_unchanged=True,
        private_absolute_paths_in_new_artifacts=0,scientific_calls=0,COMPARE='BLOCKED',
        alternative_adopted=False,push=False,links=links)
    for name,data in [('file_inventory.json',dict(files=inventory,self_and_final_verification_excluded=True)),
                      ('final_verification.json',result)]:
        with (HERE/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='links'}))


if __name__=='__main__':verify()
