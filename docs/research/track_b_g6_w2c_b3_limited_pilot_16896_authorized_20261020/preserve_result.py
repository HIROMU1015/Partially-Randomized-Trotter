"""Post-STOP byte inventory/archive and historical preservation checks only."""
from pathlib import Path
from hashlib import sha256
import json,tarfile,subprocess,sys

def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main(packet):
    p=Path(packet).resolve();r=p.parents[2];m=json.loads((p/'approved_manifest.json').read_text());root=Path(m['state_directory']);out=root/'output'
    assert (out/'STOP_marker.json').is_file() or (out/'STOP_failure_marker.json').is_file()
    inventory=[]
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():inventory.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest(path)))
    archive=p/'raw_scientific.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for v in inventory:t.add(root/v['path'],arcname=v['path'])
    with tarfile.open(archive,'r:gz') as t:
        members={v.name:v for v in t.getmembers() if v.isfile()};assert set(members)=={v['path'] for v in inventory}
        for v in inventory:assert sha256(t.extractfile(members[v['path']]).read()).hexdigest()==v['sha256']
    save(p/'raw_inventory.json',dict(state_root=str(root),files=inventory,archive=dict(path=archive.name,bytes=archive.stat().st_size,sha256=digest(archive)),archive_roundtrip_SHA256=True,local_original_preserved=True,
        scope='this adopted one-shot; no retry/resume or source modification',
        archive_access='repository dedicated branch when publication receipt confirms; original state root local owned server workspace'))
    src=json.loads((p/'source_inventory.json').read_text());pins=dict(src['source_pins']);pins.update(src['input_pins'])
    for path,h in pins.items():assert digest(r/path)==h
    source_archive=p/'measured_source_and_inputs.tar.gz'
    with tarfile.open(source_archive,'w:gz') as t:
        for path in sorted(pins):t.add(r/path,arcname=path)
    commit=m['measured_SOURCE'];tree=subprocess.check_output(['git','rev-parse',commit+'^{tree}'],cwd=r,text=True).strip()
    save(p/'measured_source_retention.json',dict(commit=commit,tree=tree,source_pins=len(src['source_pins']),input_pins=len(src['input_pins']),all_SHA256_match=True,
        source_archive=dict(path=source_archive.name,bytes=source_archive.stat().st_size,sha256=digest(source_archive)),
        commit_raw=subprocess.check_output(['git','cat-file','commit',commit],cwd=r,text=True)))
    old=r/'docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020';oi=json.loads((old/'raw_inventory.json').read_text());before=json.loads((p/'old512_raw_before.json').read_text());oldroot=Path(before['old_state_root'])
    for v in oi['files']:assert digest(oldroot/v['path'])==v['sha256']
    assert digest(old/oi['archive']['path'])==oi['archive']['sha256']
    oldpins=json.loads((p/'preservation_before.json').read_text())['pins']
    for path,h in oldpins.items():assert digest(r/path)==h
    save(p/'preservation_after.json',dict(status='PASS',old512_raw_files=len(oi['files']),old512_raw_archive_SHA256=oi['archive']['sha256'],old_SOURCE_input_pins=len(oldpins),all_unchanged=True,
        other_worktree_HEAD_status='observation only; not a freeze or stop requirement',new_scope_retry=0,new_scope_resume=False))
    print(json.dumps(dict(raw_files=len(inventory),raw_bytes=sum(v['bytes'] for v in inventory),archive_bytes=archive.stat().st_size,archive_SHA256=digest(archive),SOURCE=commit)))
if __name__=='__main__':main(sys.argv[1])
