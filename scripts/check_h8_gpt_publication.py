"""Validate only the additive public H8 documentation; no network or science."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REL='artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/'
BASE='9f33ad0b11b4febe48b802000c38c5a0e3fd267c'
PRIVATE_COMMITS=['253356942845dddf05c2bb6f4582b026e0bce942','363199c83b4cebde3d144f173e0e0aae67f7b2df']
ZIP_SHA='901e416d0588fe2e011e32705179e628a41d76f9347ea8690def46472ec11bba'

def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    b=ROOT/REL;manifest=json.loads((b/'publication_manifest.json').read_text())
    require(manifest['base_public_commit']==BASE and manifest['repository_publication_authorized'] is True,'PUBLICATION_SCOPE')
    spec=importlib.util.spec_from_file_location('h8_offline_check',b/'candidate/verify_packet.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    packet,data=m.validate(b/'candidate');mutations=m.self_test(data)
    names={r['path'] for r in manifest['files']}
    expected_new={str(p.relative_to(ROOT)) for p in b.rglob('*') if p.is_file()}|set(manifest['extra_files_outside_artifact_directory'])
    require(names|{REL+'publication_manifest.json'}==expected_new,'EXACT_PUBLICATION_FILE_INVENTORY')
    for row in manifest['files']:
        path=Path(row['path']);require(not path.is_absolute() and '..' not in path.parts and 'local_only' not in path.parts,'NO_NONSHARE_FILES')
        raw=(ROOT/path).read_bytes();require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'PUBLICATION_FILE_HASH')
        if path.suffix!='.zip':
            text=raw.decode();m.scan_text(text)
            if path.suffix=='.json':m.scan_json(json.loads(text))
    m.scan_text((b/'publication_manifest.json').read_text());m.scan_json(manifest)
    zpath=b/'H8_GPT_SHARE_PACKET_20261020.zip';require(hashlib.sha256(zpath.read_bytes()).hexdigest()==ZIP_SHA,'ORIGINAL_AUDITED_ZIP_UNCHANGED')
    with zipfile.ZipFile(zpath) as z:
        expected={'H8_GPT_SHARE_PACKET_20261020/'+str(p.relative_to(b/'candidate')) for p in (b/'candidate').rglob('*') if p.is_file()}
        require(set(z.namelist())==expected and len(z.namelist())==18 and z.testzip() is None,'ZIP_ALLOWLIST_CRC')
        for name in z.namelist():require(z.read(name)==(b/'candidate'/name.split('/',1)[1]).read_bytes(),'ZIP_EXACT_CANDIDATE_BYTES')
    g=['git','--no-optional-locks','-C',str(ROOT)]
    for private in PRIVATE_COMMITS:
        exists=subprocess.run(g+['cat-file','-e',private+'^{commit}'],capture_output=True).returncode==0
        if exists:require(subprocess.run(g+['merge-base','--is-ancestor',private,'HEAD'],capture_output=True).returncode==1,'PRIVATE_COMMIT_IN_ANCESTRY')
    changes=subprocess.check_output(g+['diff','--name-status',BASE,'--'],text=True).splitlines()
    require(all(line.startswith('A\t') and line.split('\t',1)[1] in names|{REL+'publication_manifest.json'} for line in changes),'ONLY_ADDITIVE_WHITELISTED_FILES')
    print(json.dumps(dict(status='PASS_PUBLICATION_STRUCTURE_AND_REDACTION_ONLY',publication_files=len(names)+1,packet=packet,mutation_checks=len(mutations),private_index_in_tree=False,private_commits_in_ancestry=False,existing_tracked_files_changed=0,new_scientific_calls=0,scientific_review_started=False,final_evidence_freeze=False),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
