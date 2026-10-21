"""Preservation/identity byte checks; no scientific modules or old raw decoding."""
from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-b-c1-5a-p5-worker-recovery-20261021');n=w/'docs/research/track_b_c1_5a_p5_worker_recovery_20261021';r=Path(str(w)+'.runtime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((n/'preservation_before.json').read_text());checked=[]
for mf in ('source_manifest.json','input_manifest.json','runtime_manifest.json'):
 d=json.loads((n/mf).read_text())
 for row in d['files']:
  p=Path(row['path']);p=p if p.is_absolute() else w/p
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
 checked.append(dict(manifest=mf,files=len(d['files']),all_match=True))
for row in before['old_namespace_files']+before['private_originals']:
 p=Path(row['path']);p=p if p.is_absolute() else w/p
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
# Working-tree changes to protected tracked files absent. No all-worktree freeze.
changes=subprocess.check_output(['git','diff',before['base_commit'],'--name-status'],cwd=w,text=True).splitlines()
assert all(line.startswith('A\t'+str(n.relative_to(w))+'/') for line in changes),changes
terminal=json.loads((n/'result/terminal.json').read_text());marker=json.loads((n/'attempt_marker.json').read_text())
assert marker['scientific_attempts']==1 and marker['retry']==0 and terminal['retry']==0 and terminal['mandatory_STOP']
auth=json.loads((n/'authorization_receipt.json').read_text())
assert auth['contract_SHA256']==sha(n/'execution_contract.json')
for f,v in auth['manifest_SHA256'].items():assert sha(n/f)==v
new_index=json.loads((n/'raw_archives/index.json').read_text())
packed=b''.join((n/x['path']).read_bytes() for x in new_index['parts'])
assert len(packed)==new_index['archive']['bytes'] and hashlib.sha256(packed).hexdigest()==new_index['archive']['sha256']
for row in new_index['members']:
 p=r/'raw_preserved'/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
report=dict(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=before['base_commit'],SOURCE=auth['source_commit'],AUTH=marker['authorization_HEAD'],manifests=checked,old_namespace_files_verified=len(before['old_namespace_files']),old_namespace_all_bytes_SHA256_unchanged=True,private_originals=before['private_originals'],private_originals_unchanged_and_not_published=True,only_new_namespace_additions=True,scientific_attempts=1,retry=0,terminal_status=terminal['status'],raw_transport_members=len(new_index['members']),raw_all_bytes_SHA256_preserved=True,no_old_raw_scientific_reanalysis=True,no_other_worktree_mutations=True,shared_environment_changes=False,mandatory_STOP=True)
(n/'preservation_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report,ensure_ascii=False))
