"""Read-only collection. Do not launch scientific workers."""
from pathlib import Path
import json,hashlib
from collections import Counter
import argparse
p=argparse.ArgumentParser(description='Collect task/worker/manifest metadata, without loading costs or providers')
p.add_argument('--projects-root',type=Path,required=True);p.add_argument('--output-private',type=Path,required=True)
a=p.parse_args();W=a.projects_root/'partially-randomized-trotter-worktrees';E=a.projects_root/'h6-handoff-evidence';P=a.output_private
if not P.is_absolute() or P.exists():raise ValueError('NEW_EXCLUSIVE_METADATA_OUTPUT_REQUIRED')
P.mkdir(mode=0o700)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
dirs={};H=W/'h6-hash-reuse-eight-worker-20261012/artifacts/resource_applicability'
for f in ['track_a_h6_exploration_summary_v1/2026-10-12/integrated_exploration_v1.json','track_a_h6_confirmation_summary_v1/2026-10-12/integrated_confirmation_v1.json']:
 for g in read(H/f)['groups']:
  d=Path(g['directory']) if 'directory' in g else Path(g['cost_result_path']).parent
  dirs[str(d)]=('H6',g)
for day in ['20261013','20261014']:
 for d in (E/day).glob('h8-resource-map*/run0*/*/group_*'):
  if (d/'task_binding.json').exists():dirs[str(d)]=('H8',{})
manifest_cache={};manifest_digests={};records=[]
for path,(sys,g) in dirs.items():
 d=Path(path);tb=d/'task_binding.json';cl=d/'worker_claim.json'
 if not tb.exists() or not cl.exists():continue
 t=read(tb);w=read(cl)
 if g.get('file_sha256') and sha(tb)!=g['file_sha256']['task_binding.json']:raise ValueError('HISTORICAL_TASK_HASH')
 manifests=[]
 for parent in list(d.parents)[:3]:
  for name in ['frozen_manifest.json','manifest.json']:
   f=parent/name
   if f.exists():
    if str(f) not in manifest_cache:
     manifest_cache[str(f)]=read(f)
     manifest_digests[str(f)]=hashlib.sha256(json.dumps(manifest_cache[str(f)],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    canonical=manifest_digests[str(f)]
    if t.get('manifest_digest')==canonical:manifests.append(str(f))
 r=dict(system=sys,directory=path,task=t['task'],task_sha=sha(tb),worker_sha=sha(cl),source=w['source_commit'],manifest_matches=manifests)
 records.append(r)
(P/'raw_metadata_PRIVATE.json').write_text(json.dumps(dict(records=records,manifests=manifest_cache)))
print('RECORDS',Counter(r['system'] for r in records),'MANIFESTS',len(manifest_cache),'MATCHES',Counter((r['system'],len(r['manifest_matches'])) for r in records))
for path,m in manifest_cache.items():
 print('MANIFEST_SOURCE',m.get('source_commit'),'SEED_SOURCE',m.get('seed_source_commit'))
