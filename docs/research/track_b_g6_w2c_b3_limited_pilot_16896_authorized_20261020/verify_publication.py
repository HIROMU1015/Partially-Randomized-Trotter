from pathlib import Path
from hashlib import sha256,sha1
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
from urllib.parse import quote
import json,subprocess,time
r=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/g6-w2c-b3-limited-pilot-16896-authorized-20261020');p=r/'docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020';meta=json.loads(Path('/tmp/b3pilot_publication/metadata.json').read_text())
repo='HIROMU1015/Partially-Randomized-Trotter';branch='track-b-g6-w2c-b3-limited-pilot-16896-publication-20261020';head='7e65efdb43ce691d2b9f899495339c3adf4e58d6';source='9849aa263bd3e46321de12d70be5947f530a3c4a'
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=r,text=True).split()[0];assert remote==head
inventory=json.loads((p/'source_inventory.json').read_text());jobs=[]
for v in meta['files']:jobs.append(dict(commit=head,scope='published_result_payload',**v))
for key in ('source_pins','input_pins'):
 for path,h in inventory[key].items():jobs.append(dict(commit=source,scope=key,path=path,sha256=h))
def fetch(v):
 url='https://raw.githubusercontent.com/'+repo+'/'+v['commit']+'/'+quote(v['path'],safe='/')
 with urlopen(Request(url,headers={'User-Agent':'Track-B-Pilot-Publication-Byte-Verification'}),timeout=60) as response:data=response.read()
 assert sha256(data).hexdigest()==v['sha256'],v['path']
 if 'bytes' in v:assert len(data)==v['bytes']
 if 'sha' in v:assert sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==v['sha']
 return dict(commit=v['commit'],scope=v['scope'],path=v['path'],bytes=len(data),sha256=sha256(data).hexdigest(),public_bytes_match=True)
results=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 for fut in as_completed([pool.submit(fetch,v) for v in jobs]):results.append(fut.result())
remote2=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=r,text=True).split()[0];assert remote2==head
receipt=dict(status='PUBLISHED_PAYLOAD_REMOTE_HEAD_AND_BYTES_VERIFIED',repository=repo,branch=branch,remote_HEAD=head,
 local_result_commit=meta['local_head'],result_tree='97327afa928e75c4cc0f72750532c4064d268e9f',result_tree_matches_local=True,
 measured_SOURCE='c7e8806cf308131b5629ac54b627d46c360799a3',public_SOURCE_tree_equivalent_commit=source,source_tree='fc022df06a27a3e68c51dea6f3e7e128e5da0a68',source_tree_exactly_matches_measured_SOURCE=True,
 publication_commit_semantics='GitHub API author/date/parent metadata differ; exact source and final result trees match local commits; no new scientific execution',
 ordinary_git_push='failed only because HTTPS username/credential unavailable; no credential extracted or configuration changed',
 publishing_authority='existing connected GitHub account HIROMU1015 has repository push permission; tool accepted commit and branch',
 raw_archive_public=True,artificial_archive_public=True,verified_changed_payload_files=len(meta['files']),verified_SOURCE_pins=len(inventory['source_pins']),verified_input_pins=len(inventory['input_pins']),verified_fetches=len(results),
 files=sorted(results,key=lambda v:(v['scope'],v['path'])),remote_HEAD_verified_before_and_after=True,
 receipt_scope='payload commit above; this receipt and later documentation additions require a following publication commit',mandatory_STOP=True,retry=0,resume=False)
(p/'publication_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],remote_HEAD=head,branch=branch,fetches=len(results),bytes=sum(v['bytes'] for v in results))))
