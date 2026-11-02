"""Check original commit header identities; public mirrors carry identical trees."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
for c in json.loads((root/'source_commit_proofs.json').read_text())['commits']:
    raw=c['raw_commit'].encode()
    actual=hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if actual!=c['original']:raise ValueError('Original commit identity mismatch')
    print(actual,c['mirror'],c['tree'])
