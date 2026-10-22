"""Static Git blob acquisition only. Never imports project/scientific modules."""
import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/AbeHiromu/projects/partially-randomized-trotter')
MANIFEST=ROOT/'acquisition_manifest.json'

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args])

def get(commit,path):
    commit=git('rev-parse',commit+'^{commit}').decode().strip()
    records=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    data=git('show',commit+':'+path)
    blob=git('rev-parse',commit+':'+path).decode().strip()
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob
    target=ROOT/'sources'/commit/path
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert target.read_bytes()==data
    else:
        target.write_bytes(data)
    record=dict(commit=commit,path=path,git_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),local_path=str(target.relative_to(ROOT)),retrieval='LOCAL_FIXED_GIT_BLOB',url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+commit+'/'+path)
    if not any(x['commit']==commit and x['path']==path for x in records):
        records.append(record)
        MANIFEST.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    return target

if __name__=='__main__':
    import sys
    for path in sys.argv[2:]:
        t=get(sys.argv[1],path)
        print(str(t.relative_to(ROOT)),t.stat().st_size)
