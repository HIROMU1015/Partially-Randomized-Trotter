"""P5R standard-library supervision/provenance utilities; no broad scientific imports."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import os
import resource
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021'
BASE='b022f2ea421649015a4c21ea3c904fe619d0c198'

def require(test,reason):
    if not test: raise ValueError(reason)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def digest(path): return sha(Path(path).read_bytes())
def read(path): return json.loads(Path(path).read_text())
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def journal(path,**value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value.update(pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic=time.monotonic())
    with path.open('a') as stream:
        stream.write(json.dumps(value,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())

def write_gzip(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'OUTPUT_ALREADY_EXISTS')
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_bytes(gzip.compress(raw,mtime=0))
    os.replace(temp,path)

def proc_info(pid):
    p=Path('/proc')/str(pid);out=dict(pid=pid)
    for name in ('status','smaps_rollup'):
        try:
            text=(p/name).read_text()
            out[name]={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in text.splitlines()
                if ':' in line and line.split(':',1)[0] in ('State','VmRSS','VmHWM','VmSize','Threads',
                    'Rss','Pss','Shared_Clean','Shared_Dirty','Private_Clean','Private_Dirty','Swap')}
        except OSError as exc:out[name]=type(exc).__name__
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split();hz=os.sysconf('SC_CLK_TCK')
        out['user_CPU_seconds']=int(stat[11])/hz;out['system_CPU_seconds']=int(stat[12])/hz
    except OSError:pass
    return out

def cgroup():
    rel=Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
    base=Path('/sys/fs/cgroup')/rel;out={}
    for p in (base,base.parent):
        row={}
        for key in ('memory.events','memory.events.local','memory.current','memory.peak','memory.max','cpu.max'):
            try:row[key]=(p/key).read_text()
            except OSError as exc:row[key]=type(exc).__name__
        out[str(p)]=row
    return out

def snapshot():
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic=time.monotonic(),
        self=proc_info(os.getpid()),cgroup=cgroup(),
        meminfo=Path('/proc/meminfo').read_text(),
        pressure={k:Path('/proc/pressure/'+k).read_text() for k in ('cpu','memory','io')})

def verify():
    auth=read(HERE/'authorization_receipt.json')
    require(digest(HERE/'execution_contract.json')==auth['contract_SHA256'],'CONTRACT_CHANGED')
    for name in ('source_manifest.json','input_manifest.json','runtime_manifest.json'):
        require(digest(HERE/name)==auth['manifest_SHA256'][name],'MANIFEST_CHANGED:'+name)
        for row in read(HERE/name)['files']:
            p=Path(row['path']);p=p if p.is_absolute() else ROOT/p
            require(digest(p)==row['sha256'],'FROZEN_IDENTITY:'+str(p))

def new_marker(path,data):
    with Path(path).open('x') as stream:json.dump(data,stream,indent=2)
