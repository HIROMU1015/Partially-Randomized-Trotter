"""Lossless shared immutable records with legacy logical byte identities.

Stdlib only. Blobs are compact physical JSON; all values, ordered keys, phases,
raw matrix bytes and names survive decoding. Physical writes never duplicate a
large published generation. Readers verify the complete graph before returning.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from ..df_correction_comparison import storage as legacy

MAGIC='DF_ET_LOSSLESS_RECORD_V5'
REVIEW_BYTES=512*1024
CHUNK_BYTES=256*1024
OUTPUT_BYTES=64*1024*1024-256*1024  # exactly the existing active output guard
SHARED_FIELDS={'recursive_payload','recursive_definition_index','Pauli_coefficients','all_B_event_costs','unknown_multiplicities'}


class StorageBudgetExceeded(RuntimeError):pass


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _parts(data, *, compact=False, ensure_ascii=True):
    options={'default':str,'ensure_ascii':ensure_ascii}
    if compact:options['separators']=(',',':')
    else:options['indent']=2
    for fragment in json.JSONEncoder(**options).iterencode(data):
        for i in range(0,len(fragment),CHUNK_BYTES//4):
            yield fragment[i:i+CHUNK_BYTES//4].encode('utf-8')
    yield b'\n'


def _identity(parts):
    h=hashlib.sha256();n=0
    for b in parts:h.update(b);n+=len(b)
    return n,h.hexdigest()


def _safe(root,path):
    root=Path(root).resolve();p=(root/path).resolve()
    if not p.is_relative_to(root) or p==root:
        raise ValueError('Storage reference escapes result namespace')
    return p


def _room(root,extra):
    total=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    if total+extra>OUTPUT_BYTES:
        raise StorageBudgetExceeded('Fixed active output budget; no cap increase or retry')


def _write(root,path,raw):
    if len(raw)>REVIEW_BYTES:raise ValueError('Review file exceeds 512 KiB')
    _room(root,len(raw));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


@contextmanager
def _locked(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'_records.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def _blob_parts(root,ref):
    p=_safe(root,ref['path']);digest=ref['sha256'];size=ref['bytes']
    if not re.fullmatch('[0-9a-f]{64}',digest) or type(size) is not int or size<0:
        raise ValueError('Invalid immutable blob identity')
    if ref['storage']=='raw':
        raw=p.read_bytes()
        if len(raw)>REVIEW_BYTES:raise ValueError('Oversized raw review blob')
        parts=[raw]
    elif ref['storage']=='chunks':
        index_raw=p.read_bytes()
        if len(index_raw)>REVIEW_BYTES or _sha(index_raw)!=ref['index_sha256']:
            raise ValueError('Blob index identity failed')
        index=json.loads(index_raw);parts=[]
        if index['bytes']!=size or index['sha256']!=digest:raise ValueError('Blob index binding failed')
        for i,row in enumerate(index['chunks']):
            raw=_safe(root,row['path']).read_bytes()
            if row['order']!=i or not 0<len(raw)<=CHUNK_BYTES or len(raw)!=row['bytes'] or _sha(raw)!=row['sha256']:
                raise ValueError('Blob chunk identity/order failed')
            raw.decode('utf-8');parts.append(row)
    else:raise ValueError('Unknown blob storage')
    whole=hashlib.sha256();count=0
    for item in parts:
        raw=_safe(root,item['path']).read_bytes() if isinstance(item,dict) else item
        raw.decode('utf-8');whole.update(raw);count+=len(raw);yield raw
    if count!=size or whole.hexdigest()!=digest:raise ValueError('Immutable blob bytes/SHA256 failed')


def _verify_blob(root,ref):
    for _ in _blob_parts(root,ref):pass


def _read_blob(root,ref):
    return json.loads(b''.join(_blob_parts(root,ref)))


def _blob(root,value,ensure_ascii):
    size,digest=_identity(_parts(value,compact=True,ensure_ascii=ensure_ascii))
    rawpath=Path('_objects')/(digest+'.json');indexpath=Path('_objects')/(digest+'.index.json')
    if size<=REVIEW_BYTES:
        ref={'path':str(rawpath),'storage':'raw','bytes':size,'sha256':digest}
        path=root/rawpath
        if path.exists():_verify_blob(root,ref);return ref
        raw=b''.join(_parts(value,compact=True,ensure_ascii=ensure_ascii));_write(root,path,raw)
    else:
        path=root/indexpath
        if path.exists():
            raw=path.read_bytes();ref={'path':str(indexpath),'storage':'chunks','bytes':size,'sha256':digest,'index_sha256':_sha(raw)}
            _verify_blob(root,ref);return ref
        directory=root/'_objects'/digest
        if directory.exists():raise ValueError('Incomplete prior blob cannot be retried')
        directory.mkdir(parents=True);rows=[];buffer=bytearray();whole=hashlib.sha256();written=0
        def flush():
            nonlocal written
            if not buffer:return
            raw=bytes(buffer);raw.decode('utf-8');p=directory/('chunk-%05d.txt'%len(rows))
            _write(root,p,raw);rows.append({'path':str(p.relative_to(root)),'order':len(rows),'bytes':len(raw),'sha256':_sha(raw)})
            whole.update(raw);written+=len(raw);buffer.clear()
        for part in _parts(value,compact=True,ensure_ascii=ensure_ascii):
            if len(buffer)+len(part)>CHUNK_BYTES:flush()
            buffer.extend(part)
        flush()
        if written!=size or whole.hexdigest()!=digest:raise ValueError('Blob serialization changed during publication')
        raw=(json.dumps({'bytes':size,'sha256':digest,'chunks':rows},separators=(',',':'))+'\n').encode()
        _write(root,path,raw)
        ref={'path':str(indexpath),'storage':'chunks','bytes':size,'sha256':digest,'index_sha256':_sha(raw)}
    _verify_blob(root,ref)  # verify complete disk publication, not only in-memory data
    return ref


def _key(key):
    # Match JSONEncoder's conversion of allowed nonstring dictionary keys.
    return next(iter(json.loads(json.dumps({key:0}))))


def _encode(root,value,ensure_ascii,key=None):
    if key in SHARED_FIELDS and value is not None:
        return {'t':'b','ref':_blob(root,value,ensure_ascii)}
    if key=='input_arrays' and isinstance(value,dict):
        return {'t':'d','v':[[name,{'t':'b','ref':_blob(root,item,ensure_ascii)}] for name,item in value.items()]}
    if key=='native' and isinstance(value,(list,tuple)):
        return {'t':'l','v':[{'t':'b','ref':_blob(root,item,ensure_ascii)} for item in value]}
    if isinstance(value,str) and len(value)>CHUNK_BYTES//8:
        return {'t':'b','ref':_blob(root,value,ensure_ascii)}
    if isinstance(value,dict):return {'t':'d','v':[[_key(k),_encode(root,v,ensure_ascii,_key(k))] for k,v in value.items()]}
    if isinstance(value,(list,tuple)):return {'t':'l','v':[_encode(root,v,ensure_ascii) for v in value]}
    return {'t':'s','v':json.loads(json.dumps(value,default=str,ensure_ascii=ensure_ascii))}


def _decode(root,node):
    kind=node['t']
    if kind=='b':return _read_blob(root,node['ref'])
    if kind=='s':return node['v']
    if kind=='l':return [_decode(root,item) for item in node['v']]
    if kind=='d':
        keys=[k for k,_ in node['v']]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate typed keys')
        return {k:_decode(root,item) for k,item in node['v']}
    raise ValueError('Unknown record node type')


def _snapshot(root,ref,logical_path):
    raw=_safe(root,ref['path']).read_bytes()
    if len(raw)>REVIEW_BYTES or len(raw)!=ref['bytes'] or _sha(raw)!=ref['sha256']:
        raise ValueError('Snapshot identity failed')
    snapshot=json.loads(raw)
    if snapshot['kind']!=MAGIC or snapshot['logical_path']!=logical_path:
        raise ValueError('Snapshot logical binding failed')
    return snapshot


def _root(path):
    raw=Path(path).read_bytes()
    if len(raw)>REVIEW_BYTES:raise ValueError('Root pointer exceeds review bound')
    record=json.loads(raw)
    if set(record)!={'kind','logical_path','snapshot'} or record['kind']!=MAGIC or record['logical_path']!=Path(path).name:
        raise ValueError('Record root identity failed')
    return _snapshot(Path(path).parent,record['snapshot'],Path(path).name)


def _verify_history(path,current):
    seen=set();blobs=set();snapshot=current
    def verify_nodes(node):
        if node['t']=='b':
            key=json.dumps(node['ref'],sort_keys=True)
            if key not in blobs:_verify_blob(Path(path).parent,node['ref']);blobs.add(key)
        elif node['t']=='d':
            for _,item in node['v']:verify_nodes(item)
        elif node['t']=='l':
            for item in node['v']:verify_nodes(item)
        elif node['t']!='s':raise ValueError('Unknown history node')
    while True:
        verify_nodes(snapshot['tree'])
        previous=snapshot.get('previous_snapshot')
        if previous is None:return
        if previous['sha256'] in seen:raise ValueError('Snapshot history cycle')
        seen.add(previous['sha256'])
        snapshot=_snapshot(Path(path).parent,previous,Path(path).name)


def dump_json(path,data,*,replace_existing=False,ensure_ascii=True):
    path=Path(path);root=path.parent
    # Refuse old artifacts before even creating a lock in their directory.
    if legacy.index_path(path).exists():raise FileExistsError('Legacy index cannot be replaced')
    if path.exists():_root(path)
    with _locked(root):
        previous=None
        if path.exists():
            if not replace_existing:raise FileExistsError(path)
            _root(path)  # refuse any legacy artifact before attempting an update
            load_json(path)  # validate closed prior history, not only pointer bytes
            previous=json.loads(path.read_bytes())['snapshot']
        elif legacy.index_path(path).exists():raise FileExistsError('Legacy index cannot be replaced')
        tree=_encode(root,data,ensure_ascii)
        # Keep exact legacy logical bytes as well as closed physical graph.
        size,digest=_identity(_parts(data,ensure_ascii=ensure_ascii))
        snapshot={'kind':MAGIC,'logical_path':path.name,'logical_bytes':size,'logical_sha256':digest,
                  'serialization':'json default=str indent=2 plus newline','ensure_ascii':ensure_ascii,
                  'previous_snapshot':previous,'tree':tree}
        raw=(json.dumps(snapshot,separators=(',',':'),ensure_ascii=True)+'\n').encode()
        relative=Path('_snapshots')/(_sha(raw)+'.json');p=root/relative
        if p.exists():
            if p.read_bytes()!=raw:raise ValueError('Snapshot content hash collision/mutation')
        else:_write(root,p,raw)
        pointer={'kind':MAGIC,'logical_path':path.name,'snapshot':{'path':str(relative),'bytes':len(raw),'sha256':_sha(raw)}}
        pointer_raw=(json.dumps(pointer,separators=(',',':'))+'\n').encode()
        temporary=root/(path.name+'.pending-'+uuid.uuid4().hex)
        _write(root,temporary,pointer_raw);os.replace(temporary,path)
        return {'logical_path':path.name,'original_bytes':size,'original_sha256':digest,'storage':'shared_immutable_v5','snapshot':pointer['snapshot']}


def load_json(path):
    path=Path(path)
    if not path.exists():return legacy.load_json(path)
    record=json.loads(path.read_bytes())
    if not isinstance(record,dict) or record.get('kind')!=MAGIC:return legacy.load_json(path)
    snapshot=_root(path);_verify_history(path,snapshot);value=_decode(path.parent,snapshot['tree'])
    size,digest=_identity(_parts(value,ensure_ascii=snapshot['ensure_ascii']))
    if size!=snapshot['logical_bytes'] or digest!=snapshot['logical_sha256']:
        raise ValueError('Reconstructed logical JSON identity failed')
    return value


def iter_verified_bytes(path):
    path=Path(path)
    record=json.loads(path.read_bytes()) if path.exists() else None
    if not isinstance(record,dict) or record.get('kind')!=MAGIC:
        yield from legacy.iter_verified_bytes(path);return
    value=load_json(path);snapshot=_root(path)
    yield from _parts(value,ensure_ascii=snapshot['ensure_ascii'])


def artifact_identity(path):
    size,digest=_identity(iter_verified_bytes(path))
    result={'logical_path':str(path),'bytes':size,'sha256':digest}
    path=Path(path)
    if path.exists():
        raw=path.read_bytes();record=json.loads(raw)
        if isinstance(record,dict) and record.get('kind')==MAGIC:
            result.update(physical_descriptor_bytes=len(raw),physical_descriptor_sha256=_sha(raw),
                          storage='shared_immutable_v5')
    return result


def artifact_exists(path):return Path(path).is_file() or legacy.index_path(path).is_file()
