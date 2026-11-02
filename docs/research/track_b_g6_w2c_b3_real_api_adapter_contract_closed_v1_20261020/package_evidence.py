"""Package only new owned artificial outputs; no scientific result processing."""
from pathlib import Path
from hashlib import sha256
import gzip,io,json,tarfile,sys
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    matrix,units,integration=map(Path,sys.argv[1:4]);dest=HERE/'evidence';dest.mkdir(exist_ok=False)
    rows=json.loads((matrix/'matrix.json').read_text())['cases'];members=[]
    for row in rows:
        root=Path(row['root'])
        for p in root.rglob('*'):
            if p.is_file():members.append(('cases/'+row['case']+'/'+str(p.relative_to(root)),p))
    for folder,prefix in [(matrix,'matrix'),(units,'units'),(integration,'integration')]:
        for p in folder.rglob('*'):
            if p.is_file():members.append((prefix+'/'+str(p.relative_to(folder)),p))
    # Serialize archive to owned temp, split retained bytes; never remove old evidence.
    archive=dest/'raw.tar.gz';inventory=[]
    with archive.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for name,p in sorted(members):
            assert not p.is_symlink();raw=p.read_bytes();entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=0o600
            tar.addfile(entry,io.BytesIO(raw));inventory.append(dict(path=name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    parts=[]
    with archive.open('rb') as inp:
        i=0
        while raw:=inp.read(4*1024*1024):
            p=dest/f'raw.tar.gz.part{i:03d}';p.write_bytes(raw)
            parts.append(dict(path='evidence/'+p.name,bytes=len(raw),sha256=sha256(raw).hexdigest()));i+=1
    data=dict(schema='b3_closed_adapter_artificial_archive_v1',archive_bytes=archive.stat().st_size,
        archive_sha256=sha256(archive.read_bytes()).hexdigest(),members=inventory,parts=parts,
        cases=len(rows),mutation_overlays=20,science=False)
    (dest/'archive_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    archive.unlink()  # newly generated intermediate; all bytes preserved in parts
    print(json.dumps({k:data[k] for k in ('archive_bytes','archive_sha256','cases','mutation_overlays')}))
