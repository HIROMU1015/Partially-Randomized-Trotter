"""Archive only this task's owned artificial roots and mutation overlays."""
from pathlib import Path
from hashlib import sha256
import gzip,io,json,tarfile
HERE=Path(__file__).resolve().parent
def main():
    dest=HERE/'evidence';dest.mkdir(exist_ok=False)
    matrix=json.loads((HERE/'runs_final/matrix.json').read_text());members=[]
    for row in matrix['cases']:
        root=Path(row['root'])
        for p in root.rglob('*'):
            if p.is_file():members.append(('cases/'+row['case']+'/'+str(p.relative_to(root)),p))
    for folder,prefix in [(HERE/'runs_final','harness'),(HERE/'runs_tests_final/mutations','overlays'),(HERE/'additional','additional')]:
        for p in folder.rglob('*'):
            if p.is_file():members.append((prefix+'/'+str(p.relative_to(folder)),p))
    archive=dest/'raw.tar.gz';inventory=[]
    with archive.open('wb')as out,gzip.GzipFile(fileobj=out,mode='wb',mtime=0,compresslevel=6)as gz,tarfile.open(fileobj=gz,mode='w|')as tar:
        for name,p in sorted(members):
            assert not p.is_symlink();raw=p.read_bytes();entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mtime=0;entry.mode=0o600
            tar.addfile(entry,io.BytesIO(raw));inventory.append(dict(path=name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    parts=[]
    with archive.open('rb')as f:
        i=0
        while raw:=f.read(4*1024*1024):
            p=dest/f'raw.tar.gz.part{i:03d}';p.write_bytes(raw);parts.append(dict(path='evidence/'+p.name,bytes=len(raw),sha256=sha256(raw).hexdigest()));i+=1
    data=dict(schema='b3_global_port_artificial_archive_v1',SOURCE='deade78a19f5b23859d16816d3a7bfb37765dd80',
        archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive.read_bytes()).hexdigest(),members=inventory,parts=parts,
        cases=len(matrix['cases']),overlays=45,additional_mutations=3,scientific_files=False)
    (dest/'archive_manifest.json').write_text(json.dumps(data,indent=2)+'\n');archive.unlink()
    print(json.dumps({k:data[k]for k in ('archive_bytes','archive_sha256','cases','overlays','additional_mutations')}))
if __name__=='__main__':main()
