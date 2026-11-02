"""Publish current artificial raw + mutation overlays; never rewrite originals."""
from pathlib import Path
from hashlib import sha256
import io,json,tarfile,tempfile,sys,shutil
HERE=Path(__file__).resolve().parent

def main():
    dest=HERE/'raw_parts';dest.mkdir(exist_ok=False);files=[]
    matrix=json.loads((HERE/'runs_sealed/matrix.json').read_text())
    for row in matrix['cases']:
        root=Path(row['root'])
        for p in sorted(root.rglob('*')):
            if p.is_file():files.append((p,'cases/'+row['case']+'/'+str(p.relative_to(root))))
    for p in sorted((HERE/'mutations_evidence').rglob('*')):
        if p.is_file():files.append((p,'mutations/'+str(p.relative_to(HERE/'mutations_evidence'))))
    for case in ('delivery_close','parent_dispatch_failure'):
        row=json.loads((HERE/('runs_development2/'+case+'_result.json')).read_text());root=Path(row['root'])
        for p in sorted(root.rglob('*')):
            if p.is_file():files.append((p,'development_before_repair/'+case+'/'+str(p.relative_to(root))))
    archive=HERE/'preflight/artifical_bundle_raw.tar.gz';members=[]
    with tarfile.open(archive,'w:gz',compresslevel=6)as tar:
        for path,name in files:
            assert path.is_file()and not path.is_symlink();raw=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;info.mtime=0;tar.addfile(info,io.BytesIO(raw));members.append(dict(path=name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    parts=[]
    with archive.open('rb')as f:
        for i,raw in enumerate(iter(lambda:f.read(4*1024*1024),b'')):
            p=dest/('artificial_bundle_raw.tar.gz.part-%03d'%i);p.write_bytes(raw);parts.append(dict(path=str(p.relative_to(HERE)),bytes=len(raw),sha256=sha256(raw).hexdigest()))
    result=dict(schema='b3_bundle_raw_delivery_v1',archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive.read_bytes()).hexdigest(),parts=parts,members=members,final_case_count=len(matrix['cases']),mutation_overlay_count=len(list((HERE/'mutations_evidence').glob('*/overlay.json'))),development_before_repair_cases=['delivery_close','parent_dispatch_failure'],original_raw_modified=False)
    (HERE/'artificial_raw_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ('archive_bytes','archive_sha256','final_case_count','mutation_overlay_count')}));print('parts',len(parts),'members',len(members))
if __name__=='__main__':main()
