"""Postprocessing of artificial raw files only; no scientific numeric analysis."""
from pathlib import Path
from hashlib import sha256
import gzip,io,json,subprocess,tarfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019'

def main():
    dest=HERE/'evidence';dest.mkdir(exist_ok=False)
    locations=[];members=[]
    for case in ('sealed_small','sealed_n256','S_generator','P1_angle'):
        d=HERE/('runs_'+case);root=Path(json.loads((d/'location.json').read_text())['root'])
        locations.append(dict(category='port_cases',case=case,original_root=str(root)))
        for p in root.rglob('*'):
            if p.is_file():members.append((f'port_cases/{case}/'+str(p.relative_to(root)),p))
        for p in d.iterdir():
            if p.is_file():members.append((f'harness/{case}/'+p.name,p))
    matrix=json.loads((OLD/'runs_final/matrix.json').read_text())
    for row in matrix['cases']:
        root=Path(row['root']);case=row['case']
        locations.append(dict(category='bundle_cases',case=case,original_root=str(root)))
        for p in root.rglob('*'):
            if p.is_file():members.append((f'bundle_cases/{case}/'+str(p.relative_to(root)),p))
    for p in (OLD/'runs_final').iterdir():
        if p.is_file():members.append(('bundle_harness/'+p.name,p))
    for folder,prefix in [(OLD/'mutations_final','bundle_overlays'),(HERE/'runs_checks_sealed/mutations','port_overlays')]:
        for p in folder.rglob('*'):
            if p.is_file():members.append((prefix+'/'+str(p.relative_to(folder)),p))
    members.sort()
    inventory=[];archive=dest/'raw.tar.gz'
    with archive.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for name,p in members:
            assert not p.is_symlink()
            raw=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;info.mtime=0
            tar.addfile(info,io.BytesIO(raw));inventory.append(dict(path=name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    parts=[]
    with archive.open('rb')as f:
        index=0
        while raw:=f.read(4*1024*1024):
            name=f'raw.tar.gz.part{index:03d}';(dest/name).write_bytes(raw)
            parts.append(dict(path='evidence/'+name,bytes=len(raw),sha256=sha256(raw).hexdigest()));index+=1
    info=dict(schema='b3_API_artificial_raw_archive_v1',archive_sha256=sha256(archive.read_bytes()).hexdigest(),
              archive_bytes=archive.stat().st_size,members=inventory,parts=parts,locations=locations,
              real_science_files=False,publication='parts only; join in numeric order',
              source_commit='78c571005fe766c61b32dc867e006531099ff608')
    (dest/'archive_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    archive.unlink()
    print(json.dumps(dict(members=len(inventory),archive_bytes=info['archive_bytes'],parts=len(parts))))
if __name__=='__main__':main()
