"""Package new artificial raw; public Producer raw remains a pinned reference."""
from hashlib import sha256
from pathlib import Path
import json,tarfile
HERE=Path(__file__).resolve().parent

def main():
    names=('negative_raw_delivery','independent_writer_faults_final');members=[]
    target=HERE/'new_artificial_raw.tar.gz'
    with target.open('xb') as dst,tarfile.open(fileobj=dst,mode='w:gz') as tf:
        for name in names:
            for path in sorted((HERE/name).rglob('*')):
                if path.is_file():
                    relative=path.relative_to(HERE).as_posix();tf.add(path,arcname=relative,recursive=False)
                    raw=path.read_bytes();members.append(dict(path=relative,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    parts=[];whole=sha256();total=0
    with target.open('rb') as src:
        while raw:=src.read(4*1024*1024):
            name=target.name+'.part-'+str(len(parts)).zfill(3)
            with (HERE/name).open('xb') as dst:dst.write(raw)
            parts.append(dict(path=name,bytes=len(raw),sha256=sha256(raw).hexdigest()));whole.update(raw);total+=len(raw)
    document=dict(schema='b3_v2_independent_raw_archive_v1',path=target.name,bytes=total,sha256=whole.hexdigest(),parts=parts,members=members)
    (HERE/'new_artificial_raw_manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps(dict(archive_bytes=total,parts=len(parts),files=len(members))))
if __name__=='__main__':main()
