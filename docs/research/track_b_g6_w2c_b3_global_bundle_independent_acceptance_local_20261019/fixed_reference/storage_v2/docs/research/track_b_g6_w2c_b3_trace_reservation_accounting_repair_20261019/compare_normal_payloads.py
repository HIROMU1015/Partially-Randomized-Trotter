"""Compare fixed/S/P1 canonical pair payloads, excluding measured time only."""
from pathlib import Path
from hashlib import sha256
import json,sys,tempfile,tarfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V

def main():
    archives={'fixed_normal':ROOT/'docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/forward_evidence/normal/artificial_raw.tar.gz',
        'S_normal':HERE/'fault_cases_final/S_normal/artificial_raw.tar.gz',
        'P1_normal':HERE/'fault_cases_final/P1_normal/artificial_raw.tar.gz'}
    with tempfile.TemporaryDirectory(prefix='b3-storage-canonical-')as t:
        roots={}
        for name,file in archives.items():
            dest=Path(t)/name;dest.mkdir()
            with tarfile.open(file)as tf:tf.extractall(dest,filter='data')
            roots[name]=dest/'artificial-output'
        keys=sorted(f.name for f in roots['fixed_normal'].glob('*.jsonl')if f.name not in('control.jsonl','operations.jsonl','pilot_summary.jsonl'))
        rows=[]
        for name in keys:
            signatures={}
            for label,root in roots.items():
                h=sha256();count=0
                for row in V.Reader(root/name):
                    obj=row.payload;obj.pop('generator_exclusive_ns',None)
                    h.update(V.canonical(obj).encode());count+=1
                signatures[label]=dict(rows=count,sha256=h.hexdigest())
            assert len({(v['rows'],v['sha256'])for v in signatures.values()})==1,(name,signatures)
            rows.append(dict(pair=name,signatures=signatures))
        assert len(rows)==24
    (HERE/'canonical_compatibility.json').write_text(json.dumps(dict(pairs=rows,
        archive_sha256={k:sha256(p.read_bytes()).hexdigest()for k,p in archives.items()},
        excluded_metadata=['generator_exclusive_ns'],all_payloads_equal=True),indent=2)+'\n')
    print('Canonical equality: 24 pairs across fixed/S/P1')

if __name__=='__main__':main()
