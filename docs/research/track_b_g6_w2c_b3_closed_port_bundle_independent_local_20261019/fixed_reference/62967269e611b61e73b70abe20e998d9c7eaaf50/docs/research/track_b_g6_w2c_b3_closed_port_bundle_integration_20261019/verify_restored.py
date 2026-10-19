"""Fresh restore of archived artificial evidence; separate consumer imports."""
from pathlib import Path
from hashlib import sha256
import io,json,shutil,sys,tarfile,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
def main():
    from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import offline as O
    m=json.loads((HERE/'evidence/archive_manifest.json').read_text());dest=Path(tempfile.mkdtemp(prefix='b3-port-bundle-restored-'))
    data=b''
    for row in m['parts']:
        raw=(HERE/row['path']).read_bytes();assert len(raw)==row['bytes']and sha256(raw).hexdigest()==row['sha256'];data+=raw
    assert len(data)==m['archive_bytes']and sha256(data).hexdigest()==m['archive_sha256']
    inventory={r['path']:r for r in m['members']};seen=set()
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as tar:
        for t in tar:
            assert t.isfile()and t.name in inventory and t.name not in seen and not Path(t.name).is_absolute()and'..'not in Path(t.name).parts
            raw=tar.extractfile(t).read();r=inventory[t.name];assert len(raw)==r['bytes']and sha256(raw).hexdigest()==r['sha256']
            p=dest/t.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);seen.add(t.name)
    assert seen==set(inventory)
    c=json.loads((dest/'harness/trusted_candidate.json').read_text());matrix=json.loads((dest/'harness/matrix.json').read_text());results=[]
    for row in matrix['cases']:
        v=O.verify(dest/'cases'/row['case'],c);assert v['status']==row['verification_status'],(row['case'],v)
        results.append(dict(case=row['case'],status=v['status'],kind='integrated_case'))
    for d in sorted((dest/'overlays').iterdir()):
        meta=json.loads((d/'overlay.json').read_text());p=dest/'mutation_case';shutil.copytree(dest/'cases'/meta['base_case'],p)
        for row in meta['changed_files']:
            q=p/row['path'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(d/row['path'],q)
        for name in meta['deleted_files']:(p/name).unlink()
        v=O.verify(p,c);assert v['status']==meta['status'],(d,v,meta)
        results.append(dict(case=d.name,status=v['status'],kind='mutation'));shutil.rmtree(p)
    for d in sorted((dest/'additional').iterdir()):
        if not d.is_dir():continue
        meta=json.loads((d/'observation.json').read_text());v=O.verify(d/'case',c);assert v['status']==meta['status'],(d,v)
        results.append(dict(case=d.name,status=v['status'],kind='additional_mutation'))
    report=dict(schema='b3_port_bundle_restored_verification_v1',status='PASS',members=len(seen),results=results,
        restore_root=str(dest),scientific_GO=False,mandatory_STOP=True,same_primary_agent_self_check=True)
    (HERE/'restored_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='PASS',members=len(seen),classifications=len(results))))
if __name__=='__main__':main()
