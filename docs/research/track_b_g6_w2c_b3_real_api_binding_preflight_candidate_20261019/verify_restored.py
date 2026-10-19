"""Restore artificial evidence into a fresh private directory and check bytes."""
from pathlib import Path
from hashlib import sha256
import gzip,io,json,shutil,sys,tarfile,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True

def main():
    from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import offline as O
    from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_offline_v1 as B
    m=json.loads((HERE/'evidence/archive_manifest.json').read_text());dest=Path(tempfile.mkdtemp(prefix='b3-api-restored-'))
    raw=b''
    for part in m['parts']:
        data=(HERE/part['path']).read_bytes();assert len(data)==part['bytes'] and sha256(data).hexdigest()==part['sha256'];raw+=data
    assert len(raw)==m['archive_bytes'] and sha256(raw).hexdigest()==m['archive_sha256']
    expected={x['path']:x for x in m['members']};seen=set()
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
        for t in tar:
            assert t.isfile() and t.name in expected and t.name not in seen and not Path(t.name).is_absolute() and '..' not in Path(t.name).parts
            data=tar.extractfile(t).read();row=expected[t.name]
            assert len(data)==row['bytes'] and sha256(data).hexdigest()==row['sha256']
            p=dest/t.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);seen.add(t.name)
    assert seen==set(expected)
    results=[]
    for x in m['locations']:
        p=dest/x['category']/x['case']
        if x['category']=='port_cases':
            c=json.loads((dest/'harness'/x['case']/'trusted_candidate.json').read_text())
            v=O.verify(p,c,ROOT,execution_root=x['original_root'])
            want='VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE' if x['case'].startswith('sealed') else 'VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE'
        else:
            c=json.loads((dest/'bundle_harness/trusted_candidate.json').read_text());v=B.verify(p,c)
            want=json.loads((dest/'bundle_harness'/(x['case']+'_result.json')).read_text())['verification_status']
        assert v['status']==want,(x,v)
        results.append(dict(category=x['category'],case=x['case'],status=v['status']))
    base=dest/'port_cases/sealed_small'
    execution=next(x['original_root'] for x in m['locations']if x['case']=='sealed_small')
    for d in sorted((dest/'port_overlays').iterdir()):
        want=json.loads((d/'observation.json').read_text());trusted=json.loads((d/'trusted_candidate.json').read_text())
        v=O.verify(d/'case',trusted,ROOT,execution_root=execution)
        assert v['status']==want['status'],(d,v,want)
        results.append(dict(category='port_mutation',case=d.name,status=v['status']))
    for d in sorted((dest/'bundle_overlays').iterdir()):
        want=json.loads((d/'overlay.json').read_text());p=dest/'overlay_case';shutil.copytree(dest/'bundle_cases/normal_bundle_small',p)
        for row in want['changed_files']:
            q=p/row['path'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(d/row['path'],q)
        for name in want['deleted_files']:(p/name).unlink()
        v=B.verify(p,json.loads((dest/'bundle_harness/trusted_candidate.json').read_text()))
        assert v['status']==want['status'],(d,v,want)
        results.append(dict(category='bundle_mutation',case=d.name,status=v['status']));shutil.rmtree(p)
    report=dict(schema='b3_API_fresh_archive_restore_verification_v1',status='PASS',members=len(seen),results=results,
                restore_root=str(dest),same_primary_agent_self_check=True,scientific_GO=False,mandatory_STOP=True)
    (HERE/'restored_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],members=len(seen),classifications=len(results))))
if __name__=='__main__':main()
