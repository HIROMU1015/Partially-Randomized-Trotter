"""Standard-library, read-only evidence check. No H4 calculation/import."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

def main():
    root=Path(__file__).resolve().parents[1];p=root/'artifacts/track_c_d_internal/2026-11-08'
    final=json.loads((p/'FINAL.json').read_text());assert final['status']=='COMPLETE_MANDATORY_STOP'
    raw=p/'RAW_DATA.tar.gz';assert hashlib.sha256(raw.read_bytes()).hexdigest()==final['archive']['sha256']
    index=json.loads((p/'RAW_INDEX.json').read_text())
    with tarfile.open(raw,'r:gz') as tar:
        assert len(tar.getmembers())==len(index['files'])==final['archive']['files']
        for record in index['files']:
            data=tar.extractfile(record['path']).read();assert len(data)==record['bytes']
            assert hashlib.sha256(data).hexdigest()==record['sha256'],record['path']
            local=p/record['path']
            if local.exists():assert local.read_bytes()==data,record['path']
        def saved(name):return json.load(tar.extractfile(name))
        assert saved('SEMANTICS.json')['maximum_error']<1e-10
        assert saved('QUALIFICATION.json')['all_closed']
        for file in ['QUALIFICATION_EXTENSION.json','REGISTERED_QUALIFICATION.json']:
            assert all(r['closed'] for r in saved(file)['rows'])
        result=saved('REGISTERED_COMPARISON.json');base=result['results'][0];tuned=next(x for x in result['results'] if x['name']=='tuned_0123')
        assert base['total_shots']==8598 and tuned['total_shots']==8498
        assert tuned['G'][0]<base['G'][0] and tuned['G'][1]<base['G'][1]
        assert all(r['all_rounds_correct_branch'] and r['energy_error']<1.5936e-4 for r in result['results'])
        d=saved('QUALIFIED_G.json');strong=next(x for x in d['results'] if x['name']=='Full_nonresonant_RZ')
        assert strong['G'][0]<tuned['G'][0] and strong['G'][1]<tuned['G'][1]
        assert len(saved('STAGE_A.json')['rows'])==28 and len(saved('STAGE_A2.json')['rows'])==96
        assert saved('STAGE_A2.json')['unique_roots']==[]
        assert saved('SIGNAL_SUMMARY.json')['count']==11988 and saved('STRONG_FULL.json')['rows']==7452
    assert len(list(csv.DictReader((p/'RESOURCE_TABLE.csv').open())))==18
    print('Saved evidence PASS: all raw bytes, interval closure, registered comparison, strong baseline, branch; mandatory STOP.')

if __name__=='__main__':main()
