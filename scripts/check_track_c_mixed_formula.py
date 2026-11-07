"""Read-only checks of saved scientific evidence; no H4 build or simulation."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

def main():
    root=Path(__file__).resolve().parents[1]
    directory=root/'artifacts/track_c_mixed_formula/2026-11-08'
    final=json.loads((directory/'FINAL.json').read_text())
    assert final['status']=='COMPLETE_MANDATORY_STOP'
    raw=(directory/'RAW_DATA.tar.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==final['archive']['sha256']
    index=json.loads((directory/'RAW_INDEX.json').read_text())
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        assert len(tar.getmembers())==len(index['files'])==199
        for record in index['files']:
            data=tar.extractfile(record['path']).read()
            assert len(data)==record['bytes']
            assert hashlib.sha256(data).hexdigest()==record['sha256'],record['path']
            local=directory/record['path']
            if local.exists():assert local.read_bytes()==data,record['path']
        def saved(name):return json.load(tar.extractfile(name))
        assert len(saved('SCREEN.json')['rows'])==684
        assert len(saved('FULL_ROUNDS.json')['rows'])==3564
        pairs=saved('FULL_PAIRS.json')['rows']
        for x in pairs:
            assert x['same_D_R_count']
            assert not x['candidate_feasible'] or x['reference_feasible']
            if x['candidate_feasible']:assert x['candidate_N']>=x['reference_N']
        for x in saved('RPE_CHECKS.json')['records']:
            assert x['source_mean_energy_pass'] and x['correct_branch_at_all_rounds']
        price=saved('PAIRED_PRICE_SUMMARY.json')
        assert price['all_paired_metrics_equal'] and price['all_native_topologies_equal']
        assert price['state_checks']==24 and price['maximum_readout_error']<1e-10
        assert saved('SOURCE_SEMANTICS.json')['maximum_error']<1e-10
        assert saved('THEORY_AND_LOCAL.json')['BCH']['completed_square']=='0'
    assert len(list(csv.DictReader((directory/'FULL_ROUNDS.csv').open())))==3564
    print('Saved evidence PASS: 199 raw files, hashes, paired prices, source semantics, RPE diagnostics; mandatory STOP.')

if __name__=='__main__':main()
