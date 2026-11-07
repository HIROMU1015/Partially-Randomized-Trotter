#!/usr/bin/env python3
"""Read-only archived evidence checks; no circuit/H4 computation."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_c_diagonal_fusion/2026-11-08'


def main():
    index=json.loads((OUT/'RAW_INDEX.json').read_text());archive=OUT/'RAW_DATA.tar.gz';final=json.loads((OUT/'FINAL.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==index['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        entries={x['path']:x for x in index['files']};assert set(tar.getnames())==set(entries)
        def read(n):return json.load(tar.extractfile(n))
        for name,e in entries.items():
            raw=tar.extractfile(name).read();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'],name
        assert read('FINAL.json')==final
        f=read('POLICY.json')
        for p in f['policies']:
            old=next(x for x in f['old_policies'] if (x['family'],x['primary'])==(p['family'],p['primary']))
            assert [s['signal'] for s in old['selected']]==[s['signal'] for s in p['selected']]
            assert sum(Fraction.from_float(a) for row in p['alpha'] for a in row)<=Fraction(1,20)
            for s,row,aa in zip(p['selected'],p['rounds'],p['alpha']):
                q=s['qualification'];assert q['closed'] and q['rho_lower']>0 and q['beta_stat_lower']>0
                for n,a in zip(row['shots'],aa):
                    b=4*math.log(2/a)/(q['rho_lower']**2*math.sin(q['beta_stat_lower'])**2)
                    assert n>=b-1e-8 and n-1<b+1e-8
        for name in entries:
            if name.startswith(('training_prices/','holdout_prices/')):
                r=read(name);choice=min(r['variants'],key=lambda k:tuple(r['variants'][k]));assert r['selected']==r['variants'][choice]
            if name.startswith('det_prices/'):assert all(e==0 for e in read(name)['errors'].values())
        for row in read('CORRECTNESS.json')['rows']:
            for x in row['rows']:assert max(x['source_Fro_error'],x['native_projective_Fro_error'],*x['XY_errors'])<1e-9
        for row in read('D_CORRECTNESS.json')['rows']:
            for x in row['rows']:assert max(x['source_Fro_error'],x['native_projective_Fro_error'])<1e-9 and x['rounding_error']==0
        for p in final['policies']:
            assert p['mean_branch_pass'] and p['mean_energy_error']<1.5936e-4
            for i in range(4):assert math.isclose(sum(x['total'][i] for x in p['new']['rounds']),p['new']['totals']['total'][i],rel_tol=1e-12)
        for s in read('RESOURCES.json')['short_checks']:
            for r in s['rows']:
                assert r['modular_DAG'][2]<=r['serial_depth_upper']
                assert all(r['whole_minus_modular'][i]==0 for i in (0,1,3))
                if r['readout_error'] is not None:assert r['readout_error']<1e-8
        assert read('STOP.json')['status']=='MANDATORY_STOP' and final['mandatory_STOP']
    print('PASS: raw hashes, fixed policies, rational/native correctness, paired prices, exact alpha/shots, modular accounting and STOP')


if __name__=='__main__':main()
