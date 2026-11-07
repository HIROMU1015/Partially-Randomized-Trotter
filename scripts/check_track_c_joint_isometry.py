#!/usr/bin/env python3
"""Read-only published evidence checks. No Qiskit, input cache or H4 run."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_c_joint_isometry/2026-11-08'


def main():
    index=json.loads((OUT/'RAW_INDEX.json').read_text());archive=OUT/'RAW_DATA.tar.gz';final=json.loads((OUT/'FINAL.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==index['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        entries={x['path']:x for x in index['files']};assert set(tar.getnames())==set(entries)
        def read(name):return json.load(tar.extractfile(name))
        for name,x in entries.items():
            data=tar.extractfile(name).read();assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256'],name
        assert final==read('FINAL.json')
        fixed=read('POLICY.json');cat=read('CATALOGUE_FINAL.json')['entries'];cert=read('CERTIFICATE.json')
        assert len(cat)==639 and sum(x['adopt'] for x in cat.values())==625
        assert cert['contexts']==128 and cert['unique_native_lowerings']==5380
        assert max(x['involution_error_upper'] for x in cert['columns'])<1e-12
        for p in fixed['policies']:
            assert sum(Fraction.from_float(a) for row in p['alpha'] for a in row)<=Fraction(1,20)
            old=next(q for q in fixed['old_policies'] if q['family']=='PR' and q['primary']==p['primary'])
            assert [s['signal'] for s in p['selected']]==[s['signal'] for s in old['selected']]
            for s,row,alphas in zip(p['selected'],p['rounds'],p['alpha']):
                q=s['qualification'];assert q['closed'] and q['rho_lower']>0 and q['beta_stat_lower']>0
                for n,a in zip(row['shots'],alphas):
                    b=4*math.log(2/a)/(q['rho_lower']**2*math.sin(q['beta_stat_lower'])**2)
                    assert n>=b-1e-8 and n-1<b+1e-8
        for name in entries:
            if name.startswith(('training_final/','holdout_prices/')):
                r=read(name);assert r['selected']==(r['joint_proposal'] if r['joint_proposal'][0]<r['baseline'][0] else r['baseline'])
        for x in read('TECHNICAL_FIX.json')['preserved_results']:
            assert entries['training_final/'+x['path']]['sha256']==x['sha256']
        reps=read('REPRESENTATIVES.json')['rows'];assert len(reps)==11
        for r in reps:
            assert max(r['joint_operator_Fro_errors'])<1e-9 and r['inactive_identity_Fro']<1e-9
            for m in r['means']:assert max(abs(v-m['values'][0]) for v in m['values'])<1e-9
        assert not any(r['improves'] for r in read('LAYOUTS_RESUMED.json')['full'])
        for p in final['policies']:
            assert p['mean_branch_pass'] and p['mean_energy_error']<1.5936e-4 and p['energy_statistical_bound']<1.5936e-4
            for i in range(4):assert math.isclose(sum(r['total'][i] for r in p['new']['rounds']),p['new']['totals']['total'][i],rel_tol=1e-12)
        for s in read('RESOURCES.json')['short_checks']:
            for r in s['rows']:
                assert all(r['whole_DAG'][i]==r['modular_DAG'][i] for i in (0,1,3))
                assert r['modular_DAG'][2]<=r['serial_depth_upper']
                if r['readout_error'] is not None:assert r['readout_error']<1e-8
        assert read('STOP.json')['status']=='MANDATORY_STOP' and final['mandatory_STOP']
    print('PASS: raw hashes, preserved samples, frozen union policy, Fock/native/X/Y, exact alpha/shots, all-round resources and STOP')


if __name__=='__main__':main()
