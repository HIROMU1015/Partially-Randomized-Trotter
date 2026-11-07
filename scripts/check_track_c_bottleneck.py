#!/usr/bin/env python3
"""Read-only checks of published Track C bottleneck evidence; no H4 runs."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_c_bottleneck/2026-11-08'


def main():
    final=json.loads((OUT/'FINAL.json').read_text()); index=json.loads((OUT/'RAW_INDEX.json').read_text())
    archive=OUT/'RAW_DATA.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==index['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        expected={r['path']:r for r in index['files']}
        assert set(tar.getnames())==set(expected)
        def read(path): return json.load(tar.extractfile(path))
        for name,r in expected.items():
            data=tar.extractfile(name).read()
            assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],name
        optimization=read('OPTIMIZATION.json'); validation=read('VALIDATION.json')
        profiles=read('TEMPLATE_COMPONENTS.json'); certificate=read('REPLACEMENT_CERTIFICATE.json')
        assert len(certificate['components'])==218 and certificate['contexts']==128
        assert all(r['involution_error_upper']<1e-10 for r in certificate['components'])
        for name in expected:
            if not name.startswith(('runwise_prices/','joint_prices/','holdout_prices/')):continue
            r=read(name); selected=r['proposal'] if r['proposal'][0]<r['old'][0] else r['old']
            assert r['selected']==selected and len(selected)==4
        for policy in optimization['optimized']:
            assert len(policy['selected'])==18
            assert sum(Fraction.from_float(a) for row in policy['alpha'] for a in row)<=Fraction(1,20)
            for s,row,alphas in zip(policy['selected'],policy['rounds'],policy['alpha']):
                q=s['qualification']; assert q['closed'] and q['rho_lower']>0 and q['beta_stat_lower']>0
                for n,a in zip(row['shots'],alphas):
                    bound=4*math.log(2/a)/(q['rho_lower']**2*math.sin(q['beta_stat_lower'])**2)
                    assert n>=bound-1e-8 and n-1<bound+1e-8
        for p in profiles['policies']:
            for row in p['rows']:
                for i in range(4):
                    total=sum(v[i] for group in ('D','RTE') for v in row[group].values())+row['wrapper'][i]
                    assert math.isclose(total,row['total'][i],rel_tol=1e-12,abs_tol=1e-5)
        for p in validation['policies']:
            assert p['all_source_native_closed'] and p['mean_branch_pass']
            assert p['mean_energy_error']<=1.5936e-4 and p['energy_statistical_bound']<1.5936e-4
            frozen=next(q for q in optimization['optimized'] if q['family']=='PR' and q['primary']==p['primary'])
            assert p['alpha']==frozen['alpha'] and [r['shots'] for r in p['rounds']]==[r['shots'] for r in frozen['rounds']]
        for s in validation['short_checks']:
            for row in s['rows']:
                assert all(row['whole_DAG'][i]==row['modular_DAG'][i] for i in (0,1,3))
                assert row['modular_DAG'][2]<=row['modular_serial_depth_upper']
                if row['state']:assert max(row['state']['readout_errors']+row['state']['projective_errors'])<1e-8
        assert final['PR_RZ_policy_holdout']['totals']==validation['policies'][0]['totals']
    assert final['mandatory_STOP'] and (OUT/'STOP.json').is_file()
    print('PASS: raw hashes, 218 replacements, native budgets, exact alpha, fixed holdout, additive costs, X/Y and STOP')


if __name__=='__main__':main()
