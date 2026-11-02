"""Read-only audit of committed lightweight records; no science imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE='f0dd2a9aa4e011b204c4ba6c51a050dff09bc471'
H4_SOURCE='687679a4ae20e4661bde1187dd8ab92ae176bd44'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(name):return json.loads((HERE/name).read_text())


def main():
    manifest=load('result_manifest.json')
    for row in manifest['files']:
        p=HERE/row['path']
        if not p.resolve().is_relative_to(HERE) or p.is_symlink():
            raise ValueError('UNSAFE_MANIFEST_PATH')
        raw=p.read_bytes()
        assert len(raw)==row['bytes'] and sha(raw)==row['sha256'],row['path']
    audit=load('final_completion_audit.json');plan=load('plan.json')
    assert audit['SOURCE']==SOURCE and audit['new_validated_compiles']==96
    assert audit['plan_sha256_by_SOURCE'][H4_SOURCE]==sha((HERE/'plan.json').read_bytes())
    assert audit['plan_sha256_by_SOURCE'][SOURCE]==sha((HERE/'parallel_H6_plan.json').read_bytes())
    assert audit['all_authoritative_native_exits_zero_and_reaped']
    assert all(v['native_exit_code']==0 and v['native_wait_reaped']
        for v in audit['completed_H4_children']+audit['completed_H6_children'])
    assert audit['two_post_exit_absence_checks']==[True,True]
    counts=0;compiler=set();checks=[]
    for group in plan['groups']:
        x=load(group['id']+'_final_result.json')
        expected_source=H4_SOURCE if group['system']=='H4' else SOURCE
        assert x['SOURCE']==expected_source and x['group']==group and x['status']=='COMPLETE_DIAGNOSTIC'
        assert x['plan_sha256']==audit['plan_sha256_by_SOURCE'][expected_source]
        if group['system']=='H6':
            for row in x['aggregate_from']:
                raw=(HERE/row['path']).read_bytes()
                assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
                part=json.loads(raw)
                assert part['seed_SOURCE']==H4_SOURCE and part['SOURCE']==SOURCE
        assert {(v['sample'],v['axis']) for v in x['records']}=={(i,a) for i in range(8) for a in ('cosine','sine')}
        assert len(x['records'])==16 and x['population_mean_certified'] is False
        assert x['quantum_measurement_shots_sampled']==0
        checks.extend(x['validation_checks'])
        totals=[]
        for i in range(8):
            rows={v['axis']:v for v in x['records'] if v['sample']==i}
            assert rows['cosine']['event_digest']==rows['sine']['event_digest']
            totals.append(sum(x['shots']['axes'][a]['shots']*rows[axis]['metrics']['rz_count']
                for a,axis in [('real','cosine'),('imag','sine')]))
        assert totals==x['RZ_shot_weighted_sample_costs']
        assert statistics.mean(totals)==x['RZ_shot_weighted_mean']
        assert statistics.stdev(totals)/math.sqrt(8)==x['RZ_shot_weighted_SE']
        compiler.update(v['compiler_settings_hash'] for v in x['records'])
        counts+=len(x['records'])
    assert counts==96 and len(compiler)==1
    assert len(checks)==audit['validation_check_count']
    assert max(v['error'] for v in checks)==audit['maximum_validation_error']<=1e-9
    for path,h in load('parallel_H6_source_closure.json')['paths'].items():
        raw=subprocess.check_output(['git','--no-optional-locks','-c','maintenance.auto=false',
            '-c','gc.auto=0','-C',str(ROOT),'show',SOURCE+':'+path])
        assert sha(raw)==h,path
    print(json.dumps(dict(status='PASS_SAVED_ARTIFACT_CONSISTENCY',SOURCE=SOURCE,
        groups=6,compiled_records=96,numeric_checks=len(checks),
        scientific_reexecution=False,provider_or_environment_reverification=False)))


if __name__=='__main__':main()
