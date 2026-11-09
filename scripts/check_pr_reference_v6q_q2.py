"""Verify saved Q2 output without rerunning provider, circuits or simulation."""
from fractions import Fraction as F
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pr_reference_v6q/2026-11-09/Q2_run01'
DRIVER_COMMIT='ca19707b5c0c1cefa338d47987c27f338cb4cf67'
PROVIDER_COMMIT='eaf70d886d42364a01b7e949ebc21f3b026bbe72'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def exact_literal(value):
    if value['type']=='builtins.int':return F(int(value['value']))
    assert value['type'] in ('builtins.float','numpy.float64')
    return F.from_float(float.fromhex(value['value']))


def main():
    if (OUT/'CHECKS.json').exists():raise FileExistsError('Preserve old checks')
    raw=(OUT/'RESULTS.json').read_bytes();data=json.loads(raw)
    source=json.loads((OUT/'SOURCE.json').read_text());stop=json.loads((OUT/'STOP.json').read_text())
    assert source['execution_commit']==DRIVER_COMMIT and source['provider_commit']==PROVIDER_COMMIT
    for key,path,ref in [('driver','scripts/run_pr_reference_v6q_q2.py',DRIVER_COMMIT),
                         ('provider','src/trottertracks/pr_reference_v6q.py',PROVIDER_COMMIT)]:
        actual=(ROOT/path).read_bytes()
        assert actual==subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
        assert sha(actual)==source[key+'_sha256']
    assert sha((OUT/'common.pkl').read_bytes())==source['manifest']['common_input']['pickle_sha256']
    packed=(OUT/'LITERALS.json.xz').read_bytes();literal_raw=lzma.decompress(packed)
    assert sha(packed)==data['literal_archive_sha256'] and sha(literal_raw)==data['literal_raw_sha256']
    literals=json.loads(literal_raw)
    assert len(data['rows'])==len(literals)==4
    assert [(r['identity']['q'],r['identity']['rare']) for r in data['rows']]==[(1,False),(1,True),(4,False),(4,True)]
    maxima=dict(old_XY_error=0.,new_XY_error=0.,vacuum_action_error=0.,vacuum_leakage=0.,inverse_error=0.)
    negatives=[]
    for row, lit in zip(data['rows'],literals):
        assert row['identity']==lit['identity'] and row['passed'] and row['old_tolerance']==1e-9
        values=[]
        for name in ('F','G'):
            circuit=lit[name]
            total=exact_literal(circuit['global_phase'])
            for op in circuit['ops']:
                assert op['name'] in ('rz','rzz','xx_plus_yy','z','p','id')
                if op['name'] in ('rz','rzz'):total-=exact_literal(op['params'][0])/2
            values.append(total)
            vacuum=next(v for v in row['vacuum'] if v['lane']==name)
            assert F(vacuum['exact_vacuum_phase'])==total
            assert 0<=vacuum['leakage']<1e-9 and 0<=vacuum['action_error']<1e-9
            maxima['vacuum_action_error']=max(maxima['vacuum_action_error'],vacuum['action_error'])
            maxima['vacuum_leakage']=max(maxima['vacuum_leakage'],vacuum['leakage'])
        total=sum(values,F())
        rn=row['rounding'];value=row['phase_new'];center=F.from_float(value)
        assert F(rn['exact_sum'])==total and rn['returned_hex']==value.hex()
        assert F(rn['signed_rounding_residual'])==center-total
        left=(center+F.from_float(math.nextafter(value,-math.inf)))/2
        right=(center+F.from_float(math.nextafter(value,math.inf)))/2
        even=not(struct.unpack('>Q',struct.pack('>d',value))[0]&1)
        assert left<total<right or (even and total in (left,right))
        assert F(rn['left_midpoint'])==left and F(rn['right_midpoint'])==right
        assert rn['even_significand']==even and rn['passed']
        assert value==float(total) and total!=0 and math.isfinite(value)
        snapshot_raw=json.dumps([lit['F'],lit['G']],sort_keys=True).encode()
        assert sha(snapshot_raw)==row['literal_snapshot_digest']
        assert row['provider_input_order_and_values_preserved']
        shape=row['core_shape']
        assert shape['only_ancilla_P_may_differ'] and shape['new_P']['wires']==[16]
        assert shape['new_P']['params']==[dict(type='builtins.float',value=value.hex())]
        for key in ('old','new'):
            assert 0<=row['XY_errors'][key]<1e-9
            maxima[key+'_XY_error']=max(maxima[key+'_XY_error'],row['XY_errors'][key])
        assert 0<=row['inverse_error']<1e-9
        maxima['inverse_error']=max(maxima['inverse_error'],row['inverse_error'])
        separate=json.loads((OUT/f"q{row['identity']['q']}_rare{int(row['identity']['rare'])}.json").read_text())
        assert separate==row
        if row['negative_witness'] is not None:
            negative=row['negative_witness'];assert row['identity']['q']==1 and not row['identity']['rare']
            assert negative['passed'] and negative['error_to_negative_new_z']<1e-9
            assert negative['mismatch_to_correct_z']>1e-9
            negatives.append(negative)
    assert len(negatives)==1
    assert data['candidate_scientific_calls']==4 and data['reference_XY_axis_evaluations']==18
    assert data['compile_sampling_price_RPE_T_calls']==0 and not data['full_support_or_V6P_cost_promoted']
    assert data['status']==stop['status']=='FOUR_SAVED_FIXTURES_PASS_ALL_SUPPORT_PENDING'
    assert stop['completed_fixtures']==4 and not stop['Q3_started'] and stop['next_action']=='MANDATORY_STOP'
    checks=dict(status='Q2_POSTRUN_CHECKS_PASS',execution_source=DRIVER_COMMIT,
        scientific_campaigns=1,scientific_retries=0,CLI_exit_code=0,
        CLI_invocations_total=2,pre_SOURCE_validation_failure=dict(count=1,
            cause='Git commit metadata synchronization failure; failed SOURCE lookup before output creation',
            scientific_input_or_circuit_calls=0,driver_or_provider_changed=False),
        fixed_source_and_cache_hashes=True,four_exact_input_sums_and_RN_even_checks=True,
        original_thresholds_preserved=True,maxima=maxima,negative_witness=negatives[0],
        candidate_calls=4,reference_XY_axis_evaluations=18,compile_sampling_price_RPE_T_calls=0,
        result_sha256=sha(raw),literal_archive_sha256=sha(packed),literal_raw_sha256=sha(literal_raw),
        checker_sha256=sha(Path(__file__).read_bytes()),independent_scientific_review=False,
        all_support_qualified=False,V6P_cost_promoted=False,next_action='MANDATORY_STOP')
    assert (OUT/'RESULTS.json').read_bytes()==raw
    with (OUT/'CHECKS.json').open('x') as stream:json.dump(checks,stream,indent=2);stream.write('\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
