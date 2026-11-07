"""Technical integrity checks of saved one-campaign records only.

No scientific module imports, fit, paired-statistic recomputation, model load,
PRNG, engine, circuit or price acquisition. Existing outputs stay immutable.
"""
from collections import Counter
from fractions import Fraction as F
import hashlib,json
from pathlib import Path

ROOT=Path('/tmp/scv-signed-support-448-preparation-20261107')
OUT=ROOT/'artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01'
DOC=Path('docs/research/scv_signed_support_448_science_20261107')


def read(name):return json.loads((OUT/name).read_text())
def lines(name):
    with name.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def main():
    result=read('SIGNED_RESULT.json');complete=result['status']=='COMPLETE';checks={}
    source=read('SOURCE.json');assert source['source_commit']=='f70e555fbfe08676f530d989192aade75a57ab1a'
    assert source['campaign']=='SCV-SIGNED-H4-H1-PAIRED-20261107-v1'
    assert source['seed']==20261110 and source['old_measurements_reused']==0
    assert source['control_identity']=='5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89'
    restored=read('RESTORED_CONTROL.json');assert restored['moment_recomputation']==0
    assert restored['snapshot']['control_identity']==source['control_identity']
    streams=read('STREAMS.json');assert streams['seed']==20261110 and streams['children']==24
    assert streams['mapping']==source['settings']['domain_mapping'] and len(streams['mapping'])==12
    checks['source_control_stream_binding']='PASS'
    counts=read('BOUNDED_CALL_COUNTS.json')
    assert counts['static_caps']==source['settings']['call_caps']
    for name,value in counts['observed'].items():
        assert 0<=value['returned']+value['exceptions']<=value['started']<=counts['static_caps'][name]
        if complete:assert value['in_flight']==value['exceptions']==0 and value['returned']==value['started']
    if complete:
        expected=dict(physical_trial=448,conditional_path=448,probability=448,Born=448,bit_threshold=448,price=448,
            selected_full_build=896,OLS=2,paired=6,saved_bundle_decode=1,saved_control_restore=1,stream_book=1)
        for name,n in expected.items():assert counts['observed'][name]['started']==n,(name,counts['observed'][name])
    checks['finite_call_caps']='PASS'
    stopped=False;gate_counts=Counter();first_eval=None;fit_returns=[]
    for row in lines(OUT/'ADAPTER_CALLS.jsonl'):
        if row['event']=='STOP_REQUESTED':stopped=True
        if row['event']=='CALL_START':
            assert not stopped,'new CALL_START after durable STOP'
            if row['name']=='apply_exact':gate_counts[row['ticket']]+=1
            if row['name']=='conditional_path' and json.loads(row['domain'])[2]=='prod' and first_eval is None:first_eval=row['call_id']
        if row['event']=='CALL_RETURN' and row['name']=='OLS':fit_returns.append(row['call_id'])
    assert all(n<=955 for n in gate_counts.values())
    if first_eval is not None:assert len(fit_returns)==2 and first_eval>max(fit_returns)
    checks['STOP_order_gate_cap_and_both_fits_before_eval']='PASS'
    bits={};prices={};known_rz=0;worker_failures=[]
    for path in OUT.glob('worker_*.jsonl'):
        for row in lines(path):
            event=row['event'];ticket=row['ticket']
            if event=='BIT_SAVED':
                assert ticket not in bits;bits[ticket]=row
                assert row['record']['bit'] in (0,1)
                assert not set(row['record'])&{'probability','state','E0','gap','true_variance','signal'}
            if event=='PRICE_SAVED':
                assert ticket in bits and ticket not in prices and bits[ticket]['record']==row['record']
                assert type(row['metrics']['rz_count']) is int and row['metrics']['rz_count']>=0
                known_rz+=row['metrics']['rz_count'];prices[ticket]=row
            if event=='WORKER_STOP':worker_failures.append(row)
    if complete:assert len(bits)==len(prices)==448 and not worker_failures
    checks['durable_bit_before_price_and_oracle_boundary']='PASS'
    quotas=(8,128,8,8,8);allkeys=set();received=0
    for axis in ('X','Y'):
        for stage in ('train','prod'):
            p=OUT/f'RECEIVED_{axis}_{stage}.json'
            if not p.exists():
                assert not complete;continue
            envelope=json.loads(p.read_text())
            for key in ('campaign','purpose','metadata_identity','control_identity'):assert envelope[key]==source[key]
            rows=envelope['rows'];planned=[('1',64)] if stage=='train' else list(zip(map(str,range(5)),quotas))
            assert [(h,len(b['records'])) for h,b in rows]==planned
            records=[]
            for h,batch in rows:
                n=dict(planned)[h]
                assert {r['trial_id'] for r in batch['records']}==set(map(str,range(n)))
                token=json.dumps(('A1' if stage=='train' else 'B1-OL',axis,stage,h),separators=(',',':'))
                for r in batch['records']:
                    assert r['stream_id']==token and r['weight']=='1'
                    key=(source['campaign'],token,r['trial_id']);assert key not in allkeys;allkeys.add(key)
                    records.append(r);received+=1
            assert len(envelope['costs'])==len(records)
        fit=OUT/f'{axis}_FIT.json';guard=OUT/f'{axis}_GUARD.json'
        if guard.exists():
            f=json.loads(fit.read_text());g=json.loads(guard.read_text())
            assert f['rank']==2 and g['PASS'] is True and F(g['actual_width'])<=6
            assert F(f['width'])==F(g['actual_width'])==2+2*abs(F(f['coefficient']))
    if complete:assert received==448
    checks['received_quota_uniqueness_scope_and_signed_guard']='PASS'
    account=read('SIGNED_ACCOUNTING.json')
    if complete:assert int(F(account['physical_RZ']))==known_rz and account['physical_shots_returned']==448
    checks['physical_RZ_once_matches_all_worker_prices']='PASS'
    record=dict(schema='signed448_saved_integrity_v1',status='PASS',campaign_status=result['status'],checks=checks,
        bits_saved=len(bits),priced_trials=len(prices),known_RZ=known_rz,received_records=received,
        worker_failure_count=len(worker_failures),per_probability_observed_gate_max=max(gate_counts.values(),default=None),
        complete_campaign=complete,scientific_reanalysis=False,new_science_calls=0,
        note='No fresh fit or paired recomputation; saved result statistics are quoted unchanged.')
    (DOC/'SAVED_RECORD_INTEGRITY.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
