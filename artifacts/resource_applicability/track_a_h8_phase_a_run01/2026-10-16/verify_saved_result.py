"""Read-only public metadata/count/hash checks; never evaluate scientific actions."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def read(name):return json.loads((HERE/name).read_text())


def main():
    checks=0
    summary=read('result_summary_v1.json');terminal=read('execution_terminal_v1.json')
    launch=read('resource_and_cleanup_v1.json');binding=read('execution_binding_v1.json')
    assert summary['status']==terminal['status']==launch['status']=='COMPLETE_MANDATORY_STOP';checks+=1
    assert summary['SOURCE']==terminal['SOURCE']==binding['SOURCE'];checks+=1
    raw=gzip.decompress((HERE/'H8_signal_complete_v1.json.gz').read_bytes());receipt=read('signal_export_receipt_v1.json')
    assert len(raw)==receipt['original_json_bytes'] and hashlib.sha256(raw).hexdigest()==receipt['original_json_sha256'];checks+=1
    result=json.loads(raw);rows=result['signals'];expected=set()
    for m in ('B0','B0I'):
        for f,ps in (('2nd',[p for p in range(8,21) if p!=14]),('4th',range(15,22))):
            expected.update(f'{m}_p{p}_q{q}_{f}' for p in ps for q in (1,2,4,8))
    assert len(rows)==152 and {r['cell']['id'] for r in rows}==expected;checks+=1
    assert all(r['reference']==rows[0]['reference'] for r in rows);checks+=1
    for condition in summary['eligibility_counts']:
        counts=collections.Counter()
        for row in rows:
            v=row['precision_rows'][str(condition['epsilon'])][str(condition['factor'])]
            counts[row['cell']['method']+'_'+row['cell']['formula']]+=bool(v['cost_acquisition_eligible'])
        assert dict(counts)==condition['eligible'] and sum(counts.values())==condition['total']
    checks+=1
    actual=collections.Counter(terminal['call_counts'])
    for value in terminal['settlements'].values():actual.update(value)
    assert dict(actual)==summary['actual_calls'];checks+=1
    assert set(terminal['settlements'])==set(terminal['irrevocable_reservations'])
    for task,counts in terminal['settlements'].items():
        cap=terminal['irrevocable_reservations'][task]
        assert all(value<=cap[name] for name,value in counts.items())
    checks+=1
    worker=read('owned_worker_receipts_v1.json')
    assert len(worker['workers'])==35 and all(w['reaped'] and w['exit_code']==0 for w in worker['workers']);checks+=1
    assert worker['original_result_hash_missing']==['prefix_19']
    assert sum(not w['original_result_sha256_present'] for w in worker['workers'])==1;checks+=1
    assert launch['all_reaped'] and not launch['cleanup_errors'] and launch['native_exit_code']==0
    assert launch['max_monitor_gap_seconds']<=5;checks+=1
    assert not terminal['next_stage_authorized'] and not summary['next_stage_authorized']
    assert summary['new_compile']==summary['new_cost_model_fit']==summary['new_cost_reaccount']==0;checks+=1
    count=0
    for row in read('publication_files_v1.json')['files']:
        path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
        data=(ROOT/path).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];count+=1
    checks+=1
    print(json.dumps(dict(status='PASS_SAVED_METADATA_ONLY',checks=checks,public_file_hashes=count,
                         scientific_actions=0,new_fit=0,new_compile=0)))


if __name__=='__main__':main()
