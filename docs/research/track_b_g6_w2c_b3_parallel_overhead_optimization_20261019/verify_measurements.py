"""Read-only audit of the known artificial measurement directories."""
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile


def partition(value):
    assert all(p>=0 and r>=0 and p+r==t for p,r,t in zip(value['exclusive_sum_ns'],
        value['unattributed_and_observer_residual_ns'],value['outer_total_ns']))


def main():
    root=Path(__file__).parent;references={};count=0;normal_trials=0
    for folder,expected in [('benchmark',99),('diagnostic',27)]:
        d=root/'measurements'/folder;runs=json.loads((d/'runs.json').read_text())
        assert len(runs)==expected
        with gzip.open(d/'reports.jsonl.gz','rt') as file:reports={x['run_id']:x['report'] for x in map(json.loads,file)}
        assert len(reports)==expected
        for run in runs:
            assert run['semantic']==references.setdefault(run['workload'],run['semantic'])
            report=reports[run['run_id']];assert report['fake_claim_status']=='STOPPED'
            assert len(report['pairs'])==24
            files={f['path']:f for f in run['files']}
            for i,pair in enumerate(report['pairs']):
                result=pair if run['original'] else pair['outcome']
                task=pair if run['original'] else pair['task']
                key=f"{task['phase']}_{task['pass_name']}_{task['arm']}_{task['layer']}.jsonl"
                assert files[key]['sha256']==pair['file_sha256' if run['original'] else 'file_sha']
                n=16 if i<8 else run['N'];assert sum(result['counts'].values())==2*n
                assert all(a['H_bits_supplied']==160*n and a['N_executed']==n for a in result['axes'])
                partition(pair['reconciliation'])
                if pair.get('fine_reconciliation'):partition(pair['fine_reconciliation'])
                normal_trials+=2*n
            if not run['original']:
                assert report['complete'] and report['states']['COMPLETED']==report['reserved']==report['started_observed']==report['planned']
                assert report['output']['reserved']==report['output']['observed']
                assert all(report['output']['reserved'][k]<=report['output']['limits'][k] for k in ('log','result'))
                expected_messages=(3*report['planned']+72) if run['transport']=='coalesced' else (5*report['planned']+24)
                if run['config']=='before_exact8':assert report.get('parent_messages') is None
                else:assert sum(report['parent_messages'].values())==expected_messages
            count+=1
    archives=[];raw_rows=partial=chain_files=0
    for path in sorted((root/'measurements').glob('*/*-raw.tar.gz')):
        ids=set();files=[]
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.isfile():continue
                f=tar.extractfile(member);data=f.read();f.close()
                files.append(dict(path=member.name,bytes=len(data),sha256=sha256(data).hexdigest()))
                if not member.name.endswith('.jsonl') or member.name.endswith('/control.jsonl'):continue
                previous='0'*64
                for line in data.splitlines(keepends=True):
                    try:row=json.loads(line)
                    except json.JSONDecodeError:
                        assert path.parent.name=='faults';partial+=1;break
                    assert row['previous']==previous
                    previous=sha256((previous+row['payload']).encode()).hexdigest();assert row['sha256']==previous
                    payload=json.loads(row['payload'])
                    assert json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True)==row['payload']
                    if 'trial_id' in payload:
                        assert payload['trial_id'] not in ids;ids.add(payload['trial_id']);raw_rows+=1
                chain_files+=1
        archives.append(dict(path=str(path.relative_to(root)),files=files))
    faults=json.loads((root/'measurements/faults/summary.json').read_text());assert len(faults)==17
    for f in faults:
        assert f['reserved']<=768
        if f['started'] is not None:assert f['started']<=f['reserved']
    value=dict(artificial_only=True,performance_runs=99,diagnostic_runs=27,normal_report_trials=normal_trials,
        all_stream_semantics_equal_S0=True,raw_trial_rows=raw_rows,raw_rows_may_be_uncommitted=True,
        canonical_chain_files=chain_files,partial_tails_retained=partial,duplicate_ids=0,
        counts_bits_hashes_cost_partitions_verified=True,archives=archives,scientific_calls=0,real_claims=0)
    (root/'artifact_verification.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='archives'}))


if __name__=='__main__':main()
