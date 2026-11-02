"""Read-only verification of this pilot's known artificial artifacts."""
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile


def check_partition(row):
    total=row['outer_total_ns'];parts=row['exclusive_sum_ns'];residual=row['unattributed_and_observer_residual_ns']
    assert all(p>=0 and r>=0 and p+r==t for p,r,t in zip(parts,residual,total))


def verify(root):
    scaling=root/'measurements/scaling'
    runs=json.loads((scaling/'runs.json').read_text())
    reports={}
    with gzip.open(scaling/'run_reports.jsonl.gz','rt') as file:
        for line in file:
            row=json.loads(line);assert row['run_id'] not in reports;reports[row['run_id']]=row['report']
    assert len(runs)==len(reports)==72
    verified_trials=0
    canonical_files=0
    references={}
    for run in runs:
        r=reports[run['run_id']]
        assert r['fake_claim_status']=='STOPPED' and len(r['pairs'])==24
        semantic={x['path']:x['normalized_sha256'] for x in run['files']}
        previous=references.setdefault(run['workload'],semantic)
        assert semantic==previous
        by_path={x['path']:x for x in run['files']}
        original=run['original']
        for i,pair in enumerate(r['pairs']):
            result=pair if original else pair['outcome']
            task=pair if original else pair['task']
            name=f"{task['phase']}_{task['pass_name']}_{task['arm']}_{task['layer']}.jsonl"
            stream=by_path[name]
            assert stream['sha256']==pair['file_sha256' if original else 'file_sha']
            assert stream['chain']==pair['output_chain_sha256' if original else 'chain']
            assert stream['bytes']==pair['output_bytes']
            n=16 if i<8 else run['settings']['profile_n']
            assert sum(result['counts'].values())==2*n
            assert all(a['N_executed']==n and a['H_bits_supplied']==160*n for a in result['axes'])
            check_partition(pair['reconciliation'])
            if pair.get('fine_reconciliation'):check_partition(pair['fine_reconciliation'])
            verified_trials+=2*n
        if not original:
            assert r['complete'] and r['reserved']==r['started_observed']==r['states']['COMPLETED']==r['planned']
            assert not r['failures'] and len(r['all_pair_states'])==24
            assert r['output']['observed']==r['output']['reserved']
            assert all(r['output']['observed'][k]<=r['output']['limits'][k] for k in ('result','log'))
    archives=[];raw_trials=0;partial_tails=0
    for folder in [scaling,root/'measurements/faults']:
        for path in sorted(folder.glob('*-raw.tar.gz')):
            normal=folder==scaling
            ids=set();info=[]
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    if not member.isfile():continue
                    file=tar.extractfile(member);data=file.read();file.close()
                    info.append(dict(path=member.name,bytes=len(data),sha256=sha256(data).hexdigest()))
                    if not member.name.endswith('.jsonl') or member.name.endswith('/control.jsonl'):continue
                    previous='0'*64
                    for line in data.splitlines(keepends=True):
                        try:row=json.loads(line)
                        except json.JSONDecodeError:
                            assert not normal;partial_tails+=1;break
                        assert row['previous']==previous
                        previous=sha256((previous+row['payload']).encode()).hexdigest()
                        assert row['sha256']==previous
                        payload=json.loads(row['payload'])
                        assert json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True)==row['payload']
                        if 'trial_id' not in payload:continue
                        tid=payload['trial_id'];assert tid not in ids;ids.add(tid)
                        phase,pass_name,arm,layer,axis,index=tid.split('/')
                        assert axis in ('X','Y') and int(index)>=0
                        if normal:
                            workload=path.name.split('-')[0]
                            n=16 if phase=='correctness' else {'light':128,'mixed':64,'heavy':32}[workload]
                            assert int(index)<n
                            assert ('retention' in payload)==(phase=='profile' and int(index)>=16)
                        raw_trials+=1
                    canonical_files+=1
            if normal:
                workload=path.name.split('-')[0]
                assert len(ids)==256+32*{'light':128,'mixed':64,'heavy':32}[workload]
            archives.append(dict(path=str(path.relative_to(root)),files=info))
    faults=json.loads((root/'measurements/faults/summary.json').read_text())
    assert len(faults)>=14
    for row in faults:
        assert row['reserved']<=768 and row['fake_claim_status'] in ('STOPPED','ADMITTED','RESERVED')
        if row['started'] is not None:assert row['started']<=row['reserved']
    return dict(artificial_only=True,measured_runs=len(runs),semantic_equivalence_to_S0=True,
                report_trials_verified=verified_trials,raw_trials_verified=raw_trials,
                canonical_chain_files_verified=canonical_files,partial_tails_retained=partial_tails,
                duplicate_trial_ids=0,cost_partitions_reconciled=True,archives=archives,
                scientific_calls=0,real_claims=0,author_primary_agent_checks=True,independent_review=False)


if __name__=='__main__':
    root=Path(sys.argv[1]);result=verify(root)
    (root/'artifact_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='archives'}))
