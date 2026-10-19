"""Read-only audit of this task's explicit artificial artifacts; no adapter API."""
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

BASE=Path(__file__).resolve().parent


def decode(value):
    if isinstance(value,dict):
        if set(value)=={'int'}:return int(value['int'],16)
        return {k:decode(v)for k,v in value.items()}
    if isinstance(value,list):return [decode(v)for v in value]
    return value


def stream(raw):
    previous='0'*64;rows=[];partial=b''
    for line in raw.splitlines(keepends=True):
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):partial=line;break
        assert set(row)=={'previous','sha256','payload'} and row['previous']==previous
        payload=json.loads(row['payload'])
        assert json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True)==row['payload']
        previous=sha256((previous+row['payload']).encode()).hexdigest();assert previous==row['sha256']
        rows.append(decode(payload))
    return dict(file_sha=sha256(raw).hexdigest(),chain=previous,rows=rows,partial_bytes=len(partial))


def main():
    tests=json.loads((BASE/'tests/summary.json').read_text())
    assert tests['total']==481 and tests['protected_baseline']==356 and tests['integration']==125
    assert all(x['fail']==x['error']==x['skip']==0 and x['science_imports']==[]for x in tests['suites'])
    benchmarks=json.loads((BASE/'measurements/benchmark/runs.json').read_text());assert len(benchmarks)==60
    reports={}
    for suite in ('benchmark','faults'):
        with gzip.open(BASE/'measurements'/suite/'reports.jsonl.gz','rt')as file:
            for line in file:
                row=json.loads(line);report=row['report'];reports[suite+'/'+row['run_id']]=report
                if report['schema']=='b3_parallel_crash_observation_v3':
                    assert report['total_cpu_ns']is None and report['complete']is False
                    continue
                assert report['total_cpu_ns']==report['parent_cpu_ns']+report['worker_cpu_ns']
                assert sum(report['states'].values())==report['planned']
                for pair in report['pairs']:
                    for key in ('reconciliation','fine_reconciliation'):
                        rec=pair.get(key)
                        if rec:
                            assert rec['unknown_bit_operations']is None
                            assert all(x>=0 for x in rec['unattributed_and_observer_residual_ns'])
                            assert [a+b for a,b in zip(rec['exclusive_sum_ns'],rec['unattributed_and_observer_residual_ns'])]==rec['outer_total_ns']
                if suite=='benchmark':
                    assert report['complete'] and len(report['pairs'])==24 and not report['log_errors']
                    assert report['output']['reserved']==report['output']['observed']
                    assert report['fake_claim_status']=='STOPPED'
    reference={}
    for row in benchmarks:
        assert row['semantics']==reference.setdefault(row['workload'],row['semantics'])
        assert row['total_cpu_s']==row['parent_cpu_s']+row['worker_cpu_s'] or abs(row['total_cpu_s']-row['parent_cpu_s']-row['worker_cpu_s'])<1e-9
    archive_records=[];physical_trial_rows=0;partial_files=0
    for suite in ('benchmark','faults'):
        for archive in sorted((BASE/'measurements'/suite).glob('*-raw.tar.gz')):
            run_id=archive.name[:-len('-raw.tar.gz')];report=reports[suite+'/'+run_id];files={}
            with tarfile.open(archive,'r:gz')as tar:
                for member in tar.getmembers():
                    path=Path(member.name)
                    assert path.parts[0]=='artificial-output' and '..'not in path.parts and not path.is_absolute()
                    assert not member.issym() and not member.islnk()
                    if member.isfile():files[path.name]=tar.extractfile(member).read()
            assert 'trial_states.bin'in files
            states=files['trial_states.bin']
            if report['schema']=='b3_parallel_crash_observation_v3':
                assert states.count(1)==1
            else:
                counts=Counter(states)
                for index,state in enumerate(('NOT_STARTED','START_RESERVED','STARTED','IN_FLIGHT','COMPLETED','FAILED','CANCELLED_BEFORE_START','TERMINATED_WITH_UNKNOWN_RESULT')):
                    assert counts[index]==report['states'][state]
            checks=[]
            pairs={p['task']['key']+'.jsonl':p for p in report.get('pairs',[])}
            for name,raw in files.items():
                if not name.endswith('.jsonl')or name=='control.jsonl':continue
                parsed=stream(raw);partial_files+=bool(parsed['partial_bytes'])
                if name!='summary.jsonl':
                    ids=[r['trial_id']for r in parsed['rows']if 'trial_id'in r]
                    assert len(ids)==len(set(ids));physical_trial_rows+=len(ids)
                pair=pairs.get(name)
                if pair and pair['status']!='LOST':
                    assert pair['output_bytes']==len(raw)
                    if pair['stream_prefix']['file_hash_covers_written_bytes']:assert pair['file_sha']==parsed['file_sha']
                    assert pair['chain']==parsed['chain']
                checks.append(dict(path=name,bytes=len(raw),sha256=parsed['file_sha'],chain=parsed['chain'],complete_envelopes=len(parsed['rows']),partial_bytes=parsed['partial_bytes']))
            if report.get('complete'):
                summary=stream(files['summary.jsonl']);events=[json.loads(x)for x in files['control.jsonl'].splitlines()]
                receipt=events[-1]
                assert receipt['event']=='COMPLETION_RECEIPT' and receipt['summary_sha']==summary['file_sha'] and receipt['summary_chain']==summary['chain']
                assert summary['rows'][0]['binding']['formal_digest']==receipt['formal_digest']==report['binding']['formal_digest']
                assert summary['rows'][0]['planned']==report['planned']
            archive_records.append(dict(path=str(archive.relative_to(BASE)),bytes=archive.stat().st_size,
                                        sha256=sha256(archive.read_bytes()).hexdigest(),streams=checks))
    value=dict(all_assertions_passed=True,tests=481,protected_baseline_tests=356,integration_tests=125,
               benchmark_runs=60,fault_reports=len(reports)-60,archives=len(archive_records),
               physical_trial_rows=physical_trial_rows,partial_files=partial_files,records=archive_records,
               scientific_calls=0,real_authorizations_grants_claims=0,read_only_explicit_artificial_inputs=True)
    (BASE/'artifact_verification.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items()if k!='records'}))


if __name__=='__main__':main()
