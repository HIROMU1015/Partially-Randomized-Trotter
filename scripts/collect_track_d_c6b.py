#!/usr/bin/env python3
"""Collect stopped C6b scalar evidence and local-payload hashes; no science calls.

Does not sample/build/compile/check circuits, refit, or recompute price statistics.
The existing runner owns all numerical processing. Its outputs stay immutable.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/track_d_c6b_q8/2026-11-07/run01'
SOURCE = '23725fc6f6d6d0c2bec93e31e32074711804e250'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    global OUT, SOURCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accelerated-completion',action='store_true')
    args=parser.parse_args()
    if args.accelerated_completion:
        OUT=ROOT/'artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion'
        SOURCE=read(OUT/'SOURCE.json')['execution_source']
    result = read(OUT/'result.json'); stop = read(OUT/'STOP.json')
    if not stop['mandatory_stop'] or result['source_commit'] != SOURCE:
        raise ValueError('EXISTING_SOURCE_FINAL_RESULT_STOP_REQUIRED')
    if (OUT/'verification_summary.json').exists() or (OUT/'payload_indices').exists():
        raise ValueError('FRESH_COLLECTION_ONLY_NO_OVERWRITE')
    planned = [f'q8_sample{i:02d}' for i in range(32)]
    if result['planned_ids'] != planned:
        raise ValueError('FIXED_32_IDS')
    indices = OUT/'payload_indices'; indices.mkdir()
    bindings = read(OUT/'event_bindings.json'); plan = read(OUT/'seed_plan.json')
    inherited=read(OUT/'inherited_evidence.json') if args.accelerated_completion else None
    reused_raw=set(inherited['raw_reused']) if inherited else set()
    reused_wrappers={tuple(x) for x in inherited['wrappers_reused']} if inherited else set()
    if len(bindings) != 32 or plan != read(ROOT/'artifacts/track_d_c6b_q8/2026-11-07/technical/seed_plan.json'):
        raise ValueError('COMPLETE_SAMPLING_BINDING')
    all_rows=[]; raw_rows=[]; post_rows=[]; paired_rows=[]; compile_calls=0
    for binding, seed in zip(bindings, plan['generation_order']):
        sid = binding['sample_id']; local = OUT/'local'/sid
        event = OUT/'local/events'/(sid+'.json')
        if sha(event) != binding['record_sha256']:
            raise ValueError('EVENT_RECORD_CHANGED:'+sid)
        record=read(event)
        sampling_source=result.get('original_source_commit',SOURCE)
        if record['execution_source']!=sampling_source or any(record[k]!=v for k,v in seed.items()):
            raise ValueError('EVENT_SOURCE_SEED_BINDING:'+sid)
        events_hash=hashlib.sha256(json.dumps(record['ordered_events'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if events_hash != binding['ordered_events_sha256']:
            raise ValueError('ORDERED_EVENT_DIGEST:'+sid)
        files={'events.json':event}
        for pattern in ['circuits/*.qpy','references.npz','*_observable.npz']:
            files.update({str(p.relative_to(local)):p for p in local.glob(pattern)})
        members=[]
        for n,p in sorted(files.items()):
            generated_by=sampling_source if n=='events.json' else (sampling_source if sid in reused_raw else SOURCE)
            for method in ['symmetric_directional','reference']:
                for axis in ['X','Y']:
                    if n in ['circuits/'+method+'_'+axis+'_compiled.qpy',method+'_'+axis+'_observable.npz']:
                        generated_by=sampling_source if (sid,method,axis) in reused_wrappers else SOURCE
            members.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p),generated_by_source=generated_by))
        write_new(indices/(sid+'.json'),dict(sample_id=sid,execution_source=SOURCE,members=members,
            execution_source_role='completion/evaluation; actual member generators recorded individually',
            originals_local_only=True,raw_payload_uploaded=False,
            location='task-local event and per-path circuit/reference directories; not downloadable from GitHub'))
        raw=local/'precompile_checks.json'
        if raw.exists():
            value=read(raw); raw_rows.append(dict(sample_id=sid,status=value['status'],
                checks=value.get('checks',[]),reference_leakage=value.get('reference_leakage'),
                input_basis_columns=value.get('input_basis_columns')))
        post_status={}; compile_status={}
        for method in ['symmetric_directional','reference']:
            for axis in ['X','Y']:
                name=method+'_'+axis
                marker=local/(name+'_compile.json')
                if marker.exists():
                    value=read(marker);compile_calls+=value['compile_calls'];compile_status[name]=value['status']
                check=local/(name+'_check.json')
                if check.exists():
                    value=read(check);post_status[name]=value['status']
                    post_rows.append(dict(sample_id=sid,method=method,axis=axis,status=value['status'],
                        checks=value.get('checks',[]),reference_leakage=value.get('reference_leakage')))
        paired=local/'paired_observables.json'
        if paired.exists():
            paired_rows.append(dict(sample_id=sid,checks=read(paired)))
        all_rows.append(dict(sample_id=sid,payload_members=len(members),
            raw_status=read(raw)['status'] if raw.exists() else 'MISSING',
            compile_status=compile_status,postcheck_status=post_status,paired_check_present=paired.exists()))
    recorded_calls=result.get('compile_calls_across_original_and_completion',result.get('compile_calls_started'))
    if compile_calls != recorded_calls or compile_calls > 128:
        raise ValueError('COMPILE_CALL_ACCOUNTING')
    if result['status']=='TECHNICAL_PASS_MANDATORY_STOP':
        if (len(result['completed_ids'])!=32 or result['missing_or_failed_ids'] or
                len(raw_rows)!=32 or len(post_rows)!=128 or len(paired_rows)!=32 or
                any(r['status']!='PASS' for r in raw_rows+post_rows) or
                any(r['payload_members']!=20 for r in all_rows)):
            raise ValueError('COMPLETE_TECHNICAL_EVIDENCE_REQUIRED')
    def max_check(rows, field):
        return max((check[field] for row in rows for check in row['checks'] if field in check), default=None)
    leakage = [row['reference_leakage'] for row in post_rows if row['reference_leakage'] is not None]
    write_new(OUT/'verification_summary.json',dict(status=result['status'],execution_source=SOURCE,
        raw_paths=all_rows,raw_PASS=sum(r['status']=='PASS' for r in raw_rows),
        compiled_postcheck_PASS=sum(r['status']=='PASS' for r in post_rows),compile_calls=compile_calls,
        paired_paths=len(paired_rows),raw_checks=raw_rows,compiled_checks=post_rows,paired_checks=paired_rows,
        raw_max_component=max_check(raw_rows,'max_entry'),compiled_max_component=max_check(post_rows,'max_entry'),
        compiled_observable_max_frobenius=max((check['frobenius'] for row in post_rows for check in row['checks']
            if check['label'] in ['compiled_vs_CAR_observable','compiled_vs_raw_observable']), default=None),
        reference_leakage=leakage,all_event_bindings_match=True,
        collector_source_sha256=sha(Path(__file__)),
        numerical_processing='existing runner outputs only; collector does not recompute sample statistics or circuit checks',
        scientific_calls_added=0))
    write_new(OUT/'payload_publication.json',dict(raw_payload_uploaded=False,originals_preserved_locally=True,
        member_index_directory='payload_indices',paths=len(all_rows),members=sum(r['payload_members'] for r in all_rows),
        published='SOURCE, PLAN, seed/model definitions, scalar checks, resource statistics, hashes and STOP',
        excluded='raw events, QPY/NPZ, operational CPU/RAM/runtime/thread values, private inputs and credentials',
        raw_external_publication_requires_specific_user_permission=True))
    # Protected execution source and inputs are checked by the runner itself;
    # independently verify that publishing/collection has not changed SOURCE files.
    source=read(OUT/'SOURCE.json');changed=[]
    for name,want in {**source['C6b_source'],**source.get('accelerated_source',{})}.items():
        if sha(ROOT/name)!=want:changed.append(name)
    if changed:
        raise ValueError('FIXED_EXECUTION_SOURCE_CHANGED:'+repr(changed))
    print(json.dumps(dict(status=result['status'],raw=len(raw_rows),postchecks=len(post_rows),
                         payload_members=sum(r['payload_members'] for r in all_rows),source_unchanged=True)))


if __name__=='__main__':
    main()
