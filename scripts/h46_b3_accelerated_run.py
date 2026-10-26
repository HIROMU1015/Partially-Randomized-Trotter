"""Adopt exact canonical-byte acceleration for future recovery/confirmation.

The live run03 source and workers remain unchanged. Validation uses fixed saved
rows, seeds and the original full compiler in separate directories. Its paired
observations never enter exploration or confirmation statistics.
"""
from pathlib import Path
import json
import sys
import h46_b3 as b
import h46_b3_trace_recovery as traced
from h46_b3_durable import save

UNIT='prt-h46-b3-20261026-run04-fast.service'
SCRIPT=Path(__file__).resolve()
SOURCES=('scripts/h46_b3_fast_fingerprint.py','scripts/h46_b3_accelerated_run.py',
         'scripts/h46_b3_trace_recovery.py')


def install():
    b.setup()
    import h46_b3_fast_fingerprint as fast
    from trottertracks.resource_applicability import h6_hash_reuse_v1
    h6_hash_reuse_v1.use_fingerprint_reuse=fast.use_fast_fingerprint


def validate(identity,mode):
    import faulthandler
    faulthandler.enable(all_threads=True)
    traced.check_sources()
    row=traced.fixed_row(identity,0)
    reference=b.read(b.PRIVATE/'cost/B3_EXPLORATION'/identity/'0/result.json')
    traced.check_trial(reference,row,0)
    native=b.setup();native.save=save
    if mode=='fast': install()
    native.CONTEXT[row['system']]=b.native_context(row['system'],[0])
    native.PRIVATE=b.PRIVATE/'acceleration/validation'/mode
    native.PRIVATE.mkdir(parents=True,exist_ok=True)
    result=native.cost_trial((row,'B3_EXPLORATION',0))
    traced.check_trial(result,row,0)
    directory=native.PRIVATE/'cost/B3_EXPLORATION'/identity/'0'
    original=b.PRIVATE/'cost/B3_EXPLORATION'/identity/'0/events_PRIVATE.json'
    events_equal=traced.recovery.event_digest(original)==traced.recovery.event_digest(directory/'events_PRIVATE.json')
    hashes=[p.stem for p in (native.PRIVATE/'compiled_cache').glob('*.json') if len(p.stem)==64]
    fingerprints_match=all((b.PRIVATE/'compiled_cache'/(fp+'.json')).exists() for fp in hashes)
    if result['prices']!=reference['prices'] or not events_equal or not fingerprints_match:
        save(directory/'STOP.json',dict(reason='ACCELERATION_REFERENCE_MISMATCH'))
        raise ValueError('ACCELERATION_REFERENCE_MISMATCH')
    evidence=dict(status='PASS',mode=mode,system=row['system'],candidate=identity,s=row['s'],
        wall_seconds=result['wall_seconds'],peak_RSS_bytes=result['peak_RSS_bytes'],
        all_prices_equal=True,events_equal=True,fingerprints_equal=True,
        scientific_checks=result['checks'],new_compile_axes=2-result['cache_hits'],
        result_sha256=b.signal.sha(directory/'result.json'),additional_statistical_samples=0,
        sources={name:b.signal.sha(b.ROOT/name) for name in SOURCES})
    save(directory/'acceleration_check.json',evidence)
    print(json.dumps(evidence),flush=True)


def validation_summary():
    rows=[]
    for system,identity in [('H4','95a12d1758489893ddd1'),('H6','5d859631f95f7543343e')]:
        pair={}
        for mode in ('original','fast'):
            path=b.PRIVATE/'acceleration/validation'/mode/'cost/B3_EXPLORATION'/identity/'0'
            check=b.read(path/'acceleration_check.json')
            if check['status']!='PASS' or not all(check[k] for k in ('all_prices_equal','events_equal','fingerprints_equal')):
                raise ValueError('ACCELERATION_NOT_VERIFIED')
            if b.signal.sha(path/'result.json')!=check['result_sha256']:
                raise ValueError('VALIDATION_RESULT_CHANGED')
            if check['sources']!={name:b.signal.sha(b.ROOT/name) for name in SOURCES}:
                raise ValueError('VALIDATED_SOURCE_CHANGED')
            pair[mode]=check
        rows.append(dict(system=system,candidate=identity,original=pair['original'],fast=pair['fast'],
            time_speedup=pair['original']['wall_seconds']/pair['fast']['wall_seconds'],
            peak_RSS_ratio=pair['original']['peak_RSS_bytes']/pair['fast']['peak_RSS_bytes']))
    result=dict(status='PAIRED_FIXED_INPUT_IMPLEMENTATION_VALIDATION_PASS',rows=rows,
        canonical_policy='qiskit_recursive_numeric_circuit_v2_UNCHANGED',
        compiler_settings_unchanged=True,additional_statistical_samples=0,
        sources={name:b.signal.sha(b.ROOT/name) for name in SOURCES},
        scope='Two saved representative rows; whole-study speedup is not established')
    save(b.PUBLIC/'acceleration_validation.json',result)
    return result


def configure():
    traced.SCRIPT=SCRIPT;traced.UNIT=UNIT
    original=traced.preserve_and_adopt
    def preserve(progress):
        original(progress)
        path=b.PUBLIC/'interruption_run03.json';record=b.read(path)
        record['helper_source']=str(SCRIPT.relative_to(b.ROOT))
        record['helper_source_sha256']=b.signal.sha(SCRIPT)
        record['runtime_optimization_sources']={name:b.signal.sha(b.ROOT/name) for name in SOURCES}
        record['runtime_change']='Call-local canonical-byte hashing only; original compiler and all scientific checks retained'
        record['acceleration_validation']='acceleration_validation.json'
        save(path,record)
    traced.preserve_and_adopt=preserve


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['validate','summary','worker','continue','complete'])
    parser.add_argument('--candidate');parser.add_argument('--mode',choices=['original','fast']);parser.add_argument('--job')
    args=parser.parse_args()
    if args.command=='validate': validate(args.candidate,args.mode)
    elif args.command=='summary': print(json.dumps(validation_summary()),flush=True)
    elif args.command=='worker':
        install();traced.worker(args.job)
    else:
        configure()
        if args.command=='continue':
            validation_summary();traced.continuation()
        else: traced.completion()
