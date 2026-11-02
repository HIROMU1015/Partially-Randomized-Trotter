"""Summarize only the newly authorized P5 trajectories, before its terminal."""
from collections import Counter
from fractions import Fraction as F
import gzip
import json
import math
import time
from common import HERE,read,save,require,digest

def summarize():
    from bridge import connect
    connect()
    from trottertracks.algorithm_codesign.g6_f3_contract import Evidence,_unpack
    out=HERE/'result';seen=set();logical=Counter();raw=Counter();reduced=Counter();squares=[];maximum=F(0)
    started=time.perf_counter(),time.process_time();receipts=[]
    for batch in range(32):
        rec=read(out/'receipts'/f'batch_{batch:03}.json');receipts.append(rec)
        path=out/'batches'/f'batch_{batch:03}.jsonl.gz';require(digest(path)==rec['sha256'],'BATCH_SHA')
        for line in gzip.decompress(path.read_bytes()).splitlines():
            row=json.loads(line);record=Evidence.create(_unpack(row['exact']))
            require(record.digest==row['exact_digest'],'PATH_EVIDENCE_DIGEST')
            d=record.read();trial=d['trial'];require(trial not in seen and trial//32==batch,'UNIQUE_TRIAL')
            require(d['kind']=='ACCEPTED' and d['measurement_performed'] is False,'P5_NO_FAKE_OBSERVATIONS')
            require(all(x['width']==2048 for s in d['microsteps'] for x in s['bits_draws']),'BIT_WIDTH')
            require(len(d['microsteps'])==4,'FOUR_STEPS')
            seen.add(trial);squares.append(row['observed_weight_square_binary64']);maximum=max(maximum,d['weight'])
            raw.update(str(i) for i in row['raw_degrees']);reduced.update(str(i) for i in row['reduced_lengths'])
            for key in ('signed_Q','child_rotations','controlled_gate_instructions','logical_1Q','logical_2Q','logical_multi_qubit','instructions','depth'):
                logical[key]+=row['logical'][key]
    require(seen==set(range(1024)),'ALL_PLANNED_TRIALS_REQUIRED')
    producer=read(out/'producer_summary.json');stages={}
    for r in receipts:
        for key,value in r['stages'].items():stages.setdefault(key,Counter()).update(value)
    owner_exit=read(out/'producer_supervision/summary.json')['exits'][0]
    worker_exits=read(out/'worker_supervision/summary.json')['exits']
    summary=dict(method='closed_P5_full',scope='UNCERTIFIED_EXPLORATORY_TAIL_ONLY_PLUS_SCALAR',planned=1024,completed=1024,
        microsteps=4096,accepted_paths=1024,valid_zero_paths=0,accepted_fraction_observed=1.,
        raw_degree_counts=dict(raw),reduced_length_counts=dict(reduced),
        empirical_weight_second_moment=math.fsum(squares)/1024,maximum_observed_weight=float(maximum),
        maximum_observed_weight_exact=Evidence.create(maximum).to_dict(),moments_are_empirical_not_certified=True,
        logical_totals=dict(logical),logical_per_classical_trial={k:v/1024 for k,v in logical.items()},
        logical_per_accepted_path={k:v/1024 for k,v in logical.items()},workspace_beyond_system=1,
        quantum_call_equivalent_circuits=1024,zero_avoided_circuits=0,producer=producer,
        logical_worker_stages={k:dict(v) for k,v in stages.items()},
        producer_wait4_CPU_seconds=owner_exit['user_CPU']+owner_exit['system_CPU'],
        logical_workers_wait4_CPU_seconds=math.fsum(v['user_CPU']+v['system_CPU'] for v in worker_exits),
        max_logical_worker_RSS_KiB=max(v['max_RSS_KiB'] for v in worker_exits),
        saved_bytes_before_summary=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        technical_failure=None,complete=True,coherent_signal_estimator=None,native_cost='UNKNOWN',
        deterministic_halves='UNKNOWN',state_preparation='UNKNOWN',readout='UNKNOWN',sufficient_N=None,
        summary_CPU_seconds=time.process_time()-started[1],summary_wall_seconds=time.perf_counter()-started[0])
    save(out/'summary.json',summary);return summary
