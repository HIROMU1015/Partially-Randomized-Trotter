"""Post-run stdlib metadata audit; no signal, circuit, sampler or test execution.

This publication helper was added after the stopped run. It is not part of the
frozen scientific worker and must not be described as an execution-time source.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json


def digest(value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def audit(raw):
    def read(name):return json.loads((raw/name).read_text())
    manifest=read('frozen_pilot.json');plan=manifest['plan'];terminal=read('terminal_status.json')
    assert terminal['status']=='H6_TECHNICAL_PILOT_STOP'
    assert terminal['reason']=='PHASE_WALL_CAP:wrapper_cost'
    assert terminal['worker_terminal'] is None and not (raw/'worker_terminal.json').exists()
    boundary=terminal['latest_progress'];assert boundary['correctness_completed']==7
    assert boundary['compiled_wrappers']==32 and boundary['calls_attempted']['compile']==32
    expected={c['id']+'_correctness.json' for c in plan['cells']}
    assert expected=={p.name for p in raw.glob('H6*_correctness.json')}
    evidence=[]
    for c in plan['cells']:
        row=read(c['id']+'_correctness.json');assert row['cell']==c
        assert row['evidence_kind']=='TECHNICAL_AGREEMENT' and row['certified'] is False
        assert row['N'] is None and row['G'] is None and row['accuracy_eligibility']=='UNDETERMINED'
        evidence.append({'cell_id':c['id'],'saved_log_B':row['log_B'],
            'saved_oracle_signal_discrepancies':row['oracle_signal_discrepancies'],
            'saved_total_signal_discrepancy':row['error_decomposition']['total_absolute']})
    wrappers=sorted(raw.glob('wrapper_[0-9][0-9].json'))
    assert [p.name for p in wrappers]==['wrapper_%02d.json'%i for i in range(32)]
    counts=Counter();compiler_hashes=set();trajectories={}
    for i,p in enumerate(wrappers):
        row=json.loads(p.read_text());task=row['task'];assert task==plan['wrapper_tasks'][i]
        name=task['cell_id']+'_rep%d_trajectory.json'%task['replica'];tr=read(name)
        assert tr['seed']==task['seed']
        assert row['event_digest']==tr['event_digest']==digest(tr['events'])
        assert row['N'] is None and row['G'] is None
        assert row['cost_scope']=='measured Hadamard wrapper; no state preparation'
        assert set(row['metrics'])=={'rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size'}
        assert all(type(v) is int and v>=0 for v in row['metrics'].values())
        counts[(task['cell_id'],task['replica'])]+=1;compiler_hashes.add(row['compiler_hash'])
        trajectories[name]={'seed':tr['seed'],'event_digest':tr['event_digest'],
            'saved_control_measurement_max_error':tr['control_measurement_max_error'],
            'has_random_events':tr['events'] is not None}
    assert len(counts)==8 and set(counts.values())=={4} and len(compiler_hashes)==1
    assert len(trajectories)==8 and sum(r['has_random_events'] for r in trajectories.values())==3
    assert not (raw/'cost_summary.json').exists()
    result={'schema':'track_a_h6_saved_partial_bytes_audit_v2','status':'SAVED_PARTIAL_METADATA_BINDING_PASS',
        'audit_scope':'saved JSON structure/task/event/hash metadata only; no scientific result recomputed',
        'helper_is_post_run_not_frozen_worker_source':True,'source_commit':manifest['source_commit'],
        'correctness_records':7,'wrapper_records':32,'complete_paired_replica_groups':8,
        'saved_random_trajectory_records':3,'saved_trajectory_occurrences':6,
        'calls_attempted':boundary['calls_attempted'],'counter_scope':'last saved boundary only; interrupted remainder unknown',
        'terminal_reason':terminal['reason'],'cost_summary_missing':True,'worker_terminal_missing':True,
        'remaining_wrapper_tasks':plan['wrapper_tasks'][32:],'compiler_hashes':sorted(compiler_hashes),
        'saved_cell_evidence':evidence,'saved_trajectory_metadata':trajectories,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=audit(a.input)
    result['helper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(result['status'])
