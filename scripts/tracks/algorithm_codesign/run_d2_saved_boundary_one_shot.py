#!/usr/bin/env python3
"""Future D2 cost-only pair evaluation, explicitly gated before saved IR load."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_development.boundary_launch import verify_launch
from trottertracks.algorithm_codesign.rte_development.launch import Guard,consume_marker,check_protected
from trottertracks.algorithm_codesign.rte_development.evidence import digest,atomic_json,EvidenceWriter

CONTRACT=ROOT/'artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json'


def execute(source):
    c,a,head=verify_launch(ROOT,CONTRACT,source)
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
    runtime=verify_runtime(ROOT,c)
    if digest(Path(sys.executable).resolve())!=c['runtime_executable_sha256']:
        raise PermissionError('fixed runtime required')
    if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise PermissionError('numeric threads must be one')
    from fractions import Fraction as F
    from trottertracks.algorithm_codesign.rte_development.boundary import aggregate_pairs
    directory=ROOT/c['result_directory'];guard=Guard(c['caps'])
    identity=dict(source_commit=source,execution_HEAD=head,contract_sha256=digest(CONTRACT),
                  authorization_sha256=digest(ROOT/c['authorization_path']))
    marker=None; terminal=None
    try:
        with guard:
            marker,mark=consume_marker(directory,dict(kind='D2_BOUNDARY_ONE_SHOT_CONSUMED',
                   **identity,runs=1,retries=0,mandatory_STOP=True))
            identity['marker_sha256']=mark['sha256'];guard.directory=directory
            writer=EvidenceWriter(directory,c['caps']['output_bytes'],guard.check,c['caps']['review_chunk_bytes'])
            writer.write('preflight.json',dict(runtime=runtime,**identity,
                         source_manifest_sha256=digest(ROOT/c['source_manifest'])))
            base=ROOT/c['saved_result_directory']
            cache=json.loads((base/'synthesis_cache.json').read_text())
            summaries=[];total=0
            for spec in c['registered_rows']:
                row=json.loads((base/spec['row_path']).read_text())
                if row['steps']!=2 or row['arm']!=spec['arm'] or row['degree']!=5:
                    raise ArithmeticError('fixed row identity')
                path=base/row['events']['path']
                if digest(path)!=row['events']['sha256']:
                    raise PermissionError('saved event identity')
                events=json.loads(path.read_text())
                v=aggregate_pairs(events,cache,guard.check,spec['positive_pair_bound'],c['caps']['denominator_bits'])
                if v['positive_pairs']!=spec['positive_pair_bound']:
                    raise ArithmeticError('incomplete registered positive support')
                for k in ('T','CX','1Q'):
                    if v['additive_per_whole_trial'][k]!=F(row['per_whole_trial_native'][k]):
                        raise ArithmeticError('D1 additive anchor mismatch')
                    if not 0<=v['after_per_whole_trial'][k]<=v['additive_per_whole_trial'][k]:
                        raise ArithmeticError('non-increasing exact cancellation violated')
                if v['acceptance_squared']!=F(row['reference_law_diagnostic_only']['acceptance'])**2:
                    raise ArithmeticError('law changed')
                N=row['budget']['N_per_axis']
                v.update(arm=row['arm'],steps=2,N_per_axis=N,budget_unchanged=row['budget'],
                         new_expected_native_intercept={k:2*N*x for k,x in v['after_per_whole_trial'].items()},
                         original_expected_native_intercept=row['expected_native_intercept'],
                         prep_readout_slope_unchanged=row['prep_readout_slope'],
                         old_accepted_tail_native_retained=row['accepted_tail_native'],
                         old_hard_native_retained=row['hard_native'],
                         workspace_unchanged=row['workspace_beyond_system'],
                         old_error_upper_retained=True,old_D1_classification_changed=False,
                         classical_online_cost='NOT_ACQUIRED; never zero')
                writer.write(f'rows/row_{spec["row_index"]:02d}.json',v)
                total+=v['positive_pairs'];summaries.append(v)
                del events,row,v
            if total!=c['caps']['positive_pair_evaluations']:
                raise ArithmeticError('global coverage incomplete')
            writer.write('resource_map.json',dict(rows=summaries,**identity,
                         registered_pairs=total,classification='D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'))
            check_protected(ROOT,c);guard.check()
            writer.write('evidence_manifest.json',dict(files=list(writer.files),**identity))
            terminal=dict(status='D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW',**identity,
                          rows=6,registered_pairs=total,resource=guard.snapshot(),runs=1,retries=0,
                          mandatory_STOP=True,next_science_authorized=False,
                          new_synthesis=0,new_angles=0,matrix=0,native_compile=0,sampling=0,GPU=0,
                          source_contract_protected_identity_unchanged=True)
    except BaseException as exc:
        if marker is None: raise
        terminal=dict(status='D2_TECHNICAL_INCONCLUSIVE',**identity,
                      reason=type(exc).__name__+': '+str(exc)[:1500],resource=guard.snapshot(),
                      runs=1,retries=0,mandatory_STOP=True,next_science_authorized=False,
                      prefix_scientific_use=False)
    atomic_json(directory/'terminal.json',terminal,16384)
    atomic_json(directory/'STOP.json',dict(mandatory_STOP=True,retries=0,next_science_authorized=False,
                                          terminal_sha256=digest(directory/'terminal.json')),4096)
    print(terminal['status'])
    return 0 if terminal['status']=='D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    try: sys.exit(execute(a.source))
    except Exception as exc:
        print('D2_LAUNCH_OR_RECORD_FAILURE '+type(exc).__name__+': '+str(exc)[:1500],file=sys.stderr)
        sys.exit(2)
