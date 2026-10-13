#!/usr/bin/env python3
"""Future D1 one-shot. Pending authority rejects before scientific imports."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_development.evidence import digest,atomic_json,EvidenceWriter
from trottertracks.algorithm_codesign.rte_development.launch import verify_launch,consume_marker,Guard,check_protected

CONTRACT=ROOT/'artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json'


def execute(source):
    c,auth,head=verify_launch(ROOT,CONTRACT,source)
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
    runtime=verify_runtime(ROOT,c)
    if digest(Path(sys.executable).resolve())!=c['runtime_executable_sha256']:
        raise PermissionError('fixed isolated runtime required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise PermissionError('numeric threads must be one')
    directory=ROOT/c['result_directory']; guard=Guard(c['caps'])
    identity={'execution_HEAD':head,'source_commit':source,
              'authorization_sha256':digest(ROOT/c['authorization_path']),
              'contract_sha256':digest(CONTRACT)}
    # Enter guards before consuming authority; no registered constructors yet.
    marker=None; code=1; terminal=None
    try:
        with guard:
            marker,mark=consume_marker(directory,{'kind':'D1_ONE_SHOT_CONSUMED',**identity,
                'runs':1,'retries':0,'mandatory_STOP':True,'next_science_authorized':False})
            identity['marker_sha256']=mark['sha256'];guard.directory=directory
            writer=EvidenceWriter(directory,c['caps']['output_bytes'],guard.check,c['caps']['review_chunk_bytes'])
            writer.write('preflight.json',{'runtime':runtime,**identity,'threads':1,
                'source_manifest_sha256':digest(ROOT/c['source_manifest']),
                'protected_ledger_sha256':digest(ROOT/c['protected_ledger']),
                'UTC_start_unix':time.time(),'per_key_retry':0})
            from trottertracks.algorithm_codesign.rte_development.pipeline import run_pipeline
            counts=run_pipeline(ROOT,c,identity,writer,guard)
            check_protected(ROOT,c);guard.check()
            writer.write('evidence_manifest.json',{'files':list(writer.files),**identity,
                'all_chunk_reconstruction_verified':True,'resource':guard.snapshot()})
            guard.check()
            terminal={'status':'D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW',**identity,**counts,
                'resource':guard.snapshot(),'runs':1,'retries':0,'mandatory_STOP':True,
                'method_adoption':False,'next_science_authorized':False,
                'quantum_shots':0,'sampling':0,'trajectory':0,'LP':0,'DF':0,'molecule_NPZ_GPU':0,
                'source_contract_protected_identity_unchanged':True,'outer_receipt_required':True}
    except BaseException as exc:
        if marker is None:raise
        terminal={'status':'D1_TECHNICAL_INCONCLUSIVE',**identity,
            'reason':type(exc).__name__+': '+str(exc)[:1500],'resource':guard.snapshot(),
            'runs':1,'retries':0,'mandatory_STOP':True,'next_science_authorized':False,
            'prefix_scientific_use':False,'outer_receipt_required':True}
    if terminal['status']=='D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW':code=0
    # Bounded receipts are reserved even at output failure. Never rewrite a prefix.
    atomic_json(directory/'terminal.json',terminal,16384)
    atomic_json(directory/'STOP.json',{'terminal_sha256':digest(directory/'terminal.json'),
        'mandatory_STOP':True,'retries':0,'next_science_authorized':False},4096)
    print(terminal['status'],flush=True)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    args=parser.parse_args()
    try:sys.exit(execute(args.source))
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        print('D1_LAUNCH_OR_RECORD_FAILURE '+type(exc).__name__+': '+str(exc)[:1500],file=sys.stderr)
        sys.exit(2)
