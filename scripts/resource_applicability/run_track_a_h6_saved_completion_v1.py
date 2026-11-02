#!/usr/bin/env python3
"""Saved DF completion launcher; default metadata only; fresh pinned grant required."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
STARTED=time.monotonic()
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_limits import exclusive_json,install_worker_limits
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2b_h6_saved_completion_contract_v1 import (
    preparation,validate_launch,file_hash,verify_parents,verify_sources,environment)
from trottertracks.resource_applicability.ax2b_h6_saved_completion_watchdog_v1 import CompletionProgress,supervise

def bounded_json(p):
    if p.stat().st_size>4*2**20:raise ValueError('LAUNCH_JSON_CAP')
    return json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=parser.parse_args()
    if not a.execute:
        if a.manifest or a.authorization or a.authorization_sha256 or a.worker:parser.error('Execution requires a separate saved-DF completion grant.')
        exclusive_json(a.output,preparation());print('H6_SAVED_DF_COMPLETION_NOT_AUTHORIZED');return 0
    if not a.manifest or not a.authorization or not a.authorization_sha256:parser.error('A sealed manifest and pinned new saved-DF completion grant are required.')
    if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('COMPLETION_AUTHORIZATION_SHA')
    manifest,grant=bounded_json(a.manifest),bounded_json(a.authorization);output=a.output.resolve()
    cpu=validate_launch(ROOT,manifest,grant,output,requested=True,worker=a.worker)
    caps=manifest['plan']['caps']
    if not a.worker:
        output.mkdir(parents=False,exist_ok=False);writer=AtomicWriter(output,byte_cap=caps['output_bytes'])
        writer.write('launch_binding.json',{'manifest_digest':digest(manifest),'authorization_digest':digest(grant)})
        writer.write('frozen_completion.json',manifest);writer.write('authorization.json',grant)
        payload=a.authorization.read_bytes()
        if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('COMPLETION_AUTHORIZATION_CHANGED')
        with (output/'authorization_source.json').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        command=[sys.executable,str(Path(__file__).resolve()),'--execute','--worker',
                 '--manifest',str(a.manifest.resolve()),'--authorization',str(a.authorization.resolve()),
                 '--authorization-sha256',a.authorization_sha256,'--output',str(output)]
        r=supervise(command,output,caps=caps);print(r['status']);return 0 if r['status']=='H6_SAVED_DF_COMPLETION_COMPLETE' else 1
    install_worker_limits(cpu,caps['address_space_bytes'],caps['output_bytes']-65536)
    writer=AtomicWriter(output,byte_cap=caps['output_bytes'],diagnostics_cap=caps['diagnostics'])
    writer.write('worker_claim.json',{'manifest_digest':digest(manifest),'authorization_digest':digest(grant),
                                    'assigned_cpu':cpu,'retry':False,'resume':False,'authorization_sha256':a.authorization_sha256})
    progress=CompletionProgress(writer,cap=caps['progress_records'],started=STARTED);port=None
    try:
        progress.update(point='claimed_before_numerical_import')
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_port_v1 import SavedCompletionPort
        port=SavedCompletionPort(manifest,writer,progress,root=ROOT,grant_sha256=a.authorization_sha256);port.execute()
        verify_parents(ROOT);verify_sources(ROOT,manifest['source_commit'],manifest['source_hashes'])
        if environment()!=manifest['environment']:raise ValueError('COMPLETION_ENVIRONMENT_AFTER')
        status,reason='H6_SAVED_DF_COMPLETION_COMPLETE',None
    except Exception as e:status,reason='H6_SAVED_DF_COMPLETION_STOP',type(e).__name__+':'+str(e)[:512]
    writer.write('worker_terminal.json',{'status':status,'reason':reason,
        'df_receipt_saved':(output/'df_receipt.json').exists(),
        'snapshot_receipt_saved':(output/'snapshot_receipt.json').exists(),
        'calls_completed':dict(port.completed) if port else None,
        'authorization_sha256':a.authorization_sha256,
        'calls_attempted':dict(port.calls.used) if port else None,
        'H6_input_accepted':status=='H6_SAVED_DF_COMPLETION_COMPLETE','historical_raw_bytes_identity_claim':False,
        'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False,'retry':False,'resume':False},terminal=True)
    return 0 if status=='H6_SAVED_DF_COMPLETION_COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
