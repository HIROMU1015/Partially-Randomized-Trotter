#!/usr/bin/env python3
"""H6 technical pilot. Default: metadata only. Execution requires a fresh pinned grant."""
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
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2b_h6_pilot_contract_v1 import (
    preparation,validate_launch,file_hash,verify_parent,verify_sources,environment,install_parallel_limits,resources)
from trottertracks.resource_applicability.ax2b_h6_pilot_watchdog_v1 import PilotProgress,supervise


def bounded_json(path):
    if path.stat().st_size>4*2**20:raise ValueError('PILOT_LAUNCH_JSON_CAP')
    return json.loads(path.read_text())


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
        if a.manifest or a.authorization or a.authorization_sha256 or a.worker:parser.error('A new technical-pilot grant is required for execution.')
        exclusive_json(a.output,preparation());print('H6_PILOT_NOT_AUTHORIZED');return 0
    if not a.manifest or not a.authorization or not a.authorization_sha256:parser.error('Sealed manifest and pinned new technical-pilot grant required.')
    if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('PILOT_AUTHORIZATION_SHA')
    manifest,grant=bounded_json(a.manifest),bounded_json(a.authorization);output=a.output.resolve()
    cpus=validate_launch(ROOT,manifest,grant,output,requested=True,worker=a.worker)
    caps=manifest['plan']['caps_proposed']
    if not a.worker:
        output.mkdir(parents=False,exist_ok=False);writer=AtomicWriter(output,byte_cap=caps['output_bytes'])
        writer.write('launch_binding.json',{'manifest_digest':digest(manifest),'authorization_digest':digest(grant)})
        writer.write('frozen_pilot.json',manifest);writer.write('authorization.json',grant)
        payload=a.authorization.read_bytes()
        if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('PILOT_AUTHORIZATION_CHANGED')
        with (output/'authorization_source.json').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        command=[sys.executable,str(Path(__file__).resolve()),'--execute','--worker','--manifest',str(a.manifest.resolve()),
            '--authorization',str(a.authorization.resolve()),'--authorization-sha256',a.authorization_sha256,'--output',str(output)]
        result=supervise(command,output,caps=caps);print(result['status'])
        return 0 if result['status']=='H6_TECHNICAL_PILOT_COMPLETE' else 1
    install_parallel_limits(cpus,caps['address_space_bytes'],caps['output_bytes']-65536)
    writer=AtomicWriter(output,byte_cap=caps['output_bytes'],diagnostics_cap=caps['diagnostics'])
    writer.write('worker_claim.json',{'manifest_digest':digest(manifest),'authorization_digest':digest(grant),
        'assigned_resources':resources(cpus),'retry':False,'resume':False,'authorization_sha256':a.authorization_sha256})
    progress=PilotProgress(writer,cap=caps['progress_records'],started=STARTED);port=None
    try:
        progress.update(point='claimed_before_numerical_import')
        from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import H6PilotPort
        port=H6PilotPort(ROOT,manifest,writer,progress,started=STARTED);writer.observer=port.observe
        port.setup();port.correctness();port.costs()
        if port.completed!=7 or port.compiled!=36:raise ValueError('PILOT_TERMINAL_COUNTS')
        verify_sources(ROOT,manifest['source_commit'],manifest['source_hashes'])
        if verify_parent(ROOT)!=manifest['input_identity']:raise ValueError('PILOT_INPUT_AFTER')
        if environment()!=manifest['environment']:raise ValueError('PILOT_ENVIRONMENT_AFTER')
        status,reason='H6_TECHNICAL_PILOT_COMPLETE',None
    except Exception as e:status,reason='H6_TECHNICAL_PILOT_STOP',type(e).__name__+':'+str(e)[:512]
    writer.write('worker_terminal.json',{'status':status,'reason':reason,'correctness_completed':port.completed if port else 0,
        'compiled_wrappers':port.compiled if port else 0,'calls_attempted':dict(port.calls.used) if port else {},
        'authorization_sha256':a.authorization_sha256,'source_commit':manifest['source_commit'],
        'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'ground_state_certified':False,'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False,'retry':False,'resume':False},terminal=True)
    return 0 if status=='H6_TECHNICAL_PILOT_COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
