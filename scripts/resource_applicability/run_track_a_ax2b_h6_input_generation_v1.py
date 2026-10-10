#!/usr/bin/env python3
"""Default metadata only; molecular input work requires a separate pinned grant."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h6_input_generation_contract_v1 import (
    preparation, validate_launch, verify_sources, environment, file_hash)
from trottertracks.resource_applicability.ax2b_limits import exclusive_json, install_worker_limits, output_size
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2b_h6_input_generation_watchdog_v1 import InputProgress, supervise


def read_bounded(path):
    if path.stat().st_size > 4*2**20:
        raise ValueError('LAUNCH_JSON_CAP')
    return json.loads(path.read_text())


def publish_exact_authorization(writer, source, expected_hash):
    """Retain the pinned original bytes, separately from canonical JSON copy."""
    payload = source.read_bytes()
    import hashlib
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError('INPUT_AUTHORIZATION_CHANGED')
    if output_size(writer.output)+len(payload) > writer.byte_cap-writer.reserve:
        raise RuntimeError('AUTHORIZATION_OUTPUT_CAP')
    pending = writer.output/'.pending_authorization_source.json'
    with pending.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, writer.output/'authorization_source.json')
    pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        if args.worker or args.manifest or args.authorization or args.authorization_sha256:
            parser.error('Launch arguments require --execute and a separate input-generation grant.')
        exclusive_json(args.output, preparation())
        print('H6_INPUT_GENERATION_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION')
        return 0
    if not args.manifest or not args.authorization or not args.authorization_sha256:
        parser.error('Separate pinned input-generation grant and sealed manifest required.')
    if file_hash(args.authorization) != args.authorization_sha256:
        raise ValueError('INPUT_AUTHORIZATION_HASH')
    manifest, grant = read_bounded(args.manifest), read_bounded(args.authorization)
    output = args.output.resolve()
    cpu = validate_launch(ROOT, manifest, grant, requested=True, output=output, worker=args.worker)
    caps = manifest['plan']['caps_proposed']
    if not args.worker:
        output.mkdir(parents=False, exist_ok=False)
        writer = AtomicWriter(output, byte_cap=caps['output_bytes'])
        writer.write('launch_binding.json', {'manifest_digest':digest(manifest),'authorization_digest':digest(grant)})
        writer.write('frozen_preparation.json', manifest)
        writer.write('authorization.json', grant)
        publish_exact_authorization(writer, args.authorization, args.authorization_sha256)
        command = [sys.executable, str(Path(__file__).resolve()), '--execute','--worker',
                   '--manifest',str(args.manifest.resolve()),'--authorization',str(args.authorization.resolve()),
                   '--authorization-sha256',args.authorization_sha256,'--output',str(output)]
        terminal = supervise(command, output, caps=caps)
        print(terminal['status'])
        return 0 if terminal['status'].endswith('_COMPLETE') else 1
    install_worker_limits(cpu, caps['address_space_bytes'], caps['output_bytes']-65536)
    writer = AtomicWriter(output, byte_cap=caps['output_bytes'], diagnostics_cap=caps['diagnostics'])
    writer.write('worker_claim.json', {'manifest_digest':digest(manifest),'authorization_digest':digest(grant),
                                      'assigned_cpu':cpu,'retry':False,'resume':False})
    progress = InputProgress(writer, cap=caps['progress_records'], started=STARTED)
    port = None
    try:
        progress.update(point='worker_claimed_before_numerical_imports')
        from trottertracks.resource_applicability.ax2b_h6_input_generation_port_v1 import InputGenerationPort
        port = InputGenerationPort(manifest, writer, progress, grant_sha256=args.authorization_sha256)
        port.execute()
        verify_sources(ROOT, manifest['source_commit'], manifest['source_hashes'])
        if manifest['environment'] != environment():
            raise ValueError('INPUT_ENVIRONMENT_CHANGED_DURING_RUN')
        status, reason = 'H6_INPUT_GENERATION_COMPLETE', None
    except Exception as error:
        status, reason = 'H6_INPUT_GENERATION_STOP', type(error).__name__+':'+str(error)[:512]
    writer.write('worker_terminal.json', {'status':status,'reason':reason,
        'snapshot_receipt_saved':(output/'snapshot_receipt.json').is_file(),
        'calls_attempted':dict(port.calls.used) if port else None,
        'calls_completed':dict(port.completed) if port else None,'latest_progress':progress.row,
        'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False,'retry':False,'resume':False}, terminal=True)
    return 0 if status.endswith('_COMPLETE') else 1


if __name__ == '__main__':
    raise SystemExit(main())
