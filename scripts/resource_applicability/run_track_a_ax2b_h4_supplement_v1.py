#!/usr/bin/env python3
"""Default metadata only; each H4 supplement unit requires its own new grant."""
import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.ax2b_supplement_launch_v1 import (
        UNITS, preparation, validate_launch, verify_sources, verify_input, environment)
from trottertracks.resource_applicability.ax2b_limits import exclusive_json, install_worker_limits
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter, Progress
from trottertracks.resource_applicability.ax2b_supplement_watchdog_v1 import supervise


def bounded_json(path):
    if path.stat().st_size > 4*2**20:
        raise ValueError('JSON_INPUT_SIZE')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', choices=UNITS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        if args.worker or args.manifest or args.authorization or args.authorization_sha256:
            parser.error('Launch arguments require --execute and a separate new grant.')
        exclusive_json(args.output, preparation(args.unit))
        print('H4_SUPPLEMENT_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION')
        return 0
    if not args.manifest or not args.authorization or not args.authorization_sha256:
        parser.error('Separate pinned authorization and sealed unit manifest required.')
    if file_hash(args.authorization) != args.authorization_sha256:
        raise ValueError('AUTHORIZATION_FILE_HASH')
    manifest, grant = bounded_json(args.manifest), bounded_json(args.authorization)
    if manifest.get('unit') != args.unit:
        raise ValueError('CLI_UNIT_BINDING')
    cpu = validate_launch(ROOT, manifest, grant, requested=True, output=args.output, worker=args.worker)
    caps = manifest['plan']['caps_proposed']
    if not args.worker:
        args.output.mkdir(parents=False, exist_ok=False)
        writer = AtomicWriter(args.output, byte_cap=caps['output_bytes'])
        writer.write('launch_binding.json', {'manifest_digest':digest(manifest),'authorization_digest':digest(grant)})
        writer.write('frozen_preparation.json', manifest)
        writer.write('authorization.json', grant)
        command = [sys.executable, str(Path(__file__).resolve()), '--unit', args.unit, '--execute','--worker',
                   '--manifest',str(args.manifest.resolve()),'--authorization',str(args.authorization.resolve()),
                   '--authorization-sha256',args.authorization_sha256,'--output',str(args.output.resolve())]
        result = supervise(command, args.output, unit=args.unit, caps=caps)
        print(result['status'])
        return 0 if result['status'].endswith('_COMPLETE') else 1
    install_worker_limits(cpu, caps['address_space_bytes'], caps['output_bytes']-65536)
    writer = AtomicWriter(args.output, byte_cap=caps['output_bytes'], diagnostics_cap=caps['diagnostics'])
    writer.write('worker_claim.json', {'manifest_digest':digest(manifest),'authorization_digest':digest(grant),
                                      'assigned_cpu':cpu,'retry':False,'resume':False})
    progress = Progress(writer, args.unit, cap=caps['progress_records'], started=STARTED)
    port = None
    try:
        progress.update(point='worker_claimed_before_numerical_imports')
        # No numerical imports occur until grant/source/input/environment/output
        # binding, worker claim and process limits have all passed.
        from trottertracks.resource_applicability.ax2b_h4_supplement_port_v1 import SupplementPort
        port = SupplementPort(ROOT, manifest, writer, progress, started=STARTED)
        port.execute_unit()
        verify_sources(ROOT, manifest['source_commit'], manifest['source_hashes'])
        verify_input(ROOT, manifest['input_binding'], 'H4_LIMITED')
        if environment() != manifest['environment']:
            raise ValueError('ENVIRONMENT_CHANGED_DURING_RUN')
        expected = (2,4,0) if args.unit == 'S4_MP' else (0,0,4)
        if (port.completed,progress.row['mp_completed'],progress.row['event_completed']) != expected:
            raise ValueError('SUPPLEMENT_COMPLETION_COUNTS')
        if progress.row['primitive_completed'] != 537 or port.compiled != 0:
            raise ValueError('SUPPLEMENT_PREREQUISITE_COUNTS')
        status, reason = 'H4_SUPPLEMENT_COMPLETE', None
    except Exception as error:
        status, reason = 'H4_SUPPLEMENT_STOP', type(error).__name__+':'+str(error)[:512]
    writer.write('worker_terminal.json', {'status':status,'reason':reason,'unit':args.unit,
        'completed_correctness_cells':port.completed if port else 0,'compiled_wrappers':port.compiled if port else 0,
        'completed_mp_records':progress.row['mp_completed'],'completed_event_groups':progress.row['event_completed'],
        'primitive_completed':progress.row['primitive_completed'],
        'calls_reserved_before_work':port.calls.used if port else None,'latest_progress':progress.row,
        'N':None,'G':None,'mandatory_stop':True,'next_stage_authorized':False,
        'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION'}, terminal=True)
    return 0 if status.endswith('_COMPLETE') else 1


if __name__ == '__main__':
    raise SystemExit(main())
