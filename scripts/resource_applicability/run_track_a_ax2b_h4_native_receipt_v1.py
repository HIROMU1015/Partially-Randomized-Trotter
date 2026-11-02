#!/usr/bin/env python3
"""Default draft or metadata binding only. H4-P execution needs a separate grant."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.ax2b_h4_native_receipt_v1 import (
    KIND, ProductionPort, bind_metadata, bounded_json, draft, environment,
    run_receipt, safe_path, static_receipt, validate_launch, verify_input, verify_sources,
)
from trottertracks.resource_applicability.ax2b_h4_native_watchdog_v1 import supervise
from trottertracks.resource_applicability.ax2b_h6_controller import BoundedWriter
from trottertracks.resource_applicability.ax2b_limits import exclusive_json, install_worker_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bind-source-commit')
    parser.add_argument('--assigned-cpu', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        if args.worker or args.manifest or args.authorization or args.authorization_sha256 or args.manifest_sha256:
            parser.error('Launch arguments require --execute and a separate H4-P user grant.')
        if bool(args.bind_source_commit) != (args.assigned_cpu is not None):
            parser.error('Metadata binding requires both source commit and assigned CPU.')
        result = bind_metadata(ROOT, args.bind_source_commit, args.assigned_cpu) if args.bind_source_commit else draft()
        exclusive_json(args.output, result)
        print('H4_NATIVE_RECEIPT_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION')
        return 0
    if args.bind_source_commit or args.assigned_cpu is not None:
        parser.error('Execute uses the frozen manifest, not binding options.')
    if not all((args.manifest, args.manifest_sha256, args.authorization, args.authorization_sha256)):
        parser.error('Separate pinned H4-P grant and pinned sealed preparation manifest required.')
    # Both files bounded before hashing; pin exact file bytes and semantic digest.
    manifest, auth = bounded_json(args.manifest), bounded_json(args.authorization)
    if file_hash(args.authorization) != args.authorization_sha256 or file_hash(args.manifest) != args.manifest_sha256:
        raise ValueError('PINNED_LAUNCH_FILE_HASH')
    if auth.get('manifest_sha256') != args.manifest_sha256:
        raise ValueError('AUTHORIZATION_EXACT_MANIFEST_BYTES')
    cpu = validate_launch(ROOT, manifest, auth, requested=True, output=args.output, worker=args.worker)
    caps = manifest['plan']['caps']
    static = static_receipt(ROOT, manifest['static_binding'])
    if not args.worker:
        output = safe_path(ROOT, manifest['plan']['output_path'])
        output.mkdir(parents=True, exist_ok=False)
        exclusive_json(output/'launch_binding.json', {'manifest_digest': digest(manifest), 'authorization_digest': digest(auth)})
        exclusive_json(output/'frozen_preparation.json', manifest)
        exclusive_json(output/'authorization.json', auth)
        command = [sys.executable, str(Path(__file__).resolve()), '--execute', '--worker',
            '--output', str(output), '--manifest', str(args.manifest.resolve()),
            '--manifest-sha256', args.manifest_sha256, '--authorization', str(args.authorization.resolve()),
            '--authorization-sha256', args.authorization_sha256]
        report = supervise(command, output, caps=caps, manifest=manifest, static=static)
        print(report['status'])
        return 0 if report['status'] == KIND+'_COMPLETE' else 1
    writer = BoundedWriter(args.output, byte_cap=caps['output_bytes'], diagnostics_cap=caps['diagnostics'])
    terminal = {'status': KIND+'_STOP', 'reason': 'WORKER_SETUP_INCOMPLETE',
                'mandatory_stop': True, 'next_stage_authorized': False}
    try:
        install_worker_limits(cpu, caps['address_space_bytes'], caps['output_bytes']-65536)
        exclusive_json(args.output/'worker_claim.json', {'manifest_digest': digest(manifest),
            'authorization_digest': digest(auth), 'assigned_cpu': cpu, 'resume': False})
        # First numerical imports/native preparation occur inside this factory.
        port = ProductionPort(ROOT, manifest)
        terminal = run_receipt(port, writer, manifest, static)
        if terminal['status'] == KIND+'_COMPLETE':
            verify_sources(ROOT, manifest['source_commit'], manifest['source_hashes'])
            verify_input(ROOT, manifest['input_binding'], 'H4_LIMITED')
            if environment() != manifest['environment']:
                raise ValueError('ENVIRONMENT_CHANGED_DURING_RUN')
    except Exception as error:
        terminal['status'] = KIND+'_STOP'
        terminal['reason'] = type(error).__name__+':'+str(error)[:512]
    writer.write('worker_terminal.json', terminal, terminal=True)
    return 0 if terminal['status'] == KIND+'_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
