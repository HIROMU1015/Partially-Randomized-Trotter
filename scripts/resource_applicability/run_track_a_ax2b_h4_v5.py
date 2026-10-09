#!/usr/bin/env python3
"""Default: prepare metadata. --execute needs separately approved/pinned H4 authorization."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import (
    preparation, validate_launch, file_hash, digest,
)
from trottertracks.resource_applicability.ax2b_limits import (
    exclusive_json, supervise, install_worker_limits,
)


def bounded_json(path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('JSON_INPUT_SIZE')
    return json.loads(path.read_text())


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
            parser.error('Launch arguments require --execute and separate authorization.')
        exclusive_json(args.output, preparation(ROOT))
        print('AX2B_H4_FINGERPRINT_V5_PREPARED_SCIENCE_NOT_AUTHORIZED')
        return 0
    if args.manifest is None or args.authorization is None or args.authorization_sha256 is None:
        parser.error('Explicit, pinned authorization and manifest are required.')
    if file_hash(args.authorization) != args.authorization_sha256:
        raise ValueError('AUTHORIZATION_FILE_HASH')
    manifest, authorization = bounded_json(args.manifest), bounded_json(args.authorization)
    cpu = validate_launch(ROOT, manifest, authorization, requested=args.execute,
                          output=args.output, worker=args.worker)
    if cpu not in os.sched_getaffinity(0):
        raise ValueError('ASSIGNED_CPU_UNAVAILABLE')
    caps = manifest['plan']['caps']
    if args.worker:
        install_worker_limits(cpu, caps['address_space_bytes'], caps['output_bytes'] - 65536)
        exclusive_json(args.output / 'worker_claim.json', {'assigned_cpu': cpu,
                       'manifest_digest': digest(manifest), 'authorization_digest': digest(authorization),
                       'resumption_allowed': False})
        # Scientific imports occur only here, after authorization, identities,
        # CPU affinity, address-space/file-size limits and one-thread settings.
        from trottertracks.resource_applicability.ax2b_h4_science_v5 import Pilot
        return 0 if Pilot(ROOT, args.output, manifest).run() else 1
    args.output.mkdir(parents=False, exist_ok=False)
    exclusive_json(args.output / 'launch_binding.json', {'manifest_digest': digest(manifest),
                                                       'authorization_digest': digest(authorization)})
    exclusive_json(args.output / 'frozen_preparation.json', manifest)
    exclusive_json(args.output / 'authorization.json', authorization)
    command = [sys.executable, str(Path(__file__).resolve()), '--execute', '--worker',
               '--manifest', str(args.manifest.resolve()), '--authorization', str(args.authorization.resolve()),
               '--authorization-sha256', args.authorization_sha256, '--output', str(args.output.resolve())]
    report = supervise(command, args.output, total_wall_seconds=caps['total_wall_seconds'],
                       phase_wall_seconds=caps['phase_wall_seconds'], output_bytes=caps['output_bytes'])
    print(report['status'])
    return 0 if report['status'] == 'H4_TECHNICAL_PILOT_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
