#!/usr/bin/env python3
"""Seal a fresh launch identity around the unchanged science runner v3."""
import argparse
from pathlib import Path
import sys

from h6_matched_launch_preflight_v4 import (
    ROOT, OUTPUT_PATH, PREPARATION_PATH, METADATA_SOURCES, VALIDATION_SOURCE,
    new_manifest, verify_sealed, output_parent, write_exclusive, file_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-source-commit', required=True)
    args = parser.parse_args()
    destination = ROOT / PREPARATION_PATH
    if destination.exists():
        raise ValueError('LAUNCH_PREPARATION_ALREADY_EXISTS')
    manifest = new_manifest(ROOT, args.metadata_source_commit,
                            {p: file_hash(ROOT / p) for p in METADATA_SOURCES},
                            {VALIDATION_SOURCE: file_hash(ROOT / VALIDATION_SOURCE)})
    verified = verify_sealed(ROOT, manifest)
    # All saved-byte/source/environment checks precede filesystem preparation.
    receipt = output_parent(ROOT, OUTPUT_PATH, prepare_parent=True)
    destination.mkdir(parents=True, exist_ok=False)
    write_exclusive(destination / 'sealed_manifest_launch_v4.json', manifest)
    write_exclusive(destination / 'output_parent_preparation_v4.json', receipt)
    verified.update(output_parent_ready=True, output_leaf_unused=True, approved_grants_created=0,
                    H6_status='H6_NOT_AUTHORIZED', contract_status='DRAFT_NOT_AUTHORIZATION',
                    mandatory_stop=True, next_stage_authorized=False)
    write_exclusive(destination / 'preparation_verification_v4.json', verified)
    if any(p in sys.modules for p in ('numpy', 'scipy', 'qiskit', 'numba', 'openfermion')):
        raise RuntimeError('HEAVY_SCIENCE_IMPORT_FORBIDDEN')
    print('SEALED_LAUNCH_PREPARATION_V4 SCIENCE_NOT_LAUNCHED H6_NOT_AUTHORIZED')


if __name__ == '__main__':
    main()
