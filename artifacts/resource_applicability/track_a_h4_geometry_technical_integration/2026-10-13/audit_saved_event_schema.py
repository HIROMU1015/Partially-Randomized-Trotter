"""Read names and at most 24 small records; never load arrays or replay seeds."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import signal


def audit(root):
    counts = Counter()
    samples = {}
    reads = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            family = entry.name.split('-', 1)[0]
            counts[family] += 1
            if family != 'record' or reads >= 24:
                continue
            with open(entry.path, 'rb') as stream:
                data = stream.read(4097)
            reads += 1
            if len(data) > 4096:
                raise ValueError('record exceeds metadata-only read limit')
            record = json.loads(data)
            fields = sorted(record)
            samples[entry.name] = {
                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'geometry': record['geometry'], 'saved_fields': fields,
                'event_order_component_fields_present': any(
                    key in record for key in ('events', 'order', 'orders', 'components', 'products')),
                'trajectory_seed_saved': 'trajectory_seed' in record,
                'numerical_fingerprint_saved': 'numerical_circuit_fingerprint' in record,
            }
    return {
        'status': 'EVENT_COUNTS_NOT_SAVED_IN_CLOSED_RUNTIME_SCHEMA',
        'file_family_counts': dict(counts), 'record_reads': reads,
        'record_read_limit': 24, 'max_bytes_per_record': 4096,
        'bounded_record_samples': samples,
        'sample_role': 'schema corroboration; source payload and inherited full-record audit determine coverage',
        'orders_within_K': 'NOT_SAVED; absence of metadata is not an observed zero',
        'orders_above_K': 'OUTSIDE_CUTOFF',
        'component_observations': 'NOT_SAVED',
        'seed_replay_and_generation_performed': False,
        'H6_0p00734_percent_correction_used': False,
        'new_science_sampling_compile': 0,
    }


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128*2**20, 128*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    signal.alarm(60)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'record_reads': result['record_reads'],
                      'file_family_counts': result['file_family_counts']}))


if __name__ == '__main__':
    main()
