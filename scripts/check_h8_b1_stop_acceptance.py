"""Read-only, review-only checker. Has no execution or approval command."""
import argparse
import hashlib
from pathlib import PurePosixPath, Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/trotterlib'))
import h8_b1_stop_acceptance as review

GIT = ('git', '--no-optional-locks', '-c', 'index.threads=1', '-c',
       'core.preloadIndex=false', '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false')
B0_DIR = 'artifacts/resource_applicability/track_a_h8_b0_anchor_run02_stop/2026-10-16/'
B1_DIR = 'artifacts/resource_applicability/track_a_h8_phase_b1_run01/2026-10-17/'
EVIDENCE_OBJECTS = {
    **{role: (review.B0_RESULT, B0_DIR + name) for role, name in (
        ('stop', 'STOP_assessment_v1.json'), ('binding', 'execution_binding_v1.json'),
        ('train', 'train_measurements.json'), ('holdout', 'holdout_measurements.json'),
        ('workers', 'worker_exit_reap_receipts_v1.json'), ('controls', 'preparation_control_receipts_v1.json'))},
    'plan': (review.B0_SOURCE, 'artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/acquisition_plan_UNAPPROVED_v1.json'),
    'audit': (review.AUDIT_COMMIT, 'artifacts/resource_applicability/track_a_h8_b0_saved_anchor_audit/2026-10-17/saved_anchor_audit_v1.json'),
    **{role: (review.EXISTING_B1_RESULT, B1_DIR + name) for role, name in (
        ('B1_contract', 'execution_contract_v1.json'), ('B1_binding', 'execution_binding_v1.json'),
        ('B1_terminal', 'execution_terminal_v1.json'), ('B1_phase', 'phase_terminal_v1.json'),
        ('B1_cleanup', 'resource_and_cleanup_v1.json'), ('B1_model', 'frozen_models_v1.json'))},
}


def verify(root, inventory, contract):
    review.require(inventory['schema'] == 'H8_B1_ACCEPTANCE_EVIDENCE_V1', 'EVIDENCE_SCHEMA')
    review.validate_draft(contract, review.digest(inventory))
    records = {}
    for row in inventory['files']:
        role, commit, name = row['role'], row['commit'], row['path']
        path = PurePosixPath(name)
        review.require(not path.is_absolute() and '..' not in path.parts and
                       len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'UNSAFE_GIT_OBJECT')
        review.require(role not in records, 'DUPLICATE_EVIDENCE_ROLE')
        review.require(EVIDENCE_OBJECTS.get(role) == (commit, name), 'FIXED_EVIDENCE_OBJECT:' + role)
        data = subprocess.check_output([*GIT, 'show', commit + ':' + name], cwd=root)
        review.require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                       'EVIDENCE_BYTES:' + role)
        records[role] = review.strict_json(data)
    return review.validate_records(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--draft', required=True)
    args = parser.parse_args()
    result = verify(args.repository, review.strict_json(Path(args.inventory).read_bytes()),
                    review.strict_json(Path(args.draft).read_bytes()))
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
