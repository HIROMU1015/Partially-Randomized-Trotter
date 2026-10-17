"""Metadata preparation only. Default prints the plan.

`--prepare` binds pinned inputs (bytes/SHA and their Git objects), checks the final
B2 four-axis contract, and writes an UNSEALED UNAPPROVED config plus an INVALID
grant. It never parses Phase C scalar rows, ranks, reaccounts, fits, compiles or
issues a valid grant. Sealing (execution_sealed=true with fresh protection lists)
belongs to a separately approved execution step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_phase_c_runtime as r
c = r.c


def pinned_inputs(mapping, *, git=True):
    spec = r.read(r.ROOT/r.INPUT_CONTRACT)
    c.require(spec['schema'] == 'H8_PHASE_C_INPUT_CONTRACT_V2' and spec['real_numeric_pipeline_run'] is False, 'C_INPUT_CONTRACT')
    c.require(set(mapping) == {row['role'] for row in spec['files']}, 'EXACT_INPUT_ROLES')
    files = []
    for row in spec['files']:
        path = Path(mapping[row['role']])
        c.require(path.is_absolute() and path.is_relative_to('/home/AbeHiromu') and not path.is_symlink() and
                  path.is_file() and path.stat().st_size <= 32*2**20, 'C_SAVED_INPUT_PATH')
        receipt = dict(role=row['role'], path=str(path), bytes=path.stat().st_size, sha256=r.sha(path))
        c.require(receipt['bytes'] == row['bytes'] and receipt['sha256'] == row['sha256'], 'PINNED_SAVED_INPUT_BYTES')
        if git and row.get('commit'):
            blob = subprocess.check_output([*r.GIT, 'show', row['commit'] + ':' + row['path']], cwd=r.ROOT)
            c.require(hashlib.sha256(blob).hexdigest() == row['sha256'], 'PINNED_INPUT_GIT_OBJECT:' + row['role'])
        files.append(receipt)
    return files


def prepare(source, mapping_path, python_path, directory, *, seal=False, protection_path=None):
    c.require(type(source) is str and len(source) == 40 and all(x in '0123456789abcdef' for x in source), 'FIXED_C_SOURCE')
    c.require(subprocess.check_output([*r.GIT, 'rev-parse', 'HEAD'], cwd=r.ROOT).decode().strip() == source,
              'COMMITTED_C_SOURCE_HEAD_REQUIRED')
    files = pinned_inputs(r.read(mapping_path))
    by_role = {x['role']: x for x in files}
    c.b2_gate(dict(binding=r.read(by_role['B2_binding']['path']), receipts=r.read(by_role['B2_execution_receipts']['path'])))
    protection = r.read(protection_path) if seal else dict(protected_files=[], protected_worktrees=[])
    target = Path(directory)
    c.require(target.is_absolute() and target.resolve() == target and target.is_relative_to('/home/AbeHiromu') and
              not target.exists(), 'EXCLUSIVE_NEW_C_PREPARATION_PATH')
    config = dict(schema='H8_PHASE_C_CONFIG_V2', SOURCE=source, execution_sealed=bool(seal), approved=False,
        runtime_authorization=False, plan_sha256=c.digest(r.plan()), resource_policy=r.RESOURCE_POLICY,
        source_hashes={p: r.sha(r.ROOT/p) for p in r.SOURCES}, python=str(python_path), python_sha256=r.sha(python_path),
        profile_sha256=c.PROFILE, compiler_hash=c.COMPILER, inputs=files,
        B2_receipt_set_sha256=c.digest([by_role[k] for k in r.B2_ROLES]),
        protection_policy='CAPTURE_ALL_REGISTERED_WORKTREES_AND_PROTECTED_EVIDENCE_AT_SEALING' if not seal else 'SEALED',
        protected_files=protection['protected_files'], protected_worktrees=protection['protected_worktrees'],
        output=str(target/'output01'), control=str(target/'control01'), lock=str(target/'C_ONE_SHOT'))
    r.path_set(config); r.verify(config)
    grant = dict(schema='H8_PHASE_C_NEW_GRANT_V2', scope=r.SCOPE, approved_by_user=False, runtime_authorization=False,
        valid=False, approval_text=None, SOURCE=source, config_sha256=c.digest(config), plan_sha256=config['plan_sha256'],
        model_sha256=c.MODEL_SHA, B1_result_commit=c.B1_RESULT, B2_result_commit=c.B2_RESULT,
        B2_receipt_set_sha256=config['B2_receipt_set_sha256'], resource_policy=r.RESOURCE_POLICY, call_caps=r.CAPS,
        one_shot=True, retry=False, resume=False, refit=False, noncancellation_shots_allowed=False,
        allowed_cpus=[], absolute_launch_command=None, next_stage_authorized=False)
    target.mkdir(mode=0o700)
    for name, value in (('config_UNAPPROVED.json', config), ('grant_INVALID_DRAFT.json', grant)):
        with (target/name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return dict(status='METADATA_PREPARED_NOT_SEALED_NOT_AUTHORIZED' if not seal else 'METADATA_SEALED_NOT_AUTHORIZED',
                SOURCE=source, config_sha256=c.digest(config), grant_sha256=c.digest(grant),
                new_cost_reaccount=0, new_valid_grants=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    for name in ('source', 'input-mapping', 'python', 'directory'): parser.add_argument('--'+name)
    a = parser.parse_args()
    if not a.prepare:
        if any((a.source, a.input_mapping, a.python, a.directory)): parser.error('--prepare required')
        print(json.dumps(r.plan(), indent=2)); return
    if not all((a.source, a.input_mapping, a.python, a.directory)): parser.error('all metadata arguments required')
    print(json.dumps(prepare(a.source, a.input_mapping, a.python, a.directory)))


if __name__ == '__main__': main()
