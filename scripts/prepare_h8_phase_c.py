"""Future metadata sealing only, after immutable C publication and completed B2.

Default prints the plan. Never ranks, reaccounts, fits, compiles or issues a
valid grant. The present unregistered snapshot cannot pass the Git SOURCE gate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_phase_c_runtime as r
c = r.c


def prepare(source, mapping_path, protection_path, python_path, directory):
    c.require(type(source) is str and len(source) == 40 and all(x in '0123456789abcdef' for x in source), 'FIXED_C_SOURCE')
    c.require(subprocess.check_output([*r.GIT, 'rev-parse', 'HEAD'], cwd=r.ROOT).decode().strip() == source,
              'COMMITTED_C_SOURCE_HEAD_REQUIRED')
    spec = r.read(r.ROOT/'references/input_contract_v1.json')
    mapping = r.read(mapping_path)
    c.require(set(mapping) == {row['role'] for row in spec['files']}, 'EXACT_INPUT_ROLES')
    files = []
    for row in spec['files']:
        path = Path(mapping[row['role']])
        c.require(path.is_absolute() and path.is_relative_to('/home/AbeHiromu') and not path.is_symlink() and
                  path.is_file() and path.stat().st_size <= 32*2**20, 'C_SAVED_INPUT_PATH')
        receipt = dict(role=row['role'], path=str(path), bytes=path.stat().st_size, sha256=r.sha(path))
        if row['bytes'] is not None:
            c.require(receipt['bytes'] == row['bytes'] and receipt['sha256'] == row['sha256'], 'PINNED_SAVED_INPUT_BYTES')
        files.append(receipt)
    by_role = {x['role']: x for x in files}
    proof = {k: r.read(by_role['B2_'+k]['path']) for k in ('terminal', 'cleanup', 'final', 'launcher_native_exit')}
    c.b2_gate(proof)
    protection = r.read(protection_path)
    target = Path(directory)
    c.require(target.is_absolute() and target.resolve() == target and target.is_relative_to('/home/AbeHiromu') and
              not target.exists(), 'EXCLUSIVE_NEW_C_PREPARATION_PATH')
    config = dict(schema='H8_PHASE_C_CONFIG_V1', SOURCE=source, execution_sealed=True,
        plan_sha256=c.digest(r.plan()), resources=r.RESOURCES, source_hashes={p: r.sha(r.ROOT/p) for p in r.SOURCES},
        python=str(python_path), python_sha256=r.sha(python_path), profile_sha256=c.PROFILE, compiler_hash=c.COMPILER,
        inputs=files, B2_receipt_set_sha256=c.digest([by_role['B2_'+k] for k in
            ('terminal', 'cleanup', 'final', 'launcher_native_exit')]),
        protected_files=protection['protected_files'], protected_worktrees=protection['protected_worktrees'],
        output=str(target/'output01'), control=str(target/'control01'), lock=str(target/'C_ONE_SHOT'))
    r.path_set(config); r.verify(config)
    grant = dict(schema='H8_PHASE_C_NEW_GRANT_V1', scope=r.SCOPE, approved_by_user=False, runtime_authorization=False,
        valid=False, approval_text=None, SOURCE=source, config_sha256=c.digest(config), plan_sha256=config['plan_sha256'],
        model_sha256=c.MODEL_SHA, B1_result_commit=c.B1_RESULT, B2_receipt_set_sha256=config['B2_receipt_set_sha256'],
        resources=r.RESOURCES, call_caps=r.CAPS, one_shot=True, retry=False, resume=False, refit=False,
        driver_cpu=None, observer_cpu=None, allowed_cpus=[], absolute_launch_command=None, next_stage_authorized=False)
    target.mkdir(mode=0o700)
    for name, value in (('config_UNAPPROVED.json', config), ('grant_INVALID_DRAFT.json', grant)):
        with (target/name).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)
    return dict(status='METADATA_SEALED_NOT_AUTHORIZED', SOURCE=source, new_cost_reaccount=0, new_valid_grants=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    for name in ('source', 'input-mapping', 'protection', 'python', 'directory'): parser.add_argument('--'+name)
    a = parser.parse_args()
    if not a.prepare:
        if any((a.source, a.input_mapping, a.protection, a.python, a.directory)): parser.error('--prepare required')
        print(json.dumps(r.plan(), indent=2)); return
    if not all((a.source, a.input_mapping, a.protection, a.python, a.directory)): parser.error('all metadata arguments required')
    print(json.dumps(prepare(a.source, a.input_mapping, a.protection, a.python, a.directory)))


if __name__ == '__main__': main()
