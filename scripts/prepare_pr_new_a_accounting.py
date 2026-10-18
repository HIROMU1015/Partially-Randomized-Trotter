"""Bind existing scalar files and audit schemas; never call numerical account()."""
import argparse
import fractions
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pr_new_a_accounting as r
c = r.c
BASE = Path('/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018')
PARALLEL = Path('/home/AbeHiromu/projects/h8-pr-candidate-parallel-preparation/20261018')
PUBLISHED = 'fc6bd039dc7171d5674ac04f0b3ad427f1b6495b'


def byte_row(role, path, expected=None):
    path = Path(path)
    c.require(path.is_file() and not path.is_symlink(), 'REGULAR_BYTE_INPUT')
    h = c.sha(path)
    c.require(expected is None or h == expected, 'EXPECTED_BYTES:' + role)
    return dict(role=role, path=str(path), bytes=path.stat().st_size, sha256=h)


def git_blob(source, path):
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'maintenance.auto=false',
        '-c', 'gc.auto=0', '-C', str(r.ROOT), 'show', source + ':' + path])


def profile_identity(native, binding):
    if 'profile_reference' in native:
        profile = native['profile_reference']
        c.require(profile['sha256'] == binding['profile'], 'ORIGINAL_PROFILE_BINDING')
        return profile, 'PUBLIC_FILE_BYTES'
    # Existing H8 contract carries an identifier, not a recoverable profile file
    # SHA. Keep that legacy limitation; do not manufacture a matching profile.
    c.require(native['profile_sha256'] == binding['profile'], 'H8_PROFILE_IDENTIFIER')
    return dict(identifier=native['profile_sha256'], kind='INHERITED_NATIVE_PROFILE_IDENTIFIER',
                original_profile_file_bytes_verified=False,
                old_compiler_output_bit_identity_verified=False), 'INHERITED_IDENTIFIER'


def prepare(system, root, source):
    target = root / system
    target.mkdir(parents=True, exist_ok=False)
    contracts = BASE / 'private/sealed03/contracts'
    origin_path = contracts / (system + '_A_COST') / 'technical_seal.json'
    origin = c.read(origin_path)
    a_spec_path = (PARALLEL / 'runs/resume01/sealed/technical_spec.json' if system == 'H8' else origin_path)
    d_spec_path = (BASE / 'runs/H8_D_repair02/technical_seal.json' if system == 'H8'
                   else contracts / (system + '_D_ACCOUNTING') / 'technical_seal.json')
    a_spec, d_spec = c.read(a_spec_path), c.read(d_spec_path)
    payload = c.read(origin['payload'])
    native = c.read(payload['native_adapter']['native_config'])
    d_payload = c.read(d_spec['payload'])
    public = r.ROOT / 'artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18'
    registered = c.read(public / 'a_candidates.json')[system]
    c.core.same_binding(origin['binding'], registered['binding'])
    c.core.same_binding(origin['binding'], d_spec['binding'])
    c.core.same_binding(origin['binding'], a_spec['binding'])
    profile, profile_kind = profile_identity(native, origin['binding'])
    if profile_kind == 'PUBLIC_FILE_BYTES':
        profile_path = target / 'compiler_profile.json'
        profile_path.write_bytes(git_blob(profile['commit'], profile['path']))
        profile_input = byte_row('compiler_profile', profile_path, profile['sha256'])
    else:
        profile_input = byte_row('inherited_profile_identifier_record', payload['native_adapter']['native_config'])
    inputs = [byte_row('origin_A_spec', origin_path), byte_row('A_spec', a_spec_path),
        byte_row('A_payload', origin['payload']), byte_row('A_signal', payload['A_signal']),
        byte_row('native_config_metadata', payload['native_adapter']['native_config']),
        byte_row('A_result', Path(a_spec['output']) / 'result.json'),
        byte_row('A_terminal', Path(a_spec['output']) / 'terminal.json'),
        byte_row('D_spec', d_spec_path), byte_row('D_payload', d_spec['payload']),
        byte_row('D_result', Path(d_spec['output']) / 'result.json'),
        byte_row('D_terminal', Path(d_spec['output']) / 'terminal.json'),
        profile_input,
        byte_row('A_runtime_receipt', origin['runtime_receipt'], origin['runtime_receipt_sha256']),
        byte_row('registered_candidates', public / 'a_candidates.json'),
        byte_row('published_completion', public / 'completion_status.json'),
        byte_row('published_verification', public / 'verification.json')]
    # This is a bounded receipt check, not a repeated task campaign or result audit.
    original_wait_path = BASE / 'private/sealed03/authorization/batch_native_completion.json'
    original_wait = c.read(original_wait_path)
    inputs.append(byte_row('original_batch_wait', original_wait_path))
    accepted = {}
    for stage, spec in [('A', a_spec), ('D', d_spec)]:
        if system == 'H8':
            path = (PARALLEL / 'runs/resume01/native_launcher_terminal.json' if stage == 'A'
                    else BASE / 'runs/H8_D_repair02/native_completion.json')
            wait = c.read(path)
            inputs.append(byte_row(stage + '_parent_wait', path))
        else:
            wait = original_wait['results'][system + ('_A_COST' if stage == 'A' else '_D_ACCOUNTING')]
        c.require(wait['exit_code'] == 0 and wait['reaped'] is True, 'COMPLETED_NATIVE_PARENT')
        accepted[stage] = dict(SOURCE=spec['SOURCE'], native_exit_code=0, reaped=True,
                              result_sha256=c.sha(Path(spec['output']) / 'result.json'))
    # Scalar-analysis runtime only. The original compile runtime receipt remains
    # bound as provenance; analysis does not import those scientific providers.
    for module in (statistics, fractions, hashlib, json, math):
        inputs.append(byte_row('analysis_runtime:' + module.__name__, module.__file__))
    for index, path in enumerate(sorted({line.split()[-1] for line in
            Path('/proc/self/maps').read_text().splitlines() if '/' in line and
            line.split()[-1].startswith('/') and Path(line.split()[-1]).is_file()})):
        inputs.append(byte_row('loaded_runtime:' + str(index), path))
    statistics_path = Path(statistics.__file__)
    spec = dict(schema=r.a.SCHEMA, scope=system + '_NEW_A_TOTAL_COST', system=system,
        SOURCE=source, public_result=PUBLISHED, binding=origin['binding'], cells=registered['candidates'],
        source_hashes={p: c.sha(r.ROOT / p) for p in r.FILES}, inputs=inputs,
        python=origin['python'], python_sha256=origin['python_sha256'],
        statistics_sha256=c.sha(statistics_path), completion_acceptance=accepted,
        seed_derivation_SOURCE=origin['SOURCE'], signal_hashes_digest=c.digest(payload['signal_record_hashes']),
        compiler_options=native['compiler_options'], profile_reference=profile, profile_evidence_kind=profile_kind,
        old_PR_cohort_label=r.a.OLD_LABELS[system], old_PR_ids=d_payload['PR_ids'],
        deterministic_sets=d_payload['deterministic_sets'], epsilons=list(c.core.EPS),
        factors=[1., 10.], bias_modes=list(c.core.MODES), P=0,
        calls=dict(saved_scalar_pass=1, signal=0, trajectory=0, wrapper_compile=0,
                   fit=0, shot_recalculation=0, quantum_shots=0, preparation=0),
        output=str(target / 'future_run01/output'), lock=str(target / 'future_run01/ONE_SHOT.json'),
        output_estimate_bytes=64 * 2**20, inode_estimate=32, memory_estimate_bytes=512 * 2**20,
        resource_caps=None, allowed_cpus=[], absolute_launch_command=None,
        resource_policy='admission estimates, not limits; fresh resources; no affinity or unrelated-worktree gate',
        technical_sealed=True, approved=False, runtime_authorization=False,
        retry=False, next_stage_authorized=False, missing_endpoints_stay_unresolved=True)
    spec['structure_audit'] = r.a.audit_structure(r.load_data(spec))
    spec['technical_digest'] = r.technical_digest(spec)
    c.save(target / 'technical_seal.json', spec)
    c.save(target / 'grant_INVALID.json', dict(schema='INVALID_NEW_A_TOTAL_COST_GRANT',
        scope=spec['scope'], system=system, SOURCE=source, technical_digest=spec['technical_digest'],
        approved=False, approved_by_user=False, runtime_authorization=False, one_shot=True,
        retry=False, next_stage_authorized=False, allowed_cpus=[], approval_receipt=None,
        approval_receipt_sha256=None, absolute_launch_command=None))
    c.verify_source(r.ROOT, source, spec['source_hashes'])
    c.verify_byte_rows(inputs)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    root = c.home_path(args.directory)
    root.mkdir(parents=True, exist_ok=False)
    c.require(len(args.source) == 40, 'FIXED_SOURCE_SHA')
    specs = [prepare(system, root, args.source) for system in r.a.OLD_LABELS]
    c.save(root / 'batch_PREPARATION_ONLY.json', dict(schema=r.a.SCHEMA, SOURCE=args.source,
        selected_for_execution=[], approved=False, runtime_authorization=False,
        independent_contracts={s['scope']: s['technical_digest'] for s in specs},
        max_parallel_systems=len(specs), preparation_scientific_calls=0, retry=False))
    print('3 TECHNICAL SEALS; INVALID GRANTS; SCIENTIFIC_ACCOUNTING=0')


if __name__ == '__main__':
    main()
