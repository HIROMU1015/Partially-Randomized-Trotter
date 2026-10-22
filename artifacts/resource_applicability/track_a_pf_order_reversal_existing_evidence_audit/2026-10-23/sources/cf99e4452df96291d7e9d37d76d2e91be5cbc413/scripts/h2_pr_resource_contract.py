"""H2-only F contract. Importing this module never imports scientific packages."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from h6_dba_contract import digest, read, require, save, sha

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.h2_pr_resource_private/2026-10-19'
PUBLIC = ROOT / 'artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19'
SCOPE = 'H2_PR_RESOURCE_F_V1_20261019'
PROVIDER_SOURCE = '217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
SOURCE_FILES = ['scripts/h2_pr_resource_contract.py', 'scripts/h2_tol_only_input.py',
                'scripts/run_h2_pr_f.py', 'scripts/prepare_h2_pr_resource.py', 'scripts/h6_dba_contract.py']
CALLS = dict(scf_kernel=1, integral_build=1, df_decomposition=1,
             sector_matrix_columns=4, independent_occupation_matrix=1, solver_eigh=1,
             signal=0, sampling=0, compile=0, shots=0)


def plan():
    return dict(scope=SCOPE, stage='F',
        geometry=[['H', [0., 0., 0.]], ['H', [0., 0., 1.]]],
        basis='sto-3g', charge=0, multiplicity=1, unit='angstrom', symmetry=False,
        scf_conv_tol=1e-9, scf_max_cycle=50, df_tol=1e-8,
        final_rank_supplied=False, cutoff=0., signed_weights=True,
        coefficient_order='decomposer_generation_order', n_qubits=4,
        nelec_alpha=1, nelec_beta=1, sector_dimension=4,
        matrix_backend='python', state_solver='scipy.linalg.eigh',
        state_phase='largest_sector_amplitude_real_positive_v1',
        post_save_renormalization=False, minimum_ground_gap=1e-12,
        projection_limit='9.9e-11', sector_oracle_tolerance=1e-9,
        hermiticity_tolerance=1e-10, state_residual_tolerance=1e-10,
        state_norm_tolerance=1e-12, ground_state_certified=False,
        calls=dict(CALLS), automatic_retry=False, next_stage_authorized=False,
        mandatory_STOP=True, P_science_in_scope=False)


def owned(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve().is_relative_to(PRIVATE.resolve()),
            'H2_PRIVATE_PATH_REQUIRED')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != ROOT.parent),
            'H2_PRIVATE_SYMLINK_REJECTED')
    return path


def verify(spec, grant, *, runtime=True, interpreter=False):
    # Deliberately reject pending/foreign grants before hashes, imports or writes.
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
            'SEPARATE_EXPLICIT_F_APPROVAL_REQUIRED')
    require(spec.get('scope') == SCOPE and spec.get('stage') == grant.get('stage') == 'F', 'H2_F_ONLY')
    require(spec.get('status') == 'SEALED_FOR_EXECUTION' and spec.get('plan') == plan(), 'F_PLAN_CHANGED')
    require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
            and grant.get('next_stage_authorized') is False, 'ONE_SHOT_STOP_REQUIRED')
    require(grant.get('spec_digest') == digest(spec), 'SPEC_DIGEST_CHANGED')
    require(grant.get('SOURCE') == spec.get('SOURCE') and len(spec['SOURCE']) == 40, 'SOURCE_REQUIRED')
    receipt_path = owned(grant['approval_receipt'])
    require(sha(receipt_path) == grant.get('approval_receipt_sha256'), 'APPROVAL_RECEIPT_BYTES')
    receipt = read(receipt_path)
    require(receipt.get('role') == 'user' and receipt.get('stage') == 'F'
            and receipt.get('SOURCE') == spec['SOURCE'] and receipt.get('plan_digest') == digest(plan())
            and isinstance(receipt.get('verbatim_user_message'), str)
            and bool(receipt['verbatim_user_message'].strip()), 'EXPLICIT_USER_F_RECEIPT')
    pending_path = owned(spec['pending_spec'])
    require(sha(pending_path) == spec['pending_spec_sha256'], 'PENDING_SEAL_PRESERVED')
    pending = read(pending_path)
    require(pending.get('status') == 'SEALED_PENDING_APPROVAL', 'PENDING_SEAL_STATUS')
    require({k: v for k, v in spec.items() if k not in ('status', 'pending_spec', 'pending_spec_sha256')}
            == {k: v for k, v in pending.items() if k != 'status'}, 'EXECUTION_SCOPE_DIFFERS_FROM_PENDING_SEAL')
    require(set(spec['source_hashes']) == set(SOURCE_FILES), 'EXACT_F_SOURCE_CLOSURE')
    for rel, expected in spec['source_hashes'].items():
        path = ROOT / rel
        require(sha(path) == expected, 'SOURCE_BYTES_CHANGED:' + rel)
        blob = subprocess.check_output(['git', 'show', spec['SOURCE'] + ':' + rel], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == expected, 'SOURCE_GIT_IDENTITY:' + rel)
    provider = owned(spec['provider_root'])
    require(spec['provider_SOURCE'] == PROVIDER_SOURCE, 'FIXED_PROVIDER_SOURCE')
    for rel, expected in spec['provider_hashes'].items():
        require((provider / rel).resolve().is_relative_to(provider), 'PROVIDER_PATH_ESCAPE')
        require(sha(provider / rel) == expected, 'FROZEN_PROVIDER_CHANGED:' + rel)
    manifest = owned(spec['runtime_manifest'])
    require(sha(manifest) == spec['runtime_manifest_sha256'], 'RUNTIME_MANIFEST_CHANGED')
    if runtime:
        for path, expected in read(manifest).items():
            require(sha(path) == expected, 'RUNTIME_CHANGED:' + path)
    require(sha(spec['python']) == spec['python_sha256'], 'PYTHON_CHANGED')
    if interpreter:
        require(Path(sys.executable).absolute() == Path(spec['python']).absolute(), 'FIXED_PYTHON_REQUIRED')
        require(sys.pycache_prefix is not None
                and Path(sys.pycache_prefix).absolute() == PRIVATE / 'runtime_pycache',
                'H2_PYCACHE_PREFIX_REQUIRED')
    attempt = owned(spec['attempt'])
    require(attempt == PRIVATE / 'attempts/F_run01', 'SEALED_F_ATTEMPT')
    require(spec['scientific_inputs'] == [] and spec['call_budget'] == CALLS, 'GENERATION_RECIPE_AND_BUDGET')
    return dict(status='PASS', SOURCE=spec['SOURCE'], scope=SCOPE)


def reserve(output, name, count=1):
    require(type(count) is int and count > 0 and count <= CALLS.get(name, 0), 'F_CALL_BUDGET:' + name)
    save(Path(output) / ('reserved_' + name + '.json'),
         dict(category=name, count=count, maximum=CALLS[name], nonrefundable=True))


def actual_calls(output):
    return {name: read(Path(output) / ('reserved_' + name + '.json'))['count']
            if (Path(output) / ('reserved_' + name + '.json')).exists() else 0 for name in CALLS}


def observation():
    # Observations do not impose new CPU, memory, worker or PSI limits.
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'MemTotal:')):
            key, value, _ = line.split(); values[key.rstrip(':')] = int(value) * 1024
    return dict(allowed_cpus=sorted(os.sched_getaffinity(0)), memory_bytes=values,
                cpu_pressure=Path('/proc/pressure/cpu').read_text(),
                memory_pressure=Path('/proc/pressure/memory').read_text(),
                agent_imposed_CPU_RAM_worker_wall_limits=None,
                unrelated_worktree_HEAD_status_is_stop_gate=False)
