"""Stdlib authorization boundary. No scientific imports or work on import."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

from .isolated_qiskit_execution import digest, read, save, sha

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/non_df_h4_execution_foundation/2026-10-23'
CASES = ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR')


def completed_contract():
    d = deepcopy(read(ROOT/'artifacts/non_df_h4_readiness/2026-10-23/H4_execution_contract_DRAFT.json'))
    d.update(schema='non_df_h4_execution_candidate_v1', status='IMPLEMENTED_AWAITING_BATCH_APPROVAL',
             execution_authorized=False, grant_valid=False, dispatch_enabled=False)
    d['readiness'].update(scientific_dispatch_implemented=True,
        missing_before_execution=['one explicit adoption/batch approval bound to SOURCE/contract/namespace',
            'automatic clean SOURCE/environment seal and fresh one-shot grant', 'fresh resource observation'],
        H4_inputs_generated=False)
    d['execution'].update(seed_namespace='NON_DF_H4_PILOT_V1',
        call_budget=dict(SCF=1, MO_integral_build=1, Pauli_JW_build=1, DF_decomposition=1,
            nonDF_sector_ground=1, DF_sector_crosscheck=1, FCI=1,
            input_dense_reference_signals=2, DF_sector_matvec_columns=36,
            case_evaluations=4, deterministic_trajectories=2, PR_paired_trajectories=512,
            full_wrapper_compile_axes=1028, production_RTE_events=16384,
            Pauli_sampler_batches=256, DF_occurrence_sampler_batches=2048,
            independent_trajectory_checks_per_PR=3, forced_orders_per_PR=[0,2,4],
            case_numerical_evidence='one dense/sparse/Horner evidence evaluation per fixed case; frozen H2 path',
            case_evaluation='one sparse reference/mean and one cost loop per case; five precision projections, no refit'),
        resources='four fresh case interpreters; no shared Qiskit session; small dense BLAS avoids nested oversubscription; no CPU/RAM caps',
        activation='approval names exact SOURCE/contract digest/pack namespace; immutable grant and case claims; no automatic retry')
    d['event_audit'] = dict(schema='production_rte_trajectory_v1',
        actual_returned_events=True, additional_RNG_draws=0, replay=False,
        fields=['case_id','representation','method','trajectory_id','seed','step_seeds',
            'outer_step_index','inner_RTE_position','taylor_order','application_sequence',
            'selected_component_ids','coefficient_sign','rotation_angle','phase','scalar_phase',
            'event_probability','distribution_id','normalization','compiled_RZ','compiler',
            'event_record_hash','trajectory_record_hash'],
        absent_information='null plus information_missing; no inference', additive_event_cost=False,
        population_cost_guarantee=False)
    return d


def seed_for(representation, method):
    return int.from_bytes(hashlib.sha256(f'NON_DF_H4_PILOT_V1/{representation}/{method}'.encode()).digest()[:8], 'little')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def source_manifest():
    names = [str(p.relative_to(ROOT)) for folder in ['src/trotterlib','scripts','tests']
             for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    names += ['artifacts/non_df_h4_execution_foundation/2026-10-23/H4_execution_contract_DRAFT.json']
    return {n:sha(ROOT/n) for n in sorted(names)}


def environment_manifest(plan, *, all_files=False):
    versions = {name:importlib.metadata.version(name) for name in plan['environment']}
    if versions != plan['environment']:
        raise ValueError('H4_RUNTIME_PACKAGE_VERSION')
    files = {str(Path(sys.executable).resolve()):sha(Path(sys.executable).resolve())}
    if all_files:
        # Read-only package closure: no installations or shared cache changes.
        for name in versions:
            dist = importlib.metadata.distribution(name)
            for relative in dist.files or []:
                p = Path(dist.locate_file(relative))
                if p.is_file() and p.suffix in {'.py','.so','.pyd'}:
                    files[str(p.resolve())] = sha(p)
    return dict(python=str(Path(sys.executable).resolve()), versions=versions, files=files)


def require_current_contract(plan):
    if plan != completed_contract():
        raise ValueError('H4_CONTRACT_CHANGED')


def activate(contract_file, approval_file, pack):
    """Future, explicit post-approval seal. NEVER called by readiness/tests."""
    plan = read(contract_file); require_current_contract(plan)
    approval = read(approval_file); pack = Path(pack).absolute()
    SOURCE = git('rev-parse','HEAD')
    if git('status','--porcelain'):
        raise ValueError('H4_SOURCE_MUST_BE_CLEAN')
    expected = dict(schema='h4_explicit_batch_approval_v1', SOURCE=SOURCE,
        contract_digest=digest(plan), pack_namespace=str(pack),
        approve_H4_input_and_four_cases_once=True, adopted_proposal=True,
        retry=False, push=False, scientific_review=False)
    if approval != expected:
        raise ValueError('H4_EXPLICIT_APPROVAL_BINDING_REQUIRED')
    env = environment_manifest(plan, all_files=True)
    claims = ROOT/'.h4_execution_claims'/digest(approval)
    if claims.exists() or pack.exists() or pack.is_symlink():
        raise FileExistsError('H4_APPROVAL_ALREADY_BOUND_OR_NAMESPACE_USED')
    claims.mkdir(parents=True)
    save(claims/'approval_claim.json',dict(approval=approval, pack=str(pack), no_rebind=True))
    pack.mkdir(parents=True)
    save(pack/'contract.json',plan)
    save(pack/'approval.json',approval)
    source = dict(SOURCE=SOURCE, files=source_manifest(), environment=env)
    save(pack/'SOURCE.json',source)
    grant = dict(schema='h4_one_shot_grant_v1', SOURCE=SOURCE, contract_digest=digest(plan),
        source_manifest_sha256=sha(pack/'SOURCE.json'), approval_sha256=sha(pack/'approval.json'),
        pack_namespace=str(pack), claim_directory=str(claims), authorized_cases=list(CASES),
        execution_authorized=True, grant_valid=True, dispatch_enabled=True,
        one_shot=True, automatic_retry=False, mandatory_STOP=True)
    grant['grant_id'] = digest(grant)
    save(pack/'grant.json',grant)
    return grant


def checked_pack(pack, *, allow_consumed=False):
    """All authorization checks precede molecular/quantum module imports."""
    pack = Path(pack).absolute()
    if pack.is_symlink() or not (pack/'grant.json').is_file():
        raise PermissionError('H4_NO_VALID_GRANT')
    plan, source, approval, grant = [read(pack/n) for n in ['contract.json','SOURCE.json','approval.json','grant.json']]
    require_current_contract(plan)
    verify_authority_records(pack, plan, source, approval, grant)
    claims = Path(grant['claim_directory'])
    if claims.is_symlink() or claims.stat().st_uid != os.getuid():
        raise PermissionError('H4_CLAIM_DIRECTORY_IDENTITY')
    if read(claims/'approval_claim.json') != dict(approval=approval,pack=str(pack),no_rebind=True):
        raise PermissionError('H4_APPROVAL_CLAIM')
    verify_sealed_source(source, plan)
    check_consumption(pack, allow_consumed=allow_consumed)
    return plan, source, grant


def verify_sealed_source(source, plan):
    if source['SOURCE'] != git('rev-parse','HEAD') or source['files'] != source_manifest():
        raise PermissionError('H4_SOURCE_BYTES_OR_COMMIT_CHANGED')
    env = source['environment']
    if environment_manifest(plan)['versions'] != env['versions'] or str(Path(sys.executable).resolve()) != env['python']:
        raise PermissionError('H4_RUNTIME_CHANGED')
    for path, expected in env['files'].items():
        if sha(path) != expected:
            raise PermissionError('H4_RUNTIME_FILE_CHANGED:'+path)


def verify_authority_records(pack, plan, source, approval, grant):
    """Pure binding validation; does not issue or consume any authority."""
    required = dict(schema='h4_one_shot_grant_v1',SOURCE=source['SOURCE'],contract_digest=digest(plan),
        source_manifest_sha256=sha(pack/'SOURCE.json'),approval_sha256=sha(pack/'approval.json'),
        pack_namespace=str(pack),claim_directory=str(ROOT/'.h4_execution_claims'/digest(approval)),
        authorized_cases=list(CASES),execution_authorized=True,grant_valid=True,dispatch_enabled=True,
        one_shot=True,automatic_retry=False,mandatory_STOP=True)
    required['grant_id'] = digest(required)
    if grant != required:
        raise PermissionError('H4_INVALID_GRANT_BINDING')
    if approval != dict(schema='h4_explicit_batch_approval_v1',SOURCE=source['SOURCE'],
        contract_digest=digest(plan),pack_namespace=str(pack),approve_H4_input_and_four_cases_once=True,
        adopted_proposal=True,retry=False,push=False,scientific_review=False):
        raise PermissionError('H4_APPROVAL_BINDING')


def check_consumption(pack, *, allow_consumed=False):
    if not allow_consumed and any((pack/p).exists() for p in ['CONSUMED.json','STOP.json','run']):
        raise PermissionError('H4_GRANT_CONSUMED_OR_STOPPED')
