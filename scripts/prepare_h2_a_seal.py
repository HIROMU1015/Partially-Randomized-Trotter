"""Finalize existing H2 A66 metadata only; no grant activation or science."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

import h2_panels_contract as c
from prepare_h2_panels import preservation, primitive_count, row

SCIENCE_SOURCE = 'de72e5d4e48177734c0ddd79dffac14222410658'
FP_COMMIT = 'af724928689be3592b391fab3a14538bf4752b76'
D_COMMIT = '7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb'
D_SOURCE = 'cf99e4452df96291d7e9d37d76d2e91be5cbc413'
STORE = c.PRIVATE/'A_seal01'
PUBLIC = c.PUBLIC/'A_seal01'
MESSAGE = '既存のA66契約にFP・Dの依存SHA、結果監査、保存PF対照を結合し、seed衝突・SOURCE・call予算を再確認して最終sealして'
PENDING = ['SEPARATE_EXPLICIT_A_APPROVAL', 'FRESH_RESOURCE_ADMISSION',
           'SEED_INVENTORY_ACTIVATION_REFRESH']
TOOL_FILES = ('scripts/prepare_h2_a_seal.py', 'tests/test_h2_a_seal.py')


def verify_rows(rows):
    c.require(len({r['role'] for r in rows}) == len(rows), 'UNIQUE_SEAL_INPUT_ROLES')
    for r in rows:
        p = c.owned(r['path'])
        c.require(p.is_file() and p.stat().st_size == r['bytes'] and c.sha(p) == r['sha256'],
                  'SEAL_INPUT_CHANGED:'+r['role'])


def dependency_payload(stage, result, done, native, audited, accepted, result_hash):
    c.require(stage in ('FP', 'D') and result['stage'] == stage and result['binding'] == c.BINDING
              and result['status'] == done['status'] == audited['status'] == 'COMPLETE_MANDATORY_STOP',
              'COMPLETE_SAME_BINDING_DEPENDENCY')
    expected = SCIENCE_SOURCE if stage == 'FP' else D_SOURCE
    c.require(result['SOURCE'] == done['SOURCE'] == audited['SOURCE'] == expected
              and done['stage'] == audited['stage'] == stage, 'FIXED_DEPENDENCY_SOURCE')
    c.require(done['result_sha256'] == audited['result_sha256'] == accepted['result_sha256'] == result_hash
              and accepted.get('accepted_by_user') is True and bool(accepted.get('verbatim_user_message')),
              'ACCEPTED_DEPENDENCY_HASH')
    c.require(native['exit_code'] == done['native_exit_code'] == audited['native_exit_code'] == 0
              and all(x['same_parent_reaped'] and x['wait4_complete'] for x in (native, done, audited)),
              'DEPENDENCY_NATIVE_EXIT_WAIT4')


def inspect_dependency(stage, dep):
    for key in ('result', 'completion_audit', 'native_completion_audit', 'acceptance', 'native_exit'):
        c.require(c.sha(c.owned(dep[key])) == dep[key+'_sha256'], 'DEPENDENCY_BYTES:'+key)
    result, done, native, audited, accepted = [c.read(dep[k]) for k in
        ('result', 'completion_audit', 'native_exit', 'native_completion_audit', 'acceptance')]
    c.require(dep['sha256'] == dep['result_sha256'], 'LEGACY_DEPENDENCY_HASH_FIELD')
    dependency_payload(stage, result, done, native, audited, accepted, dep['sha256'])
    public = c.ROOT/dep['public_result']
    blob = subprocess.check_output(['git', '--no-optional-locks', 'show',
                                   dep['completion_commit']+':'+dep['public_result']], cwd=c.ROOT)
    c.require(hashlib.sha256(blob).hexdigest() == c.sha(public) == dep['sha256'],
              'DEPENDENCY_COMPLETION_COMMIT_BYTES')
    return result


def pf_payload(saved, result, result_hash):
    c.require(saved['D_result_sha256'] == result_hash and saved['binding'] == c.BINDING
              and saved['winners'] == result['accounting']['winners'], 'SAVED_PF_COUNTERPART_COPY')
    c.require(set(saved['winners']) == {str(e) for e in c.EPS}, 'FIVE_SAVED_PF_ENDPOINTS')
    for e, value in saved['winners'].items():
        c.require(value is not None and value in result['accounting']['rows'] and value['eligible']
                  and str(value['epsilon']) == e and value['cohort'] == 'DETERMINISTIC'
                  and value['n'] == 1 and value['G_RZ_exact'] is not None, 'SAVED_PF_ROW_IDENTITY')


def unchanged_science(spec, original):
    # All original values except resolved dependency/pending fields stay identical.
    for key, value in original.items():
        if key in ('inputs', 'dependencies', 'pending'):
            continue
        c.require(spec[key] == value, 'ORIGINAL_A_SEAL_CHANGED:'+key)
    c.require(spec['inputs'][:len(original['inputs'])] == original['inputs'], 'ORIGINAL_INPUT_ROWS_PRESERVED')
    c.require(spec['SOURCE'] == SCIENCE_SOURCE and spec['plan'] == c.plan('A')
              and spec['calls'] == c.budget('A', 90) and spec['primitive_probes'] == 90,
              'UNCHANGED_A66_SOURCE_GRID_BUDGET')
    c.require(spec['status'] == 'SEALED_PENDING_APPROVAL' and spec['pending'] == PENDING
              and spec['worker_cpus'] == [] and spec['resource_receipt'] is None,
              'NO_ACTIVATION_DURING_SEAL')


def seed_check(spec, destination):
    registry = c.read(spec['seed_registry']); fp = c.read(c.STORE/'FP_seeds.json')
    c.require(registry['tasks'] == c.seed_tasks('A', SCIENCE_SOURCE), 'ORIGINAL_A_SEEDS_UNCHANGED')
    new = c.seed_values(registry['tasks']); fp_values = c.seed_values(fp['tasks'])
    old = set(c.read(c.STORE/'old_seed_values.json')['values'])
    c.require((len(registry['tasks']), len(new), len(fp_values)) == (528, 3888, 4)
              and not new & (old | fp_values), 'A_FP_OLD_SEED_COLLISION')
    observations = []
    origins = c.read(c.PRIVATE/'FP_activation01/seed_origins.json')
    geometry = c.read(c.STORE/'seed_origins.json')[-1]
    for origin in [*origins, geometry]:
        p = Path(origin['origin']); frozen = Path(origin.get('snapshot', origin.get('path')))
        c.require(c.sha(frozen) == origin['sha256'], 'PRESERVED_SEED_SNAPSHOT_CHANGED')
        observed = c.sha(p)
        c.require(observed == origin['sha256'], 'REGISTERED_SEED_METADATA_CHANGED_RESEAL_REQUIRED')
        observations.append(dict(role=origin['role'], origin=str(p), snapshot=str(frozen),
                                 observed_sha256=observed, changed=False))
    prior = c.read(spec['seed_check'])
    c.require(prior['new_values_digest'] == c.digest(sorted(new))
              and prior['joint_FP_A_values_digest'] == c.digest(sorted(new | fp_values))
              and all(not r['collisions'] for r in prior['geometry_checks'])
              and sum(r['old_seed_slots_checked'] for r in prior['geometry_checks']) == 7737216,
              'GEOMETRY_PROOF_COVERS_UNCHANGED_A_SEEDS')
    c.save(destination/'known_seed_observations.json', observations)
    c.save(destination/'fresh_seed_exclusions.json', dict(values=sorted(old | fp_values),
                                                       own_A_reservation_excluded_from_exclusions=True))
    receipt = dict(status='COLLISION_ZERO', checked_at_UTC=datetime.now(timezone.utc).isoformat(),
        new_values_digest=c.digest(sorted(new)), seed_registry_sha256=c.sha(spec['seed_registry']),
        master_seeds=528, new_seed_slots=3888, old_explicit_values=len(old),
        FP_diagnostic_values=len(fp_values), exclusion_values=len(old | fp_values),
        known_campaigns_checked=[r['role'] for r in observations],
        known_origin_observations=len(origins), geometry_registration_unchanged=True,
        geometry_case_sha256=geometry['sha256'], geometry_checks=prior['geometry_checks'],
        geometry_reserved_slots=7737216, reused_geometry_collision_proof_sha256=c.sha(spec['seed_check']),
        fresh_exclusions_sha256=c.sha(destination/'fresh_seed_exclusions.json'),
        observations_sha256=c.sha(destination/'known_seed_observations.json'),
        inherited_inventory_cutoff=prior['inherited_inventory_cutoff'],
        metadata_rechecked_without_RTE_sampling=True, seed_redrawn=False, new_RTE_sampling=0,
        future_unregistered_campaigns_covered=False, activation_refresh_required=True,
        unrelated_HEAD_status_frozen=False, zero_collision_is_independence_proof=False)
    c.save(destination/'seed_refresh.json', receipt)
    return receipt


def verify(spec, *, runtime=True):
    original = c.owned(spec['original_pending_spec'])
    c.require(c.sha(original) == spec['original_pending_spec_sha256'], 'ORIGINAL_PENDING_BYTES')
    unchanged_science(spec, c.read(original)); c.validate_metadata(spec); verify_rows(spec['inputs'])
    paths = {r['path']:r['sha256'] for r in spec['inputs']}
    used = [spec[k] for k in ('original_pending_spec', 'dependency_bindings', 'saved_PF_controls',
                              'seed_refresh', 'fresh_seed_exclusions')]
    used += spec['preservation_manifests']
    used += [dep[k] for dep in spec['dependencies'].values() for k in
             ('result', 'completion_audit', 'native_completion_audit', 'native_exit', 'acceptance')]
    c.require(all(paths.get(p) == c.sha(c.owned(p)) for p in used), 'ACTUALLY_USED_SEAL_INPUTS')
    for rel, expected in spec['seal_tool_hashes'].items():
        blob = subprocess.check_output(['git', '--no-optional-locks', 'show',
                                      spec['seal_tool_SOURCE']+':'+rel], cwd=c.ROOT)
        c.require(c.sha(c.ROOT/rel) == hashlib.sha256(blob).hexdigest() == expected, 'SEAL_TOOL_SOURCE')
    expected = c.read(spec['dependency_bindings'])
    c.require(set(spec['dependencies']) == {'FP', 'D'} and spec['dependencies'] == expected,
              'FROZEN_ACTUAL_DEPENDENCY_BINDINGS')
    results = {stage: inspect_dependency(stage, dep) for stage, dep in expected.items()}
    pf_payload(c.read(spec['saved_PF_controls']), results['D'], expected['D']['sha256'])
    receipt = c.read(spec['seed_refresh'])
    c.require(receipt['status'] == 'COLLISION_ZERO' and receipt['activation_refresh_required']
              and receipt['seed_redrawn'] is False and receipt['new_RTE_sampling'] == 0
              and receipt['new_values_digest'] == c.digest(sorted(c.seed_values(c.read(spec['seed_registry'])['tasks'])))
              and receipt['seed_registry_sha256'] == c.sha(spec['seed_registry']), 'SEALED_SEED_REFRESH')
    excluded = set(c.read(spec['fresh_seed_exclusions'])['values'])
    c.require(not excluded & c.seed_values(c.read(spec['seed_registry'])['tasks']), 'FRESH_EXPLICIT_SEED_COLLISION')
    for path in spec['preservation_manifests']:
        for r in c.read(path):
            c.require(c.sha(r['path']) == r['sha256'], 'EXISTING_EVIDENCE_PRESERVED')
    before = preservation()
    if runtime:
        for path, h in c.read(spec['runtime_manifest']).items():
            c.require(c.sha(path) == h, 'FIXED_RUNTIME_BYTES')
    c.require(not Path(spec['attempt']).exists() and not (c.PRIVATE/'A_activation01').exists(),
              'A_ONE_SHOT_NOT_STARTED')
    return dict(status='FINAL_SEAL_MANDATORY_STOP', stage='A', SOURCE=spec['SOURCE'],
        seal_tool_SOURCE=spec['seal_tool_SOURCE'], candidates=66, n=8,
        dependency_result_sha256={s:d['sha256'] for s,d in expected.items()},
        saved_PF_controls_sha256=c.sha(spec['saved_PF_controls']), calls=spec['calls'],
        source_files=len(spec['source_hashes']), provider_files=len(spec['provider_files']),
        kernel_files=len(spec['kernel_files']), runtime_files=len(c.read(spec['runtime_manifest'])),
        F_records_preserved=before['records_checked'], FP_files_preserved=204,
        original_D_files_preserved=541, completed_D_files_preserved=314,
        seeds=dict(masters=528, integers=3888, exclusions=len(excluded), collisions=0,
                   geometry_reserved_slots=7737216, seed_redrawn=False),
        pending=spec['pending'], valid_grant_created=False, A_attempt_created=False,
        scientific_execution=0, new_RTE_sampling=0, compile=0, cost_reaccounting=0,
        independent_scientific_review=False, next_stage_authorized=False, push_performed=False)


def seal(tool_source):
    c.require(len(tool_source) == 40, 'FULL_COMMITTED_SEAL_TOOL_SOURCE')
    for rel in TOOL_FILES:
        blob = subprocess.check_output(['git', '--no-optional-locks', 'show', tool_source+':'+rel], cwd=c.ROOT)
        c.require(hashlib.sha256(blob).hexdigest() == c.sha(c.ROOT/rel), 'COMMITTED_SEAL_TOOL_BYTES')
    original = c.STORE/'A_spec_PENDING.json'; original_spec = c.read(original)
    c.validate_metadata(original_spec)
    c.require(primitive_count(c.registrations()['A'], original_spec['native_config']['provider_root']) == 90,
              'STRUCTURAL_PRIMITIVE_CALL_COUNT')
    c.require(not Path(original_spec['attempt']).exists(), 'A_NOT_STARTED')
    STORE.mkdir(exist_ok=False); PUBLIC.mkdir(exist_ok=False)
    preservation()
    # D's existing metadata auditor has no numerical kernel and still permits unstarted A.
    from audit_h2_d_recovery_completed import audit
    c.save(STORE/'D_dependency_reaudit.json', audit())
    c.save(STORE/'D_acceptance.json', dict(role='user', stage='D', accepted_by_user=True,
        completion_commit=D_COMMIT, result_sha256=c.sha(c.PRIVATE/'attempts/D_run02/output/result.json'),
        verbatim_user_message=MESSAGE, scope='Accept completed D as dependency of the requested A final seal',
        A_scientific_execution_authorized=False, push_authorized=False))
    dependencies = {}
    for stage, attempt, activation, acceptance, commit in [
        ('FP', 'FP_run01', 'FP_activation01', c.PRIVATE/'D_activation01/FP_acceptance.json', FP_COMMIT),
        ('D', 'D_run02', 'D_recovery_activation01', STORE/'D_acceptance.json', D_COMMIT)]:
        run = c.PRIVATE/'attempts'/attempt
        paths = dict(result=run/'output/result.json', completion_audit=run/'completion_audit.json',
                     native_completion_audit=c.PRIVATE/activation/'native_completion_audit.json',
                     native_exit=run/'native_exit.json', acceptance=acceptance)
        dep = dict(completion_commit=commit, public_result=str((c.PUBLIC/
            ('FP_completed_run01' if stage == 'FP' else 'D_completed_run02')/'result.json').relative_to(c.ROOT)))
        for key, path in paths.items():
            dep[key] = str(path); dep[key+'_sha256'] = c.sha(path)
        dep['sha256'] = dep['result_sha256']; inspect_dependency(stage, dep); dependencies[stage] = dep
    c.save(STORE/'dependency_bindings.json', dependencies)
    d = c.read(dependencies['D']['result'])
    c.save(STORE/'saved_PF_controls.json', dict(binding=c.BINDING, D_completion_commit=D_COMMIT,
        D_result_sha256=dependencies['D']['sha256'], winners=d['accounting']['winners'],
        copied_from_saved_accounting=True, cost_reaccounting=False, new_PF_selection=False))
    seed_check(original_spec, STORE)
    spec = deepcopy(original_spec)
    manifests = [c.PRIVATE/'FP_activation01/FP_completed_preservation_manifest_PRIVATE.json',
        c.PRIVATE/'D_interruption_audit01/original_D_preservation_manifest_PRIVATE.json',
        c.PRIVATE/'D_recovery_activation01/D_completed_preservation_manifest_PRIVATE.json',
        c.STORE/'preparation_preservation.json']
    spec.update(dependencies=dependencies, pending=PENDING, final_A_seal_version=1,
        original_pending_spec=str(original), original_pending_spec_sha256=c.sha(original),
        dependency_bindings=str(STORE/'dependency_bindings.json'),
        saved_PF_controls=str(STORE/'saved_PF_controls.json'),
        seed_refresh=str(STORE/'seed_refresh.json'), seed_refresh_sha256=c.sha(STORE/'seed_refresh.json'),
        fresh_seed_exclusions=str(STORE/'fresh_seed_exclusions.json'),
        preservation_manifests=[str(p) for p in manifests], seal_tool_SOURCE=tool_source,
        seal_tool_hashes={rel:c.sha(c.ROOT/rel) for rel in TOOL_FILES},
        activation_rule='Verify this immutable final pending spec before activation; preserve dependencies and inputs; bind separate user approval, fresh resources and fresh seed receipt',
        actual_scientific_execution=0)
    additions = [(original, 'original_A_pending_spec'), *[(p, 'preservation_'+str(i)) for i,p in enumerate(manifests)]]
    for stage, dep in dependencies.items():
        additions += [(Path(dep[k]), stage+'_'+k) for k in
                      ('result', 'completion_audit', 'native_completion_audit', 'native_exit', 'acceptance')]
    additions += [(p, p.stem) for p in sorted(STORE.iterdir()) if p.is_file()]
    # D_acceptance is already present as a dependency row; bind every path once.
    seen = {r['path'] for r in spec['inputs']}
    for path, role in additions:
        if str(path) not in seen:
            spec['inputs'].append(row(path, role)); seen.add(str(path))
    result = verify(spec)
    pending = STORE/'A_spec_FINAL_PENDING.json'; c.save(pending, spec)
    c.save(STORE/'final_seal_audit.json', result)
    c.save(PUBLIC/'A_INVALID_GRANT.json', c.invalid_grant('A'))
    for name in ('dependency_bindings.json', 'saved_PF_controls.json', 'seed_refresh.json',
                 'D_acceptance.json', 'D_dependency_reaudit.json', 'final_seal_audit.json'):
        (PUBLIC/name).write_bytes((STORE/name).read_bytes())
    c.save(PUBLIC/'A_final_preregistration.json', dict(SOURCE=spec['SOURCE'], seal_tool_SOURCE=tool_source,
        final_pending_spec_sha256=c.sha(pending), final_pending_spec_digest=c.digest(spec),
        original_pending_spec_sha256=c.sha(original), plan=spec['plan'], calls=spec['calls'],
        source_hashes=spec['source_hashes'], seal_tool_hashes=spec['seal_tool_hashes'],
        inputs=spec['inputs'], dependencies=dependencies, pending=PENDING,
        runtime_manifest_sha256=spec['runtime_manifest_sha256'],
        seed_registry_sha256=c.sha(spec['seed_registry']), seed_refresh_sha256=spec['seed_refresh_sha256'],
        valid_grant=False, scientific_execution=0, automatic_retry=False, mandatory_STOP=True))
    c.save(STORE/'seal_preservation_manifest_PRIVATE.json', [row(p, str(p.relative_to(STORE)))
        for p in sorted(STORE.rglob('*')) if p.is_file()])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seal', action='store_true'); group.add_argument('--verify', action='store_true')
    parser.add_argument('--tool-source')
    args = parser.parse_args()
    if args.seal:
        c.require(args.tool_source, 'COMMITTED_SEAL_TOOL_SOURCE_REQUIRED'); result = seal(args.tool_source)
    else:
        result = verify(c.read(STORE/'A_spec_FINAL_PENDING.json'))
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    main()
