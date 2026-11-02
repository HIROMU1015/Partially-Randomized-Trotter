"""Seal final H6 metadata and INVALID grants only; no science dispatch option."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h6_pr_confirmation as h
import h6_pr_confirmation_seal as final
import prepare_h6_pr_confirmation as old
c = h.c
PROJECTS = Path('/home/AbeHiromu/projects')
PHASE0 = PROJECTS / 'h6-pr-confirmation-preparation/20261019/phase0-audit02'
HIGHER = PROJECTS / 'h46-higher-pf-confirmation-native/20261018/private'
H4 = PROJECTS / 'partially-randomized-trotter-worktrees/h4-p-seal-20261019/.h4_p_seal_private'


def campaign_sources():
    fixed = PROJECTS / 'h6-handoff-evidence/20261015/h6-fixed-delta-readiness'
    return {
        'H6_LEGACY_AND_A128': [(PHASE0 / 'seed_exclusions_PRIVATE.json', 'values')],
        'H6_FIXED_DELTA_RESERVATIONS': [(fixed / f'materialized0{i}/config.json', 'fixed_delta') for i in range(1, 6)],
        'H6_CONDITIONAL': [(PROJECTS / 'h6-dba-validation/20261018/private/B_attempt02/output/seed_registry.json', 'tasks')],
        'HIGHER_PF_LEGACY_AND_ACQUISITION': [(HIGHER / 'run01/excluded_seeds_PRIVATE.json', 'values'),
            (PROJECTS / 'h46-higher-pf-acquire/20261018/private/seed_registry_PRIVATE.json', 'flat_registry')],
        'H6_HIGHER_PF_CONFIRMATION_RESERVED': [(HIGHER / 'proposal01/seed_registry_PROVISIONAL_PRIVATE.json', 'flat_registry')],
        'H6_HIGHER_PF_CONFIRMATION_USED': [(HIGHER / 'run01/seed_registry_PRIVATE.json', 'flat_registry')],
        'H8_NEW_A_AND_CONFIRMATION': [(PROJECTS / 'h8-new-pr-confirmation/20261018/sealed01/seed_registry.json', 'tasks')],
        'H4_RECORDED_OLD_SEEDS': [(H4 / 'old_seed_exclusions_PRIVATE.json', 'values')],
        'H4_P_CONFIRMATION': [(H4 / 'new_confirmation_seeds_PRIVATE.json', 'all_master_and_step_values')],
    }


def collect_inventory(destination, sources=None):
    campaigns, origins = {}, []
    store = destination / 'seed_metadata_store'
    store.mkdir()
    for campaign, pairs in (sources if sources is not None else campaign_sources()).items():
        rows = []
        for index, (path, mode) in enumerate(pairs):
            origin = old.byte_row(path, f'seed_origin:{campaign}:{index}')
            origins.append(origin)
            copy = store / origin['sha256']
            if not copy.exists():
                copy.write_bytes(Path(path).read_bytes())
            row = old.byte_row(copy, f'seed_snapshot:{campaign}:{index}')
            c.require(row['sha256'] == origin['sha256'], 'SEED_SNAPSHOT_BYTES')
            rows.append(dict(row, seed_mode=mode))
        campaigns[campaign] = rows
    inventory = dict(schema='H6_FINAL_SEED_INVENTORY_V1',
        known_registered_campaigns_complete=True, campaigns=campaigns,
        cutoff_UTC=datetime.now(timezone.utc).isoformat(),
        scope='KNOWN_FIXED_H6_CAMPAIGNS_AND_RECORDED_H4_H8_REGISTRIES; NOT_UNKNOWN_JOBS',
        future_campaign_policy='Later allocations must exclude this sealed registry; refresh only the exclusion audit before launch, never redraw.',
        reserved_or_stopped_identifiers_retained=True,
        mathematical_PRNG_independence_proved=False, actual_event_sampling=0)
    values = final.inventory_values(inventory)
    c.save(destination / 'seed_inventory_PRIVATE.json', inventory)
    return inventory, values, origins


def prepare(destination, source):
    destination = Path(destination).resolve()
    c.require(destination.is_relative_to(PROJECTS), 'OWN_PROJECT_DESTINATION')
    source_hashes = {name: c.sha(h.ROOT / name) for name in h.FILES}
    c.verify_source(h.ROOT, source, source_hashes)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = c.read(PHASE0 / 'input_manifest_PRIVATE.json')
    c.verify_byte_rows(manifest['inputs'])
    plan = c.read(h.ROOT / h.PLAN)
    h.verify_proposal(plan)
    stats_path = h.ROOT / 'artifacts/resource_applicability/track_a_h6_pr_confirmation_s_stats/2026-10-19/result.json'
    c.require(c.sha(stats_path) == 'e51f91d9db4fae2d48547a575331b75af0a07f788f6af04af5ae0b223c817a87',
              'ACCEPTED_S_STATS_RESULT_BYTES_ONLY')
    selection = dict(schema='H6_FINAL_SELECTION_V1', targets=plan['targets'], n_by_scope=final.FINAL_N,
        design_digest=c.digest(plan), S_STATS_result_sha256=c.sha(stats_path),
        n_adopted_from_prior_recommendation=True,
        authorization='2026-10-19 user: fix final n/PF/direction, seeds, SOURCE, calls and invalid grant; mandatory STOP',
        scientific_execution_authorized=False, public_push_authorized=False)
    c.save(destination / 'final_selection.json', selection)
    inventory, excluded, origins = collect_inventory(destination)
    inputs = deepcopy(manifest['inputs'])
    inputs += [r for rows in inventory['campaigns'].values() for r in rows]
    inputs += [old.byte_row(destination / 'seed_inventory_PRIVATE.json', 'final:seed_inventory'),
               old.byte_row(destination / 'final_selection.json', 'final:selection'),
               old.byte_row(stats_path, 'final:accepted_S_STATS')]
    # Fresh observation of exact known runtime leaves. Unchanged stat facts reuse
    # the prior full SHA audit; changed leaves are rehashed, never trusted blindly.
    origin = c.read(next(r['path'] for r in manifest['inputs'] if r['role'] == 'accounting:origin_A_spec'))
    runtime = old.runtime_audit(origin, destination,
        cache_path=manifest['runtime']['path'])
    inputs.append(old.byte_row(runtime['path'], 'final:runtime_receipt'))
    seed_plan = dict(SOURCE=source, design_digest=c.digest(plan), cohort=h.COHORT,
        namespace=h.NAMESPACE, targets=plan['targets'], n_by_scope=final.FINAL_N,
        seed_inventory_sha256=c.sha(destination / 'seed_inventory_PRIVATE.json'),
        final_selection_sha256=c.sha(destination / 'final_selection.json'),
        source_hashes=source_hashes, immutable_inputs=deepcopy(inputs))
    c.save(destination / 'seed_plan_PRIVATE.json', seed_plan)
    tasks = h.derive_registry(plan['targets'], source, c.digest(seed_plan), excluded, final.FINAL_N)
    fresh = [s for t in tasks for s in h.shared.hierarchy(t['seed'], t['cell']['q'])]
    c.require(len(tasks) == 152 and len(fresh) == len(set(fresh)) == 1080
              and excluded.isdisjoint(fresh), 'EXACT_FINAL_SEED_COLLISION_AUDIT')
    registry_path = destination / 'seed_registry_PRIVATE.json'
    c.save(registry_path, dict(schema='H6_FINAL_INDEPENDENT_SEED_REGISTRY_V1', SOURCE=source,
        design_digest=c.digest(plan), seed_plan_digest=c.digest(seed_plan), n_by_scope=final.FINAL_N,
        seed_inventory_sha256=c.sha(destination / 'seed_inventory_PRIVATE.json'),
        excluded=sorted(excluded), excluded_inventory_complete=True, tasks=tasks,
        old_cohorts_not_reused=True, redraw_on_collision=False, actual_RTE_event_sampling=0))
    inputs += [old.byte_row(destination / 'seed_plan_PRIVATE.json', 'final:seed_plan'),
               old.byte_row(registry_path, 'final:registry')]
    specs = []
    for target in plan['targets']:
        scope, n = target['scope'], final.FINAL_N[target['scope']]
        directory = destination / scope
        directory.mkdir()
        spec = dict(schema='H6_CONFIRMATION_EXECUTION_V1', scope=scope, SOURCE=source,
            target=target, binding=plan['binding'], n=n, n_by_scope=final.FINAL_N, cohort=h.COHORT,
            design_digest=c.digest(plan), seed_plan_digest=c.digest(seed_plan),
            seed_inventory=str(destination / 'seed_inventory_PRIVATE.json'),
            seed_plan=str(destination / 'seed_plan_PRIVATE.json'),
            final_selection=str(destination / 'final_selection.json'),
            registry=str(registry_path), registry_sha256=c.sha(registry_path),
            calls=h.calls_for(target, n), shots=target['shots'], law=target['law'], payload=manifest['payload'],
            source_hashes=source_hashes, inputs=inputs, provider_closure=manifest['provider_closure'],
            runtime_receipt=runtime['path'], runtime_receipt_sha256=runtime['sha256'],
            python=manifest['python'], python_sha256=manifest['python_sha256'],
            profile_reference=manifest['profile_reference'],
            output=str(directory / 'output'), lock=str(directory / 'ONE_SHOT.json'),
            output_estimate_bytes=2**31, inode_estimate=16384,
            memory_estimate_per_process_bytes=6*2**30,
            memory_estimate_source='Copied H6 A_COST conservative admission estimate, not a resource cap; refresh resource allocation before launch.',
            approved=False, runtime_authorization=False, technical_sealed=True,
            retry=False, next_stage_authorized=False, resource_caps=None, allowed_cpus=[],
            absolute_launch_command=None)
        spec['technical_digest'] = h.shared.technical_digest(spec)
        spec_path = directory / 'technical_seal_PRIVATE.json'
        c.save(spec_path, spec)
        verification = h.verify(spec)
        grant_path = directory / 'grant_INVALID.json'
        grant = dict(scope=scope, SOURCE=source, technical_digest=spec['technical_digest'],
            approved=False, approved_by_user=False, runtime_authorization=False, one_shot=True,
            retry=False, next_stage_authorized=False, approval_receipt=None, approval_receipt_sha256=None,
            spec_path=str(spec_path), grant_path=str(grant_path),
            allowed_cpus=[], driver_cpu=None, concurrent_job_cpus=None,
            memory_admission_estimate_bytes=None, quota_policy=None, absolute_launch_command=None)
        c.save(grant_path, grant)
        specs.append(dict(scope=scope, n=n, path=str(spec_path), sha256=c.sha(spec_path),
            technical_digest=spec['technical_digest'], verification=verification))
    c.verify_byte_rows(origins)
    c.save(destination / 'seed_origin_preservation_PRIVATE.json', origins)
    c.save(destination / 'resource_observation_PRIVATE.json', h.shared.effective_fresh(destination))
    calls = {t['scope']: h.calls_for(t, final.FINAL_N[t['scope']]) for t in plan['targets']}
    c.save(destination / 'science_approval_INVALID.json', dict(stage='H6_CONFIRMATION_SCIENCE',
        approved_by_user=False, runtime_authorization=False, SOURCE=source,
        n_by_scope=final.FINAL_N, seed_registry_sha256=c.sha(registry_path), calls_by_scope=calls,
        scopes=list(final.FINAL_N), retry=False, next_stage_authorized=False,
        mandatory_stop=True, resource_admission='FRESH_RESOURCE_AND_SEED_RECHECK_REQUIRED'))
    receipt = dict(status='SEALED_TECHNICAL_ONLY_MANDATORY_STOP', SOURCE=source, specs=specs,
        seed_registry_sha256=c.sha(registry_path), seed_inventory_sha256=c.sha(destination / 'seed_inventory_PRIVATE.json'),
        seed_plan_digest=c.digest(seed_plan), runtime_receipt_sha256=runtime['sha256'],
        source_files=len(source_hashes), input_rows=len(inputs), runtime_files=runtime['files'],
        provider_files=sum(len(v) for v in manifest['provider_closure'].values()),
        excluded_unique_seed_values=len(excluded), new_masters=152, new_master_and_step_values=1080,
        collisions=0, calls_by_scope=calls,
        total_calls={k: sum(v[k] for v in calls.values()) for k in next(iter(calls.values()))},
        valid_grants_created=0, actual_scientific_calls=0, statistics_recomputed=0,
        output_directories_created=0, science_ledgers_or_markers_created=0,
        next_stage_authorized=False, pushed=False)
    c.save(destination / 'preparation_receipt_PRIVATE.json', receipt)
    c.save(destination / 'STOP.json', dict(reason='FINAL_TECHNICAL_SEAL_COMPLETE_SCIENCE_NOT_AUTHORIZED',
        mandatory_stop=True, next_stage_authorized=False, retry=False))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    result = prepare(args.destination, args.source)
    print(result['status'])
