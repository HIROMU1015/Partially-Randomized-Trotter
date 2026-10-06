#!/usr/bin/env python3
"""Preparation-only identity/JSON helper. Never launches scientific processing."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[2]
BUNDLE = 'artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06'
SCIENCE_ROOT = Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-parallel-source-20261006')
SOURCE = '6a121725ce751affd2d3d131a84944728e6b2343'
REVIEW = '88461f3930b9fef511739f91edae88231c33a3f5'
SOURCE_AUDIT_SHA = '5c9a1997339fa0f1f5479c62b11b6e2ef2ee024ce5a584aa958cfa80c4addd5f'
PARALLEL_MANIFEST_SHA = 'a9cb2b03a9ff505b5964c7740818b6b41df3f7176105baadd52108330a0ad53a'
STATUS = 'H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW'


def production_modules():
    # Read the reviewed execution checkout, never the preparation checkout copy.
    sys.path.insert(0, str(SCIENCE_ROOT/'src'))
    from trottertracks.resource_applicability.h4_geometry import gates, identity, resources
    identity.require(Path(gates.__file__).absolute() == SCIENCE_ROOT/'src/trottertracks/resource_applicability/h4_geometry/gates.py', 'actual loaded science checkout')
    return gates, identity, resources


def git(*args, root=SCIENCE_ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def verify_identity():
    gates, identity, _resources = production_modules()
    sha, require = identity.sha, identity.require
    require(git('rev-parse', 'HEAD').decode().strip() == REVIEW, 'review checkout HEAD')
    require(git('remote', 'get-url', 'origin').decode().strip() == 'https://github.com/HIROMU1015/Partially-Randomized-Trotter', 'origin identity')
    require(not git('diff', '--name-only') and not git('diff', '--cached', '--name-only'), 'science tracked worktree changed')
    for older, newer in ((gates.BASE, SOURCE), (SOURCE, REVIEW)):
        require(subprocess.run(['git', '-C', str(SCIENCE_ROOT), 'merge-base', '--is-ancestor', older, newer]).returncode == 0, 'commit ancestry')
    audit_raw = (SCIENCE_ROOT/gates.SOURCE_AUDIT).read_bytes()
    require(sha(audit_raw) == SOURCE_AUDIT_SHA, 'source audit bytes')
    audit = json.loads(audit_raw)
    require(audit['source_commit'] == SOURCE and audit['status'] == 'SOURCE_BLOBS_VERIFIED', 'source audit identity')
    require(len(audit['new_source_hashes']) == 17 and len(audit['namespace_parent_hashes']) == 2, 'full source closure')
    closure = {**audit['new_source_hashes'], **audit['namespace_parent_hashes']}
    for path, expected in closure.items():
        require(path.endswith('.py') and sha(gates.git_blob(SCIENCE_ROOT, SOURCE, path)) == expected, 'SOURCE blob')
        require(sha((SCIENCE_ROOT/path).read_bytes()) == expected, 'actual source checkout hash')
        require(sha((ROOT/path).read_bytes()) == expected, 'preparation copy source unchanged')
    manifest_path = str(Path(gates.SOURCE_AUDIT).parent/'artifact_manifest_v1.json')
    manifest_raw = (SCIENCE_ROOT/manifest_path).read_bytes()
    require(sha(manifest_raw) == PARALLEL_MANIFEST_SHA, 'parallel manifest bytes')
    manifest = json.loads(manifest_raw)
    for entry in manifest['files']:
        commit = SOURCE if entry['commit_stage'] == 'SOURCE_COMMIT' else REVIEW
        require(sha(gates.git_blob(SCIENCE_ROOT, commit, entry['path'])) == entry['sha256'], 'parallel manifest blob')
        require(sha((SCIENCE_ROOT/entry['path']).read_bytes()) == entry['sha256'], 'parallel checkout artifact')
    contract = gates.verify_contract(SCIENCE_ROOT)
    observed = gates.environment_matches(audit['dependency_observations'])
    options = gates.compiler_matches(contract['compiler_environment_reference']['compiler'])
    for path, expected in audit['installed_source_hashes'].items():
        require(path.endswith('.py') and sha(Path(path).read_bytes()) == expected, 'installed reviewed source')
    bundles = {}
    for directory, count in (
        ('artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05', 25),
        ('artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06', 26),
        (gates.CONTRACT, 30),
        ('artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06', 28),
        (str(Path(gates.SOURCE_AUDIT).parent), 11)):
        paths = git('ls-tree', '-r', '--name-only', '-z', REVIEW, '--', directory).decode().split('\0')[:-1]
        require(len(paths) == count, 'old bundle inventory')
        hashes = {}
        for path in paths:
            expected = sha(gates.git_blob(SCIENCE_ROOT, REVIEW, path))
            require(sha((SCIENCE_ROOT/path).read_bytes()) == expected and sha((ROOT/path).read_bytes()) == expected, 'old bundle modified')
            hashes[path] = expected
        bundles[directory] = {'files':count, 'byte_identical':True, 'sha256':hashes}
    static_path = 'artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05/static_audit_v0.json'
    static = json.loads(gates.git_blob(SCIENCE_ROOT, gates.BASE, static_path))
    for entry in static['source_hashes'] + static['allowed_json_identity']:
        require(entry['path'].endswith(('.py','.json')), 'allowed old evidence path')
        require(sha((SCIENCE_ROOT/entry['path']).read_bytes()) == entry['sha256'], 'old source/evidence changed')
        require(sha((ROOT/entry['path']).read_bytes()) == entry['sha256'], 'old preparation source/evidence changed')
    require(len(static['source_hashes']) == 247 and len(static['allowed_json_identity']) == 6, 'old inventory')
    # The previous transpile ledger is read only; no Qiskit compiler invocation.
    require(audit['final_synthetic_tests']['synthetic_transpile_cumulative'] == 28, 'frozen synthetic count')
    report = {'schema_version':'h4-input-generation-identity-audit-v1', 'status':STATUS,
        'observed_utc':datetime.now(timezone.utc).isoformat(), 'branch_base_commit':REVIEW,
        'contract_base_commit':gates.BASE, 'science_source_commit':SOURCE,
        'preparation_checkout_root':str(ROOT), 'actual_science_checkout_root':str(SCIENCE_ROOT),
        'artifact_anchor':gates.ARTIFACT_ANCHOR, 'future_output_root':gates.OUTPUT,
        'source_audit_sha256':SOURCE_AUDIT_SHA, 'parallel_manifest_sha256':PARALLEL_MANIFEST_SHA,
        'contract_plan_sha256':gates.PLAN_SHA, 'contract_plan_fingerprint':gates.PLAN_FP,
        'contract_manifest_sha256':gates.MANIFEST_SHA, 'contract_manifest_entries_verified':37,
        'science_source_and_parent_hashes':closure, 'science_source_and_parent_paths_verified':19,
        'installed_source_hashes':audit['installed_source_hashes'], 'installed_source_paths_verified':11,
        'dependency_count':len(observed), 'dependency_observations':observed, 'dependency_mismatches':[],
        'compiler_options':options, 'compiler_defaults_plugins_match':True,
        'compiler_fingerprint':audit['compiler_fingerprint'], 'environment_fingerprint':audit['environment_fingerprint'],
        'old_bundles':bundles, 'old_sources_verified':247, 'saved_JSON_verified':6,
        'additional_transpile':0, 'preserved_source_series_transpile_cumulative':28,
        'science_source_modified':False, 'excluded_scientific_paths_inspected':False,
        'full_binary_independent_reproducibility_claimed':False, 'mandatory_stop':True}
    return contract, audit, report


def documents(contract, audit, allowed_cpus):
    gates, identity, _resources = production_modules()
    identity.require(type(allowed_cpus) is list and all(type(c) is int and c >= 0 for c in allowed_cpus)
                     and len(set(allowed_cpus)) == len(allowed_cpus), 'explicit CPU draft list')
    plan = {'schema_version':'h4-native-execution-plan-v1', 'stage':'input_generation',
        'run_id':gates.RUN_ID, 'base_commit':gates.BASE, 'contract_plan_fingerprint':gates.PLAN_FP,
        'source_commit':SOURCE, 'source_hashes':{**audit['new_source_hashes'], **audit['namespace_parent_hashes']},
        'source_audit_sha256':SOURCE_AUDIT_SHA, 'source_root':str(SCIENCE_ROOT),
        'artifact_anchor':gates.ARTIFACT_ANCHOR, 'output_root':gates.OUTPUT,
        'distances':list(gates.DISTANCES), 'requested_workers':6, 'binding':'SOURCE_BOUND',
        'inputs':None, 'generation_freeze_digest':None, 'templates':contract['templates'],
        'compiler_fingerprint':audit['compiler_fingerprint'], 'environment_fingerprint':audit['environment_fingerprint']}
    plan_fp = identity.fingerprint('h4-execution-plan-v1', plan)
    authorization = {'schema_version':'h4-native-authorization-v1', 'stage':'input_generation',
        'run_id':gates.RUN_ID, 'permission':'input_generation', 'one_shot':True, 'result_prior':True,
        'plan_fingerprint':plan_fp, 'allowed_cpus':list(allowed_cpus)}
    review = {'schema_version':'h4-native-stage-review-v1', 'stage':'input_generation',
        'run_id':gates.RUN_ID, 'approved':False, 'plan_fingerprint':plan_fp,
        'authorization_digest':identity.fingerprint('h4-authorization-v1', authorization)}
    gates.structural_gate(plan, authorization, review)
    return plan, authorization, review


def resource_record(allowed_cpus, explicit_cpu_evidence):
    gates, identity, resources = production_modules()
    cpu_raw = next(line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('Cpus_allowed_list:'))
    process_cpus = sorted(resources.cpus(cpu_raw))
    host_available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                              if line.startswith('MemAvailable:')).split()[1])*1024
    utc = datetime.now(timezone.utc).isoformat()
    try:
        memory = resources.observe_memory()  # read-only metadata; never admission/launch
        memory['process_cpus'] = sorted(memory['process_cpus'])
        memory_status, memory_error = 'OBSERVED_PREPARATION_METADATA_ONLY', None
    except (identity.Stop, OSError, ValueError, KeyError) as exc:
        memory, memory_status, memory_error = None, 'UNRESOLVED_FRESH_LAUNCH_OBSERVATION_REQUIRED', str(exc)
    evidence = explicit_cpu_evidence if allowed_cpus else None
    identity.require(not allowed_cpus or bool(evidence), 'explicit CPU evidence required')
    subset = set(process_cpus) <= set(allowed_cpus) if allowed_cpus else False
    unresolved = ['FINAL_STAGE_REVIEW_NOT_APPROVED', 'USER_EXPLICIT_LAUNCH_NOT_PERFORMED', 'FRESH_LAUNCH_MEMORY_AND_PRESSURE_REQUIRED']
    if not allowed_cpus:
        unresolved.append('EXPLICIT_CPU_PERMISSION_UNCONFIRMED')
    if not subset:
        unresolved.append('LAUNCH_PROCESS_CPUS_MUST_BE_SUBSET_OF_EXPLICIT_ALLOWED_CPUS')
    if memory is None:
        unresolved.append('PREPARATION_MEMORY_OBSERVATION_FAILED: '+memory_error)
    return {'schema_version':'h4-input-generation-resource-review-v1', 'status':STATUS,
        'observed_utc':utc, 'preparation_checkout_root':str(ROOT), 'science_checkout_root':str(SCIENCE_ROOT),
        'requested_workers':6, 'generation_independent_tasks':6, 'contract_worker_cap':12,
        'signal_compile_requested_workers':None, 'actual_admitted_workers':None,
        'allowed_cpus':list(allowed_cpus), 'explicit_cpu_permission_evidence':evidence,
        'cpu_permission_status':'EXPLICIT_LIST_PROVIDED_REVIEW_PENDING' if allowed_cpus else 'UNCONFIRMED_NOT_GRANTED',
        'observed_process_cpu_list':cpu_raw, 'observed_process_cpus':process_cpus,
        'candidate_cpu_list_for_review_only':process_cpus[:6],
        'observation_is_permission':False, 'process_cpus_subset_allowed_cpus':subset,
        'launch_context_proposal':'After explicit CPU approval and final review, a separately approved launch context must expose only approved CPUs. No affinity/cgroup/job changes are made by this preparation.',
        'affinity_or_shared_context_changed':False, 'preparation_host_mem_available_bytes':host_available,
        'memory_observation':memory, 'memory_observation_status':memory_status, 'memory_observation_error':memory_error,
        'observation_substitutes_for_fresh_launch_admission':False, 'host_available_is_reservation':False,
        'driver_AS_bytes':8*2**30, 'worker_AS_bytes':8*2**30, 'RSS_monitored_separately':True,
        'headroom_bytes':16*2**30, 'required_available_formula_GiB':'8 + 8*w + 16',
        'required_available_for_6_workers_GiB':72, 'required_available_for_1_worker_GiB':32,
        'generation_and_map_cumulative_wall_seconds':72*3600, 'fixed_run_total_output_bytes':10*2**30,
        'internal_thread_process_count':1, 'fixed_run_id':gates.RUN_ID,
        'artifact_anchor':gates.ARTIFACT_ANCHOR, 'future_output_root':gates.OUTPUT,
        'production_output_registry_resolve_stat_created':False,
        'failure_interruption':'STOP; no retry/resume/replacement/refund/rescue/alternative input',
        'after_six_input_freeze':'MANDATORY STOP; no signal/sampling/build/compile continuation',
        'unresolved_conditions':unresolved, 'execution_ready':False, 'actual_authorization_effective':False,
        'final_review_approved':False, 'explicit_launch_performed':False, 'mandatory_stop':True}


def write_new(relative, value):
    target = ROOT/relative
    if not relative.startswith(BUNDLE+'/') or '..' in Path(relative).parts:
        raise ValueError('preparation-only output scope')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--explicit-cpu-list', default='', help='Explicit approved list only; empty preserves unresolved permission.')
    parser.add_argument('--cpu-evidence', default='')
    args = parser.parse_args(argv)
    gates, identity, resources = production_modules()
    allowed = sorted(resources.cpus(args.explicit_cpu_list)) if args.explicit_cpu_list else []
    identity.require(not allowed or bool(args.cpu_evidence), 'explicit CPU evidence missing')
    contract, audit, report = verify_identity()
    plan, authorization, review = documents(contract, audit, allowed)
    resource = resource_record(allowed, args.cpu_evidence)
    files = {'input_generation_plan_v1.json':plan, 'authorization_draft_v1.json':authorization,
        'stage_review_v1.json':review, 'identity_audit_v1.json':report, 'resource_cpu_review_v1.json':resource}
    for name, data in files.items():
        write_new(BUNDLE+'/'+name, data)
    print(json.dumps({'status':STATUS, 'source_paths':19, 'templates':218, 'requested_workers':6,
        'allowed_cpus':allowed, 'saved_review_approved':False,
        'plan_fingerprint':identity.fingerprint('h4-execution-plan-v1', plan),
        'authorization_digest':identity.fingerprint('h4-authorization-v1', authorization),
        'review_digest':identity.fingerprint('h4-review-v1', review),
        'unresolved_conditions':resource['unresolved_conditions'], 'science_executed':False}, sort_keys=True))


if __name__ == '__main__':
    main()
