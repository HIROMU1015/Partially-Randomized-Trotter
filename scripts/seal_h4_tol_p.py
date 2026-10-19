"""Metadata-only P final seal. Produces no valid grant or scientific launch."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h4_tol_p_contract as p
import seal_h4_tol_a as a
import seal_h4_tol_d as d
import h4_p_seed_inventory as inventory
c = p.c
PRIVATE = p.ROOT/'.h4_tol_p_private'
PUBLIC = p.ROOT/'artifacts/resource_applicability/track_a_h4_tol_p_seal/2026-10-19'
OLD = p.ROOT.parent/'h4-tol-only-main-phase0-20261019/.h4_tol_main_private'
OLD_STATS = p.ROOT.parent/'h4-tol-s-stats-preparation-20261019/.h4_tol_s_stats_private'


def preserved():
    rows = {}
    for path in (OLD_STATS/'prior_preservation_manifest_PRIVATE.json',
        OLD_STATS/'technical_seal_preservation_manifest_PRIVATE.json',
        OLD_STATS/'completed_S_STATS_preservation_manifest_PRIVATE.json'):
        for row in c.read(path):
            c.require(row['path'] not in rows or rows[row['path']] == row['sha256'], 'OLD_MANIFEST_CONSISTENT')
            rows[row['path']] = row['sha256']
        rows[str(path)] = c.sha(path)
    for path in OLD_STATS.glob('*.json'): rows[str(path)] = c.sha(path)
    for path, h in rows.items(): c.require(c.sha(path) == h, 'OLD_SOURCE_RESULT_GRANT_LEDGER_MARKER_STOP')
    return [dict(path=path, sha256=h) for path, h in sorted(rows.items())]


def refusal(spec_path, grant_path, directory):
    guard = directory/'refusal_guard'; guard.mkdir()
    (guard/'sitecustomize.py').write_text("import sys\ndef guard(event,args):\n if event=='import' and (args[0]=='h4_tol_p_science' or args[0].startswith(('numpy','scipy','qiskit','trottertracks'))): raise AssertionError('SCIENCE_IMPORT_BEFORE_APPROVAL')\nsys.addaudithook(guard)\n")
    rows = []
    for witness in (False, True):
        command = [sys.executable, str(p.ROOT/'scripts/run_h4_tol_p.py'), '--spec', str(spec_path),
            '--grant', str(grant_path)]+(['--witness'] if witness else [])
        result = subprocess.run(command, capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=str(guard), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1'))
        c.require(result.returncode != 0 and 'SEPARATE_EXPLICIT_P_SCIENTIFIC_APPROVAL_REQUIRED' in result.stderr
            and 'SCIENCE_IMPORT_BEFORE_APPROVAL' not in result.stderr, 'REFUSAL_BEFORE_IMPORT_OR_DISPATCH')
        path = directory/('witness.stderr' if witness else 'direct.stderr'); path.write_text(result.stderr)
        rows.append(dict(witness=witness, exit_code=result.returncode, stderr_sha256=c.sha(path),
            refused_before_science=True, scientific_inputs_loaded=False, driver_launched=False))
    return rows


def seal(source):
    c.require(not PRIVATE.exists() and not PUBLIC.exists(), 'UNUSED_P_SEAL_NAMESPACE')
    prior = preserved(); PRIVATE.mkdir(); store = a.Store(PRIVATE); inputs = []
    def bind(role, path, expected=None, **extra):
        target = store.copy(role, path, expected)
        inputs.append(dict(role=role, path=str(target), sha256=c.sha(target), **extra))
        return target
    def fixed(role, commit, folder, name):
        repo = 'artifacts/resource_applicability/'+folder+'/2026-10-19/'+name
        raw = subprocess.check_output(['git', 'show', commit+':'+repo], cwd=p.ROOT)
        path = bind(role, p.ROOT/repo, hashlib.sha256(raw).hexdigest(), commit=commit, repo_path=repo)
        return c.read(path) if name.endswith('.json') else path
    stages = {}
    for stage, commit, folder in [('D', p.stats.D_COMMIT, 'track_a_h4_tol_d_completed'),
        ('A', p.stats.A_COMMIT, 'track_a_h4_tol_a_completed'),
        ('S_STATS', p.STATS_COMMIT, 'track_a_h4_tol_s_stats_completed')]:
        stages[stage] = fixed(stage+'_result', commit, folder, 'result.json')
        audit = fixed(stage+'_audit', commit, folder, 'completion_audit.json')
        native = fixed(stage+'_native_exit', commit, folder, 'native_exit.json')
        execution = fixed(stage+'_execution_binding', commit, folder, 'execution_binding.json')
        manifest = fixed(stage+'_manifest', commit, folder, 'artifact_manifest.json')
        result_hash = next(x['sha256'] for x in inputs if x['role'] == stage+'_result')
        c.require(audit['status'] == 'COMPLETE_MANDATORY_STOP' and audit['result_sha256'] == result_hash
            and native['exit_code'] == 0 and native['same_parent_reaped'] and native['result_sha256'] == result_hash,
            'ACCEPTED_COMPLETED_'+stage)
        expected_source = p.stats.SCIENCE_SOURCE if stage != 'S_STATS' else '8752489bde3903d80d306c4500535c79ac65ef31'
        c.require(execution['SOURCE'] == stages[stage]['SOURCE'] == expected_source, 'UPSTREAM_SOURCE')
        for name, h in manifest['files'].items():
            c.require(c.sha(p.ROOT/'artifacts/resource_applicability'/folder/'2026-10-19'/name) == h,
                'UPSTREAM_PUBLISHED_BYTES')
    eps = p.endpoints(stages['A'], stages['D'], stages['S_STATS'])
    original = c.read(OLD/'A_seal01_20261019/A_contract_SEALED_PENDING_APPROVAL_PRIVATE.json')
    bind('accepted_A_execution_contract', OLD/'A_seal01_20261019/A_contract_SEALED_PENDING_APPROVAL_PRIVATE.json')
    provider_hashes = {}
    for path, h in original['provider_hashes'].items():
        oldpath = Path(path); target = PRIVATE/oldpath.relative_to(OLD)
        c.require(c.sha(oldpath) == h, 'FROZEN_PROVIDER_SOURCE')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(oldpath.read_bytes())
        provider_hashes[str(target)] = h
    original_config = c.read(OLD/'rank10_native_config_PRIVATE.json')
    native_config = dict(provider_root=str(PRIVATE/'provider'), binding=deepcopy(p.main.BINDING),
        compiler_options=deepcopy(original_config['compiler_options']),
        mapping={key: str(bind(key, original_config['mapping'][key])) for key in ('tol_snapshot', 'tol_receipt')})
    def own(role, name, value):
        path = PRIVATE/name; c.save(path, value)
        inputs.append(dict(role=role, path=str(path), sha256=c.sha(path)))
        return str(path)
    config_path = own('native_config', 'native_config_PRIVATE.json', native_config)
    eps_path = own('fixed_endpoints', 'endpoints_PRIVATE.json', eps)
    preparations_path = bind('saved_A_preparations', OLD/'A_seal01_20261019/run01/output/preparations.json')
    compressed = fixed('saved_A_signals_gzip', p.stats.A_COMMIT, 'track_a_h4_tol_a_completed', 'signals.json.gz')
    saved_signals = __import__('json').loads(gzip.decompress(Path(compressed).read_bytes()))
    signals_path = own('saved_A_signals', 'saved_A_signals_PRIVATE.json', saved_signals)
    signals = {x['cell']['id']: x for x in saved_signals['signals']}
    reg = p.registry(source, eps, c.read(preparations_path)['rows'], signals)
    reg_path = own('seed_registry', 'seed_registry_PRIVATE.json', reg)
    print('Fixed SOURCE/inputs and 1186 seed tasks; checking known concurrent inventory', flush=True)
    excluded, coverage, cases, inherited = a.inventory(store)
    rank10 = store.copy('H4_rank10_A50_new392', OLD/'A_seal01_20261019/seed_registry_PRIVATE.json')
    values = a.identifiers(c.read(rank10), 'tasks'); excluded.update(values)
    coverage.append(dict(scope='H4_RANK10_A50_392', seed_values=len(values), status='COVERED'))
    c.require(not excluded & set(reg['values']), 'EXPLICIT_OLD_OR_CONCURRENT_SEED_COLLISION')
    geometry, workers = inventory.check_geometry(cases, reg['values'])
    c.require(all(not row['collisions'] for row in geometry), 'GEOMETRY_SEED_COLLISION')
    exclusions_path = own('seed_exclusions', 'old_seed_exclusions_PRIVATE.json', dict(values=sorted(excluded)))
    seed_receipt = dict(status='COMPLETE_COLLISION_ZERO', all_registered_scopes_covered=True,
        cutoff_UTC=datetime.now(timezone.utc).isoformat(), coverage=coverage, inherited_H4_coverage=inherited,
        old_explicit_values=len(excluded), old_explicit_exclusions_sha256=c.sha(exclusions_path),
        geometry_checks=geometry, geometry_workers=workers, geometry_registered_candidates=1308,
        geometry_stochastic_candidates=1164, new_master_seeds=1186, new_seed_slots=len(reg['values']),
        new_seed_values_digest=c.digest(reg['values']), RTE_event_sampling=0, redraw_on_collision=False,
        zero_collision_is_independence_proof=False, future_unregistered_campaigns_covered=False,
        activation_refresh_required=True, unrelated_worktree_HEAD_status_frozen=False)
    receipt_path = own('seed_inventory_receipt', 'seed_inventory_receipt_PRIVATE.json', seed_receipt)
    origin_path = own('seed_origins', 'seed_origins_PRIVATE.json', store.rows)
    font = deepcopy(original['font_seed']); font['path'] = str(bind('font_seed', font['path'], font['sha256']))
    receipt = dict(kind='P_FINAL_SEAL_PREPARATION_ONLY', scope=p.SCOPE, fixed_S_STATS_commit=p.STATS_COMMIT,
        explicit_preparation_approved=True, adopted_n=p.N, scientific_execution_approved=False,
        P_sampling_compile_approved=False, new_saved_value_analysis_approved=False,
        public_projection_contains_seed_values=False, new_publication_authorization_recorded=False)
    own('preparation_authorization', 'preparation_authorization_PRIVATE.json', receipt)
    spec = dict(scope=p.SCOPE, stage='P', SOURCE=source, plan=p.plan(),
        status='TECHNICALLY_SEALED_PENDING_EXPLICIT_APPROVAL', pending=p.PENDING,
        source_hashes=p.closure(), inputs=inputs, provider_roots=[str(PRIVATE/x) for x in ('provider', 'A_kernel')],
        provider_hashes=provider_hashes, runtime_hashes=original['runtime_hashes'], python=original['python'],
        python_sha256=c.sha(sys.executable), native_config=config_path, endpoints=eps_path,
        saved_signals=signals_path, saved_preparations=str(preparations_path), seed_registry=reg_path,
        old_seed_exclusions=exclusions_path, seed_inventory_receipt=receipt_path, seed_origins=origin_path,
        import_paths=[str(PRIVATE/'A_kernel/scripts'), str(PRIVATE/'provider/src')], font_seed=font,
        worker_cpus=[], fresh_resources=dict(admitted=False, receipt=None, sha256=None),
        seed_refresh_receipt=None, seed_refresh_sha256=None, approved_by_user=False, runtime_authorization=False,
        output=str(PRIVATE/'run01/output'), lock=str(PRIVATE/'run01/ONE_SHOT.json'),
        actual_scientific_calls=0, automatic_retry=False, next_stage_authorized=False)
    print('Seed collision checks PASS; verifying complete SOURCE/provider/runtime bytes', flush=True)
    p.verify_bytes(spec)
    path = PRIVATE/'P_contract_SEALED_PENDING_APPROVAL_PRIVATE.json'; c.save(path, spec)
    grant_path = PRIVATE/'P_grant_INVALID_PRIVATE.json'; c.save(grant_path, p.invalid_grant(spec))
    refusals = refusal(path, grant_path, PRIVATE)
    c.require(not Path(spec['output']).parent.exists() and not Path(spec['lock']).exists(), 'NO_P_ATTEMPT')
    for row in prior: c.require(c.sha(row['path']) == row['sha256'], 'ALL_PRIOR_BYTES_STILL_PRESERVED')
    c.save(PRIVATE/'prior_preservation_manifest_PRIVATE.json', prior)
    PUBLIC.mkdir(parents=True)
    c.save(PUBLIC/'contract_projection.json', dict(scope=p.SCOPE, SOURCE=source, contract_digest=c.digest(spec),
        status=spec['status'], pending=p.PENDING, plan=p.plan(), endpoints=eps,
        seed_registry_sha256=c.sha(reg_path), seed_inventory_receipt_sha256=c.sha(receipt_path),
        source_hashes=spec['source_hashes'], input_files=len(inputs), provider_files=len(provider_hashes),
        runtime_files=len(spec['runtime_hashes']), native_provider_SOURCE='217c971dd537b1cc7fbc13742bc0f6c263ff7f45',
        A_kernel_SOURCE='31ef62a12407010e9c1a07f9b37facedb8dff5fe',
        scientific_execution_authorized=False, launch_command=None))
    c.save(PUBLIC/'seed_inventory_summary.json', {k:v for k,v in seed_receipt.items() if k != 'inherited_H4_coverage'})
    c.save(PUBLIC/'grant_INVALID.json', p.invalid_grant(spec))
    c.save(PUBLIC/'refusal_report.json', dict(status='PASS', cases=refusals))
    c.save(PUBLIC/'preparation_authorization.json', receipt)
    c.save(PUBLIC/'resource_observation.json', d.resource_snapshot())
    audit = dict(status='PASS_TECHNICAL_SEAL_MANDATORY_STOP', SOURCE=source, contract_digest=c.digest(spec),
        prior_records=len(prior), source_files=len(spec['source_hashes']), provider_files=len(provider_hashes),
        runtime_files=len(spec['runtime_hashes']), inputs=len(inputs), new_seed_master_count=1186,
        new_seed_integer_slots=1810, old_explicit_seed_values=len(excluded), geometry_seed_slots=7737216,
        seed_collisions=0, actual_scientific_calls=0, actual_numerical_analyses=0, valid_grants=0,
        scientific_output_or_lock_created=False, actual_P_sampling=0, actual_P_compile=0,
        source_inputs_runtime_verified=True, all_prior_records_preserved=True,
        separate_AI_reviewer_used=False, next_stage_authorized=False, mandatory_STOP=True)
    c.save(PUBLIC/'completion_audit.json', audit)
    c.save(PRIVATE/'P_FINAL_SEAL_MANDATORY_STOP_PRIVATE.json', audit)
    c.save(PRIVATE/'technical_seal_preservation_manifest_PRIVATE.json', [dict(path=str(x),sha256=c.sha(x))
        for x in sorted(PRIVATE.glob('*.json'))])
    print('PASS: P final technical seal, invalid grants refused, science=0; mandatory STOP', flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True)
    args = parser.parse_args(); seal(args.source)
