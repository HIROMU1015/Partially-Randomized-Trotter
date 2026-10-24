"""Metadata-only pack builder. Does not issue grants or evaluate molecular data."""
import argparse
import copy
import csv
import json
from pathlib import Path
import secrets
import subprocess
import sys
import h46_standard_y4_contract as c

ENDPOINTS = 'artifacts/resource_applicability/track_a_h246_reversal_batch_run01/2026-10-24/primary_endpoints_60.csv'
OLD_DIR = 'artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/'


def fixed_blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=c.ROOT)


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def collect_exclusions(config_paths):
    values = set()
    records = []
    for path in config_paths:
        cfg = c.read(path)
        for k in ('registry_path', 'excluded_seeds_path'):
            p = Path(cfg[k])
            data = c.read(p)
            found = data['values'] if k == 'excluded_seeds_path' else [
                v for row in data.values() for v in row.values()]
            c.require(all(type(v) is int for v in found), 'INTEGER_OLD_SEEDS')
            values.update(found)
            records.append(dict(**c.seal(p, 'available_historical_' + k),
                value_count=len(found)))
        records.append(c.seal(path, 'historical_exploration_config_readonly'))
    return values, records


def phase_a(assets):
    """Read receipts and identifiers, not new means, variances or shots."""
    a = Path(assets)
    recovery = (c.read(a / 'recovery_manifest_PRIVATE.json') +
                c.read(a / 'additional_import_closure_PRIVATE.json'))
    # Verify every recovered source/input against the recorded original Git blob.
    for r in recovery:
        p = Path(r['local'])
        c.require(c.sha(p) == r['sha256'] and p.read_bytes() ==
            fixed_blob(r['commit'], r['path']), 'EXACT_RECOVERED_GIT_BLOB:' + r['path'])
    audit = c.read(a / 'fixed/end_audit.json')
    c.require(audit['all_native_exits_zero_and_reaped'] and
        audit['all_worker_exits_zero_and_reaped'] and
        audit['original_numerical_SOURCE_files_unchanged'] and
        audit['H2_H4_regenerated_array_hashes_identical'] and
        audit['H6_original_snapshot_SHA_identical'], 'OLD_END_AUDIT_PASS')
    rows = read_csv(a / 'fixed/fixed_evaluation/paired_samples_118.csv')
    input_metadata = c.read(a / 'fixed/stage_contracts.json')['contracts']
    reconstructed = c.read(a / 'fixed/input_reconstruction.json')
    design = read_csv(a / 'fixed/n_E_sensitivity_216.csv')
    endpoints = read_csv(c.ROOT / ENDPOINTS)
    c.require((c.ROOT / ENDPOINTS).read_bytes() == fixed_blob(c.BASE, ENDPOINTS),
        'CURRENT_FIXED_ENDPOINT_SOURCE')
    result = {}
    for system, g in c.GROUPS.items():
        planned = [r for r in design if r['system'] == system and r['formula'] == '4th'
            and r['q'] == '4' and float(r['relative_MCSE_target']) == .01 and
            float(r['variance_inflation']) == 4]
        c.require(len(planned) == 1 and int(planned[0]['n_E_nominal_only']) == g['n'],
            'REUSE_SAVED_N_DESIGN_NO_NEW_NUMERICAL_SAMPLE_SIZE_CALCULATION')
        selected = [r for r in rows if r['system'] == system and r['candidate_id'] == g['S2']]
        c.require(len(selected) == g['n_s2'] and sorted(int(r['sample']) for r in selected)
            == list(range(g['n_s2'])) and all(r['SOURCE'] == c.OLD_SOURCE and
            r['stratum'] in c.law.STRATA and len(r['result_sha256']) == 64 and
            int(r['cosine_RZ']) > 0 and int(r['sine_RZ']) > 0 for r in selected),
            'COMPLETE_S2_PAIRED_SAMPLE_LINEAGE:' + system)
        old = next(r for r in input_metadata if r['stage'] == system + '_CONFIRMATION')
        fixed = {kind: next(r for r in endpoints if r['system'] == system and
            r['candidate'] == g[kind] and r['epsilon'] == '0.001') for kind in ('S2', 'Y4')}
        for kind, formula, q in [('S2', '2nd', 8), ('Y4', '4th', 4)]:
            r = fixed[kind]
            c.require(r['formula'] == formula and int(r['q']) == q and
                int(r['prefix']) == g['prefix'] and r['method'] == 'PR' and
                [int(r['N_cos']), int(r['N_sin'])] == g['N_' + kind], 'EPSILON_001_FIXED_SHOTS')
        inp = reconstructed[system]
        if system == 'H4':
            c.require(all(inp['array_hash_comparisons'].values()) and
                inp['same_hamiltonian_fingerprint'] and inp['same_state_fingerprint'],
                'H4_SCIENTIFIC_ARRAYS_NOT_CONTAINER_BYTE_REQUIREMENT')
        result[system] = dict(S2_n=len(selected), S2_candidate=g['S2'],
            Y4_candidate=g['Y4'], binding=old['input_binding'],
            preparation=g['preparation'], compiler=c.law.COMPILER_HASH,
            S2_pair_SOURCE=c.OLD_SOURCE, result_commit=c.RESULT_SOURCE,
            old_namespace_sha256=old['campaign_namespace_sha256'],
            old_registry_sha256=old['registry_sha256'],
            original_S2_raw_and_registry_recovered=False,
            reuse='ACCEPTED_SAVED_PAIR_CSV_AND_PINNED_SUCCESSFUL_END_AUDIT',
            H4_container_byte_identity_required=False,
            current_shots={k: g['N_' + k] for k in ('S2', 'Y4')},
            old_epsilon_0005_G_reused=False, new_S2_sampling=0, new_S2_compile=0,
            cohort='CONFIRMATION', historical_seed_coverage_complete=False)
    return result


def prepare(assets, python, site_packages, historical_configs, pack_name):
    a = Path(assets).resolve()
    c.install_paths(a, site_packages)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip()
    for rel in c.SOURCE_PATHS:
        c.require((c.ROOT / rel).read_bytes() == fixed_blob(source, rel), 'COMMIT_CODE_BEFORE_PACKS')
    c.require(str(Path(sys.executable).resolve()) == str(Path(python).resolve()), 'PREPARE_USING_FIXED_PYTHON')
    import numpy as np
    import qiskit
    import symengine
    import scipy
    import concurrent.futures.thread
    import concurrent.futures._base
    import queue
    import _queue
    import threading
    c.require(np.__version__ == '1.26.4' and qiskit.__version__ == '1.3.0', 'PINNED_RUNTIME')
    reuse = phase_a(a)
    namespace = c.SERIES + '_' + source + '_' + secrets.token_hex(32)
    prior, protected = collect_exclusions(historical_configs)
    # Reconstruct the source-derived, originally proposed seeds as another known
    # subset. The actual migrated-confirmation registry remains unavailable.
    import h246_confirmation_contract as old
    for stage in old.STAGES:
        values = c.law.registry(old.tasks_for(stage), old.OLD_SOURCE if hasattr(old, 'OLD_SOURCE')
            else '8825391ff73b531be5834bf852b9bf2479d1799b',
            'H246_HIGHER_PF_CONFIRMATION_PROPOSED_RUN01_8825391ff73b531be5834bf852b9bf2479d1799b',
            'CONFIRMATION', [])
        prior.update(v for row in values.values() for v in row.values())
    original_signals = c.read(a / 'frozen/artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json')
    records = [r for r in original_signals['records'] if
        r['candidate']['id'] in {g['Y4'] for g in c.GROUPS.values()}]
    c.require(len(records) == 2, 'TWO_STANDARD_Y4_SIGNALS')
    c.require(pack_name in ('standard_y4_run01', 'standard_y4_run02'), 'FIXED_PREPARATION_ATTEMPT')
    packdir = a / pack_name
    packdir.mkdir(exist_ok=False)
    signal_path = packdir / 'signals_PRIVATE.json'
    c.save(signal_path, dict(records=records, source=c.OLD_SOURCE))
    tasks = {s: c.tasks_for(s, next(r['candidate'] for r in records if
        r['candidate']['system'] == s)) for s in c.GROUPS}
    registries = {s: c.law.registry(tasks[s], source, namespace, 'CONFIRMATION', sorted(prior))
                  for s in c.GROUPS}
    seeds = {s: {v for row in r.values() for v in row.values()} for s, r in registries.items()}
    c.require(not (seeds['H4'] & seeds['H6']) and not (set.union(*seeds.values()) & prior),
        'NEW_SEEDS_DISJOINT_KNOWN_AND_OTHER_STAGE')
    files = {r['path']: r for r in protected}

    def add(path, role):
        r = c.seal(path, role)
        files[r['path']] = r

    for r in (c.read(a / 'recovery_manifest_PRIVATE.json') +
              c.read(a / 'additional_import_closure_PRIVATE.json')):
        add(r['local'], 'recovered:' + r['path'])
    add(a / 'recovery_manifest_PRIVATE.json', 'recovery_provenance')
    add(a / 'additional_import_closure_PRIVATE.json', 'additional_static_import_closure')
    prior_stop = a / 'prior_attempt_preservation_PRIVATE.json'
    if prior_stop.exists():
        prior_evidence = c.read(prior_stop)
        c.check_files([dict(r, role='prior_stopped_attempt') for r in prior_evidence['files']])
        add(prior_stop, 'technical_recovery_prior_attempt')
        for row in prior_evidence['files']:
            add(row['path'], 'prior_stopped_attempt:' + row['logical_path'])
    for rel in c.SOURCE_PATHS:
        add(c.ROOT / rel, 'new_SOURCE:' + rel)
    add(c.ROOT / ENDPOINTS, 'fixed_current_epsilon_001_endpoints')
    add(signal_path, 'saved_Y4_signals_only_no_evaluation')
    modules = [np, qiskit, symengine, scipy, concurrent.futures.thread,
               concurrent.futures._base, queue, _queue, threading]
    for module in modules:
        add(module.__file__, 'loaded_runtime_identity')
    # Native compiler/array/scipy/symbolic binaries affect correctness. No
    # unrelated site-packages closure, shared installation or runtime mutation.
    for module in [np, qiskit, symengine, scipy]:
        for path in sorted(Path(module.__file__).parent.rglob('*.so')):
            add(path, 'compiler_array_oracle_binary')
    add(Path(python).resolve(), 'fixed_python_binary')
    alias_rows = []
    for s in c.GROUPS:
        for name, origin in [('reference.npz', 'forward_reference.npz'),
                             ('receipt.json', 'forward_receipt.json')]:
            src, dst = a / 'inputs' / s / origin, a / 'inputs' / s / name
            if dst.exists():
                c.require(dst.read_bytes() == src.read_bytes(), 'IDENTICAL_EXISTING_SAVED_REFERENCE_ALIAS')
            else:
                dst.write_bytes(src.read_bytes())
            add(dst, 'byte_identical_saved_reference_alias')
            alias_rows.append(dict(system=s, alias=name, origin=origin, sha256=c.sha(dst)))
    input_path = packdir / 'input_manifest_PRIVATE.json'
    c.save(input_path, dict(SOURCE=source, files=list(files.values()), phase_A=reuse,
        known_historical_seed_count=len(prior), historical_seed_coverage_complete=False,
        aliases=alias_rows, no_new_numerical_evaluation=True))
    runtime = dict(python=python, site_packages=site_packages, numpy=np.__version__,
        qiskit=qiskit.__version__, symengine=symengine.__version__, scipy=scipy.__version__,
        provider_root=str(a / 'provider'), provider_SOURCE=c.PROVIDER_SOURCE)
    package_path = packdir / 'execution_package_DRAFT_PRIVATE.json'
    capacities = json.loads(fixed_blob(c.OLD_SOURCE,
        'artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/stage_readiness.json'))
    stages = []
    for s, g in c.GROUPS.items():
        dest = packdir / (s + '_CONFIRMATION')
        dest.mkdir(exist_ok=False)
        root = dest / 'run01'
        rp, ep = dest / 'registry_PRIVATE.json', dest / 'known_exclusions_PRIVATE.json'
        c.save(rp, registries[s])
        c.save(ep, dict(values=sorted(prior | seeds['H6' if s == 'H4' else 'H4']),
            complete_historical_seed_noncollision_guaranteed=False))
        binding = copy.deepcopy(reuse[s]['binding'])
        names = [('H4_tol_snapshot.npz', 'tol_only_snapshot.npz'),
                 ('H4_tol_receipt.json', 'tol_only_receipt.json')] if s == 'H4' else [
                 ('H6_snapshot.npz', 'h6_input_snapshot.npz'),
                 ('H6_snapshot_receipt.json', 'snapshot_receipt.json'),
                 ('H6_df_receipt.json', 'df_receipt.json')]
        inputs = {k: c.seal(a / 'inputs' / s / v, 'saved_scientific_input') for k, v in names}
        c.require(next(v['sha256'] for k, v in inputs.items() if k.endswith('_snapshot.npz'))
            == binding['snapshot'], 'SNAPSHOT_BINDING')
        resources = copy.deepcopy(next(x['resources'] for x in capacities['stages'] if
            x['stage'] == s + '_CONFIRMATION'))
        # Old mixed-formula proxies are observations, not caps. Final size and
        # time scale with fixed n; admission is recomputed from fresh resources.
        old_n = 34 if s == 'H4' else 32
        for k in ('final_growth_proxy_bytes', 'inode_proxy', 'workload_proxy_seconds'):
            resources[k] = __import__('math').ceil(resources[k] * g['n'] / old_n)
        body = dict(SOURCE=source, stage=s + '_CONFIRMATION', cohort='CONFIRMATION',
            assets=str(a), package_path=str(package_path), provider_root=str(a / 'provider'),
            input_manifest_path=str(input_path), input_manifest_sha256=c.sha(input_path),
            runtime=runtime, campaign_id=namespace, tasks=tasks[s],
            call_scope=c.acquisition_scope(tasks[s]), context_scope=c.context_scope(),
            analysis_scope=c.analysis_scope(), registry_path=str(rp), excluded_seeds_path=str(ep),
            systems={s: dict(binding=binding, inputs=inputs, reference_root=str(a / 'inputs' / s))},
            inputs=inputs, binding=binding, preparation_hash=g['preparation'],
            signal_path=str(signal_path), storage=c.storage.storage_policy(), resources=resources,
            compiler_settings_hash=c.law.COMPILER_HASH, compile_tolerance=c.COMPILE_TOLERANCE,
            other_tolerance=c.OTHER_TOLERANCE, run_root=str(root), output=str(root / 'output'),
            ledger=str(root / 'ledger'), marker=str(root / 'CONSUMED.json'),
            supervisor={k: str(dest / n) for k, n in [('entered', 'SUPERVISOR_ENTERED.json'),
                ('exit', 'external_exit_witness_PRIVATE.json'), ('log', 'native_PRIVATE.log')]},
            complete_historical_seed_noncollision_guaranteed=False,
            sealed_files=list(files.values()) + [c.seal(rp, 'new_registry'), c.seal(ep, 'known_exclusions')])
        pp = dest / 'plan_DRAFT_PRIVATE.json'
        c.save(pp, body)
        cfg = dict(body, plan=body, plan_path=str(pp), plan_sha256=c.sha(pp),
            authorized=False, grant=dict(valid=False))
        cp = dest / 'config_DRAFT_PRIVATE.json'
        c.save(cp, cfg)
        stages.append(dict(stage=body['stage'], plan_path=str(pp), plan_sha256=c.sha(pp),
            draft_config_path=str(cp)))
    package = dict(schema=c.SERIES, SOURCE=source, base=c.BASE, old_SOURCE=c.OLD_SOURCE,
        assets=str(a), input_manifest_path=str(input_path), input_manifest_sha256=c.sha(input_path),
        runtime=runtime, stages=stages, n={s: g['n'] for s, g in c.GROUPS.items()},
        reused_S2_n={s: g['n_s2'] for s, g in c.GROUPS.items()},
        evaluation_scope=c.evaluation_scope(), analysis_scope=c.analysis_scope(),
        scientific_execution_authorized=False, valid_grants=0,
        evaluation_root=str(packdir / 'evaluation_run01'),
        sample_size_design=dict(source=c.OLD_SOURCE, receipt='n_E_sensitivity_216.csv',
            relative_MCSE_target=.01, variance_inflation=4, guarantee=False,
            new_sample_size_numeric_calculation=0, adopted=False),
        retry=False, mandatory_STOP=True)
    c.save(package_path, package)
    preflight = metadata_preflight(package_path)
    c.save(packdir / 'metadata_preflight_PRIVATE.json', preflight)
    return dict(package_sha256=c.sha(package_path), SOURCE=source,
        input_manifest_sha256=c.sha(input_path), preflight=preflight)


def metadata_preflight(package_path):
    p = c.verify_package(package_path)
    for r in p['stages']:
        cfg = c.read(r['draft_config_path'])
        c.verify(cfg)
        c.require(not cfg['authorized'] and not cfg['grant']['valid'] and
            not Path(cfg['run_root']).exists(), 'DISABLED_UNUSED_DRAFT')
        try:
            c.authorize(cfg)
        except ValueError as exc:
            c.require(str(exc) == 'EXPLICIT_STANDARD_Y4_SCIENCE_APPROVAL_REQUIRED',
                'FAIL_BEFORE_SCIENTIFIC_IMPORT')
        else:
            raise ValueError('UNAPPROVED_DRAFT_ACCEPTED')
    return dict(status='METADATA_PREFLIGHT_PASS_AWAITING_ONE_USER_APPROVAL',
        SOURCE=p['SOURCE'], package_sha256=c.sha(package_path),
        input_manifest_sha256=p['input_manifest_sha256'],
        new_scientific_calls=0, valid_grants=0, mandatory_STOP=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--site-packages', required=True)
    parser.add_argument('--historical-config', action='append', required=True)
    parser.add_argument('--pack-name', choices=('standard_y4_run01', 'standard_y4_run02'),
                        default='standard_y4_run02')
    args = parser.parse_args()
    print(json.dumps(prepare(args.assets, args.python, args.site_packages, args.historical_config,
        args.pack_name)))


if __name__ == '__main__':
    main()
