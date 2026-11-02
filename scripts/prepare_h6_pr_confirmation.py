"""Read-only H6 evidence audit and Phase 0 proposal; never allocate fresh seeds.

Only hashes, metadata equality checks, and verbatim saved scalar projections.
No paired_statistics, sample-size, power, visibility or native-provider calls.
The destination must be new; existing SOURCE/results/grants are never rewritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h6_pr_confirmation as h
c = h.c

FIXED = {
    'new_A_results': ('64c85f7a833b5950ef2ccae623708b73081edcd6',
        'docs/research/track_a_pr_new_a_total_cost_results_20261018.md'),
    'H6_A_D_results': ('fc6bd039dc7171d5674ac04f0b3ad427f1b6495b',
        'docs/research/track_a_pr_cost_batch_completed_results_20261018.md'),
    'H8_design': ('434e67a8bb205711d22d5bf1aa5d143fd08b7765',
        'docs/research/track_a_h8_new_pr_confirmation_design_20261018.md'),
    'H8_SOURCE': ('745a674546f2cb6258526fbfde1d9a00601689f8', 'scripts/h8_pr_confirmation.py'),
    'H8_results': ('e6eea5387a6e84d1e39a9adb3b61975acc74e200',
        'docs/research/track_a_h8_new_pr_confirmation_results_20261019.md'),
    'H4_P_results': ('8a18b277663bf2f8998f79e9b8c0a9e61d2b184c',
        'docs/research/track_a_h4_p_completed_20261019.md'),
}
COMPARISONS = 'artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv'


def git_bytes(commit, path):
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'maintenance.auto=false',
        '-c', 'gc.auto=0', '-C', str(h.ROOT), 'show', commit + ':' + path])


def byte_row(path, role):
    path = Path(path)
    c.require(path.is_file() and not path.is_symlink(), 'REGULAR_EVIDENCE_FILE')
    return dict(role=role, path=str(path.resolve()), bytes=path.stat().st_size, sha256=c.sha(path))


def runtime_audit(origin, destination, cache_path=None):
    """Hash the explicit frozen inventory once; computational I/O threads only."""
    path = origin['runtime_receipt']
    c.require(c.sha(path) == origin['runtime_receipt_sha256'], 'ORIGINAL_RUNTIME_RECEIPT_SHA')
    prior = c.read(path)
    cache = c.read(cache_path) if cache_path is not None else None
    if cache is not None:
        c.require(cache['status'] == 'PASS_FRESH_RUNTIME_BYTES'
                  and cache['inventory_source_sha256'] == origin['runtime_receipt_sha256']
                  and cache['expected'] == prior['expected'], 'EXACT_RUNTIME_CACHE_PROVENANCE')
    def check(item):
        name, expected = item
        st = Path(name).stat()
        facts = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
        reused = cache is not None and cache['verified_stats'].get(name) == facts
        actual = cache['expected'][name] if reused else c.sha(name)
        c.require(actual == expected, 'RUNTIME_SHA_MISMATCH:' + name)
        after = Path(name).stat()
        c.require(facts == [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns],
                  'RUNTIME_CHANGED_DURING_AUDIT:' + name)
        return name, actual, facts, reused
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        rows = list(pool.map(check, prior['expected'].items()))
    receipt = dict(expected={name: value for name, value, _, _ in rows},
        verified_stats={name: facts for name, _, facts, _ in rows},
        status='PASS_FRESH_RUNTIME_BYTES', files=len(rows), scientific_imports=0,
        inventory_source_sha256=origin['runtime_receipt_sha256'],
        SHA_reused_on_unchanged_stat=sum(reused for _, _, _, reused in rows),
        SHA_rehashed=sum(not reused for _, _, _, reused in rows),
        cache_receipt_sha256=None if cache_path is None else c.sha(cache_path))
    c.save(destination / 'runtime_receipt_PRIVATE.json', receipt)
    return dict(path=str(destination / 'runtime_receipt_PRIVATE.json'),
        sha256=c.sha(destination / 'runtime_receipt_PRIVATE.json'), files=len(rows),
        status=receipt['status'])


def prepare(accounting_spec, destination, runtime_cache=None):
    destination = Path(destination).resolve()
    c.require(destination.is_relative_to(Path('/home/AbeHiromu/projects')), 'OWN_PROJECT_DESTINATION')
    destination.mkdir(parents=True, exist_ok=False)
    accounting = c.read(accounting_spec)
    c.require(accounting['system'] == 'H6', 'H6_SAVED_ACCOUNTING_ONLY')
    by_role = {r['role']: r for r in accounting['inputs']}
    c.verify_byte_rows(accounting['inputs'])
    origin = c.read(by_role['origin_A_spec']['path'])
    c.require(origin['scope'] == 'H6_A_COST' and origin['binding'] == accounting['binding'], 'H6_ORIGINAL_BINDING')
    c.verify_byte_rows(origin['inputs'])
    payload = c.read(by_role['A_payload']['path'])
    c.require(payload['native_adapter']['binding'] == origin['binding'], 'H6_NATIVE_ADAPTER_BINDING')
    native_config = c.read(payload['native_adapter']['native_config'])
    c.verify_byte_rows(native_config['inputs'])
    c.require(native_config['compiler_options'] == dict(basis_gates=['rz', 'sx', 'x', 'cx'],
              optimization_level=1, seed_transpiler=17), 'FIXED_COMPILER_OPTIONS')
    c.require(c.sha(origin['python']) == origin['python_sha256'], 'PYTHON_BINARY_SHA')
    refs = {}
    for role, (commit, path) in FIXED.items():
        value = git_bytes(commit, path)
        refs[role] = dict(commit=commit, path=path, bytes=len(value),
                          sha256=hashlib.sha256(value).hexdigest(), retrieved=True)
    # Check original execution SOURCE against Git; current worktree HEAD is unrelated.
    for path, expected in origin['source_hashes'].items():
        c.require(hashlib.sha256(git_bytes(origin['SOURCE'], path)).hexdigest() == expected,
                  'H6_A_EXECUTION_SOURCE_BLOB:' + path)
    for root, files in origin['provider_closure'].items():
        for path, expected in files.items():
            c.require(c.sha(Path(root) / path) == expected, 'FROZEN_PROVIDER_BYTES:' + path)
    runtime = runtime_audit(origin, destination, runtime_cache)
    old_terminal = c.read(by_role['A_terminal']['path'])
    c.require(old_terminal['status'] == 'COMPLETE_MANDATORY_STOP'
              and old_terminal['result_sha256'] == by_role['A_result']['sha256'], 'SAVED_A_TERMINAL_SHA')
    result = c.read(by_role['A_result']['path'])
    c.require(result['scope'] == 'H6_A_COST' and result['status'] == 'COMPLETE_MANDATORY_STOP', 'SAVED_A_COMPLETE')
    representatives_path = Path(origin['output']) / 'first_control/parallel_receipt.json'
    representatives = c.read(representatives_path)
    c.require(representatives['status'] == 'PASS' and representatives['all_reaped'] is True
              and not representatives['cleanup_errors'], 'SAVED_A_NATIVE_RECEIPT')
    prep_path = Path(origin['output']) / 'preparation_receipt.json'
    prep = c.read(prep_path)
    signals = {r['cell']['id']: r for r in c.read(payload['A_signal'])['signals']}
    registry = c.read(h.ROOT / c.REGISTRY)['systems']['H6']
    c.require(registry['binding'] == accounting['binding'], 'PINNED_H6_SCIENTIFIC_BINDING')
    public_rows = list(csv.DictReader(io.StringIO(git_bytes(FIXED['new_A_results'][0], COMPARISONS).decode())))
    d_rows = c.read(by_role['D_result']['path'])['rows']
    saved_costs_path = Path(accounting['output']) / 'result.json'
    saved_costs = c.read(saved_costs_path)
    cost_terminal_path = Path(accounting['output']) / 'terminal.json'
    cost_terminal = c.read(cost_terminal_path)
    c.require(cost_terminal['result_sha256'] == c.sha(saved_costs_path)
              and cost_terminal['status'] == 'COMPLETE_MANDATORY_STOP', 'SAVED_COST_TERMINAL_SHA')
    inputs = [byte_row(accounting_spec, 'saved_new_A_accounting_spec')]
    inputs += [dict(row, role='accounting:' + row['role']) for row in accounting['inputs']]
    inputs += [dict(row, role='origin:' + row['role']) for row in origin['inputs']]
    inputs += [dict(row, role='native:' + row['role']) for row in native_config['inputs']]
    inputs += [byte_row(representatives_path, 'old_A_representative_native_receipt'),
        byte_row(prep_path, 'old_A_p4_preparation_receipt'), byte_row(saved_costs_path, 'old_new_A_saved_costs'),
        byte_row(cost_terminal_path, 'old_new_A_saved_cost_terminal')]
    targets = []
    for original in h.TARGETS:
        target = deepcopy(original)
        cid, epsilon = target['cell']['id'], target['epsilon']
        record = signals[cid]
        c.require(record['cell'] == target['cell'] and c.digest(record) == payload['signal_record_hashes'][cid]
                  and c.digest(record) == next(r['record_sha256'] for r in registry['records']
                  if r['cell']['id'] == cid), 'PINNED_H6_SIGNAL_RECORD')
        saved = record['precision_rows'][str(epsilon)]['1.0']
        c.require(saved['cost_acquisition_eligible'] is True, 'ORIGINAL_ELIGIBLE_ENDPOINT')
        comparison, = [r for r in public_rows if r['system'] == 'H6' and r['bias_mode'] == 'USUAL_SIGNED'
            and r['factor'] == '1.0' and r['epsilon'] == str(epsilon)
            and r['deterministic_scope'] == 'strengthened_S2_S4']
        c.require(comparison['new_best_id'] == cid and comparison['PF_best_available_id'] == target['fixed_PF_id'],
                  'ORIGINAL_SELECTION_PROVENANCE')
        pf, = [r for r in d_rows if r['cell_id'] == target['fixed_PF_id'] and r['epsilon'] == epsilon
               and r['factor'] == 1. and r['bias_mode'] == 'USUAL_SIGNED']
        c.require(pf['eligible'] is True and pf['G_exact'] == comparison['PF_G_exact']
                  and set(pf['axis_costs']) == set(c.core.AXES)
                  and all(v['kind'] == 'DIRECT_MEASURED' and v['compiler_hash'] == accounting['binding']['compiler']
                          and v['SOURCE'] and v['result_sha256'] for v in pf['axis_costs'].values()),
                  'SAVED_DIRECT_PF_PROVENANCE')
        saved_row, = [r for r in saved_costs['new_rows'] if r['cell_id'] == cid and r['epsilon'] == epsilon
                     and r['factor'] == 1. and r['bias_mode'] == 'USUAL_SIGNED']
        c.require(saved_row['n'] == 8 and saved_row['G_RZ_exact'] == comparison['new_n8_G_exact']
                  and str(saved_row['G_RZ_SE']) == comparison['new_n8_G_SE'], 'SAVED_N8_STATISTICS_COPY_ONLY')
        handle, = [r for r in representatives['workers'] if r['task_id'] == cid + '_0']
        directory = Path(handle['directory'])
        c.require(handle['exit_code'] == 0 and handle['reaped'] is True
                  and handle['result_sha256'] == c.sha(directory / 'task_result.json'), 'OLD_REPRESENTATIVE_EXIT_SHA')
        trace = c.read(directory / 'trajectory.json')
        control = c.read(directory / 'control.json')
        c.require(trace['cell'] == target['cell'] and trace['cohort'] == 'EXPLORATION_N8'
                  and trace['conditional'] is False and control['status'] == 'PASS', 'OLD_NATIVE_CELL_CONTROL')
        law = trace['original_RTE_distribution']
        h.check_law(law)
        target.update(n=None, A_signal_record_sha256=c.digest(record), accounting_sha256=c.digest(saved),
            shots={a: saved['axes'][ch]['shots'] for a, ch in c.core.CHANNEL.items()},
            fixed_PF_G_exact=pf['G_exact'], fixed_PF_receipt=pf, law=law,
            original_law_sha256=c.digest(law), preparation_sha256=prep['prefixes']['4'],
            saved_log_B=record['log_B'], log_B_margin=record['allowance']['log_B_margin'],
            saved_n8=dict(n=8, mean_exact=saved_row['G_RZ_exact'], paired_SE=saved_row['G_RZ_SE'],
                          SD=None, variance=None, new_statistics_computed=False),
            source_selection_comparison=comparison)
        for name in ('trajectory.json', 'control.json', 'task_result.json', 'started.json'):
            inputs.append(byte_row(directory / name, 'old_representative:' + target['scope'] + ':' + name))
        targets.append(target)
    old_exclusions_path = payload['new_seed_registry']['old_seed_exclusions']
    excluded = set(c.read(old_exclusions_path)['values'])
    old_count = len(excluded)
    c.require(len(payload['new_seed_registry']['tasks']) == 128, 'EXCLUDE_ALL_H6_A128')
    for task in payload['new_seed_registry']['tasks']:
        excluded.update(h.shared.hierarchy(task['seed'], task['cell']['q']))
    # Reuse the old inventory, but do not assert coverage of later concurrent campaigns.
    seed_inventory = dict(values=sorted(excluded), old_receipt_values=old_count,
        original_A_task_count=128, derived_from_existing_identifiers_only=True,
        new_confirmation_seed_count=0, final_exclusion_inventory_complete=False,
        pending='Reconcile later H6 runs and in-flight reservations before final seed seal; never inspect broad roots.')
    c.save(destination / 'seed_exclusions_PRIVATE.json', seed_inventory)
    inputs.append(byte_row(old_exclusions_path, 'old_seed_exclusions'))
    inputs.append(byte_row(destination / 'seed_exclusions_PRIVATE.json', 'phase0:seed_inventory'))
    inputs.append(byte_row(runtime['path'], 'phase0:runtime_receipt'))
    c.verify_byte_rows(inputs)
    c.save(destination / 'input_manifest_PRIVATE.json', dict(inputs=inputs,
        provider_closure=origin['provider_closure'], payload=by_role['A_payload']['path'],
        preparation_policy=payload['preparation_policy'], runtime=runtime,
        python=origin['python'], python_sha256=origin['python_sha256'],
        profile_reference=accounting['profile_reference'], original_A_SOURCE=origin['SOURCE']))
    plan = dict(schema='H6_CONFIRMATION_PHASE0_V1', system='H6', cohort=h.COHORT, SOURCE=None,
        binding=accounting['binding'], targets=targets, metric='saved_shots_times_compiled_RZ_cosine_plus_sine',
        bias_mode='USUAL_SIGNED', allowance_factor=1, P=0, seed_namespace=h.NAMESPACE,
        seed_registry=None, calls=None, seed_inventory_complete=False,
        approved=False, runtime_authorization=False, technical_sealed=False, retry=False,
        next_stage_authorized=False, mandatory_STOP=True,
        compiler_options=native_config['compiler_options'], qiskit_version='1.3.0',
        numerical_design=dict(approved=False, SOURCE_inputs='fixed saved n8 means/SE, PF prices and K2 law above',
            family_alpha_proposed=.05, power_proposed=.8, SD_multipliers_proposed=[1, 2, 4],
            H8_n48_reference_only=True, n_results=None, power_results=None, visibility_results=None,
            n_formula='max(2, ceil(((z_(1-alpha_family/2)+z_power)*f*SE8*sqrt(8)/abs(PF_G-mean8))**2))',
            visibility_formula='pi2=1-(1-p2)**R; visibility(n)=1-(1-pi2)**n',
            caveat='Selected n8 normal plug-in scenarios only; no certified power, CLT or population superiority.'),
        budget_formulas=dict(trajectory='n_q2+n_q8', wrapper_compile='2*(n_q2+n_q8)',
            native_evolution='n_q2+n_q8', occurrence='2*n_q2+8*n_q8', RTE_event='8*n_q2+32*n_q8',
            control_probe='4 per candidate (8 total)', independent_event_oracle='1 per candidate (2 total)',
            new_PF_compile=0, new_signal=0, shot_recalculation=0, SCF_DF_state_solver=0),
        resource_policy=dict(CPU_RAM_worker_wall_caps=None, internal_threads=1,
            workers='fresh effective CPU/memory availability and own-scope noninterference; no historical cap',
            transient_PSI_CPU_contention='record and recover monitoring; never kill solely for transient pressure',
            shared_environment_changes=False, unrelated_worktrees='read-only observations, not launch/STOP gates'),
        pending=['separate approval for numerical design', 'final n and complete old seed inventory',
                 'final SOURCE/input/runtime/seed/call seal', 'separate approval for scientific sampling/compile'])
    h.verify_proposal(plan)
    c.save(destination / 'plan_PROPOSED.json', plan)
    c.save(destination / 'fixed_references.json', refs)
    c.save(destination / 'input_binding_projection.json', dict(
        inputs=[{k: r[k] for k in ('role', 'bytes', 'sha256')} for r in inputs],
        provider_roots=[dict(file_count=len(v), inventory_digest=c.digest(v))
                        for v in origin['provider_closure'].values()],
        runtime=dict(files=runtime['files'], receipt_sha256=runtime['sha256'], status=runtime['status']),
        seed_inventory=dict(old_explicit_values=old_count, all_A_tasks_excluded=128,
            complete_for_final_seal=False, new_confirmation_seeds=0), private_paths_omitted=True))
    c.save(destination / 'grant_INVALID.json', dict(schema='INVALID_H6_PHASE0_GRANT', SOURCE=None,
        approved=False, approved_by_user=False, runtime_authorization=False, one_shot=True,
        n_by_scope=None, registry_sha256=None, calls_by_scope=None,
        retry=False, next_stage_authorized=False, allowed_cpus=[], absolute_launch_command=None))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting-spec', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--runtime-cache', type=Path, help='Reuse prior verified SHA only for unchanged stat identities')
    args = parser.parse_args()
    prepare(args.accounting_spec, args.destination, args.runtime_cache)
    print('PASS_PHASE0_METADATA_ONLY_NO_NEW_SEEDS_NO_SCIENTIFIC_CALLS')
