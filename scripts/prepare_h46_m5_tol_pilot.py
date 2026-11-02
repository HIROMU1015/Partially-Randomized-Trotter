"""Seal INVALID proposals by hashing existing bytes; never restore molecular arrays."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import h46_m5_tol_contract as c

F_RESULT = '97d25cc431516d720ce91c7cb4f148e061d8ea34'
F_BUNDLE = 'artifacts/resource_applicability/track_a_h4_followup/2026-10-18/'


def fixed_json(commit, path):
    return json.loads(subprocess.check_output(['git', 'show', commit+':'+path], cwd=c.ROOT))


def seal(path, role):
    path = Path(path).resolve()
    return dict(path=str(path), role=role, bytes=path.stat().st_size, sha256=c.sha(path))


def prepare(old_bundle_path, fp_config_path, destination, source, extra_seed_receipts=()):
    destination = Path(destination).resolve()
    c.require(not destination.exists(), 'NEW_PREPARATION_DIRECTORY_REQUIRED')
    for path in c.SOURCE_PATHS:
        c.require(subprocess.check_output(['git', 'show', source+':'+path], cwd=c.ROOT)
                  == (c.ROOT/path).read_bytes(), 'COMMITTED_SCIENTIFIC_SOURCE:'+path)
    old_bundle = c.read(old_bundle_path)
    old = {k: c.read(v) for k, v in old_bundle['configs'].items()}
    c.require(set(old) == set(c.legacy.LANES), 'FIXED_OLD_PREPARATION_BUNDLE')
    checked = {}
    for cfg in old.values():
        for row in cfg['sealed_files']:
            if row['path'] not in checked:
                current = seal(row['path'], row['role'])
                c.require(current['sha256'] == row['sha256']
                          and current['bytes'] == row['bytes'], 'OLD_PREPARATION_BYTES_CHANGED')
                checked[row['path']] = current
    fp = c.read(fp_config_path)
    published = fixed_json(F_RESULT, F_BUNDLE+'FP_binding.json')
    c.require(c.core.digest(fp) == published['config_digest'], 'ORIGINAL_FP_CONFIG_DIGEST')
    c.require(fp['binding'] == published['input_binding'], 'PUBLISHED_FP_BINDING')
    h4_inputs = {}
    for role, name in (('tol_snapshot', 'H4_tol_snapshot.npz'), ('tol_receipt', 'H4_tol_receipt.json')):
        previous = next(row for row in fp['inputs'] if row['role'] == role)
        current = seal(fp['mapping'][role], role)
        c.require(current['sha256'] == previous['sha256']
                  and current['bytes'] == previous['bytes'], 'H4_TOL_ORIGINAL_INPUT_SHA')
        h4_inputs[name] = current
    f_completion = fixed_json(F_RESULT, F_BUNDLE+'F_completion_receipt.json')
    c.require(f_completion['native_exit']['exit_code'] == 0
              and f_completion['final_gate']['status'] == 'PASS'
              and h4_inputs['H4_tol_receipt.json']['sha256'] == f_completion['result_uncompressed_sha256'],
              'H4_F_NORMAL_COMPLETION_AND_RECEIPT_BYTES')
    h6_inputs = copy.deepcopy(old['H6_P2_DET']['inputs'])
    for row in h6_inputs.values():
        c.require(c.sha(row['path']) == row['sha256'], 'H6_ORIGINAL_INPUT_SHA')
    inputs = {'H4': h4_inputs, 'H6': h6_inputs}
    provider = Path(old['H6_P2_DET']['provider_root']).resolve()
    runtime = copy.deepcopy(old['H6_P2_DET']['runtime'])
    c.require(all(cfg['runtime'] == runtime for cfg in old.values()), 'ONE_ORIGINAL_RUNTIME')
    common = [row for row in checked.values() if Path(row['path']).is_relative_to(provider)
              or '/site-packages/' in row['path'] or row['path'] == runtime['python_target']]
    c.require(common, 'USED_PROVIDER_RUNTIME_CLOSURE')
    common += [seal(c.ROOT/path, 'scientific_SOURCE:'+path) for path in c.SOURCE_PATHS]
    # Keep every old provisional seed excluded, including unused old pilot seeds.
    excluded = set()
    seed_receipts = []
    for cfg in old.values():
        for key in ('excluded_seed_path', 'registry_path'):
            if cfg[key] in seed_receipts:
                continue
            seed_receipts.append(cfg[key])
            data = c.read(cfg[key])
            if key == 'excluded_seed_path':
                excluded.update(data['values'])
            else:
                excluded.update(v for lane in data['lanes'].values() for task in lane.values() for v in task.values())
    for path in extra_seed_receipts:
        path = str(Path(path).resolve())
        if path not in seed_receipts:
            data = c.read(path)
            excluded.update(v for lane in data['lanes'].values() for task in lane.values() for v in task.values())
            seed_receipts.append(path)
    seeds = c.registry(source, excluded)
    # All molecular operations remain behind authorization; this is seed metadata only.
    destination.mkdir(parents=True, exist_ok=False)
    excluded_path = destination/'excluded_seeds_PRIVATE.json'
    registry_path = destination/'seed_registry_PROVISIONAL_PRIVATE.json'
    c.save(excluded_path, dict(values=sorted(excluded),
        scope='KNOWN_OLD_HIGHER_PF_AND_OLD_M5_PROVISIONAL_RECEIPTS',
        evidence=[seal(p, 'old_seed_receipt') for p in seed_receipts],
        unknown_external_registry_coverage_claimed=False))
    c.save(registry_path, dict(SOURCE=source, purpose='TOL_ONLY_TECHNICAL_PILOT_NOT_CONFIRMATION', lanes=seeds))
    common += [seal(excluded_path, 'seed_exclusion'), seal(registry_path, 'provisional_seed_registry')]
    configs = {}
    reports = {}
    root = destination/'future_run01'
    for lane in c.LANES:
        system = lane[:2]
        inp = copy.deepcopy(inputs[system])
        files = common + [seal(v['path'], 'saved_input:'+k) for k, v in inp.items()]
        binding = copy.deepcopy(c.TARGETS[system]['binding'])
        plan = dict(status='TECHNICALLY_SEALED_EXECUTION_UNAPPROVED', SOURCE=source,
            scope=c.scope(lane), native_policy=c.POLICY,
            closure_digest=c.core.digest(files), input_digest=c.core.digest(inp),
            binding_digest=c.core.digest(binding), runtime_digest=c.core.digest(runtime),
            seed_registry_sha256=c.sha(registry_path), phase3_authorized=False,
            main_comparison_adopted=False, AD03='DEFERRED_ADDITIONAL_PF')
        plan_path = destination/(lane+'_plan_DRAFT_PRIVATE.json')
        c.save(plan_path, plan)
        paths = dict(output=str(root/(lane+'_output')), ledger=str(root/(lane+'_ledger')),
                     marker=str(root/(lane+'_CONSUMED.json')))
        grant = dict(valid=False, grant_id=None, approval_reference=None, lane=lane,
            SOURCE=source, plan_sha256=c.sha(plan_path), maximum_axes=c.scope(lane)['compile_axes'],
            maximum_trajectories=6 if lane.endswith('PR') else 0,
            maximum_reference_jobs=c.scope(lane)['reference_jobs'], retry=False, phase3_authorized=False, **paths)
        cfg = dict(authorized=False, lane=lane, system=system, SOURCE=source,
            native_policy=c.POLICY, provider_root=str(provider), runtime=runtime,
            inputs=inp, binding=binding, sealed_files=files, plan=plan, plan_path=str(plan_path),
            plan_sha256=c.sha(plan_path), registry_path=str(registry_path),
            excluded_seed_path=str(excluded_path), grant=grant, run_root=str(root),
            reference_root=str(root/(system+'_reference')),
            RSS_proxy_bytes=old[system+'_P2_'+c.scope(lane)['kind']]['RSS_proxy_bytes'], **paths)
        c.validate_inputs(cfg)
        c.verify_seals(cfg)
        config_path = destination/(lane+'_config_DRAFT_PRIVATE.json')
        c.save(config_path, cfg)
        c.save(destination/(lane+'_grant_INVALID.json'), grant)
        configs[lane] = str(config_path)
        reports[lane] = dict(plan_sha256=c.sha(plan_path), closure_files=len(files),
            input_series=binding['series'], maximum_axes=grant['maximum_axes'], grant_valid=False)
    c.save(destination/'bundle_DRAFT_PRIVATE.json', dict(configs=configs))
    result = dict(status='TOL_ONLY_FOUR_CONTRACTS_TECHNICALLY_SEALED_EXECUTION_UNAPPROVED',
        SOURCE=source, lanes=reports, fresh_seed_values=sum(len(v) for l in seeds.values() for v in l.values()),
        known_excluded_seed_values=len(excluded), collisions=0, molecular_arrays_restored=False,
        sampling=0, signal=0, compile=0, SCF_DF_state_generation=0, valid_grants=0,
        used_runtime_file_entries=len(common), unknown_external_seed_coverage=False,
        old_original_files_verified=len(checked), future_run_root_exists=root.exists(), mandatory_STOP=True)
    c.save(destination/'preparation_summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-bundle', type=Path, required=True)
    p.add_argument('--fp-config', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--extra-seed-receipt', type=Path, action='append', default=[])
    a = p.parse_args()
    print(json.dumps(prepare(a.old_bundle, a.fp_config, a.destination, a.source, a.extra_seed_receipt)))


if __name__ == '__main__':
    main()
