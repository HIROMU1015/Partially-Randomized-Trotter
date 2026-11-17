#!/usr/bin/env python3
"""Build a RomuTail operational overlay; never start a controller or science.

Archived service/scientific settings are read only. Host-specific paths are
mapped explicitly and every unresolved controller file remains a visible HOLD.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap(value, paths):
    if isinstance(value, dict):
        return {key: remap(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, paths) for item in value]
    if isinstance(value, str):
        for old, new in sorted(paths.items(), key=lambda pair: len(pair[0]), reverse=True):
            if value == old or value.startswith(old.rstrip('/') + '/'):
                return new.rstrip('/') + value[len(old.rstrip('/')):]
    return value


def prepare(source_directory, snapshot_manifest, layout, source, output):
    source_directory, output = Path(source_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new deployment output directory')
    manifest_path = Path(snapshot_manifest).resolve()
    snapshot = read(manifest_path)
    paths = {row['original_root']: str(manifest_path.parent / row['snapshot_root']) for row in snapshot['roots']}
    paths.update(layout['controller_path_map'])
    code_root = Path(layout['code_root']).resolve()
    resolved_source = subprocess.check_output(['git', 'rev-parse', source + '^{commit}'], cwd=code_root, text=True).strip()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code_root, text=True).strip() != resolved_source:
        raise ValueError('Deployment code_root must be checked out at the selected integrated source')
    original_service = read(source_directory / 'SERVICE.json')
    output.mkdir(parents=True)
    profiles = {}
    for host, entry in layout['hosts'].items():
        profile = read(entry['base_profile'])
        profile = remap(profile, entry.get('profile_path_map', {}))
        dependencies = entry.get('preparation_sources', {})
        operational_paths = entry.get('preparation_paths', {})
        for extra in entry.get('additional_preparation_profiles', []):
            profile.setdefault('preparation_profiles', []).extend(read(extra).get('preparation_profiles', []))
        seen, selected = set(), []
        for rule in profile.get('preparation_profiles', []):
            key = (rule['stage'], rule['provider'], rule.get('source_commit'))
            if key in seen:
                continue
            seen.add(key)
            for commit, directory in dependencies.items():
                rule['profile'].setdefault('preparation_sources', {})[commit] = directory
            rule['profile'].setdefault('preparation_paths', {}).update(operational_paths)
            selected.append(rule)
        profile['preparation_profiles'] = selected
        profile.setdefault('preparation_paths', {}).update(operational_paths)
        profile.setdefault('preparation_sources', {}).update(dependencies)
        profile['work_dir'] = entry['work_dir']
        profile['artifact_cache_directory'] = entry['artifact_cache_directory']
        profile.setdefault('slot_policy', {'mode': 'auto', 'initial_slots': 1})
        # Worker resources are operational; preserve verified Q3 interpreter,
        # scientific root, route views and timed adapter manifest.
        profile['slots'] = entry.get('slots', profile.get('slots', 1))
        for nested in profile.get('route_profiles', []):
            nested['profile']['work_dir'] = profile['work_dir']
        put(output / 'profiles' / (host + '.json'), profile)
        profiles[host] = profile
    holds = []
    problem = remap(read(source_directory / 'PROBLEM.json'), paths)
    put(output / 'PROBLEM.json', problem)
    admission = dict(schema='RPE_Q3_HOST_ADMISSION_V1', hosts={
        'server2': dict(accepting=True), 'local': dict(accepting=True),
        'server1': dict(accepting=False, max_slots=0, reason='USER_REPORTED_STOPPED'),
        'local0-wsl': dict(accepting=False, optional=True, reason='EXISTING_PAUSE_PRESERVED')})
    put(output / 'HOST_ADMISSION.json', admission)
    original_rounds = {}
    for name in ('ROUND.default.json', 'ROUND.absolute-kappa.json'):
        original = read(source_directory / name)
        original_rounds[name] = original
        round_value = remap(original, paths)
        # One controller-owned store across releases and all run queues.
        # Its saved state must be reconciled before activating this deployment.
        round_value.setdefault('price_validation', {})['store'] = str(
            Path(layout['state_directory']).parent/'price-validation'/'models.sqlite3')
        round_value['price_validation']['publish_shared_models']=True
        distributed = round_value['executor']['distributed']
        distributed.update(computation_root=profiles['local']['root'],
            local_worker=deepcopy(profiles['local']), worker_script=str(code_root / 'scripts/run_rpe_q3_worker.py'),
            host_admission=str(output / 'HOST_ADMISSION.json'), required_hosts=[],
            preparation_profiles={host: str(output / 'profiles' / (host + '.json')) for host in profiles})
        distributed['affinity_wait_seconds'] = 0
        distributed.setdefault('auto_allocation', {}).update(
            priority_groups=[['server2', 'local', 'server1'], ['local0-wsl']], discovery_grace_seconds=0)
        if layout.get('relay_workers'):
            distributed['relay']=dict(workers=deepcopy(layout['relay_workers']),poll_seconds=1,request_timeout_seconds=30)
        distributed.setdefault('preparation', {}).setdefault('budgets', dict(signal=0, compile=0, audit=0))
        completion = round_value.setdefault('candidate_completion', {})
        completion.update(source=layout.get('candidate_completion_source', resolved_source),
                          controller_paths=deepcopy(layout['controller_paths']))
        if round_value.get('price_rules_overrides', {}).get('registry_file'):
            completion['price_registry_file'] = round_value['price_rules_overrides']['registry_file']
        # Explicitly preserve all finite budgets and the original unapproved
        # template. The caller's ordinary scope controls future authorized work.
        if round_value['max_new_q3_per_method'] != original['max_new_q3_per_method']:
            raise ValueError('Operational overlay changed Q3 budgets')
        if round_value['science_approved'] != original['science_approved']:
            raise ValueError('Operational overlay changed scientific permission')
        if distributed['preparation']['budgets'] != original['executor']['distributed']['preparation']['budgets']:
            raise ValueError('Operational overlay changed preparation budgets')
        put(output / name, round_value)
    service = deepcopy(original_service)
    service.update(code_root=str(code_root), code_source=resolved_source, problem=str(output / 'PROBLEM.json'),
        round=str(output / 'ROUND.default.json'), python=layout['controller_python'],
        state_directory=layout['state_directory'], config_path=str(output / 'SERVICE.json'), entry='schedule')
    service['absolute_kappa_route'].update(problem=service['problem'], round=str(output / 'ROUND.absolute-kappa.json'))
    if layout.get('publication'):
        service['publication'] = deepcopy(layout['publication'])
    service.pop('previous_service', None)
    service['fixed_settings'] = {str(path): checksum(path) for path in output.rglob('*.json')}
    service['migration'] = dict(original_service=str(source_directory / 'SERVICE.json'),
        original_service_sha256=checksum(source_directory / 'SERVICE.json'), original_code_source=original_service['code_source'],
        scientific_computation_source=original_service['computation_source'], snapshot_manifest=str(manifest_path),
        snapshot_manifest_sha256=checksum(manifest_path), archived_source_inputs_modified=False)
    put(output / 'SERVICE.json', service)

    def inspect(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('local_worker', 'preparation_profiles', 'relay'):
                    continue
                inspect(item, where + '/' + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, where + '/' + str(index))
        elif isinstance(value, str) and value.startswith('/'):
            if not Path(value).exists():
                holds.append(dict(field=where, path=value, reason='CONTROLLER_LOCAL_PATH_MISSING'))
    inspect(problem, 'PROBLEM')
    for name in original_rounds:
        inspect(read(output / name), name)
    for key, value in layout['controller_paths'].items():
        if isinstance(value, str) and not Path(value).exists():
            holds.append(dict(field='controller_paths/' + key, path=value, reason='CONTROLLER_LOCAL_PATH_MISSING'))
    report = dict(schema='ROMUTAIL_CONTROLLER_OPERATIONAL_DEPLOYMENT_V1', source=resolved_source,
        service=str(output / 'SERVICE.json'), state_directory=layout['state_directory'],
        computation_source=original_service['computation_source'], prior_run_dirs=read(output / 'ROUND.default.json')['executor']['prior_run_dirs'],
        controller_path_map=paths, controller_paths=layout['controller_paths'],
        host_availability=dict(server2='AVAILABLE', local='AVAILABLE', server1='OFFLINE_DEFERRED', **{'local0-wsl':'PAUSED_OFF'}),
        q3_and_preparation_budgets_preserved=True, archived_inputs_modified=False,
        holds=holds, status='CONFIGURED' if not holds else 'CONFIGURED_WITH_PATH_HOLDS',
        scientific_calls=0, compile_calls=0, queue_operations=0, controllers_started=0)
    put(output / 'DEPLOYMENT.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--snapshot-manifest', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source_directory, args.snapshot_manifest, read(args.layout), args.source, args.output)
    print(json.dumps(dict(status=report['status'], service=report['service'], holds=len(report['holds']))))


if __name__ == '__main__':
    main()
