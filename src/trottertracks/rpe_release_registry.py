"""Resolve validated controller releases while retaining each run's fixed inputs.

The registry is an ordinary deployment pointer, not an execution permission.
Scientific authorization and frozen run contracts remain with the normal entry.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import os


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def load_service(path):
    """Read the old CURRENT_SERVICE pointer or the new registry bootstrap."""
    path = Path(path).resolve(); value = read(path)
    if 'service_config' in value:
        return load_service(value['service_config'])
    value['config_path'] = str(path)
    return value


def input_files(config):
    files={key:config[key] for key in ('problem','round')}
    if config.get('pbs_walltime_policy'):
        files['pbs_walltime_policy'] = config['pbs_walltime_policy']
    for key in ('problem','round'):
        if config.get('absolute_kappa_route',{}).get(key):
            files['absolute_kappa_route.'+key]=config['absolute_kappa_route'][key]
    return files


def register(registry_path, release_id, service_path, *, activate=False, evidence=None, expected_current=None):
    """One registration/switch operation; retain existing registered versions."""
    service_path = str(Path(service_path).resolve())
    config = load_service(service_path)
    for key in ('code_root', 'code_source', 'computation_source', 'problem', 'round', 'python'):
        if not config.get(key):
            raise ValueError('Release service missing ' + key)
    inputs=input_files(config)
    for path in inputs.values():
        if not Path(path).is_file():
            raise ValueError('Release input missing: ' + path)
    registry = read(registry_path) if Path(registry_path).exists() else dict(schema='RPE_VALIDATED_RELEASES_V1', releases={})
    if activate and expected_current is not None and registry.get('current')!=expected_current:
        raise ValueError('Current release changed; reconcile before adoption: '+str(registry.get('current')))
    entry = dict(service_config=service_path, status='VALIDATED', implementation_source=config['code_source'],
                 computation_source=config['computation_source'], service_sha256=fingerprint(config),
                 input_sha256={key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key,path in inputs.items()})
    if config.get('release_manifest'):
        from .rpe_release_deployment import manifest_reference
        manifest=read(config['release_manifest'])
        if manifest['release_id']!=release_id or manifest['components']['optimizer']['source']!=config['code_source']:
            raise ValueError('Component manifest differs from the selected release/controller')
        entry['components']=manifest_reference(manifest)
    if evidence is not None:
        entry['validation_evidence'] = evidence
    previous = registry['releases'].get(release_id)
    if previous and previous != entry:
        raise ValueError('Registered release is immutable: ' + release_id)
    registry['releases'][release_id] = entry
    if activate:
        registry['current'] = release_id
    write(registry_path, registry)
    return registry


def select(config, request):
    """Return selected service; resume/status/stop prefer a run's pinned service."""
    if not config.get('release_registry'):
        return config
    state = Path(config['state_directory'])
    run_id = request.get('run_id')
    target = state / 'CURRENT_TARGET.json'
    if not request.get('scope') and (run_id or target.exists()):
        if run_id:
            index = read(state / 'RUNS.json') if (state / 'RUNS.json').exists() else {}
            if run_id not in index:
                raise ValueError('Unknown existing run_id: ' + str(run_id))
            run = Path(index[run_id]['run_directory'])
        else:
            run = Path(read(target)['run_directory'])
        pin = run / 'CONTROLLER_CONFIG.json'
        if pin.exists():
            selected = read(pin)
            chosen = selected.get('release', {}).get('id')
            if request.get('release') and request['release'] != chosen:
                raise ValueError('Existing run keeps its pinned release: ' + str(chosen))
            selected['state_directory'] = str(state)
            selected['config_path'] = str(pin)
            return selected
    registry = read(config['release_registry'])
    release_id = request.get('release') or registry.get('current')
    entry = registry.get('releases', {}).get(release_id)
    if not entry or entry.get('status') != 'VALIDATED':
        raise ValueError('No validated release selected: ' + str(release_id))
    selected = load_service(entry['service_config'])
    if entry.get('service_sha256') and fingerprint(selected) != entry['service_sha256']:
        raise ValueError('Registered service changed: ' + release_id)
    for key, expected in entry.get('input_sha256', {}).items():
        if hashlib.sha256(Path(input_files(selected)[key]).read_bytes()).hexdigest() != expected:
            raise ValueError('Registered release input changed: ' + key)
    selected['state_directory'] = str(state)
    selected['release'] = dict(id=release_id, registry=str(Path(config['release_registry']).resolve()),
                               implementation_source=selected['code_source'],
                               computation_source=selected['computation_source'])
    if entry.get('components'):
        from .rpe_release_deployment import manifest_reference
        if manifest_reference(read(selected['release_manifest']))!=entry['components']:
            raise ValueError('Registered component versions changed: '+release_id)
        selected['release']['components']=entry['components']
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--expected-current',help='Refuse adoption if another session changed current')
    parser.add_argument('--evidence', help='Location of successful integration validation record')
    args = parser.parse_args()
    value = register(args.registry, args.release, args.service, activate=args.activate, evidence=args.evidence,
                     expected_current=args.expected_current)
    print(json.dumps(dict(current=value.get('current'), release=args.release)))


if __name__ == '__main__':
    main()
