#!/usr/bin/env python3
"""Merge portable DF deployment fixes into a new mixed-host release.

Only configuration is written. Existing releases, run state, host admission and
scientific settings remain owned by their original contracts. Installation and
ordinary worker dry-checks must precede registration/activation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.rpe_release_deployment import manifest_reference


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def profile(value):
    return deepcopy(value) if isinstance(value, dict) else read(value)


def unique(rows, **keys):
    matches = [row for row in rows if all(row.get(k) == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError('Expected one deployment rule: ' + str(keys))
    return deepcopy(matches[0])


def prepare(base_service, accepted_service, output, release_id, *, source_bundle=None):
    output = Path(output).resolve()
    if not release_id or Path(release_id).name != release_id or release_id in ('.', '..'):
        raise ValueError('Use a simple new release identifier')
    if output.exists():
        raise ValueError('Use a new output directory; pinned releases remain unchanged')
    service, accepted = read(base_service), read(accepted_service)
    if service['computation_source'] != accepted['computation_source']:
        raise ValueError('Scientific Q3 SOURCE differs')
    reference_round = read(accepted['round'])
    reference_dist = reference_round['executor']['distributed']
    settings = reference_round['candidate_completion']
    sources = {kind: settings['df_' + kind + '_source'] for kind in ('signal', 'price')}
    if any(len(s) != 40 or any(c not in '0123456789abcdef' for c in s)
           for s in [accepted['code_source'], *sources.values()]):
        raise ValueError('Deployment requires full fixed SOURCE commits')
    manifest = read(service['release_manifest'])
    manifest['release_id'] = release_id
    manifest['components']['optimizer']['source'] = accepted['code_source']
    hosts = list(service['host_layouts'])
    old_round = read(service['round'])
    old_dist = old_round['executor']['distributed']
    alternate_path = service.get('absolute_kappa_route', {}).get('round')
    if alternate_path:
        alternate = read(alternate_path)
        alternate_profiles = alternate['executor']['distributed']['preparation_profiles']
        if ({h: profile(p) for h, p in alternate_profiles.items()} !=
                {h: profile(p) for h, p in old_dist['preparation_profiles'].items()}):
            raise ValueError('Alternate route has different host profiles; prepare it separately')
        for kind in ('signal', 'price'):
            key = 'df_' + kind + '_source'
            if alternate['candidate_completion'][key] != old_round['candidate_completion'][key]:
                raise ValueError('Alternate route has different fixed DF sources')
    pbs_hosts = {w['host_id'] for w in old_dist['pbs']['workers']}
    rules = [('signal', 'DF_SIGNALS', 'signal'),
             ('compile', 'DF_CALIBRATION', 'price'), ('compile', 'DF_HOLDOUT', 'price'),
             ('audit', 'DF_MODEL_FREEZE', 'price'), ('audit', 'DF_HOLDOUT_COMPARE', 'price')]
    layouts, profiles = {}, {}
    for host in hosts:
        layout = read(service['host_layouts'][host])
        original = profile(old_dist['preparation_profiles'][host])
        updated = deepcopy(original)
        install_parent = Path(layout['component_roots']['optimizer']).parent
        optimizer = str(install_parent / ('mixed-df-' + accepted['code_source'][:12]))
        if host == 'local':
            optimizer = accepted['code_root']
        layout['component_roots']['optimizer'] = optimizer
        roots = {kind: str(install_parent / ('mixed-df-' + source[:12]))
                 for kind, source in sources.items()}
        if host == 'local':
            roots = {kind: reference_dist['preparation']['bundle_validation_sources'][source]
                     for kind, source in sources.items()}
        accepted_profile = None
        if host in pbs_hosts:
            accepted_profile = profile(reference_dist['preparation_profiles'][host])
        for stage, provider, kind in rules:
            source = sources[kind]
            if accepted_profile:
                selected = unique(accepted_profile['preparation_profiles'], stage=stage,
                                  provider=provider, source_commit=source)
                # Retain the tested interpreter and source views, while using
                # this host's normal work/cache directories rather than isolation.
                roots[kind] = selected['profile']['preparation_sources'][source]
            else:
                prior_source = old_round['candidate_completion']['df_' + kind + '_source']
                selected = unique(original['preparation_profiles'], stage=stage,
                                  provider=provider, source_commit=prior_source)
                selected['source_commit'] = source
            selected['profile'].setdefault('preparation_sources', {})[source] = roots[kind]
            for key in ('work_dir', 'artifact_cache_directory', 'preparation_metadata_cache_directory',
                        'storage_limit', 'pbs', 'slots', 'slot_policy'):
                if key in original:
                    selected['profile'][key] = deepcopy(original[key])
            # Add exact-source rules; old rules remain available for old inputs.
            if any(r.get('stage') == stage and r.get('provider') == provider and
                   r.get('source_commit') == source for r in updated['preparation_profiles']):
                raise ValueError('Portable DF rule already present; inspect before rebuilding')
            updated['preparation_profiles'].append(selected)
            runtime_name = 'portable_df_' + provider.lower()
            component = 'portable_df_' + kind
            layout.setdefault('runtimes', {})[runtime_name] = dict(
                python=selected['profile']['interpreter'], modules={}, components=[component])
            manifest.setdefault('runtime_contracts', {}).setdefault(host, []).append(dict(
                stage=stage, provider=provider, source=source, runtime=runtime_name,
                request_runtime_contract='preparation.runtime_requirements',
                compiler_binding='preparation.compiler_bindings',
                runtime_observation_status='REQUIRES_FRESH_DRY_CHECK'))
        for kind, source in sources.items():
            layout['component_roots']['portable_df_' + kind] = roots[kind]
        layout['runtime_contracts_status'] = 'REQUIRES_FRESH_DRY_CHECK'
        profiles[host], layouts[host] = updated, layout
    for kind, source in sources.items():
        manifest['components']['portable_df_' + kind] = dict(
            source=source, hosts=hosts, required_paths=[
                'scripts/run_pr_df_candidate_signal_jobs.py' if kind == 'signal'
                else 'scripts/run_pr_df_candidate_prices.py',
                'src/trottertracks/pr_df_descriptor_extension.py',
                'src/trottertracks/rpe_preparation_paths.py'])
    for stage, _, kind in rules:
        name = 'portable_df_' + kind
        names = manifest['stage_components'].setdefault(stage, [])
        if name not in names:
            names.append(name)
    reference = manifest_reference(manifest)
    policy = read(service['pbs_walltime_policy'])
    accepted_policy = read(accepted['pbs_walltime_policy'])
    additions = [('q3', unique(accepted_policy['q3'], method='PR_DF',
                             route_kind='prdf_d2_common_entry'))]
    for stage, provider, kind in rules:
        additions.append(('preparation', unique(accepted_policy['preparation'], stage=stage,
                                               provider=provider, source=sources[kind])))
    for group, row in additions:
        keys = ('method', 'route_kind', 'view_commit', 'adapter_source') if group == 'q3' else (
            'stage', 'provider', 'source')
        matches = [r for r in policy[group] if all(r.get(k) == row.get(k) for k in keys)]
        if matches:
            if len(matches) != 1 or any(matches[0].get(k) != row.get(k)
                                       for k in ('seconds', 'memory_bytes')):
                raise ValueError('Existing PBS resource estimate conflicts')
        else:
            row['id'] = 'mixed-' + row['id']
            policy[group].append(row)
    policy['id'] = release_id
    manifest_path = output / 'RELEASE_COMPONENTS.json'
    put(manifest_path, manifest)
    service['host_layouts'] = {}
    for host, layout in layouts.items():
        profile_path = output / 'profiles' / (host + '.json')
        layout_path = output / 'hosts' / (host + '.json')
        # Central metadata paths are used locally; remote copies are explicitly
        # staged before the host is allowed to participate in this release.
        metadata = output if host == 'local' else Path(layout['manifest_path']).parent.parent / release_id
        layout.update(manifest_path=str(metadata / 'RELEASE_COMPONENTS.json'),
                      profile_path=str(profile_path if host == 'local' else metadata / 'PROFILE.json'),
                      layout_path=str(layout_path if host == 'local' else metadata / 'HOST.json'))
        if source_bundle is not None:
            layout.setdefault('bundle_files', []).append(str(metadata / 'DF_FIXES.bundle'))
        put(profile_path, profiles[host])
        put(layout_path, layout)
        service['host_layouts'][host] = str(layout_path)
    round_paths = [('round', service['round'])]
    if service.get('absolute_kappa_route', {}).get('round'):
        round_paths.append(('absolute_kappa_route', service['absolute_kappa_route']['round']))
    for route, path in round_paths:
        value = read(path)
        completion = value['candidate_completion']
        completion.update(df_signal_source=sources['signal'], df_price_source=sources['price'])
        dist = value['executor']['distributed']
        prep = dist['preparation']
        for key in ('df_snapshot_directory', 'bundle_fixed_source_only'):
            prep[key] = deepcopy(reference_dist['preparation'][key])
        for source in sources.values():
            prep.setdefault('bundle_validation_sources', {})[source] = reference_dist['preparation']['bundle_validation_sources'][source]
        dist['preparation_profiles'] = {host: str(output / 'profiles' / (host + '.json')) for host in hosts}
        dist.update(required_release=reference, release_manifest=str(manifest_path),
                    local_deployment_layout=service['host_layouts']['local'])
        if 'local_worker' in dist:
            dist['local_worker'] = deepcopy(profiles['local'])
        for worker in dist.get('relay', {}).get('workers', []):
            host = worker['host_id']
            if host not in layouts or worker.get('enabled') is False:
                continue
            layout = layouts[host]
            worker.update(root=layout['component_roots']['optimizer'],
                          profile=layout['profile_path'])
            worker['deployment'] = dict(
                entry=str(Path(worker['root']) / 'scripts/manage_rpe_release_deployment.py'),
                manifest=layout['manifest_path'], layout=layout['layout_path'])
        for worker in dist['pbs']['workers']:
            worker['profile'] = deepcopy(profiles[worker['host_id']])
        if len(pbs_hosts) != 1:
            raise ValueError('Expected one PBS deployment layout')
        pbs_layout = layouts[next(iter(pbs_hosts))]
        dist['pbs']['deployment'] = dict(manifest=pbs_layout['manifest_path'],
                                         layout=pbs_layout['layout_path'])
        target = output / ('ROUND.template.json' if route == 'round' else 'ROUND.absolute-kappa.json')
        put(target, value)
        if route == 'round':
            service['round'] = str(target)
        else:
            service['absolute_kappa_route']['round'] = str(target)
    put(output / 'PBS_WALLTIME_POLICY.json', policy)
    if source_bundle is not None:
        (output / 'DF_FIXES.bundle').write_bytes(Path(source_bundle).read_bytes())
    service.update(code_root=accepted['code_root'], code_source=accepted['code_source'],
                   release_manifest=str(manifest_path), config_path=str(output / 'SERVICE.json'),
                   pbs_walltime_policy=str(output / 'PBS_WALLTIME_POLICY.json'))
    # Keep normal scaled-shot, publication, admission, state, budgets and all
    # continuation/STOP/descriptor settings. Never import the LD8 run override.
    fixed = service.setdefault('fixed_settings', {})
    for path in output.rglob('*.json'):
        fixed[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    put(output / 'SERVICE.json', service)
    report = dict(release_id=release_id, status='PREPARED_REQUIRES_HOST_VALIDATION',
                  base_service=str(Path(base_service).resolve()),
                  accepted_service=str(Path(accepted_service).resolve()),
                  service=str(output / 'SERVICE.json'), hosts=hosts,
                  scientific_calls=0, queue_operations=0, registry_modified=False,
                  note='PBS estimates inherited as reservation estimates. No new scientific coverage claim. '
                       'Scoped LD8 descriptor, signals, retry, budgets and NEHALEM override are not promoted.')
    put(output / 'PREPARATION.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-service', 'accepted-service', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--source-bundle', type=Path,
                        help='Existing fixed-code Git bundle to stage with each host layout')
    args = parser.parse_args()
    print(json.dumps(prepare(args.base_service, args.accepted_service, args.output, args.release,
                             source_bundle=args.source_bundle)))


if __name__ == '__main__':
    main()
