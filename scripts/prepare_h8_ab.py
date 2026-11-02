"""Metadata-only H8 A/B snapshots; no numerical imports or grant issuance."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import h8_ab_contract as c


def collect_seeds(value):
    out = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if 'seed' in key.lower():
                if type(child) is int: out.add(child)
                elif isinstance(child, list): out.update(x for x in child if type(x) is int)
            out.update(collect_seeds(child))
    elif isinstance(value, list):
        for child in value: out.update(collect_seeds(child))
    return out


def flat_alternates(objects):
    seen = set(); ordered = []
    def visit(directory):
        directory = Path(directory).resolve()
        if directory in seen: return
        seen.add(directory); alt = directory / 'info/alternates'
        if alt.exists():
            for line in alt.read_text().splitlines():
                p = Path(line); visit(p if p.is_absolute() else directory / p)
        ordered.append(directory)
    visit(objects)
    return ordered


def local_source_closure(names, excluded=()):
    known = set(names); pending = list(known)
    while pending:
        tree = ast.parse((c.ROOT / pending.pop()).read_text()); modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == 'import_module' and node.args
                  and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                modules.append(node.args[0].value)
        for module in modules:
            path = c.ROOT / 'scripts' / (module.replace('.', '/') + '.py')
            if path.exists():
                relative = str(path.relative_to(c.ROOT))
                if relative in excluded: continue
                if relative not in known: known.add(relative); pending.append(relative)
    return known


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--old-config', type=Path, required=True)
    args = parser.parse_args(); args.private.mkdir(parents=True, exist_ok=False)
    old = c.read(args.old_config); inputs = args.private / 'inputs'; inputs.mkdir()
    mapping = {}; rows = []
    def add(role, source, name=None):
        source = Path(source); dest = inputs / (name or role)
        shutil.copyfile(source, dest)
        c.require(c.sha(source) == c.sha(dest), 'SNAPSHOT_COPY_BYTES')
        rows.append(dict(role=role, path=str(dest), bytes=dest.stat().st_size, sha256=c.sha(dest)))
        mapping[role] = str(dest)
    for role in ('h8_input_snapshot.npz', 'snapshot_receipt.json', 'df_receipt.json', 'projection_receipt.json',
                 'h8_preparation_export', 'preparation_identity_receipts', 'full_H_reference', 'font_seed'):
        row = next(x for x in old['inputs'] if x['role'] == role)
        c.require(c.sha(row['path']) == row['sha256'], 'OLD_INPUT_BYTES')
        add(role, row['path'])
    pilot = Path('/home/AbeHiromu/projects/h8-pr-s2-conditional-pilot/20261017-01')
    for role in ('exploration_registration', 'confirmation_registration', 'original_trajectory_q4', 'original_trajectory_q8'):
        add(role, pilot / 'inputs' / role)
    for q in (4, 8): add('old_pilot_q' + str(q), pilot / 'run02' / f'pilot_q{q}_accepted.json')
    origin = Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-resource-map-v1/run02/signal')
    for q, index in ((2, 1), (4, 2), (8, 3)):
        add('primitive_q' + str(q), origin / f'group_{index:04d}' / 'primitive_validation.json')
        if q in (4, 8): add('old_signal_q' + str(q), origin / f'group_{index:04d}' / 'signal_summary.json')
    seeds = set()
    for role in ('exploration_registration', 'confirmation_registration', 'original_trajectory_q4',
                 'original_trajectory_q8', 'old_pilot_q4', 'old_pilot_q8'):
        seeds.update(collect_seeds(c.read(mapping[role])))
    c.require(bool(seeds), 'SAVED_SEED_REGISTRY_REQUIRED')
    provider = args.private / 'provider'
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', old['provider_root'], str(provider)], check=True)
    (provider / '.git/objects/info/alternates').write_text(''.join(str(p) + '\n' for p in
        flat_alternates(Path(old['provider_root']) / '.git/objects')))
    subprocess.run(['git', 'checkout', '--detach', old['provider_source']], cwd=provider, check=True)
    config = deepcopy(old)
    for key in ('predictions', 'targets'): config.pop(key, None)
    config.update(schema='H8_AB_CONFIG_V1', provider_root=str(provider), inputs=rows, mapping=mapping,
                  old_seed_values=sorted(seeds))
    config['font_seed']['path'] = mapping['font_seed']
    sources = set(old['source_hashes'])
    sources.update('scripts/' + x for x in ('h8_ab_contract.py', 'h8_ab_native.py', 'h8_ab_sampling.py',
        'h8_ab_science.py', 'h8_ab_signal.py', 'run_h8_ab.py', 'h8_ab_witness.py', 'prepare_h8_ab.py',
        'h6_dba_science.py', 'h6_dba_native.py', 'h6_dba_sampling.py', 'h6_dba_signal.py'))
    sources.add('src/trotterlib/h8_rare_tail_selection.py')
    sources.add('tests/test_h8_ab_validation.py')
    sources = local_source_closure(sources)
    config['source_hashes'] = {name: c.sha(c.ROOT / name) for name in sorted(sources)}
    c.save(args.private / 'config_prepared.json', config)
    public = c.ROOT / 'artifacts/resource_applicability/track_a_h8_ab/2026-10-18'
    public.mkdir(parents=True, exist_ok=False)
    c.save(public / 'plan_v1.json', c.plan())
    c.save(public / 'input_binding_v1.json', dict(binding=config['binding'],
        inputs=[{k: v for k, v in row.items() if k != 'path'} for row in rows],
        provider_SOURCE=config['provider_source'], saved_seed_values=len(seeds),
        runtime_file_count=len(config['runtime_file_hashes']), preparation_metadata_only=True,
        grant_issued=False, SOURCE_not_yet_sealed=True))
    print(dict(inputs=len(rows), saved_seed_values=len(seeds), source_files=len(sources)))


if __name__ == '__main__': main()
