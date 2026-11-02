"""Explicit new-environment generation grants; immutable old recipes, new provenance.

This module has no import-time scientific operations. It is loaded by file path so
the separately pinned scientific provider remains the only trotterlib package.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def claim(path, grant):
    require(grant.get('approved_by_user') is True and grant.get('valid') is True,
            'EXPLICIT_MIGRATION_APPROVAL_REQUIRED')
    require(grant.get('retry') is False, 'NO_GRANT_RETRY')
    save(path, dict(state='CONSUMED_NO_RETRY_NO_REFUND', grant=grant))


def prepare_generation(root, private):
    root, private = Path(root).resolve(), Path(private).resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    old_h2 = read(private/'recovered/H2/result.json')
    old_h4 = read(private/'recovered/H4/plan_v1.json')
    recovery = read(private/'recovery_manifest.json')
    provider = next(r for r in recovery if 'provider_SOURCE' in r)
    for system in ('H2', 'H4'):
        calls = (copy.deepcopy(old_h2['generation_policy']['calls']) if system == 'H2' else
                 dict(scf_kernel=1, integral_build=1, df_decomposition=1,
                      sector_matrix_columns=36, independent_occupation_columns=36,
                      solver_eigh=1, signal=0, sampling=0, compile=0, shots=0))
        recipe = private/'original_generation_source'/('h2_tol_only_input.py' if system == 'H2' else 'h4_tol_only_input.py')
        recipe_commit = subprocess.check_output(['git', 'rev-parse',
            ('37b1cf72' if system == 'H2' else 'dfef38fc')+'^{commit}'], cwd=root, text=True).strip()
        original = subprocess.check_output(['git', 'show', recipe_commit+':scripts/'+recipe.name], cwd=root)
        require(original == recipe.read_bytes(), 'ORIGINAL_GENERATION_RECIPE_BYTES')
        target = private/'generation'/system
        plan = dict(kind='NEW_MIGRATION_GENERATION_PACK_NOT_OLD_PACK_RESTORATION', SOURCE=source,
            system=system, scientific_policy=(old_h2['generation_policy'] if system == 'H2' else old_h4['F']),
            call_budget=calls, recipe=str(recipe), recipe_sha256=sha(recipe), recipe_commit=recipe_commit,
            provider=provider, output=str(target/'run01'),
            source_hashes={p: sha(root/p) for p in
                ('src/trotterlib/h246_migration.py', 'scripts/run_h246_migration.py')},
            provider_hashes={str(p.relative_to(Path(provider['provider_root']))):sha(p)
                             for p in sorted(Path(provider['provider_root']).rglob('*.py'))},
            expected_rank=3 if system == 'H2' else 10,
            old_snapshot_sha256=(old_h2['sha256'] if system == 'H2' else
                '12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96'),
            equivalence_energy_tolerance=1e-9, retry=False)
        grant = dict(valid=True, approved_by_user=True, retry=False, SOURCE=source,
            plan_digest=digest(plan), call_budget=calls, grant_id=system+'_MIGRATION_GENERATION_'+digest(plan))
        save(target/'plan.json', plan)
        save(target/'approval_receipt.json', dict(role='user', execution_authorized=True,
            scope='Regenerate missing H2/H4 using the original fixed algorithms and scientific conditions; validate, then continue to fixed independent confirmation without intermediate reapproval.',
            SOURCE=source, plan_digest=digest(plan), call_budget=calls))
        save(target/'grant.json', grant)
    return dict(SOURCE=source, systems=['H2', 'H4'], H6='RECOVERED_ORIGINAL_NO_REGENERATION')


def verify_generation(root, pack, grant):
    require(grant.get('approved_by_user') is True and grant.get('valid') is True,
            'EXPLICIT_MIGRATION_APPROVAL_REQUIRED')
    require(grant.get('plan_digest') == digest(pack) and grant.get('call_budget') == pack['call_budget']
            and grant.get('SOURCE') == pack['SOURCE'] and grant.get('retry') is False, 'EXACT_NEW_GENERATION_GRANT')
    root = Path(root)
    for rel, expected in pack['source_hashes'].items():
        blob = subprocess.check_output(['git', 'show', pack['SOURCE']+':'+rel], cwd=root)
        require(sha(root/rel) == expected == hashlib.sha256(blob).hexdigest(), 'GENERATION_SOURCE_CHANGED:'+rel)
    require(sha(pack['recipe']) == pack['recipe_sha256'], 'ORIGINAL_RECIPE_CHANGED')
    for rel, expected in pack['provider_hashes'].items():
        require(sha(Path(pack['provider']['provider_root'])/rel) == expected, 'PINNED_PROVIDER_CHANGED:'+rel)


def run_generation(root, private, system):
    import sys
    root, private = Path(root).resolve(), Path(private).resolve()
    require(system in ('H2', 'H4'), 'ONLY_MISSING_INPUT_SYSTEMS')
    base = private/'generation'/system
    plan, grant = read(base/'plan.json'), read(base/'grant.json')
    verify_generation(root, plan, grant)
    claim(base/'CONSUMED.json', grant)
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    (out/'scratch').mkdir()
    # Nonrefundable complete-stage reservations precede every scientific call.
    for kind, count in plan['call_budget'].items():
        if count:
            save(out/'ledger'/f'{kind}.json', dict(state='RESERVED_NONREFUNDABLE',
                kind=kind, count=count, SOURCE=plan['SOURCE'], grant_id=grant['grant_id']))
    sys.path.insert(0, str(Path(plan['provider']['provider_root'])/'src'))
    used = {}

    def reserve(unused_output, name, count=1):
        used[name] = used.get(name, 0)+count
        require(used[name] <= plan['call_budget'][name], 'GENERATION_CALL_BUDGET:'+name)

    def verify(*args, **kwargs):
        verify_generation(root, plan, grant)

    policy = plan['scientific_policy']
    bridge = SimpleNamespace(require=require, save=save, read=read, sha=sha, reserve=reserve,
        actual_calls=lambda unused: dict(used), verify=verify,
        SCOPE=policy.get('scope', 'H4_FOLLOWUP_RARE_TIME_DF_20261018'),
        plan=lambda: policy if system == 'H2' else {'F': policy})
    tree = ast.parse(Path(plan['recipe']).read_text())
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.Import) and any(alias.asname == 'c' for alias in node.names))]
    namespace = dict(c=bridge, __name__='immutable_original_generation_recipe')
    exec(compile(tree, plan['recipe'], 'exec'), namespace)
    try:
        result = namespace['generate'](dict(SOURCE=plan['SOURCE']), grant, out)
        require(result['rank'] == plan['expected_rank'], 'FIXED_DF_RANK')
        if system == 'H2':
            old = read(private/'recovered/H2/result.json')
            require(used == {k:v for k,v in plan['call_budget'].items() if v}, 'EXACT_H2_GENERATION_CALLS')
        else:
            import gzip
            old = json.loads(gzip.decompress((private/'recovered/H4/F_result.json.gz').read_bytes()))
        energy_error = abs(result['state_energy']-old['state_energy'])
        require(energy_error < plan['equivalence_energy_tolerance'], 'OLD_GROUND_ENERGY_COMPARISON')
        comparison = dict(status='INPUT_GENERATED_OLD_ENERGY_AND_RANK_PASS_SIGNAL_GATE_PENDING',
            SOURCE=plan['SOURCE'], old_snapshot_sha256=plan['old_snapshot_sha256'],
            new_snapshot_sha256=result['sha256'], same_snapshot_bytes=result['sha256']==plan['old_snapshot_sha256'],
            original_restored=False, regenerated=True,
            same_hamiltonian_fingerprint=result['hamiltonian_hash']==old['hamiltonian_hash'],
            same_state_fingerprint=result['state_hash']==old['state_hash'],
            old_ground_energy=old['state_energy'], new_ground_energy=result['state_energy'],
            ground_energy_abs_error=energy_error, rank=result['rank'],
            array_hash_comparisons={k:v['sha256']==old['arrays'][k]['sha256'] for k,v in result['arrays'].items()},
            input_scientific_equivalence='PENDING_FIXED_SIGNAL_AND_TAIL_CHECKS',
            call_budget=plan['call_budget'], exact_before_call_counters=used if system=='H2' else None,
            H4_reservation_accounting='FIXED_HASHED_RECIPE_FULL_STAGE_RESERVATIONS; successful known loops 36 sector + 36 oracle columns',
            grant_id=grant['grant_id'], retry=False)
        verify_generation(root, plan, grant)
        save(out/'comparison.json', comparison)
        return comparison
    except BaseException as exc:
        save(out/'STOP.json', dict(status='STOP', reason=str(exc), retry=False,
            grant_id=grant['grant_id'], before_call_counters=used))
        raise
