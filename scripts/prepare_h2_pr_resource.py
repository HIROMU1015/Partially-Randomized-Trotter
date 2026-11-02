"""Static H2 F seal: copy fixed source, audit metadata, freeze runtime, no science."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import h2_pr_resource_contract as c

BASE = 'd728d088c010089c6b672eef0521261791b2d209'
PROTOCOL = '9036d8a87ad1c38e83b9ff654b412dde7041e9d3'
OLD_CONFIG = Path('/home/AbeHiromu/projects/h4-followup-validation/20261018/private/config_FP.json')
PREPARATION = 'phase0_preparation01'


def blob(source, path):
    return subprocess.check_output(['git', 'show', source + ':' + path], cwd=c.ROOT)


def prepare(source):
    source = subprocess.check_output(['git', 'rev-parse', source + '^{commit}'], cwd=c.ROOT, text=True).strip()
    private, public = c.PRIVATE / PREPARATION, c.PUBLIC / PREPARATION
    c.require(not private.exists() and not public.exists(), 'FRESH_PREPARATION_NAMESPACE')
    source_hashes = {name: c.sha(c.ROOT / name) for name in c.SOURCE_FILES}
    for name, expected in source_hashes.items():
        c.require(hashlib.sha256(blob(source, name)).hexdigest() == expected, 'COMMITTED_F_SOURCE_REQUIRED:' + name)
    private.mkdir(); public.mkdir()
    # Immutable Git bytes, never a link to H4's working directory.
    provider = private / 'provider'; provider.mkdir()
    archive = subprocess.check_output(['git', 'archive', c.PROVIDER_SOURCE, 'src'], cwd=c.ROOT)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        for entry in tree.getmembers():
            if entry.isdir(): continue
            c.require(entry.isfile(), 'REGULAR_PROVIDER_FILES_ONLY')
            target = provider / entry.name
            c.require(target.resolve().is_relative_to(provider.resolve()), 'PROVIDER_ARCHIVE_PATH')
            target.parent.mkdir(parents=True, exist_ok=True)
            data = tree.extractfile(entry).read()
            with target.open('xb') as out: out.write(data)
            hashes[entry.name] = hashlib.sha256(data).hexdigest()
    upstream = c.read(OLD_CONFIG)
    # -B plus H2 pycache_prefix avoids reading/writing the old runtime's .pyc.
    # Static archives, headers, test fixtures and typing stubs are not F runtime.
    paths = [p for p in upstream['runtime_file_hashes']
             if Path(p).suffix not in ('.pyc', '.a', '.o', '.h', '.pyi')
             and not any(part in ('__pycache__', 'tests', 'test', 'include') for part in Path(p).parts)]
    runtime = {p: c.sha(p) for p in paths}
    python = upstream['python']; runtime[str(Path(python).resolve())] = c.sha(python)
    manifest = private / 'runtime_hashes_PRIVATE.json'; c.save(manifest, runtime)
    recipe = private / 'F_generation_recipe.json'; c.save(recipe, c.plan())
    protocol = private / 'common_protocol_v1.md'
    protocol.write_bytes(blob(PROTOCOL, 'docs/research/track_a_pr_resource_evaluation_protocol_v1.md'))
    spec = dict(scope=c.SCOPE, stage='F', status='SEALED_PENDING_APPROVAL', SOURCE=source,
        plan=c.plan(), call_budget=c.CALLS, scientific_inputs=[], source_hashes=source_hashes,
        provider_root=str(provider), provider_SOURCE=c.PROVIDER_SOURCE, provider_hashes=hashes,
        runtime_manifest=str(manifest), runtime_manifest_sha256=c.sha(manifest),
        python=python, python_sha256=c.sha(python), attempt=str(c.PRIVATE / 'attempts/F_run01'))
    c.save(private / 'F_spec_SEALED_PENDING_APPROVAL.json', spec)
    invalid = dict(scope=c.SCOPE, stage='F', SOURCE=source, spec_digest=c.digest(spec),
        approved_by_user=False, runtime_authorization=False, one_shot=True,
        automatic_retry=False, next_stage_authorized=False, approval_receipt=None)
    c.save(public / 'F_grant_INVALID.json', invalid)
    inventory = []
    # Saved metadata only. No phase/shot/cost/statistical fields are recomputed.
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, 'artifacts'],
                                    cwd=c.ROOT, text=True).splitlines()
    h2_names = [n for n in names if 'h2' in n.lower()]
    for name in h2_names:
        data = blob(BASE, name)
        row = dict(path=name, fixed_commit=BASE, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                   reuse='HISTORICAL_CONTEXT_ONLY_NOT_CURRENT_INPUT')
        if name.startswith('artifacts/pf_c_system_size_validation/') and name.endswith('.json'):
            saved = json.loads(data)
            if 'hamiltonian' in saved:
                ham = saved['hamiltonian']
                row.update(saved_input_metadata=ham['metadata'], saved_n_qubits=ham['n_qubits'],
                    saved_sector_dimension=ham['sector_dimension'],
                    mismatch_reasons=['CONFIG_SELECTED_RANK_NOT_TOL_ONLY', 'TOTAL_NUMBER_SECTOR_NOT_NALPHA_NBETA_SPIN_SECTOR'],
                    provenance=saved.get('provenance'))
        inventory.append(row)
    c.save(public / 'existing_H2_inventory.json', dict(search_scope='FIXED_BASE_TRACKED_H2_ARTIFACT_FILENAMES_AND_RESOURCE_METADATA',
        records=inventory, identical_current_tol_only_input_found=False,
        private_or_untracked_H2_directories_searched=False,
        limitation='No claim of absence outside this fixed tracked evidence scope; no broad shared-directory scan.'))
    c.save(public / 'F_contract_summary.json', dict(status=spec['status'], SOURCE=source,
        private_spec_path=str(private / 'F_spec_SEALED_PENDING_APPROVAL.json'), spec_digest=c.digest(spec),
        plan=c.plan(), source_hashes=source_hashes, provider_SOURCE=c.PROVIDER_SOURCE,
        provider_files=len(hashes), runtime_files=len(runtime),
        provider_manifest_digest=c.digest(hashes), runtime_manifest_sha256=c.sha(manifest),
        python=python, python_sha256=spec['python_sha256'],
        historical_runtime_hash_mismatches=sum(upstream['runtime_file_hashes'].get(p, h) != h for p,h in runtime.items()),
        mutable_pyc_excluded=True, H2_pycache_prefix=str(c.PRIVATE / 'runtime_pycache'),
        recipe_sha256=c.sha(recipe), protocol_commit=PROTOCOL, protocol_sha256=c.sha(protocol),
        actual_scientific_calls={k: 0 for k in c.CALLS}, valid_scientific_grant_created=False,
        output_or_one_shot_created=False, automatic_next_stage=False, independent_scientific_review=False))
    for stage in ['FP', 'D', 'A', 'S_STATS', 'P']:
        c.save(public / (stage + '_contract_DRAFT.json'), dict(stage=stage,
            status='STATIC_DEPENDENCY_DRAFT_NOT_SEALED', approved_by_user=False,
            SOURCE=None, rank=None, candidates=None, n=None, seeds=None, scientific_call_budget=None,
            pending=['ACCEPTED_F_INPUT_AND_ACTUAL_RANK', 'H2_SPECIFIC_SOURCE_INPUT_AND_BUDGET',
                     'SEPARATE_EXPLICIT_STAGE_APPROVAL'],
            fixed_primary_epsilons=[.05,.0005] if stage in ('S_STATS','P') else None,
            mandatory_STOP=True, automatic_retry=False, P_science_authorized=False))
    c.save(public / 'session_adoption.json', dict(user_message='これがその新規チャットです',
        adopted_chat_role='H2_DEDICATED_INDEPENDENT_OF_ONGOING_H4',
        authorizes_static_H2_preparation=True, authorizes_scientific_execution=False,
        original_isolation_commit='f9d2e186887007ba773c3d2fbfa4adb9ff9d7b4d'))
    return dict(SOURCE=source, spec_digest=c.digest(spec), provider_files=len(hashes),
                runtime_files=len(runtime), status='SEALED_PENDING_F_APPROVAL', public=str(public))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True)
    print(json.dumps(prepare(parser.parse_args().source), ensure_ascii=False))
