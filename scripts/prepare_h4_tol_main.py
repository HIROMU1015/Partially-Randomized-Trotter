"""Phase0 byte/provenance audit and contract preparation. Never imports science.

Saved numerical fields are copied verbatim. No NPZ decoding, raw cost aggregation,
new shots, rankings, power/sample-size calculations, or quantum work is performed.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import h4_tol_main_contract as c

OLD = Path('/home/AbeHiromu/projects/h4-followup-validation/20261018/private')
BATCH = Path('/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/private/sealed03')
COMMITS = dict(F_SOURCE='dfef38fc8107dd76f5d71dcdeca31f8d3dc39f64',
    FP_SOURCE='64e0fc91cd273cff02ae8cffb729ee8e40c302a9', FP_RESULT='97d25cc431516d720ce91c7cb4f148e061d8ea34',
    D_RESULT='4e8ad1936363266d1b55bb689c36846e6c3dfcdb', P_RESULT='8a18b277663bf2f8998f79e9b8c0a9e61d2b184c',
    A_COST='fc6bd039dc7171d5674ac04f0b3ad427f1b6495b', A_TOTAL='64c85f7a833b5950ef2ccae623708b73081edcd6',
    H8_P_SOURCE='745a674546f2cb6258526fbfde1d9a00601689f8', H8_P_RESULT='e6eea5387a6e84d1e39a9adb3b61975acc74e200')
PUBLIC = c.ROOT/'artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19'


class Collector:
    def __init__(self, private):
        self.private = Path(private); self.private.mkdir(parents=True, exist_ok=True)
        self.store = self.private/'store'; self.store.mkdir(exist_ok=True)
        self.rows = []; self.by_path = {}

    def keep(self, role, path, expected=None):
        path = Path(path).resolve(); actual = c.sha(path)
        c.require(expected is None or actual == expected, 'EXISTING_BYTES_CHANGED:'+role)
        target = self.store/actual
        if not target.exists(): shutil.copyfile(path, target)
        c.require(c.sha(target) == actual, 'SNAPSHOT_COPY')
        self.rows.append(dict(role=role, path=str(path), copy=str(target), sha256=actual, bytes=path.stat().st_size))
        self.by_path[str(path)] = str(target)
        return str(target)

    def git(self, source, name, role):
        blob = subprocess.check_output(['git', 'show', source+':'+name], cwd=c.ROOT)
        target = self.store/hashlib.sha256(blob).hexdigest()
        if not target.exists(): target.write_bytes(blob)
        self.rows.append(dict(role=role, SOURCE=source, repo_path=name, path=str(target),
            copy=str(target), sha256=c.sha(target), bytes=len(blob)))
        return str(target)

    def walk_run(self, directory, role):
        # Only the two known completed F/FP output trees; scratch caches are excluded.
        directory = Path(directory)
        for item in sorted(directory.iterdir()):
            if item.name == 'scratch': continue
            if item.is_file(): self.keep(role+':'+item.name, item)
            elif item.is_dir(): self.walk_run(item, role+':'+item.name)


def inventory(private):
    collect = Collector(private); p = collect.private
    config = c.read(OLD/'config_FP.json')
    config_copy = collect.keep('original_FP_config', OLD/'config_FP.json')
    for stage in ('F', 'FP'):
        run = OLD/(stage+'_run01'); end = c.read(run/'native_exit.json')
        term = c.read(run/'output/terminal.json'); gate = c.read(run/'output/final_gate.json')
        c.require(end['exit_code'] == 0 and end['wait4_complete'] and end['same_parent_reaped']
            and term['status'] == 'COMPLETE_MANDATORY_STOP' and gate['status'] == 'PASS', 'F_FP_NATIVE_COMPLETION')
        c.require(end['SOURCE'] == COMMITS[stage+'_SOURCE'], 'FIXED_F_FP_SOURCE')
        collect.keep(stage+'_native_exit', run/'native_exit.json')
        collect.keep(stage+'_grant', run/'grant.json')
        collect.walk_run(run/'output', stage+'_output')
        receipt = c.read(c.ROOT/'artifacts/resource_applicability/track_a_h4_followup/2026-10-18'/(stage+'_completion_receipt.json'))
        c.require(c.sha(run/'output/result.json') == receipt['result_uncompressed_sha256'], 'PUBLIC_PRIVATE_RESULT_IDENTITY')
        zipped = c.ROOT/'artifacts/resource_applicability/track_a_h4_followup/2026-10-18'/(stage+'_result.json.gz')
        c.require(c.sha(zipped) == receipt['result_gzip_sha256'] and
            hashlib.sha256(gzip.decompress(zipped.read_bytes())).hexdigest() == receipt['result_uncompressed_sha256'], 'GZIP_RESULT_IDENTITY')
        for public_path in (zipped, zipped.parent/(stage+'_completion_receipt.json')):
            original=collect.git(COMMITS['FP_RESULT'],str(public_path.relative_to(c.ROOT)),stage+'_fixed_public_result')
            c.require(c.sha(original)==c.sha(public_path),'FIXED_RESULT_COMMIT_IDENTITY')
    f = c.read(OLD/'F_run01/output/tol_only_receipt.json')
    c.require(f['rank'] == 10 and f['sha256'] == c.BINDING['snapshot']
        and f['hamiltonian_hash'] == c.BINDING['hamiltonian'] and f['state_hash'] == c.BINDING['state'], 'RANK10_FIXED_IDENTITY')
    c.require(f['hamiltonian_metadata']['final_rank_supplied'] is False and
        f['hamiltonian_metadata']['decomposition_kwargs'] == dict(truncation_threshold=1e-8), 'TOL_ONLY_DF_POLICY')
    c.require(c.sha(config['mapping']['tol_snapshot']) == f['sha256'], 'FP_INPUT_IDENTICAL_TO_F')
    own = deepcopy(config)
    for row in config['inputs']: collect.keep('FP_input:'+row['role'], row['path'], row['sha256'])
    own['inputs'] = [dict(row, path=collect.by_path[str(Path(row['path']).resolve())]) for row in config['inputs']]
    own['mapping'] = {k: collect.by_path.get(str(Path(v).resolve()), v) for k, v in config['mapping'].items()}
    own['font_seed']['path'] = collect.by_path[str(Path(config['font_seed']['path']).resolve())]
    provider = p/'provider'; provider.mkdir(exist_ok=True)
    for name, expected in config['provider_hashes'].items():
        origin = Path(config['provider_root'])/name
        collect.keep('FP_provider:'+name, origin, expected)
        target = provider/name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copyfile(origin, target)
        c.require(c.sha(target) == expected, 'OWN_PROVIDER_SNAPSHOT')
    own['provider_root'] = str(provider.resolve())
    for name, expected in config['source_hashes'].items():
        c.require(c.sha(c.ROOT/name) == expected, 'UNCHANGED_REUSED_FP_SOURCE:'+name)
        stored = collect.git(COMMITS['FP_SOURCE'], name, 'reused_FP_SOURCE:'+name)
        c.require(c.sha(stored) == expected, 'REUSED_FP_GIT_BLOB')
    for name, expected in config['runtime_file_hashes'].items():
        c.require(c.sha(name) == expected, 'SAVED_RUNTIME_CHANGED')
    c.save(p/'rank10_native_config_PRIVATE.json', own)
    fp = c.read(OLD/'FP_run01/output/result.json')['new']
    signals = c.read(OLD/'FP_run01/output/signals.json')['signals']
    c.require({r['cell']['id'] for r in signals} == set(c.FP_SIGNALS_D+[c.FP_PR]), 'EXACT_SEVEN_FP_SIGNAL_RECORDS')
    c.require(len(fp['measurements']) == 36 and len(fp['signals']) == 5, 'FP_FIXED_FIVE_CELL_COVERAGE')
    evidence = dict(binding=c.BINDING, SOURCE=COMMITS['FP_SOURCE'], result_sha256=c.sha(OLD/'FP_run01/output/result.json'),
        signals=signals, saved_rows=fp['rows'], measurements=fp['measurements'],
        preparation_identities={r['cell']['id']: r['hash'] for r in c.read(OLD/'FP_run01/output/preparations.json')['rows']},
        statistics_recomputed=False, scientific_numbers_transferred_from_rank12=False)
    by_id = {x['id']: x for x in c.registrations()['D']+c.registrations()['A']}
    for row in fp['measurements']:
        c.reuse_measurement(row, c.BINDING, by_id[row['cell_id']],
            accepted_result_sha256=evidence['result_sha256'], source=COMMITS['FP_SOURCE'])
    c.save(p/'FP_evidence_PRIVATE.json', evidence)
    collect.keep('saved_rank10_reference', OLD/'FP_run01/output/reference.json')
    c.save(p/'native_input_binding_PRIVATE.json', dict(native_config=str(p/'rank10_native_config_PRIVATE.json'),
        FP_evidence=str(p/'FP_evidence_PRIVATE.json'), saved_reference=collect.by_path[str((OLD/'FP_run01/output/reference.json').resolve())],
        font_seed=own['font_seed'], runtime_hashes=config['runtime_file_hashes'],
        original_FP_config_sha256=c.sha(config_copy)))
    for label, source in COMMITS.items():
        subprocess.check_output(['git', 'cat-file', '-e', source+'^{commit}'], cwd=c.ROOT)
    return collect, own, f


def policies(collect, f):
    evidence = {}; p = collect.private
    for system in ('H6', 'H8'):
        file = BATCH/'contracts'/(system+'_A_COST')/'technical_seal.json'
        collect.keep(system+'_fixed_A_seal', file)
        seal = c.read(file); collect.keep(system+'_fixed_A_payload', seal['payload'])
        adapter = c.read(seal['payload'])['native_adapter']
        collect.keep(system+'_native_config', adapter['native_config'])
        config = c.read(adapter['native_config'])
        snap = c.read(config['mapping']['snapshot_receipt.json'])
        df = c.read(config['mapping']['df_receipt.json'])
        for role in ('snapshot_receipt.json', 'df_receipt.json', 'projection_receipt.json'):
            if role in config['mapping']: collect.keep(system+':'+role, config['mapping'][role])
        md = snap['metadata']; hm = md['hamiltonian_metadata']
        evidence[system] = dict(binding=adapter['binding'], state_policy=md['state_policy'],
            sector=md['sector'], global_phase_policy=md['global_phase_policy'],
            df_policy={k: hm.get(k) for k in ('coefficient_order', 'df_tol_requested', 'df_rank_actual',
                'final_rank_supplied', 'subsequent_coefficient_cutoff', 'input_policy')},
            projection_policy=df['projection_receipt']['policy'], source_ancestors=hm['parents'],
            numerical_values_copied_not_recalculated=True)
    parent = evidence['H6']['source_ancestors']['integrals']
    name = parent['input_path'].replace('integrals.npz', 'integral_receipt.json')
    receipt = collect.git(parent['old_result_commit'], name, 'H6_original_integral_receipt')
    c.require(c.sha(receipt) == parent['receipt_sha256'], 'H6_ORIGINAL_INTEGRAL_RECEIPT')
    h6_integral = c.read(receipt)
    h8_parent = evidence['H8']['source_ancestors']
    h8_file = Path(h8_parent['directory'])/'integral_receipt.json'
    collect.keep('H8_original_integral_receipt', h8_file, h8_parent['file_hashes']['integral_receipt.json'])
    h8_integral = c.read(h8_file)
    for name in ('raw_df_receipt.json', 'worker_terminal.json'):
        collect.keep('H8_original_'+name, Path(h8_parent['directory'])/name, h8_parent['file_hashes'][name])
    c.require(h6_integral['convention'] == h8_integral['convention'], 'SAVED_H6_H8_INTEGRAL_CONVENTION')
    evidence['H4'] = dict(binding=c.BINDING, generation_source=COMMITS['F_SOURCE'],
        SCF=dict(scf='RHF', unit='Angstrom', symmetry=False, conv_tol=1e-9, max_cycle=50,
            orbital_order='canonical MO returned by same openfermionpyscf helpers; no post-sort'),
        spin_mapping='spinorb_from_spatial even-alpha/odd-beta, two-body multiplied by 1/2',
        df_policy=f['hamiltonian_metadata'], projection=f['projection'],
        state_policy=dict(solver='scipy.linalg.eigh on complete 36x36 sector', selection='lowest eigenvector',
            phase='largest-amplitude canonical phase', normalization='eigh normalized vector; no post-save renormalization',
            residual_gate=1e-10, ground_state_certified=False), numerical_values_copied_not_recalculated=True)
    differences = [
        dict(item='SCF/integrals', status='ALIGNED_SOURCE_AND_SAVED_SETTINGS',
            note='RHF, 1e-9/50, no symmetry, canonical MO helper, spin mapping and 1/2 convention agree'),
        dict(item='DF decomposition', status='ALIGNED_POLICY',
            note='OpenFermion low_rank_two_body_decomposition; threshold 1e-8; no final rank; retain generation order and signed coefficients; cutoff0'),
        dict(item='weighted Hermitian projection', status='ALIGNED_DEFINITION',
            note='same weighted triangle inequality; full-Fock N=8/12/16; engineering decision 9.9e-11; no interval certificate'),
        dict(item='saved-state solver', status='DECLARED_NUMERICAL_IMPLEMENTATION_DIFFERENCE',
            note='H4 dense eigh sector36; H6/H8 eigsh SA with HF initial state; same lowest-sector-state target and canonical phase; preserve frozen states'),
        dict(item='normalization/residual gates', status='DECLARED_NUMERICAL_IMPLEMENTATION_DIFFERENCE',
            note='H4 uses normalized eigh vector/residual1e-10; H6/H8 one binary64 normalization and absolute+relative residual gate; no re-solve authorized'),
        dict(item='compiler profile', status='DIFFERENT_LINEAGE_IDENTIFIERS',
            note='same declared Qiskit1.3.0/rz-sx-x-cx/opt1/seed17; H8 historical profile identifier differs; old binary outputs not certified identical'),
        dict(item='candidate search and cost evidence', status='DIFFERENT_REGISTERED_DOMAINS',
            note='H4 proposed D190/A50; H6/H8 A16 and strengthened PF domains differ and include model evidence; no pure size-scaling attribution'),
        dict(item='new common-protocol input regeneration', status='NOT_AUTHORIZED_OR_NECESSARY_FOR_SAVED_INPUT_REUSE',
            note='Any decision to require identical solver implementations or change saved inputs is a separate scientific approval; no silent normalization/projection change')]
    c.save(p/'policy_evidence_PRIVATE.json', evidence)
    return evidence, differences


def preserve_fixed_records(collect):
    base = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees')
    records = [
        ('rank12_D', base/'h4-d-preparation-fix-20261019/.h4_dp_phase0_private/D_run02_20261019'),
        ('rank12_P', base/'h4-p-seal-20261019/.h4_p_seal_private/P_run01_20261019')]
    names = {'rank12_D': ['spec_SEALED_PRIVATE.json', 'grant_PRIVATE.json', 'native_driver_exit.json',
        'ONE_SHOT.json', 'output/result.json', 'output/terminal.json'],
        'rank12_P': ['spec_SEALED_PRIVATE.json', 'grant_PRIVATE.json', 'native_driver_exit.json',
        'ONE_SHOT.json', 'output/result.json', 'output/terminal.json']}
    for role, directory in records:
        for name in names[role]:
            file = directory/name
            # Older D filenames are resolved from its own directory only.
            if not file.exists() and role == 'rank12_D':
                choices = {'spec_SEALED_PRIVATE.json': 'spec_SEALED.json', 'grant_PRIVATE.json': 'grant.json'}
                file = directory/choices.get(name, name)
            c.require(file.is_file(), 'FIXED_PRIOR_RECORD_NOT_FOUND:'+str(file))
            collect.keep(role+':'+name, file)
    for source in (COMMITS['D_RESULT'], COMMITS['P_RESULT']):
        for name in ('scripts/h4_dp_contract.py', 'scripts/h4_dp_science.py', 'scripts/run_h4_dp_verification.py'):
            collect.git(source, name, 'rank12_implementation_reference')


def materialize(private, source):
    source=subprocess.check_output(['git','rev-parse',source+'^{commit}'],cwd=c.ROOT,text=True).strip()
    p = Path(private).resolve(); collect, own, f = inventory(p)
    evidence, differences = policies(collect, f)
    preserve_fixed_records(collect)
    policy_sources = c.read(p/'policy_source_manifest_PRIVATE.json')
    for row in policy_sources:
        collect.keep('fixed_policy_SOURCE:'+row['repo_path'], row['path'], row['sha256'])
    # These are project-private source snapshots, not installed/shared packages.
    a_provenance = c.read(p/'A_kernel_provenance_PRIVATE.json')
    for name, expected in a_provenance['source_hashes'].items():
        collect.keep('frozen_A_kernel:'+name, p/'A_kernel'/name, expected)
    for name in ('rank10_native_config_PRIVATE.json', 'FP_evidence_PRIVATE.json',
                 'native_input_binding_PRIVATE.json', 'policy_evidence_PRIVATE.json', 'Phase0_authorization_PRIVATE.json'):
        collect.keep('own_binding:'+name, p/name)
    source_hashes = c.source_closure()
    for name, expected in source_hashes.items():
        blob = subprocess.check_output(['git', 'show', source+':'+name], cwd=c.ROOT)
        c.require(hashlib.sha256(blob).hexdigest() == expected, 'ACTUAL_SOURCE_COMMIT_REQUIRED:'+name)
    providers = {str(p/'provider'/name): expected for name, expected in own['provider_hashes'].items()}
    providers.update({str(p/'A_kernel'/name): expected for name, expected in a_provenance['source_hashes'].items()})
    binding = c.read(p/'native_input_binding_PRIVATE.json')
    c.save(p/'preservation_manifest_PRIVATE.json', collect.rows)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    public_contracts = {}
    for stage in ('D', 'A', 'S_STATS', 'P'):
        pending = ['SEPARATE_EXPLICIT_EXECUTION_APPROVAL', 'FRESH_RESOURCE_ADMISSION']
        if stage == 'A': pending += ['PENDING_D_RESULT', 'COMPLETE_NEW_SEED_INVENTORY']
        elif stage in ('P', 'S_STATS'): pending += ['PENDING_D_A_RESULT', 'SEPARATE_FINAL_NUMERICAL_CONTRACT']
        contract = dict(stage=stage, status='TECHNICALLY_BOUND_PENDING_APPROVAL' if stage == 'D'
            else 'PENDING_D_RESULT_AND_SEED_INVENTORY' if stage == 'A' else 'PENDING_D_A_RESULT',
            SOURCE=source if stage in ('D', 'A') else None, implementation_SOURCE=source,
            binding=c.BINDING, plan=c.plan(stage), pending=pending,
            source_hashes=source_hashes, inputs=[dict(role=str(i)+':'+r['role'],
                path=r['path'] if r['role'].startswith('own_binding:') else r['copy'], sha256=r['sha256'])
                for i, r in enumerate(collect.rows)], runtime_hashes=own['runtime_file_hashes'],
            provider_hashes=providers, provider_roots=[str(p/'provider'),str(p/'A_kernel')],
            python=own['python'], python_sha256=c.sha(own['python']),
            native_config=binding['native_config'], FP_evidence=binding['FP_evidence'],
            saved_reference=binding['saved_reference'], font_seed=binding['font_seed'],
            import_paths=[str(p/'A_kernel/scripts'), str(p/'provider/src')],
            accepted_D=None, seed_inventory_complete=False, seed_registry=None,
            old_seed_exclusions=None, seed_inventory_receipt=None,
            fresh_resources=dict(admitted=False, receipt=None, sha256=None), worker_cpus=[],
            output=str(p/'attempts'/(stage+'_run01')/'output'), lock=str(p/'attempts'/(stage+'_run01')/'ONE_SHOT.json'),
            approved_by_user=False, runtime_authorization=False, next_stage_authorized=False,
            automatic_retry=False, actual_scientific_calls=0)
        c.save(p/(stage+'_contract_DRAFT_PRIVATE.json'), contract)
        grant = c.invalid_grant(stage, contract)
        c.save(p/(stage+'_grant_INVALID_PRIVATE.json'), grant)
        projection = {k: deepcopy(v) for k, v in contract.items() if k in (
            'stage', 'status', 'SOURCE', 'implementation_SOURCE', 'binding', 'plan', 'pending',
            'approved_by_user', 'runtime_authorization', 'next_stage_authorized', 'automatic_retry', 'actual_scientific_calls')}
        projection.update(contract_digest=c.digest(contract), source_hashes=source_hashes,
            input_count=len(contract['inputs']), provider_file_count=len(providers), runtime_file_count=len(contract['runtime_hashes']),
            output_exists=Path(contract['output']).exists(), one_shot_exists=Path(contract['lock']).exists(),
            resource_assignment='FRESH_AT_SEPARATE_EXECUTION_APPROVAL', scientific_execution_authorized=False)
        c.save(PUBLIC/(stage+'_contract_DRAFT.json'), projection)
        c.save(PUBLIC/(stage+'_grant_INVALID.json'), grant)
        public_contracts[stage] = projection['contract_digest']
    fp = c.read(p/'FP_evidence_PRIVATE.json')
    c.save(PUBLIC/'FP_reuse_evidence.json', dict(binding=c.BINDING, SOURCE=fp['SOURCE'], result_sha256=fp['result_sha256'],
        signal_records=[r['cell']['id'] for r in fp['signals']],
        cohort='SAVED_FP_PANEL_N32_ANCHOR; not new A n8 or independent P',
        saved_rows=[r for r in fp['saved_rows'] if r['bias_mode'] == 'USUAL_SIGNED' and r['allowance_factor'] == 1.],
        measured_axes=[dict(cell_id=r['cell_id'], replica=r['replica'], axes=r['rows']) for r in fp['measurements']],
        reused_deterministic_axes=8, reused_PR_axes=64, same_input_state_compiler_and_fixed_native_SOURCE=True,
        no_new_fingerprint_construction=True, old_FP_raw_statistics_recomputed=False))
    c.save(PUBLIC/'F_saved_input_metadata.json', {k: f[k] for k in ('rank', 'sha256', 'arrays', 'hamiltonian_hash',
        'state_hash', 'hamiltonian_metadata', 'state_energy', 'state_residual', 'projection', 'actual_calls')})
    c.save(PUBLIC/'policy_comparison.json', dict(differences=differences,
        H6_H8_integral_convention='same saved convention and original source helpers',
        state_solver_policy_identical=False, saved_inputs_reused=True,
        numerical_policy_evidence_sha256=c.sha(p/'policy_evidence_PRIVATE.json'),
        source_provenance=[{k: r[k] for k in ('SOURCE', 'repo_path', 'sha256')} for r in policy_sources],
        scientific_input_changes_performed=False, new_numerical_analysis=False))
    c.save(PUBLIC/'input_manifest.json', dict(binding=c.BINDING,
        records=[{k: row[k] for k in ('role', 'sha256', 'bytes')} for row in collect.rows],
        NPZ_decoded=False, private_input_paths_published=False,
        preservation_manifest_sha256=c.sha(p/'preservation_manifest_PRIVATE.json')))
    c.save(PUBLIC/'candidate_registrations.json', c.registrations())
    c.save(PUBLIC/'proposed_calls.json', c.budgets())
    c.save(PUBLIC/'SOURCE_provenance.json', dict(implementation_SOURCE=source, fixed_references=COMMITS,
        own_source_hashes=source_hashes, frozen_A_kernel_SOURCE=a_provenance['SOURCE'],
        frozen_A_kernel_files=len(a_provenance['source_hashes']), provider_SOURCE=own['provider_source'],
        reused_FP_sources_unchanged=True, Phase0_scientific_calls=0))
    c.save(PUBLIC/'preservation_receipt.json', dict(records=len(collect.rows), origin_and_copy_SHA_verified=True,
        shared_environment_modified=False, prior_grants_ledgers_markers_STOP_modified=False,
        unrelated_worktree_updates_are_gates=False, AI_subagents_used=False,
        new_scientific_grants=0, scientific_output_or_marker_created=False))
    c.save(PUBLIC/'phase0_status.json', dict(status='PHASE0_PREPARED_PENDING_ARTIFICIAL_TESTS',
        SOURCE=source, contracts=public_contracts, science_calls=0, actual_numerical_analyses=0,
        next_stage_authorized=False, automatic_execution=False, mandatory_STOP=True))
    return dict(records=len(collect.rows), own_source_files=len(source_hashes), provider_files=len(providers),
        runtime_files=len(own['runtime_file_hashes']), scientific_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private', type=Path, default=c.ROOT/'.h4_tol_main_private')
    p.add_argument('--source', required=True)
    args = p.parse_args()
    print(json.dumps(materialize(args.private, args.source)))


if __name__ == '__main__': main()
