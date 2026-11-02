"""Bind technical DRAFTs to a local SOURCE. Never grants scientific authority."""
import argparse
from pathlib import Path
import subprocess

import h4_dp_contract as c
import prepare_h4_dp_phase0 as p

FILES = ['scripts/h4_dp_contract.py', 'scripts/h4_dp_science.py',
         'scripts/run_h4_dp_verification.py', 'scripts/h4_dp_witness.py',
         'scripts/prepare_h4_dp_phase0.py', 'scripts/seal_h4_dp_phase0.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    c.require(len(args.source) == 40, 'FIXED_LOCAL_SOURCE')
    source_hashes = {name: c.sha(p.ROOT / name) for name in FILES}
    for name, h in source_hashes.items():
        raw = subprocess.check_output(['git', 'show', args.source + ':' + name], cwd=p.ROOT)
        c.require(__import__('hashlib').sha256(raw).hexdigest() == h, 'SOURCE_BLOB')
    baseline = c.read(p.PRIVATE / 'preservation_baseline.json')
    for row in baseline:
        c.require(c.sha(row['path']) == row['sha256'] and c.sha(row['copy']) == row['sha256'], 'OLD_BASELINE_CHANGED')
    original = c.read(p.NATIVE / 'config_SEALED_PRIVATE.json')
    native = dict(original, mapping=dict(original['mapping']))
    for row in baseline:
        if row['role'].startswith('native_input:'):
            role = row['role'].split(':', 1)[1]
            if role in native['mapping']:
                native['mapping'][role] = row['copy']
    c.save(p.PRIVATE / 'native_for_D.json', native)
    store = p.Store()
    a_spec = c.read(p.BATCH / 'contracts/H4_A_COST/technical_seal.json')
    payload = c.read(a_spec['payload'])
    store.file('old_seed_exclusions', payload['new_seed_registry']['old_seed_exclusions'])
    prep = store.file('old_A_preparation_receipt', Path(a_spec['output']) / 'preparation_receipt.json')
    c.save(p.PUBLIC / 'P_saved_preparation_identities.json', c.read(prep))
    # Retain a separate supplemental baseline; never reset the first observation.
    c.save(p.PRIVATE / 'preservation_supplement.json', store.private)
    c.save(p.PUBLIC / 'source_inventory_supplement.json', store.public)
    adapter = c.read(p.PRIVATE / 'A_native_adapter.json')
    old_a = c.read(adapter['native_config'])
    c.require(old_a['binding'] == original['binding'], 'A_INPUT_BINDING')
    a_native = dict(old_a, mapping=dict(native['mapping']))
    c.save(p.PRIVATE / 'native_for_P.json', a_native)
    adapter = dict(adapter, native_config=str(p.PRIVATE/'native_for_P.json'),
                   preparation_identities={k: v['preparation_hash'] for k, v in c.read(prep)['prefixes'].items()})
    c.save(p.PRIVATE / 'A_adapter_for_P.json', adapter)
    records = c.read(p.PUBLIC / 'saved_signal_catalog.json')
    by_id = {r['cell']['id']: r for r in records}
    tasks = c.read(p.PUBLIC / 'D_tasks.json')
    reg = c.read(p.PUBLIC / 'P_registration.json')
    raw_d = c.read(p.NATIVE / 'D_run01/output/result.json')
    previous = [dict(bias_mode=r['bias_mode'], epsilon=r['epsilon'], deterministic=r['deterministic'])
                for r in raw_d['comparisons'] if r['allowance_factor'] == 1. and 'deterministic' in r]
    c.require(len(previous) == 10, 'SAVED_SUBSET_COMPARISONS')
    c.save(p.PRIVATE / 'previous_subset_comparisons.json', previous)
    def row(role, path):
        return dict(role=role, path=str(path), sha256=c.sha(path))
    inputs = [row('snapshot_or_metadata:' + k, v) for k, v in native['mapping'].items()
              if k in ('snapshot', 'snapshot_receipt', 'df_receipt', 'saved_reference')]
    for name in ['saved_signal_catalog.json', 'saved_endpoints.json', 'saved_measurements.json', 'D_tasks.json']:
        inputs.append(row(name, p.PUBLIC / name))
    inputs.extend([row('native_config', p.PRIVATE/'native_for_D.json'),
                   row('previous_subset_comparisons', p.PRIVATE/'previous_subset_comparisons.json')])
    providers = {str(Path(original['provider_root'])/name): h for name, h in original['provider_hashes'].items()}
    common = dict(schema=c.SCHEMA, SOURCE=args.source, source_hashes=source_hashes,
        technical_sealed=True, status='DRAFT_UNAUTHORIZED', approved_by_user=False, runtime_authorization=False,
        runtime_hashes=original['runtime_file_hashes'], provider_hashes=providers,
        python=original['python'], python_sha256=c.sha(original['python']),
        workers=None, worker_count_status='FRESH_ADMISSION_AT_AUTHORIZED_SEAL', resource_caps=None,
        cpu_affinity_changes=False, transient_PSI_or_other_worktree_change_stops_science=False,
        one_shot=True, retry=False, next_stage_authorized=False)
    d = dict(common, stage='D', native_config=str(p.PRIVATE/'native_for_D.json'), inputs=inputs,
        registered_tasks=str(p.PUBLIC/'D_tasks.json'),
        cells=[by_id[x]['cell'] for x in tasks['compile_ids']],
        tasks=[dict(id=cid, cell=by_id[cid]['cell'], record=by_id[cid]) for cid in tasks['compile_ids']],
        signal_catalog=str(p.PUBLIC/'saved_signal_catalog.json'), saved_endpoints=str(p.PUBLIC/'saved_endpoints.json'),
        saved_measurements=str(p.PUBLIC/'saved_measurements.json'),
        previous_subset_comparisons=str(p.PRIVATE/'previous_subset_comparisons.json'),
        calls=dict(native_evolution=55, wrapper_compile=110, control_probe=220, deterministic_accounting_pass=1,
                   new_signal=0, sampling=0, SCF=0, DF=0, state=0, reference=0, shot_recalculation=0),
        output=str(p.PRIVATE/'D_future_run01/output'), lock=str(p.PRIVATE/'D_future_run01/ONE_SHOT.json'))
    p_inputs = [row('P_registration', p.PUBLIC/'P_registration.json'),
        row('P_signal_bindings', p.PUBLIC/'P_signal_bindings.json'),
        row('P_preparation_identities', p.PUBLIC/'P_saved_preparation_identities.json'),
        row('P_native_config', p.PRIVATE/'native_for_P.json'), row('P_native_adapter', p.PRIVATE/'A_adapter_for_P.json'),
        row('saved_preparation_receipt', prep)] + inputs[:4]
    prospective_p = dict(common, stage='P', native_config=str(p.PRIVATE/'native_for_P.json'),
        A_native_adapter=str(p.PRIVATE/'A_adapter_for_P.json'), cells=reg['candidates'], inputs=p_inputs,
        tasks=[], n_fixed=None, D_result_sha256=None, D_result_path=None, evaluation_endpoints=[],
        registration=str(p.PUBLIC/'P_registration.json'), seed_plan_digest=None, seed_exclusions=None,
        signal_bindings=str(p.PUBLIC/'P_signal_bindings.json'),
        saved_preparation_identities=str(p.PUBLIC/'P_saved_preparation_identities.json'),
        seed_collision_check_complete=False, stats_approval_receipt_sha256=None,
        pending=reg['P_pending'], frozen_stage_source='31ef62a12407010e9c1a07f9b37facedb8dff5fe',
        frozen_stage_closure=a_spec['provider_closure'],
        calls=dict(trajectory=None, native_evolution=None, occurrence=None, RTE_event=None,
                   wrapper_compile=None, control_probe=36, independent_event_oracle=9,
                   endpoint_accounting_pass=1, new_signal=0, shot_recalculation=0),
        output=str(p.PRIVATE/'P_future_run01/output'), lock=str(p.PRIVATE/'P_future_run01/ONE_SHOT.json'))
    for spec in [d, prospective_p]:
        stage = spec['stage']
        Path(spec['output']).parent.mkdir(parents=True, exist_ok=True)
        c.require(not Path(spec['output']).exists() and not Path(spec['lock']).exists(), 'UNUSED_FUTURE_RUN')
        c.save(p.PRIVATE/(stage+'_contract_DRAFT_PRIVATE.json'), spec)
        grant = dict(schema='INVALID_H4_DP_GRANT', stage=stage, SOURCE=args.source, spec_digest=c.digest(spec),
            approved_by_user=False, runtime_authorization=False, approval_receipt=None,
            approval_receipt_sha256=None, one_shot=True, retry=False, next_stage_authorized=False)
        c.save(p.PRIVATE/(stage+'_grant_INVALID.json'), grant)
        public_spec = {k: v for k, v in spec.items() if k in ('schema','stage','SOURCE','status','technical_sealed',
            'approved_by_user','runtime_authorization','calls','pending','n_fixed','retry','next_stage_authorized')}
        public_spec.update(private_spec_digest=c.digest(spec), source_hashes=source_hashes,
                           input_hashes=[{k:v for k,v in x.items() if k!='path'} for x in spec['inputs']])
        c.save(p.PUBLIC/(stage+'_contract_DRAFT.json'), public_spec)
        c.save(p.PUBLIC/(stage+'_grant_INVALID.json'), grant)
    s_stats = dict(status='DRAFT_UNAUTHORIZED', stage='S_STATS', SOURCE=args.source,
        inputs=['P_registration.json:saved_statistics', 'ACCEPTED_D_RESULT:PENDING_D_RESULT'],
        numeric_design_pass=1, maximum_normal_scenarios=84, fixed_reference_sign_comparisons=14,
        candidate_selection_or_reranking=0, raw_trajectory_aggregation=0,
        bootstrap=0, sampling=0, compile=0, reranking=0, approved_by_user=False,
        runtime_authorization=False, retry=False, next_stage_authorized=False,
        law_visibility_values='ONLY_IF_CANDIDATE_SPECIFIC_SAVED_LAW_INPUTS_EXIST; OTHERWISE_NULL')
    c.save(p.PUBLIC/'S_STATS_contract_DRAFT.json', s_stats)
    c.save(p.PUBLIC/'technical_source_binding.json', dict(SOURCE=args.source, source_hashes=source_hashes,
        local_only=True, remote_published=False, scientific_authorization=False,
        D_draft_digest=c.digest(d), P_draft_digest=c.digest(prospective_p),
        runtime_full_revalidation_in_phase0=False, runtime_file_manifest_count=len(original['runtime_file_hashes'])))
    print('TECHNICAL_DRAFTS_BOUND; INVALID_GRANTS; SCIENCE=0; SOURCE=' + args.source)


if __name__ == '__main__':
    main()
