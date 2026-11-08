"""Audit completed run02 bytes and scalar records; never run scientific engines."""
from collections import Counter
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FAMILY = ROOT / 'artifacts/research_c_cdf0_relocated/2026-11-08'
RUN = FAMILY / 'science/run02'
PUBLIC = FAMILY / 'publication_run02'
PUBLIC.mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def table(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


c = read(RUN / 'contract.json')
old_contract = read(ROOT / 'docs/research/research_c_cdf0_relocated_contract_proposal_20261108.json')
assert [k for k in c if c[k] != old_contract[k]] == ['fresh_output']
assert (RUN / 'contract.json').read_bytes() == (ROOT / 'docs/research/research_c_cdf0_relocated_contract_rerun02_20261108.json').read_bytes()
assert sha(RUN / 'SOURCE.json') == sha(ROOT / c['SOURCE']) == c['SOURCE_sha256']
source = read(ROOT / c['SOURCE'])
for group in (source['files'], c['inputs']):
    assert all(sha(ROOT / name) == expected for name, expected in group.items())
old_source = read(ROOT / 'artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/SOURCE.json')
assert all(sha(ROOT / name) == expected for name, expected in old_source['files'].items())
old_audit = read(FAMILY / 'execution/result_audit.json')
assert all(sha(ROOT / name) == entry['sha256'] for name, entry in old_audit['raw_files'].items())
assert read(FAMILY / 'science/run01/STOP.json')['failed'] is True

stop = read(RUN / 'STOP.json')
result = read(RUN / 'CDF0.json')
final = read(RUN / 'final_summary.json')
frozen = read(RUN / 'frozen_selection.json')
counts = result['call_counts']
assert stop == dict(failed=False, completed=['CDF0'], reason='REGISTERED_TASKS_COMPLETE', retry=0, resume=0, mandatory_stop=True)
assert result['status'] == 'CDF0_REGISTERED_SCOPE_COMPLETE'
assert result['selection'] == frozen['selection'] and result['selection']['selected_r'] == 64
assert len(final['rows']) == 64 and final['result']['r'] == 64

rounds = []
policies = []
for r in (8, 16, 32, 64, 128):
    normal = read(RUN / f'r{r}/reference/normalization.json')
    law = read(RUN / f'r{r}/reference/law.json')
    assert normal['status'] == law['status'] == 'PASS'
    rows = [read(RUN / f'r{r}/reference/m{m:02d}.json') for m in range(13)]
    selected = next(row for row in frozen['rows'] if row['r'] == r)
    eligible = selected['eligible']
    sel = read(RUN / f'r{r}/selection/result.json') if eligible else None
    ideal = sel['lanes']['ideal_exact_native'] if eligible else None
    for row in rows:
        ref = row['reference']
        rounds.append(dict(r=r, m=row['m'], pf_operator_error=ref['pf_operator_error'],
            rte_corrected_operator_error=ref['rte_corrected_operator_error'],
            attenuation=ref['attenuation'], classification=row['feasibility']['status'],
            reason=row['feasibility'].get('reason'), native_policy_admitted=eligible))
    m12 = rows[-1]['reference']
    policies.append(dict(r=r, native_admitted=eligible,
        m12_classification=rows[-1]['feasibility']['status'],
        m12_pf_operator_error=m12['pf_operator_error'],
        m12_rte_corrected_operator_error=m12['rte_corrected_operator_error'],
        m12_attenuation=m12['attenuation'],
        selection_samples=sel['samples'] if eligible else 0,
        selection_total_RZ=selected['total_rz'],
        ideal_shots=ideal['total_shots'] if eligible else None,
        selection_mean_RZ=sel['calibration']['sample_mean']['rz_count'] if eligible else None))
table(PUBLIC / 'reference_rounds65.csv', rounds)

sample_rows = []
domains = set()
worker_count = Counter()
checks = []
for r, split, batches in ((64, 'selection', 1), (128, 'selection', 1), (64, 'final', 2)):
    parent = RUN / (f'r{r}/selection' if split == 'selection' else f'final_r{r}')
    for batch in range(batches):
        folder = parent / f'batch{batch:02d}'
        workers = [json.loads(line) for line in (folder / 'worker_samples.jsonl').read_text().splitlines()]
        assert len(workers) == 32 and {x['sample'] for x in workers} == set(range(32))
        assert all(x['status'] == 'PASS' and x['failure'] is None for x in workers)
        for worker in workers:
            worker_count.update(x['kind'] for x in worker['calls'])
        for sample in range(32):
            prefix = folder / f'samples/sample{sample:02d}'
            events = read(str(prefix) + '_events.json')
            assert events['SOURCE'] == c['SOURCE_sha256']
            assert events['JU'] == result['selection']['candidate_sha256']
            assert events['sampling']['events_per_sample'] == r
            assert len(events['events']) == r and events['old_cohort_pooled'] is False
            domain = (events['sampling']['cohort'], events['sampling']['axis'], sample)
            assert domain not in domains
            domains.add(domain)
            metrics = read(str(prefix) + '_metrics.json')
            raw = read(str(prefix) + '_raw.json')
            compiled = read(str(prefix) + '_compiled.json')
            adjoint = read(str(prefix) + '_adjoint.json')
            xy = read(str(prefix) + '_XY.json')
            assert xy['status'] == 'PASS'
            assert raw['reference']['max_entry'] <= 1e-10
            assert compiled['reference']['max_entry'] <= 1e-10
            assert all(v['reference']['max_entry'] <= 1e-10 and v['returned']['max_entry'] <= 1e-10 for v in adjoint.values())
            assert all(v['error'] <= 2e-10 for v in xy['signals'].values())
            assert all((folder / f'circuits/sample{sample:02d}_{kind}.qpy').exists()
                for kind in ('raw', 'compiled', 'raw_adjoint', 'compiled_adjoint'))
            check = dict(raw_reference=raw['reference']['max_entry'], compiled_reference=compiled['reference']['max_entry'],
                raw_inverse_reference=adjoint['raw']['reference']['max_entry'],
                compiled_inverse_reference=adjoint['compiled']['reference']['max_entry'],
                raw_inverse_return=adjoint['raw']['returned']['max_entry'],
                compiled_inverse_return=adjoint['compiled']['returned']['max_entry'],
                raw_XY_error=xy['signals']['raw']['error'], compiled_XY_error=xy['signals']['compiled']['error'])
            checks.append(check)
            sample_rows.append(dict(split=split, r=r, batch=batch, sample=sample,
                cohort=events['sampling']['cohort'], SOURCE_sha256=events['SOURCE'], **metrics, **check))
assert len(sample_rows) == len(domains) == 128
assert sum(worker_count.values()) == 1152
assert all(counts[k] == v for k, v in worker_count.items())
table(PUBLIC / 'native_samples128.csv', sample_rows)

call_scope = dict(candidates=1, relocations=counts['relocated_model'],
    original_factor_gates=counts['original_factor_gate'], full_H_gate=counts['Hamiltonian_gate'],
    saved_ground_gate=counts['ground_gate'], PR_PF_kernels=counts['kernel_reference'],
    exact_evolutions=counts['exact_evolution'], PR_RTE_policies=counts['law_prepare'],
    PR_reference_rounds=counts['round_reference'], selection_native_samples=64,
    final_native_samples=64, native_samples=counts['trajectory_sample'],
    native_kernel_builds=counts['kernel_build'], native_compiles=counts['compile'],
    actual_inverse_builds=2 * counts['adjoint_pair_gate'],
    native_kernel_full512_decodes=4 * counts['trajectory_sample'], shell_full512_decodes=2 * counts['shell_gate'],
    final_policy_pairs=1, primary_comparisons=counts['primary_comparison'],
    fits=0, DF_norms=0, D_evaluations=0, ENTRY_recomputations=0, quantum_shots=0)
for key, actual in call_scope.items():
    assert actual <= c['maximum_scientific_scope'][key], (key, actual)
assert sum(counts.values()) == 1710

files = []
for path in sorted(RUN.rglob('*')):
    if path.is_file():
        files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path),
            classification='runtime_cache' if path.suffix in ('.pkl', '.npy') else 'scientific_evidence'))
inventory = dict(schema='raw_file_inventory_v1',
    root=str(RUN.relative_to(ROOT)), files=files, includes_runtime_cache=True,
    numeric_arrays_or_circuits_decoded_during_audit=False)
manifest_bytes = (json.dumps(inventory, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
with (PUBLIC / 'raw_files.json.gz').open('xb') as stream:
    stream.write(gzip.compress(manifest_bytes, mtime=0))

comparison = result['comparison']
CDF0, ENTRY = comparison['total_points']['PR'], comparison['total_points']['ENTRY']
assert final['result']['lanes']['ideal_exact_native']['conditional_totals']['totals']['rz_count'] == CDF0
assert CDF0 - ENTRY == comparison['difference_point']
summary = dict(status=result['status'], case='C_PRECISION_TARGET_UNRESOLVED',
    scientific_SOURCE_commit='bde646c2b0d275c66d9890eee835d4692f55cafb', SOURCE_sha256=c['SOURCE_sha256'],
    run02_contract_sha256=sha(RUN / 'contract.json'), changed_contract_fields=['fresh_output'],
    candidate_id=c['candidate']['id'], candidate_sha256=c['candidate']['sha256'], input_identity=c['input_identity'],
    selected_r=64, selection_only=True, final_samples=64, ENTRY_new_samples=0,
    policies=policies, tail=read(RUN / 'r64/reference/law.json'),
    final_bindings=final['bindings'], final_calibration=final['result']['calibration'],
    final_conditional_totals={k:v['conditional_totals'] for k,v in final['result']['lanes'].items()},
    primary_comparison=comparison,
    display_arithmetic=dict(ENTRY_point_reduction_relative_to_CDF0=(CDF0-ENTRY)/CDF0,
        halfwidth_relative_to_ENTRY=comparison['halfwidth']/ENTRY,
        note='Arithmetic display of the registered pair only; no new inference, interval or ranking'),
    call_counts=counts, call_counts_total=sum(counts.values()), actual_scientific_scope=call_scope,
    maximum_scientific_scope=c['maximum_scientific_scope'],
    correctness=dict(samples=128, all_worker_status_PASS=True, raw_compiled_actual_inverse_XY_records_PASS=True,
        full_input_output_dimension=512, component_allowance=1e-10, signal_allowance=2e-10,
        maxima_over_executed_samples={k:max(x[k] for x in checks) for k in checks[0]},
        original_factor_gates=9, Hamiltonian_gate=1, sector_and_full_CAR_pass=True,
        evidence_for_unsaved_gate_reports='Successful fixed SOURCE callbacks in persisted call_counts; no residual values invented',
        qualification='finite binary64 numerical pilot, not all-law synthesis or actual CA'),
    raw_inventory=dict(files=len(files), bytes=sum(x['bytes'] for x in files),
        evidence_files=sum(x['classification']=='scientific_evidence' for x in files),
        evidence_bytes=sum(x['bytes'] for x in files if x['classification']=='scientific_evidence'),
        cache_files=sum(x['classification']=='runtime_cache' for x in files),
        manifest='raw_files.json.gz', manifest_sha256=sha(PUBLIC/'raw_files.json.gz'),
        full_raw_binary_public=False, full_raw_retained_at=str(RUN)),
    preservation=dict(SOURCE112=True, inputs32=True, old_SOURCE68=True, old_run01_raw=True,
        old_failed_STOP_unchanged=True, original_proposal_unchanged=True, shared_environment_unchanged=True),
    execution=dict(new_run_execute_invocations=1, technical_relaunch_from_run01=1,
        retry_within_run=0, resume_within_run=0, wall_seconds_GNU_time=156.88,
        user_seconds=4103.34, system_seconds=153.67, peak_RSS_KiB_GNU_time=1060708),
    STOP=stop, additional_scientific_work_after_STOP=0,
    qualifications=dict(actual_CA_achieved=None,whole_law_synthesis=None,ideal_law_price_transfer=False,
        guaranteed_resource_superiority=None,ratio_interval=None,prospective_two_arm_trial=False),
    audit_method='JSON/JSONL scalar transcription, bytes/hash and scope checks only; no NPZ/QPY decode, scientific functions or new confidence calculation')
assert summary['display_arithmetic']['halfwidth_relative_to_ENTRY'] > .05
assert comparison['precision_met'] is False and comparison['decision'] == 'UNRESOLVED_PRECISION_TARGET'
save(PUBLIC / 'summary.json', summary)
save(FAMILY / 'execution_run02/result_audit.json', summary)
save(FAMILY / 'execution_run02/completion.json', dict(status='REGISTERED_SCOPE_COMPLETE_PRECISION_UNRESOLVED',
    recorded_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    execute_invocations=1,exit_status=0,retry=0,resume=0,mandatory_stop=True,
    raw_manifest=str((PUBLIC/'raw_files.json.gz').relative_to(ROOT)),
    raw_manifest_sha256=sha(PUBLIC/'raw_files.json.gz')))
print(json.dumps(dict(status=summary['status'],case=summary['case'],selected_r=64,
    CDF0_total_RZ=CDF0,ENTRY_total_RZ=ENTRY,display=summary['display_arithmetic'],
    callbacks=sum(counts.values()),actual_scientific_scope=call_scope,
    raw=summary['raw_inventory'],correctness_maxima=summary['correctness']['maxima_over_executed_samples']),ensure_ascii=False))
