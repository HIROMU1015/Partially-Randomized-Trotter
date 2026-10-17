"""Metadata-only evidence audit for the future noncancellation diagnostic.

Checks bytes/SHA, lineage joins, schema presence, finiteness types, pairing and
seed identity. It performs no sum, difference, absolute value, logarithm,
shot, cost, SE, selection or ranking on saved scientific values, and writes no
scientific value. Default CLI prints the scope only.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_noncancellation_contract as contract

AXES = ('real', 'imag')
WRAPPER_AXES = ('cosine', 'sine')
METRICS = ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size')
# Relative to the sealed attempt directory of the B2 recovery SOURCE.
B2_RECEIPT_PATHS = dict(driver_terminal='output01/terminal.json',
                        saved_B0_acceptance='output01/saved_B0_acceptance.json',
                        launcher_terminal='control01/launcher_terminal.json',
                        final_gate='control01/B2_final_gate.json',
                        launcher_log_terminal='launcher_admission01/terminal.json',
                        witness_native_exit='native_exit.json')
B2_STATUS_FIELDS = ('status', 'SOURCE', 'scope', 'first_stop', 'reason', 'error', 'native_exit_code',
                    'reported_exit_code', 'wait_completed', 'all_reaped', 'cleanup_errors')


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def complex_present(value):
    return type(value) is dict and set(value) >= set(AXES) and all(is_number(value[a]) for a in AXES)


def load(manifest, root):
    raw, receipts = {}, []
    for row in manifest['files']:
        path = contract.home_path(root/row['snapshot_path'])
        try:
            raw[row['role']] = contract.read_verified(path, row)
            receipts.append(dict(role=row['role'], bytes=row['bytes'], sha256=row['sha256'], PASS=True))
        except (OSError, ValueError) as exc:
            receipts.append(dict(role=row['role'], bytes=row['bytes'], sha256=row['sha256'], PASS=False,
                                 reason=type(exc).__name__ + ':' + str(exc)))
    return raw, receipts


def PR_record_fields(record):
    d = record.get('error_decomposition', {})
    parts = d.get('signed_parts', {})
    s, f, u = record.get('signals', {}), record.get('independent_forward_signals', {}), record.get('allowance', {})
    cell = record.get('cell', {})
    return dict(
        cell_R_equals_q_times_r=type(cell.get('R')) is int and cell.get('R') == cell.get('q', 0)*cell.get('r', 0),
        signed_part_names=sorted(parts), outer_PF_both_axes=complex_present(parts.get('outer_PF')),
        finite_RTE_both_axes=complex_present(parts.get('finite_RTE')),
        total_signed_both_axes=complex_present(d.get('total_signed')),
        endpoint_signals_raw_corrected_exact_tail=all(complex_present(s.get(k)) for k in ('raw', 'corrected', 'exact_tail')),
        forward_raw_corrected_exact_tail=all(complex_present(f.get(k)) for k in ('raw', 'corrected', 'exact_tail')),
        reference_both_axes=complex_present(record.get('reference')),
        normalization_b_present=is_number(record.get('b')), log_B_present=is_number(record.get('log_B')),
        normalization_log_discrepancy_present=is_number(record.get('normalization_log_discrepancy')),
        empirical_u_real_imag_margin=all(is_number(u.get(k)) for k in ('real', 'imag', 'log_B_margin')),
        u_kind_empirical_uncertified=u.get('kind') == 'EMPIRICAL' and u.get('certified') is False,
        numerical_allowance_certified=record.get('numerical_allowance_certified'),
        complex_modulus_absolute_parts_present_not_used=set(d.get('absolute_parts', {})) == {'outer_PF', 'finite_RTE'})


def audit(manifest, root, b2_config=None):
    raw, receipts = load(manifest, root)
    out = dict(schema='H8_NONCANCELLATION_EVIDENCE_AUDIT_V1', scope='METADATA_SCHEMA_LINEAGE_ONLY',
               input_receipts=receipts, input_bytes_all_PASS=all(r['PASS'] for r in receipts))
    if not out['input_bytes_all_PASS']:
        out['status'] = 'STOP_INPUT_BYTES'
        return out
    o = {k: contract.strict(v) for k, v in raw.items() if k not in ('A_signals', 'wrapper_metrics')}
    export = o['A_export']
    decoded = gzip.GzipFile(fileobj=io.BytesIO(raw['A_signals'])).read()
    gzip_ok = (len(decoded) == export['original_json_bytes'] and
               hashlib.sha256(decoded).hexdigest() == export['original_json_sha256'])
    o['A_signals'] = contract.strict(decoded)
    wrappers = contract.d0.parse_wrappers(raw['wrapper_metrics'])
    registry, selection, completion = o['registry'], o['confirmation_selection'], o['old_completion']
    binding, signed = o['old_D0_contract']['binding'], o['signed_axis_results']
    cells = {c['id']: c for c in registry['old'] + registry['additional']}
    union = selection['selected_ids']
    lineage = dict(A_gzip_original_bytes_sha=gzip_ok)
    sb = o['signed_binding']
    lineage['signed_run01_raw_result_sha_bytes'] = (
        sb['raw_axis_result_sha256'] == contract.receipt_of(manifest['files'], 'signed_axis_results', 'role')['sha256'] and
        sb['raw_axis_result_bytes'] == len(raw['signed_axis_results']) and sb['SOURCE'] == signed['SOURCE'] ==
        contract.numeric.SIGNED['SOURCE'] and sb['plan_digest'] == signed['plan_sha256'] and
        sb['grant_digest'] == signed['grant_sha256'])
    lineage['signed_run01_shared_input_roles'] = all(
        contract.receipt_of(signed['input_receipts'], role, 'role')['sha256'] ==
        contract.receipt_of(manifest['files'], role, 'role')['sha256'] for role in contract.SIGNED_ROLES)
    for name, role in (('signal_summary.json', 'old_signals'), ('wrapper_metrics.csv', 'wrapper_metrics'),
                       ('selection_before_confirmation.json', 'confirmation_selection'),
                       ('completion_audit.json', 'old_completion')):
        row = contract.receipt_of(o['old_D0_contract']['input_receipts'], name)
        lineage['D0_receipt:' + name] = row['sha256'] == contract.receipt_of(manifest['files'], role, 'role')['sha256']
    lineage['D0_selection_digest'] = binding['selection_digest'] == contract.d0.digest(selection)
    lineage['selection_signal_digest'] = selection['signal_digest'] == contract.d0.digest(o['old_signals']['signals'])
    lineage['selection_fixed_before_confirmation_unpooled'] = (
        selection['fixed_before_confirmation'] is True and selection['exploration_confirmation_pooled'] is False)
    lineage['completion_SOURCE_seed_input_compiler_union'] = (
        completion['SOURCE'] == contract.PROVIDER and completion['seed_source'] == binding['seed_source'] and
        completion['input_SHA'] == binding['input'] and completion['compiler_hash'] == binding['compiler'] ==
        contract.numeric.COMPILER and completion['selected_ids'] == union)
    acc = contract.receipt_of(o['D0_terminal']['output_file_receipts'], 'candidate_accounting_v1.json', 'name')
    lineage['D0_terminal_complete_and_accounting_receipt'] = (
        o['D0_terminal']['status'] == 'COMPLETE_MANDATORY_STOP' and o['D0_terminal']['first_stop'] is None and
        acc['sha256'] == contract.receipt_of(manifest['files'], 'D0_accounting', 'role')['sha256'])
    lineage['D0_provenance_SOURCE_contract'] = (
        o['D0_provenance']['SOURCE'] == contract.D0['SOURCE'] and
        o['D0_provenance']['contract_sha256'] == contract.receipt_of(manifest['files'], 'old_D0_contract', 'role')['sha256'])
    files = o['B0I_file_receipts']['files']
    lineage['old_B0I_accounting_and_signal_receipts'] = (
        contract.receipt_of(files, 'case_accounting_v1.json', 'name')['sha256'] ==
        contract.receipt_of(manifest['files'], 'B0I_accounting', 'role')['sha256'] and
        contract.receipt_of(files, 'case_scalar_signal_receipts_v1.json', 'name')['sha256'] ==
        contract.receipt_of(manifest['files'], 'old_B0I', 'role')['sha256'])
    lineage['B1_terminal_model_native_exit'] = (
        o['B1_terminal']['status'] == 'COMPLETE_MANDATORY_STOP' and o['B1_terminal']['SOURCE'] == contract.B1['SOURCE'] and
        o['B1_terminal']['model_sha256'] == contract.numeric.MODEL_SHA == contract.d0.digest(o['B1_models']) and
        o['B1_launcher_native_exit']['native_exit_code'] == 0 and o['B1_launcher_native_exit']['wait_completed'] is True)
    reg = o['B2_saved_B0_registry']
    lineage['B2_saved_B0_registry_two_axes_from_STOPPED_run01'] = (
        reg['original_execution_status'] == 'STOP' and reg['model_sha256'] == contract.numeric.MODEL_SHA and
        [(r['case_id'], r['axis']) for r in reg['axes']] == [x[:2] for x in contract.numeric.B2['axes'][:2]] and
        all(r['model_check']['PASS'] is True for r in reg['axes']))
    out['lineage'] = lineage
    # PR confirmation union6 / n32.
    records = {x['cell']['id']: x for x in o['old_signals']['signals']}
    tasks = {}
    for item in selection['registrations']:
        t = item['task']
        tasks.setdefault(t['cell_id'], []).append(t)
    D0_conf = [r for r in o['D0_accounting']['H8_CONFIRMATION_UNION6'] if r['cell_id'] in union]
    wrapper_conf_ids = sorted({r['cell_id'] for r in wrappers if r['phase'] == 'confirmation'})
    PR = dict(union_ids=union, membership_consistent=(
        sorted(union) == sorted(completion['selected_ids']) == wrapper_conf_ids ==
        sorted({r['cell_id'] for r in o['D0_accounting']['H8_CONFIRMATION_UNION6'] if r['method'] == 'B2'})),
        candidates={})
    for cid in union:
        record = records[cid]
        mine = sorted(tasks.get(cid, []), key=lambda t: t['replica'])
        seeds = {t['replica']: t['seed'] for t in mine}
        conf = [r for r in wrappers if r['cell_id'] == cid and r['phase'] == 'confirmation']
        explo = [r for r in wrappers if r['cell_id'] == cid and r['phase'] == 'exploration']
        try:
            pairs = contract.d0.paired_rows(conf, cell_id=cid, phase='confirmation', compiler_hash=binding['compiler'])
            pairing = 'PASS'
        except ValueError as exc:
            pairs, pairing = [], 'FAIL:' + str(exc)
        rows = [r for r in D0_conf if r['cell_id'] == cid]
        fields = PR_record_fields(record)
        entry = dict(fields=fields,
            registrations=len(mine), registration_replicas_0_to_31=[t['replica'] for t in mine] == list(range(32)),
            registration_seed_derivation=all(t['seed'] == contract.task_seed(binding, cid, 'confirmation', t['replica'])
                                             for t in mine),
            registration_expected_signal_is_saved_corrected=all(t['expected_signal'] == record['signals']['corrected']
                                                                for t in mine),
            wrapper_rows=len(conf), paired_trajectories=len(pairs), pairing=pairing,
            wrapper_symmetric_directional=all(r['control'] == 'symmetric_directional' for r in conf),
            wrapper_compiler_bound=all(r['compiler_hash'] == binding['compiler'] for r in conf),
            wrapper_sources=sorted({r['source'] for r in conf}),
            wrapper_sources_allowed=all(r['source'] in binding['allowed_wrapper_sources'] for r in conf),
            wrapper_seed_equals_registration=all(p['cosine']['seed'] == seeds.get(p['cosine']['replica']) for p in pairs),
            wrapper_integer_metrics=all(type(r[m]) is int and r[m] >= 0 for r in conf for m in METRICS),
            wrapper_fingerprints_present=all(type(r['fingerprint']) is str and len(r['fingerprint']) == 64 for r in conf),
            exploration_rows_kept_separate=len(explo),
            D0_saved_accounting_rows=len(rows),
            D0_rows_confirmation_n32=all(r['sample_phase'] == 'confirmation' and r['sample_count'] == 32 for r in rows),
            D0_derived_rows_paired_samples_32=all(r['metrics']['RZ']['paired_samples'] == 32 for r in rows
                                                  if r['status'] == 'DERIVED_SAVED_SCALAR'))
        entry['data_sufficient_for_future_approved_computation'] = (
            fields['outer_PF_both_axes'] and fields['finite_RTE_both_axes'] and fields['total_signed_both_axes'] and
            fields['endpoint_signals_raw_corrected_exact_tail'] and fields['forward_raw_corrected_exact_tail'] and
            fields['normalization_b_present'] and fields['log_B_present'] and fields['empirical_u_real_imag_margin'] and
            fields['u_kind_empirical_uncertified'] and fields['cell_R_equals_q_times_r'] and
            fields['numerical_allowance_certified'] is False and entry['paired_trajectories'] == 32 and
            entry['registration_seed_derivation'] and entry['wrapper_seed_equals_registration'] and
            entry['wrapper_sources_allowed'] and entry['wrapper_integer_metrics'] and entry['D0_saved_accounting_rows'] == 10
            and entry['D0_rows_confirmation_n32'])
        PR['candidates'][cid] = entry
    PR['rare_order_histograms'] = completion['Taylor_orders']
    PR['confirmation_unobserved_orders'] = [k for k in (0, 2, 4) if completion['Taylor_orders']['confirmation'].get(str(k), 0) == 0]
    PR['not_checked_numerically_now'] = [
        'outer_PF/finite_RTE endpoint differences and closure', 'forward endpoint agreement',
        'unrotated full-H reference agreement', 'log_B == R*log(b) exact binding',
        'usual saved accounting and paired metric reproduction', 'noncancellation shots, paired totals, SE']
    out['PR_confirmation'] = PR
    # Deterministic 184 and cost-unit inputs: presence only.
    det = {x['cell']['id']: x for x in o['old_signals']['signals'] if x['cell']['id'] in cells}
    det.update({w['signal']['cell']['id']: w['signal'] for w in o['old_B0I']})
    det.update({x['cell']['id']: x for x in o['A_signals']['signals']})
    signed_rows = {(r['cell']['id'], r['axis']): r for r in signed['rows']}
    usual = set()
    for row in o['D0_accounting']['H8_EXPLORATION92']:
        if row['cell_id'] in cells:
            usual.add((row['cell_id'], float(row['epsilon']), float(row['factor'])))
    for row in o['B0I_accounting']:
        usual.add((row['cell_id'], float(row['epsilon']), float(row['factor'])))
    for s in o['A_signals']['signals']:
        for e, v in s['precision_rows'].items():
            usual.update((s['cell']['id'], float(e), float(f)) for f in v)
    usual.update((r['cell_id'], float(r['epsilon']), float(r['factor'])) for r in D0_conf)
    expected = {(c, e, f) for c in [*cells, *union] for e in contract.numeric.EPSILONS for f in contract.numeric.FACTORS}
    full_pf = [c for c, x in cells.items() if x['method'] == 'B1']
    explo_pairs = {}
    for cid in full_pf:
        rows = [r for r in wrappers if r['cell_id'] == cid and r['phase'] == 'exploration']
        try:
            explo_pairs[cid] = len(contract.d0.paired_rows(rows, cell_id=cid, phase='exploration',
                                                           compiler_hash=binding['compiler']))
        except ValueError:
            explo_pairs[cid] = None
    anchors = o['B1_anchors']['rows']
    out['deterministic'] = dict(
        candidates=len(cells), old=len(registry['old']), additional=len(registry['additional']),
        records_present=len(det) == 184 and set(det) == set(cells),
        signed_axes_PASS=sum(1 for k in [(c, a) for c in cells for a in AXES]
                             if signed_rows.get(k, {}).get('status') == 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY'),
        signed_axes_reconstructed=sum(1 for r in signed['rows'] if r['reconstructed']),
        signed_part_sets_by_method={m: sorted({tuple(sorted(r['signed_parts'])) for r in signed['rows']
                                               if r['cell']['method'] == m}) for m in ('B0', 'B0I', 'B1')},
        u_logB_margin_present=sum(1 for x in det.values() if all(is_number(x['allowance'].get(k)) for k in
                                  ('real', 'imag', 'log_B_margin')) and is_number(x.get('log_B')) and
                                  x['allowance'].get('kind') == 'EMPIRICAL' and x['allowance'].get('certified') is False),
        saved_usual_rows=len(usual & expected), saved_usual_rows_expected=len(expected),
        saved_usual_rows_extra_or_missing=len(usual ^ expected))
    out['cost_inputs'] = dict(
        B1_anchor_axes=len(anchors), B1_model_strata=len(o['B1_models']['models']),
        B1_model_status=sorted({m['status'] for m in o['B1_models']['models']}),
        B1_S4_status=sorted({m['S4_status'] for m in o['B1_models']['models']}),
        full_PF_cells=len(full_pf), full_PF_exploration_N1_pairs=explo_pairs,
        old_B0_wrapper_costs_reused=False, B2_saved_B0_axes=len(reg['axes']),
        B2_required_SOURCE=contract.numeric.B2['SOURCE'], B2_required_scope=contract.numeric.B2['scope'])
    out['B2_status'] = B2_probe(b2_config) if b2_config else None
    out['cost_comparison_inputs_complete'] = bool(out['B2_status'] and out['B2_status']['B2_full_four_axis_PASS_receipts']
                                                  and not out['B2_status']['B2_process_running'])
    out.update(status='AUDIT_COMPLETE_MANDATORY_STOP', scientific_values_written=False,
               arithmetic_on_saved_scientific_values=0, noncancellation_values=0, shots=0, costs=0, rankings=0,
               new_signal_compile_sampling_fit=0, rigorous_bias_guarantee=None, next_stage_authorized=False)
    return out


def B2_probe(config_path):
    """Read-only: config SOURCE, final receipt status fields and own process names.

    Opens small terminal receipts for reading only; never writes, renames,
    signals or locks anything in the B2 areas. Axis measurements are not copied.
    """
    config_path = Path(config_path)
    config = json.loads(config_path.read_bytes())
    root = config_path.parent
    present, statuses, digests = {}, {}, {}
    for role, rel in B2_RECEIPT_PATHS.items():
        path = root/rel
        present[role] = path.is_file()
        if present[role]:
            data = path.read_bytes()
            digests[role] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            value = json.loads(data)
            statuses[role] = {k: value[k] for k in B2_STATUS_FIELDS if k in value}
            if role == 'driver_terminal':
                statuses[role]['completed_axis_rows'] = len(value.get('axes', []))
    accepted = sorted(p.name for p in (root/'output01').glob('axis_*_accepted.json'))
    running = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if p.stat().st_uid != os.getuid():
                    continue
                cmd = (p/'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if any(x.endswith(b'run_h8_phase_b2_recovery.py') for x in cmd):
                running.append(int(p.name))
    complete = (all(present.values()) and statuses['driver_terminal'].get('status') == 'COMPLETE_MANDATORY_STOP'
                and statuses['final_gate'].get('status') == 'PASS_MANDATORY_STOP'
                and statuses['witness_native_exit'].get('native_exit_code') == 0)
    return dict(config_SOURCE=config['SOURCE'], required_SOURCE_matches=config['SOURCE'] == contract.numeric.B2['SOURCE'],
                final_receipts_present=present, receipt_status_fields=statuses, receipt_digests=digests,
                new_B0I_axis_acceptance_files=accepted, all_final_receipts_present=all(present.values()),
                B2_full_four_axis_PASS_receipts=complete, B2_process_running=bool(running),
                B2_driver_process_count=len(running), access='READ_ONLY', B2_files_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-saved-metadata', action='store_true')
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--B2-config', type=Path)
    args = parser.parse_args()
    if not args.audit_saved_metadata:
        print(json.dumps(dict(scope='H8_NONCANCELLATION_EVIDENCE_AUDIT_V1', metadata_only=True,
                              saved_H8_numerical_analysis=0)))
        return 0
    manifest = contract.strict((contract.ROOT/contract.MANIFEST).read_bytes())
    result = audit(manifest, contract.home_path(args.input_root), args.B2_config)
    result['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with contract.home_path(args.output).open('x') as stream:
        json.dump(result, stream, indent=1, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], input_bytes_all_PASS=result['input_bytes_all_PASS'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
