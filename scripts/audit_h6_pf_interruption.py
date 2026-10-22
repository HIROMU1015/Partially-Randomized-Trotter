"""Read-only acceptance of complete pairs after a lost COST process.

No evolution, state restoration, wrapper construction, compile, qualification,
shot evaluation or ranking is performed. Failed-stage reservations stay charged.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

import pf_common_run_contract as c

BASE = '0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'
OUTPUT = 'artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22'
DRIVERS = {'run_pf_common_run.py', 'run_h6_pf_common_successors.py',
           'finish_h6_pf_successful_results.py', 'close_h246_pf_common_success.py'}


def active_processes(private):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            args = [x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if str(private) in args and any(Path(a).name in DRIVERS for a in args):
                found.append(dict(pid=int(p.name), args=args))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def stage_control(private, name, successful):
    root = private/name
    grant = c.read(root/'grant.json')
    marker = c.read(root/'CONSUMED.json')
    ledger = [c.read(p) for p in sorted((root/'ledger').glob('*.json'))]
    counts = dict.fromkeys(grant['caps'], 0)
    for entry in ledger:
        c.require(entry['entered_nonrefundable'] is True, 'NONREFUNDABLE')
        counts[entry['kind']] += entry['count']
    c.require(counts == c.read(root/'reserved_counts.json'), 'LEDGER_COUNTER:'+name)
    c.require(all(counts[k] <= grant['caps'][k] for k in counts), 'CAPS:'+name)
    c.require(marker['grant'] == grant, 'MARKER:'+name)
    c.require(grant['plan_sha256'] == c.sha(private/'plan_APPROVED.json'), 'PLAN:'+name)
    receipt = c.read(root/'completion.json') if (root/'completion.json').exists() else None
    if successful:
        c.require(receipt['status'] == 'PASS' and receipt['actual_calls'] == counts, 'RECEIPT:'+name)
    else:
        c.require(receipt is None, 'COST_HAS_NO_COMPLETION')
    return dict(grant=grant, marker=marker, ledger=ledger, counts=counts, receipt=receipt)


def standard_fresh_budget(rows, consumed, template):
    n = len(rows)
    fresh = dict(snapshot_restore=1, saved_reference_read=1, occupation_oracle=1,
        full_preparation=1, prefix_preparation=len({r['prefix'] for r in rows if r['method'] != 'FULL'}),
        native_evolution=n, native_piece=sum(r['q']*c.PIECES[r['formula']] for r in rows),
        independent_occupation=n, native_low=n, native_high=n, wrapper_build=2*n,
        uncompiled_action=2*n, wrapper_compile=2*n, compiled_action=2*n,
        sampling=0, new_signal=0, new_full_H_reference=0)
    names = dict(native_piece='native_signed_S2_pieces', native_low='native_low_actions',
        native_high='native_high_actions', independent_occupation='independent_occupation_actions',
        uncompiled_action='uncompiled_wrapper_actions', compiled_action='compiled_wrapper_actions')
    comparison = []
    for k in fresh:
        if k in ('snapshot_restore', 'saved_reference_read', 'occupation_oracle',
                 'full_preparation', 'prefix_preparation'):
            continue
        cap = template.get(names.get(k, k))
        if cap is not None:
            total = consumed[k]+fresh[k]
            comparison.append(dict(kind=k, consumed=consumed[k], fresh=fresh[k],
                cumulative=total, fixed_upper_bound=cap, excess=max(0, total-cap)))
    return dict(status='UNAPPROVED_NOT_A_GRANT', retry=False, resume=False,
        candidates=rows, fresh_caps=fresh, cumulative_comparison=comparison,
        exceeds_fixed_internal_budget=any(r['excess'] for r in comparison),
        interpretation='Conservative fresh paired acquisition by the unchanged original method; '
            'not a proof of the minimal possible recovery budget. Partial axes have no complete paired receipt.',
        new_PR_sampling=0, additional_signal=0, additional_qualification=0)


def audit(private, metadata, output):
    private, metadata, output = map(lambda p: Path(p).resolve(), (private, metadata, output))
    c.require(not output.exists(), 'NEW_AUDIT_OUTPUT')
    observed = datetime.now(timezone.utc).isoformat()
    observations = [dict(time_utc=observed, processes=active_processes(private))]
    time.sleep(1)
    observations.append(dict(time_utc=datetime.now(timezone.utc).isoformat(), processes=active_processes(private)))
    c.require(all(not r['processes'] for r in observations), 'PROCESS_STILL_ALIVE')
    plan = c.read(private/'plan_APPROVED.json')
    c.verify(plan)
    controls = {s: stage_control(private, s, s != 'COST') for s in ('SIGNAL', 'QUALIFY', 'COST')}
    c.require(len({v['grant']['grant_id'] for v in controls.values()}) == 3, 'DISTINCT_GRANTS')
    selected = c.read(private/'COST/selected_candidates.json')
    saved = c.read(private/'saved_prices.json')
    signals = c.read(private/'SIGNAL/output/all_signals.json')
    endpoints = c.read(private/'QUALIFY/output/endpoints.json')
    c.require(len(signals) == 276 and len(endpoints) == 1380, 'FULL_SIGNAL_QUALIFICATION')
    c.require(all(e['status'] in ('ELIGIBLE', 'INELIGIBLE') for e in endpoints), 'NO_UNKNOWN_ELIGIBILITY')
    complete, missing, archives, orphan_files, maxima = [], [], [], [], {}
    import numpy as np
    for row in selected:
        ident = c.cid(row)
        root = private/'COST/output'/ident
        if not (root/'result.json').exists():
            missing.append(row)
            for p in sorted(root.glob('*')):
                if p.is_file():
                    orphan_files.append(dict(candidate=ident, path=str(p.relative_to(private)),
                        bytes=p.stat().st_size, sha256=c.sha(p),
                        accepted_price=False, original_archive_digest_receipt_available=False))
            continue
        result = c.read(root/'result.json')
        c.require(result['status'] == 'PASS' and result['candidate'] == row and result['SOURCE'] == plan['SOURCE'], 'NATIVE_SOURCE')
        c.require(result['hamiltonian'] == plan['binding']['hamiltonian'] and result['state'] == plan['binding']['state'], 'NATIVE_INPUT')
        for check in result['checks']:
            tol = 1e-7 if check['check'].startswith('compiled_vector_') else 1e-9
            c.require(check['PASS'] and check['phase_alignment'] is False and check['strictly_below'] == tol
                and math.isfinite(check['error']) and 0 <= check['error'] < tol, 'NATIVE_GATE')
            maxima[check['check']] = max(maxima.get(check['check'], 0.), check['error'])
        c.require(len(result['archives']) == 6, 'SIX_PAIRED_ARCHIVES')
        for entry in result['archives']:
            path = root/entry['path']
            blob = path.read_bytes()
            raw = gzip.decompress(blob)
            c.require(len(raw) == entry['original_bytes'] and hashlib.sha256(raw).hexdigest() == entry['original_sha256']
                and hashlib.sha256(blob).hexdigest() == entry['archive_sha256'], 'RAW_ARCHIVE_BYTE_RESTORATION')
            if entry['path'].endswith('_vectors.npz.gz'):
                ax = entry['path'].split('_', 1)[0]
                with np.load(io.BytesIO(raw), allow_pickle=False) as vectors:
                    error = float(np.linalg.norm(vectors['after']-vectors['before']))
                    before = vectors['before']; n = len(before)//2
                    value = float(np.vdot(before[:n], before[:n]).real-np.vdot(before[n:], before[n:]).real)
                expected = signals[ident]['signal']['corrected']['real' if ax == 'cosine' else 'imag']
                c.require(error < 1e-7 and abs(value-expected) < 1e-9, 'SAVED_VECTOR_RECHECK')
            archives.append(dict(candidate=ident, **entry))
        for ax in ('cosine', 'sine'):
            price = result['prices'][ax]
            c.require(price['compiler'] == c.COMPILER and price['preparation_hash'] == result['preparation_hash'], 'COMPILER_PREPARATION')
            c.require(all(type(v) is int and v >= 0 for v in price['metrics'].values()) and price['metrics']['rz_count'] > 0, 'INTEGER_PRICE')
        complete.append(result)
        print('Verified complete native pair', len(complete), ident, flush=True)
    c.require(len(complete)+len(missing) == len(selected), 'EXACT_SELECTED_PARTITION')
    consumed = controls['COST']['counts']
    draft = standard_fresh_budget(missing, consumed, plan['cost_cap_template'])
    c.require(draft['exceeds_fixed_internal_budget'], 'FRESH_BUDGET_ASSESSMENT_CHANGED')
    native_prices = {c.cid(r['candidate']):dict(cosine=r['prices']['cosine']['metrics']['rz_count'],
        sine=r['prices']['sine']['metrics']['rz_count'], kind='AUDITED_COMPLETE_PAIR_FROM_INTERRUPTED_STAGE',
        result_sha256=c.sha(private/'COST/output'/c.cid(r['candidate'])/'result.json')) for r in complete}
    all_prices = dict(saved, **native_prices)
    rows = []
    for r in plan['rows']:
        ident = c.cid(r)
        eligible = [e['epsilon'] for e in endpoints if c.cid(e['candidate']) == ident and e['status'] == 'ELIGIBLE']
        rows.append(dict(r, id=ident, signal='AVAILABLE', eligibility='KNOWN', eligible_epsilons=eligible,
            cost_status='NOT_REQUIRED_INELIGIBLE' if not eligible else 'AVAILABLE_UNACCOUNTED' if ident in all_prices else 'MISSING_COMPILED_PAIR',
            cost_kind=all_prices.get(ident, {}).get('kind'), fixed_total_cost_accounted=False))
    eligible_ids = {r['id'] for r in rows if r['eligible_epsilons']}
    c.require(eligible_ids-set(all_prices) == {c.cid(r) for r in missing}, 'EXACT_MISSING_ELIGIBLE')
    output.mkdir(parents=True)
    stop = dict(status='MANDATORY_STOP', reason='COST_PROCESS_ABSENT_WITHOUT_SUCCESS_RECEIPT',
        observed_at_utc=observed, exact_failure_time=None, cause='UNDETERMINED',
        SOURCE=plan['SOURCE'], grant_id=controls['COST']['grant']['grant_id'],
        actual_calls=consumed, reservations_nonrefundable=True, retry=False, resume=False,
        all_original_processes_absent_at_two_observations=True, original_native_exit_codes=None,
        original_parent_join_reap_receipt_available=False, new_scientific_calls_in_this_audit=0)
    # This is a new observation of interruption, never a fabricated native exit.
    c.save(private/'COST/STOP.json', stop)
    c.save(private/'EXTERNAL_INTERRUPTION_OBSERVED.json', dict(stop, observations=observations))
    for name in ('SUCCESSOR_DRIVER_STOP.json', 'FINAL_RESULT_DRIVER_STOP.json', 'LOCAL_COMMIT_DRIVER_STOP.json'):
        c.save(private/name, dict(status='MANDATORY_STOP', reason='EXTERNAL_PROCESS_ABSENCE_OBSERVED',
            observed_at_utc=observed, exact_failure_time=None, retry=False, scientific_calls=0))
    protected = []
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], cwd=c.ROOT, text=True).splitlines()
    for rel in paths:
        if rel.startswith(('artifacts/resource_applicability/track_a_pf_common_H2_results/',
                           'artifacts/resource_applicability/track_a_pf_common_H4_results/',
                           'artifacts/resource_applicability/track_a_pf_common_results/')):
            c.require((c.ROOT/rel).read_bytes() == c.gitbytes(BASE, rel), 'OLD_RESULT_BYTES')
            protected.append(dict(path=rel, sha256=c.sha(c.ROOT/rel)))
    old_inventory = c.read(c.ROOT/'artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/original_control_inventory.json')
    for rel, sha in old_inventory.items():
        c.require(c.sha(metadata.parent/'METADATA63_run02'/rel) == sha, 'OLD_FAILED_CONTROL_BYTES')
    old_controls = [c.seal(private/rel, rel) for rel in ['plan_APPROVED.json', 'approval_receipt.json']+
        [s+'/'+n for s in controls for n in ('grant.json', 'CONSUMED.json', 'reserved_counts.json')]]
    summary = dict(status='COST_INTERRUPTED_MANDATORY_STOP', scientific_SOURCE=plan['SOURCE'],
        metadata_SOURCE=c.read(metadata/'METADATA/completion.json')['SOURCE'],
        registered=276, metadata_observations=63, old_signal_checks=52, new_raw_signals_in_metadata=11,
        common_new_signals=102, signal_records=276, qualification_endpoints=1380,
        eligible_candidates=len(eligible_ids), complete_native_pairs=len(complete), complete_native_axes=2*len(complete),
        reused_eligible_native_prices=len(eligible_ids & set(saved)),
        acquisition_requirements_available=sum(r['cost_status'] != 'MISSING_COMPILED_PAIR' for r in rows),
        missing_compiled_pairs=len(missing), compile_reservations=consumed['wrapper_compile'],
        full_H6_final_comparison='UNRESOLVED', full_H246_final_comparison='UNRESOLVED',
        H2_H4_historical_completed_candidates=312, H2_H4_recomputed=False,
        accepted_partial_archives=len(archives), incomplete_unaccepted_files=len(orphan_files),
        all_workers_exit0_reaped=None, scientific_processes_present=False,
        original_failure_time=None, original_failure_cause='UNDETERMINED',
        new_scientific_calls=0, PR_sampling=0, GitHub_push=False,
        fixed_budget_extension_required_for_standard_fresh_method=True)
    values = {'summary.json':summary, 'COST_STOP.json':stop, 'process_absence_observations.json':observations,
        'candidate_status_276.json':rows, 'unresolved_candidates_22.json':missing,
        'raw_archive_inventory.json':archives, 'unaccepted_partial_files.json':orphan_files,
        'native_gate_maxima.json':maxima, 'fresh_completion_contract_UNAPPROVED.json':draft,
        'old_results_preserved.json':dict(base=BASE, files=protected), 'original_control_inventory.json':old_controls,
        'prior_failed_controls_preserved.json':dict(status='PASS', files=old_inventory),
        'resource_receipt.json':c.read(private/'COST/resource_receipt.json')}
    for name, value in values.items():
        c.save(output/name, value)
    for name, value in [('controls_and_ledgers.json.gz', controls), ('complete_native_results_187.json.gz', complete),
                        ('signal_records_276.json.gz', signals), ('qualification_endpoints_1380.json.gz', endpoints),
                        ('available_prices_UNACCOUNTED.json.gz', all_prices)]:
        with (output/name).open('xb') as f:
            f.write(gzip.compress(c.encoded(value), compresslevel=1, mtime=0))
    c.save(output/'MANDATORY_STOP.json', dict(status='MANDATORY_STOP', reason='Fixed internal budget insufficient for unchanged fresh paired completion',
        old_grant_reused=False, next_scientific_stage_started=False, GitHub_push=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--private', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--output', default=OUTPUT)
    a = p.parse_args()
    audit(a.private, a.metadata, a.output)
