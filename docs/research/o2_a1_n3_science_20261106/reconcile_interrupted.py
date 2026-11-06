"""Preserve an interrupted campaign using durable journals only.

No bundle decoding, state access, fit, RNG, gate/Born, compile, calibration or N.
The scientific driver and its frozen SOURCE are not modified or invoked.
Run once after observing that the parent and all recorded workers have exited.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from trottertracks.algorithm_codesign import a1_n3_campaign as c


def preserve(root):
    out = root / c.OUTPUT
    observation = json.loads((out / 'INTERRUPTION_OBSERVATION.json').read_text())
    assert not Path('/proc', str(observation['parent_pid'])).exists()
    assert all(not Path('/proc', str(pid)).exists()
               for pid in observation['worker_pids_from_journals'])
    assert not (out / 'STOP.json').exists()
    assert not (out / 'ACCOUNTING.json').exists()
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir() if p.is_file()}
    c.save(out / 'RECOVERY_INPUT_HASHES.json', original)
    shutil.copyfile('/tmp/a1-n3-science-run-20261106.log', out / 'RUN.txt')
    shutil.copyfile('/tmp/a1-n3-science-invocation-20261106.json', out / 'INVOCATION.json')
    shutil.copyfile('/tmp/a1-n3-science-SOURCE-verified-20261106.json',
                    out / 'SOURCE_PUBLICATION_VERIFIED.json')

    # Reuse only the fixed metadata/cost reconciliation function.
    methods = c.reconcile(out)
    accounting = json.loads((out / 'ACCOUNTING.json').read_text())
    accounting['reconciliation_process_peak_RSS_bytes'] = accounting['parent_peak_RSS_bytes']
    accounting['parent_peak_RSS_bytes'] = None
    accounting['parent_peak_RSS_status'] = 'UNKNOWN: scientific parent exited without finalization'
    accounting['worker_RSS_status'] = 'recorded lower bounds only; interrupted tail highwaters UNKNOWN'
    accounting['campaign_total_classical_cost_status'] = 'UNKNOWN: interrupted worker costs and parent peak/wall duration missing'
    accounting['reconciled_after_process_loss'] = True
    c.save(out / 'ACCOUNTING.json', accounting)
    counts = json.loads((out / 'CALL_COUNTS.json').read_text())
    counts['counter_semantics'] = 'durable function-entry records, not a certificate of call completion'
    counts['termination_cause'] = 'UNKNOWN'
    counts['new_scientific_calls_during_reconciliation'] = 0
    c.save(out / 'CALL_COUNTS.json', counts)

    planned, fixed, complete, tapes = {}, {}, {}, {}
    for line in (out / 'JOURNAL.jsonl').read_text().splitlines():
        row = json.loads(line)
        target = {'TRIAL_PLANNED': planned, 'PATH_FIXED': fixed,
                  'TRIAL_COMPLETE': complete}.get(row['event'])
        if target is not None:
            assert row['ticket'] not in target
            target[row['ticket']] = row
    for line in (out / 'engine_internal_words.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert row['ticket'] not in tapes
        tapes[row['ticket']] = row
    assert set(planned) == set(fixed) == set(tapes)
    assert all(len(r['path_words']) <= 64 and type(r['measurement_word']) is int
               and 0 <= r['measurement_word'] < 2**53
               and all(type(w) is int and 0 <= w < 2**53 for w in r['path_words'])
               for r in tapes.values())
    bits, priced, starts, last_call = {}, {}, {}, {}
    for path in sorted(out.glob('worker_*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ticket = row['ticket']
            assert ticket in planned
            if row['event'] == 'WORKER_STARTED':
                assert ticket not in starts
                starts[ticket] = row
            elif row['event'] == 'CALL':
                last_call[ticket] = row['name']
            elif row['event'] == 'BIT_SAVED':
                bits[ticket] = row
                assert row['record']['bit'] in (0, 1)
                assert row['domain'] == planned[ticket]['domain']
            elif row['event'] == 'PRICE_SAVED':
                priced[ticket] = row
                assert row['metrics']['compiler_settings_hash']
    assert set(complete) == set(priced)
    assert set(starts) == set(planned)
    assert set(bits) == set(priced)
    assert all(complete[t]['rz_count'] == priced[t]['metrics']['rz_count'] for t in complete)
    pending = [{'ticket': t, 'domain': starts[t]['domain'],
                'trial': starts[t]['trial'], 'last_logged_call': last_call.get(t),
                'measurement_bit': 'UNKNOWN', 'price': 'UNKNOWN',
                'classical_cost': 'UNKNOWN'} for t in sorted(set(starts) - set(bits))]
    c.save(out / 'INTERRUPTED_TRIALS.json', pending)
    partial_y = [r for r in bits.values() if r['domain']['axis'] == 'Y']
    c.save(out / 'A1_Y_train_PARTIAL.json', {
        'status': 'PARTIAL_NOT_FITTED', 'quota_per_stratum': 64,
        'records': sorted(partial_y, key=lambda r: (r['domain']['cell'],
                                                   int(r['record']['trial_id']))),
        'fit_calls': 0, 'coefficients': 'UNKNOWN'})
    c.save(out / 'CAMPAIGN_RESULT.json', {
        'status': 'INTERRUPTED_PROCESS_LOST', 'methods': methods,
        'precision_achieved': 'UNKNOWN for all four methods and both axes; no production',
        'CV_improvement': 'UNDETERMINED; B1-OL comparison not acquired',
        'expected_cost_guarantee': 'UNKNOWN', 'truth_error_measured': False,
        'finalized_from_journals': True})
    c.save(out / 'CORRECTNESS.json', {
        'scope': 'saved journals/bytes/metadata/cost consistency only; no new scientific evaluation',
        'duplicate_trial_check': 'PASS', 'bit_before_price_check': 'PASS',
        'measurement_price_record_match': 'PASS', 'parent_worker_price_match': 'PASS',
        'path_word_bounds_and_cap': 'PASS', 'one_tape_per_trial': 'PASS',
        'trials_planned': len(planned), 'bits_saved': len(bits),
        'prices_saved': len(priced), 'interrupted_without_bit': len(pending),
        'saved_bits_per_axis_stratum': dict(Counter(
            r['domain']['axis'] + ':' + r['domain']['cell'] for r in bits.values())),
        'source_and_fixed_input_hashes': 'see PRESERVATION.json',
        'additional_scientific_calls': 0})
    assert all(hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
               for name, digest in original.items())
    c.save(out / 'STOP.json', {
        'status': 'MANDATORY_STOP', 'campaign_status': 'INTERRUPTED_PROCESS_LOST',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'recorded_by': 'journal-only postprocessing after confirmed process loss; not driver finalizer',
        'reason': {'termination_cause': 'UNKNOWN', 'exit_code': 'UNKNOWN',
                   'facts': 'parent and all 77 journaled workers absent; exec session unavailable; no original STOP/exception'},
        'no_retry': True, 'resumed_scientific_calls': 0,
        'additional_campaign_authorized': False,
        'condition_seed_kernel_changes': False,
        'next_actions': 'result publication and remote content verification only'})
    print(c.dumps({'status': 'MANDATORY_STOP', 'counts': counts,
                   'methods': {k: {key: value for key, value in v.items()
                                  if key != 'stages'} for k, v in methods.items()}}))


if __name__ == '__main__':
    preserve(Path(__file__).resolve().parents[3])
