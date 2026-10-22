#!/usr/bin/env python3
"""Read-only provenance/schema checks and literal evidence transcription.

No scientific-module import, signal/allowance/normalization evaluation, shot
calculation, cost accounting, event statistics, or changes to original evidence.
Only hashes, schemas, stored statuses, log records and source text are inspected.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OLD = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-execution-foundation-20261023')
BASE = OLD/'artifacts/non_df_h4_pilot/2026-10-23'
PACK = BASE/'run01'
COMMIT = '6892e95ffdfc78e75425e7432e9e4ccdb9958df1'
SOURCE = '0c329b30a81accb35c212be8defa66fee0dd0f12'
CASES = ('pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR')
AXES = ('cosine','sine')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def check(value, message):
    if not value:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(['git','-C',str(OLD),*args])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1024*1024):
            h.update(part)
    return h.hexdigest()


def preservation():
    source = read(PACK/'SOURCE.json')
    scientific = {}
    for name, expected in source['files'].items():
        actual = sha(OLD/name)
        check(actual == expected, 'SCIENTIFIC_SOURCE_CHANGED:'+name)
        scientific[name] = actual
    grant = read(PACK/'grant.json')
    prefixes = [str(BASE.relative_to(OLD)), str(Path(grant['claim_directory']).relative_to(OLD))]
    entries = git('ls-tree','-r','-z',COMMIT,'--',*prefixes).split(b'\0')
    result = {}
    for entry in entries:
        if not entry:
            continue
        meta, name = entry.split(b'\t',1)
        mode, kind, expected = meta.decode().split()
        check(mode == '100644' and kind == 'blob', 'UNEXPECTED_EVIDENCE_FILE_MODE')
        path = OLD/name.decode()
        check(not path.is_symlink(), 'EVIDENCE_SYMLINK')
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        check(actual == expected, 'COMMITTED_RESULT_CHANGED:'+name.decode())
        result[name.decode()] = dict(git_blob=actual, sha256=hashlib.sha256(data).hexdigest())
    check(sha(PACK/'SOURCE.json') == grant['source_manifest_sha256'], 'SOURCE_MANIFEST_SEAL')
    check(sha(PACK/'approval.json') == grant['approval_sha256'], 'APPROVAL_SEAL')
    return dict(scientific_SOURCE=SOURCE, evidence_commit=COMMIT,
        source_files=scientific, evidence_and_claim_files=result,
        source_file_count=len(scientific), evidence_file_count=len(result))


def remote_check():
    remote = read(HERE/'github_retrieval.json')
    branch = json.loads(remote['branch_result']['structuredContent']['content'])
    commit = json.loads(remote['commit_result']['structuredContent']['content'])
    check(branch['commit']['sha'] == commit['sha'] == COMMIT, 'GITHUB_COMMIT_MISMATCH')
    check(commit['parents'][0]['sha'] == SOURCE, 'SCIENTIFIC_PARENT_MISMATCH')
    check(git('rev-parse',COMMIT+'^{tree}').decode().strip() == commit['tree']['sha'], 'GITHUB_COMMIT_TREE')
    comparisons = []
    for item in remote['files']:
        check(not item.get('isError',False), 'REMOTE_FILE_FETCH_FAILED')
        data = item['content'].encode()
        # fetch_file represents CSV CRLF using LF. Compare normalized UTF-8
        # text and separately verify the reported canonical Git blob SHA.
        committed = git('show',item['ref']+':'+item['path'])
        oid = hashlib.sha1(b'blob '+str(len(committed)).encode()+b'\0'+committed).hexdigest()
        check(oid == item['sha'], 'REMOTE_BLOB_SHA:'+item['path'])
        check(data.replace(b'\r\n',b'\n') == committed.replace(b'\r\n',b'\n'), 'REMOTE_CONTENT:'+item['path'])
        check((OLD/item['path']).read_bytes() == committed, 'LOCAL_PINNED_CONTENT:'+item['path'])
        comparisons.append(dict(path=item['path'],ref=item['ref'],git_blob=oid,
            byte_equal=data == committed, newline_normalized_equal=True))
    return comparisons


def source_excerpt_inventory():
    selections = {
        'src/trotterlib/fixed_signal_accounting.py':[(26,80)],
        'src/trotterlib/h2_non_df_pilot.py':[(236,293)],
        'src/trotterlib/h4_audited_evaluation.py':[(43,58),(105,135)],
        'src/trotterlib/h4_pilot_science.py':[(122,171)],
        'src/trotterlib/h4_pilot_execution.py':[(54,92),(106,146)],
        'src/trotterlib/fixed_signal_evaluation.py':[(94,105)],
        'src/trotterlib/pauli_signal_adapter.py':[(193,205)],
        'src/trotterlib/isolated_qiskit_supervisor.py':[(255,264),(295,369)],
        'src/trotterlib/rte.py':[(217,245),(1450,1494)],
    }
    output = []
    for name, ranges in selections.items():
        text = git('show',SOURCE+':'+name).decode()
        ast.parse(text)  # Parse only; never execute scientific source.
        lines = text.splitlines()
        output.extend(dict(path=name,SOURCE=SOURCE,start=lo,end=hi,
            text='\n'.join(f'{i}: {lines[i-1]}' for i in range(lo,min(hi,len(lines))+1))) for lo,hi in ranges)
    return output


def precision_inventory():
    plan = read(PACK/'contract.json')
    states = []; causes = Counter(); cases = {}
    for case in CASES:
        folder = PACK/'run'/case
        result = read(folder/'result.json'); evidence = read(folder/'numerical_evidence.json')
        rows = read(folder/'precision_rows.json')
        check([r['epsilon_sig'] for r in rows] == plan['precision']['epsilon_sig'], 'FIVE_SAVED_PRECISIONS')
        for row in rows:
            check(row['shots']['N_total'] is None and row['total_RZ'] is None and not row['shots']['usable'], 'PRESERVE_NULLS')
            check(row['shots']['evidence_kind'] == 'EMPIRICAL' and row['shots']['certificate_verified'] is False, 'EMPIRICAL_STATUS')
            axes = {}
            for axis in AXES:
                saved = row['shots']['axes'][axis]
                check(saved['status'] == 'UNDETERMINED' and saved['shots'] is None, 'STORED_AXIS_STATE')
                # Read the saved sign/status only. No subtraction, ratio, log,
                # bound or shot evaluation is performed by this investigation.
                if saved['headroom'] > 0:
                    check(saved['log_shot_bound'] is not None, 'EXPECTED_PRIOR_POSITIVE_HEADROOM_PATH')
                    cause = 'PRECISION_ACCOUNTING_RATIO_GUARD'
                else:
                    check(saved['log_shot_bound'] is None, 'EXPECTED_PRIOR_NONPOSITIVE_HEADROOM_PATH')
                    cause = 'BASE_ACCOUNTING_ALLOWANCE_OVERLAP_UNDETERMINED'
                causes[cause] += 1
                axes[axis] = dict(saved=saved, cause_from_static_source_and_saved_status=cause)
            states.append(dict(case=case,epsilon_sig=row['epsilon_sig'],axes=axes,
                stored_N_total=None,stored_G=None,stored_status='UNDETERMINED'))
        cases[case] = dict(normalization=result['normalization'],signed_bias=result['signal']['signed_bias'],
            numerical_evidence=evidence,confidence_allocation_as_saved=dict(alpha_union=rows[0]['shots']['alpha_union'],
                alpha_axes={a:rows[0]['shots']['axes'][a]['alpha'] for a in AXES}),
            all_input_values_present=True,source_paths=dict(result=str((folder/'result.json').relative_to(OLD)),
                evidence=str((folder/'numerical_evidence.json').relative_to(OLD))))
    csv_rows = list(csv.DictReader(io.StringIO((BASE/'postprocessing_recovery01/precision_table.csv').read_text())))
    check(len(csv_rows) == len(states) == 20, 'SAVED_20_ROWS')
    check(all(row['N_total'] == row['G_mean'] == '' and row['accounting_usable'] == 'False' for row in csv_rows), 'SAVED_CSV_NULLS')
    return dict(rows=states,cases=cases,axis_counts=dict(causes),
        method='Literal transcription, stored sign/status checks, static control-flow attribution only.',
        no_shot_or_G_recalculation=True,no_new_numerical_diagnostic=True)


def monitor_inventory():
    progress = [json.loads(line) for line in (PACK/'progress_observations.jsonl').read_text().splitlines()]
    finalizer = (BASE/'postprocessing_recovery01/original_finalizer_log_preserved.log').read_text().splitlines()
    names = ['STOP.json','summary.json','run/STOP.json','STOP.request']
    absence = {name: not (PACK/name).exists() for name in names}
    cases = {}
    for case in CASES:
        folder = PACK/'run'/case
        cases[case] = dict(launch=read(folder/'launch.json'),ready=read(folder/'ready.json'),
            status=read(folder/'status.json'),process_record_present=(folder/'process.json').exists(),
            failed_record_present=(folder/'FAILED.json').exists(),
            stdout_bytes=(folder/'stdout.log').stat().st_size,stderr_bytes=(folder/'stderr.log').stat().st_size)
    return dict(original_progress_records=len(progress),last_original_progress=progress[-1],
        original_finalizer_log_records=len(finalizer),last_original_finalizer_line=finalizer[-1],
        input_stage_native_completion=read(PACK/'input_run/STOP.json'),
        input_stage_native_process=read(PACK/'input_run/input_generation/process.json'),
        original_case_ledger=[json.loads(x) for x in (PACK/'run/ledger.jsonl').read_text().splitlines()],
        cases=cases,original_absence=absence,
        recovery_observation=read(BASE/'postprocessing_recovery01/recovery_authorization.json'),
        terminal_observation=read(BASE/'postprocessing_recovery01/terminal_observation.json'),
        original_supervisor_disappearance_cause='UNKNOWN',
        original_native_completion_verified=False,worker_reap_proven=None,
        input_stage_completion_does_not_prove_case_stage_completion=True,
        reconstructed_native_records_created=False)


def main():
    global OLD, BASE, PACK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after',action='store_true',help='Compare immutable input hashes to the first check.')
    parser.add_argument('--source-worktree',type=Path,default=OLD,help='Unmodified checkout of the original evidence commit.')
    args = parser.parse_args()
    OLD = args.source_worktree.resolve()
    BASE = OLD/'artifacts/non_df_h4_pilot/2026-10-23'
    PACK = BASE/'run01'
    preserved = preservation()
    if args.after:
        check(preserved == read(HERE/'preservation_before.json'), 'INPUTS_CHANGED_DURING_INVESTIGATION')
        save(HERE/'preservation_after.json',preserved)
        print('PASS: original source, evidence and claim bytes unchanged')
        return
    save(HERE/'preservation_before.json',preserved)
    remote = remote_check()
    save(HERE/'source_excerpts.json',source_excerpt_inventory())
    precision = precision_inventory(); save(HERE/'precision_state_inventory.json',precision)
    monitor = monitor_inventory(); save(HERE/'monitor_inventory.json',monitor)
    save(HERE/'verification.json',dict(status='PASS_READ_ONLY_STRUCTURE_ONLY',
        checked_UTC=datetime.now(timezone.utc).isoformat(), evidence_commit=COMMIT,scientific_SOURCE=SOURCE,
        github_file_comparisons=remote,scientific_source_files_checked=preserved['source_file_count'],
        original_evidence_and_claim_files_checked=preserved['evidence_file_count'],
        precision_rows_transcribed=20,axes_transcribed=40,axis_static_causes=precision['axis_counts'],
        original_N_G_null_preserved=True,original_native_completion_verified=False,
        scientific_modules_executed=False,numerical_reanalysis=False,shot_recalculation=False,
        G_reaccounting=False,new_science_calls=0,original_records_written=False,push=False,mandatory_STOP=True))
    print(json.dumps(dict(status='PASS_READ_ONLY_STRUCTURE_ONLY',axis_static_causes=precision['axis_counts'],
        source_files=preserved['source_file_count'],evidence_files=preserved['evidence_file_count'])))


if __name__ == '__main__':
    main()
