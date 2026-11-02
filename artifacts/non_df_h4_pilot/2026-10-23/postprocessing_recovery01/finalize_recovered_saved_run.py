#!/usr/bin/env python3
"""Observe the existing orphan H4 worker, audit saved data and commit locally.

Stdlib only. No scientific dispatch, process signals, original run writes, or
claim reuse. Missing original supervisor exit/reap receipts remain unverified.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
spec = importlib.util.spec_from_file_location('original_saved_finalizer', BASE/'finalize_saved_run.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
RECEIPT = Path('/tmp/h4-recovery-859ce9f6.receipt.json')
WORKER_PID = 71955
WORKER_START = 5479158


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid, expected_start=None):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return dict(pid=pid, present=False, active=False, reused=False)
    start = int(fields[19])
    reused = expected_start is not None and start != expected_start
    return dict(pid=pid, present=True, start_ticks=start, state=fields[0], ppid=int(fields[1]),
                reused=reused, active=not reused and fields[0] != 'Z')


def observation(pack):
    cases = {}
    for case in f.CASES:
        folder = pack/'run'/case
        cases[case] = dict(joined=len(list((folder/'trajectories').glob('*/price_join.json'))),
                          events=len(list((folder/'trajectories').glob('*/events.json'))),
                          status=f.read(folder/'status.json') if (folder/'status.json').exists() else None)
    return dict(UTC=now(), cases=cases, worker=identity(WORKER_PID, WORKER_START))


def wait_for_worker(pack, log_path):
    with log_path.open('x') as stream:
        while True:
            value = observation(pack)
            stream.write(json.dumps(value, sort_keys=True)+'\n')
            stream.flush()
            print(value['UTC'], 'EXISTING_WORKER_ONLY', value['cases']['df_native_PR']['joined'], flush=True)
            f.check(not value['worker']['reused'], 'WORKER_PID_REUSED_NO_SCIENTIFIC_RETRY')
            if not value['worker']['active']:
                return value
            time.sleep(30)


def recovered_summary(pack):
    rows = []; missing = []; states = {}
    for case in f.CASES:
        folder = pack/'run'/case
        complete = all((folder/name).exists() for name in
                       ['result.json', 'status.json', 'validation.json', 'precision_rows.json'])
        states[case] = f.read(folder/'status.json')['status'] if complete else 'MISSING_OR_FAILED'
        if not complete or states[case] != 'COMPLETE_FIXED_CASE':
            missing.append(case)
            continue
        rows.extend(dict(case=case, **row) for row in f.read(folder/'precision_rows.json'))
    return dict(status='FOUR_CASES_COMPLETE' if not missing else 'PARTIAL_SAVED_CASES',
                rows=rows, missing_cases=missing, case_states=states,
                original_pipeline_completion_verified=False, scientific_superiority_claimed=False)


def commit_saved_recovery():
    grant = f.read(f.PACK/'grant.json')
    claims = Path(grant['claim_directory'])
    prefixes = [str(BASE.relative_to(f.ROOT)), str(claims.relative_to(f.ROOT))]
    f.check_repository(f.ROOT, f.SOURCE, f.BRANCH, prefixes)
    f.check(not f.same_process(71878, 5478336), 'ORIGINAL_SUPERVISOR_STILL_ACTIVE')
    old_claim = f.read(BASE/'finalizer_claim.json')
    f.check(not (Path('/proc')/str(old_claim['pid'])).exists(), 'OLD_FINALIZER_PID_PRESENT_NO_DUPLICATE')
    f.save(HERE/'claim.json', dict(pid=os.getpid(), UTC=now(), SOURCE=f.SOURCE,
        previous_claim_preserved=True, scientific_calls=0, scientific_retry=False, push=False))
    terminal = wait_for_worker(f.PACK, HERE/'observations.jsonl')
    f.check_repository(f.ROOT, f.SOURCE, f.BRANCH, prefixes)
    f.check(not identity(WORKER_PID, WORKER_START)['active'], 'WORKER_STILL_ACTIVE_NO_COMMIT')
    f.save(HERE/'terminal_observation.json', dict(observation=terminal,
        original_supervisor=identity(71878, 5478336), original_finalizer=identity(old_claim['pid']),
        original_native_STOP_present=(f.PACK/'STOP.json').exists(),
        original_stage_STOP_present=(f.PACK/'run/STOP.json').exists(),
        all_children_reaped=None, worker_exit_code=None,
        reason='Original supervisor disappeared; an observer cannot recover its exit/reap receipts.',
        scientific_calls=0, scientific_retry=False, original_run_files_written=False))
    f.save(HERE/'recovered_case_summary.json', recovered_summary(f.PACK))
    try:
        result = subprocess.run([sys.executable, '-B', str(HERE/'verify_saved_recovery.py'), '--runtime-files'],
                                capture_output=True, text=True, check=True)
        verification = json.loads(result.stdout)
    except Exception as exc:
        verification = dict(status='SAVED_DATA_VERIFICATION_FAILED', error=str(exc),
            traceback=traceback.format_exc(), stderr=getattr(exc, 'stderr', None),
            original_native_completion_verified=False, all_children_reaped=None, scientific_retry=False)
    f.save(HERE/'verification.json', verification)
    records, rows = f.collect_cases(f.PACK)
    summary = dict(schema='h4_saved_recovery_summary_v1', SOURCE=f.SOURCE, branch=f.BRANCH,
        recorded_UTC=now(), scientific_STOP=None,
        finalization_status='ORIGINAL_SUPERVISOR_LOST_NATIVE_STOP_MISSING',
        verification_status=verification['status'],
        verification_error='Original supervisor completion/exit/reap evidence is missing; no native PASS.',
        cases=records, precision_rows=len(rows), null_N_G_rows=sum(r['total_RZ'] is None for r in rows),
        scope='H4 parallel research only; no integration into original chat research',
        original_native_completion_verified=False, all_children_reaped=None,
        detailed_scientific_review=False, push=False, scientific_retry=False, mandatory_STOP=True)
    f.save(HERE/'result_summary.json', summary)
    with (HERE/'precision_table.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case','epsilon_sig','N_cosine','N_sine','N_total','G_mean','accounting_usable'])
        for row in rows:
            shots = row['shots']; cost = row['total_RZ']
            writer.writerow([row['case'],row['epsilon_sig'],shots['axes']['cosine']['shots'],
                shots['axes']['sine']['shots'],shots['N_total'],None if cost is None else cost['mean'],shots['usable']])
    text = f.render_report(summary)
    text = text.replace('監査はSOURCE/runtime・入力・数値gate・event/価格hash・固定予算・子孫回収・旧証拠保全を確認する。',
        '保存データ監査はSOURCE/runtime・入力・数値gate・event/価格hash・固定予算・旧証拠保全を確認する。'
        '元のcase監視によるexit/reapと正常STOPは未確認。')
    text = text.replace('`verification.json`、`result_summary.json`、`precision_table.csv`、',
        '本directoryの`verification.json`、`result_summary.json`、`precision_table.csv`、')
    text = text.replace('`/tmp/h4-finalize-859ce9f6.receipt.json`', '`/tmp/h4-recovery-859ce9f6.receipt.json`')
    text += ('\n## 監視消失後の保存処理\n\n'
        '元の監視・後処理はDF PRの161件前後で記録が途絶え、確認時に消失していた。原因は未特定。'
        '元のDF PR workerのみを継続観測し、新規科学計算は起動していない。'
        '元runのsummary/STOP/ledger/marker/claimは新造・上書きしていない。'
        'このdirectoryの保存データ検査は、元の入力・SOURCE・runtime・event/価格・予算・数値gate検査を維持するが、'
        '元監視のexit codeとreapを確認できないため、通常実行の完全受理PASSとは区別する。'
        '上記の保存物は元runに実在するもののみを指し、欠けたSTOPを存在すると扱わない。'
        '検査JSON・表・inventoryは本directory内。commit receiptは'
        '`/tmp/h4-recovery-859ce9f6.receipt.json`。\n')
    with (HERE/'RESULTS.md').open('x') as stream:
        stream.write(text)
    (BASE/'README.md').write_text('# H4非DF/DF固定pilot\n\n'
        '[監視消失後の保存結果・検証の限界](postprocessing_recovery01/RESULTS.md)。'
        '元の監視exit/reap/STOPが欠けるため通常実行の完全受理PASSにはしない。'
        'H4専用branchのみ、元run保全、科学retry・追加sampling・pushなし。mandatory STOP。\n')
    inventory = {}
    for directory in [BASE, claims]:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                inventory[str(path.relative_to(f.ROOT))] = dict(symlink=str(path.readlink()))
            elif path.is_file():
                inventory[str(path.relative_to(f.ROOT))] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    f.save(HERE/'result_inventory.json', dict(files=inventory, inventory_self_excluded=True,
        SOURCE=f.SOURCE, original_run_files_written=False))
    f.check_repository(f.ROOT, f.SOURCE, f.BRANCH, prefixes)
    subprocess.run(['git','-C',str(f.ROOT),'add','-f','--',*prefixes], check=True)
    f.check_repository(f.ROOT, f.SOURCE, f.BRANCH, prefixes, staged=True)
    subprocess.run(['git','-C',str(f.ROOT),'diff','--cached','--check','--','*.py','*.md'], check=True)
    subprocess.run(['git','-C',str(f.ROOT),'-c','user.name=AbeHiromu','-c',
        'user.email=AbeHiromu@users.noreply.github.com','commit','--quiet','-m',
        'Preserve H4 saved results and explicit supervisor evidence gap'], check=True)
    receipt = dict(status='COMMITTED_SAVED_RESULTS_WITH_SUPERVISION_GAP',
        commit=f.git(f.ROOT,'rev-parse','HEAD'), branch=f.BRANCH,
        verification_status=verification['status'], worktree_clean=not bool(f.git(f.ROOT,'status','--porcelain')),
        original_native_completion_verified=False, all_children_reaped=None,
        scientific_retry=False, push=False, mandatory_STOP=True)
    f.save(RECEIPT, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-and-commit', action='store_true', required=True)
    parser.parse_args()
    try:
        commit_saved_recovery()
    except Exception:
        failure = dict(status='SAVED_RECOVERY_FAILED_NO_SCIENTIFIC_RETRY', traceback=traceback.format_exc())
        if not (HERE/'FAILED.json').exists():
            f.save(HERE/'FAILED.json', failure)
        print(json.dumps(failure), flush=True)
        raise
