#!/usr/bin/env python3
"""Wait for existing H4 STOP, verify saved evidence, report and commit locally.

No science imports, dispatch, sampling, compilation, retries or process signals.
All writes stay in this H4 artifact/claim scope, apart from the /tmp receipt.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PACK = BASE/'run01'
SOURCE = '0c329b30a81accb35c212be8defa66fee0dd0f12'
BRANCH = 'codex/non-df-h4-execution-foundation-20261023'
CASES = ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR')
RECEIPT = Path('/tmp/h4-finalize-859ce9f6.receipt.json')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def allowed(path, prefixes):
    return any(path == prefix or path.startswith(prefix+'/') for prefix in prefixes)


def check_repository(root, source, branch, prefixes, *, staged=False):
    check(git(root, 'rev-parse', 'HEAD') == source, 'SOURCE_HEAD_CHANGED_NO_COMMIT')
    check(git(root, 'branch', '--show-current') == branch, 'BRANCH_CHANGED_NO_COMMIT')
    check(not git(root, 'diff', '--name-only'), 'TRACKED_WORKTREE_CHANGES_NO_COMMIT')
    staged_names = git(root, 'diff', '--cached', '--name-only').splitlines()
    check(not staged_names or staged, 'PREEXISTING_INDEX_NO_COMMIT')
    check(all(allowed(name, prefixes) for name in staged_names), 'OUT_OF_SCOPE_INDEX_NO_COMMIT')
    untracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--others',
        '--exclude-standard', '-z']).decode().split('\0')
    check(all(allowed(name, prefixes) for name in untracked if name), 'OUT_OF_SCOPE_UNTRACKED_NO_COMMIT')


def same_process(pid, start_ticks):
    path = Path('/proc')/str(pid)/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return int(fields[19]) == start_ticks and fields[0] != 'Z'


def last_observation(pack):
    path = pack/'progress_observations.jsonl'
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-8192))
        lines = stream.read().decode().splitlines()
    try:
        return json.loads(lines[-1])
    except (ValueError, IndexError):
        return {}


def wait_for_existing_run(pack, pid, start_ticks):
    last = None
    while not (pack/'STOP.json').exists():
        check(same_process(pid, start_ticks), 'SUPERVISOR_GONE_WITHOUT_STOP_NO_RETRY')
        observation = last_observation(pack)
        counts = {c:v.get('joined') for c,v in observation.get('cases', {}).items()}
        if counts != last:
            print(datetime.now(timezone.utc).isoformat(), 'WAITING_FOR_EXISTING_RUN', counts, flush=True)
            last = counts
        time.sleep(15)
    while same_process(pid, start_ticks):
        time.sleep(1)
    # The separate read-only monitor writes one final STOP observation then exits.
    deadline = time.monotonic()+65
    while 'STOP' not in last_observation(pack) and time.monotonic() < deadline:
        time.sleep(1)
    check('STOP' in last_observation(pack), 'OBSERVER_NOT_SETTLED_NO_COMMIT')
    return read(pack/'STOP.json')


def collect_cases(pack):
    records = {}; rows = []
    for case in CASES:
        folder = pack/'run'/case
        status = read(folder/'status.json') if (folder/'status.json').exists() else {'status':'MISSING_OR_FAILED'}
        record = dict(status=status['status'])
        if (folder/'result.json').exists():
            result = read(folder/'result.json')
            record.update(samples=result['circuit_cost']['sample_count'], C=result['circuit_cost']['axes'],
                signed_bias=result['signal']['signed_bias'], normalization=result['normalization'],
                primary_shots=result['shots'], primary_G=result['total_RZ'],
                compiled_probability_max_error=result['circuit_cost']['compiled_signal_max_error'])
        if (folder/'precision_rows.json').exists():
            precision = read(folder/'precision_rows.json')
            rows.extend(dict(case=case, **row) for row in precision)
        records[case] = record
    return records, rows


def render_report(summary):
    verified = summary['verification_status'] == 'PASS_SAVED_H4_FIXED_PILOT'
    lines = ['# H4非DF/DF固定pilot：保存結果と技術検証', '',
        '**'+('4方式取得・保存証拠検証PASS。' if verified else '保存証拠の検証は未受理。部分結果と失敗を保全。')+' mandatory STOP。**', '',
        'ユーザーの一括実行承認と「検証結果をまとめてコミット」指示に基づく専用branchの記録。'
        'H4非DFは並行研究側であり、本チャット本来の研究・発表資料へ統合しない。', '',
        '科学SOURCE：`'+summary['SOURCE']+'`。branch：`'+summary['branch']+'`。', '',
        '条件：linear H4・隣接1.00 Å・STO-3G、共通未閾値MO積分、Pauli-JW/DF-native、'
        'DF tol-only1e-8/rank10、Pauli184項のPR prefix27、DF PR prefix1/10＋one-body。'
        '非DF Nalpha=Nbeta=2 sector36最低状態を共有、各表現自身のfull-H reference。'
        'S2/T=.8/q8/δ=.1、PR r4/K4/各256 paired trajectories。'
        'Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし、ordinary-controlled full wrapper RZ、P=0。', '',
        '| case | 保存状態 | paired本数 | cosine RZ平均 ± MC SE | sine RZ平均 ± MC SE |',
        '|---|---|---:|---:|---:|']
    for case, r in summary['cases'].items():
        def cost(axis):
            c = r.get('C', {}).get(axis)
            return f"{c['mean']:.8g} ± {c['standard_error']:.6g}" if c else '未取得'
        lines.append(f"| {case} | {r['status']} | {r.get('samples', '未取得')} | {cost('cosine')} | {cost('sine')} |")
    lines.extend(['', f"保存精度行：{summary['precision_rows']}、N/G null行：{summary['null_N_G_rows']}。",
        'ε=.05/.01/.005/.001/.0005の保存会計をそのまま転記する。精度不適格・数値証拠不足はN/G nullで、'
        'allowance・閾値・候補・標本数を変更していない。総費用Gが未定義の行について方式の総費用順位を主張しない。', '',
        'PRのCは観測256本のfull-wrapper価格平均であり、MC SE/nominal Student-tは有限標本の診断。'
        '希少eventの未観測費用・母平均・一般的優位性は保証しない。event記録はproductionの実戻り値を観測し、'
        '追加抽選・seed replayを行わず、trajectory IDとcos/sin価格をhash joinした。', '',
        '実行基盤の事前検査は既存371＋新規65＝436 PASS、FAIL/skip0、分子計算3件除外。'
        'これは事前の人工/回帰結果であり、今回の実H4保存証拠監査と区別する。'
        '監査はSOURCE/runtime・入力・数値gate・event/価格hash・固定予算・子孫回収・旧証拠保全を確認する。', '',
        '自動後処理のGit範囲・STOP待ち・他作業保全guardは、人工temp repositoryの8検査でPASS。'
        'この後処理検査による科学callは0。詳細は`finalizer_checks.json`とlogに保存した。', '',
        '保存物：`run01/`の原結果・SOURCE/承認/grant/CONSUMED・入力snapshot・実event/価格join・'
        '監視ledger/marker/STOP、`verification.json`、`result_summary.json`、`precision_table.csv`、'
        '`result_inventory.json`。raw・旧SOURCE・旧grant・旧STOPは後から書き換えない。', '',
        '科学retry・追加sampling・再compile・詳細科学レビュー・pushは行っていない。'
        'ローカルcommitの実SHAと成否はGit logおよび`/tmp/h4-finalize-859ce9f6.receipt.json`で確認する。', '',
        '次の作業なし。この保存・ローカルcommitの範囲を終了しmandatory STOP。', ''])
    if not verified:
        lines.extend(['検証未受理の理由：`'+summary.get('verification_error', '未確定')+'`。科学caseは再実行しない。', ''])
    return '\n'.join(lines)


def finish():
    grant = read(PACK/'grant.json')
    claims = Path(grant['claim_directory'])
    prefixes = [str(BASE.relative_to(ROOT)), str(claims.relative_to(ROOT))]
    check_repository(ROOT, SOURCE, BRANCH, prefixes)
    save(BASE/'finalizer_claim.json', dict(pid=os.getpid(), SOURCE=SOURCE, phase='RESERVED_SAVED_ONLY_FINALIZER',
        scientific_calls=0, retry=False, push=False, allowed_prefixes=prefixes))
    stop = wait_for_existing_run(PACK, 71878, 5478336)
    check_repository(ROOT, SOURCE, BRANCH, prefixes)
    verification_error = None
    try:
        completed = subprocess.run([sys.executable, '-B', str(BASE/'verify_saved_run.py'), '--runtime-files'],
            capture_output=True, text=True, check=True)
        verification = json.loads(completed.stdout)
        check(verification['status'] == 'PASS_SAVED_H4_FIXED_PILOT', 'SAVED_VERIFIER_DID_NOT_PASS')
    except Exception as exc:
        verification_error = str(exc)
        verification = dict(status='SAVED_VERIFICATION_FAILED', error=verification_error,
            traceback=traceback.format_exc(), stderr=getattr(exc, 'stderr', None),
            mandatory_STOP=True, scientific_retry=False, push=False)
    save(BASE/'verification.json', verification)
    records, rows = collect_cases(PACK)
    summary = dict(schema='h4_saved_pilot_summary_v1', SOURCE=SOURCE, branch=BRANCH,
        recorded_JST=datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Tokyo')).isoformat(),
        scientific_STOP=stop['status'], verification_status=verification['status'], verification_error=verification_error,
        cases=records, precision_rows=len(rows), null_N_G_rows=sum(r['total_RZ'] is None for r in rows),
        scope='H4 parallel research only; no integration into the original chat research',
        detailed_scientific_review=False, push=False, scientific_retry=False, mandatory_STOP=True)
    save(BASE/'result_summary.json', summary)
    with (BASE/'precision_table.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case','epsilon_sig','N_cosine','N_sine','N_total','G_mean','accounting_usable'])
        for row in rows:
            shots = row['shots'];G = row['total_RZ']
            writer.writerow([row['case'],row['epsilon_sig'],shots['axes']['cosine']['shots'],
                shots['axes']['sine']['shots'],shots['N_total'],None if G is None else G['mean'],shots['usable']])
    with (BASE/'RESULTS.md').open('x') as stream:
        stream.write(render_report(summary))
    # README is our mutable report, not a sealed source/raw/contract record.
    with (BASE/'README.md').open('w') as stream:
        stream.write('# H4非DF/DF固定pilot\n\n[最終結果・検証・限界](RESULTS.md)、'
            '[検証JSON](verification.json)、[原結果](run01/)、[SOURCE](run01/SOURCE.json)、'
            '[inventory](result_inventory.json)。\n\n専用H4 branch内で保存し、主題側の研究資料には統合しない。'
            '科学retry・追加sampling・push・詳細科学レビューなし。mandatory STOP。\n')
    inventory = {}
    for directory in [BASE, claims]:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                inventory[str(path.relative_to(ROOT))] = dict(symlink=str(path.readlink()))
            elif path.is_file():
                inventory[str(path.relative_to(ROOT))] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(BASE/'result_inventory.json', dict(schema='h4_pilot_result_inventory_v1', files=inventory,
        inventory_self_excluded=True, SOURCE=SOURCE, raw_scientific_records_unchanged=True))
    check_repository(ROOT, SOURCE, BRANCH, prefixes)
    subprocess.run(['git','-C',str(ROOT),'add','-f','--',*prefixes], check=True)
    check_repository(ROOT, SOURCE, BRANCH, prefixes, staged=True)
    subprocess.run(['git','-C',str(ROOT),'diff','--cached','--check','--','*.py','*.md'], check=True)
    message = 'Record verified H4 fixed pilot results and mandatory STOP' if verification_error is None else 'Preserve H4 pilot evidence with failed saved verification'
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=AbeHiromu','-c',
        'user.email=AbeHiromu@users.noreply.github.com','commit','--quiet','-m',message], check=True)
    commit = git(ROOT, 'rev-parse', 'HEAD')
    receipt = dict(status='COMMITTED_VERIFIED_RESULTS' if verification_error is None else 'COMMITTED_WITH_VERIFICATION_FAILURE',
        commit=commit, branch=BRANCH, SOURCE=SOURCE, verification_status=verification['status'],
        worktree_clean=not bool(git(ROOT,'status','--porcelain')), push=False, mandatory_STOP=True)
    save(RECEIPT, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-and-commit', action='store_true', required=True)
    parser.parse_args()
    try:
        finish()
    except Exception:
        failure = dict(status='LOCAL_FINALIZATION_FAILED_NO_SCIENTIFIC_RETRY', traceback=traceback.format_exc())
        if not (BASE/'FINALIZATION_FAILED.json').exists():
            save(BASE/'FINALIZATION_FAILED.json', failure)
        print(json.dumps(failure), flush=True)
        raise
