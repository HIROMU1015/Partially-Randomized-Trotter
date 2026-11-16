# ローカル担当の観測範囲

2026-11-17の実行セッションはUbuntu 22.04系 / Linux 6.8.0-47-generic。
WSL marker、`wsl.exe`、`powershell.exe`、Windows drive mountを確認できず、Windows本体の資源を測定したものではない。
後続の利用者確認で、この環境の接続aliasがRomuTailであり、ローカル0にqg1/qg2/RomuTailが定義されていると判明した。
採用経路はローカル0からのSSH mailbox relay。このセッションからローカル0へ接続する経路は無い。

| 項目 | このLinuxでの確認 |
|---|---|
| CPU | affinityで32 logical CPUを使用可能 |
| RAM | MemTotal 65,522,476 KiB（約62.5 GiB）、最初のMemAvailable 55,137,680 KiB |
| ディスク | 初回空き452,212,506,624 bytes（約421 GiB）。固定view 5つの展開後 `.local-q3` は約15 GiB |
| GPU | nvidia-smiはdriverに接続できず。GPU利用可能とは判定しない |
| SSH | ssh/rsyncあり。`~/.ssh/config` は空。qg1/qg2は `ssh -G` でも別名展開なし。qg2の接続試行は名前解決失敗 |
| Python | default 3.10.12には数値依存なし。既存 `.venv311` は3.11.0rc1でnumpy/scipy/numba/psutil/pytestあり。flintなし |
| 実機間の経路 | 利用者確認済み：ローカル0→qg1/qg2/RomuTail。このセッションからのSSH実測はしていない |

新環境のinstall、共有設定変更、SSH鍵/認証設定変更は行っていない。
flint未導入でも下記PF/Pauli metadata検査と人工cacheテストは通過した。
DFのpath不足を直しても、DF全依存と入力の検査が完了するとは判定しない。

## 再利用した実装

- `scripts/run_rpe_optimizer_q3_job.py`: 既存 `dry-check` / 人工Q3。配布fixtureは `../../fixtures/ARTIFICIAL_REQUEST.json`。
- `tests/test_rpe_optimizer_concurrent.py`: 人工problem/信号/価格fixture。準備コマンドのsmokeは同fixtureと `rpe_optimizer_concurrent.ConcurrentExecutor` を使用する。
- `tests/test_rpe_dp_prefix_reuse.py`: 人工tables、exact mass、forkした既存worker、disk cache再開、SOURCE guard、MSE判定一致を検査。`RPE_FIXED_VIEW` でbd42dd1e9のローカルviewへ対応付けた。
- 共通89f51a4のworker/relayを再利用した。ローカルbranchではhost別DP worker設定をRESOLVED_REQUESTに適用する最小拡張と、relay再開時の状態表示の修正を追加した。DP拡張はcontroller/workerの固定SOURCEへ統合されるまで89f51a4の実profileでは有効にしない。

## 固定routeと入力

`configs/rpe_method_optimizer/h4_df10_problem.json` と
`configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json` からrouteを読んだ。
各 `view_commit` を所有するdetached cloneへ展開し、requestのローカルコピーだけに
`view` / `interpreter` を指定した。サーバーpathのdirectoryやsymlinkは作っていない。
各routeの最初のtemplate（common-entryはcontract template）に保存されたshot vectorで
既存adapterの `--mode dry-check` を実行した。新しいMSE評価は行っていない。

| route kind | 固定view SHA | metadata/count/runtime dry-check |
|---|---|---|
| pf_b_pf6（FULL/TRUNCATED共通） | cbc471c913072de45c25a0ed22d51dff8a9b7034 | PASS |
| pauli_b_ld3 | a844ac25f88c5cdf39e9e1837d85b6d4341774ff | PASS |
| pauli_b_ld5 | b6d4dea49c39f364048af422bc2928ee7c169261 | PASS |
| pauli_b_common_entry | bd42dd1e95eb1d4152216d62792ad3421b803f7c | PASS |
| prdf_d2_split | 23160cc876807c0300be3de811f767b5810640a7 | BLOCKED_PATH |

PF/Pauliの通過範囲では必要なtemplate/input Git objectsが解決できた。
これは全候補入力・大規模入力・将来のSOURCE inventoryを網羅する検査ではない。

DFの実際の不足:
`run_pr_df_ld_split_q3_resume.py` は旧固定caller
`e0b3cad2a024070389f757160f3a20d7da02c5ff:scripts/run_pr_df_ld_split_q3.py`
を読み込み、そこで固定view
`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115`
から `scripts/run_pr_df_fullh_total_q3_fixed.py` をloadする。このdirectoryがローカルに無い。
その固定viewの要求SHAは `27e70efe533862f38f4080aabb9cdd3ad1749fe9`。
外側routeの `view` をローカルにしても内側の固定pathが残る。
固定SOURCE/AST条件を変更せずにローカルviewへ解決できる共通adapterの対応が必要である。
その後DFのmetadata/count/runtime/source/input依存を再確認する。DFは実行していない。

## 保存した検査と限界

[checks/](checks/) に配布人工入力のdry-check、人工OUTCOME、設定2/2と参照1/1の比較、
今回の検査一覧を保存した。Linux単一hostの技術検査であり、Windows・CI・独立再現・
実3host検収・H4性能の証拠ではない。人工worker内の `scientific_evaluation_calls=1` は
人工callのカウンタである。新規の実H4科学callは0。

DP cache検査コマンド（ローカルviewの配置に合わせる）:

```bash
RPE_FIXED_VIEW="$PWD/.local-q3/views/bd42dd1e95eb1d4152216d62792ad3421b803f7c" \
  "/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" \
  -B tests/test_rpe_dp_prefix_reuse.py
```

10/10 PASS（約0.95秒）。SOURCEの固定guardを維持し、人工13-roundで末尾のみ変更した
次jobの12-round再利用、別processでのdisk cache再開、MSE上下界と判定の一致を確認した。
共通workerの通信を経由するcache再利用はまだ未検収。

## 共通実装公開後の検査

`89f51a4fef47ecd2364d22c4c5cc2e225df3f2ce` の専用detached view
`.local-q3/common-89f51a4` を用意した。SOURCE manifestにHEAD一致条件があるので、
ローカル成果branchのHEADを実workerとして代用しない。固定viewのtracked sourceは変更していない。

`scripts/check_rpe_q3_local_join.py` は同固定viewの人工producer・通常schedule入口・worker・fault proxyを再利用する。
両初期workerのARTIFICIAL_EVALUATION_STARTを確認してからローカル役を起動し、
その最初の人工job開始後に通信を遮断してpolling daemonだけを停止した。
detached supervisorの結果保存と他hostの完了増加を確認してから同じspoolで復帰した。
同一attempt・保存結果不変・そのjobのexecute1回・重複完了通知の冪等性を確認した。
参照18件のMSE上下界/判定と方式別最安も一致した。[結果](checks/LOCAL_JOIN_CHECKS.json)。

```bash
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B \
  scripts/check_rpe_q3_local_join.py --common-root .local-q3/common-89f51a4 \
  --output .local-q3/checks/late-join-NEW
```

共通workerの7テストは固定89f51a4でPASS。relayの既存2テストは同固定SHAで1件PASS、
再開時に前daemonのstopped状態が残る競合で1件失敗した。
ローカルbranchのrelayはtick前に新しいstopped=false状態を保存する修正を行い、2件ともPASS。
host別DP設定のローカル拡張を含むworkerテストは8件PASS。
これらは単一Linux上の人工検査であり、実SSH経路の検収ではない。

通常sandboxではloopback socket作成がPermissionErrorになったため、HTTPを使う人工検査だけ
実行環境の許可された制限外実行で行った。実サーバー・H4入力には接続していない。
