# RomuTail PR-DF：内側SOURCEの配置解決と実入力dry-check

2026-11-17。`prdf_d2_split` のREQUEST `48494ae31276a64c46ce` は
**RomuTail実機で `REAL_HOST_DRY_CHECK_PASS`**。
[READINESS](checks/pass01/READINESS.json) / [adapterログ](checks/pass01/ADAPTER.log) /
[検査一覧](CHECKS.json)。実Q3・compile・本番queue操作はすべて0。
実装・検査はいずれも担当Codex自身が行い、独立レビューではない。

## サーバー2が取り込む変更

専用branch：`codex/rpe-prdf-romutail-source-resolution`。
**実装commit：`e37f6544dea3e021d3694264f7dae87add7ad30e`**（計算SOURCE 28d976bを基点）。

- `scripts/run_rpe_optimizer_q3_job.py`：固定callerが要求する内側commitを読み、
  `route_views[commit]` にあるcheckoutを使用する。該当する1回のimportだけを正しいファイルへ転送し、
  callerの `VIEW` の実行時bindingとimport pathを対応付ける。保存callerや科学コードのbytes/ASTは書き換えない。
- `src/trottertracks/rpe_q3_worker.py`：PR-DFのresolved copyにprofileの `route_views` を渡す。
  元REQUEST・job ID・hashは変更しない。
- `scripts/check_rpe_q3_host_readiness.py`：計画33928bfの既存readiness入口を再利用し、
  任意の `--adapter-root` を追加。元assignmentは元SOURCEに照合し、新adapterのcommit/hashを別欄へ記録する。
- 配置切替・誤SHA・dirty SOURCEの人工テスト2件、profile伝達の人工テスト1件がPASS。
  対応testは `tests/test_prdf_source_resolution.py` と
  `tests/test_rpe_q3_worker.py::WorkerTests::test_prdf_dependency_paths_are_only_added_to_resolved_request`。

この固定計画の元assignment manifestは28d976bのまま保存し、書き換えていない。
今回のPASSは **元SOURCE/入力の照合＋新adapterでのdry-check** である。
新adapterを実workerへ導入する場合はサーバー2が共通SOURCEへ取り込み、
controller/workerの新しい固定SHA・manifestを通常の方法で揃える。古いmanifestへの偽装はしない。
本番実行を今回のPASSから開始しない。

## 固定条件と実配置

| 項目 | 今回の値 |
|---|---|
| 固定計画 | `33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b` |
| 元計算SOURCE | `28d976b99cdf8329adcc1b79327e329e947ef774` |
| route | `prdf_d2_split` |
| 外側checkout | `23160cc876807c0300be3de811f767b5810640a7` |
| 旧caller | `e0b3cad2a024070389f757160f3a20d7da02c5ff:scripts/run_pr_df_ld_split_q3.py` |
| 内側checkout | `27e70efe533862f38f4080aabb9cdd3ad1749fe9` |
| Python | `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python`（3.11.0rc1 exact） |
| 新adapter専用checkout | `/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-prdf-romutail-source-resolution` |
| 内側SOURCE実パス | 上記checkout内 `.local-prdf-inner` |

内側は所有するdetached cloneで、`scripts/`・`src/` を展開した。
サーバー絶対pathを模倣するdirectory/symlinkは作っていない。
元の外側checkout・元28d976b checkout・過去のprofile/結果は変更していない。
`.local-prdf-inner` は共有Git objectsを使うため、元repositoryを残す。

今回のprofileは [PROFILE.json](PROFILE.json)。既存profileの `route_views` に次の1件を追加した。
別ホストでは同じkeyへ、そのホストの正しいdetached checkoutの絶対pathを設定する。
サーバー2では既存の27e70efe checkoutを検査して再利用し、重複checkoutを作る必要はない。

```json
{
  "route_views": {
    "27e70efe533862f38f4080aabb9cdd3ad1749fe9": "/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-prdf-romutail-source-resolution/.local-prdf-inner"
  }
}
```

新adapterはHEAD一致・tracked scripts/srcの変更なし・内側entry bytesの一致を確認してからimportする。
その後、元の `U.check_source()` と `runtime_check()`、元metadata/count検査を実行した。
AST guardは `3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8`、
`constructor=PASS_UNCHANGED_GUARD`、`guard_modified=false`。
REQUEST・configuration・shot vector・入力blob hash・評価契約は元計画のものを保持した。

## ローカル0から次回同じ乾式検査を行う

次のbashコマンドをRomuTailで実行する。ローカル0からは `ssh RomuTail` で入る。
Python探索やFull PF/B0Iの再検査は不要。outputには新しいdirectory名を指定する。

```bash
cd '/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-prdf-romutail-source-resolution'
Q3_PYTHON='/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python'
"$Q3_PYTHON" -B scripts/check_rpe_q3_host_readiness.py \
  --assignment docs/research/rpe_prdf_romutail_source_20261117/ASSIGNMENT.json \
  --profile docs/research/rpe_prdf_romutail_source_20261117/PROFILE.json \
  --adapter-root "$PWD" --output .local-prdf-readiness/check02
```

`READINESS.json`・`ADAPTER.log`・`adapter/DRY_CHECK.json` を回収する。
新しい実行許可や本番queue投入を意味するコマンドではない。

## 拒否検査と確認範囲

同じ実REQUESTで27e70efeの設定先を外側23160ccのcheckoutへ誤指定したところ、
`PR-DF inner SOURCE HEAD differs` で拒否され、乾式PASSにならなかった。
[拒否時READINESS](checks/wrong-inner01/READINESS.json) / [ログ](checks/wrong-inner01/ADAPTER.log)。
同一SHAのtracked SOURCE変更の拒否は人工Git fixtureで確認した。

PASSしたのはこのPR-DF REQUESTのSOURCE・入力・metadata・count・runtimeだけ。
MSE評価、科学的性能・精度判定、Q3終了条件や本番queueの検収は行っていない。

次はサーバー2担当が上記実装commitとprofileの内側mappingを共通実装へ取り込み、
ローカル0担当が必要時に同じ乾式コマンドと設定を再利用する。実Q3/compile/queue投入は別の許可範囲である。
