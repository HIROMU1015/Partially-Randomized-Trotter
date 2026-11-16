# 再現条件と移植上の注意

今回の検査は環境/人工限定。分子Q3のexecute、signal生成、compile、H4本番runの再開は実行していない。

## 固定runtime

元のDP AST guardはPython3.12のAST形状を拒否する。guard定数やDP SOURCEを変更してはいけない。
既に準備されていたCPython3.11.0rc1を再利用し、専用addonへ不足の`_ctypes`を同じソースからビルドした。
libffi-devはdownload/extractのみでOSへインストールしていない。
numpy1.26.4 / scipy1.14.1 / mpmath1.3.0のcp311/pure wheelを専用directoryへ展開。
共有venvは変更していない。入力hashは [RUNTIME_INPUTS.json](checks/RUNTIME_INPUTS.json)、
ビルドcommandは [CTYPES_BUILD_COMMAND.json](checks/CTYPES_BUILD_COMMAND.json)。
launcherは`PYTHONPATH`をそのaddonとsite-packagesへ設定し、BLAS/OMP/MKL threadsを1にする。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-server1-relay
SERVER1_Q3_PYTHON=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/runtime-acceptance/bin/python3
RPE_FIXED_VIEW=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/routes/b-bd42dd1e9 \
  nice -n 10 "$SERVER1_Q3_PYTHON" -B tests/test_rpe_dp_prefix_reuse.py
```

Python3.12で同じtestを実行した最初のFAILは [PREFIX_REGRESSION.log](checks/PREFIX_REGRESSION.log) に保持した。
3.11.0rc1の成功は [PREFIX_REGRESSION_PY311RC1.log](checks/PREFIX_REGRESSION_PY311RC1.log)。
これは共通分散workerの不具合ではなく、既存固定routeのruntime要件。

## PR-DFのネストした絶対view

固定route `23160cc876807c0300be3de811f767b5810640a7` がgit objectから読む旧caller
`e0b3cad2a024070389f757160f3a20d7da02c5ff:scripts/run_pr_df_ld_split_q3.py` には
`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115`
が固定されている。worker profileのroute_viewsだけではこの内部参照は解決しない。

現在このpathには、今回専用の `routes/prdf-kernel-27e70efe5` への既存symlinkがある。
そのtarget HEADは `27e70efe533862f38f4080aabb9cdd3ad1749fe9`、trackedファイルはclean。
この対応を確認したうえでdry-checkを通した。固定caller/source guardは変更していない。
別ホストへ移す場合は同じviewを確認して対応を用意するか、共通担当側で明示的path resolverを設計する。
サーバー1側に独自queueやselectorを実装する必要はない。

```bash
# 既存失敗を再現する場合だけ、互換pathのない新環境で元REQUESTをdry-checkする。
# 稼働中の互換symlinkは消さない。
# 成功の再検査：--outputには未使用directoryを指定（既存DRY_CHECKはO_EXCLで保存される）
nice -n 10 "$SERVER1_Q3_PYTHON" -B scripts/run_rpe_optimizer_q3_job.py \
  --request docs/research/rpe_distributed_q3/server1/checks/prdf-23160cc87_REQUEST_PY311RC1.json \
  --output /tmp/server1-prdf-dry-NEW --mode dry-check
```

Pauli LD3/LD5も同じ`*_REQUEST_PY311RC1.json`と別outputでdry-checkできる。
dry-check対象の入力commit/path/hashは [ROUTE_INPUTS.json](checks/ROUTE_INPUTS.json)。
保存H4信号/構成は読取のみ。数値結果の追加・再資格はしていない。

## mailbox部品jobの再照会

[ASSIGNMENT](checks/mailbox_component/ASSIGNMENT.json) と [PROFILE](checks/mailbox_component/PROFILE.json) が原本。
共通の`fixtures/ARTIFICIAL_REQUEST.json`を使い、job IDと人工sleep=2秒だけを部品検査用に変更した。
submit応答後にRPCを行わず、ローカル状態が完了へ遷移したことを観測してから再接続した。

```bash
SERVER1_Q3_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/common-89f51a4fef
python3 -B "$SERVER1_Q3_ROOT/scripts/run_rpe_q3_worker.py" \
  --profile docs/research/rpe_distributed_q3/server1/checks/mailbox_component/PROFILE.json --mailbox status
```

保存spoolは`COMPLETED_PENDING_TRANSFER`のまま保持。中央採用がないためackは送らない。
同一assignmentの再submitはduplicateとして照合され、再計算を起動しない設計。
この試験ではstatus2回のbundle/JOB_STAGES一致とLAUNCH1回を検査した。SSH切断の実2台検収は別途必要。
