# R08のqueue予算応答修正と継続

`KeyError: 'calls'` の発生元を修正した。queueが新しい予約を拒否した `BUDGET_EXHAUSTED` を、探索器が数値Q3結果として消費していた。修正前に通常幅CLIと共有queueの人工経路で同じ例外を再現し、修正後は正常な `Q3_BUDGET_EXHAUSTED` でbatchを返す。未知の数値回答の欠落値を0で埋める変更はしていない。

修正SOURCE：`2c2196bee51589b6765039691c1e7509169ef6af`（中核差分 `f3867a1f3dc697e65d4441e9a19f54915cefe819`）。branchは `codex/b2-rpe-pr-budget-continuity-20261118`。旧fdb4453a0のSOURCE checkoutを保全するため、独立worktreeを使った。

## 発生条件と修正

1. V2の回答数にまだ余裕があっても、先読みがそのbatchの全予約枠を使っている場合がある。
2. `ConcurrentExecutor._enqueue` は新しいjobを予約せずFalseを返す。その制御応答には `calls` がなく、HOLDでもなかった。
3. `run_priority` が `answer['calls']` へ進み、全体の例外になった。旧探索器では同じ誤分類でMSE欄の欠落も起こり得る。

`rpe_optimizer_concurrent.py` の予約拒否は `calls=0`、`scientific_evaluation_calls=0`、sourceを明記し、`BUDGET_WAIT` を記録する。旧・V2探索器はこの状態を数値知識・PASS/FAIL・candidate holdへ入れず、未実行の要求を残してbatchを返す。既存WidthDriverが、元の全体予算内で次batch/cycleへ進める。未開始の先読みをキャンセルして予算枠が戻った場合も、準備完了を待たず同じqueueのQ3へ戻す。この空き枠の見落としも通常CLIで人工再現し、`BEFORE_RELEASE_FIX.txt` と修正後5件で確認した。

queueの予約上限、先読みを含む実起動数の計上、`take_count()`、保存回答の0件会計は維持。`adopted_new_calls` とqueue起動累計の `new_calls` を区別し、未採用の先読みも実起動予算へ計上する。残るIN_PROGRESS/STALE_JOB/NO_ROUTEの制御回答にも0件を明記した。

変更はこの制御境界と再開時の出力epoch指定のみ。順位・shot提案・κ/幅選択、price、MSE/τ、scalar、native emitter/lowering/compiler、Q3 worker数値kernelは変更していない。既存AST比較は追加した予算分岐だけを明示的に除き、順位・価格・更新ロジックの同一性を保つ。

## 検証

- 追加人工5件PASS：拒否時の0件会計、同じqueueで次batchへ継続、旧探索器の制御応答処理、通常 `run_rpe_schedule_search.main` と同じrun再開0起動、予約解放後の準備中Q3再開。
- 既存並列回帰15件PASS、準備並行・例外回収・引継ぎ回帰17件PASS。今回の異なるテストは計37件。
- native Python3.11.0rc1で追加5件PASS。引継ぎ4件も別途再確認。
- `checks/BEFORE_FIX.txt` は修正前の同じcalls例外、`AFTER_FIX.txt` は修正後。`NORMAL_ENTRY.json` は人工provider/workerでの通常CLI機能入口の記録。本番H4の速度倍率とはしない。
- 実H4の数値・回路・compiler作用を人工PASSだけから認定しない。実再開の進捗はrun02のstage/queue logで確認する。

## 旧結果と§10.5

旧停止は11/17 21:31 JST、RUNTIME_EXCEPTION/KeyError/'calls'。元128jobは全てOUTCOME/PROCESS_RETURNがあり、状態は適合92、不適合36、未回収0。数値再評価・fitは行わず、元の値を保持した。元の全結果・停止tracebackは旧branchの `f9e2ff4ecfdc41b254e316d5e74e42538a6bc518` に追記公開した。旧SOURCE `fdb4453a0` と旧STOPを変更していない。

再開前、関連controllerなしと全128receiptを確認した。既存prior_run_dirsを維持し、旧run02を追加した。元Executorがjob key/route/worker/SOURCE/bindingを照合して完了分を再利用する。出所が合わないjob・過去の失敗科学stage・曖昧STARTEDは再起動しない。

`PRELAUNCH_ACCEPTANCE.json` は元WidthDriverの固定入力/family/physical gateを通した記録。lambda_sourceに関する5pointの不一致は元guardによる個別HELDのまま引き継いだ。共通数値UNKNOWN、SOURCE不整合、保存失敗は従来の停止保護を維持する。

## R08再開範囲

`RESUME_PLAN.json`、`PROBLEM.json`、`ROUND.json`、run02/SOURCEを科学再開前に固定した。R08 ID、PR-Pauli .1/.2・PR-DF .1/.05、S2、L_D1〜9、元mapper/target/decoder/law/loss/exact τ、条件付き価格を維持する。元cycle4から上限200まで残り196で、予算をリセット/拡張しない。固定の新しい出力epochを使う。

実行時資源は利用可能64 CPU、MemAvailable約426 GiB、load約12。従来の共通queue20・DP3・checkpoint m5/m11・同じprefix cacheを維持。共有設定や他processには変更しない。科学retry loopは追加していない。

ロードマップ差分はR08がbatch予約到達で異常終了する技術不具合の解消と、保存結果を使った継続。新しい最安・全探索完了・価格資格の拡張は認定しない。統合担当は[HANDOFF](HANDOFF.md)の固定locatorと実run進捗を参照する。
