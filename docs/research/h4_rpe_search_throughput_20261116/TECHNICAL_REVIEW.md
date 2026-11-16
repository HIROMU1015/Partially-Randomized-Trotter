# 通常入口・再開・保存再利用・準備並行の技術確認

2026-11-17 JST。対象branchは `codex/h4-rpe-search-throughput-20261116`、基点は B2 `d8e806265c3ded8e1184181c37c2d7ec71f08ade`。

実装担当はroot。rootが変更した入口・接続コードを、別のsub-agent `saved_log_analysis` が読んだ技術確認である。重要な科学レビュー、実H4の数値一致確認、実性能認定ではない。`saved_log_analysis` 自身が作成した保存ログ分析と追加回帰testは、この文書によって独立確認済みとしない。

## 対象と経路

- `scripts/run_rpe_schedule_search.py` → `scripts/run_rpe_kappa_neighbourhood.py` のbatch → `scripts/run_rpe_method_optimizer.py`。
- `src/trottertracks/rpe_optimizer_concurrent.py` の通常起動・予約予算・完了保存・live再開・既完了prior job再利用。
- `scripts/run_rpe_optimizer_q3_job.py` の既存route、実行profile、DP-prefix context。
- `src/trottertracks/pr_candidate_completion.py` の既存準備状態と、通常optimizerを維持する継続callback。

読取と人工workerだけで確認した。新規の実信号・較正・compile・Q3、保存H4のDP再計算は行っていない。

## 指摘して修正を確認した事項

| 経路と元の問題 | 確認した修正 |
|---|---|
| 空の新runでは `continues.run_dir` の状態を取り込まず、旧LD1・2・3・5のfamily constantsが欠け、保存点を広域planから落とせた。 | 新runの初期化時だけ前runのpoints・family constants・saved packets・splitsをコピーする。元の状態ファイルを書き換えず、前round/STOPはcontinuationの来歴へ残す。 |
| plan-only/未承認で実行しなかったCOMPLETIONがHELDのまま残り、通常再開で処理対象に戻らなかった。 | 理由が厳密に `COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION` のものだけCOMPLETIONへ戻す。実際の科学的HELD/STOPを一括で開かない。 |
| 準備中に走るready batchがUNKNOWN等で止まっても、通常の次batchへ進めた。保存済みのdriver STOPも再開入口で参照していなかった。 | interimのUNKNOWN/STALE/NO_ROUTE/予算停止を外側driverへ伝達し、同じrunにある非一時的STOPはplan・準備・Q3投入より前で保持する。 |
| ConcurrentExecutorを直接使う場合、`has_job` はprior完了を認識する一方、answer/prefetch側はpriorを見ず再投入できた。通常optimizerのprior一括取込に隠れた穴だった。 | enqueue前にprior jobを除外し、answerでは既存のroute/identity照合を使って `PRIOR_ROUND_JOB`、calls=0として返す。 |
| 再開時の未採用先読みworkerはslotに数えるが、他に選択候補がないと `finish()` がその終了前に戻った。 | `wait_for_running=True` の終了処理は、該当method/rootのexternal resumed jobがdrainされるまで監視を維持する。workerの停止・再起動はしない。 |

有限予算のbatch入口では `launched + batch` を元roundの総予算でcapする。先読みの予約・起動数をscheduler側で計上する構造を維持している。通常最適化後の準備callbackにも同じoptimizerを渡し、共有executorを最終段階まで開いたまま使う接続を確認した。

## reviewerが追加した人工回帰

`tests/test_rpe_throughput_entry.py::EntryTests.test_finish_waits_for_unselected_resumed_worker` を追加した。短い人工legacy workerを起動してからconcurrent executorを再開し、候補を一つも要求せず `finish()` する経路で、戻った直後のPROCESS_RETURN保存、external集合の解消、新規起動0を確認する。

修正前の再現では `finish()` が約0.00062秒で戻り、PROCESS_RETURN未保存・external 1件だった。root修正後の当該人工testは1/1 PASS（約0.333秒）。このtestはreviewer自身の作成・実行であり、そのtestの独立レビューではない。既存のscheduler/prefix/入口suiteの最終再実行は、rootがSOURCEを確定してから行う。

## 最後の限定指摘の解消と確認範囲

`overlap_ready_batch` のinterim subprocessが失敗してres=Noneになった場合、callbackのRuntimeErrorだけで終了し、外側状態へ `V2_INVOCATION_FAILED` を保存しない経路を最後に指摘した。rootは、失敗のreturncodeをinterim recordへ保持し、既起動の準備futureを終了・取込した後、次batchを起動する前に同じSTOPを保存するよう修正した。reviewerはこの経路を読み、rootが追加した `EntryTests.test_interim_batch_failure_is_persisted_before_any_second_batch` だけを実行して1/1 PASS（約0.001秒）を確認した。失敗結果をabsorbしないこと、returncode=17の保持、再開時に再batchしないことを人工入力で検査する。

準備との並行化はopt-inであり、interim batchでは `candidate_completion` を外して、準備担当が更新するSTATE/registryを別Completion instanceから読書きしない。結果の採択は既存V2順、準備futureの吸収は選択順を維持する。この確認だけで実H4の高速化や、旧ログにないready待ち時間を認定しない。

この限定確認で未解消の具体的な指摘は残っていない。最終SOURCE全体のsuite再実行・通常入口のplan-only検査・最終公開はrootが担当する。

資料作成の前提として、reviewerは `docs/research/研究概要・現状.md` 全589行を分割して読んだ。これは委任された概況読取であり、root自身が全文を読んだとの表示には用いない。旧2026-09-20本文・rank12等の過去条件を今回のH4 DF10へ移さず、現行の条件付き価格・精度契約・固定版の状態は今回指定されたB2/D2資料を用いる。

## rootによる後続の作者検査

上記の限定レビュー後、rootはV2の方法全体previewから外れた別構成の未起動予約を取消す差分と、実際の通常executor再開で観測窓がresetする回帰を追加した。これらはroot自身の実装・検査であり、上記reviewerによる別担当レビュー済みとはしない。最終入口13検査と共通scheduler15検査はchecks/VALIDATION.jsonに記録する。
