# Full-H κ2・2倍shot：固定1評価の実行前確認

2026-11-15 JST。ユーザーは計画4525e1b68883bb6a4960346f6406da4d01af2ccdの2倍vector・新有限scopeを採択し、Full-H κ2だけの最大1 Q3 callとexact閾値判定を明示許可した。承認証明書等は作成しない。

## 計画後の重複・入力照合

- Bの公開ref/HEADは4525e1b6、既存worktreeはclean。a82b0f0結果以降の関連local履歴・B/Full-H S1/C/roadmap/D P2の公開refを読取確認した。提案vectorの同等Q3結果は確認範囲で未同定。今回の出力先は未存在だった。
- 自ユーザーの関連Python script名・cwdを読取確認した。稼働中のFull-H P2 checkpoint/accelerated callerは価格処理で、Q3重複ではない。process・他worktree・設定を変更しない。資料未取得を全記録の不存在とは扱わず、今後同等結果の存在が判明した場合は二重投入しない。
- 4525e1b6のPROPOSAL.jsonをbyte照合。元54ac3eb3→total265b90dc、binding579c5de2、全13physical行・元/total round ID、SOURCEとlawをa82b0f0 INPUTSから照合。元ce6aacfa PLAN/B_SIGNAL_INPUTと既存Full-H metadata routeを再利用した。one-body込みκ2/K2、grouping・q/R/event数・scalar・normalizationは変更しない。R448・PR-DF・Pauli・PFを投入しない。
- 元13rawは再取得せず、normalization/scalarを再適用しない。solver proxy・ideal finite-K raw/IID・EXACT_AFFINE_OF_BINARY64_RAW・Z1＋T1・全事象平方loss・exact十進εE²を維持。旧Minkowskiは別参照、AδE加算なし。詳細正本は固定計画と今回INPUTS。

## count-only包含

X/Y各軸`[128,120,112,104,94,86,76,68,60,50,42,34,24]`、allocation ID `FULL_H_KAPPA2_TOTAL_Q3_CONTROL_SCALE2_PROPOSAL_01`だけ。

`C_m=(N_X+1)(N_Y+1)`の最大16,641、全隣接組上限847,519,356、`C0+8192*Σ(C1..C12)`の遷移上限620,478,721を既存純整数helperで確認し、採択scopeと一致した。非終端frontier8192、ρは全13round1.0。**旧N0=64の2593初期状態・残余0というsupportを新N0=128へ移送していない**。新supportの閉包・重み・残余品質は実行内gateまで未確認。

## SOURCE・worker保存・人工検証

新callerは`run_full_h_kappa2_scale2_q3_fixed.py`。旧3候補runnerは起動せず、そのFull-H metadata・`worker`・`supervise`をimportして使う。対象・配分・scopeだけを新INPUTSで固定し、旧kernel710aa69b・accelerator9e294778と旧結果は不変。execute入口でcaller/INPUTSのcommit一致と固定proposal全体を確認し、run01を新規作成する。一度作成したrun01は再実行で上書きしない。

- 新caller2件：実H4 risk呼出しを禁止したmetadata/13round/count確認、異vector・scope・ID拒否。
- 再利用worker5件：人工decoder評価、原API return/partials/CPU/RSSのworker保存、親monitor故障中の子保存、未開始call拒否、R448等の混在拒否。
- **計7技術test PASS**（TESTS_SCALE2.txt、TESTS_REUSED_WORKER.txt）。人工以外のQ3・DP・確率計算はこの準備で0件。

親monitorが例外になっても、開始済みworkerが原API return・RESULT・RESOURCE_FINAL・PERSISTEDを保存してから小さいreceiptを返す既存処理を使う。partialsは各stageで保存。保存状態と数値判定を分離する。

## 実行・gate・保全

観測時64 CPU、available約500,633,292 KiB、load約2.23。既存8worker/batch8の高速backendを1構成で再利用する。これはCPU/RAM/workerの恒常的上限ではなく、共有設定を変更しない。新vectorの時間・メモリ保証は未測定。資源監視は自分のworker subtreeだけに限定する。

caller/INPUTSをcommitしてから最大1call。ρ閉包、mass・残余loss、principal guard、allowance、MSE上下界・momentsを既存kernelと結果整合検査で確認する。m0を含むpruning質量を捨てない。有効upper≤τなら条件付きモデル適合、lower>τならモデル不適合、閾値をまたげばINCONCLUSIVE、gate不成立ならUNKNOWN。原API returnを保全し、保存状態とモデル判定を別記する。

重大異常・最大1call終了・同等結果再利用でmandatory STOP。無断retry・別vector・追跡拡張・許容差/backend変更・価格・G・必要N探索なし。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`、ground/実sampler移送/数値certificate未資格を維持する。旧input/raw/結果/SOURCE/grant/ledger/marker/KEEP/STOPは変更しない。
