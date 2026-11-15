# H4 Full PF / B0I p7：固定B2探索round

実行SOURCE `53bb0a1d7b1da70644da1c22552224291205af78`の独立checkoutを使用。B2の作業checkoutとrun directoryは共有しない。元problemは同SOURCEの`configs/rpe_method_optimizer/h4_df10_problem.json`。実引数はEXECUTION_COMMANDS.json。

- 対象：Full PF、B0I p7、S2/Yoshida4/M5、要求幅hの1/2～16倍の既存index。信号/26価格/Q3 routeが揃う構成だけ評価し、入力不足の構成は別報告する。
- 採択済みモデル：Z1+T1 total、保存DF10 solver energy proxy、全事象二乗loss、保存raw/IID二項、exact τ=(0.00159360010199040 Ha)^2。十分N・物理ground保証・正式Gへ昇格しない。
- 方式別のQ3件数は固定しない。公開schemaのUNLIMITEDとscience_approved=trueを使う。有限探索規則はFINITE_LOCAL_DESCENT_V1、同SOURCEの既定100000 step/方式（UNLIMITED時）。この手数終了を候補枯渇と呼ばない。
- programが選ぶ候補・全round X/Y配分・適応的選択順を保持。方式間は独立に並列、方式内は結果取込後に次を選ぶ。
- STEP_LIMIT、INPUTS_MISSING、UNKNOWN_RESULT、NO_ROUTE/JOB_IN_PROGRESS/STALE、改善候補なしを区別。無断retry、別規則や幅の追加、新signal/compile/fit/MCを行わない。
- 価格はC417a73e73固定catalogに束縛された各候補自身の価格・直接/条件付き閉包/model資格を保持。Full PF対PR-Pauliは同catalogの保存固定点（LD5 FULL_PF__B_PRICE、3792207339/7 RZ）と比較する。PRの新Q3や価格は計算しない。

§10.5：Cの保存catalogと関連PF/p7 Q3履歴・関連worktree・現在のoptimizer/Q3プロセスを照合。同条件の未反映PF結果/進行中科学jobは確認されなかった。保存Full PF 11点・p7 14点をinitial=allで再利用する。旧PF6の部分保存状態・過去のUNKNOWN/STOPを保全する。

BASELINE_RESULT.json：Full PF 559457386 RZ、p7 372035410 RZ。入力readyはFull PF3/p7 4構成、範囲内不足はFull PF17/p7 5構成。

PREFLIGHT.json：最初の自動候補2件について、実際の固定routeのmetadata/count/runtime dry-check PASS。Q3は0件。初回の検査準備で、保存JSONのcost有理数をFractionへ復元する前のTypeErrorがあったが、科学call前の自分の読取scriptの問題で修正済み。B2 codeは変更していない。

workerは原API/partials/resourcesを自身のjob directoryへ保存する。親CLIは既存programを/usr/bin/time経由で実行し、consoleとreturn codeを別に残す。ライブ進捗はjobs/*/*/JOB_STAGES、workerのSEGMENTS/STAGES、OUTCOMEを参照し、CPU生存だけで前進と判定しない。二方式とも終了または所定の異常で停止し、SOURCE付きC索引を公開する。
