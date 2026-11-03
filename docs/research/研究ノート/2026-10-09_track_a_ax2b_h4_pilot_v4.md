# 2026-10-09：AX-2B H4 v4一回実行・時間上限STOP

利用者の「ではそのまま進めて」を、固定v4の同一H4限定pilot一回実行・保存監査・STOPへの認可として適用した。
元v4 preparation inventoryを変えないため、新しいexecution worktreeへ既存1,721ファイルをbyte一致でコピーした。
153 source hashes・manifest・保存input・環境・CPU3/thread1・8 GiB/phase900秒の上限を照合し、
155 synthetic testsがpassした後、新しい認可とexclusive `run_v1`で一回だけ起動した。

linear H4 1.00 Å/STO-3G/legacy rank12/generation-prefix/T=0.8、q1/q4（δ0.8/0.2）、
B0 prefix6、B1 prefix12、B2 prefix6/R8/K2,K4、B3 prefix0/R8/K6、四次B1の登録8 cell検査はpass。
保存correctness8件は旧v3とbyte一致。16/28 measured full wrappers（state preparationを除く）を保存後、
wrapper phaseの900秒上限に達しparent watchdogがSIGKILLした。parent wall921.208770秒。
最後のmarkerはB3 replica1/ordinary/cosineのwrapper準備内numeric fingerprint。
worker terminal/tracebackは欠け、凍結保存verifierはearly partial pathとして停止結果を保持した。

追加stdlib監査で16 wrapperのidentity/metrics、5 trajectoryの旧v3一致、246診断を確認。
42完了fingerprint区間のwall合計845.846秒に対し、16 transpile/metrics区間は15.420秒。
今回の主な古典処理時間はfingerprintであり、stream方式の性能改善が次の技術課題と判断した。
内部の支配的処理は未profileなので、JSON/generator/definition構築の原因を確定しない。
保存peak RSS約0.961 GiBだが、全28件・四次baselineに未到達で、memory問題解消はまだ結論できない。

判断：`H4_CORRECTNESS_PASS_COST_PARTIAL_PHASE_WALL_STOP`で停止。
同じbytes/mutation拒否を維持した性能改善と合成検証を次の候補にする。
今回はsource変更、retry/resume、cap拡大、baseline削除、H6/H8/GPUへ進めない。
数値allowance・accuracy・shot/total cost・方式順位の結論は更新しない。
元v4/v3/dirty worktree、旧研究結果・原稿・Track Bを保持し、commit/pushしない。

[結果・証拠・次の一段階](../track_a_ax2b_h4_pilot_v4_result.md)に詳細を記録する。
このノートは実行後の履歴で、先のv4準備ノートや旧v3停止理由を書き換えない。
