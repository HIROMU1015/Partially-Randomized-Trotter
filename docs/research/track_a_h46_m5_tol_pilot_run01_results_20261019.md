# H4/H6 m5_best：tol-only technical pilot run01 結果・STOP

2026-10-19。**29/30 task PASS、H6 full m5_best/q2のcosine検証不一致で部分STOP。** 13:03:26〜13:30:50 JST、wall 1,644.77秒（約27分25秒）。native launcherはexit1、全24 pool workerはexit0で回収済み。予定最大30 workerに対して実際にspawnされたpool workerは24だった。今回の一回実行は終了し、mandatory STOP。再compile・retry・閾値変更・再fit・後続Phaseは実行していない。

SOURCE：`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。branch：`track-a-h46-m5-tol-native-readiness-20261019`。準備資料：`54f3e8da374b2ad60755dd6bbc6ae3b8150c6502`、起動記録：`2cb0f023f23e5ba893393b65d05e7cae3a819a7c`。結果入口：[summary](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/summary.json)、[終了witness](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/completion_witness.json)、[外部wait4](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/external_exit_witness.json)、[全60予定軸の状態](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/axis_status.csv)。

## 固定範囲とlane別結果

線形H4/H6、隣接1 Å、STO-3G、保存tol-only1e-8、rankはH4=10/H6=19の観測値、T=.8、P=0。S2/Yoshida4/m5、q2、H4 p3/H6 p5（fullは全rank）、PR K4・signed S2 piece当たりr2。m5係数は `PF_M5_BEST_20D64C2_BINARY64_V1`、11 signed S2 pieceをそのまま使用した。symmetric-directional、full Hadamard wrapper両軸、scalar一回、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。H4 legacy DF12や旧ordinary高次PFの費用は混合していない。

|lane|task PASS/予定|受理費用軸|compile予約|終了状態|
|---|---:|---:|---:|---|
|H4_TOL_PILOT_DET|9/9|18|18|COMPLETE_MANDATORY_STOP|
|H4_TOL_PILOT_PR|6/6|12|12|COMPLETE_MANDATORY_STOP|
|H6_TOL_PILOT_DET|8/9|16|17|STOP|
|H6_TOL_PILOT_PR|6/6|12|12|COMPLETE_MANDATORY_STOP|

PR12 trajectoryは全て両軸の費用取得まで完了。系ごとの共有full-H reference2件は両PASS。全60予定軸のうち、58軸は受理済み、1軸は検証不一致、1軸はSTOP後未投入。59予約を58成功として書き換えたり、未受理・未投入の単価を0補完したりしていない。

## STOPした固定点

H6 tol-only DF19、full PF、`4th(m5_best)`、prefix19、q2、代表sample0、cosine。taskは `H6_TOL_FULL_0e289606747ea369__0`。理由は `NUMERIC_DISAGREEMENT:compiled_vector_cosine`。

保存された検査値は次のとおり（再計算せず転記）。

|検査|保存error|固定基準での状態|
|---|---:|---|
|normalization_log_B|0|PASS|
|native_low|7.966375899717263e-14|PASS|
|native_leakage|0|PASS|
|independent_occupation_vector|2.0510309204898196e-12|PASS|
|hadamard_cosine|1.1957101975212936e-13|PASS|
|compiled_vector_cosine|1.504043707445665e-9|FAIL|

判定は各errorがfiniteかつ**1e-9未満**。最後の値が超過したため、cosineのcompile予約は消費済みのまま費用を未受理とし、sineは予約・compileしていない。該当点のsignalはcompile前に保存されているが、task全体のPASS証拠にはならない。[STOP projection](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/STOP_projection.json)はprivate原本SHAと全検査値を保持し、private pathを含むtracebackは原本で保全する。

これはcompiled状態の検証不一致という観測であり、原因を丸め誤差・global phase・compiler故障のいずれかへ確定したものではない。基準を事後的に緩めず、失敗点をPASSに変更しない。係数の次数条件の判定や、PRの費用優位性の否定・保証とは区別する。

## 終了監査と保全

全29 PASS taskの両軸cost原本・result・ledgerを照合し、task/axis/SOURCE/plan/fingerprint/compiler/native policy、非負integer費用、代表sampleと費用専用sampleを確認した。全保存検査値は190件（189件PASS、上記1件FAIL）。これは同一pilot内の検査件数であり、独立190実験や独立reviewではない。代表sample0は23 task全検査PASS、残り1代表はcosineでFAIL。sample1の6本は費用取得専用で、代表と同じ全検査を行ったと表示しない。

4plan、SOURCE16 file、provider/runtime、保存入力、seed封印・導出と原proposal02のDRAFT/INVALID grantが実行前後で不変だった。今回の有効grantは独立authorization_run01で保存し、消費marker・59予約・原STOPは保持した。closureの共通fileを共有した4laneも、それぞれのscopeで照合した。他worktreeへ書込・signal・freezeは行っていない。共有OS/package/cgroup設定の変更もない。

pool witnessは全24 worker exit0/reap、外部wait4 witnessはlauncher exit1/reapを記録した。exit1は今回のlane別検証STOPを反映し、worker crashを示すものではない。SOURCE整合の成功と全pilot数値検証の成功を混同しない。

公開bundleは58受理費用軸を含む60予定軸の状態CSV、190検査値、29 result、24 signal原本、lane別completion、2 reference receipt、59 ledger、STOP projection、承認/有効grantの公開projection、private raw inventory SHA、保全receiptを持つ。入力NPZ・seed値・private events・private config・private tracebackは公開しない。資料の構造/hash/privacy照合は作者自身の技術確認であり、別reviewerによる独立検証ではない。

## 得られた範囲と次の担当

固定q2のtechnical pilotでH4全lane・H6 PR・H6決定論8点のnative経路と両軸単価が取得できた。一方、H6 full m5点は未合格で、pilot全体を60軸PASSとは扱えない。signal24件は経験的diagnosticのみ、shot allowance/headroom/厳密bias certificateはnullのまま。shot会計、総費用順位、model fit、希少層保証、5精度比較、独立確認は行っていない。m5主比較採用AD03と後続Coreは保留。

**Codex（次の技術作業案、未実施）**：保存STOPと封印SOURCEから不一致原因の静的調査・修正候補を整理する。新しい数値検査・sampling・compile・閾値変更は含めない。

**ユーザー（次の判断）**：その修正案と追加検証範囲を確認し、必要な数値検査・再compileを別途承認するか判断する。現行のSTOP・成功済み58軸・旧grantを保全し、成功点を自動で再取得しない。

今回の結果はlocal commit後mandatory STOP。公開pushと後続科学計算は行わない。
