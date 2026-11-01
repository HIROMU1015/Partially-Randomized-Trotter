# 2026-11-02：P3 Stage 3 run02のnative回復と部分価格

ユーザーの追加実行指示により、旧失敗`aab0b483fdd8e57fe17305a11c5cb750a9c80a4d`を保持して、
新SOURCE`9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`から一回だけ新attemptを行った。
project内Givensの近零・微小角処理だけをexact-zero/atan2へ修正し、共有library/旧runner/kernelは変更していない。

[結果・価格・未取得対象・保全](../../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run02/REPORT.md)。
同じ厳しいd8基準でone-particle差は約4.05e-16、共通保持/DF basis/scalar/18 bodyの81項目PASS。
人工21PASS、H4/1Å/STO-3G/full256/保存DF rank10/案A/T.8/S2/q8/r4/K4/M_B/M_Cを維持。
8 trajectory/256 event位置を一度取得し、36compile中23件を保存。
Qiskitのshared equivalence libraryで`RuntimeError: Already mutably borrowed`が生じて停止した。
同一processのthread並列方式に問題があり、残り13件の価格は未取得。

旧run01のFAILを上書きせず、run02の入力/event/QPY/部分metricsも可逆archiveへ保存。
Stage 2有限bias UNRESOLVED、母平均価格・全lawρ・N/G等は未評価を維持。
自動retryは行わず、次案を「保存済みinputの未取得13件だけを独立processでcompile」として具体化した。
新sampling・native再構築・既存23件の再compileを自動追加しない。
公開照合後mandatory STOPし、次はユーザーがその限定回復を実行するか判断する。
