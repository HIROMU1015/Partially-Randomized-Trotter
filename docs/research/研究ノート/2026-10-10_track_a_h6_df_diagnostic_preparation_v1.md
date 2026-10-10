# 2026-10-10 Track A H6 DF診断準備

ユーザー指示に従い、既存証拠の確認と[診断source/plan/seal](../track_a_h6_df_hermitization_diagnostic_preparation_v1.md)だけを準備した。
ローカル別worktreeも検索したが失敗rawを回収できず、旧STOP・欠測を保持する。
全returned rawを先に保存し、lambdaと非Hermiticity、normal-order coefficient影響を分離する新診断を設計。
43 local synthetic tests pass。CPU2・480秒/AS8GiB/output32MiBをsealしたが新grantなし、runner/実DF実行0。
DF政策・target/state・H6 GOを決定しない。source/旧integrals/旧結果/dirty・Track Bを保全し、実行認可待ちSTOP。
