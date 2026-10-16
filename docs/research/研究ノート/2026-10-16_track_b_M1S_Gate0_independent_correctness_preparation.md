# 2026-10-16：Gate 0 v5独立correctness検証の準備

ユーザーは実施準備のみを承認し、新規科学計算・実/模擬入力での数値テスト・新authority/marker発行を禁止した。
既存の未完成d6d35cfb草稿4ファイルを別worktreeで再利用し、元worktreeを変更せず監査・補完した。
整数occupation/CAR/Pauli traceを保持し、全state/metricをexact rationalへ変更。Decimalは表示のみ。
SOURCE4787cf90固定、result cdcb7f6f・raw hash 8e32925e…へbinding。旧SOURCE helper非利用、result parseはreference全3行完成/hash後のみ。
旧rawのfull-H/HD行列欠落はUNDETERMINEDとして残し、全scope PASSを許さない。既存数値結果は変更しない。
[独立検証契約候補](../../tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_candidate_20261016.md)を作成し、静的検査・公開照合を行う。
新科学計算0、実/模擬入力test0、実行authorization/grant/marker0、mandatory STOP。次はユーザー契約採択と別途明示実施承認。
