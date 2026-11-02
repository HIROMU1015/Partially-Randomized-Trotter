# 2026-11-02：P3 Stage 3の共通basis preflight停止

ユーザーの実行許可により、契約`be3de04e6fb501fecc4a07f38ceea7c24ba10472`、
科学SOURCE`d3c258eb019e59234a874e1d8458e383f8317d68`から固定H4の1batchを起動した。
H4/1Å/STO-3G/中性singlet/system8/full256、保存DF tol-only1e-8/rank10、案A/T.8/S2/q8/r4/K4/M_B/M_C。

[失敗数値・全件数・未到達範囲](../../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run01/REPORT.md)。
共通T_eff native Gaussian basisのone-particle差は2.376341682004808e-13、
d8の事前固定FAIL境界1.1368683772161626e-13を超えた。他4 basis検査はPASS。
保持半指数を生成する前に停止し、sampling/trajectory/compileは0。
閾値変更・gauge修復・dense fallback・科学retryは行っていない。

人工検査は最終13PASS。初回の人工DF測定wrapper compileで全体位相差を確認し、
測定observableでの回帰検査へ修正した経緯は科学SOURCEの実装文書に保存した。
その人工PASSから実H4 native basisのPASSを推定しない。

失敗後の調査は保存値とSOURCEの静的読取だけ。qiskit-natureの近零判定等は調査候補であり、原因は未同定。
価格・資源順位は未取得。Stage 2の既存結果と`STOP_STAGE2_UNRESOLVED_COMPONENTS`は保全。
`STOP_STAGE3_PREFLIGHT_OR_EXECUTION_FAILURE_NO_RETRY`とし、公開照合後mandatory STOP。
次はユーザーが原因調査・実装/数値政策変更と必要な再実行の範囲を判断する。
