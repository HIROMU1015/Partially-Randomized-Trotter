# 2026-10-16 M1-S Gate 0 v4候補：絶対残差policyへ改訂

ユーザー指定のGPT限定レビュー原本をbytes不変で取り込み、新namespaceでv4候補を準備した。
eta*Sを使わず、保存oracle生成元の絶対quality 1e-8 Haと新reference B_resを別管理する。
旧matrix-free/sector/solver-energyと新dense/full-Fock/Rayleigh経路の差は未測定・未認証と記録する。
8 formulaの算術・8 scalar・科学scope・runtime/capsは維持し、証拠区分をCONDITIONAL_ENGINEERING_NOT_CERTIFIEDへ統一。
input/reconstruction/oracle/precision失敗はtechnical STOP、科学的no-goへ読み替えない。全3行・失敗receiptを保存する設計。

[v4最終契約候補](../../tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_candidate_20261016.md)、
[品質/coverage説明](../../tracks/algorithm_codesign/m1s_gate0_v4_oracle_residual_quality_and_coverage_20261016.md)、
[監査artifact](../../../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1)を参照。
99静的AST/schema checks PASS、数値correctness/runtime/保存経路は未検証。新数値test・科学計算・formula評価・Pauli生成0。
旧SOURCE/結果/採択記録/auth/grant/marker・親worktreeを保全。全17候補が具体化したが正式採択なし、coverage_model_adopted=false。
次はユーザーのv4最終契約採択、その後にさらに別の科学実施承認。新authority/grant/marker0、mandatory STOP。
