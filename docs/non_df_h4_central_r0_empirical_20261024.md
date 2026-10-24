# H4 Pauli central R0：binary64経験的実装残差診断

## SOURCE固定前

基点 `2734b97c2dd5f017e93cd81619ff52a6a0c6f16b`、専用branch `codex/non-df-h4-central-r0-empirical-20261024`。
固定Linear H4/1Å/STO-3G/n8+ancilla8、Pauli184/prefix27、S2/T=.8/q8/r4/K4。
D2保存PF・PR index0のcosineに限定し、新状態作用最大5。
新10＋既存15人工PASS。実QPY8個をdecodeしSOURCE token・phase・wrapper・input/output fingerprint・nativebasis・measurementを照合、状態作用0。
PF C-targetは保存intervalの同一ray scalar変換、PR0は保存event列のC整合vector作用。
新SOURCEをcommitし、runtime/input/output schema/承認と未使用grantを結合した後にだけ状態作用を行う。

runtime metadataの参照先をA_AXES直下と仮定した処理が失敗したため、既存sealのruntime_manifest参照を使用するよう修正。
SOURCE前・science grant0・状態作用0であり、失敗logを保存してmetadataだけ再実行した。

出力はEMPIRICAL_ONLY_NO_RHO_UPPER、numerical_error_upper/rho_upper=null。
PF保存区間からexact scalar midpointを取り、PR0 C-targetはbinary64点評価とする。
Arb/precision escalation/R1/R2/sine/新compile/sampling/Hamiltonian-state生成/C-Lフル計算/N_G/正式受理0。
旧hash FAIL/原因UNKNOWN・元native UNKNOWN・旧N/G null・C/L/旧compiler/PR全law/tailを保持。

[承認原文](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/approval.txt)、
[GitHub照合](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/github_verification.json)、
[構造照合](../artifacts/non_df_h4_central_r0_empirical/2026-10-24/structural_preflight.json)。

最新版規約§8.4/10.1/10.2を適用。§10.3は利用可能版に存在せず、今回の直接指示を適用する。
単一workerで固定5program walkを行う。入力QPY・state・runtimeを共有して再読込を抑え、CPU/RAM/threads/affinity上限や共有環境変更は加えない。
完了後mandatory STOP。
