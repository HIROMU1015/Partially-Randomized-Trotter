# M1-S Gate 0 v4：絶対oracle残差品質と条件付きcoverage

**候補のみ・未採択。CONDITIONAL_ENGINEERING_NOT_CERTIFIED。新科学計算・算定式数値評価・数値correctness testは全て0、mandatory STOP。**

## 1. 改訂の根拠と適用範囲

[GPT限定科学レビュー原本](../../research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md)を受領し、
ユーザーがv4のSOURCE・文書・静的検証・公開を承認した。レビュー推奨も今回の作業承認も最終契約の採択ではない。
基点は `e749bf8844f5d78817f18f738ade87269c2d70e1`。
[v3契約](m1s_gate0_headroom_execution_contract_v3_candidate_20261016.md)、
[v3誤差式原本](m1s_gate0_preregistered_error_formula_design_20261016.md)、全旧SOURCE/結果/採択記録/auth/markerを保全する。

科学scopeはlinear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12、保存順G3/G6/G9、非冗長Pauli-tail、保存数値oracle。
delta windowなし（finite-step評価なし）。JW signed aggregate、cutoff0、identityをHDへ移すlawと既存比較を維持する。
H4 holdout、新rank、T1c/Q4/Q5、finite-step dressing、simulation/compile、総資源評価へ拡張しない。

## 2. 残差gate：品質成分と演算誤差成分を分離

v3の `eta_oracle*S+B_res` を廃止し、新v4では次を候補とする。

```
quality_policy = absolute_original_solver_acceptance_v1
oracle_absolute_residual_quality_hartree = 1e-8
eps_state_residual = 1e-8 Hartree + B_res
```

1e-8 Haはsolver由来の数値品質policy、B_resは新referenceの条件付き演算誤差予算。
`budget.global_limits()`はこの二つをFractionで加算し、最終値を上向きbinary64へ変換する設計である。
今回その計算は呼んでいない。1e-8にSを掛けず、Sからetaを評価する処理もない。
品質成分は実行契約のschemaで固定し、別の値・裸scalar・旧v3残差IDへの差替えは拒否する。

由来となるSOURCEは、保存development metadataのsource_hashesとbytes/SHA256一致を確認した。

| 出典 | 内容 | SHA256 |
|---|---|---|
| `src/trotterlib/pr2_s0_s1_validation.py:830` | `solve_df_ground_state(..., matrix_free_backend="python", tol=1e-12)`、converged確認後にsnapshotを保存 | `1ccf6505d2933699504db187e3849251d1fef7bc58d8dca0b5f19871896ca6fe` |
| `src/trotterlib/df_hamiltonian.py:750` | `converged and residual_norm <= max(100.0*tol, 1e-8)` | `7d711e1c3c093079ae890ffc1907813d55d211a4fb622e4f36edee05fdea0c52` |

これは旧SOURCEの受理基準を引き継ぐengineering policyである。旧残差値の再解析や、旧算術誤差の厳密な移送を行ったものではない。
PASSしても厳密固有状態への距離、full-Fock ground性、非縮退性、SP条件、真のground上のW/v/rho誤差は認証しない。
レビューが指摘した「eta*Sの品質成分がenergy offsetに依存する」問題は、絶対品質成分を使って解消する。
B_resがSに依存する点は演算誤差の尺度依存として残り、物理的品質との混同を避ける。

## 3. 旧solverと新referenceの算術経路差

| 観点 | 旧生成solver | 新v4 reference／残差gate |
|---|---|---|
| 空間 | physical sector、保存metadataではdimension36 | mode0 MSBのfull-Fock dimension256、exact sector embeddingを別gateで確認 |
| H作用 | Python matrix-free `df_linear_operator` の作用 | occupation-bit one-body matrixと12 DF blockのdense matrix products、独立JW再構成比較 |
| energy | eigshが返したenergy | 保存stateのRayleigh quotient `vdot(psi,Hpsi)/norm2` |
| state | solver内でnormを割ってからresidualを計算、その後保存側でphase canonicalization | 保存stateを上書きnormalize/phase変更しない。明示norm2 quotientを使用 |
| residual | `norm(linop@state-energy*state)` | `norm(Href@psi-E_R*psi)/sqrt(norm2)` |
| 誤差の扱い | 旧受理判定の丸め誤差は今回認証しない | B_resは新経路の条件付き予算のみ。新旧経路差を数値評価していない |

旧gateを通過したことから、新経路でも必ず合格するとは推論しない。
保存後のphase canonicalization、normの扱い、energy、作用経路が異なる。
今回のpolicyはprovenance整合のquality screenであり、新旧の厳密同値や差のcertificateではない。
失敗すれば `ORACLE_QUALITY_UNRESOLVED` を伴うtechnical STOPとして保存し、閾値を緩和しない。

将来の出力では `residual_quality_gate` とglobal error ledgerに少なくとも次を残す。

- quality policy、固定quality component（Ha）、B_res component（Ha）、合計gate limit、評価残差とPASS/STOP。
- `old_solver_vs_new_reference=DIFFERENT_PATHS_NOT_DIFFERENCE_CERTIFIED`。
- `old_solver_residual_arithmetic_error=NOT_CERTIFIED_NOT_TRANSFERRED`、旧残差の再評価なし。
- `eigenstate_distance_certified=false`、`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。

state receiptは残差gateの前に作成し、FAIL時もfinallyで監査出力へ取り込む。
nonfinite診断はstrict JSON内の文字列として保持する設計で、値の修復・gate合格扱い・計算再開はしない。
これらの保存動作もruntimeでは未検証である。

## 4. 他の8式・8 scalar・B_resを維持

eps_H、eps_hermiticity_H、eps_phase、eps_imag_coeff、eps_imag_energy、W_error_budget、v_error_budget、W_floorの
式IDと算術はv3を維持する。B_resを計算するconstructor、rational conversion、gamma、outward conversion、
W/v/floor law resolverは、coverage文字列の変更を除きAST比較で不変を確認した。
**AST一致は誤差式のcoverageや数値correctnessの証明ではない。**

8式の定義・必要入力・導出は[v3式原本](m1s_gate0_preregistered_error_formula_design_20261016.md)の該当式を参照。
v4では同原本の残差eta未決部分と証拠ラベルだけを本書で置き換える。旧文書自体は書き換えない。

| 17項目の区分 | 項目 | v4候補 |
|---|---|---|
| 構造2 | eps_basis、eps_hermiticity_factors | 各0、exact equality |
| 品質scalar3 | eps_state_norm、eps_imag_expectation、eps_prob | 2^-40、2^-40、2^-46 |
| 報告resolution3 | eps_W、eps_v、eps_rho | 2^-24、2^-24、2^-12 |
| 条件付き式8 | 上記8項目 | v3式のまま、観測値からthresholdを作らない |
| 絶対oracle残差品質1 | eps_state_residual | 1e-8 Ha+B_res、品質成分と演算成分を分離 |

全17項目に具体的な値又は式の候補があり、未設定のetaは残らない。**全項目未採択**。
reporting resolutionを科学的有意性・資源改善の閾値へ読み替えない。

## 5. 条件付きcoverageと未検証条件

証拠区分は `CONDITIONAL_ENGINEERING_NOT_CERTIFIED` に統一し、contract、budget ledger、state/row/result、監査に記録する。
通常のbinary64相対誤差モデル、複素BLAS gamma_(8k+32)、fsum gamma_(m+32)×L1、round-to-nearest、
未会計overflow/underflow/FTZなし、exact Hermitian input等は未検証条件として明示する。
NumPy/BLASのversion/build hash一致やFraction・nextafterだけでは、それらの成立・式の網羅性を認証できない。
silent underflowを完全検出する実装でもない。rigorous certificate、true-eigenstate error intervalとは表現しない。
`coverage_model_adopted=false` のままであり、ユーザーによる条件付きモデルの採択を待つ。

input exact Hermiticity FAILは `INPUT_MODEL_NOT_APPLICABLE_TECHNICAL_STOP`。
再構成FAILは `RECONSTRUCTION_UNRESOLVED`、残差FAILは `ORACLE_QUALITY_UNRESOLVED`。
lambda下端非正・eW/evのresolution超過はmetric評価前の `PRECISION_UNRESOLVED_BEFORE_METRICS`。
small denominator/rho精度不足もtechnical precision unresolvedとして記録し、thresholdやlawを変更しない。
これらの失敗を含むrun-level statusは `TECHNICAL_STOP`。rowの詳細NA/UNDETERMINED/UNRESOLVEDを併記する。
数値empty tail／W=0のNAと、精度不足は区別する。rho単独で科学GO/STOPを自動化しない。
global failureでも全G3/G6/G9の未到達statusを保存する。独立splitの処理は固定scopeの範囲のみで、retryではない。

## 6. Runtime・guard・残る採択

CPU1 worker、GPUなし、各thread1、4 GiB AS、result/log各256 MiB、context4 MiB、M65536、C1000000、wall-time capなしを維持。
Python3.11.0rc1はpre-release、NumPy1.26.4と6 build-file hashを固定する候補。
既存cache/共有DF blocks/bit referenceを維持。新しい性能検証・pilot・import smokeは行っていない。
メモリ制限・log capture・例外保存の実効性、数値correctness、全3splitの完遂は未検証。

新v4 authorization schemaを用い、source commit < 採択contract commit < 別実施authorization commit、
clean checkout、scope/input/source/runtime/build/thread identity、exclusive result namespaceを維持。
新authorization/grant/marker/resultは作成しない。旧v3を流用して実行しない。

残るのはユーザーによる17項目・絶対残差policy・条件付きcoverage・runtime/caps・失敗/one-shot/STOPを含む最終契約の一括採択。
追加の科学的変更が必要な場合はその新論点を明示してSTOPする。
採択後も、初回数値correctnessを含むGate 0実施には別の明示承認が必要。mandatory STOP。
