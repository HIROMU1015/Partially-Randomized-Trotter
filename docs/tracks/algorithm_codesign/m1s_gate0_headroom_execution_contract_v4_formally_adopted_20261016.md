# M1-S Gate 0 最終実験契約 v4：ユーザー正式採択（2026-10-16）

**ADOPTED_BY_USER / SCIENTIFIC_EXECUTION_NOT_AUTHORIZED / mandatory STOP。**
証拠区分は **CONDITIONAL_ENGINEERING_NOT_CERTIFIED**。SOURCEはIMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTEDのまま。

## 1. 正式採択と承認境界

ユーザーはcommit `44277e03448ee51dbe9afdd2dc6560337757cc6d` のv4最終実験契約を、17項目・残差品質・条件付きcoverage・runtime制限・one-shot/failure STOPを含め正式採択した。
採択原文：

> commit `44277e03448ee51dbe9afdd2dc6560337757cc6d` のv4最終実験契約を正式採択します。17項目、残差品質基準、条件付きcoverage、runtime制限、one-shot・failure時STOPを含め、正式契約と採択記録の固定・公開を進めてください。科学計算は承認しません。

[採択記録](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)、[正式machine契約](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json)、
[候補との差分4項目](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/candidate_to_adopted_semantic_diff.json)、
[SOURCE・入力・契約identity](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/source_input_contract_identity.json)を参照。
基点repository `HIROMU1015/Partially-Randomized-Trotter`、基点branch `track-b-m1s-gate0-v4-contract-candidate-20261016`。
採択固定branch `track-b-m1s-gate0-v4-formal-adoption-20261016`。
[共通研究指示2026-10-16](../../research/research_project_instructions_unified_2026-10-16.md)を適用。

採択対象は[固定v4候補](m1s_gate0_headroom_execution_contract_v4_candidate_20261016.md)と
[品質/coverage説明](m1s_gate0_v4_oracle_residual_quality_and_coverage_20261016.md)。
既存候補・SOURCE・レビュー・監査・scope採択記録を変更しない。
新正式machine契約で変えるのはstatus、adopted、coverage_model_adopted、adoption_recordだけ。
科学scope、17項目の数値/式、runtime/caps、SOURCE/input hash、比較/STOP規則は候補と同一。

## 2. 採択済み科学scopeと固定入力

linear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12、保存順prefix G3/G6/G9、非冗長Pauli-tail、保存数値oracle。
delta windowなし（finite-step評価なし）。mode0 MSB、alpha-even/beta-odd。
保存DF順、JW signed aggregation後abs、IXYZ lexicographic、cutoff0、identityをHDへ移すlawを維持。
入力 `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`、8652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
declared H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` は保存値で、今回数値再計算していない。
SOURCE commit `606036db52da2f080cf846e48824e5e418430762`、7 SOURCE/dependency filesのhashを正式machine契約に固定する。

保存stateを上書きnormalize/phase変更せずnorm2 quotientを取る。
W=1−〈h²〉、v=Var〈signed P〉、rho=(W+v)/(2W)、h=HR/lambda_R。
同じideal taskでR conceptual event positions：shared/ordinary r=R、independent各branch R/2（R even）。
lambda²/Rあたりの先導係数はshared (W+v)/4、ordinary/independent W/2。
これは保存oracleのheadroom診断で、full-Fock ground・非縮退性・SP・state距離・finite-step/RPE/native資源優位を認証しない。
T1c/Q4/Q5、holdout、新rank、総費用評価等は含めない。

## 3. 採択済み17項目

[正式17項目仕様](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/precision_specification_17_adopted.json)を参照。

| 区分 | 項目 | 採択値・式 |
|---|---|---|
| 構造2 | eps_basis、eps_hermiticity_factors | 各0、exact equality |
| scalar品質3 | eps_state_norm、eps_imag_expectation、eps_prob | 2^-40、2^-40、2^-46 |
| 報告分解能3 | eps_W、eps_v、eps_rho | 2^-24、2^-24、2^-12 |
| 条件付き式8 | eps_H、eps_hermiticity_H、eps_phase、eps_imag_coeff、eps_imag_energy、W_error_budget、v_error_budget、W_floor | v4候補のformula ID/算術をそのまま採択 |
| oracle残差1 | eps_state_residual | **1e-8 Ha+B_res**、absolute_original_solver_acceptance_v1 |

式の詳細は[v3事前登録式原本](m1s_gate0_preregistered_error_formula_design_20261016.md)の8式と、v4品質/coverage説明の残差変更を一体として用いる。
1e-8 Haは旧solverの数値受理policyで、B_resは新referenceの条件付き演算誤差。両成分を別管理する。
旧sector matrix-free/solver energy/保存前phaseと新full-Fock dense/Rayleigh/保存norm2の経路差は未測定・未認証。
旧gate通過から新gate合格を保証しない。品質PASSから厳密固有状態への距離を推論しない。

## 4. 条件付きcoverage・失敗・STOPの採択

`coverage_model_adopted=true` は条件付きengineering評価としての方針受理で、IEEE binary64、BLAS、math.fsum、
rounding、underflow/FTZ等の仮定が検証済みという意味ではない。証拠区分はCONDITIONAL_ENGINEERING_NOT_CERTIFIEDを維持する。
Fraction/nextafter、固定hash、過去の静的99 PASSはrigorous numerical certificateやruntime correctnessの証拠ではない。
静的99 PASSと資料214 PASSは旧候補準備の記録を保持し、今回それら全体を再実施したとは扱わない。

input exact Hermiticity、再構成、oracle残差等のgate FAILは証拠を保存してtechnical STOP。
metric前にlambda/予算を判定し、eW/ev超過ではW/v/rhoを未評価のPRECISION_UNRESOLVEDとして残す。
small denominator/rho precision不足も技術的不確定で、研究上のno-goやrho=1へ読み替えない。
閾値緩和・law変更・入力変更・自動retryで継続しない。rhoのみの科学GO/STOPは禁止。
G3/G6/G9を全行status付きで記録し、global failureでは未到達を明示する。
run後は成功・失敗のいずれでもmandatory STOP。scope拡張や新しい科学判断は別承認へ戻す。

## 5. Runtime・resourceの採択

Python3.11.0rc1（pre-release）、NumPy1.26.4、既存6 build-file SHA、CPU1 worker、GPUなし、各種thread1。
4 GiB RLIMIT_AS、result/log各256 MiB、context JSON4 MiB、M65536、contribution cap1000000、scientific wall-time capなし。
既存cache/DF block共有・mask cacheを維持。runtime、BLAS経路、capの実効性、保存経路、数値correctness・完遂は未検証。
今回の採択固定ではruntime smoke・benchmark・模擬/実入力の数値testを行わない。

[採択static checker](../../../scripts/tracks/algorithm_codesign/check_m1s_gate0_v4_adoption_static.py)はJSON・raw bytes・hash・guard構造だけを検査し、科学処理を実行しない。

## 6. 採択と実行許可の分離

正式machine契約のstatus=ADOPTED、adopted=true、coverage_model_adopted=trueとし、採択原文をadoption_recordへ固定する。
既存の `validate_config` が返す **execution_ready=trueは契約JSONの完備性**を意味するだけで、科学実行の許可ではない。
今回の監査ではcontract_configuration_completeとactual_execution_permittedを別々に記録する。

将来実行にはSOURCE commit < 正式契約commit < 別実施authorization commit、clean dedicated checkout、
全scope/input/source/runtime/build/thread identity、exclusive result namespaceが必要。
schema `m1s_gate0_separate_execution_authorization_v4` の実行承認記録は今回作成しない。
本採択記録、正式契約JSON、既存scope採択記録は実行authorizationにならない。
初回数値correctnessを含め、Gate 0実施にはさらに別のユーザー明示承認が必要。

**契約正式採択済み・科学計算未承認。新科学計算/Pauli生成/数値test/算定式評価/実行authorization/grant/markerは全て0。mandatory STOP。**
