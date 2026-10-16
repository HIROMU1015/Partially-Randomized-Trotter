# Gate 0 v5 精度契約の設計候補：tail局所上界（2026-10-16）

**DESIGN_CANDIDATE_NOT_EXECUTABLE_NOT_ADOPTED。正式な実行契約ではなく、SOURCE接続前の精度設計候補。**
新モデルのcoverage_model_adopted=false、scientific_execution_authorized=false、execution_ready=false。
旧v4の採択・TECHNICAL_STOP・null結果・消費済みone-shotはそのまま保持する。

対象はH4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12、HD=保存順G3/G6/G9、非冗長signed Pauli tail、
保存数値oracle、delta/finite-stepなし。新しい科学条件・精度ceiling・比較方式を採択しない。

[条件付き証明](m1s_gate0_tail_local_bound_proof_20261016.md)、
[machine-readable設計候補](../../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/precision_contract_v5_design_candidate.json)、
[演算経路台帳](../../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/tail_local_model_candidate.json)、
[宣言的spec](../../../src/trottertracks/algorithm_codesign/m1s_gate0_tail_bound_spec.py)を一組で扱う。

## 変更案

global係数誤差alphaをHR lawへ流用する代わりに

    G_l = sum_pq (abs(Re g_lpq)+abs(Im g_lpq))
    S_R = sum_l_in_tail abs(lambda_l)*G_l²
    alpha_R = gamma_(713+r)*S_R

を導入する。r=9/6/3でindex722/719/716。raw保存binary64値はexact入力として扱い、
将来resolverはFractionで尺度・予算を計算し、最後にbinary64へ上向き変換する設計。
observed reconstruction差、residual、W/v/rho、実測counter27,781をこの式の入力にしない。

既存law_limitsのL・lambda誤差・ec・matrix assembly・dot・variance・rho floorの代数は維持し、
alphaだけをalpha_Rへ置き換える。M=65536も維持する。
新allowlisted formula IDはW_forward_error_tail_local_v1、variance_forward_error_tail_local_v1、rho_resolution_floor_tail_local_v1。
任意expression評価や観測結果からのthreshold調整を実装しない。

## 精度17項目の扱い

|項目|設計候補|
|---|---|
|eps_basis、eps_hermiticity_factors|旧exact equality（0）のまま|
|eps_state_norm、eps_imag_expectation、eps_prob|旧2^-40、2^-40、2^-46のまま|
|eps_W、eps_v、eps_rho|旧2^-24、2^-24、2^-12のまま|
|eps_H、eps_hermiticity_H、eps_phase|旧global式・旧full-H尺度を維持|
|eps_imag_coeff、eps_imag_energy|旧global式を維持|
|eps_state_residual|旧absolute_original_solver_acceptance_v1、1e-8 Ha+B_resを維持|
|W_error_budget、v_error_budget、W_floor|同じ伝播代数をlocal alpha_Rで解く新selector候補|

global residual/reconstruction/phase/error ledgerと、HR local coefficient/law ledgerは別キーで保存する設計。
HR identityの除去はnonidentity係数を丸めない。HD scalarへの加算誤差はglobal ledgerのままにする。
local alphaで旧global許容値を狭めたり、旧結果を再分類したりしない。

## 精度要求を維持できる解析的領域

証明ではS_R/lambda_hat<=8、L/lambda_hat<=2の領域で、上記indexと旧伝播式からeW、evがともに2^-24以下となる十分条件を導いた。
このratio条件は新しい実行gateや結果選択規則ではない。外側の領域でも局所式は有効であり、予算超過なら従来通り停止する。
実入力へ新式を代入した値は今回取得していない。
rhoの小分母、Pauli probability/expectation品質、全17gate、runtime完走までは保証しない。

## Runtime、SOURCE接続、実行防止

CPU1 worker、GPUなし、thread5項目各1、4 GiB address space、result/log各256 MiB、context4 MiB、
Pauli cap65536、global contribution cap1000000、scientific wall capなしを維持。
runtime/build pinも旧契約のまま。count713+rはroundoffの経路indexであり、resource capを引き下げる値ではない。

今回のspecは整数・文字列・状態flagの宣言だけで、resolverもGate0 runnerも持たない。
machine候補も既存v4実行schemaとは異なり、旧runnerへ与える正式契約ではない。
新しい実行SOURCEのcommitはnull、SOURCE接続は未実装であり、この設計候補を直接採択・executeできる状態と表示しない。

今後の実装では、独立namespace、schema/guard、SOURCE/input hash、global/local台帳を接続し、
source固定→完全な契約のユーザー採択→別科学実施承認を揃える必要がある。
新しいprecision resolverの最初の実入力評価と数値correctnessも、その別承認の範囲へ明記する。
旧source、authorization、result directory、markerを再利用・変更しない。

## 未決事項と完了状態

[未決一覧](../../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/unresolved_items.json)にSOURCE接続、platform coverage、
初回数値correctness、正式採択、rho/全gateの未検証を記載した。
係数上界は旧モデルを前提に条件付き導出済み。実測counterの保証値化や閾値緩和を残していない。
新しいGPT詳細レビューは開始せず、厳密有理数によるPauli独立再構成も行わない。

[静的checker](../../../scripts/tracks/algorithm_codesign/check_m1s_gate0_tail_bound_static.py)はAST/schema/countと2冪不等式を整数だけで確認する。
数値correctness test、実入力・合成入力の予算評価、科学計算、simulation、compileは0。
[evidence index](../../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/evidence_index.json)、
[publication receipt](../../../artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/publication_receipt.json)を参照。
CONDITIONAL_ENGINEERING_NOT_CERTIFIED、候補未採択、mandatory STOP。
