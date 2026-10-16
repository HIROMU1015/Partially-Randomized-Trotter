# M1-S Gate 0 最終実験契約 v4 候補（2026-10-16）

**CANDIDATE_NOT_ADOPTED / IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED。CONDITIONAL_ENGINEERING_NOT_CERTIFIED。**
新科学計算・Pauli生成・算定式数値評価・数値correctness・simulation・compileは全て0。mandatory STOP。

## 1. 根拠・固定SOURCE・承認境界

Repository `HIROMU1015/Partially-Randomized-Trotter`。
基点branch `track-b-m1s-gate0-formula-contract-20261016`、commit `e749bf8844f5d78817f18f738ade87269c2d70e1`。
新branch `track-b-m1s-gate0-v4-contract-candidate-20261016`、SOURCE commit `606036db52da2f080cf846e48824e5e418430762`。
[GPTレビュー原本](../../research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md)をbytes不変で取り込み、ユーザー承認の技術改訂を実施した。
[共通研究指示2026-10-16](../../research/research_project_instructions_unified_2026-10-16.md)を適用する。
レビュー推奨・v4作成承認は、本契約の正式採択や科学実行承認ではない。

[machine契約候補](../../../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1/execution_contract_v4_candidate.json)、
[SOURCE identity](../../../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1/source_identity.json)、
[17項目表](../../../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1/precision_candidate_table.json)、
[未決採択・未検証条件](../../../artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1/unresolved_items.json)を一体として扱う。
旧v3契約・SOURCE・scope採択記録、全旧成果/auth/grant/marker・親worktreeを保全する。

## 2. 科学scope・入力・比較

採択済み限定scopeはlinear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12、保存順prefix G3/G6/G9、
非冗長signed Pauli-tail、保存数値oracleのheadroom診断。delta windowなし（finite-step評価なし）。
入力 `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`、8652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、
H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
mode0 MSB、alpha-even/beta-odd、保存DF順を用い、cutoff0・signed aggregation後abs・IXYZ lexicographic、identityをHDへ移す。
旧stateをnormalize/phase変更せず、期待値は保存norm2でquotientを取る。
full-Fock ground、非縮退性、exact eigenstateへの距離、SP条件は認証しない。

W=1−〈h²〉、v=Var〈signed P〉、rho=(W+v)/(2W)、h=HR/lambda_R。
同じideal taskでR conceptual event positions：shared r=R、ordinary r=R、independent各branch R/2（R even）。
lambda²/Rで正規化した先導係数はshared (W+v)/4、ordinary/independent W/2。
native shots/費用優位、finite-step signal、総資源結論を意味しない。T1c/Q4/Q5・holdout等へ拡張しない。

## 3. 17項目と残差gate

[残差品質・coverage説明書](m1s_gate0_v4_oracle_residual_quality_and_coverage_20261016.md)を規範的なv4差分とする。
他8式の数式は[v3事前登録式原本](m1s_gate0_preregistered_error_formula_design_20261016.md)を維持し、旧原本は改変しない。

- 構造2：eps_basis=eps_hermiticity_factors=0、exact equality。
- 品質scalar3：eps_state_norm=eps_imag_expectation=2^-40、eps_prob=2^-46。
- 報告分解能3：eps_W=eps_v=2^-24、eps_rho=2^-12。
- 条件付き8式：eps_H、eps_hermiticity_H、eps_phase、eps_imag_coeff、eps_imag_energy、W_error_budget、v_error_budget、W_floor。
- oracle残差1：**eps_state_residual=1e-8 Ha+B_res**、ID `absolute_original_solver_acceptance_v1`。

品質成分は旧solver `max(100*tol,1e-8)`（保存生成tol=1e-12）から継承した固定policy。
新v4 schemaはこのquality value、残差formula ID、evidence labelを固定し、裸scalarや旧eta*Sへの差替えを拒否する。
B_resは新referenceの条件付き演算誤差成分として別管理し、結果に合わせて1e-8又は予算を緩和しない。
全17欄に具体的な候補がありeta未決は解消したが、**17項目全体は未採択**。

旧経路は36次元sector matrix-free作用とsolver energy、正規化後residual、保存前phase canonicalization。
新経路は256次元occupation-bit dense referenceとRayleigh quotient、保存state/norm2を用いた残差。
新旧の算術差・旧residualの演算誤差は未測定・未認証で、新gate合格を保証しない。
品質gate PASSはground/state-distance/theorem適用の認定ではない。

## 4. 条件付きcoverage・出力・failure意味論

証拠区分は `CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。
IEEE binary64／rounding、complex BLAS、math.fsum、未会計underflow/FTZなし等は未検証の前提である。
Fraction、nextafter、hash固定、静的PASSはそれらの成立やrigorous numerical certificateを証明しない。
`coverage_model_adopted=false`、`adopted=false`、`adoption_record=null`を維持する。

入力h/gのexact Hermiticity不成立はtechnical STOP、修復・projectionによる継続なし。
再構成FAILはRECONSTRUCTION_UNRESOLVED、残差FAILはORACLE_QUALITY_UNRESOLVEDとして保存する。
global raw尺度はH構築/energy/residualより前、law予算はhpsi/Pauli期待値/W/v/rhoより前に解決し、ledgerをmetric前にhash化する。
lambda下端非正、eW>eps_W又はev>eps_vは、そのsplitのW/v/rho未評価でPRECISION_UNRESOLVED_BEFORE_METRICS。
small denominator/rho分解能不足は詳細statusを残しtechnical STOP。NAとUNDETERMINEDも区別する。
rho pointをclampせず、外向きrectangle intervalの判定を維持し、rho単独で研究GO/STOPを決めない。

G3/G6/G9の全行を保存する。global failureで未到達ならNOT_EVALUATED_GLOBAL_TECHNICAL_STOPを明記する。
残差receiptはgate前に作り、FAIL時もquality component・B_res・limit・評価値を監査出力に残す設計。
nonfinite失敗診断はJSON文字列として保持し、成功値にはしない。保存の実効性も未検証。
完了/technical STOPのいずれでもretry/入力変更/別threshold実行を自動開始しない。

## 5. Runtime・resource・実行guard

Python3.11.0rc1（pre-release）、NumPy1.26.4、既存6 build-file SHA、CPU1 worker・GPUなし、各種thread1。
4 GiB RLIMIT_AS、result/log各256 MiB、context JSON4 MiB、M65536、contribution cap1000000、scientific wall-time capなし。
既存cached bilinear/DF map/reference・mask cacheを維持し、未承認benchmarkやparallel tuningを行わない。
runtime import smoke、BLAS経路、capの実効性、数値correctness、完遂・速度は今回未検証。

新[v4 runner](../../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v4.py)のdefault/status/schema/validate-configは科学処理なし。
実行はsource commit < 採択contract commit < separate authorization commit、clean dedicated checkout、全identity照合が必要。
separate schemaは `m1s_gate0_separate_execution_authorization_v4`。v3 authority／scope採択記録は実行承認にならない。
resultは新namespaceへexclusive作成し、既存directoryへのretry/overwriteを拒否する。
今回、新authorization/grant/marker/resultは0。将来最初の数値correctness gateを含む実行も別承認を要する。

## 6. 残る採択とSTOP

ユーザーが、絶対残差policy・条件付きcoverage・17項目・runtime/caps・全3行/one-shot/failure/STOPを含む本v4最終契約を正式採択する。
現在は候補資料準備完了・実行不可。初回数値correctnessとGate 0計算には、採択後さらに別途明示承認が必要。
新しい科学的重要問題が見つかった場合はその論点を明示しSTOPする。忠実な実装のみに追加詳細レビューを自動要求しない。
**新科学計算0、mandatory STOP。**
