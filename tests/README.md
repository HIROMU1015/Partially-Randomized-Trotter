# tests の役割

## 研究C：CDF7 native32の人工検証（2026-10-30）

[新test](management/test_research_c_cdf7_native32.py)／[技術報告](../docs/research/research_c_cdf7_native32_preparation_20261030.md)。人工rank7保存119相当→332 mock/QPY128、実人工128-event full512 raw/compiled/adjoint/X-Y、SOURCE/sector/law/phase不一致、seed分離、fresh STOP、shared-SEを検証。既存CDF7 PR・Qiskit shared-lock・rank8 gateも回帰する。実H4の性能・CA保証を人工PASSから推定しない。

## 研究C：CDF7 PRの人工end-to-end・回帰（2026-10-30）

[新人工11件](management/test_research_c_cdf7_PR.py)＋既存回帰28件＝39 PASS。[技術報告](../docs/research/research_c_cdf7_PR_preparation_20261030.md)。Invented rank7 J/Uで既存CAR/PF/RTE/RPEを用い119 dispatch、63終了、correctness STOP、未資格null、component consumer、input/SOURCE/fresh不一致、native/sample遮断を検証。旧rank8 gateも回帰。実H4の科学的性能やCA達成を示すものではなく、今回実科学call0。

## 研究C：通常CDF7・限定1-fitでDF予算PASS（2026-10-30）

[結果・初期/終了誤差・CDF8比較・後続案](../docs/research/research_c_cdf7_onefit_execution_20261030.md)。保存直鎖H4/1Å/STO-3G、Nα=Nβ2/sector36、通常number CDF7/元順LD2、γ0 normalized tensor loss、ffsim0.0.83/修正codec/L-BFGS-B options=None。SOURCEc13d6f82a無変更で25/25 COMPLETE、correctness FAIL0、rank7 Cholesky初期化1/fit1/retry-resume0。初期sector+guard9.0647035e-4Haは予算未達の診断としてfitを許容し、終了3.02264175e-7HaでCA/8=.00019875Ha PASS。Optimizerは215iterの相対目的減少停止、正規化gradient∞=.05556で停留/大域保証なし。人工45 PASS＋29 subtests、旧保護2153件不変。PR/PF-RTE/ground/native/shots/総資源0。後続119-call案はrank7 adapter/SOURCE未固定でexecutable=false、native32は別案。公開照合後mandatory STOP。


## 研究C：通常CDF7の限定1-fit準備（2026-10-30）

[実装・人工45 PASS・固定25-call](../docs/research/research_c_cdf7_onefit_preparation_20261030.md)。新専用module/CLIは既存ffsim Cholesky initializer・γ0 loss・修正codec・SectorCARを再利用。初期予算は診断のみ、表現correctness後に1-fit。保存H4/1Å/STO-3G、sector36、rank7/LD2、options=None、最終CA/8、fresh run01/retry-resume0。SOURCE35/input8/旧保護2153件固定、後続PRは技術案のみ。Shared-basisのexecutable=false契約は保全。


## 研究C：Shared-basisは通常低rank CDFと同値（2026-10-29）

[定義・証明・native対応・61人工PASS](../docs/research/research_c_shared_basis_df_design_20261029.md)。一般対称Jの共有G groupは通常CDF≤Gと同じ表現集合で、CDF8のtail F2/F3共有はCDF≤7に対応。冗長Jと通常rank削減を新DF方式と呼ばず、専用fitter/重複fitはNO-GO提案。固定H_D/H_R・h/r/Kで現partial PF/補正RTEは不変だが、atom相殺でλ/law/減衰は変化し得る。Signed-zeroのtail keyと決定論tuple判定、head相対融合とtail共有を区別した。新21＋回帰40＝61 PASS＋29 subtests、旧SOURCE/原本2140件照合、実H4 decode/fit/新再解析/build/compile/科学call0。非実行の未承認案はcall0/executable=false。通常CDF7の物理予算は未評価、次方向は未採択。公開照合後mandatory STOP。


## 2026-10-27：固定CAR reference・Proposed検証の技術準備

`management/test_research_c_pr_aware_df_fixed_car.py` は人工J/U・event・QPYだけの27試験。既存関連143と隔離旧general-K/native40もPASS（計210、別に29 subtests）。旧general-Kのsys.modules隔離前提のため新Qiskit試験とは別processで実行する。[記録](../docs/research/research_c_pr_aware_df_fixed_car_reference_readiness_20261027.md)。旧C2並列STOPの非決定性は未修正。

`management/test_research_c_c1_stage2_native.py` は[C1 Stage2接続準備](../docs/research/research_c_c1_stage2_native_preparation_20261025.md)の新規27人工test。実QuantumCircuit／Operator／Statevector／compileをmockまたは禁止し、係数・phase・spin・順序・API dispatch・call／fresh output／STOPを確認する。既存一般K／CAR／frame修正／評価専用testsを含む80件＋29 subtests PASS、分子回路の科学的PASSではない。証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`。

`tests/` は、ライブラリAPI、数値恒等式、成果物schema、ガード条件の回帰検査を置く。
基本的に `src/trotterlib/<name>.py`、`scripts/run_<name>.py`、
`tests/test_<name>.py`、`docs/<name>.md`、`artifacts/<name>/`を一組として読む。

テスト通過は、実装が期待した規約を満たすことを示す。一方で、次を単独では意味しない。

- 大きな系でも同じ近似精度になること
- 実量子backendやnoise下で成立すること
- 未検証のパラメータ範囲への外挿が正しいこと
- 最終的な総コストや手法間の優位性が確定したこと

研究上の証拠statusは[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を参照する。

最新のRPE長round検証は`test_rpe_target_round_horizon_validation.py`と
`test_rpe_delta_round_schedule_validation.py`、`test_rpe_delta_compiled_cost_validation.py`を、
対応する検証文書・artifactと一組で読む。最後の検証は中央RTEブロックだけを扱い、
Hadamard 1 shot全体や最終総コストの検証ではない。

`management/test_pr_dictionary_f2_attribution.py` は、保存F2寄与分析の分類保存則・共有metadata・joint origin・欠落／重複拒否・非零の記述分解を人工recordだけで検査する。新しい科学回路は生成しない。[結果とSOURCE対応](../docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)。
