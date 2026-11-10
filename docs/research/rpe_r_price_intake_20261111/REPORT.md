# R可変compiled価格のB受入interface：C1 Phase 1価格recordの統合準備（2026-11-11）

Cの`describe_request`と固定C1 PLANを、最新B（`d5d9823e17ecf78a82216fc02b7fcc13a21bb2cb`）のcandidate identity・law binding・価格record形式へ接続した。C1の校正・独立検証・model予測を、公開後にpinned commit/pathから受け入れられる。

- 結果は130技術test（129 PASS、既存の任意skip 1）。
- 新しい科学計算は0件。sampling、compile、構造S再計数、fit、実価格の予測・残差評価、signal/phase/radius/N、実N×C/G、R最適化はいずれも行っていない。
- 既存のB moduleは1行も変更していない。追加したのは、薄いadapter 1本、Cから無変更で取り込んだmodule 1本とtest、metadata CLI、test、文書だけ。

成果：
- [受入adapter](../../../src/trottertracks/rpe_r_price_intake.py)／[取込んだC module](../../../src/trottertracks/r_variable_compiled_price.py)
- [metadata CLI](../../../scripts/plan_rpe_r_price_intake.py)／[新test](../../../tests/check_rpe_r_price_intake.py)
- [HANDOFF](HANDOFF.md)／[READINESS](READINESS.json)／[COVERAGE](COVERAGE.json)／[TESTS](TESTS.txt)／[CHECKS](CHECKS.txt)

## 1. 再利用したもの・取り込んだもの

| 対象 | 固定SOURCE | 扱い |
|---|---|---|
| B共通pipeline・R選択・provider統合・A0受入 | `d5d9823e` | 無変更で再利用：`pf_pr_rpe_pipeline`、`rpe_r_selection`、`rpe_provider_integration`（`FixedSavedPrices`、`prepare_menu`、`run_missing`）、`rpe_normalization_intake`、exact会計 |
| H4/H6固定R価格provider | `d5d9823e` | 無変更。R4/R16の保存lookupだけ |
| C `r_variable_compiled_price.py`・そのtest | `c596e88d2a88fb219c2e373fb0d72c6e39cb723a` | byte一致で取込み。`describe_request`をBのadapterから使う |
| C `r_price_phase1.py`の`validate_plan`・`digest` | SOURCE `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7` | Git objectから読み込んで呼ぶだけ。fit/評価/価格関数は呼ばない（testでも呼出し禁止を確認）。runner/science backendは取り込んでいない |
| C1 PLAN | `c596e88d:docs/research/r_price_phase1_preparation_20261111/PLAN.json` | Cのvalidate_planで検査。plan digest `4bbedd54…` |

Cの`RVariableSavedPriceProvider`はB最新版へそのまま差し込まなかった。最新Bでは固定providerの返す価格recordへlaw binding・full domainを付ける処理（`FixedSavedPrices`）が入っているが、Cの版はその処理を経由しないためである。代わりにBの`RScopedFixedPrices`が次を組み合わせる。
- 既存の`FixedSavedPrices`
- Cの`describe_request`によるR scope metadata
- 同じtransplant禁止の検査

## 2. 何ができるようになったか

1. **C1結果の受入（`C1PriceIntake`）**：
   - `read_published(commit, run_dir)`は、指定された公開commitのrun directoryから次を読む：PLAN、FROZEN_MODELS、CALIBRATION_EVALUATION、VALIDATION、RARE_DIAGNOSTICS、STOP。
   - 受入時に次を検査する：stage順序、Git凍結、役割、候補、law、価格値とevaluationの一致。
   - 価格recordはcopyするだけで、値を計算しない。
2. **8セルの照合**：C1の8セルすべてで、Cが保存したrequest identityとBの`build_menu/request_context`によるidentityが一致した。比較対象はH/state/snapshot、configuration、T・qPF・δ・prefix・R・K・formula、allocation・event・scalar、compiler・control・wrapper・P0。law bindingは定義以外（candidate_id・R_outer・K・q_PF）が一致する。
3. **固定R価格の新Rへの移植防止**：
   - `RScopedFixedPrices`は、H4 R4/H6 R16以外のPR行に値が入っていれば`FIXED_R_PRICE_CANNOT_BE_TRANSPLANTED`で停止する。
   - 未校正のRは、既存providerのnull（`OUTSIDE_PREFIX_R_K`）のまま。
   - C1の新R点の価格はC1 recordだけで、該当する(m, candidate)のexact identityにしか付かない（`records_for`）。
4. **static coverage**：既存coverageに4つのlaneを追加した：normalization、signal/phase/radius/N、C、G。R価格状態も表示する。

## 3. C1価格recordとB interfaceの対応

| C1の区分 | Bでの`c1.price_kind` | 使い方 |
|---|---|---|
| 保存済み校正（H4 q8/q16 R4、H6 q1/q2 R16のdirect） | `SAVED_CALIBRATION_DIRECT` | 固定catalogと同じ保存cohortのC側copy。`same_saved_cohort_as_fixed_catalog`で明示し、別scenario `C1:DIRECT:<cohort>`とする |
| 新Rで取得した校正（H4 q8 R8、H6 q1 R32のdirect） | `NEW_CALIBRATION_DIRECT` | 有限cohort |
| Git凍結後の独立検証（H4 q16 R8、H6 q2 R32のdirect） | `INDEPENDENT_VALIDATION_DIRECT` | `frozen_model_commit`付き。calibration data groupはnull |
| 校正modelによる予測 | `MODEL_PREDICTION_AT_CALIBRATION_POINT`／`…_AT_VALIDATION_POINT` | `qualified_for_candidate=false`。適用domainは`C1_EVALUATED_POINT_ONLY` |
| C1外のR/q | recordなし。`R_price_status=NEW_R_PRICE_NOT_ACQUIRED`等 | null/欠測のまま |

cohort名には`C1:`を付ける。固定catalogの同名cohort（例：`DIRECT:EXPLORATION`）と同じ候補の別scenarioとして並び、重複エラーにも同一視にもならない。

検証と校正は混ぜない。受入時の主な検査は次の通り。
- CALIBRATIONの6セル／VALIDATIONの2セルは、それぞれ自分の役割のcell batchが揃っていること。
- FROZEN_MODELSのcalibration_cellsが校正3セル/systemだけであること。
- validationのfrozen commitの内容がpublication時点と同一であること。
- validation directにcalibration data groupが付いていないこと。

## 4. law・cohort・modelの資格

- **law**：
  - B（N側）のrequest lawは`b.LAW`（Aのideal finite RTE mean、`5ca2dce:pr_r_dependent_n.py`）。
  - C1のsampler宣言は`PCG64_numpy_float64_CDF_saved_cohort`（`c31ef424:src/trotterlib/rte.py`）。
  - 両commitの`src/trotterlib/rte.py`は同一blob（sha256 `620a7007…`）であることを確認した。これを根拠に、同じcandidate/R/K/qPFのrequestへ対応付ける。
  - B recordの`law_binding`はBのrequestを表す。Cの元宣言は`c1_original_law_binding`、実samplerは`law_context.price_law`、対応関係は`law_correspondence`に残す。
  - `ideal_sampler_distance_qualified=false`。文字列を書き換えて一致させたのではなく、blobの一致は距離の保証でもない。
- **cohort**：
  - PR価格はすべて`FINITE_COHORT_ONLY`。既定の`qualified_expectation`では価格として使えない（`UNQUALIFIED_C`）。
  - 明示的な`finite_cohort_scenario`ではdirect cohortをscenarioとして扱えるが、全law期待価格にはしない。
  - `n_paths`は古典的な価格path数で、量子shot数Nではない。X/Yは同じpathのpairで、独立標本数を2倍にしない。
- **forced rare**：RARE_DIAGNOSTICSは`rare_diagnostics`に別unitとして保持する（`price_record=false`、`pooled_with_cohort=false`）。forced cohortの価格recordは拒否する。
- **model**：
  - CのPLANは`acceptance_threshold=null`。qualified=trueのmodel recordは拒否する。
  - 予測値を計算できること（Cが出力済み）と、総費用に使う資格は別。B側でmodelを全q/Rへ適用可能にしない。
- **不確かさ**：Cのevaluationから次をcopyする：path間共分散、paired X/Y、nominal mean共分散、validation residualとsummary、shared calibration group。
  - R/q systematic、cross-R・cross-round共分散、joint共分散、total CI、全law期待価格はnullのまま。値を入れたrecordは拒否する。
  - 同じ係数を共有する予測は`dependence_groups`に列挙し、独立な推定量として扱わない。

## 5. 人工/mockテスト

[TESTS.txt](TESTS.txt)：**130 test method（129 PASS＋既存の環境変数依存skip 1）**。内訳：新15、C取込み9（無変更）、既存回帰106。回帰の内訳は、B pipeline 12、R選択12、provider統合8、A0受入8、alignment 7、H4結合7、H6 coverage 7、H4 provider 8、H6 provider 10、会計15、A N 12。

新15件は、指示の確認項目1–14と、pinned module・Git読取の確認1件。
- 人工C1 publicationは、Cのpinned `price_records`で作った。evaluationの数値は人工値で、実C1価格は使っていない。
- 主なguard（role、transplant、law付替え）は、一時的に無効化するとtestが失敗することを確認し、元に戻した。
- 作者自身の技術確認であり、独立した科学レビューではない。

## 6. 静的coverage（metadata例）

[COVERAGE.json](COVERAGE.json)は次のmenuを使ったmetadata例で、R候補の採択ではない。
- H4 {m3: R4/8, m4: R4/8, m12: R4/8192}
- H6 {m4: R16/32, m5: R16/32, m12: R16/8192}

| 例 | normalization | signal/phase/radius/N | C | G |
|---|---|---|---|---|
| H4 m12 qPF4096 R8192 | A0記録済み（`NORMALIZATION_SCORED_NOT_N_QUALIFIED`） | 未評価 | 未取得（`NEW_R_PRICE_NOT_ACQUIRED`） | 未評価 |
| H6 m12 qPF256 R8192 | A0記録済み | 未評価 | 未取得 | 未評価 |
| H4/H6 C1新校正・独立検証の4セル | H6 m4 R32だけA0記録済み | 未評価（N欠測） | `C1_NOT_INGESTED` | 未評価 |

normalizationはstatusだけを参照し、N・Cへの換算や乗算はしない。保存N（H4補正済み巨大整数を含む）は既存loaderのcopyのまま。

## 7. 未取得・未資格と残る科学的不足

- **C1の実結果は今回取り込んでいない。**
  - 作業中に、Cが次のcommitを公開していたことを確認した。
    - `7eeabf0ae4…`（02:00）：校正と凍結
    - `2e6a0cdf5…`（02:46）：校正・独立検証・rare、`STOP=PHASE1_COMPLETE_MANDATORY_STOP`
  - 指示の前提（C1は未完了、実価格は存在すると仮定しない）に従い、取り込んでいない。
  - 確認したのはフィールド名の構造だけで、adapterの期待と一致した。run01のPLANは固定PLANと同一blob。数値は読んでいない。
- 長q（H4 m12 q4096、H6 m12 q256）と大R（8192等）の価格は、C1の小q 8セルからは資格されない。C2等の別の有限検証が必要。
- 新RのN（R-aware allowance、phase/radius）は、C1の新4セルではすべて欠測。同じ点で実N×Cを見るには、Aの別の科学実行が必要。
- 次は未資格のまま：modelの採用閾値・必要精度、全law期待価格、rare寄与、actual samplerとideal lawの距離、cross-R/round共分散、total CI、全RPE成功。

## 8. 実施範囲・保全

- 既存B branch `codex/r-variable-rpe-cost-pipeline-20261111`（worktree清浄、remote一致を確認）を継続した。
- A/Cのbranch、worktree、process、cache、private rawは読み書きしていない。Cの成果は公開Git objectだけを参照した。
- SOURCE、input、raw、result、grant、ledger、marker、KEEP、過去のSTOPは変更していない。共有環境も変更していない。
- commit/pushとremote照合の後、**mandatory STOP**。C1実結果の取込み、新しい科学計算、実G/R比較へは自動で進まない。
