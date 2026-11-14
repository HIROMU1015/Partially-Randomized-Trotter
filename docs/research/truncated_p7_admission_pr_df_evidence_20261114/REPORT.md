# p7保存価格の受入とPR-DF期待価格の証拠整理

2026-11-14。**p7 B0I M5/m4のX/Y保存receiptは受入PASS。** 両軸60,950 RZを再利用する。新compile・実価格平均/fit・予測・Gの計算は実施していない。

## C1：保存receipt

|条件|照合結果|
|---|---|
|物理候補|`13d1cd4869951cdf1e2c`、H4/1 Å/STO-3G/tol-only DF10、B0I p7、M5、m4/T1.6/q1/actual16h|
|H/state/snapshot|原carrier provenance、保存H4 context、現C configurationのbindingが一致。context原本はSOURCE_INDEXのSHAと一致|
|保持領域|one-body＋fragment0..6の8 block、元順。元native_context/compose/M5 scheduleを再利用。新prepareは作っていない|
|scalar|c＋保存p7 tail identity。元SOURCEのbinary64加算規則で **2.293146549384713** と一致。総Tに一度、piece内scalarは0|
|回路SOURCE|前回C `2fa88065` のnative/wrapper/contract一致の証拠を再利用。D `62ae1bd5` と段階分離 `c848bbef` のplans/checked_trial/price_bindingsも同一|
|制御・wrapper|symmetric-directional controlled evolution、full Hadamard X/Y＋測定、P0|
|compiler|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、hash `1239304c…`。原cache receipt/provenanceと現profileで一致|
|元価格/gate|原CSVのcosine/sine行、carrier n1、原trialの全6 metricが一致。元7個の作用/compiled-vector gateは元tolerance内・phase alignmentなし|
|分類|`DIRECT_COMPILED_DETERMINISTIC` / `DETERMINISTIC`、route `TRUNCATED_PF`、RTE lowering `NOT_APPLICABLE_NO_RTE`、scalar quantity `HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY`|

原SOURCEは `155529e20a8fac39e00786c40b57830b71203c4f:artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`。原cohort **EXPLORATION/index0**、trial locator・hash・preparation hash `fc83f26f…`を維持した。[ORIGINAL_TRIAL.json](ORIGINAL_TRIAL.json)はbyte copy、[ORIGINAL_TABLE_ROWS.json](ORIGINAL_TABLE_ROWS.json)は原CSVの2行の文字列。新規取得cohortへ移していない。

### X/Yを個別に識別した証拠

元trial/CSVにはwrapper fingerprintが直接記録されていないため、旧studyの保存cacheとQPYを参照した。同じ全6 metricを持つ別回路が存在し、**metric一致だけでは採用していない**。

|軸|保存full-wrapper pre-transpile fingerprint|suffix（ancilla8→classical0）|
|---|---|---|
|X|`b0026b0eb54be07801a02a628e27f19302a93ba4c1cc9ddf8dbf0c91de5b11b6`|H、measure|
|Y|`8b523455c6b57793b1edf4cabf57f84e24d5777dca11ad606bced1edfc3389e5`|Sdg、H、measure|

保存QPYをdeserializeし、既存canonical writerで各fingerprintが一致することを確認した。axis suffixを除くword（entry H・evolution・scalarを含む）のexact canonical digestも一致する。両軸は別fingerprintであり、同じ価格を二重に流用したのではない。

ancillaのscalar角は元の`-scalar*T`のbinary64値と一致し、総Tへの適用は一度。同じmetricの別T/B0回路はこの照合で区別した。原nativeが各axisのfingerprintでcacheを書き、そのmetricを元trialの同axisへ保存するSOURCE chain、元trialの個別gateと併せて受け入れた。保存回路を再生成・再compile・価格再計数していない。各QPY/receipt/compiled-artifactのhashと照合結果は[SAVED_WRAPPER_CHECKS.json](SAVED_WRAPPER_CHECKS.json)。

固定bundle **`43c15bfa34a22f313e55b824223b200332b7f23f`** に証拠を保存した後、実際のGit JSONを用いて無変更のC628 guardを通した。[GUARD_CHECK.json](GUARD_CHECK.json)は両軸とも `IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`、価格tierは **DETERMINISTIC_DIRECT**。受入公開は **`2b94539784995790c8ff1ba7dee578137bfc6034`**。正式な比較policyの採択やG評価とは区別する。

## 新版CELL_PLAN

[CELL_PLAN_V2.json](CELL_PLAN_V2.json)は旧 `2fa88065:docs/research/m5_m4_receipt_admission_20261114/CELL_PLAN.json` の**p7 M5/m4 X/Yのactionだけ**を `NEW_COMPILE` → `SAVED_RECEIPT_FOR_C_ACCEPTANCE` に変更。既存action名をD interfaceとの互換のため用いるが、両軸のC受入は完了している。旧版は変更していない。

|方式|保存較正|新規較正|新規holdout|
|---|---:|---:|---:|
|Full PF p10|2|10|6|
|Truncated p7 B0I|2|10|6|
|合計|4|20|12|

全36行、候補・formula・q・actual幅・round・軸・model_group・roleは不変。新規compileの対象は32件へ減る。m4/m5較正→12モデルをfit/freeze→m6 holdoutという順序は維持する。観測値分母2%も**未採択の提案のまま**。実fitはしていない。

D c848の段階分離で前回の一括取得問題は解消した。ただし現SOURCEは旧CELL_PLAN・Full PFだけの保存入力・Truncated較正12件を前提としている。新版をそのまま渡すと既存guardが停止することを人工testで確認した。Dがpin・p7保存参照・較正件数を更新する必要がある。[HANDOFF.md](HANDOFF.md)と固定locatorを揃えた[SAVED_RECEIPT_SOURCES.json](SAVED_RECEIPT_SOURCES.json)を利用する。

## C2：PR-DF coverage

対象は既存AのPR-LD1/LD2 schedule（保存H4 DF10/S2/Δ.1/K2/h.1/M12/P0）。C `3ae2e605` の固定coverageを再利用し、D P1参照を追加した。52軸の個別physical ID、SOURCE、R/law、資格は[PR_DF_COVERAGE.csv](PR_DF_COVERAGE.csv)／[JSON](PR_DF_COVERAGE.json)。価格・残差の値は再集計していない。

|m（各X/Y）|qPF|LD1 R|LD2 R|LD1の同条件compiled証拠|LD1 E[S]|LD2の同条件S/C|
|---|---:|---:|---:|---|---|---|
|0|1|2|1|P1 n2、P2 fresh n4（別cohort）|あり|未取得|
|1|2|4|2|未取得|あり|未取得|
|2|4|8|3|未取得|あり|未取得|
|3|8|16|5|未取得|あり|未取得|
|4|16|32|10|未取得|あり|未取得|
|5|32|64|20|未取得|あり|未取得|
|6|64|128|40|P1 n2、P2 fresh n4（別cohort）|あり|未取得|
|7|128|255|80|未取得|あり|未取得|
|8|256|510|160|未取得|あり|未取得|
|9|512|1020|320|未取得|あり|未取得|
|10|1024|2040|640|未取得|あり|未取得|
|11|2048|4080|1280|未取得|あり|未取得|
|12|4096|8160|2559|未取得|あり|未取得|

両軸で同じcoverage区分だが、価格値の一般的同一性を認定したものではない。P1/P2は新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`・full wrapperの同path/axis C/S。P2のreplayはP1の再現確認であり、新独立cohortとして足さない。P1とP2 freshもpoolしない。consumer identity/law/scalar/source bridgeは未受入のまま。

**26 E[S]は構造期待値**（C `9d95d3ea…`、ideal finite-K law）。K0/K1・区間・preparation324 component/9 basisは原本のまま。E[C]ではない。**資格付きE[C]は52軸すべて未取得**で、現domainの残差modelの較正/独立holdoutも未完了。

旧C1は別catalog [PR_DF_C1_CATALOG.json](PR_DF_C1_CATALOG.json)。H4のq8/R4 n9、q16/R4 n33、q8/R8 n8を校正、q16/R8 n8を独立検証した保存価格がある。旧generic controlled RZZ lowering・固定prefix1のdomainであり、現在のq8/R16・q16/R32やprefix2へ移植しない。元role・cohort・SOURCE・qualified=falseを保全。旧rare診断も別unitで、新ZZ/現lawのrare寄与保証ではない。

`rpe_mean_price_model.STRUCTURAL_PLUS_RESIDUAL`、既存MeanKernel、guard、paper_nominal_costを再実装していない。E[S]＋単一残差という経路は使えるが、同context・元S・lawと不確かさの適合が必要で、自動資格は付かない。実値で呼んでいない。第1水準への接続と追加検証は[PR_DF_VALIDATION_PLAN.md](PR_DF_VALIDATION_PLAN.md)。

## 検証・重複照合・STOP

今回、新8＋変更したFull PF受入helperの回帰7＝**15 test method PASS**。P1参照追加後は影響するmetadata testを再確認した。無変更のC guard/modelの既存48 PASS記録（C2fa）と、E[S]/C1/P1/P2の固定検証記録は再利用し、科学再実行・同じ回帰48件の重複実行をしない。作者自身の技術確認で、独立科学レビューではない。

指定`research_project_instructions_unified_2026-11-14.md`の実体は、既知の共通位置/各worktree直下では取得できず、保存パスを照会中。本文の未確認条項を適用済みとは記載しない。依頼で明示された§10.5の重複照合は、既存実装・固定証拠の再利用と差分検査として実施した。既存AGENTSの保全・科学実行境界を維持した。

同目的のC branch `codex/m5-m4-receipt-admission-20261114`を継続。A/B/Dのworktree・実行領域には介入せず、PRについては公開Git metadataだけを参照した。原C1/C2/T2/旧CELL_PLAN/guard/model係数/資格/SOURCE/raw/grant/ledger/marker/KEEP/STOPを変更していない。

現時点で新compile、sampling、新S/E[S]、保存C/Sの新平均・残差fit/予測、12 PFモデルの実fit、G_paper/N/G、policy採択は**0件**。公開・remote SHA/tree照合後 **mandatory STOP**。
