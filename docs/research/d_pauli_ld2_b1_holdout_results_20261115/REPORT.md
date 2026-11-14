# PR-Pauli LD2 m5/m6：独立holdoutの記述的比較

2026-11-15 JST、Codex D2。ユーザーがC2の記述的規則を採択し、固定fresh3 path・X/Y最大6 compileと4件の誤差算出を許可した。**campaign = `DESCRIPTIVE_COMPLETE`、model_verdict = `NOT_ASSESSED_NO_ADOPTED_TOLERANCE`**。予定最大6、受入済み保存holdoutの再利用0、新規実compile6、compile cache hit0、有効個別receipt6/6。モデルのPASS/FAIL、数値許容差は設定していない。

[HANDOFF](HANDOFF_C2.md)／[exact4比較](COMPARISON.json)／[個別receipt受入](RECEIPT_ADMISSION.json)／[個別価格CSV](OBSERVATIONS.csv)／[取得・資源](ACQUISITION_STATUS.json)／[観測前固定](PREOPEN.json)／[最終静的照合](FINAL_CHECK.json)。本報告は指定有限cohortの差の記録である。

## 1. 固定SOURCE・入力と開封順序

- 規則：`f4480ee3f831a7b106d287c401ee78fbd62f8615` の `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/HOLDOUT_RULE_PROPOSAL.md`。固定文書の「提案・未採択」は当時の履歴で、今回はユーザーの新指示により記述的規則を採択した。
- 凍結予測正本：`b74e4e536392c2bbe94cd371436615fc56c0564f` の `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`。配列順でなく(m,axis)で4件を対応付け、`predicted_RZ`を直接読んだ。係数からの予測再生成・再fit0、C2 branch merge0。
- cell/cohort/seed/canonical physical ID：`043c7567451e8f502a1cf0f425d32efe0a50f5dc` の [CAL_AFTER_W_METADATA](../d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json)。m5=M5_ROUND、q32/R33/T3.2、fresh p0/p1。m6=M6_ROUND、q64/R66/T6.4、fresh p0。
- D2固定実装：`13ab1acfaebe5f33958e1db67862f5c45725ab91`。実行時HEAD＝`5dac74959ed9ea49465d393fc792ef5b4727edba`。caller/core/connection/receipt/decoderの実行bytesは固定13abと一致、科学SOURCE変更0。新しい[ADMIT](ADMIT.py)は既存receipt exporter/guardの呼出し、[COMPARE](COMPARE.py)はexact誤差算術だけである。
- 今回の元raw公開commit：`0e1ce866e8f8d4e042801260fb8d543d0b1bee53`。各cellのRESULT/C_INPUTS/PRICES/COMPILE_RECEIPTS/IMPLEMENTATION_EVIDENCE/SIDECARS_UNBOUNDを固定し、その後のsidecar/guard/比較を別保存した。実行SOURCEと公開SOURCEを区別する。

§10.5では固定計画と前回CALの同等性・別ID監査を再利用し、関連remote tip、計画後に進んだD price/Full-H/C2 branch、local同名科学job、新公開/専用private namespaceのmetadata差分を確認した。元physical IDのhitは計画・凍結予測・alignment資料のみで、同等holdout raw/receipt候補なし。旧DF-tail/Full-H/PF/Pauli-JWはlaw/head/K/lowering/quantityの条件差を元監査から再確認した。同一候補のlocal実行jobなし、出力先未使用、受入可能な再利用0、出所不明で保留した候補0。確認は既知の公開refとlocal保存先に限定し、外部非公開jobまで確認済みとはしない。照合時にholdoutの観測価格は開かなかった。

[PREOPEN](PREOPEN.json)を取得前に保存し、対象・4予測とselector・規則・SOURCE/file hash・出力先を固定した。凍結model gateは `FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION`。較正seedとの重複なし、holdout内128 step seedも重複なしをmetadataで確認した。JOBに元7-cell計画が残るのは既存design検査のためで、今回投入したのはHOLDOUTの2 cellだけ。B0/Wや較正pathの再実行・標本への繰入れ0。

## 2. 物理条件と価格の範囲

保存H4 linear1.00Å/STO-3G/tol-only DF10、LD2、one-body＋fragment0,1のDF head S2、保存Pauli tail184項、K2、step time0.1、元normal sampler（PCG64/NumPy float64 CDF）。保存H/state/snapshot・辞書の固定locatorとhashはPREOPENに収録した。辞書SHA-256 `dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7`、P0 SOURCE `4220f3006d0d8d413e195ed948e250578513b557` を維持した。

canonical physical IDはm5=`RMSE:01489f6e391e9917e96eaa56a0006a91b6099e60956cb5fcea3a516b24df5802`、m6=`RMSE:8826b5c0b36fdb91115109a85c6f2eca40cf1989bb3b3e67d8cbcd1598043529`。cohortは元 `D_PAULI_LD2_B1-CAL_M5_ROUND_PROPOSED_V1` / `...M6_ROUND_PROPOSED_V1`、全seedはPREOPEN/JOBS/各path・receiptで一致を確認した。

scalar=c_H＋保存s_R=2.515477156730702を各pathの総Tへ一回、Pauli D2 lowering `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional、測定付きfull controlled Hadamard wrapperを維持した。状態準備費を含まないcompiled RZ数が価格quantity。X/Yは同一pathの別軸で、独立6標本ではない。

## 3. 個別6 receiptの観測価格と受入

| round | path | 軸 | compiled RZ | RZ depth | CX | CX depth | total depth | size | 個別receipt |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0 | X | 68228 | 20052 | 36776 | 23057 | 51785 | 145486 | `p0X` |
| 5 | 0 | Y | 68228 | 20052 | 36776 | 23057 | 51785 | 145486 | `p0Y` |
| 5 | 1 | X | 68300 | 20102 | 36820 | 23110 | 51933 | 145659 | `p1X` |
| 5 | 1 | Y | 68300 | 20102 | 36820 | 23110 | 51933 | 145659 | `p1Y` |
| 6 | 0 | X | 149435 | 47363 | 92334 | 63295 | 128349 | 328666 | `p0X` |
| 6 | 0 | Y | 149435 | 47363 | 92334 | 63295 | 128349 | 328666 | `p0Y` |

| round/軸/path | 元receipt locator（raw公開commitは下記共通） | 実行SOURCE / fingerprint |
|---|---|---|
| m5-X-p0 | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M5_ROUND/COMPILE_RECEIPTS.json` → `['p0X']` | `5dac74959ed9ea49465d393fc792ef5b4727edba` / `897261f8c317d6af8705abb56d3d2221e55082bb48cbdfe2e300f5ef654ac5e1` |
| m5-X-p1 | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M5_ROUND/COMPILE_RECEIPTS.json` → `['p1X']` | `5dac74959ed9ea49465d393fc792ef5b4727edba` / `a215ad433ae961e12113584011480e3b76347c5e37260970fd23f5788546c3bb` |
| m5-Y-p0 | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M5_ROUND/COMPILE_RECEIPTS.json` → `['p0Y']` | `5dac74959ed9ea49465d393fc792ef5b4727edba` / `6e27a62055749a742854b12ebbe3cabbc3cc663719debc9508df5121a9a464b5` |
| m5-Y-p1 | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M5_ROUND/COMPILE_RECEIPTS.json` → `['p1Y']` | `5dac74959ed9ea49465d393fc792ef5b4727edba` / `c24e864e764814c7d86a4170a4ca6d3647d0d795b3afeca72ecba92d810db957` |
| m6-X-p0 | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M6_ROUND/COMPILE_RECEIPTS.json` → `['p0X']` | `5dac74959ed9ea49465d393fc792ef5b4727edba` / `1f4d49d9e94f91666907470ca3f1943bc4224484c27f08f9178402285595a8a7` |
| m6-Y-p0 | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M6_ROUND/COMPILE_RECEIPTS.json` → `['p0Y']` | `5dac74959ed9ea49465d393fc792ef5b4727edba` / `beff662a88996377a332c63062148fd49ae3405caf35a0d844f5eb3cb573bfe6` |

全locatorのcommitは `0e1ce866e8f8d4e042801260fb8d543d0b1bee53`。個別receiptは既存 `pauli_b1_receipts.compile_receipts` でprivate cacheから元recordとQPY所在/hashをコピーした。固定callerのcanonical-round経路は個別COMPILE_RECEIPTSを自動出力しないため、この既存helperだけを取得後に呼び出した。新しいcompile・回路再simulationは行っていない。

公開rawに対し、固定 `check_path` でrequest・辞書・cohort/index/seed・作用/compiled-vector gate・測定wrapperを、個別receiptでaxis・実行SOURCE・compiler・fingerprint・全metricsと整数RZ値を照合した。canonical `check_round_records` による実装guardは各cellの `C_GUARD_CHECK.json`、公開後bindingは `SIDECARS_PUBLISHED.json` に保存した。guardの既存policy名 `B1_TECHNICAL_COMPATIBILITY_NOT_ADOPTED` は技術照合の識別子で、モデル精度判定ではない。**受入を終えてから**、receiptの整数RZを4件の平均・誤差算術へ渡した。

## 4. 凍結予測とのexact4比較

| 単位 | P（凍結exact） | O（receipt観測exact） | d=P−O | abs(d) | d/O | abs(d)/O |
|---|---|---|---|---|---|---|
| m5-X | 6169957875465/90343469 | 68264 | 2751307649/90343469 | 2751307649/90343469 | 2751307649/6167206567816 | 2751307649/6167206567816 |
| m5-Y | 6169957875465/90343469 | 68264 | 2751307649/90343469 | 2751307649/90343469 | 2751307649/6167206567816 | 2751307649/6167206567816 |
| m6-X | 13521761923033/90343469 | 149435 | 21285633018/90343469 | 21285633018/90343469 | 1637356386/1038498176155 | 1637356386/1038498176155 |
| m6-Y | 13521761923033/90343469 | 149435 | 21285633018/90343469 | 21285633018/90343469 | 1637356386/1038498176155 | 1637356386/1038498176155 |

m5のOは同軸p0,p1の算術平均、m6のOは同軸p0の値。平均・差・比はすべて `{numerator,denominator}` の既約有理数で [COMPARISON](COMPARISON.json) に保存した。d=P−Oで、正は過大予測、負は過小予測。相対誤差の分母は観測価格Oで固定した。参考の符号付き相対誤差表示：m5-X 0.044612%、m5-Y 0.044612%、m6-X 0.157666%、m6-Y 0.157666%（%は表示だけの丸め）。

`COMPARE.py`は観測前に作成しhashをPREOPENへ固定、exact算術・符号・観測分母・(m,axis)対応・O=0・0/0・m5片path欠測・gate失敗を人工値で確認した（[log](ARITHMETIC_TOY_CHECK.txt)）。m5の欠測を1 path平均へ変更せず、無効単位はOと全誤差をnull、O=0なら相対誤差をnull/`ZERO_OBSERVED_PRICE`とする。同じ4観測から新しい閾値を設定していない。許容差必須の `price_model_validation.validate` は呼ばず、仮のtoleranceも渡していない。

終了状態は全4単位有効かつO>0ならDESCRIPTIVE_COMPLETE、全4有効で0を含むならDESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED、欠測/無効があればINCOMPLETE_OR_INVALID。今回の状態は `DESCRIPTIVE_COMPLETE`。いずれもモデルverdictはNOT_ASSESSED_NO_ADOPTED_TOLERANCEであり、取得gateのPASSをモデルのPASSへ読み替えない。

## 5. 作用gateと実event

| 固定gate | 件数 | 保存誤差最大値 | 固定tol | 取得gate状態 |
|---|---|---|---|---|
| low_control | 3 | 2.08439212e-12 | 1e-09 | 全PASS |
| high_control_leakage | 3 | 0 | 1e-09 | 全PASS |
| independent_full_space_vector | 3 | 2.08093036e-12 | 1e-09 | 全PASS |
| Hadamard_cosine | 3 | 9.15489906e-13 | 1e-09 | 全PASS |
| Hadamard_sine | 3 | 4.05764311e-12 | 1e-09 | 全PASS |
| compiled_vector_cosine | 3 | 2.08289156e-11 | 1e-07 | 全PASS |
| compiled_vector_sine | 3 | 2.07588243e-11 | 1e-07 | 全PASS |

bit順は取得済み各pathの固定gateで確認した。保存stateに対するfull-space vector/control/scalar/Hadamard/compiled-vectorの確認で、全入力演算子の新資格ではない。既存26人工/toy記録は再利用し、旧suite再起動・保存H4 P0演算子の独立数値再検査0。ここに示したgateは今回の実pathに必要な固定caller内の検査だけである。

| cell | path | event数 | order別実観測 | 先頭3 component | 先頭label | event phase/count | D2 phase集合 |
|---|---|---|---|---|---|---|---|
| M5_ROUND | 0 | 1056 | {0: 1056} | 0, 13, 50 | `IZIIIZII` | [{'real': 1.0, 'imag': 0.0, 'count': 1056}] | [0.0] |
| M5_ROUND | 1 | 1056 | {0: 1056} | 2, 23, 66 | `IIIZIIIZ` | [{'real': 1.0, 'imag': 0.0, 'count': 1056}] | [0.0] |
| M6_ROUND | 0 | 4224 | {0: 4224} | 92, 16, 64 | `IIIIIIIZ` | [{'real': 1.0, 'imag': 0.0, 'count': 4224}] | [0.0] |

実LD2 order2 productは未観測UNKNOWNを維持した。 非自明event/D2位相も未観測UNKNOWNを維持した。 全順序列、Pauli label/係数/phase/order/product、D2 token/phase owner/mappingは元RESULTのevent_metadataに保存した。[EVENT_COUNTS](EVENT_COUNTS.json)は実観測の転記と固定locatorで、理想law確率の再解析ではない。未観測を理由とする追加seed/path/強制sampling0、toy事例の繰入れ0。

## 6. 資源と実行状態

| cell | wall (m:ss) | user / system (秒) | peak RSS (KiB) | exit / swaps |
|---|---|---|---|---|
| M5_ROUND | 3:59.38 | 357.72 / 593.09 | 791416 | 0 / 0 |
| M6_ROUND | 4:35.03 | 487.39 / 937.82 | 946088 | 0 / 0 |

独立2 cell jobを並行で各一回起動した。各GNU time -vのwallはimport/context準備/作用gate/個別compile/compiled-vectorを含み、軸別compile時間ではない。並行なのでwallの和を全体経過時間とせず、RSSの和を同時peakともしない。実際の起動commandは各RESOURCE先頭に保存した。execution stdout/stderrとRESULT/private child/cacheを保持し、無断retry0、seed差替え0、guard緩和0。CPU affinity64、任意resource cap0、共有設定・他利用者プロセスの変更0。実行中の一時的CPU競合を理由に正常workerを停止していない。

## 7. 引渡し・ロードマップ影響・STOP

今回解消したのは、C2の保存4予測に対する指定fresh有限cohortの観測と記述差の未取得である。**モデル資格・全law E[C]・統計CI・m7以後の長round価格・N/G・winnerは未認定のまま**。m6の1 pathから分散/SE/CIを作らず、X/Yを独立pathへ水増ししない。凍結modelsのCALIBRATED_UNVALIDATED/qualified_for_candidate=falseは変更していない。

統合ロードマップのPauli項目には、前回CAL18/10 record、C2の2モデル/4予測凍結、今回の独立holdout6/4記述差まで進んだことを反映する候補となる。ロードマップのWのみ完了という旧記述はこの3段階の一次資料で更新可能だが、本作業でロードマップ本体は編集しない。将来の科学的資格や下流費用への採用には、用途・証拠を踏まえた別の判断と実行指示が必要であり、今回から自動継続しない。

旧CAL/B0/W raw・SOURCE・STOP、SCOPE_DEVIATIONとlog、元plan/PROPOSALS/JOB_TEMPLATES、C2凍結blobを保全した。既存D2 branchへ結果・REPORT・HANDOFFをcommit/pushし、remote照合して**mandatory STOP**。追加path、事後閾値、再fit、feature変更、LD1、E[C]/CI/N/G・winnerには進まない。
