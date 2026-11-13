# B → A/C/D：予備分析と全Gへの接続

2026-11-13 JST。[REPORT](REPORT.md)／[DECISION_PACKET_V0](DECISION_PACKET_V0.md)／[有限再解析P0〜P4](NEXT_OPTIMIZATION_PLAN.md#1-承認を求める有限保存再解析b-prelim-v0)／[固定SOURCE/ID](SOURCE_SCOPE.json)。今回は計画・metadata照合のみ。新しい数値再解析・科学実行を他担当へ指示していない。

## 1. 共通task・価格quantity

元H4 DF10/H-state/snapshot/sector、energy目標・risk claim・prior/decoder/alias/loss、state仮定、P0を保つ。同一のq/R/幅/Nを全方式へ強制せず、目標精度と会計quantityを共通にする。全RPEのNはwhole-vectorとして評価し、roundごとの独立shot最適化を仮定しない。

構造quantity=`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`とcompiled quantity=`FULL_WRAPPER_MEAN_COMPILED_PRICE`を区別する。CのE[S]はideal finite-K law、D pilotのC−Sは具体的samplerのfinite development cohort。Sはcompiled Cの厳密な上下界でなく、law距離を無視した加算でE[C]へ資格しない。costは測定を含むcontrolled full Hadamard wrapper RZ/shot、状態準備P0。

## 2. A/Cのcandidateとsignal

既存`generate`/source-bound schedule adapterのcanonical configuration/round identity/lawと、元physical candidate ID/SOURCEをそのまま渡す。PR現構成だけをanchorにし、変数変更を旧identityへ隠さない。共通κ・λ/source・integer R/signed allocation・K・scalar一回・PF実幅を保持する。

B入力は既存`rpe_decoder_statistics.prepare`のmean/target形式。raw、corrected、logB/attenuation、bias/reference、数値allowance/qualification/sourceを保持し、rawだけから観測確率を作る。新candidateのsignal生成・normalizer・referenceはA/Cの担当側で管理し、Bが再実装しない。A821cの`pr_rpe_signal_plan`はmetadataの再利用候補で、今回取り込んでrunしていない。

4方式screen699178ではm6 Full PF4セルだけを局所証拠として保持する。m12/M5の数値gate失敗、Full-H/B0I未開始、空例外は元status/nullのまま。失敗時norm/unitarity残差の新評価は別科学scope。Aの技術調査・既存STOPを本予備分析で再開しない。

## 3. D価格とC構造

要求/返却は既存V2 identity・law_binding・axis・SOURCE/physical bridge、compiler/control/wrapper/measurement/profile/P0、明示lowering、DIRECT/MODEL/cohort/外挿・domain資格・価格区間/不確かさを保持する。compiler hashだけでlowering同一とは判断しない。旧Full PF receiptと新controlled ZZの違いをscenarioとして分ける。

Cの全26recordはaxis=cosine/sineで、BのN_X/Yへ明示対応させる。元axis名を変更せず、m/axisのkeyで結合する。既存m0/m6 copyと新11roundのkernel_sourceも保持する。今回C kernelを再実行せず、numerical enclosureをcompiled/actual lawの保証へ変換しない。

D pilotのfresh summaryはm0/m6・各4 normal path、replay/P1と別。承認後に許すC/S・D/S比も有限cohortの局所診断に限定する。長時間roundの分散/残差・rare寄与・CI・全lawmeanをBで推定/fitしない。Full PF価格はe7dcaの既存receipt分類を再利用し、元gate数の再計数をしない。

## 4. 既存コードへの接続と再利用

| 既存API | 今回/承認後の使い方 |
|---|---|
| `pf_pr_rpe_pipeline.SavedArtifacts.read`, `same` | 今回実際に再利用：pinned locatorの読取、全identity/law/metadata照合。科学providerを呼ばない |
| `h4_df10_rpe_connection.exact_scalar/exact_value` | 承認後のproxy積和・比のexact会計。巨大整数Nをfloat化しない。表示丸めと保存値の数値誤差を分離 |
| `rpe_shot_allocation.prepare/diagnose/compare`、元controller | 今回呼ばない。将来、供給全rawと明示loss/risk/shot menu、資格N＋同candidate compiled価格が揃う範囲で使用。nominal/Sを通すためguardを緩めない |
| `rpe_decoder_statistics`とR3 | 元逐次branch/zero/tie/count規則のまま。全RMSE評価は別の有限許可。局所phase/Tを代用しない |
| D `full_pf_saved_inventory`/`full_pf_price_receipts` | 既存e7dcaの選択結果とcarrierを再利用。元136候補の再走査・private preparation再構築は不要 |
| D `p2_direct_pilot.fresh_summary`、`compiled_residual.cohort_summary` | 元fresh summaryの由来を確認。今回再計算せず、次案も既存summaryからの有限比だけ。再fit/新CI/新cohortなし |
| D24b0/B既存価格精度計画 | 必要N又は許容集合、同時区間、law/rare/共有校正の資格が揃った後の精密化request。N未資格なら正式Gへの影響順位は出さない |

正式会計と予備proxyは別出力にする。`cost/select/compare`へ偽のQUALIFIED N/価格を渡してproxy値を得ない。P0〜P4は小さな既存exact算術で足り、新しい共通framework/価格model/N solverは不要。

## 5. 後続担当と必要情報

- **ユーザー**：まずP0〜P4の有限保存再解析を許可するか判断。これは未解決loss・confidence・N探索範囲の採択や新科学runの許可とは別。
- **B**：許可後だけ数値欄を埋め、条件付き構造寄与・短round閾値・pilot比から次の研究判断を提案。正式G/winnerやPR全体最適性は返さない。
- **A/Cの候補・signal担当**：少数PR対照とbaselineの全mean不足、PF数値安定性、target/数値/finite-K/law適用範囲を管理。どのL_D/K/κ対照を選ぶかは本結果とユーザー判断後。
- **A/Bのrisk担当**：共通全事象loss/risk claim・target誤差・raw/law移送・同時confidenceを準備し、whole-N資格を評価する有限科学案を出す。S3点RMSEを十分Nへ昇格しない。
- **D**、必要なら**C**：使用domainの平均compiled C・区間・モデル/外挿資格・law/rare/bias/covarianceを供給。全domainの課題と特定価格精密化の必要性を分ける。

本納品は他担当へのチャット送信や作業変更を含まない。既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを保全し、commit/push・remote照合後mandatory STOP。次の実行は許可された有限scopeだけ。
