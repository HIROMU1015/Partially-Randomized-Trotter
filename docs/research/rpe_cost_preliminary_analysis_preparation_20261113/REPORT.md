# B：RPE総コスト最適化に向けた予備分析の準備

2026-11-13 JST。**固定SOURCE・比較範囲・算術・有限再解析案を整理した。保存科学値を使う新しい寄与率、比率、感度、再集計、ランキングは未実行。** 数値結果欄はnullのまま、正式G・十分N・winnerも未資格である。

[DECISION_PACKET_V0](DECISION_PACKET_V0.md)／[優先事項・有限実行案](NEXT_OPTIMIZATION_PLAN.md)／[A/C/D HANDOFF](HANDOFF.md)／[固定locator・candidate](SOURCE_SCOPE.json)／[7 metadata確認](METADATA_CHECKS.txt)。既存B branchを継続し、production code・科学runner・新modelは追加していない。

## 今回判断できること

| 結論 | 証拠水準・限界 | 研究判断への意味 |
|---|---|---|
| PRの全13round nominalとCの全26軸E[S]は同canonicalに結合可能 | **保存metadata照合**。元identity/lawを一致確認。Cのaxisはcosine/sineでありX/Yとの対応を明示。十分NやE[C]を得た意味ではない | 新signal/構造kernelを計算せず、承認後に条件付き構造費用の寄与を計算できる |
| 長roundが正式Gをどの程度支配するかは未確定 | **未資格**。nominalの積和も今回は未実行。真のN配分・compiled補正がround比率を変える | E[S]の増大やT²だけでm12重視・他round省略を決定しない |
| 保存Full PF価格は強いbaselineの短roundを支える | **保存SOURCE照合済み証拠の再利用**。6公式/幅laneのm3/m4に8候補対の直接X/Y価格があり、他4対は価格欠測。これはRMSE適格性ではない | 保存可能セルだけで損益分岐式の係数を作り、長時間の費用差はunknownにする |
| PRとFull PFの同round直接compiled価格比較は今回scopeで成立しない | **metadata照合**。Dのfresh PRはm0/m6、上記Full PF価格はm3/m4。PR構造価格は全roundあるがcompiled Cではない | m3/m4ではFull PF C対PR Sの条件付き閾値だけ。m6ではFull PF Cが不足。異round価格を割らない |
| 4方式screenは完了していない | **A699178の保存報告**：Full PF m6の4セル保存、6 gate失敗、9未開始、1空例外未解決。m12の有資格signalなし | 数値gate失敗をPF方式の科学的FAILやPR優位に変換しない。Aの原因調査を重複しない |
| S3は固定nominal1配分の経験的条件付き評価 | **B2084791の保存観測**。主population CI、十分N/global boundはnull。全trialを今回開いていない | 必要N比をこの1実験から推定・他方式へ転用しない。全decoderのrisk契約が後続の制約 |
| 残差pilotと全law価格は別 | **D39dddの有限development cohort**。m0/m6各fresh4 path。replayは独立標本でなく、rare/law/長q systematicは未資格 | pilot内のC/S比は測定法の診断用。ideal E[S]へD平均を足して全13roundのE[C]を作らない |
| まず絞るべきものは不確かな依存関係 | **条件付き設計判断**。全N資格と長時間C・PF数値安定性が未解決 | 新しい大格子やmodelより、どの欠測が判断を妨げるかを特定する有限案を優先 |

8候補対・4欠測対は既存B51500の6lane表とD receiptの一致を確認したmetadata件数で、科学値を再評価した数ではない。固定PRだけの記述をL_D/K/κ全体の最適性へ拡張しない。

## 固定SOURCE・入力

| SOURCE | 使用する保存物・用途 |
|---|---|
| A PR `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60` | `h4_pr_s1_s2_rmse_results_20261113/run01/N_HANDOFF.json`：13round nominal・元physical ID・SOURCE。全十分Nはnull |
| C構造 `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | `h4_structural_schedule_results_20261113/run01/COVERAGE.json`：26 E[S]と数値enclosure。compiled資格false |
| D pilot `39ddd59edf54a3a901e5302207b2765ddbd03fbb` | `d_p2_direct_pilot_20261113/run01/RESULT.json:summary`：既存fresh要約のみ。新compile/再fitを行わない |
| D Full PF `e7dca5f7301c8633b17a59b6a7161331303aef2d` | `d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.json`：既存73/136のreceipt分類。今回はH4の選択12セルだけを対象候補にする |
| A screen `699178305cec241b83220f6c307c65e0539a7856` | `h4_four_method_signal_screen_results_20261113/run02/`：4局所record・20状態。gate残差を再計算しない |
| B S3 `2084791b1b09cff65e07fa8566bb734e1e6bff8e` | `rpe_s3_readiness_20261113/metadata01/STATISTICS_INPUTS.json`と保存PRIMARY/HANDOFF。全trial archive再解析は今回/次案とも含めない |
| D区間 `24b0a6eb1d806391a3c6ad454fb0c018148d8545` | `d_rpe_price_search_design_20261113/HANDOFF.md`：law/lowering/同時区間・精密化の設計 |
| B adapter `b9b9ffa6233db9dd2b413dc07b27e7039324c666` | `rpe_shot_allocation.py`と既存controller/exact会計。nominalや構造Sで正式cost gateを通さない |
| A探索 `821c5c1ffd7c95e1690d45d2b6e8ff4ccb15bd1c` | `pr_rpe_search_design_20261113/HANDOFF.md`：whole schedule・候補生成/signal担当との境界 |
| 既存B menu `51500b724a75f9535a2c831d50bcfe09eabb9326` | 6 Full PF laneと保存locatorを再利用。11方式案は依然未採択、今回新候補を生成しない |

共通物理はlinear H4/1Å/STO-3G、保存tol-only DF10、sector(2,2)/36、保存H/state/snapshot、P0。PR canonicalは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、S2/L_D1/ΔPF=.1/K2/h=.1/M12・元exactκ。binding参照のT=.8を各round時間へ代用せず、R255等を2冪へ丸めない。

旧Full PFのnative receiptとDの新`CONTROLLED_ZZ_CX_CRZ_CX_V1`は別SOURCE。compiler hashが同じでもloweringの同一性を意味しない。後の閾値は指定実装scenarioの条件式であり、全方式へ同じ最適化が適用済みという主張をしない。

## 再利用と検証

既存`SavedArtifacts.read`で固定JSONを読み、`p.same`でidentity/law、nominalとS3入力の一致を確認した。`exact_scalar/exact_value`、`rpe_shot_allocation`、controller、`compiled_residual`、Dの`fresh_summary/cohort_summary`とPF inventory/receipt関数を調査した。疎なcheckoutにないD moduleも固定Git objectのSOURCEを参照し、他worktreeへ入らない。

数値再解析をまだ許可されていないため、既存cost/select/MC/mean/compile/fit/構造kernelを呼んでいない。proxyの算術は承認後にexact補助関数だけで十分で、誤ったquantityを受入れるためcontrollerを変更する必要はない。新production変更がないので数値回帰suiteの重複実行はせず、7 metadataチェックと文書/JSON/差分確認を行った。独立科学レビューではない。

Claudeの中間推定を数値入力・検証済み結論として使用していない。Track Aの有限T winnerも採用していない。

## 引渡しとSTOP

**ユーザー**へ求めるのは[NEXT_OPTIMIZATION_PLAN §1](NEXT_OPTIMIZATION_PLAN.md#1-承認を求める有限保存再解析b-prelim-v0)の有限保存再解析の実行許可。承認証明書・grant発行は不要。**B**は許可後だけ構造寄与・局所閾値・pilot比を計算し、正式G/winnerはnullに保つ。**A/C/D**の後続科学計算はそれぞれ別の有限scopeと許可を要する。

準備成果をcommit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。今回も、次に求める承認範囲でも、新H4 signal/MC/compile/残差fit/新PR候補の最適化は行わない。
