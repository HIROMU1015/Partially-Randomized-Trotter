# 論文ストーリーと旧原稿の再利用案

2026-11-03 JST。推奨候補O1の構成設計であり、新しい完成原稿・abstract・投稿判断ではない。番号は設計用。図表は既存保存値・既存assetの配置計画で、今回の再描画、数値join、再ランキングはない。[範囲候補](PAPER_SCOPE_OPTIONS.md)、[主張対応CSV](CLAIM_EVIDENCE_MAP.csv)

## 1. 論理の順序

「部分ランダム化を実装した」から始めるより、**何を対照にすると、どの条件で資源を節約できるか**を主問にする。taskと費用単位を固定し、小さなdevelopment事例から強い対照、入力／精度／時間、負結果、証拠の限界へ進める。最終energyの優位はこの問いから自動的に導かれない。

| 節 | 役割・主張 | 使用証拠 | 図表候補 | 読者へ伝える結論／避ける飛躍 |
|---|---|---|---|---|
| 1 Introduction | PRの一般原理と、本稿の実装資源・適用条件という問いを区別 | LANDSCAPE、PMMAP、旧稿先行研究、Günther v2 | 本文は概念図が必要なら方式模式図。数値図不要 | 一般PR・DF split・discard比較を新規性にしない。中心claimの独立性／論文価値の正式認定はしない |
| 2 Task and accounting | A01/A02/A03/F01。finite-time signal、bias/B/N、full-wrapper C、G、選抜と価格cohortを定義 | COEFF/FINITE/COST、旧Methodsと補足 | 表1：各系列のHamiltonian、rank、state、T、PF、law、compiler、精度、候補・cohort・価格版 | nは古典価格標本、Nは十分量子shots。実shots・物理価格ではない。C_useは経験的窓。RZ/native T/logicalを分ける |
| 3 A controlled H4 development case | A04/A05。旧H4 DF12で近接discard後の低B2点値、固定5構成transferと事後感度 | M1/M2、PM0–2、旧C1–C6 | 旧図1の限定panel、旧図2/3は必要部分を補足へ | 旧218／固定5という明確な比較。held-out全再最適化や302独立実験へ拡張しない |
| 4 Strong deterministic alternatives | A06/A07/A10。B0I・高次PFで適用領域が変わる | B0I geometry、H6 p7、H8強化184、PF588 | 表2／図2：別domainの保存対照表。各表に実測/model価格を表示 | Full PFだけとの比較では不十分。rank・state・候補数の差を同一scaling実験に見せない |
| 5 Precision, time and confirmation | A08/A09/A11。固定別価格方向、高次の限定利点、長時間B2 | H468主表、HIGHCONF、LONG、G01/G02 | 表3：H468の6固定比較。図3：既存長時間40セル表を探索／確認／追加集合別に配置 | REPRODUCEDはPR勝利と同義でない。33/40は取得集合内。旧642、q32追加22、追加7、確認価格を分離 |
| 6 Limits of randomizing the two-body part | A12/A13。DF B3負結果とH8 finite-tail scope | B3/そのSOURCE、H8 closure | 図3のB3対照panel又は表4。H8層別詳細はSI | one-body保持B3の有限域負結果。全RTE不可能性ではない。未観測層は0にしない、既存STOPを保持 |
| 7 Discussion | 実装価格とshot余裕の競合、比較集合依存、表現・状態・compiler・保証の限界 | 旧same-R例、全DF表、任意Pauli、RPE補足 | 新規数値図を前提にしない。制限と証拠層の表 | 新しいq因果則、表現単独効果、母平均winner、noise/FT/実通貨一般化を導かない |
| 8 Conclusion | 固定taskでの正負の適用条件をまとめる | 上記限定claimのみ | 追加図なし | 研究目的全体のenergy費用実証完了としない。正式なclaim文はユーザー選択後に執筆 |

表1の条件は元SOURCE・原結果・各検証文書を参照する。未記録fieldを分子名から補わず、legacy DF12とtol-only DF10、H6 weighted-Hermitian completion等を別条件にする。共通δ/T・基準state・rank policyをそろえたような新しい合算図を作らない。[全体地図][LANDSCAPE]、[共通PF][SRC_PF588]、[長時間][SRC_LONG]

## 2. 旧v0.1の保持・書換え・追加

旧稿の日付・比較domain・結果は履歴としてそのまま保全する。本案は別稿構成であり、旧fileを更新していない。[旧本文][OLDMAIN]、[補足][OLDSUPP]、[執筆者audit][OLDAUDIT]、[PM後設計][PMDESIGN]

| 旧資産 | 再利用可能な部分 | 新構成で必要な扱い |
|---|---|---|
| Abstract/Introduction | H4固定taskの問題設定、一般partial randomizationを先行法とする位置付け | O1なら強い対照と負結果を中心へ。旧約43.07%だけを広い研究の要約値にしない |
| Methods §§2.1–2.4／S1–S3 | finite-RTE、適格性、十分N、paired軸価格、P0、development/transfer分離 | 後続rank/state/law/PF/opt/候補・価格cohortを系列別に追加。旧32標本規則を全研究に一律適用しない |
| C1／旧図1 | 旧218、ε=.05、B0 L5対B2という局所点差 | 本文の導入事例又はSI。B0I・高次PFへの最良性を示す図にはしない |
| C2／旧図2 | 固定302表示点とB2内部q設定変化 | 同じ価格を共有するPOSTHOCであるとcaptionに保持。後続B0I域のmethod切替と別panel |
| C3／旧図3 | 事前固定5構成のM2と後続precision適格性の分離 | 最適化移送という表現を避ける。元精度TRANSFER_SUPPORTEDを残す |
| C4／旧S3 selector | 同M1内q8対全体、primary保持と他metric Pareto損失 | 旧selectorの監査として保持。後続方式の実行計画を否定／検証したことにはしない |
| C5／旧図4 same-R | 保存一組でのbias/B/N/C/G競合 | 説明例として保持。compiled det/random/basis分解やq一般因果則を足さない |
| C6／旧図S1 | 共通仮想Pと候補集合の依存 | 補足向き。実状態準備回路・全RPEの準備費用へ読み替えない |
| References／claim audit | 一次版への固定参照と過大claimを避ける境界 | 後続証拠を追加し、書誌名の旧暫定表現をコピーしない。旧執筆者auditを新原稿の独立reviewとしない |
| 新規節4–6 | 旧稿にはない | B0I、H6/H8強対照、共通PF、別価格確認、長時間B2/B3を明示的な後続として追加 |

旧稿が「未評価」とした項目のうち、後続で高次・長時間・RPEの限定検証が完成した部分は、新稿の現状説明を更新する。ただし旧条件の結果を遡って差し替えない。物理state-prep、実compiled全law残差、最終energyでの頑健資源優位等は依然区別する。

## 3. 図表設計と保全

| 図表資産 | 配置案と出典 | 表示上の条件 |
|---|---|---|
| 旧主図1–4、S1、表示CSV | 旧条件の小事例・Methods補足として利用候補 | 元asset manifest、candidate ID、nominee、価格版を保持。今回再生成・コピー改変なし |
| B0I／強対照／PF588保存表 | 節4の複数panel・比較表の設計入力 | ε順、signed/noncancellation policy、実測/modelを別表示。異なる候補集合を一本の系列にしない |
| H468独立固定主表 | 節5の6比較表 | H4 legacy DF12確認を加算しない。H8 2行のMechanism aliasを注記。科学方向と技術statusを別欄 |
| 長時間のbest-by-method／G01対応 | 節5–6のB2/B3比較 | 旧33/40と拡張探索33/40を混ぜない。追加7は異なる候補集合、探索・確認価格のpoolなし |
| Pauli最新保存精度表 | 別節／別稿候補、Table1のdomainを利用 | 最新H6 Hybrid候補を反映した表を選ぶ。旧図のHybrid水平線は旧解析版としてのみ利用 |
| RPE V1–R1既存表 | SIで成功条件と価格不確かさの二表 | V2 CONDITIONAL_SUCCESSとR1 UNDETERMINEDを一緒に示す。長q価格を実測点として描かない |
| Track B G9/D3/C1表 | 別note候補 | native T/CX/logicalとwhole RZを換算・合算しない。構成証拠と分子全task費用を分ける |

必要な新しい作図や文章化は、採択する範囲が決まった後の別指示で行う。この設計の完成のために新しい数値CSVや図を生成する必要はない。既存図のlabelだけを最新値へ変更する方法も採らない。[主張図表対応][PAULICLAIMS]、[G01/G02][GREPORT]

## 4. 補足と分岐案

**SI-A Methods/evidence**：rank/state/reference、SOURCE/input、finite law、compiler/control/測定境界、nominee固定時点、cohort共有、旧selector、原本への固定URLを置く。全rawを公開したという主張はしない。

**SI-B RPE条件付き接続**：V1共通E0/半径、V2の18round十分成功条件、V3の直接価格tuple、R1の共有較正／長q感度の保存結論の順にする。結論は「条件付き成功は支持、費用優位は未確定」。研究全体のRQ3を完了したとは書かない。[V1][V1]、[V2][V2]、[V3][V3]、[R1][R1]

**Pauli独立節を選ぶ場合**：DF結論の後に、別Methods＋最新H2/H4/H6内部比較＋H4粗精度反例＋N/C競合＋DF比較の交絡を置く。14/15とH6追加を出典のまま使い、表現の因果機構はDiscussionの未保証事項とする。分量と主問の明確さを優先するなら別稿へ分ける。[Pauli統合][PAULISYNTH]

**Track Bを分ける場合**：構成の入力・出力・law・有限bit保証、既知法との対応、matched CTS、G9/D3の負担交換、H4 C1の到達点を主系列にする。G6の構成的独立性は既存未解決事項のまま。これをDFの実測compiled優位の証拠として挿入しない。[利用可能性][SRC_BSURVEY]、[最新claim表][SRC_BLATEST]、[C1][SRC_C1]

## 5. 執筆へ進む前の分岐

ユーザーが中心範囲と配置を選び、正式に採用するclaimの強さを決める。Codexはその次の指示に合わせて原稿を作成できるが、本作業では完成稿・詳細科学レビュー・投稿を開始しない。論文を強める場合だけ必要な科学検証と、単に保存資料を回収すればよい問題は[status表](STATUS_AND_OPEN_QUESTIONS.md)で分けた。公開・照合後mandatory STOP。

<!-- fixed-reference-definitions -->
[GREPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/REPORT.md>
[LANDSCAPE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md>
[OLDAUDIT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_claim_audit_v0_1.md>
[OLDMAIN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_v0_1.md>
[OLDSUPP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_supplement_v0_1.md>
[PAULICLAIMS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_claim_figure_map_20261101.md>
[PAULISYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md>
[PMDESIGN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_post_pm2_manuscript_design.md>
[R1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fde197c64da671f1d428f0d3e8fdb762dc21930/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md>
[SRC_BLATEST]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/CLAIM_EVIDENCE_MATRIX.md>
[SRC_BSURVEY]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/results_usability_map.md>
[SRC_C1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md>
[SRC_LONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md>
[SRC_PF588]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md>
[V1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md>
[V2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md>
[V3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md>
