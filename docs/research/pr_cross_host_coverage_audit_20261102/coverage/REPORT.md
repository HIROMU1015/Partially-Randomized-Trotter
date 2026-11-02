# PR研究3環境Coverage監査 Phase 3：主張別評価案

LOCAL Codex／観測日2026-11-03 JST。metadata確認時点：2026-11-03T02:07:51+09:00。
**既存証拠に基づく暫定評価案であり、主要claimの正式採択、GPT/Claudeの独立した詳細科学レビュー、投稿可否の判断ではない。** 新しい科学値・平均・SE・N/G・比・順位・fit・数値感度を計算していない。

## 結論と判断の単位

現在の資料が掲げる「固定物理入力・有限候補集合・保存価格標本・明記した誤差仮定での結果」は、多くの必要な証拠接続を持つ。強い決定論対照を入れるとPRが負ける結果、H4粗精度の非再現、DF B3の負結果も、その条件を検証した証拠である。PRの勝利をcoverage完成条件にしていない。

一方、**実精度を保証した母平均の方式一般優位、Hamiltonian表現だけによる因果的費用差、目標energy精度での最終RPE資源優位**は、これらの局所結果から認定できない。全18roundの費用モデルは既に存在し、その完成と最終保証の未完成を分けた。

既存MatrixのA01–A13、R01–R02、F01、P01–P05の**21claim**を照合入口にした。これらは元資料の問い／限定結果のIDであり、新しく採択したclaimではない。[CLAIM_SCOPE.csv](CLAIM_SCOPE.csv) に原文固定参照、状態、適用域、比較対象、必要保証、直接証拠、現在要求しない検証を記録した。

[COVERAGE_MATRIX.csv](COVERAGE_MATRIX.csv) は44行。単位はclaim×明示された適用域×必要証拠接続で、歴史的照合2行を含む。31行を `SUPPORTED_WITHIN_RECORDED_SCOPE` とした。これは元数値・保証を再認証したscientific PASSではなく、記録された強さの主張に必要な保存証拠の接続が確認できたという判断案である。

| 状態 | metadata行数 | 読み方 |
|---|---:|---|
| SUPPORTED_WITHIN_RECORDED_SCOPE | 31 | 明記した限定条件で接続が揃う |
| PARTIAL_EVIDENCE | 4 | 旧proxy全系列、M1後続218集合join、B3最終版保全差、全law実ρ |
| NOT_EXECUTED_DOCUMENTED | 3 | H12直接係数、Y4/m5のq32、最終energy資源実証。各原資料の明示に限定 |
| EVIDENCE_UNLOCATED | 2 | 「追加7候補」の一意な対応とUWC原結果。未実施とは判定しない |
| NOT_ACCESSIBLE | 1 | H8確認主表に対応する詳細auditの公開アクセス制約 |
| NOT_APPLICABLE | 2 | H8現scopeの追加sampling義務、現P05に対する表現単独因果証明 |
| INVALIDATED_ONLY | 1 | 旧DF screeningの歴史的利用可否。後続有効研究へ失効を波及させない |

件数はmetadata行数で、独立科学実験数・成功率・検証不足件数ではない。矛盾するように見えた旧新結果は、候補集合、入力rank、cohort、bias政策、解析版の差を保持している。この限定照合で、同条件の重要な未解決数値矛盾を新たに確定したわけではない。

## 固定入力・規約・調査範囲

- 共通仕様：[PROTOCOL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md)、commit `968218c2b0c6280af6c1c02822603e3312aa410c`。特に§2、§4–7、§8–10を適用。
- Phase 2：[統合REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/604f0235840feb18d368f99ea0828f370ce70fc8/docs/research/pr_cross_host_coverage_audit_20261102/integration/REPORT.md)、commit `604f0235840feb18d368f99ea0828f370ce70fc8`。CANONICAL_CATALOG全3part、RECORD_LINKS全2part、UNRESOLVED全2part、RELATIONSHIPS、SERIES_INDEX、REFERENCE_INDEX、SOURCE_COMPONENTS、INPUT_FILES、VALIDATIONを再利用。
- 主張入口：[Scientific Landscape](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md) と [Claim–Evidence–Gap Matrix](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md)、commit `9a0bc7f17f47823e46c3945734127e9f0d09154c`。
- 原結果・SOURCE・終了監査：上記の固定参照からclaimに必要なJSON/CSVのheader、ID、cohort、状態、保存summary/特定fieldを読取。[EVIDENCE_INDEX.csv](EVIDENCE_INDEX.csv) に固定URL、blob、読取範囲、selectorを記録。

PROJECT_MAPを入口とし、前段で読了した研究概要、Track A横断inventory、70カードとSOURCE索引、VALIDATION_STATUS/manifestを探索の導線として再利用した。巨大な研究概要をこのPhase3で再び全行読了したとはしていない。既存統合文書は原結果の特定に用い、科学保証・正式受理は元の保存状態と採択記録の範囲に限った。

実際の主worktreeのAGENTS.md、出力branch固定版のAGENTS.md、親ディレクトリの規約候補、正式共通運用指示 `research_project_instructions_unified_2026-10-25.md` の§8.1、§8.4、§10を確認した。primary agent自身が実施し、sub-agentや独立reviewerへ委譲していない。Phase 0実担当はユーザー訂正どおり **SERVER 2**。固定PROTOCOLのLOCAL担当記述を変更していない。

読取対象はLOCALの既存固定Git objectと公開済みPhase 0–2・原研究資料。SERVER 1/2のfilesystemへSSH・転送・調査をしていない。公開資料の取得は別hostの現存確認ではない。IDEのactive file/open tabs、別作業のbackup TSVを実験実体・復元保証の根拠にしていない。未知の保存先の広域探索、archive全展開、raw/QPY decode・全hashは行っていない。

## Q1：限定条件で必要な証拠が揃う結論

研究全体としては、**入力と誤差仮定→保存N、lawと回路実装→保存価格、両者→固定Tの保存G**という接続が主要な実証範囲である。次の表はその強さのまま記述できる範囲を示す。表中の科学的件数・方向は元記録の引用で、今回の再集計ではない。

| 結論と範囲 | 直接支持する保存証拠 | 保持する条件 |
|---|---|---|
| 小系PF係数・finite RTEは、保存窓・元lawで数値整合する | [PF係数summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json)、[H4 finite-RTE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_v1.json)、A01/A02の各元grid | H2–H5 DF3/5/7/9、旧H6 DF11/L_D5と後続tol-onlyDF19を別にする。C_useは経験包絡。H3単一位相不通過を隠さない |
| H4 developmentのB2点Paretoと、固定構成の1.30Å transferがある | [M1元結果検証](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/pr2_matched_accuracy_m1_b1_result_validation_v1.json)、[M2保存結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json) | legacyDF12、STO-3G、T.8、元候補/精度/compiler/P0。固定5構成transferとgeometry別再最適化を分ける |
| 強いB0I/PF対照でPRの優位が変わる | [H4 geometry地図](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/precision_grid_1812.csv)、[H6実測比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/artifacts/resource_applicability/track_a_v1_p7_measured_reaccount_run01/2026-10-16/comparison.csv)、[H8更新比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/winner_cost_differences_exact.csv)、[共通PF比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/comparisons_60.csv) | H4 legacyDF12、H6/H8 tol-onlyDF19/28、通常signed/非相殺、実測/modelの区別。全面優位・単調サイズ利得とはしない |
| 長時間B2は取得価格集合内の33/40で低G、対応方向を別価格cohortで確認 | [選択表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/best_configurations.csv)、[確認比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/independent_comparisons.csv)、[入力/SOURCE binding](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json) | H4/H6 linear1Å/STO-3G/DF10/19、T=.8/1.2/1.6/2、5精度、S2/Y4/m5、候補別L_D/q/R/K、P0、経験的u、元compiler。33/40は旧取得集合に限定 |
| DF B3の取得域ではB2/PFより低Gでなかった | [40比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/comparisons.csv)、[固定nominee確認](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/fixed_nominee_confirmation.csv)、[元summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/accounting_summary.json) | 同H4/H6 DF10/19・T≤2・5精度、二体L_D0＋one-body保持、q≤64/R≤512/K2/4/6。保存0/40・高次対S2の0/8を未実施へ落とさない |
| Pauliでは最新有限集合の14/15精度条件で純RTEがHybridより低G | [既存15条件統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md)、[H2原表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv)、[H4/H6原選択表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/selected_cost_breakdown.csv)、[H6追加後比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/COMPARISON_RESULTS.csv) | linear1Å/STO-3G/中性singlet/Pauli14/184/918/T.8、central、Qiskit1.3 opt1 seed17、P0・仮想ρ0。H4 ε.05のみHybrid。15独立実験ではない |
| H6 Hybridの大event方向は追加検証済み | [固定4候補原表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv)、[元報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md) | p78/q8/r16/32/64/K2等の登録4候補。探索nomineeと確認後min、純RTE旧n48とHybrid新n64を区別 |
| 短4roundのsignal/成功率診断/分枝と、全18roundの資源モデルがある | [4round原結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/rpe_four_round_phase_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json)、[全roundモデル原結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json) | H4DF12。短δ.1/L_D3/r4/K2/q1,2,4,8と、後続δ.01/.02/L_D3対12/opt2を分離。モデル区間重複・直接価格domainの制約を保持 |

DF長時間研究は、保存H6 weighted-Hermitian completionを含む入力を用いる。別のcompletion前Hamiltonianとの同一性を推定しない。DF B3はone-body Gaussianを保持し、Pauli純RTEは全非identity項をrandom化するため、L_D0という表記だけで同一方式にしない。

SOURCEのprovider/native/confirmationはPhase2の13 component bindingと原scientific_provenanceで役割を分けた。例えばB2のprovider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` とnative `27ebabdd2943a4de787ec8b976c38e06fda5a497`、B3のnative `638e0a989b4471a075549ccb58617b1030e6748f` とconfirmation `ae3d301e2318a3f9f1331c047c68d310771b6849` は、異なる実験を示すという理由だけで扱わず、元bindingを保持する。carrierを科学SOURCEへ置換していない。

## Q2：公平性・独立性・誤差保証・回路価格の残る境界

| 接続 | 確認できたこと | 未解決／強められない主張 |
|---|---|---|
| bias→headroom→N | 原条件のB一回適用、保存bias/u/ρ仮定、軸別N、null不適格を参照 | DFの経験的allowanceを厳密bias保証へ、Pauliの仮想ρ0を実ρ0へ置換できない |
| law→sampling→価格→G | 元law/候補/cohort、paired両軸、実compiled RZと保存N/Gを対応 | nは価格標本数で量子shot Nではない。未観測event費用・ゼロ標本分散を0保証にしない |
| 強い対照と候補機会 | B0I、高次PF、共通588、長時間拡張、H6 Hybrid追加を保持 | 全prefix/q/r/K/次数の対称全域探索は未完成。proxyで選んだ未価格候補を実測劣位にできない |
| 探索→確認 | 固定nominee・別価格cohort・確認後min/逐次延長・回復再利用の記録を区別 | 全歴史seed非衝突や選抜後同時coverageは保証しない。99関係を99独立cohortと数えない |
| native→compiled | occupation検査と保存6局所Arbの包含結果がある | [R2 proof_summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/proof_summary.json) はPROOF_INCOMPLETE。局所6点のρ上端を全lawや最新全候補へ転送しない |
| 回路価格と機構 | 旧H4同物理input/ray、D1のfull/retained/eventとD2同body control対照がある | D1は固定cos/index0、D2はcontrol＋compiler合成効果。表現、law、Gaussian往復、biasの一意な因果寄与率は得られない |
| round→energy | 4round位相復元と18round cost modelを別に保持 | q≤32の限定直接holdoutを全長round signal/branch/energy保証にしない。モデルの点差を最終優位へ採択しない |

長時間の旧33/40は、signal 11,632構成と同じ範囲の全価格最適性ではない。元native selectionは基本格子の未価格適格候補を明記し、q32拡張はS2に限定する。後続B3 comparison_bindingにも旧33/40保持が保存される。「追加7候補」は本調査で固有IDと固定結果の一意な対応を得られずG01へ残した。33/40を書き換えたり、q32の22価格候補を7件と読み替えたりしていない。

## Q3：科学的未検証と区別すべき証拠整理

**G01–G05、G14–G15はD：既存証拠の照合が先である。** 新しい平均・順位・simulationで埋める事項ではない。

- G01：7候補の元ID/固定commit/pathを、長時間研究の元索引と照合する。
- G02：claimが要求する範囲だけ、nominee固定時点、匿名cohort/seed-index索引、価格共有、再利用・回復の対応を結ぶ。LONGSEEDSはseparate_cohort=trueかつcomplete_historical_coverage_claimed=false。完全独立の保証を捏造しない。
- G03：B3公開版と既存調査中local最終版の差。先行固定調査は主要表一致と6file差を記録するが、今回LOCAL object storeでは短縮ref f794a2aaを解決できず、その比較を再認証していない。公開主要結果と最終private custodyを分ける。
- G04：H8確認の詳細audit公開制約。主表2行はMechanism内同証拠aliasであり、追加実験としない。必要資料は元担当環境が公開可能な匿名projectionとして準備し、LOCALから接続しない。
- G05：M1後続218集合・PM解析版のjoin。元210cellと後続reviewは利用でき、未展開の解析表を未実施とはしない。
- G14：UWC原JSON等は既存限定探索で未発見。proseの値を性能証拠として復活させない。
- G15：旧DF無効化37表のhistorical bytesは発見済みだが、修正canonical/anchor raw/run provenanceは未回収。manifestの `invalidated_pending_regeneration` を維持する。

Phase2の全103,141groupに未解決metadataがあることを、103,141科学的不足へ換算していない。Phase2の原状態も変更していない。必要な固定結果fileに一致する6,528groupを検索用に結び、同一fileの別引用aliasも含む6,568リンクを [EVIDENCE_GROUPS.csv](EVIDENCE_GROUPS.csv) に置いた。これは網羅監査・独立実験数の認定ではない。元状態・未解決metadataを横に保持し、coverageの判断をその一律のUNKNOWNだけで決めない。Phase1にgroup未登録の直接保存表も、固定URLから引用できることと独立認定を分けて扱う。

## 不足候補A–Eと暫定優先度

[GAP_CANDIDATES.csv](GAP_CANDIDATES.csv) の18候補は、D7、C7、B2、E2。**Aの確定候補は0**。これは現在の限定条件付き主張を越えない評価の結果であり、原研究目的の全達成や、論文の正式な中心claimの受理を意味しない。

- **D**：上記7件。claimの解釈を変え得るG01/G02をmetadata照合の優先候補とし、他を一律に閉じる義務は設けない。
- **C**：G06実ρ、G07母平均tail、G08方式一般の探索/選抜公平性、G09全価格域、G10最終energy/RPE、G11因果機構、G17物理state-prep。現在の限定報告には必須でなく、強い主張を選ぶ場合の不足。
- **B**：G12物理domainの一般性、G13proxy再利用/再現性の補強。分子数・標本数が少ないことだけを優先理由にしない。
- **E**：G16アルゴリズム改善方式自体、G18noise/実機/FT/実通貨・別hardware比較。改善研究の直接baselineを使う場合も元SUPPORT_ONLY条件を越えない。

原RQ3の最終energy費用実証や、実精度保証・母平均winner・単因子機構を論文中心claimに**ユーザーが選ぶ場合**、G06/G07/G10/G11等はAへ再評価する必要がある。現在その強いclaimを自動採択していないため、先にCとして境界を明示した。過去U01–U10の分類・優先度を無条件に継承していない。

H8 rareについては [正式終了政策](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md) を優先する。層別finite mean＋残余条件付き感度は既存正式採択済みで、現scopeの追加sampling不要という判断とSTOPを維持する。残余S2_MULTI/S4が未知でも、現scopeを未完了へ戻さない。

## Q4：結論を変え得る追加作業の候補

全候補を実行する提案ではない。現在の有限比較を記述するだけなら、新しい科学計算を一律に要求しない。次を明確に分ける。

| 作業種別 | 必要な場合の具体的候補 | 判断が変わる問い／担当 |
|---|---|---|
| 既存データの照合で完結 | G01のcandidate ID対応、G02のcohort/nominee共有索引、G03/G04の公開可能な終了metadata、G05の保存解析版join | 独立確認・比較集合・保全の表現が変わるか。Codex候補。所有環境の担当資料をユーザーが別途指定 |
| 保存データの新しい数値解析 | 主張を強める場合だけ、対象を選んだ実ρ/保存guardの適用、価格tail/選抜/モデル移送・stateprepの新感度 | 点方向や保証水準が変わり得るか。ユーザーの別明示許可後にCodex。今回は新N/G/SE/順位を作らない |
| 新規科学計算 | 最終energyを中心にするなら必要長roundのsignal/成功率/分枝・価格移送。機構を中心にするなら既存D1/D2を再利用した限定central部品対照。母平均主張なら対象lawのtail境界/必要層の限定検証 | 最終応用への接続、因果説明、母平均順位が成立するか。先にユーザーがclaimと判別したい問いを決め、必要scopeのみ別承認 |
| 追加作業不要 | 本REPORTの限定条件と未保証を明記した既存結果の記述、実施済み負結果、H8現scope | 自動追加計算なし。研究上の正式採択・文章化はユーザーの判断と別指示 |

計算条件・新候補集合・SOURCE・budgetをここで正式固定していない。既済価格やholdoutを使うだけで足りる場合は取り直さず、同じ科学結果を再ランキングして「照合」と呼ばない。H8終了研究や他の科学STOPの再開は提案の既定にしない。

## Q5：追加計算なしで論文に記述し得る範囲

案として、(1)固定DF入力でのmatched signal精度・有限価格標本の資源trade-off、(2)強い決定論対照と候補集合変更で結果が変わること、(3)H4固定transferと長時間B2/B3の対照的な結果、(4)Pauli最新有限集合での純RTE/Hybrid比較、(5)短RPE検証と全roundモデルの到達点を、出典・仮定・選抜・cohort共有・STOPとともに記述できる。

独立確認を記述する際は「記録された別価格cohortによる方向確認」とし、全履歴独立や外部再現へ拡張しない。比較集合は各固定保存版と明記し、「追加7候補まで確認済み」「全価格候補で最適」と書かない。Pauli14/15は確認後選抜・共通価格・ρ0の条件を本文に残す。単回CとN込みG、固定Tと全energy、実測/model/proxyを分ける。

これは**投稿準備完了の認定や論文執筆の実施ではない**。G01/G02等、採用する具体的な文言に必要なmetadataの追加照合が残り得る。原論文・概要・研究採択記録は変更していない。

## Q6：独立した詳細科学レビューが必要になり得る論点

今回のmetadata評価案作成・技術検査を完了するために、別の詳細レビューを自動開始しない。ユーザーが以下を主要claimへ採る場合が候補である。

1. **保証された資源優位を主張する場合**：経験的u、局所包含、全law実ρ、未観測価格tail、選抜後coverageをどこまで要求するか。必要資料は対象claim/入力/law、R2STATUS、ARBSIX、保存guard、sampling/nominee計画、rare表、H8終了政策。
2. **固定Tから最終energy/RPEへ進む場合**：直接domainと外挿、state/prior、成功率とbranch、費用境界を合わせるか。必要資料は原目標、RPE4JSON、RPEJSON、horizon/schedule、直接holdoutと不確かさの定義。
3. **Pauli対DFの因果機構を中心にする場合**：同物理入力でも分割/law/controlが交絡する比較から何を帰属できるか。必要資料はPXINPUT、D1/D2、B3 one-body SOURCE、最新Pauli選抜表、compiler/control対応。
4. **独立確認の強い表現が必要で、cohort照合で実際の重複・事後利用が判明した場合**：G02で得た匿名cohort対応、固定時点、共有価格・回復履歴を先に提示する。metadataが未整理なだけならCodex照合で足りる。

いずれもレビュー候補と必要資料の提示だけ。GPT/Claudeによる詳細科学レビュー、研究方針の変更、主要claimの正式採択は **NOT_STARTED / NOT_ADOPTED**。最終判断の担当はユーザー。

## 技術検査・効率・原本保全

Phase2の `metadata_integration.py`、`integrate_catalog.py`、`verify_outputs.py`、`test_metadata_integration.py` を静的に確認した。適合する固定Git参照、selector、literal転記、匿名化、CSV読取の考え方と小関数を再利用した。Phase2は実験identityを統合する43列系、Phase3は17列/12列のclaim評価なので、既存groupの再統合・全119,304観測の再詳細監査は行わず、小さな [check_coverage.py](check_coverage.py) を追加した。

独立した3catalog partの読取を並列化し、全partをstreamingで一度選別してclaim関連fileのgroupへ限定した。RECORD_LINKS/UNRESOLVED/関係・SOURCE/参照/入力索引を再利用し、元状態を保持した。CPU affinity32、起動時MemAvailable約52GiB、一時領域空き約225GiBを確認。人工的CPU/RAM上限、共有設定変更、他ユーザー/job操作は行わない。

検査対象は17/12列順・UTF-8/RFC4180、claim/参照/gap/selectorの整合、原表の保存fieldとのliteral一致、選択groupの固定Phase2への一致、公開情報、原本HEAD/tracked status、差分範囲。科学値の算術検算・新ranking・scientific PASS発行はしていない。結果は [VALIDATION.json](VALIDATION.json) に記録する。原JSONのnull、不適格N/G、UNKNOWN/NOT_RECORDED/NOT_ACCESSIBLE/NOT_FOUNDは原本で保全され、0補完しない。

証拠treeは `GIT_OPTIONAL_LOCKS=0` / `GIT_NO_LAZY_FETCH=1` のread-only。独立した出力Git/indexで新branchのみ作成。元worktreeのcheckout/reset/merge/clean/gc/prune、branch/worktree/archive削除、Phase0–2成果物・科学SOURCE/input/state/reference/raw/QPY/result/grant/ledger/marker/STOP、正式採択記録・既存論文・整理保全資料の変更はない。並行する他作業のref/未追跡追加は非atomic観測として記録し、巻き戻さない。

内部host/IP、絶対保存path、資格情報、private raw/seed値・逆引きは公開しない。公開CSVは固定公開URLと匿名の元観測IDだけを持つ。旧文書の内部pathを本文から転載しない。原本のprivate/raw保存状態は元環境の記録として扱い、今回の独立backup/全復元確認とは呼ばない。

## 公開・mandatory STOP・次の担当

新branch `codex/pr-cross-host-coverage-phase3-20261103` へ本coverageディレクトリだけをcommit/pushする。force push、main merge、既存branch更新は行わない。公開後にremote SHA・全公開fileのblob/sizeと本文bytesを照合し、最終SHAと結果をチャットで報告する。本REPORTに自己参照の最終SHAは埋め込まない。

**技術調査・暫定評価案・検査・公開照合の完了後にPhase3 mandatory STOP。** 新規科学計算、保存値の新数値解析、詳細科学レビュー、論文執筆、新しい研究方向や主要claimの自動採択へ進まない。旧研究のSTOPも維持する。

次の担当は **ユーザー**：限定claimの採用範囲と、必要ならG01/G02等の既存証拠照合・Q6レビューのどれを別途指示するか判断する。Codexの自動継続作業なし。
