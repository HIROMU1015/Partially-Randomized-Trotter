# PR研究の成果・主張・証拠と論文化候補

2026-11-03 JST。LOCALが主担当となり、3環境に由来するGitHub公開済み成果を統合した。これは既存資料の整理と構成候補であり、新しい独立科学レビュー、中心claim・論文分割・投稿先の正式採択ではない。新科学計算・数値再解析・平均/SE/順位算出・sampling・compile・完成原稿執筆は行っていない。

**最も直接的な中心メッセージ候補は、固定DF表現の有限時間signal推定において、partial randomizationの競争力が、強い決定論対照、要求精度、時間、入力・候補集合に依存することを実装資源で示した、という適用条件の研究である。** B2が有利な保存条件と、B0/B0+I・高次PFが有利な条件、DF B3の負結果を同じ論旨に置く。partial randomizationという一般原理や、常にPRが最良という論文にはしない。[全体地図][LANDSCAPE]、[旧claim matrix][LANDMATRIX]

論文化の編集上の推奨候補は、**DF資源・適用条件を本体、Pauli-JWを分離可能な副研究、RPEを条件付き接続の補足、Track Bを構成・native会計の別系列**とする案。比較task・保証・価格単位の違いを保つためであり、正式判断はユーザーへ戻す。[範囲候補](PAPER_SCOPE_OPTIONS.md)、[構成案](PAPER_STORYLINE_OUTLINE.md)

## 1. 研究目的と比較単位

元の問いは、Hamiltonianのどこを決定論的に実行し、どこを有限RTEで補うと、同じ精度・成功条件の推定資源が減るか、である。主要な実測研究のestimandは保存状態における有限時間複素信号 `<psi|exp(-iHT)|psi>`。主費用は状態準備を除いた `N_cos C_cos+N_sin C_sin` で、Nは解析上の十分量子shot数、Cはfull Hadamard wrapperの保存compiled RZ価格、nは古典trajectory価格標本数。この三者を混同しない。

DFではone-bodyと固定二体fragment列を使う。表現のDF rank、決定論prefix長L_D、PF次数、Taylor偶数cutoff K、outer q、内部r/R、物理時間と無次元時間を区別する。

| 方式 | 保存研究での意味 |
|---|---|
| Full PF／旧B1 | one-bodyと全二体fragmentを決定論実行 |
| B0 | prefix以外のtailをdiscard |
| B0+I | discard tailの既知identity成分をscalarへ一回戻す |
| B2／Hybrid PR | one-bodyと非空prefixを保持し、tailをfinite RTEで補完 |
| DF B3 | 二体prefix0、one-bodyは決定論的に保持。全Hamiltonian純RTEではない |

finite-RTE bias、正規化Bによるattenuation、基準信号半径、headroomとN、whole compiled Cの競合を同時に扱う。経験的allowance・実行窓のC_useは厳密error certificateではない。RZ count、native Clifford+TのT/CX、logical計数、物理runtimeは別単位である。[旧Methods][OLDMAIN]、[finite RTE][SRC_FINITE]、[価格検証][SRC_COST]

## 2. DF主要成果と成立範囲

以下の数値・方向・件数はすべて原保存結果の引用。今回勝敗・比率・平均を計算し直したものではない。

| 系列・条件 | 直接支持される保存結果 | 同時に書く限定・負結果 |
|---|---|---|
| PF係数・finite RTE基礎 | 小系dense/state-action、H4係数・有限lawのsignal/phase/radius等の局所検証 | H3単一位相近似の不通過を含む。旧H6 DF11/L_D5と後続DF19は別。全δ・H12・全lawの保証ではない [係数][SRC_COEFF] |
| H4 development、1.00Å/STO-3G/legacy DF12/T=.8/S2 | M1の210、PM-1近接discard8を含む218。旧ε=.05/P0ではB2の登録discard比約43.07%低いprimary点値 | 全prefix・強い高次PF/合成への優位ではない。旧16cell selectorは6指標Pareto一件を落とす [旧稿][OLDMAIN]、[PM対応][PMMAP] |
| H4 fixed transfer、1.30Å、同DF12/T=.8 | developmentで結果前に固定した5構成の元精度移送はTRANSFER_SUPPORTED。B2 r4/r8がpoint Pareto | held-out全再最適化ではない。別精度の事後感度では固定B2が不適格となる領域がある [M2][SRC_M2] |
| H4追加6geometry、legacy DF12/T=.8/S2 | 旧218集合で全1812精度点B2点最小。B0Iの120構成/240軸追加後はB0I397/B21415 | 同じ価格標本の事後地図で1812独立実験ではない。B2全面優位という旧集合の読み方を拡張できない [B0I][SRC_B0I] |
| H6 tol-only DF19/T=.8、強化対照 | ε=.05のB0I p7/q1を実測すると保存PR確認点より安い。限定PR削減率−13.29012029% | 元136全候補を実測したのではない。他精度のmodel単価、保存PRcohortを保持 [p7実測][SRC_H6STRONG] |
| H8 tol-only DF28/T=.8、強化184 | 通常signed・allowance1でPR削減率は5精度順に−8.0230/33.2373/31.4549/22.9613/−50.8125% | 一部model価格が残る。非相殺policyの別表や独立固定主表の方向と混ぜない [6軸反映][SRC_H8STRONG] |
| H2/H4/H6、DF3/10/19/T=.8、共通588決定論候補、q≤8、S2/Y4/m5 | H2全5精度でPF低G。H4/H6は粗い3精度PF、厳しい2精度PR低G | PR側は保存探索集合。全domain対称探索や一般系サイズ則ではない [共通PF][SRC_PF588] |
| 固定H4/H6/H8別価格cohort主表、DF10/19/28/T=.8 | H4粗はNOT_REPRODUCED・PF低G、H6粗はPF低G方向REPRODUCED。厳精度は保存PR低G。H8主表2点はPR低G | REPRODUCED≠PR勝利。H8はMechanism内同cohortのaliasで追加独立証拠ではない。強化184域とも異なる [6比較][H468SYNTH] |
| 高次PR固定確認 | H2/H4/H6の登録3比較・118価格trajectoryで固定S2に対する改善方向を確認 | 最良強化PFへの勝利ではない。高次は一律有利／不利のどちらでもない [高次確認][SRC_HIGHCONF] |
| H4/H6長時間、DF10/19、T=.8/1.2/1.6/2、5精度 | 旧取得集合でB2が決定論PF envelopeより33/40点低G。対応する方向を別価格cohortでも確認 | signal11632構成と価格642候補を分離。envelopeはB0/B0I/Fullを含む。高次の一部探索優位は確認で反転 [長時間][SRC_LONG] |
| 同長時間DF B3、q/R/K拡張 | 取得40条件でB2より低G0/40、Full PF／決定論envelopeより低Gも0/40。厳精度m5対S2確認0/8低G | one-body保持のDF B3に対する有限域負結果。全RTE不可能性でもPauli純RTEの否定でもない [B3][SRC_B3] |

tol-onlyは返却rankを強制する指定ではなく、保存入力の観測rank。H4 legacy DF12とtol-only DF10、H6 weighted-Hermitian completion、state solver/正規化/fragment順の差を維持する。分子名だけで全結果を合算してscaling曲線にしない。

### 長時間のG01/G02追補

G01の追加7物理候補は原登録・未価格状態・スクリーニング・実測SOURCE・保存input・探索／確認価格へすべてMATCHED。旧642候補にもS2 q32拡張22候補にも含まれない。旧集合33/40と、追加後の拡張探索集合33/40は別の保存結果として併存し、確認価格を探索minへpoolしない。[追補REPORT][GREPORT]、[7候補binding][G01]

G02の100対応行は独立100実験ではない。DOCUMENTED_DISJOINTは列挙cohort/indexと保存監査の範囲、RECORDED_SEPARATEはそれより限定した記録、H8 aliasは再引用である。回復・同seed継続・cache再利用を独立確認へ加えない。Phase3原statusは[対応CSV](CLAIM_EVIDENCE_MAP.csv)の原欄に保持し、追補を別欄にした。[G02][G02]

## 3. Pauli-JW：関連するが独立した問い

linear H2/H4/H6、1Å/STO-3G、中性singlet、非identity項14/184/918、S2/T=.8、central制御、Qiskit1.3/opt1/seed17、P0・仮想compiled残差ρ0の保存比較では、純RTEがHybridより14/15精度条件で低G、H4 ε=.05だけHybrid低G。H6 Hybridのr32等の追加後も保存方向は維持された。5精度は同じ価格標本を共有するため、15独立実験や母平均winnerの証明ではない。[Pauli統合][PAULISYNTH]、[主張・図対応][PAULICLAIMS]

N/Cの会計競合は説明できる。H2粗・H4厳では純RTEの大きいNを小さいCが補う。H4粗ではHybridのshot節約が勝る。H6厳では純RTEの小bias・広headroomが大Bでも小さいNへつながる。prefix、tail、law、Gaussian基底、control、compilerへ一意な因果寄与率を配分した研究ではない。

Pauli純RTEは全非identity項をランダム化しconstantだけ保持する。DF B3はone-bodyを保持するため、L_D0というラベルだけで同方式として比較しない。Pauli内部Hybrid対純RTEと、Pauli対DF-nativeの表現・実装比較は別。旧H4共通H/rayやD1/D2の操作的対照はあるが、最新最良候補の完全matched比較や表現単独の因果効果ではない。Pauli単価・利得率をDFへ移送しない。[表現比較][SRC_PX]、[原因調査][SRC_MECH]

Pauliを独立の節／別論文候補にする価値は、同じ費用会計でも保持・全ランダム化の選択がDFと異なる有限観測にある。科学的に同じ一因子を変えた結果という物語にはしない。

## 4. Track B：構成・実装・native資源の成果

return集約、局所生成、finite-bit補正、既知normalizer／拒否／provider取得の会計には完成した構成成果がある。正有理p、0<x≤1、奇数m、Hermitian involution等の前提に限定し、既知Green/Euler/importance weightingを新発見としない。構成の独立新規性・一般量子優位は未確立。[利用可能性表][SRC_BSURVEY]、[v1.2 claim表][SRC_BLATEST]

固定3qubit syntheticのG9/G10ではnative T、CX、shotの交換がある。G9 Closed P5のT-primary利点と、matched CTSのCX利点を両方残す。D3では同finite targetの保証×価格4代替ledgerでr・方式の保存選択が変わるが、新しい普遍selectorを実証した結果ではない。native加法価格をwhole compiled RZへ換算しない。[G9][SRC_G9]、[D3][SRC_D3]

後続H4 C1は1Å/STO-3G/DF12/generation-prefix3、13basis/324labels、T4/5/q1/r4/m5でOrdinary／General Full／Closed P5各1024pathを完成。古い「syntheticのみ」は更新する。ただしTAIL_ONLY_PLUS_SCALARのlogical観測で、zero-path0、各pathに4 child rotationsが残る。共通認証N、deterministic halves、strict native全wrapper、同精度全task費用は未取得。一時凍結を保持し、本文ではDF資源比較の成功例として混ぜない。[C1][SRC_C1]

## 5. RPE：後続検証を反映した現在地

固定H4/1Å/STO-3G/legacy DF12、Partial L_D3対Full12、δ=.02、18round、q_max131072、元最終r/K/N/α/β、状態準備なしという条件を保つ。

| 後続 | 現在保持する成果 | 残る境界 |
|---|---|---|
| V1 | 両候補全18roundの共通E0に対するPF位相残差は保存予算内 | PF有効固有状態と元H状態は別。厳密単位半径未保証、Full半径7点の数値不一致を保持 [V1][V1] |
| V2 | 保存平均・IID・alias条件下で分枝、energy精度、成功の十分事象を評価し両候補CONDITIONAL_SUCCESS | 実機保証、厳密joint decoder失敗率、状態準備保証ではない [V2][V2] |
| V3 | 直接価格とモデル単価の72軸対応、A0 r32限定を整理 | 最終Partial直接q1/2/8、Full q1/2/16/32。q>32価格は外挿 [V3][V3] |
| R1 | 元費用を完全再現し、Partial約4.858%点選好を維持。共分散・systematicの感度を評価済み | 総合UNDETERMINED。R2/R3は順位決定目的では非推奨 [R1][R1] |

したがって「全RPEは未検証」という表現にも、「同energy taskで最終資源優位を実証した」という表現にもしない。条件付き成功の数値支持と、資源差の不確かさ未解決を同じ節で示す。今回R1の平均・SE・反転閾値を再計算していない。

## 6. 先行研究と寄与候補

旧稿・PM対応表の先行研究整理を再利用した。Güntherらのv2はpartial randomizationとsingle-ancilla phase estimationに加え、Sec.VII.BでDFの部分保持、Appendix D.2で切断との比較を扱う。該当箇所を一次資料で限定確認したため、これらの一般原理を本研究の新規性にはしない。[既存対応表][PMMAP]、[Güntherら・固定v2](https://arxiv.org/pdf/2503.05647v2)

Hagan–Wiebeのhybrid分割、CasaresらSPRINT/GRADE、OumarouらRC-DF、Kanasugiらの化学QPE end-to-end評価、Cuginiらの費用とvarianceを共同で扱うimportance samplingも旧稿の既知法として保持する。今回は既存引用の公式metadata/abstractと上記限定箇所を確認した範囲で、網羅的priority監査や新しい独立性レビューではない。[旧稿References][OLDMAIN]

寄与として検討できるのは、**強い決定論対照の下での実装資源台帳、精度・時間・入力域による正負の適用地図、finite-RTE bias／normalization／N／whole Cの具体的競合、条件付きRPE成功と価格不確かさの接続**である。これが独立論文として十分か、technical/resource noteが適切かは未判断。「初めて」「世界初」「resource-optimal」の語は採用しない。

## 7. 旧原稿から何を保持し、何を更新するか

v0.1は2026-10-05のH4 DF12・固定T・二次PF・登録218／固定5構成の事例研究。Methodsのtask/十分N/paired価格/費用scope、C1–C6の限定、4主図＋補足は文章・構成資産として再利用できる。

後続のB0I、多分子・高次・長時間は**別条件の新節**にする。旧Abstractへ後続値を差し込んで、当時既に検証していたようには書かない。旧稿が比較対象外としていた高次・RPEの一部は現在完成したが、旧条件での結論を取り消す理由でも、旧稿全体を無条件に拡張する理由でもない。旧図のcohort・nominee・価格版を最新結果へラベルだけ差し替えない。[旧audit][OLDAUDIT]、[構成差分表](PAPER_STORYLINE_OUTLINE.md)

## 8. 調査範囲・取得限界と次の科学判断

LOCALは取りまとめ担当であり、証拠対象をLOCAL実験に限定していない。Phase0–3の共通索引・3環境統合とG01/G02、各研究のGitHub固定成果を優先し、70カード／SOURCE索引と既存監査を再利用した。[固定Phase0仕様][PROTOCOL]、[LOCAL棚卸し][P1LOCAL]、[SERVER1棚卸し][P1S1]、[SERVER2棚卸し][P1S2]、[Phase2統合][P2REPORT]、[Phase3][P3REPORT]を入口にした。Phase0の実担当はSERVER2というユーザー訂正を維持し、固定PROTOCOLのLOCAL担当記載は変更していない。固定Git objectの取得・照合は非公開rawや他host原本の再現性保証ではない。

重要な不足資料については既存server inventoryも候補所在の確認に用いた。H8詳細provenanceの候補refはGitHubで404、LOCAL objectにもなく、2026-11-02のSERVER2 inventoryに対応候補がある。これは現存を直接確認した意味ではない。現在の保存点比較には公開主表を用い、詳細auditまで確認したとは書かない。より強いH8監査・独立性claimを選ぶ場合だけ、該当SERVER2候補の限定read-only照合と匿名projection公開を検討する。SERVER1を含む形式的な全再棚卸しは不要。[取得・未取得の整理](STATUS_AND_OPEN_QUESTIONS.md)

現在の限定claimに不可欠な公開一次結果は取得できた。資料の欠落、科学的未保証、強める場合だけ必要な追加検証を区別し、RPE R2/R3・DF B3・H8残余samplingを一律再開しない。原SOURCE・結果・契約・STOP、旧原稿・図・Phase3表は変更していない。

ユーザーが決めるのは、DF中心の範囲、Pauliを別稿にするか、RPEを補足に置くか、Track Bを構成noteとして分けるか、現在の限定証拠でどの主張を正式採択するかである。新しい完成稿・詳細科学レビュー・投稿は別指示の対象。**本統合成果を公開・remote本文照合後mandatory STOP。**

<!-- fixed-reference-definitions -->
[G01]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G01_CANDIDATE_BINDINGS.csv>
[G02]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G02_COHORT_BINDINGS.csv>
[GREPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/REPORT.md>
[H468SYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h468_paper_evidence_integration_20261019.md>
[LANDMATRIX]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md>
[LANDSCAPE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md>
[OLDAUDIT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_claim_audit_v0_1.md>
[OLDMAIN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_v0_1.md>
[P1LOCAL]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/17aacf67ee854657652ade8035c5fdfad1eb077a/docs/research/pr_cross_host_coverage_audit_20261102/inventory/LOCAL/REPORT.md>
[P1S1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de84ed7af00e73339886429aabab0bf8180e9f0/docs/research/pr_cross_host_coverage_audit_20261102/inventory/SERVER_1/REPORT.md>
[P1S2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8834de0cf938b13490c08e05006e3c566fed7c7a/docs/research/pr_cross_host_coverage_audit_20261102/inventory/SERVER_2/REPORT.md>
[P2REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/604f0235840feb18d368f99ea0828f370ce70fc8/docs/research/pr_cross_host_coverage_audit_20261102/integration/REPORT.md>
[P3REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md>
[PAULICLAIMS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_claim_figure_map_20261101.md>
[PAULISYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md>
[PMMAP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_post_pm2_claim_evidence_map.md>
[PROTOCOL]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md>
[R1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fde197c64da671f1d428f0d3e8fdb762dc21930/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md>
[SRC_B0I]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md>
[SRC_B3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md>
[SRC_BLATEST]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/CLAIM_EVIDENCE_MATRIX.md>
[SRC_BSURVEY]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/results_usability_map.md>
[SRC_C1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md>
[SRC_COEFF]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/pf_c_system_size_validation.md>
[SRC_COST]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/rte_compiled_cost_validation_summary.md>
[SRC_D3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md>
[SRC_FINITE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/finite_rte_signal_validation.md>
[SRC_G9]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md>
[SRC_H6STRONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md>
[SRC_H8STRONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md>
[SRC_HIGHCONF]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_reconstructed_confirmation_results_20261022.md>
[SRC_LONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md>
[SRC_M2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md>
[SRC_MECH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5432857b94c3714ea9f4d6031e493e8dfc308a91/docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md>
[SRC_PF588]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md>
[SRC_PX]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_comparison_synthesis_20261025.md>
[V1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md>
[V2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md>
[V3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md>
