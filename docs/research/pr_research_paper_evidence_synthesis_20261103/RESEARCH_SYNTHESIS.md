# PR研究の成果・主張・証拠と論文化候補

**2026-11-05更新**：[後続結果・資源機構・派生成果](#integration-20261105)と[必要十分性A/B/C/D表](STATUS_AND_OPEN_QUESTIONS.md)を追記した。現在の限定claimの記述的判断は `CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT`。以下の11月3日原統合は時点を保って読む。

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

<a id="integration-20261105"></a>

## 2026-11-05追補：検証を一区切りにできる範囲

**記述的判断：`CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT`。** 固定Hamiltonian・保存候補域・解析的shot会計・固定compilerの取得価格について、有限標本／モデル条件付きのPR資源比較を述べる範囲では、今回照合した資料に主要結論を変える具体的な証拠欠落・入力不整合は見つからなかった。追加科学計算を現在の完了条件にしない。これは主要claimの正式採択・独立科学レビュー・既存正式statusの置換ではない。

現在の範囲は、同じ有限時間signal推定課題での **G=N_cos C_cos+N_sin C_sin** の保存比較である。Nは解析上の十分量子shot数、Cは測定付きfull Hadamard wrapperのcompiled RZ、nは古典価格trajectory標本数。状態準備P=0。保存bias/reference・数値allowance、law、compiler残差の仮定を各原結果のまま付す。「要求signal精度を実compiled/noisy実装で無条件に達成」「PR母平均がPFに勝つ」「全PF候補の最適性」「同energy taskの頑健な総費用優位」へは拡張しない。

本書§1–8と旧対応表の原欄は11月3日時点の統合として保持する。以下はその後続であり、旧H8 T=.8・DF12/DF10・RPE V2等を上書きしない。最新対応表には候補取得範囲、価格・compiler、入力/SOURCE、確認水準、支持区分、必要な残作業を追記した。[最新CSV](CLAIM_EVIDENCE_MAP.csv)、[必要十分性表](STATUS_AND_OPEN_QUESTIONS.md)

### 1. H8 Stage A/BとStage Cを分けた最新結果

Linear H8／1.00 Å／STO-3G／保存tol-only1e-8 DF28、16 system qubit・保存spin sector4900、同state・fragment順0..27。T=1.6、signal ε=.0005、P=0。PRは4th(m5_best)/prefix8/q1/R_outer32/K4、固定PFはB0I/4th(m5_best)/prefix15/q1。compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/topologyなし。[探索][H8_AB]、[確認][H8_C]

| 保存cohort | 取得・固定の範囲 | 引用結果と確認水準 |
|---|---|---|
| Stage A/B EXPLORATION | signal69候補、適格40・不適格29。PR4候補各n9、PF15候補各1回、51trajectory/102価格 | m5/p8/q1探索平均G=67323362591920。PF15候補内最小は上記B0I。探索19代表133検査PASS。最小という語は取得集合内だけ |
| Stage C CONFIRMATION | 固定PR1候補、別seed namespace n9/18価格。固定PFはStage B由来の価格を引用しEXPLORATION cohortを維持 | 平均G=67335317264200、固定PF G=89291219516568、PR/PF=0.754109056061、観測削減24.58909439%。記述的SE=103180211675.02368、固定h_G=121655623062.28098に対する保存判定達成。代表index0の7検査PASS |

PRの固定N_cos/N_sin=84916951/88339169、PFは71893270/73757762、PF両軸C=613049 RZ。探索平均・確認平均・比・SEはすべて既存表示の転記で、再計算していない。確認後の候補選抜、探索とのpool、新しい信頼区間はない。[原確認JSON][H8_C_RAW]、[paired価格CSV][H8_C_PRICES]

確認でもorder0=288、order2/4は未観測。未観測群の条件付き価格・population tail boundはnullのまま。小さいsample SEと独立seed namespaceは、PRNGの数学的IID性や母平均精度・母平均優位を保証しない。追加sampling/compile・希少費用研究は現在の限定観測の完了条件でない。位相補助は実小gate operator由来のfull-wrapper global phase会計であり、statevectorの位相整列ではない。元HOLD/失敗を保全する。[確認と限界][H8_C]

結果commitと科学SOURCEは別。Stage C実行SOURCE=82fdbdd098e9d3edae01a930437e3d76c1f8bf36、signal=d1272016592ff06c0567862f770ed796fd61b366、native=df93fbb88761b61c2691bf01f761ed20a7987d7f、位相補助=18ce41942051065fa097644a8a937bd11adba708。原JSONのH/state/DF binding、candidate ID、source_files hashを固定Git objectへ照合する。private raw/QPYの外部再現まで認定しない。

### 2. 資源競合：何を説明できるか

以下は旧長時間研究の保存比・保存価格を引用する。新しい比・順位・統計量・scaling則は作らない。H4/H6長時間はlinear1.00 Å/STO-3G/tol-only1e-8 DF10/19、H6は保存weighted-Hermitian completion込み。Qiskit1.3.0/opt1/seed17/rz-sx-x-cx、symmetric-directional制御・scalar一回・P0・full wrapper。R_outerは1 outer内の総short-step数で、総event数q R_outer。δは原表のT/q。[元条件・結果][SRC_LONG]

| 要因・保存対照 | 直接支持される説明 | 解釈・未分離の境界 |
|---|---|---|
| 次数×時間×精度：H4 PF、m5/S2保存比 | T=.8/ε=.05ではq1→1、prefix5→7、N比0.7878447452676515、C_eff比14.463692453725676、G比11.395144096835159。shot減を単価増が上回る。T=1.6/ε=.0005ではq32→1、同prefix7、N比0.9927333346654413、C_eff比0.34489002563333576、G比0.34238382523983096 | C_effは元表の実効単価で、片軸Cやdet/random/basis別価格ではない。各公式の取得域内選択を比較し、q単独の因果効果ではない [保存分解][CNG] |
| 同じH4のPR m5/S2 | T=.8/ε=.0005でq8→1、prefix1→3、N比0.4937986570180657、C_eff比2.0302213375166436、G比1.0025205699151396。T=1.6ではq32→1、prefix1→3、N比0.7104277461318325、C_eff比0.5461571223523415、G比0.3880051734666214 | q減少だけでは高次PRの利益を決めない。厳精度・長時間の取得条件ではN/Cの交換が改善するが、一般漸近優位ではない [分解][CNG] |
| matched q/prefix/R/K：H4 T=.8/ε=.05、p3/q1/R32/K4 | S2: log_B=0.006803284160668892、N=11197/7515、G=125783311.46666667。m5: log_B=0.016709032397037187、N=7250/7250、G=526014303.030303。qとprefixを揃えても高次の費用は増え得る | 次数がsigned piece・law・bias・B・compiler入力も変える。normalization単独の寄与率ではない [matched保存値][MATCHED] |
| r/R_outer・K：同H4/p3/q1/m5/ε=.05 | R16 K2/4/6で保存Nは各7529/7529、C_cos=C_sinは34312.11111111111／34223.666666666664／34243.0。R32 K4ではN=7250/7250、C=36276.84848484849。R増によるshot節約と価格負担を別欄で確認できる | R32 K2/K6やR64の一部は適格でもNOT_ACQUIRED。空欄Gを0にしない。K間の有限標本差から母平均順位や単調則を作らない [感度原表][RK] |
| partition：B0/B0I/B2/B3 | B0Iはdiscard tailの既知identityを一回戻す強い対照。H6 T=.8/ε=.05の実測p7で旧PR点より低費用。長時間DF B3は取得40条件でB2・決定論envelopeより低G0/40 | prefix増はtail負担と保持回路を同時に変える。DF B3はone-body保持で、全Hamiltonian純RTEではない。全ランダム化一般の不可能性ではない [H6][SRC_H6STRONG]、[B3][SRC_B3] |
| 表現・basis/control | Pauli内部ではH2/H4/H6の保存比較14/15で純RTE低G、H4粗精度ではHybrid低G。低Cが大Nを補う場合と、bias/headroomによるN減が効く場合がある | DFとPauliで保持配分の観測が違う。law、Hamiltonian/ray、basis、control、候補域も違い、表現単独の因果効果ではない。D1/D2対照と最新family比較を分ける [既存原因整理][SRC_MECH]、[表現対照][SRC_PX] |

有限RTE biasと正normalization/attenuationは別。補正後signalを推定するNには元のrange/B²会計とbias後headroomを使い、raw meanを補正signalに読み替えない。R/K変更はbias・B・N・Cを一緒に動かす。これらは元SOURCEの会計上の直接関係で、個別機構の因果寄与率の新推定ではない。

粗精度でS2が両系・全Tの取得域内点最良だった結果、長時間・厳精度で高次が低Gだった結果、確認で探索方向が反転したH4 T=1.6/ε=.005も併記する。H8 T=.8の強化184域とT=1.6の固定確認は別条件。保存されていない時間・精度のscaling則、未使用入力のselectorは導出しない。[長時間原報告][SRC_LONG]、[H8探索][H8_AB]

説明水準は、**matched対照で直接支持／複数観測と整合する解釈／未分離の因果／新検証を要する仮説**を区別する。上表のN/C交換は前二者。q・表現・basis/control各々の一意寄与、未使用入力へ移る選択則は後二者であり、今回の自動レビュー・計算対象ではない。

### 3. 本体から分けて活用できる成果

| 派生成果 | 既存成果・再利用できる部分 | 新規性と将来利用の境界 |
|---|---|---|
| Policy-aware平均certificate | 保存component certificateをrun≤1 policy、order/role、source-basis隣接確率で重み付けする期待telescoping。固定ideal IID平均の全18round/36軸局所十分条件が成立。m17追加上端の保存上向き表示7.55735552e-9 | telescoping・三角不等式・期待値の線形性自体は既知。今回の具体的な追加部分はpolicy依存のsingleton端/内部確率とproduct/rotation係数を結んだSOURCE固有の閉形式・認証接続。独立な定理的新規性・priorityは未判定。適用仮定を満たす別basis簡略化・回路近似の平均誤差と価格の共同設計へ利用候補となるが、別policy/lawへ自動適用しない [導出§5][MEAN_ANALYSIS]、[結果][MEAN_REPORT] |
| 固定H4条件付きRPE保証 | 同一legacy DF12/L_D3/δ=.02/K2/N4 sector70の保存18round/36軸について、指定ideal controlled IID測定＋数学的decoder＋外部alias条件で失敗≤0.0496929886343339467、成功≥0.950307011365666053、energy誤差≤0.000138689200360119854 Ha | 一般RPE成功理論を新発見としない。保存matrix/mean包含、確率tail、矩形幾何、strict branch、reference込みenergy接続を固定入力で数値認証した成果。成功頻度の実測でも真の分子energy保証でもない。新入力での再利用には再bindingと別科学実行許可が必要 [保証報告][ENERGY_REPORT]、[SOURCE証明][ENERGY_PROOF] |
| PF次数×partition共同設計 | 上記C/N/Gとmatched対照、DF B0I/B2/B3、H8高次固定確認により、次数だけを選んでも資源順位は定まらないことを固定域で説明できる | 適用条件の分類・未使用入力の選択則を考える材料。独立なselectorや移送保証は未実証。新しい選択・再ランキングは行わない |
| DF／Pauli-JW | 同じG会計でも保持・全ランダム化の有利な配分が異なるという有限観測、既存D1/D2の操作的対照 | 表現・law・control・basis・候補域の交絡を残す。独立研究テーマの候補として価値を整理できるが、表現単独の因果効果・独立論文成立は未判定 [既存統合][PAULISYNTH] |
| Track B構成・native会計 | return集約・局所生成・finite-bit補正、matched CTS、G9/D3のnative T/CX/shot交換、H4 C1各1024pathのtail-only logical接続 | DF本体のRZ×shotと別系列に置く。既知Green/Euler/importance weightingとSOURCE固有の構成を区別する。独立新規性・共通認証N付き分子full-wrapper優位は未確立。一時凍結を維持 [構成claim表][SRC_BLATEST]、[C1限界][SRC_C1] |

policy-aware保証は単一event平均の積への誤因数分解や符号相殺を使わず、word依存の実装差の加法上端を期待化する。旧任意trajectory一様上界18/36未判定を、新平均上界で上書きしない。平均対象の違いが核心で、全trajectory最悪保証を完成条件に戻さない。[平均結果][MEAN_REPORT]

H4のenergy対象は保存DF exact polynomialの4電子sector最小固有値。各軸内fresh独立uniform53 trajectoryとideal±1測定を仮定し、round/axis間独立は不要。数学的decoderと外部alias/liftを使う。固定PCG64の数学的IID性、Float64 decoder、compiled/noisy全回路、実状態準備、上流DF誤差・真分子energy、実装全体の成功は保証していない。正式`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持する。[確率・energy原結果][ENERGY_RAW]

RPE V3/R1のPartial約4.858%低い点費用は旧opt2モデルの結果で、共有較正共分散・長q systematicを含む総合順位は引き続きUNDETERMINED。今回の条件付き95%成功保証をFullとのRPE費用優位に移さない。測定backend契約とFloat64追補は技術準備・限定証明の成果であり、production接続・停止済みkernel証明を自動再開しない。[R1][R1]、[backend判断][BACKEND]、[Float64追補][FLOAT64]

### 4. 終了判断と保全

現在の限定的な保存比較の必要証拠は揃い、A/B/C/D表のBに該当する具体的不足は今回見つからなかった。母平均優位・全PF最適性・無条件実装保証を中心claimに選べば必要証拠は変わるが、それを今回の主要claimに正式採用していない。新規性の独立性や論文成立性も未判定のまま残す。

今回変更したのは統合文書、既存対応CSV・文書用metadata checkerと入口・当日ノートだけ。SOURCE/raw/既存manifestの科学status/grant/ledger/marker/STOPを変更せず、新しい科学runnerをimport・起動しない。全repository再棚卸し・新規証拠計算・詳細科学レビューは行わない。**公開SHAと変更公開fileのblob/bytes照合後mandatory STOP。Codexに追加科学作業はない。ユーザーによる主要claimの正式採否は別判断であり、今回の終了の前提にはしない。**


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

[H8_AB]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84/docs/research/h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md>

[H8_C]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md>

[H8_C_RAW]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/confirmation.json>

[H8_C_PRICES]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/paired_prices.csv>

[CNG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/C_N_G_decomposition.csv>

[MATCHED]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/same_q_prefix_R32_K4.csv>

[RK]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/r_K_sensitivity.csv>

[MEAN_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1789a8ca731a1de631e80a34d2b8c90657d24458/docs/research/rpe_support_weighted_transfer_execution_20261105/REPORT.md>

[MEAN_ANALYSIS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a39a78eccdbe457786978e14d36b2f0d7abc290/docs/research/rpe_support_bound_sharpness_20261105/SUPPORT_BOUND_ANALYSIS.md>

[ENERGY_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4c016960ff4b1ca16cc73770147cde326989427/docs/research/rpe_controlled_iid_success_energy_execution_20261105/REPORT.md>

[ENERGY_RAW]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4c016960ff4b1ca16cc73770147cde326989427/artifacts/rpe_controlled_iid_success_energy/2026-11-05/run01/RESULT.json>

[ENERGY_PROOF]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffa1e22b3001ad1c4eb275ff1121ed17111452ec/docs/research/rpe_controlled_iid_success_energy_preparation_20261105/CONNECTION_PROOF.md>

[BACKEND]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/655bf6cb84ab161706baf843a8ae255da39bf7a6/docs/research/rpe_measurement_backend_contract_20261105/NEXT_STAGE_DECISION.md>

[FLOAT64]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccbd7aa974eda1e017138066a70dc8ac35221907/docs/research/rpe_three_gaps_preparation_20261104/INPUT_KERNEL_ADDENDUM_20261105.md>
