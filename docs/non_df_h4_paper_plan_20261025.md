# H4 Pauli／DF比較研究：論文構成・主要図表・主張範囲の案

2026-10-25 JST。**現在の保存証拠から、固定H4・固定実装のempirical / conditional case studyのResults・Methods・主要captionを構成できる。** 本文の中心候補は、実装系列によって変わる回路価格、同じPauli bodyでの制御方式の効果、保存価格と残差guardを条件とする総資源比較である。PR母平均の費用tailと実compiled精度を含む無条件のG優位は、未保証として残す。

本資料は原稿の設計案であり、中心主張・投稿先・掲載可否の正式採択やGPT詳細レビューではない。**新しい科学計算・数値再解析・統計集計・再ランキング・作図を行っていない。** 図番号・優先順位・英語captionはすべて候補である。

基点：`codex/non-df-h4-comparison-synthesis-20261025`、`03efd057a7e77c40ae7e90b9c77885d6606e793b`。GitHub branch refとlocal HEADの一致を確認した。専用branch：`codex/non-df-h4-paper-plan-20261025`。

## 1. タイトルと論旨の候補

| 候補 | 仮タイトル | 強調する範囲 |
|---|---|---|
| A：条件付きcase study | **Circuit and Conditional Resource Costs of Partially Randomized Product Formulas: A Pauli–DF Case Study for H4** | 4方式、Pauliの2制御系列、精度会計までを限定付きで扱う |
| B：実装依存性 | **Control Synthesis Changes Pauli–DF Cost Comparisons in an H4 Product-Formula Case Study** | 同じPauli bodyのD2対照を前面に置く。表現一般の優劣は主張しない |
| C：観測と限界 | **Compiled Circuit Costs and Conditional Sampling Costs for Pauli and Double-Factorized H4 Simulation** | 費用指標と条件をタイトルで明示する |

日本語作業題：「固定H4におけるPauli-JW／DF-nativeのPF・部分ランダム化PF：回路実測と条件付き総資源比較」。いずれも未採択。新規性・先行研究との位置付け・投稿方針の最終判断はユーザーへ返す。

導入の問いは「同じ分子入力でも、分割・event粒度・制御合成を含む具体的実装が変わると、PF/PRの回路単価と条件付き総費用の比較はどう変わるか」とする。『Pauli表現そのものがDFより優れる』を問いへの既成の答えにしない。

## 2. 章構成案と各章の証拠

本文は以下の9章相当を候補とする。Methodsを後置する形式でも、条件表とGの条件はResultsの前に提示する。

| 章 | 内容・書くべき結論の範囲 | 既存証拠 | 図表候補 |
|---|---|---|---|
| 1 Introduction | PRの保持部／ランダム部と、解析的費用モデルとcompiled費用の隔たり。固定H4の検証課題を提示 | [PR原論文v2](https://arxiv.org/abs/2503.05647v2)、[構造解析](non_df_h4_cost_reversal_structure_20261024.md) | Table 1を予告 |
| 2 Implementations and scope | 同じ積分・ray、異なるPauli/DF分割・law。generic Pauliとcentral Pauliを別系列にする | [統合報告§1–2](non_df_h4_comparison_synthesis_20261025.md)、入力receipt、C model、D2 SOURCE | Table 1 |
| 3 Resource evaluation | full-wrapper RZ/shot、補助CX/depth。価格MC256と量子shot Nを分離。bias・normalization・未知ρ・guardを定義 | [条件付き会計](non_df_h4_central_conditional_cng_20261024.md)、保存conditional_functions/axis_results | Methods、Fig. 3 |
| 4 Circuit prices | generic PauliではPFはDFが低単価、PRはPauliが低単価。central系列では対応するPF/PRともPauliが低い、という保存値 | cost_summary、A_AXES statistics、6系列引用CSV | Fig. 1 |
| 5 Effect of control synthesis | 同じbodyでPF255500→15028、PR0 9652→1145 RZ。保存256本のpaired価格比較を併記 | [D2](non_df_h4_d2_execution_20261024.md)、A_COHORT_COS statistics/既存図 | Fig. 2、Table S1 |
| 6 Structural diagnostics | Z/ZZ保持、event粒度、DF Gaussian basis/reuse。D1 SOURCE基準Bの9測定と非加法性。因果寄与率は未識別 | [D1 SOURCE-B](non_df_h4_d1_source_execution_20261024.md)、metrics/interaction CSV、構造解析 | Fig. 4またはTable 2 |
| 7 Conditional total resources | 5精度の仮想ρ=0と独立guard全域の保存比較。単価改善をshot増と合わせて評価 | central conditional_results/comparison_regions、旧generic/DFの保存会計 | Fig. 3、Table S2 |
| 8 Uncertainty and limitations | 未観測order4・DF component・有限標本。6局所Arbと全law保証の区別。表現間交絡・target scope・旧来歴 | rare_cost_sensitivity、A_AXES、six_local_results、[R2-Cの保存判定](non_df_h4_r2c_compiler_phase_proof_20261025.md) | Fig. S1、Table 3、Table S3 |
| 9 Discussion and conclusion | 固定実装の観測と条件付きGをまとめ、他分子・無条件母平均・実精度へ外挿しない | 上記証拠のみ | 追加数値なし |

補足資料は、(S1)全9構成とD2全5指標、(S2)5精度の既存N/G/guard行、(S3)局所Arbと再現性の来歴、(S4)旧generic会計の感度断面、とする。長い監査手順を本文の主題にはしないが、失敗履歴や未保証状態を隠さない。

## 3. 主要図表の優先順位と出典

**最優先はFig. 1（回路価格）、Fig. 2（同じbodyの制御方式）、Fig. 3（条件付きG）、Table 1（条件と制約）**。Fig. 4/Table 2は機構説明を支える。分布・希少event・6局所保証は本文の限界を明記したうえで補足へ置ける。

ここでの出典IDは[図表計画JSON](../artifacts/non_df_h4_paper_plan/2026-10-25/figure_table_plan.json)の`evidence`に対応する。そこにrepo相対path・JSON pointer・CSV selector・元科学SOURCEを記録した。既存PNGを実際に目視し、SVG/PDFの存在を確認した。図の原本は変更していない。

### Fig. 1／P1：4方式とPauli制御系列の回路価格

- **何を示すか**：Pauli generic／Pauli central／DF nativeの3系列×PF/PR。4方式の比較に、Pauliの実装差を明示する。
- **データ**：E1の元cost_summary、E2のA_AXES statistics。E0の`circuit_costs_QUOTED.csv`6行は元pointer付きの配置用引用表として再利用する。
- **軸・単位**：横軸は上記6カテゴリ、縦軸はcompiled RZ/shot（log）。PF singletonとPR n=256をラベルで分ける。PRのerror barには保存nominal t95半幅だけを使い、PFにMC CIは付けない。
- **伝える主張**：固定実装の回路価格は、表現名だけでなく制御方式に依存する。genericのPFとPRの比較関係はcentralへそのまま移せない。
- **再利用**：旧[4方式図](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_comparison.png)はgeneric系列の補足としてそのまま使用可能。ただし図中のPauli labelにgenericの明示がないためcaptionで必須。6系列の新しい配置案は**未描画**。既存mean/半幅の転記だけで構成でき、新集計を要しない。
- **注意**：状態準備なし、PF deterministic、PR finite cohort。cosine/sine価格の等値は保存cohortの観測だけ。名目CIは全law費用upperではない。

**Caption draft.** *Compiled full-wrapper circuit prices for fixed H4. Pauli-JW circuits are shown separately for generic whole-body control and central-rotation control; DF-native circuits use their saved native construction. PF prices are deterministic singleton counts, whereas PR prices are saved means over 256 trajectories. Error bars on PR means denote saved nominal Student-t 95% half-widths, not rigorous bounds on population costs or unseen events. Cosine and sine prices coincide in the saved cohort. State preparation is excluded, and the different splits, event laws and control constructions prevent a representation-only interpretation.*

配置時に使う値（元の6行を引用、再計算なし）：

| 実装系列 | PF RZ/shot | PR mean RZ/shot | PR nominal t95半幅 |
|---|---:|---:|---:|
| Pauli generic | 255500 | 9321.1015625 | 27.270074727002452 |
| Pauli central | 15028 | 1140.8046875 | 2.0873215844123516 |
| DF native | 135355 | 23678.44140625 | 92.00231643184831 |

### Fig. 2／P1：同じPauli bodyのcontrol対照

- **何を示すか**：A_COHORT_COSの保存3-panel図を再利用する。左はcentral PR価格分布、中央は同じIDのgenericに対する削減率分布、右は保存価格の比較。PF/PR0のD2対照はTable S1で根拠を明示する。
- **データ**：E3の[既存図](../artifacts/non_df_h4_a_cohort_cos/2026-10-24/run01/cost_comparison.png)・statistics、E4のD2 comparison CSV。
- **軸・単位**：左x=cosine RZ、y=保存trajectory数。中央x=同一ID generic比の削減率[%]、y=本数。右x=既存5カテゴリ、y=RZ/wrapper（log）。右panelはgeneric PFを含まないため、6系列のFig. 1の代用にはしない。
- **伝える主張**：制御方式の変更とそれへのcompiler応答が価格を大きく変える。同じbody/trajectoryの対照であって、Pauli対DFのpaired実験ではない。
- **再利用**：既存PDF/SVG/PNGを**新しいbin設定・平均・百分率の算出なし**で再利用できる。D2の未作成plotを新たに要求しない。
- **注意**：PR0は1代表、256平均と分離。control方式とcompiler応答の合成効果で、compiler pass単独の寄与率ではない。DF barは別lawのunpaired参照。

**Caption draft.** *Pauli control synthesis on saved bodies and trajectories. The saved panels show central-control PR prices, per-trajectory reductions relative to the same-ID generic circuits, and reference circuit prices. The D2 singleton contrasts give PF prices of 255500 versus 15028 RZ and PR-index0 prices of 9652 versus 1145 RZ for generic versus central control. The contrasts hold the body, trajectory, angles and compiler settings fixed; they measure the combined effect of control construction and compiler response. The DF reference cohort is unpaired. Histogram bins and summaries are reused from the saved analysis; they do not establish population-tail bounds.*

### Fig. 3／P1：要求信号精度ごとの条件付きG

- **何を示すか**：左6系列の仮想ρ=0、右central指定15比較の独立guard全域比。RZ単価の改善が保存shot会計を介してGへどう現れるか。
- **データ**：E5/E6/E7。[既存conditional_resources](../artifacts/non_df_h4_central_conditional_cng/2026-10-24/recovery02/run01/conditional_resources.png)を優先する。元10行のconditional_results、reused_reference_rows、15行のcomparison_regionsを参照。E0の30行引用CSVも再利用可能。
- **軸・単位**：x=ε_sig=.05/.01/.005/.001/.0005（既存図はカテゴリ等間隔、連続log精度軸ではない）。左y=G [RZ]、log。右y=無次元G比、1の水平参照線。右の範囲はCIではなく独立な方式・軸別ρ guardの端点。
- **伝える主張**：保存価格とguardの下で、central Pauli PF/PRは対応するDFより低G。PRのshot負担を含んでも保存比較では利益が残る。既存15比較を超えるpairや順位は追加しない。
- **再利用**：既存SVG/PDF/PNGをそのまま利用可能。図内の“actual rho unknown”は作成時の全law/会計接続状態を表す。後続の6局所Arbが得られた事実はcaptionとTable S3で補足し、実ρ=0や全law受理へ読み替えない。
- **注意**：PRは保存平均価格固定、ρ=0は仮想条件。各表現の保存reference/biasを使用。guard所属、PR母平均tail、共通厳密Hへの新しい精度転送は未保証。Gはenergy/RPE/end-to-end費用ではない。

**Caption draft.** *Conditional total RZ resources at five saved signal tolerances. Left: saved prices combined with the saved shot sufficient condition on the hypothetical zero-implementation-residual slice. Right: the three specified central-control comparisons at each tolerance, with ranges over independent method- and axis-specific residual guard boxes. These ranges are deterministic sensitivity ranges, not statistical confidence intervals. All comparisons keep the saved circuit-price means fixed. Selected later Arb enclosures cover the central PF singleton and PR indices 0 and 1 on both axes; they do not establish PR-law residuals or authorize replacing the hypothetical slice by actual residuals. Population-cost tails and an unconditional end-to-end advantage remain unresolved. Lines connect saved conditions for display only.*

### Fig. 4／P2、またはTable 2：D1隔離費用と非加法性

- **何を示すか**：SOURCE基準Bで完了した9構成。full／retained-only／event-only／emptyを横並び、別panelに保存符号付きΔ。ページ制約なら下の表とΔの2行で十分。
- **データ**：E8 metrics_table.csvの9行、E9 interaction_table.csvのRZ/CX4行。Δは保存fieldを転記し、再減算しない。
- **軸・単位**：価格panelのx=表現×構成、y=RZまたはCX数。Δ panelはx=表現×指標、y=保存Δ[count]、0線と符号を表示。depthは表に示すだけで加法分解しない。
- **伝える主張**：generic Pauliの選択PR0では保持/event双方のRZがDFより小さく、DFの正の非加法性が大きい。全価格を成分へ一意に配賦したわけではない。
- **再利用**：該当する既存D1画像は対象ディレクトリにない。CSVからの表組み案のみ作成し、新しいplotは作らない。
- **注意**：初回D1のhistorical fingerprint FAILと後続SOURCE-Bの9/9成功を両記。single index0/generic限定、ablationは有効なPF/PR候補ではない。**加算stack、因果的な百分率、basis単独やcompiler passへのΔ帰属を避ける。**

| SOURCE-B構成 | RZ | CX | depth |
|---|---:|---:|---:|
| Pauli PF full | 255500 | 164096 | 358563 |
| Pauli PR0 full | 9652 | 6872 | 12973 |
| DF PF full | 135355 | 78628 | 134134 |
| DF PR0 full | 23021 | 11134 | 18306 |
| Pauli PR retained | 6524 | 4896 | 8803 |
| Pauli PR event | 3110 | 1976 | 4200 |
| DF PR retained | 13483 | 8100 | 13990 |
| DF PR event | 5773 | 1932 | 2825 |
| Empty wrapper | 0 | 0 | 1 |

保存Δ=full−retained−event+empty：Pauli RZ +18/CX 0、DF RZ +3765/CX +1102。Emptyの0とPauli ΔCX=0は保存された測定／診断値であり、未観測費用nullの0埋めではない。Event-only CXはPauli1976/DF1932であり、全指標でPauliが低いとは述べない。

**Caption draft.** *Operational D1 ablations under the later SOURCE-based criterion B. Nine configurations were measured, including the four full circuits, retained-only and event-only PR-index0 circuits, and an empty wrapper. The separately displayed saved interaction is Δ=C_full−C_retained−C_event+C_empty. It measures nonadditivity in the full construction and compilation context, rather than a uniquely attributable basis or compiler-pass cost. These ablations are diagnostics, not valid alternative simulation algorithms. The initial Pauli-PR historical fingerprint failure remains unresolved; the successful later criterion-B run does not recover the missing old program. Pauli values in this diagnostic use generic control.*

### Fig. S1／P2：保存PR価格分布と両軸の対応

- **何を示すか**：E10の[paired_costs](../artifacts/non_df_h4_a_axes/2026-10-24/run01/paired_costs.png)を再利用。両軸の価格分布、same-ID差、散布図の3panel。
- **軸・単位**：第1panel x=RZ、y=本数。第2 x=sine−cosine RZ、y=本数。第3 x=cosine RZ、y=sine RZ。
- **主張・注意**：保存256では全価格が両軸同値。cohort外の測定軸等値定理、tail不在、母平均CIの保証ではない。新しいhistogram/KDE/quantileは計算しない。

**Caption draft.** *Saved cosine–sine price pairing for 256 central-control Pauli PR trajectories. The histogram, paired differences and scatter plot are reused without rebinning or recomputing summaries. The observed axis equality and covariance describe this cohort only; they do not bound unseen-event costs or establish equality for the entire event law.*

### Table 1／P1：科学条件と比較上の制約

4方式を列、以下を行にする。単位は幾何Å、時間は保存Hamiltonianに対するatomic units、費用はRZ/shot。主要条件とcontrol familyを表内に置き、長いhashはMethods/SIへ分ける。

| 条件 | Pauli PF | Pauli PR | DF PF | DF PR |
|---|---|---|---|---|
| 共通物理条件 | linear H4、隣接1.00 Å、STO-3G、charge0/singlet、system8＋ancilla8 | 同左 | 同左 | 同左 |
| 決定論的分割 | 全184非identity項＋constant | prefix27 Z/ZZ＋constant | one-body＋全10fragment＋scalar | one-body＋fragment0、tail identityをphaseへ |
| prefix規則 | 全保持 | abs(c)降順50%累積 | 全保持 | abs(λ)‖g‖F²降順50%、one-body別枠 |
| 時間発展 | S2、T=.8、q8 | S2、T=.8、q8、r4、K4 | S2、T=.8、q8 | S2、T=.8、q8、r4、K4 |
| PR law | なし | finite paired Taylor、k=0/2/4、tail Pauli | なし | finite paired Taylor、k=0/2/4、rotated Z/ZZ |
| 制御実装 | genericとcentralを別表示 | genericとcentralを別表示 | native、basis非制御 | native、occurrence内basis reuse |
| 単価標本 | singleton | 保存256本 | singleton | 別lawの保存256本 |
| 共通compiler | Qiskit1.3.0／rz,sx,x,cx／opt1／seed17／backend・coupling・layout・routing指定なし | 同左 | 同左 | 同左 |

E11入力receipt/C models、E12の構造SOURCE、E4 D2、E1/E2が出典。DF生成はtol-only1e-8/rank10、同じMO積分・同じ保存raw ray。prefix27対1、同じ50%基準は同一の物理分割を意味しない。state準備・fault-tolerant synthesis・RPE全体は費用外。

**Caption draft.** *Fixed scientific and implementation conditions. Pauli and DF share the saved molecular integrals and raw state but use different split units, ranking weights, event laws and control constructions. Prefix sizes are therefore not matched physical retention fractions. DF uses the saved tol-only factorization with tolerance 10⁻⁸ and actual rank 10. Counts exclude state preparation and hardware mapping.*

### Table 3／P2：高次RTE eventと未観測費用

**図より表を優先する。** 未観測order4を高さ0の費用barにしない。E13の保存確率と観測数をそのまま示す。

| 保存項目 | Pauli PR | DF PR |
|---|---:|---:|
| nominal per-event p(k=2) | .003934819613987418 | .002549087577779853 |
| nominal per-event p(k=4) | 2.5990406907583435e-6 | 1.0880131215414273e-6 |
| 保存8192 eventのk=0/2/4 | 8157/35/0 | 8167/25/0 |
| implemented lawでany-order4 trajectoryの保存質量 | 8.316595170215553e-5 | 3.481583274276441e-5 |
| order4 trajectory観測 | 0/256 | 0/256 |
| order4条件付き費用／母平均費用upper | unknown/null | unknown/null |
| 未観測component集合U | 空 | 保存164 component |
| no4-with-U trajectoryの保存質量 | 0 | .1001190406163259 |

Nominal per-event確率とimplemented finite-gridのtrajectory質量を同じ列の意味にしない。Uは同じ標本から選ばれた記述的集合で、独立coverage検定ではない。原論文の偶数高次eventも費用に寄与し得る。本研究のK4 lawを、無限lawや論文の解析的Toffoliモデルと同一視しない。

**Caption draft.** *Rare-event evidence and missing population-cost information. Nominal per-event probabilities and saved implemented trajectory masses are reported separately. No order4 trajectory was observed in either 256-trajectory cohort, so its conditional cost is unknown rather than zero. The unseen DF component set was selected from the same data and is descriptive, not an independent coverage test. Nominal intervals for observed prices do not certify population means, and the original paper's analytical cost model is distinct from these compiled full-wrapper RZ counts.*

### Table S3／P2：6局所Arb保証の被覆範囲

E14の`rows`から`key`、`rho_upper_decimal_outward`、`rho_budget_decimal_outward`、`precision_bits`、`relation`を転記する。数値の丸め直し・再縮約は行わない。掲載用6行は[既存一覧](non_df_h4_comparison_synthesis_20261025.md#7-6局所arb保証の適用範囲)を再利用できる。

**Caption draft.** *Six saved local Arb residual enclosures for central-control Pauli circuits. Both axes of the PF singleton and selected PR trajectories 0 and 1 have certified numerical upper endpoints within their saved local budgets, all at 64-bit precision. PR1 includes an order2 event. These enclosures apply to the fixed saved ray, selected C-targets and compiled QPYs with the original phase and normalization conventions. They do not cover the whole PR law, unseen order4 events or DF circuits, and formal certificate acceptance remains false.*

### Table S1／P2：D2の全5指標

E4の10行をそのまま、PF/PR0×RZ/CX/depth/rz_depth/cx_depth、generic・central・保存差・保存既約比の列に配置する。count/depthは無次元の回路指標で、時間単位ではない。PR0とcohort平均を混ぜず、D1 full対照は再compileされた新対照ではなく保存値の再利用とする。表はCSV転記で構成でき、差や比の新計算は不要。

**Caption draft.** *All five saved D2 circuit metrics for the PF singleton and PR-index0 body. Generic D1 reference values, central D2 values, signed differences and exact ratios are quoted from the existing comparison table. These are same-body control-construction contrasts under the fixed compiler, not cohort averages, execution times or a decomposition into individual compiler-pass effects.*

### Table S2／P2：5精度の条件付きN/Gとguard

E0の30行を実装系列×方式×精度で配置し、N_cos/N_sin/N_total、G/G_exact、PRの保存SE・nominal半幅を表示する。Gの単位はRZ。E6の15比較にあるguard範囲は別blockに置き、同じerror barへ混ぜない。actual_N/G/rhoは未確定のまま、表示行はhypothetical ρ=0と明記する。保存精度値以外の補間や新pairの評価はしない。

**Caption draft.** *Saved conditional shot and resource rows at five signal tolerances. Counts and total RZ resources use the hypothetical zero-residual slice with the original bias and normalization policy. PR standard errors and nominal intervals refer to paired saved prices; residual-guard ranges are listed separately and are not statistical intervals. No actual-N/G fields or population-cost guarantees are inferred from these conditional rows.*

## 4. Methodsと再現性情報

[Methods英語草案](non_df_h4_paper_methods_DRAFT_20261025.md)を別ファイルにまとめた。物理入力、split/law、control、compiler、price/MC、bias/normalization/shot、guard、Arb、D1の来歴、データ利用を含む。

元SOURCEの最低引用セット：元H4 `0c329b30a81accb35c212be8defa66fee0dd0f12`、C `e80d7510880f8db993b777c627bb8e95b05a070d`、D1初回 `48d4d27a05a529e05b11599f5f3c671302826617`、D1-B `9843b803878db6c3aee7ab5e7c6bb4818572c944`、D2 `74515d8eb4f073c40fc783681a3775dbca89e9b1`、A_COHORT `34ef0a7ca2412a61f5215a4ce1192e99f3226b2a`、A_AXES `a37232f85812688c3b9ab957c5073c5ce143b17e`、generic会計 `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8`、central会計 `d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34`。6局所Arb個別SOURCEはE14の保存`SOURCE`を参照。

状態は保存raw rayとexact normで特定し、今回stateを再生成・再正規化していない。原本の数値的なH一致診断と、各表現の別reference scopeもMethodsに残す。再現性情報を「元runのnative PASS」と言い換えない。

## 5. 主張文の候補と限界

| 原稿に使える文の候補 | 根拠 | 必須の限定 |
|---|---|---|
| 固定H4の保存回路では、Pauli/DF間のRZ価格の関係がPFとPR、さらにPauli制御方式によって変わる | Fig. 1/E1/E2 | 固定split/law/compiler、PRは保存cohort |
| 同一Pauli bodyでcentral controlを用いると、genericより低いcompiled価格が得られた | Fig. 2/E3/E4 | control構成＋compiler応答。新PRアルゴリズムの提案ではない |
| 選択PR0のD1 ablationは、保持・eventのRZ差と非加法性を可視化する | Fig. 4/E8/E9 | SOURCE-B、初回FAIL保持、generic1代表、basis単独の因果配賦なし |
| 保存平均価格と固定誤差会計では、Pauli centralの条件付きGは対応DFより小さく、指定比較はguard全域で反転しない | Fig. 3/E5/E6/E7 | 実ρの所属・PR母平均tail・共通厳密Hへの新精度転送は未保証 |
| 6局所Arbで、選択回路の数値残差上端が保存予算内と確認された | Table S3/E14 | PF固定singletonとPR0/1だけ。全law・DF・正式受理へ昇格しない |

『PauliはDFより本質的に安い』『PRの母平均Gが厳密に優位』『ρ=0が観測された』『order4費用は0』『DF Gaussian basisが差の大半を生む』『全compiler数値保証が現在のcase studyの必須条件』は、いずれも今回の原稿案に採用しない。

## 6. 次の研究判断：二つの方向

| 観点 | 方向1：現在のempirical / conditional case study | 方向2：PR母平均費用・実精度まで強める |
|---|---|---|
| 支持する主張 | 保存価格、同じbodyのcontrol対照、条件付きGとguard頑健性 | 主張に合う期待価格boundsと実measurement残差を閉じた後の、保証付き比較が目標 |
| 現在揃うもの | 主要Resultsの保存数値、構造説明、Methods、caption案 | C/選択Arb、law descriptor、費用counter等は再利用可能だが十分ではない |
| 追加科学計算 | **現scopeの記述には必須でない** | 新しい数値上界・費用検証・残差転送等は別途対象/範囲を定めて承認が必要 |
| 最小の残作業 | ユーザーの主張scope選択、本文・Abstract作成、既存値による図の編集配置、出典・captionの最終確認 | 下記の二つの不足を独立に扱う設計。必要性を確かめてから限定実行 |
| 利点 | 現在ある科学的知見を過度な保証研究で遅らせず報告できる | 母平均や実精度について強い結論に進める可能性 |
| 制約 | 単一分子/単一条件、交絡、tail/全lawρ未保証。掲載可否や新規性は未判定 | 成功・予算達成は未確定。数値kernel/費用上界に時間を要し、主要比較と独立の問題も残る |

**追加検証候補は、主要結論を実質的に変える次の二点に限定する。** 本段階ではどちらも未採択・未実行。

1. **母平均Gを主張する場合の費用bounds**：費用担当（Codex等）が既存lawとstructural counterを再利用し、full-wrapperの妥当な価格envelopeまたはlaw-weighted bounds、remainderの不確かさ、PR対PRで必要な両側の比較を設計する。新H4 scalar評価/compile/samplingは明示承認後。単一order4 compileや標本追加だけでは全law上界にならないため、既定工程にしない。
2. **実精度込みGを主張する場合のmeasurement残差接続**：精度担当（Codex等）がPF両軸の既存Arb、PR selected証拠、DF/C/Lの保存scopeを再利用し、PR全law平均・DF・共通targetに必要な未被覆部分を特定して限定診断を設計する。新数値解析/科学作用/正式受理は別の許可範囲。一般compiler全体の形式保証を自動的な要求にせず、必要な信号残差に合う証拠を選ぶ。

新しいD1再実行、centralの全成分配賦、DF reuse-off対照、prefix探索、別分子の追加、R2-C証明継続、アルゴリズム改善は、現在のcase studyの既定の残作業に含めない。機構の一意な因果寄与を中心主張に変える場合のみ、別研究判断となる。

次の担当は**ユーザー**：方向1/2と論文の主張scopeを選ぶ。**Codex**：選択後の明示指示に基づき原稿編集を行う。追加科学実行や重要レビューが必要なら、対象と範囲の別承認を受けてから行う。今回GPT詳細レビュー・最終採択は開始していない。

## 7. 今回の実施・保全・終了

6指定報告と最新統合報告、対応JSON/CSV、SOURCEの保存来歴、既存5図を確認した。図表計画は元証拠へのpointer/selectorを使い、新しい数値データセットや作図pipelineを作っていない。保存値の等値、リンク、出典SHA、差分範囲の構造確認のみを行う。主agent自身の確認であり、独立科学レビューではない。

運用は最新版`research_project_instructions_unified_2026-10-25.md` §8.4/§10/§10.3と共通AGENTSに従う。同版の§10.3は取得済み。旧文書が記した当時の版差・push失敗・STOPを修正していない。共有設定/他ユーザー/processへの操作、科学grantの発行/消費、科学entrypoint実行はない。

旧SOURCE/入力/raw/結果/grant/ledger/marker/STOPを無変更で保全。旧N/G null、元native UNKNOWN、D1初回historical FAIL/原因UNKNOWN、PR母平均upper/null、全lawρ/null、R2-C PROOF_INCOMPLETE、正式受理falseを維持した。今回は論文資料の追加であり、科学証拠statusの変更ではないため、既存validation manifest・研究結果・実行記録は更新しない。直接の資料索引だけを追記する。

[構造・保全確認](../artifacts/non_df_h4_paper_plan/2026-10-25/verification.json)、[公開状態](../artifacts/non_df_h4_paper_plan/2026-10-25/publication_status.json)。専用branchをcommit・公開してGitHub remote SHAを照合後、**mandatory STOP**。次の科学実行・研究方針採択・正式certificate受理へ自動移行しない。
