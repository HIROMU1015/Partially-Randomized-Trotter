# 先行研究調査報告：PR資源評価の独自性と主張の境界

調査日：2026-10-21 JST。これは先行研究と保存成果の比較判断であり、正式研究方針の採択、論文本文、科学結果の再解析ではない。担当はprimary Codexのみ。

## 1. 判断の要点

最も強い貢献候補は、**PRの実用的資源優位を、phaseを保持した切り捨て対照・細かいPF候補集合・実コンパイル・探索と分離した確認標本によって検証し、優位が成立する条件と崩れる事例を具体的に示したこと**である。これは新しい定量的知見になり得るが、一般的な新アルゴリズムや新しい資源理論としての新規性は弱い。

最重要の比較先はGüntherら [P01](references.md#p01)。2025年のarXiv初版だけでなく、2026年5月19日のPRX Quantum誌版と2026年7月10日のarXiv v2を確認した。同論文にはPR+RTE、single-ancilla QPE、DFへの適用、normalizationに伴う反復増加、誤差予算最適化、truncation対照、Toffoli/2-qubit資源が既にある。「PRをDF量子化学に適用」「shotを掛けた資源を初めて評価」「切り捨てと初めて比較」は支持できない。

科学的妥当性は別問題である。保存された固定scopeの点比較・非再現・会計感度は、限定された実証結果として有用である。一方、希少eventを含むcompiled費用の母平均順位、経験的bias allowanceの厳密保証、全PF/PR法に対する最適性は確立していない。独創性が限定的でも局所的な結果は直ちに無効にならず、局所的な結果が正しくても独創性や一般的優位が保証されるわけではない。

|判断|確信度|理由|
|---|---|---|
|PR・RTE・DFという構成自体の新規性は弱い|高|P01/P02/P03/P10に直接対応する構成・式がある|
|強化対照とcompiled費用を用いた本研究の特定の逆転・非再現は追加知見|高（保存scope内）|既存結果に具体的な対照差・確認cohortがある。新しい数値の算出は不要|
|同じ実証研究として先行研究と十分に差別化できる|中|問いと証拠の組合せには差があるが、一般化とfull-support費用の不足が残る|
|独立論文として成立し得る|中・条件付き|限定resource benchmark/適用条件研究として。採択可能性や高いimpactは断定しない|
|母平均PR優位、FT優位、新しいPR法として成立する|支持不足|現在の証拠と先行研究比較では満たさない|

## 2. SOURCEと比較対象を固定する

主たる基点は `80b7a35c4dbf683f952293eeb55944bc10b9efc8`。共通AGENTS、worktreeのAGENTS、PROJECT_MAP、研究概要、プロトコルv2、関連結果文書を読んだ。GitHub connectorで指定3commitのSHAと存在を確認した。これはremote全rawや過去run全体の再監査ではない。[監査](audit.md)と[SOURCE一覧](audit/source_inventory.json)に取得範囲を保存した。

現行比較はlinear H2/H4/H6/H8、隣接1 Å、STO-3G、tol-only 1e-8のDF rank 3/10/19/28、保存状態、T=.8、P=0を基本とする。ただしstate solver、入力生成工程、候補域まで全て同一とは認定しない。H4固定DF12のgeometry/Mechanism/理論比較は別系列である。

費用はfull cosine/sine Hadamard wrapperのcompiled RZに保存analytic shotsを掛けたもの。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし。nは古典的にコンパイル費用の平均を推定するtrajectory数、Nは必要量子測定回数の解析値であり、同じ量ではない。実機N shots、T/Toffoli、physical runtime、全energy/RPE費用を測定した結果ではない。

本報告は新しいmean、SE、比、差、CI、n、順位を一切生成しない。以下の数字は保存文書・表の引用で、再集計ではない。出典ごとにresult/SOURCE/cohortを分離する。

### 2.1 独立確認の保存結果

[プロトコルv2](../track_a_pr_resource_evaluation_protocol_v2.md) §5/§10と[H468統合](../track_a_h468_paper_evidence_integration_20261019.md)の転記。全行T=.8/P=0、通常signed/allowance倍率1。

|系・DF|ε_sig|PR L_D/q/r/K（δ）|n|固定PF|保存判定・低費用側|
|---|---|---|---:|---|---|
|H2 DF3|.05|1/1/2/4（.8）|80|B1 p3/q1 S2|REPRODUCED・PR|
|H2 DF3|.0005|1/4/2/4（.2）|12|B1 p3/q4 S2|REPRODUCED・PR|
|H4 DF10|.05|3/1/4/2（.8）|1108|B0I p5/q1 S2|NOT_REPRODUCED・PF|
|H4 DF10|.0005|2/8/2/4（.1）|78|B0 p7/q8 S2|REPRODUCED・PR|
|H6 DF19|.05|4/2/4/2（.4）|48|B0I p7/q1 S2|REPRODUCED・PF|
|H6 DF19|.0005|4/8/4/2（.1）|104|B0I p11/q4 S4|REPRODUCED・PR|
|H8 DF28|.05|8/2/4/4（.4）|48|B0I p11/q1 S2|POINT_REPRODUCED・PR|
|H8 DF28|.0005|7/16/4/4（.05）|48|B0I p15/q4 S4|POINT_REPRODUCED・PR|

H6粗精度のREPRODUCEDは、事前のPF優位方向が再現した意味。全てを「PRの勝利」と数えない。H2追加2条件は研究資料で受入済みだが既存論文主表6条件への掲載採択とは別。H8主表・Figure2B・MechanismのH8_CONFIRMは同じn48のaliasで、独立証拠を増やさない。

### 2.2 追加価値のある保存事例

- **対照追加で全面優位が崩れる。** H4 legacy DF12、上記とは別の6距離、T=.8、S2、L_D=3/4/5/6/9、q=1/2/4/8。元集合では1812 geometry/precision点全てでB2が点最小だったが、実測B0+I追加後はB0+I397/B2 1415。240軸のcompiled RZ差は旧B0比0。既知tail identityの保持が、このcompilerでは単価を増やさずbias/shot会計を変えた。[B0I結果](../track_a_h4_b0i_map_results_20261014.md)。1812独立実験や厳密な連続交点ではない。
- **理論とcompiledで固定pairの方向が異なる。** H4 legacy DF12、1 Å、T=.8/P=0、PR p3/q8/r2/K4（δ=.1、fixed_anchor n32）対full PF p12/q1/S4（δ=.8）、ε=.05、倍率1。保存比は理論0.8716321313、compiled1.0393357766。[固定結果/図](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f15752625613956d087bab221e48f1ce7dd59fef/docs/research/track_a_h46_theory_compiled_total_plots_20261019.md)。特定NATIVE_MIXED分解と有限compiled標本の差であり、P01のToffoliモデルの誤りや母平均逆転を証明しない。
- **探索優位が新標本で再現しない。** H4 DF10 ε=.05の上表。これは候補選択と費用推定の分離が実際に必要な理由を示す。非再現の原因を選択後バイアスだけ、希少eventだけに断定する分解はない。
- **単価モデルと大局方向の頑健性は別。** 80b7a35cの採用済み100 scope（重複除外40 endpoint）はmixedからdirect価格へ置換し、最良PF候補変更2、最良費用/全順位変更各50、PR/PF方向逆転0、PR低費用90/PF10。H6/H8、1 Å/STO-3G/DF19/28、T=.8/P=0、登録prefix/q、旧n32 cohortの条件付き診断。[採用結果](../track_a_pf_q_model_evidence_integration_20261020.md)。新しいP表の独立確認や、100独立実験ではない。

## 3. 最も近い先行論文との詳細比較

### 3.1 Güntherら：既にどこまで示しているか

[P01](references.md#p01) の§V Eqs.(32)–(39)、Fig.7は、H_Dの決定論PFにRTEでH_Rを挿入し、Hadamard期待値をnormalizationで減衰させる構成を与える。Appendix A2 Eqs.(A20)–(A29)は偶数/奇数Taylor項をまとめたLCU、その確率分布、B²のsampling overheadを記述し、有限次数への打切りもWanらへ参照している。本研究のcanonical finite-RTEの全偶数次数重みはこの系列に直接対応する。finite Kで実装・検証したことは有用だが、新しいRTE原理ではない。

§VII B Eqs.(42)–(43)、Fig.10はDF Hamiltonian、Givens basis change、第一/第二factorizationのpartial randomizationを扱う。さらにorbital basisの異なるinvolutionをtailに用いることと、supportに応じたbasis change費用も議論する。したがって「DF-native」「PauliではなくDFを保持」「basis変換を節約」は、具体的な新規回路・保証を特定しない限り新手法の根拠にしない。

Appendix E、特に誌版Eqs.(E21)–(E23)は、roundごとの反復数とdeterministic/random gate単価を合算し、κ、δ、誤差配分、L_Dを最適化する。単価とshotsの積、normalizationと深さの競合、費用指標で最良分割が変わるという原理は既出。同論文はRZだけより広いToffoli/2-qubit資源とQPE全体まで評価する。本研究の方が資源評価の「総範囲」が広い、とは言えない。

Appendix D1/Fig.18にはpartitionによるTrotter errorの変化、D2/Figs.19–20にはtailを落とした基底energy誤差がある。Fig.19はPRで選んだpartitionのH_Dを調べ、Fig.20はalanine active spacesについて残存項数を走査する。ただしD2はTrotterization誤差を含めないと本文に明記する。この点と、B0/B0Iごとにq・PF次数・compiled wrapper・signal headroomを組み合わせる本研究の比較は異なる。

この差は「先行論文にはdiscardがない」ではなく、**同一の固定complex-signal taskに対する、phaseを保持した決定論的候補の実装込み比較と、その結論の確認方法を追加した**と述べるべきである。P01のenergy-QPE設定と本研究の固定T設定は異なるので、本研究の局所反例をP01全体の反証としない。

### 3.2 Composite、cost-aware sampling、統計誤差

Hagan–Wiebe [P03](references.md#p03) §4–5は高次PFとqDRIFTを組み合わせ、diamond distanceに対するoperator-exponential complexityを与える。Jin–Li [P07](references.md#p07) はhybridのbias/variance、SparSto [P06](references.md#p06) はsparsificationとexpected budgetを扱う。qDRIFTの原論文 [P04](references.md#p04) Fig.2には小項のtruncationとPF次数選択もある。強化対照という発想が未発見だったとは言えない。

Kissら [P08](references.md#p08) §3.3 Eqs.(38)–(39)はterm別costを認めるCompositeの最大/平均費用を扱い、importance sampling、bias/variance、実験数の関係を検討する。本研究の単価測定を新しいcost-aware方法論と呼ぶには、この先行研究にない最適化則や保証が必要である。現在は固定lawを維持した実装比較・検証に当たる。

Wanら [P02](references.md#p02) Lemma 2とGüntherらはLCU重みとsample burdenの交換を既に与える。SCU [P28](references.md#p28) Table1/Eqs.(5)–(8)、randomized multiproduct [P15](references.md#p15)、single-ancilla LCU [P35](references.md#p35)にも近接した一般構図がある。回路が浅くなるほど必ず総費用が下がるわけではない、という一般説明を中心的新規性には置かない。

一方、本研究は**回路費用Cの古典MC推定誤差**と、**signalを推定する量子測定Nの誤差**を分ける。この区分は通常の統計学で既知だが、PRの実際の費用比較に即した観測・未観測eventと独立確認のデータは追加知見となる。concentration for random product formulas [P14](references.md#p14) は典型unitary/channel誤差を扱うため、その定理をcompiled Cの母平均保証へそのまま使えない。

### 3.3 DF・low-rank・高次PF・コンパイル

Mottaら [P10](references.md#p10) はlow-rank/truncationとbasis rotation、Berryら [P11](references.md#p11) とvon Burgら [P12](references.md#p12) はtensor構造とFT回路資源を扱う。SCDF [P23](references.md#p23)、spectrum amplification [P24](references.md#p24)、active-volume compilation [P38](references.md#p38) はさらにrepresentation/architectureを最適化する。本研究はこれらとのFT総資源比較を完了していない。

PFの誤差定数・次数・構成長を同時に比較する考え方はMoralesら [P21](references.md#p21) に明確。Lane–Browne [P22](references.md#p22) はS2にもordering等の自由度があることを示す。Abeら [P33](references.md#p33) は公開済みの別のdeterministic higher-order成果であり、その新S4の発明を今回のPR研究の追加成果として数えない。進行中の4次PF研究の未確定結果は本報告へ取り込まない。

明示回路で資源を比較する一般的意義はChildsら [P18](references.md#p18) にもあり、単に「解析的gate countとcompileを比較した」だけでは独創性が弱い。本研究固有なのは選んだnative DF/full-wrapperの境界融合・control・scalar・RTE列に対して実際に観測した差と、同じ保存shotsのもとでの比較である。理論分解の期待値とcompiled有限標本は異なる対象で、差をすべてcompiler最適化の因果効果としない。

## 4. 2025–2026年の近接研究が与える制約

|文献|確認した近接点|本研究の主張への影響|
|---|---|---|
|P25 Paganelliら、2026|§V B、Figs.3–4に精度・不均一係数依存のrandomized/deterministic crossover。App.CはgridsynthによるT gate会計|精度領域依存そのものは既知。典型約10^-3というabstract表現を普遍的境界として輸入しない|
|P26 Kanasugiら、2026|v2 §II B4 Eqs.(17)–(18)でPR rotations×shots、UWC、STAR物理資源|PR化学資源を初めて総合評価という主張は不可。native compiled benchmarkという差を限定する|
|P27 Casaresら、2026|SPRINT/GRADE、Fig.1にordering randomization、near-integrable PF、qDRIFT/RTE等の組合せ|「S4を加えたので最強PFと公平比較済み」は過大。構造化PFやrepresentation最適化の余地が残る|
|P29 Zlokapaら、2026|Theorems1–2、term-norm profileとworst-case gate/query bounds、time-dependent family|Compositeの既存理論を強化。本研究の固定状態signal/DF小系へ直接の最適性定理にはならない|
|P30 Mohammadipour–Li、2026 v2|Theorem1のindex-sharing MLMC、§2.3で量子shot-noiseとaugmented-state条件|本研究のMC費用診断と別。単純bootstrapやn増加を新しいvariance reduction法と呼ばない|
|P31 Morisaki–Fujii、2026|Haar平均誤差、Hilbert–Schmidt sampling、grouping、fixed Rz depth|最良sampling・groupingの比較がないため、現lawでの結果をrandomized一般の限界にしない|
|P32 Kim–García-Pintos、2026|H=A+αB、access model別のrandomized高次PF優位|randomization一般の優位/不利を固定実験から断定しない|
|P37 FreeQuantum、雑誌2026|AppendixでPR/qDRIFT化学資源評価を応用|PRを実分子へつなぐ将来性自体の独自性は弱い|

P25のEq.(29)は一般のspectral normの代わりにoperator differenceの固有値絶対値最大を用いるproxyである。一般の非normal行列では両者は同一でない。同論文の「randomizationに有利なensemble」という説明も、全化学Hamiltonianに対する厳密な普遍上界として採用しない。本研究のsignal precision、P25のerror proxy、P01のenergy RMSE、P29のtrace distanceを同一εとして比較しない。

P01から引用された基礎系列、P03のComposite引用系列、2026年のPR応用、DF/高次PF系列を追跡した。探索は39論文の記録に到達したが網羅性の証明ではない。本文を全取得・全読了したとはせず、[各文献の確認範囲](references.md)を根拠に評価した。

## 5. 費用会計と公平性の独立評価

本研究では軸aについて、保存bias b_aと経験的allowance u_aを用いて

\[
h_a=\epsilon_{sig}/\sqrt 2-|b_a|-u_a,\qquad
N_a=\left\lceil 2 B_{upper}^{2}\log(2/\alpha_a)/h_a^2\right\rceil,\quad
\widehat G=\sum_a N_a\overline C_a
\]

と評価する。h_a>0、固定normalization、対応する独立sampling等の前提下の十分shot会計である。Hoeffding式が解析的でも、b/uを厳密certificateへ昇格できない。biasとnormalizationの関係、hが小さくなるとshotsが増えることは既存式から当然に導かれる。新規性はその代数ではなく、実装と対照に結び付いた具体的な量と結論の変化にある。

B0+Iは既知scalar phaseの復帰であり、新しいHamiltonian simulation原理ではない。controlled/Hadamardではglobal phaseがancillaの相対位相として観測されるため、比較で落とすとtaskを変え得る。P01のAppendix Eにもphase補正を扱う箇所がある。本研究の価値はscalar保持そのものの発見より、その合理的対照がcompiled単価とbias/shotの競合を通じて実際の結論を変えたことにある。

PF側をfullだけからB0/B0I、細かいprefix、S4へ広げたのは重要。ただしH6/H8ではPF q≤8に対しPRにq16個別点があり、両側の最適化域は完全に対称ではない。PF/PR候補数も異なり、全順序・全分割・全高次formulaを最適化していない。登録集合での結論として提示すれば妥当だが、PRという方法全体と最適deterministic法の勝敗とは呼べない。

厳しいεでPRが低い本研究の一部条件を、高精度で一般にrandomized法が有利という知見にしない。固定q/prefix/K格子やbias適格境界が関係し得る。P25等の高精度crossoverと逆方向に見えても、task、誤差、候補域、randomization方式が異なるため矛盾ではない。真のε→0漸近性を本データから推測しない。

signed biasで相殺を許す評価と、経験的非相殺を分けた意義は、結論の会計依存性の可視化にある。非相殺版も入力・浮動小数工程の厳密保証が自動で得られるわけではない。allowance倍率10の感度は有用な感度であり、未観測eventや未知model discrepancyを全て覆う保証ではない。

## 6. 実験設計・証拠の強さ

独立確認では候補/PF/期待方向/n/保存shots等を新sampling前に固定し、元law、paired cos/sin、新seed namespaceを維持した。この分離は新しい統計方法ではないが、実際のPR費用順位の信頼性を判断するうえで重要な実験設計である。探索から確認への因果分解・selection-adjusted CIが新たに得られたわけではない。

H4のMC補足は同じP cohortの保存経験分布の解析であり、新しい独立確認ではない。約1/√nのSE減少は通常の統計則。観測S2群のfull PR費用平均が高くても群内変動・範囲の重なりがあり、S2件数だけで費用が決まらないという具体例を与える。bootstrapは経験分布にないeventを生成できず、未観測tailの費用/分散はunknownのまま。[MC補足固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bee9ae996fffec988d72651e295481b88afd9d73/docs/research/track_a_h4_mc_supplement_20261019.md)。希少RTEの追加調査は別担当に委ね、ここでは既存の未保証範囲だけを引用した。

H8はa6246f03のredacted公開packetにより、結果表に加えて実行・worker・seed/cohort・SOURCE/input/runtime等の追跡情報が増えた。5f267e96の補足では原profileが契約内JSONとして回収され、canonical digestとruntime mapの一致が報告される。**古い「原profile未取得」は現行判断に使わない。** ただし本報告はredacted receiptを読んだものでprivate原bytesの再監査ではない。保存PR回路は0/192のまま、元回路からのfingerprint再導出・旧compiled output bit同一性・final angle/T資源は未検証。匿名化版だけでseed replayや原profile digestを独立再計算できない点も残る。

## 7. 最終研究判断：7つの問い

1. **最も強い科学的貢献は何か。** phaseを保持した対照と実装費用・固定確認により、PRの点優位を成立条件付きのものとして定量化し、崩れる事例を示せること。特にB0I追加、理論/compiled固定pairの方向差、H4探索の非再現が中核候補。確信度：保存scope内は高、一般性は低〜中。
2. **既存PR研究と本質的にどこが違うか。** アルゴリズムの原理ではなく、固定complex-signalに対するnative full-wrapperの実装単価、合理的対照、費用標本を分離した検証という問いと証拠。P01のより広いenergy-QPE/Toffoli解析を置き換える成果ではない。確信度：高。
3. **独立論文になり得るか。** 条件付きで可能。新理論・新PR法としては弱いが、適用条件・実装依存性・比較の頑健性を問う検証研究として筋はある。現時点で「完成した一般的優位性論文」とは判断しない。確信度：中。出版可能性は投稿先・主張の限定・再現資料に依存する。
4. **中心に置くと価値が明確な主張は何か。** 「強い決定論対照とcompiled費用を使ったとき、PR資源節約のどの部分が残り、どの部分が消えるか」。式だけからは分からない具体的な逆転・非再現・条件付き頑健性を軸にする案を推奨。正式採択はしない。
5. **中心にすべきでない主張は何か。** PR/finite RTE/DF適用の発明、N×C会計や1/√nの発見、普遍的な高精度PR優位、世界初compiled比較、全PF最適性、希少event込み母平均優位、T/FT/energy全体の優位、既存higher-order PFの発明を今回に重複計上すること。確信度：高。
6. **新規性の観点で不足しているものは何か。** 反例の集積から一段進んだ、どのHamiltonian/precision/control/費用指標で結論が変わるかの移送可能な説明または判定則。加えて、主張を母平均へ置くならfull-support費用の扱い、一般PF比較へ置くなら候補域の対称性、FTへ置くならangle/synthesis/architecture会計が必要。新しい科学実行をこの報告で要求・開始するものではない。既存証拠でどこまで回収できるかは別判断。確信度：中〜高。
7. **進行中の4次PF研究が何を補強し得るか。** 高次化によるbias減少がnative compiled単価・負時間stage・RTE normalization/shotsの増加を上回るか、という非自明な総費用比較を補強できる可能性。現行S4対照はdeterministic側であり、S4-PRの有効性を示してはいない。新S4自体の独創性はP21/P33/P27等と照合し、両側の公平最適化と独立cohortが必要。優位が出るとも、科学的貢献が必ず強くなるとも予告しない。確信度：条件付き中。

[Claim Matrix](claim_prior_art_matrix.md)、[査読想定](novelty_and_reviewers.md)、[方向性の順位](directions.md)、[未解決一覧](unresolved.md)を併読する。本報告は文献探索の限界を含む暫定的な研究判断であり、先行研究の不存在や全証拠のfreezeを宣言しない。

**MANDATORY STOP。** 論文原稿・新しい科学処理・別研究への介入・新規pushへ進まない。
