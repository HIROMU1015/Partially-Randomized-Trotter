# 統合評価論文の構成案

2026-10-26。**構成の提案であり、中心claim・投稿先・一論文／二論文の正式採択ではない。** 論文本文の執筆、詳細科学レビュー、新たな数値再解析は今回行わない。条件と出典の正本は[EVIDENCE_MATRIX.csv](EVIDENCE_MATRIX.csv)、横断上の注意点は[RESEARCH_SYNTHESIS.md](RESEARCH_SYNTHESIS.md)を参照する。

## 1. 推奨する問いと貢献の置き方

仮題：**「部分ランダム化時間発展の測定込み資源：決定論的対照、PF次数、ランダム化範囲、制御実装への依存性」**。

一貫する問いは、低い単回回路価格が、bias・normalization・shot負担まで含めても低い総費用になる条件である。Aのbaseline強化、Bの時間／次数、Cの保持範囲／event数、Dの実装対照を、この問いに対する異なる操作として並べる。すべてを同じ比較軸に換算しない。

推奨する貢献候補は次の4点である。

1. Full PFだけでなくB0/B0+I・formula/q/prefixを含む対照を用い、PR利益の比較集合依存性を保存された負の結果とともに示す。
2. H4/H6の長時間・厳精度条件で、m5の大きいδが低い取得域Gと両立する一方、緩精度ではS2が低費用となる条件を示す。
3. Pauli内のHybrid対純RTEについて、確認coverageとevent上限の変更で観測順位が変わり、B/N/Cの釣合いが重要であることを示す。
4. 固定H4・同じPauli bodyのgeneric/central対照から、表現横断比較をcontrol/compiler応答と切り離せないことを示す。

どれも新アルゴリズムの発明、純RTEやPauliの普遍的優位、母平均Gの証明を意味しない。既存の限定的な正式採択結果と、これらを統合した論文claim候補の採否を別にする。

## 2. 主案：一つの統合評価論文

### 1 Introduction

**目的：** 読者が「単回ゲート数が減れば有利」という見方から、測定込み・比較集合付きの評価へ進めるようにする。

扱う順序は、PF/PRの評価対象、biasとBによる測定負担、強い対照、実装依存性、今回の問いと4証拠ブロックとする。既存v0.4のbaseline問題提起とPauli総合資料の背景を再利用する。優先権・新規性は今回新しくレビューしていないので、「初めて」「一般に優れる」などの表現は置かない。

**使う証拠：** AのCEM CE1/CE4、BのREPORT、Cのlatest15、D2。図1の概念図案を対応させる。

**主張候補：** 優位性はHamiltonian表現という一語で決まらず、同じ精度での保持部・近似・normalization・単価・候補集合を含めて定義する必要がある。

### 2 Methods

**2.1 入力と評価scope。** H2/H4/H6/H8、geometry、basis、DF rank政策、保存state/referenceをTable1で区別する。legacy H4 DF12・6距離は別行に置く。H6保存completion、H4共通rayの数値対応、state生成規則差を脚注にする。H8を長時間／Pauli実験まで含む分子と誤認させない。

**2.2 比較family。** Full PF、B0、B0+I、Hybrid PR、純RTEを定義する。Pauli B0/B0Iのconstant保持による同等範囲と、DFのdiscarded-tail identity保持を区別する。BのL_D=0未評価を明記する。

**2.3 PFとfinite RTE。** S2・標準Yoshida4・m5_best、signed piece、DFのABS_TIME_PER_OUTER整数配分、Pauliのs=q*r、Kと多項式次数K+1の意味。PF次数とevent次数を混同しない。

**2.4 回路とcompiler。** full cosine/+Im sine wrapper、scalar一回、state準備P=0、DF Gaussian reuse、generic/central/symmetric-directional制御、Qiskit1.3.0・固定basis/opt1/seed17。同じ設定名でも違うbody・lawのcompiled価格は同一でない。

**2.5 信号と費用。** Nとn、CとG、保存reference/bias/B/u/ρを区別する。Pauliの主Gは仮想ρ=0、DF長時間は経験的allowanceに基づく。既存の各shot政策を共通政策で再計算せず、その差を記載する。実機測定・noise・FT・energy/QPE/RPE全roundは対象外。

**2.6 選抜と不確かさ。** 事前固定nominee、探索選抜後の独立価格確認、確認集合内min、逐次延長を分ける。同じ候補の複数εが価格標本を共有すること、paired軸と非paired候補、K4/K6未観測費用、近似区間の限界を説明する。

**必要な証拠：** 既存input receipt、CEM/ledger、各SOURCE、saved candidate descriptor、n/SE、comparison policy。CSVにunknownのまま残したものを推測で埋めない。新規科学計算は不要。

### 3 DF-native Resource Evaluation

**目的：** 強い決定論的baselineを先に置き、後の長時間・Pauli比較の読み方を作る。

**3.1 固定独立確認。** Aの6主表を使用。H4 DF10粗精度NOT_REPRODUCEDとH6のPF低費用再現を両方示す。H8の同一cohort aliasと公開監査の限界を脚注に置く。

**3.2 FullからB0/B0+Iへの対照強化。** 旧geometryと共通588候補を、対照を選べることの効果として説明する。旧DF12の改善率を新DF10へ流用しない。

**3.3 公式内と全公式の比較。** 保存PR探索を相手とする共通PFの15比較と、同公式45比較を区別する。標準Y4の採択済み固定独立比較はSupplementから参照し、長時間結果に二重加算しない。

**図表：** Figure2（元2A/2Bのscope付き）、Table2（固定独立6比較）。共通588の細表、geometry、B0/B0I詳細はSupplement。

**接続文の要旨：** 一つのT・一つの分割での勝敗だけでは、時間と公式を選ぶときの費用は決まらない。次節でTを拡張する。

### 4 Long-Time and Higher-Order PF/PR

**目的：** 同じDF系列で時間・次数・刻み幅を変えたときの資源の変化を示す。

**4.1 比較域。** 11,632 signal構成と642価格候補を区別し、q32がS2のT1.6/2限定追加であることを明記。価格取得proxyを実測Cや下界として扱わない。

**4.2 q/δとG。** cost_vs_time、selected_delta_vs_time、selected_formula_gridを再利用。ε=.0005でのm5 q1とS2の細分化、緩いε=.05のS2点最良を併記する。

**4.3 C/N/Gとr/K・prefix。** 各公式がパラメータを調整した比較と、同じq/prefix/R32/K4対照を区別する。r/K感度・固定δ図はSupplement。高次化の効果をpiece増とnormalizationから切り離して断定しない。

**4.4 独立確認。** 39候補1,295標本、59固定比較と例外1件をTable3または短い結果欄で提示。探索点最良の図を確認後winnerの図と呼ばない。

**図表：** Figure3、Table3。独立比較全59行、rare-event、r/KはSupplement。

**接続文の要旨：** Bでは非ゼロprefixのDF Hybridを調べた。次節では別のPauli表現内で、非identity保持部をゼロにする境界そのものを比較する。

### 5 Pauli-JW Hybrid versus Pure RTE

**目的：** 「PF対PRが低い」ことと「Hybridに保持すること自体が有利」を分離する。

**5.1 同一Pauli表現内の最新比較。** 3分子×5精度の保存latest15をTable4として使う。14/15は点比較行の内訳であり、独立試行の勝率でも普遍的優位でもない。

**5.2 H2のcoverage。** 旧RTE s2 n96から有望s4の追加確認へ変わった経緯を記載。大s拡張の成功例として説明しない。確認後minと逐次延長を明示する。

**5.3 H6のevent数。** s64→512のB/N低下とC増、s1024のG増加を既存図で示す。r>8の新matched Hybridが未取得であることをcaptionに置く。

**5.4 H4の非改善。** s128/256が改善せず、粗精度だけHybrid低Gである対照を残す。最初の事前固定nomineeと後続確認集合minのKの違いはSupplementで追跡する。

**図表：** Figure4はH6のG対sとB/N/Cを中心にし、Table4に全15行。H2/H4の既存図と旧比較集合はSupplement。

**接続文の要旨：** Pauliでの純RTEの低GをDFへ一般化する前に、同じH4 bodyでもcontrol実装で単価が変わることを確認する。

### 6 Implementation and Representation Dependence

**目的：** 横断比較の交絡を隠さず、制御方式と表現内の実装選択の影響を定量的な保存対照で示す。

**6.1 固定H4の入力対応。** 共通MO積分・raw ray、full-Hの保存一致診断、Pauli項とDF fragmentの保持基準差を示す。近似同一のHamiltonianを厳密に同一と呼ばない。

**6.2 同じbodyのgeneric/central。** D2のPFとPR0、同一Pauli n256価格を優先する。6系列の単価表を用い、control＋compiler応答の効果と表現差を区別する。

**6.3 条件付きGと残差。** ρ=0会計、guard内感度、6局所Arb、全lawρを別の段階として説明する。主文には重要な限界だけを置き、詳細receiptはSupplement。

**6.4 限定母平均C。** 履歴から確認した07d4f6ffの条件付きC分離を短く参照する。詳細はSupplement。全Gや表現一般の保証へ使わない。

**図表：** Figure5、Table5。既存6系列表・D2 CSV・guard表を使用。absolute Gを分子横断の総合順位図へしない。

### 7 Discussion

次の論点を、各ブロックの証拠IDを伴って接続する。

- **何が再現されたか。** Aの採択済み固定方向、Bの58/59と例外、Cの確認集合minは異なる証拠形式である。件数を合算した再現率を作らない。
- **何を最適化したか。** q/δ、PF公式、prefix、RTEイベント数の調整機会がfamilyごとに違う。有限候補最良と理論最適を区別する。
- **何を因果分離できたか。** D2は同じbodyのcontrol対照を持つが、Pauli/DF全差を表現単独へ分解できない。
- **何が残るか。** rare-event全law費用、compiledρ、共通入力手順差、H8公開監査、FT/QPE/RPE外挿。限定ベンチマークの報告に不要な追加計算を必須化しない。

### 8 Conclusion

まとめる対象は「どの固定条件で、どの比較集合の、どの会計値が低かったか」と「対照・時間・保持範囲・制御が評価を変えること」。普遍winnerや一般scalingを結論しない。次の研究方針は本文と区別した将来課題候補とし、今回実行計画を採択しない。

## 3. MainとSupplementaryの分担

|Mainに残すもの|Supplementへ置くもの|理由|
|---|---|---|
|共通の評価scope、4証拠ブロックの条件差|全input hash、SOURCE・cohort・receiptの索引|条件は読者判断に必要。全実行履歴を主文に載せない。|
|A独立6比較と強いbaselineの意味|旧DF12 geometry、588候補の全比較、採択済みY4固定比較|採択と探索を混ぜず、主文の重複を抑える。|
|BのT/δ/formulaと例外|同q/prefix、r/K、59行、rare-event|価格subsetと確認結果の意味を保つ。|
|C最新15行とH6のトレードオフ|H2/H4のs図、旧Hybrid優位snapshot、選抜履歴|14/15という要約が何を数えたかを追跡できる。|
|D同body control対照と6系列単価|D1、guard、局所Arb、条件付き母平均C|主要な実装効果と保証上の細部を分ける。|

## 4. 代替案：二論文

|案|中心課題と使用ブロック|利点|注意点|
|---|---|---|---|
|一論文（推奨）|A→B→C→Dを測定込み費用と比較集合依存性で接続|baseline・高次・純RTE・controlを一つの問いに回収できる。条件依存性そのものが評価研究の骨格になる。|Methodsと保証scopeが長くなる。geometryと技術詳細をSupplementへ置く。|
|二論文案1|長時間・高次DF：Aの必要なbaseline＋Bを主軸|T/δ/formula/r/Kに集中できる。|A既存結果をどこまで再掲するか、C/Dとの非重複を決める必要がある。|
|二論文案2|DF/Pauli実装評価：Aの比較設計、C、Dを主軸|保持範囲・control・comparison-domainの問題が明瞭。|Bを出典以上に重複掲載しない。新アルゴリズム論文や表現一般の優劣へ変えない。|

分割は証拠が独立だから自動的に採択されるものではない。同じ固定H4やC/N/G方法を使う重複を説明でき、各論文に別の中心的な問いが立つ場合の代替である。新規性・投稿方針は今回判断しない。

## 5. 次の判断

現行の限定claimを記述するための必須新規科学計算はない。ユーザーが中心課題と構成・図表scopeを判断した後、必要なら本文作成を別途依頼できる。Figure2を新cohortへ数値更新する、全域最適や実compiled精度へclaimを強める場合は、具体的な追加数値作業を別に検討する。今回のCodex作業は文書保存・静的確認・専用branch公開後mandatory STOP。

## 2026-10-27追加：境界検証の配置案

既存Section4と7の構成を維持し、[境界追加証拠](docs/data_collection/2026-10-27_pr_boundary_addendum/REPORT.md)を次の位置から参照する。新章追加・中心claim変更・一論文／二論文の方針変更は採択しない。

|既存の位置|追加する限定内容|証拠ID・掲載案|
|---|---|---|
|4.3 C/N/Gとr/K・prefix|linear H4/1Å/STO-3G/tol-only DF10、T2/δ2、m5_best、prefix3/q1/K2/P0。R16/R128の独立確認は固定R64よりGが+19.98337%/+11.19767%、ともにHIGHER。CとNの競合があり、R64大域最適性は示さない。|BND_H4_R16_R64、BND_H4_R128_R64。Mainは短い言及、Supplementary表が第一案。|
|4.3／4.4 下限検証と不確かさ|linear H6/1Å/STO-3G/tol-only DF19、保存weighted-Hermitian completion、T2/δ2、m5_best、11 signed piece、prefix7/q1/K6/P0。R11はR16より点G−0.17209%だがINCONCLUSIVE。R11は現allocationの最小値。|BND_H6_R11_R16。表とcaptionにINCONCLUSIVE、旧対照SEを明記。|
|4.4 比較集合・cohort|R32は探索n9だけ。元33/40、7候補追加後33/40、今回T2限定E8/10・C8/10を別記し、全40セルの新順位としない。|BND_H4_R32_EXPLORATION_ONLY、BND_COHORT_CELL_SCOPE。旧独立59比較も保全。|
|2.6／7／Data availability|H8 q16の.001 E8と.0005 C48は既存証拠。.001の正確なA_signal不足で新n48会計・q8比較は未成立。固定実行完了・公開要約と非公開完全SOURCEを区別する。|BND_H8_Q16_REUSE_GAP、BND_EXECUTION_PROVENANCE。欠測と不適格を区別。|

H4の非改善とH6の不確定を、既存探索の境界外を検証しても一般最適性を保証できない実例としてDiscussionへ接続する。Pauli-JWの順位や未採択B3結果には移さない。新図の計算・再集計・renderは行わず、既存図captionとSupplementary表への参照で十分とする案である。掲載位置・中心claimの正式採択はユーザー判断。新計算や詳細科学レビューは開始せずmandatory STOP。
