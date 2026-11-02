# Track B：BF-A後の研究方針再設計

作成日：2026-10-05  
位置付け：研究方針・研究計画の提案。新規性の確定、実装済みmethod、科学実行authorizationではない。

## 0. 結論

B-Fの現行仮説を、登録されたfamily・入力・精度・探索予算の範囲で閉じる。科学再実行、欠落bridge補完、7-stage等への自動拡張は不要である。

ただし、BF-Aから「部分ランダム化内部の最適化余地が一般に尽きた」「構造を変えれば改善する」とは推論しない。いずれも現証拠を超える。

次の第一候補は、**DF構造に基づく実行頻度削減の構成研究**とする。便宜上B-Mと呼ぶ。既存のmethod名B2、旧BF-2とは別である。

> DF Hamiltonianの一部のfragment群を低頻度で実行するとき、増える分割誤差と減る実装費用を、利用可能なDF情報から評価し、有利な場合だけ非一様な実行列を構成できるか。

**multi-rate／coalescing、三層分割、norm以外の重要度、cost-aware選択そのものは既知である。** その名称を変えたものを新algorithmとはしない。B-Mで狙うのは、具体的なDF構造の評価法と、それに結び付く実行列の作成手順である。

最初に取り組むのは数学・構成・少数の反証可能なmodel familyの設計まで。既に用意された分子snapshotを再度開いてwinnerを探す段階ではない。研究Aは独立に現在のscopeで原稿化を進め、Bの成否に依存させない。

## 1. 根拠を固定するcommitと、今回の読解範囲

| 役割 | commit |
|---|---|
| 再設計handoff | `0da4d18acf3f5d32d1bc32c9661b667885bcf5f2` |
| Bの最終科学履歴・read-only recovery | `6d2645a09440f50e5b869ef42a1b73a1b625a1af` |
| 元BF-1 science source | `e59344a564e70d64dc3ea39d640581c72676df31` |
| Aの今回の参照snapshot | `4c23453c541700c6a41ba71fc5ec9323b53858d6` |

Repository：`HIROMU1015/Partially-Randomized-Trotter`。

今回用いたのは、handoff、Bの復元結果報告、過去STOPの直接文書、FRの公開snapshot、AのPM-0と主張・証拠対応表、関連一次文献である。全raw artifact・全sourceの独立再監査を行ったとはしない。引用する科学数値は保存報告の値であり、新しい再採点ではない。

以下を区分する。

- **保存証拠**：リポジトリの契約・実行・復元・既存解析に記録された内容。
- **文献確認**：今回取得した一次本文・該当節・図等から確認した内容。
- **設計用導出**：下記の近似式、恒等式、仮説整理。分子で実証された結果ではない。
- **提案**：今後採択・実装・実行を別途判断する内容。

今回、新しい分子生成、NPZ/state読込、signal評価、係数探索、trajectory、circuit build/compile、research tests、repository変更は行っていない。

## 2. B-Fの正式な着地

### 2.1 復元結果の正確な意味

BF1-R0は、元の一回の科学データから、元sourceの探索・採点規則を事後的にreplayした。原BF-1の`INCONCLUSIVE`は変更しない。別の復元結果において、preregistered primaryはBF-Aとなった。[R1]

| 指標 | 保存値 |
|---|---:|
| Fの最良finite action-work | 20,709,936.722970817 |
| L集合の共通finite採点最良 | 同じ値、同じSuzuki5 q1/R10/K2 |
| 共通参照最良 | native S2 q2/R5/K2、12,924,223.160897588 |
| F/L | 1.0 |
| F/共通参照 | 1.6024124982326993 |
| primary比の区間 | [1.6021983874, 1.6025769383] |

BF-Aの機械条件はratio >= .99であり、「全方法がほぼ同じ」を意味しない。FとLのbestは同じだが、Fは最良参照より約60.24%高い。

`SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED`というflagも過読しない。この例ではF winnerはL自身のbestで、L集合に実際に含まれている。Lが良い係数を取り逃したという事実を示したflagではない。

### 2.2 閉じるclaim

> H4 1.00 Å、STO-3G、DF rank12、generation-prefix L_D=3、T=.8、epsilon=.01、指定5-stage familyと32評価/armにおいて、finite-task objectiveを用いた係数探索は、leading-tail objectiveを用いた探索集合に対する独自の資源利得を与えなかった。

これは限定されたnegative evidenceである。全PF、別精度・系・family、他の費用指標についての不可能性証明ではない。[R1]

BF-AはBの設計履歴・negative benchmarkとして保存する。BF-A一件だけで新しい独立論文が成立すると約束しない。Aの本文へ無理に混ぜない。技術補足・研究ノートとして閉じることが、現時点の最小で正確な着地である。

### 2.3 次のために修正する、以前の解釈

「有限補正・係数最適化がこの例で効かなかった」ことと、「現architectureの全最適化余地が小さい」ことは別である。B-Sの小さいheadroomも、3個の経験分布上のcost-resampling目的についての診断であり、全samplerへの上界ではない。[R2]

過去STOPは現在の計画の優先順位を下げる根拠だが、分野全体の再研究を永久に禁止する根拠ではない。別の具体的な機構・入力・対照によって新しい問いを定義できたときに限り、別計画として検討する。BF-Aを取り消して同じrunへ条件を足すことはしない。

## 3. 過去証拠から残すもの

| 系列 | 保存証拠から言えること | 次の研究での使い方 |
|---|---|---|
| B-F | 限定familyでF/L best一致、S2参照のaction-workが低い | finite refinementを主要価値にする仮説は閉じる |
| P-D | 固定308候補の選択近傍では主要モデル一致。ただし全域ではfalse acceptanceや大きい差もある | 「finiteモデルは常に不要」と一般化しない。内部workを数える |
| B-S | 元32標本×3設定のplug-in headroom約0.03–0.12% | 現在の記録を根拠としたsampler本格実装は保留 |
| R3 | 既存multi-fidelity／SPRINTを超える具体的構造・保証を固定できなかった | 一般optimizerに特徴量を渡すだけの研究へ戻さない |
| FR | 具体化された多くの主張は既知幾何・平均式の帰結。RTE制約付き最適性は未定義部分が残る | 平均operator/channel、情報取得費用、certificate/heuristicを区別する |
| P-A等 | interval固有の差は得られなかった一方、以前のsupport/basis実装には効果と競合があった | 実装の再利用と、新しいcompiler方法の発明を区別する |

上表は同じ条件の独立反復試験の集合ではない。すべてを足して「統計的に改善余地がない」とはしない。[R2–R6]

## 4. Track Aから得る示唆と、得られない示唆

Aは固定DF・二次PF・canonical finite-RTE・所定shot規則の有限信号resource case studyとして閉じている。[R7]

M1/PM-1では近接discard rank4/5を追加しても、登録集合内でB2の低いprimary点推定が残った。PM-2では同じ保存値の精度感度によりB2内部のq4/q2/q1が切り替わった。[R7]

同じR=qrを持つ比較では、normalizationとrandom action期待値が同じでも、qを増やすと1-shot費用とshot数が競合した。[R8]

ここから得るのは、**反復の実装費用を減らす研究を検討する動機**である。

得られないのは、各fragmentの誤差寄与、cross-group commutator、deterministic/random/basis別compiled費用の完全分解である。これらは保存集約値から復元できないとAのPM-0にも明記されている。[R8]

したがって、「強いfragmentだけ頻繁にすれば安くなる」といった新案の成功を、Aの成功から演繹してはいけない。Aでq=1が有力な緩い精度域では、macro反復をさらに減らす余地自体が乏しい可能性もある。

## 5. 新規性監査：multi-rate推奨の修正

### 5.1 直接近い古い先行研究

Poulin et al.のcoalescingは、異なるHamiltonian項を異なる間隔で実行するmulti-resolution Trotterizationを明示する。§VI、Eq.(26)は実行頻度を変えた具体例である。単なる係数の大小による優先付けで有意な改善が得られなかったこと、別の重要度規則も記載されている。結論ではsum-of-squares分解とcoalescingの組合せも試し、改善が得られなかったと報告する。[W1]

この最後の結果は、今回のDF構成全体の不可能性を示すものではない。一方、**「DFにmultirateを入れる発想に直接の先行例はない」と説明することはできない。**

### 5.2 現在の三層hybridも広く既知

SPRINTのFig.1、§III.B–C、§IVでは、dominant群に高次、別群に低次、残差にqDRIFT/RTE等を配する構造と、processing・実装の調整が統合されている。次数を変えることと更新頻度を変えることは完全に同一ではないが、三層へ異なる手法を配る一般構想を新規性にはできない。[W2]

Compositeはcoalescingを背景にTrotter/qDRIFTのpartitionを定式化する。channelを主に扱うため、その保証を今回の平均振幅へそのまま移せるとはしない。[W3]

### 5.3 誤差評価を安くすることも既に進んでいる

Maxwell et al.はBCHのcompactな表現、commutator grouping、重要な寄与の抽出、classical state表現を使う評価を提案する。§III.B–C、§IV.Aが今回の具体的な比較対象になる。[W4]

したがって「commutatorで重要度をつけた」「多数のcommutatorをまとめた」「errorとcostで選んだ」だけでも新規性には足りない。

### 5.4 Claim単位の評価

| Claim候補 | 現時点の評価 |
|---|---|
| fragmentごとにTrotter stepを変える | 既知、W1 |
| fine/coarse/randomの三層に分ける | 一般発想の重複が強い、W2/W3 |
| norm以外の重要度を使う | W1ほか既知 |
| relative Gaussian basisを融合してcostを下げる | factorized simulationの既知実装と現sourceを比較すべき。融合だけを新methodにしない |
| 安価なcommutator評価を使う | W4やpartition-error研究との具体的比較が必要 |
| **DF表現だけから、特定groupの低頻度化による追加誤差・実装削減を評価し、実行列へ落とす具体手順** | **今回残す技術的目標。新規性・有用性はまだ未確定** |
| 未使用条件でsame-taskのactual resourcesを減らす | 方法上の差を裏付ける必要な実証。これだけで「世界初」は証明しない |

今回の監査は関連一次本文の重点照合であり、すべての引用網を網羅する不存在証明ではない。新規性を断定できないまま研究を終了するのでも、空白と仮定して大量計算するのでもなく、最後の具体的な技術目標を構成して比較する。

## 6. 再設計する主RQと成果物

### 主RQ

> 固定されたDF Hamiltonianとcoherent-signal taskについて、各fragment群の実行頻度を下げたときの追加誤差と実装費用の変化を、exact many-body signalを必須としない情報から評価し、uniform／既知coalescing／near-integrable設計より有用な非一様実行列を作れるか。

「有用」は、正しい演算・誤差の扱い、古典設計費用、同じtaskでの量子資源、再利用範囲を含む。あらゆる入力で勝つことや、新しい分割を必ず選ぶことは要求しない。

### 副RQ

1. 群内部を細かくすることで減る誤差と、群間の分割で残る誤差を分離できるか。
2. DF小行列の構造から、その分離に使える量を安く評価できるか。既知の一般BCH評価・norm重要度より何が良いか。
3. 実行回数削減がbasis遷移・controlled implementation・RTE normalization・shotsまで数えて利益として残るか。
4. 開発した構成／規則が、開発に使っていない構造条件でも役立つか。

### 作りたい成果物

- 明示したnative primitivesから実行列を生成する手順。
- 低頻度化操作ごとの誤差評価、または適用可能性を説明する定量的指標。
- 指標の取得情報と古典計算量。
- 強い対照を含むmatched-task評価。
- 成立しない条件と撤回可能なclaim。

単なるwinner表・一般optimizer・たくさんの合格statusを最終成果物にしない。厳密な新定理は唯一の成立形ではないが、heuristicならその旨と独立評価が必要である。

## 7. 数学的な出発点：内部誤差と群間誤差を分ける

以下は**本計画の設計用導出**。新定理・分子での実証ではない。

### 7.1 二層の理想化した説明

Aのnative symmetric PFをS_A(u)とし、

\[
\log S_A(u)=-iuA+u^3K_A+O(u^5)
\]

と書く。説明用にBのexponentialをexactとして、

\[
V_m(h)=e^{-ihB/2}[S_A(h/m)]^m e^{-ihB/2}
\]

を考える。固定m、十分小さいhのBCH展開は

\[
\log V_m(h)=-ih(A+B)+h^3\{K_{\rm cut}+K_A/m^2\}+O(h^5)
\]

となる。K_cutはA/Bの分割に由来する。B内部もPFなら、その内部誤差もmで減らない部分に含まれる。

**mを増やしても、群間のcross errorは一般に消えない。** 大きいnormのAを選ぶことと、細分化が有効なAを選ぶことは別である。Aがexactに実装可能な一つのgeneratorなら、反復は融合でき、内部誤差を減らす自由度自体がない。

### 7.2 設計用の連続cost診断

三角不等式に基づくleading評価を

\[
E_{\rm model}\simeq\frac{T^3}{q^2}
\left(\chi_{\rm floor}+\frac{\chi_{\rm fine}}{m^2}\right)
\]

と書く。**chi_floorは評価式に残る項であって、実際の誤差の下界ではない。** 誤差相殺や高次項があり得る。これを物理的な精度限界にしない。

さらに1 macro-stepの費用をm c_f+c_cと近似し、shot数とtail費用の変化をいったん固定、qを連続として除去すると、費用のm依存は

\[
f(m)=(mc_f+c_c)\sqrt{\chi_{\rm floor}+\chi_{\rm fine}/m^2}
\]

である。全定数が正なら、停留点は

\[
m_*^3=\frac{c_c\chi_{\rm fine}}{c_f\chi_{\rm floor}}.
\]

これは初等微分による診断であり、新しい一般最適化定理ではない。q>=1の整数条件、actual fusion、RTE normalization、統計予算、高次項を戻すと実際のwinnerは変わり得る。zero denominator等の退化caseは別扱いにする。

この式を用いる価値は、「coarse群が高価」「fine群の内部誤差が大きい」「cutに由来する残りが小さい」の三条件が同時に必要になり得ることを、実装前に確認できる点にある。

## 8. DF情報で何を計算するか

### 8.1 入力にある小行列からの恒等式

spin orbital数n、粒子数N、

\[
F(G)=\sum_{pq}G_{pq}a_p^\dagger a_q,
\qquad D_l=\lambda_l F(G_l)^2
\]

とする。係数・one-body補正は実際のsnapshot規約に合わせ、1/2等を勝手に落とさない。

\[
[F(G_i),F(G_j)]=F([G_i,G_j])
\]

はfermionic bilinearの恒等式である。固定N sectorで

\[
\|F(G)\|\le N\|G\|
\]

を使えば、展開から

\[
\|[D_i,D_j]\|\le
4|\lambda_i\lambda_j|N^3\|G_i\|\|G_j\|\|[G_i,G_j]\|
\]

が得られる。これは[D_i,D_j]をmany-body matrixとして作らない粗いboundの出発点で、**この式だけを新規成果にはしない**。

また、このsector boundをalgorithm全体へ使うなら、各primitiveとsampled operationが当該sectorを保つことを検証する。保存stateの粒子数だけで全演算にsector normを使わない。

pairwise小行列診断は通常のdense積ならO(L^2 n^3)程度で計算できる。一方、nested commutator全体やtightなcut評価も同じ計算量で得られるとは、まだ主張しない。

### 8.2 実際に研究すべき技術的な差

上の粗いboundを一般optimizerへ渡すだけなら、R3を名前だけ変えたものになりやすい。

B-Mの技術目標は、例えば次の形まで具体化する。

> 一つのgroupの更新頻度を落とす局所的な実行列変更について、変わらない演算を共通化し、変更が作るcommutatorの組合せをDF小行列で評価する。得られる追加誤差budgetと実装削減から、その変更を採用・棄却する。

必要なのは「共通部分を消せる」という代数だけではない。既知一般boundより情報を失わない評価、既知の評価法より安い計算、あるいは同じ情報予算で実用的に安定した選択を与える具体的な差が要る。

この技術的目標が初等式・既知法の直接利用だけに還元されるなら、new-method claimを立てず、application/benchmarkとしての価値を別に評価する。

### 8.3 basis費用との接続

DF blockの実装費用は回数だけでなく、隣接するbasis変換、対角演算、control phaseで変わる。U_i†U_j等の相対basisの融合は既知であり、現行sourceの既存改善を対照にも適用する。

saved wrapper totalから各fragmentのcostを推定して埋めない。まずcompileしない構造countを使う場合はproxyと表示し、最終候補では同じcompiler条件でactual full wrapperを比較する。

## 9. 最初に扱う実装可能な構成

H=A+B+R+cIとする。A/BはDF generatorの互いに素な集合で、Rは初期段階では現在のresidualのまま固定する。再factorization、BLISS、sampling分布変更、PF係数探索を同時に入れない。

A内部はnative symmetric S_A、Bはnative forward sweep F_Bとそのreverse sweepで扱う。例えば一macro-stepの理想構成を、作用順の規約を明示したうえで

\[
V_m(h)=F_B(h/2)[S_A(h/(2m))]^m\,
 e^{-ihR}\,[S_A(h/(2m))]^m F_B^{\rm rev}(h/2)
\]

とする。右端から作用する通常のoperator記法と、実装の時系列listを一致させる。S_Aはself-adjoint、F_Bのreverseとは同符号の逆順sweepでありadjointそのものではない。

tailを有限RTEへ置くと、同じq/r/Kの下ではtail total time・normalizationを固定したままmの寄与を調べられる。全signed time、corrected mean、physical sample mean、scalar relative phaseを保持する。

**m=1のnested構成と元のflat S2は、同じ演算列とは限らない。** 構成を入れ子にしただけで余計なdeterministic actionを増やしていないか、次の二つを必ず別に比較する。

- 同じnested構成のm=1：rate変更そのものの対照。
- 元の最良flat S2：nested構成を採用する価値の対照。

A/Bのどちらをfineに置くかも固定構造の一部であり、大小normだけで決めない。理想演算のzero/adjacent fusionは有限化前、sampled circuitのexact simplificationは有限化後に行う。finite polynomial同士を時間和へ置き換えることは別algorithmであり、compiler最適化として行わない。

## 10. 共通taskと情報の境界

最初のtaskは有限時間coherent signal

\[
z(T)=\langle\psi|e^{-iHT}|\psi\rangle
\]

を維持する。random unitary channelと平均振幅operatorを混同しない。

canonical finite RTEの補正後平均をM、normalizationをBとすると、sample mean operatorはM/B。独立occurrenceの期待値積を使う。今回新しい相関samplerは入れない。

同じshot規則では、各axisのbias b_aと数値guard u_aから

\[
s_a=\epsilon/\sqrt2-b_a-u_a>0,
\quad N_a=\left\lceil 2B^2\log(2/\alpha_a)/s_a^2\right\rceil,
\quad G=\sum_aN_a\overline C_a
\]

を共通に使う。これは当該十分shot式であって全推定器の最低shot数ではない。

情報は次を分離する。

| 段階 | 使ってよい情報 | 主張できること |
|---|---|---|
| I1設計 | DF係数、小行列、証明済みsector情報、生成列、明示cost model | instance-awareな構成規則 |
| I2開発診断 | 小系exact state/signal、実際のbias、exhaustive oracle | 何が設計を妨げるかの機構解析 |
| 評価のみ | 評価用exact signal、actual cost | 凍結した規則の成績 |

I2で決めた閾値を未使用条件へ移す場合はtraining手順を明記する。評価用のexact signalを設計へ戻してからoracle-freeと呼ばない。厳密boundが実用上緩い場合に、無言でheuristicへ切り替えない。

## 11. 比較baselineと公平性

| Baseline | 役割 |
|---|---|
| current flat S2 + canonical finite RTE | 基本の部分ランダム化構成。各精度でq/r/Kを公平に選ぶ |
| 同じnested構成m=1 | grouping／native overheadとrate変更を分離 |
| norm-based coalescing | 単純な構造基準との比較 |
| 一般commutator-based rate/group設計 | DF固有評価が必要かを比較。一般optimizerを弱くしない |
| 適用可能な既知near-integrable／SPRINT構成 | 高次数群と低次数群という既知解との比較 |
| exhaustive small-domain oracle | 評価専用。提案規則が取り逃す量と古典費用を測る |
| full deterministic／discard | partialそのものの優位を主張する場合に必要 |

最初のtoy段階では全method×全parameterを一括実行しない。まず同じ構成内の機構を検査し、方法の最終claimに必要な対照を後段に追加する。ただし比較していない範囲への優位は主張しない。

古いcoalescingのPauli項countや他論文のToffoli数を、今回のcompiled-RZへ直接並べない。task、precision、primitive、phase/control、入力情報、探索機会を揃える。完全なSPRINT再実装を第一pilotの必須条件にせず、適用する構成を先に特定する。

## 12. 次の作業を四段階に分ける

### BM-0：構成・差分を具体化する（直近）

新しい分子計算なし。今回の文献照合を再び抽象的な新規性議論へ戻さず、以下を作る。

1. B-F closure note：限定negative claim、原INCONCLUSIVEとR0 BF-Aの二層、再実行なし。
2. B-M数学仕様：native list、A/B/Rのscope、m/q、m=1/flat対応、finite semantics、cut/internal誤差分解。
3. DF-specific技術目標：何を既知法より改善するか。必要量、取得方法、古典計算量、証明済み／未証明を明記。
4. Model-familyとbaseline表：次節の反証試験を結果前に固定。
5. 小規模pilotの有限budget案：新規実行をせず、既存資産で必要な入力が揃うかを一覧化。

BM-0の出口は、一般的な『multirateを試す』ではなく、**何を変えた二つの構成を、何の改善原理で比較するかが書けていること**である。

### BM-1：構造機構を小型modelで検査する

分子結果を探す前に、DF形を保つmodel familyを使う案を推奨する。

初期案は4種類、rate m={1,2,4}、少数のqのみ。出発点の例は4 spin orbitals・2粒子sector（次元6）、最大6 fragmentである。単一粒子・involutionだけを選んでF(G)^2がidentityへ退化するtoyは避け、非自明な二体DF構造を保持する。これらは未採択のscope案で、toyの具体係数・時間・精度・seed・上限はBM-0で凍結する。新しいrandom algorithm／7-stage等は入れない。

| Family | 固定／変化させる構造 | 見ること |
|---|---|---|
| commuting control | 全G_iが同時対角化可能 | 不要な細分化を選ばない。融合後の実装が最良となるか |
| internal-error dominated | A内部の非可換性を維持し、A/B crossを小さくする | mで減らせる誤差が本当に現れるか |
| cut-error dominated | A内部を簡単に、A/B crossを大きくする | mを増やしても改善しない条件を誤認しないか |
| cost/adaptation contrast | 各fragment normを保ちながら相対basisを変える | 同じnorm重要度で区別できない場合に、DF構造が選択へ寄与するか |

G_j(eta)=exp(eta X)G_j(0)exp(-eta X)のようなunitary相似変換は、固有値を保持して相対構造を変えるための一例である。物理分子の証拠ではない。normを保ったからすべての実装costが同じとも仮定しない。

成功例だけを採るのでなく、全登録controlと失敗領域を保存する。既知法と同じ結果なら、その事実を残す。

BM-1の目的は一つの未確認の新規性を確定することではなく、**予定したDF-specific量が、一般norm量で見えない有用な選択に本当に結び付くか**を判別すること。

### BM-2：development実装・少数の実回路比較

BM-1で方法上の差が残った場合、既知H4 snapshotへ適用する。H4 1.00/1.30 Åはdevelopment／既参照でありblindではない。

最初はRの物理splitを固定し、deterministic側の二群とmだけを変更する。全subset、全order、全samplerを同時探索しない。候補生成規則を固定してから、各方式に同じq/r/K再最適化機会を与える。

actual compileへ進むのは少数の非支配候補と強い参照に限定する。proxyだけでcompiled利得を主張しない。量子costだけでなくclassical design time、memory、候補評価数も保存する。

### BM-3：独立条件・論文化

係数／group規則／cost policy／判定を固定したあと、未使用の構造条件一つ以上で検査する。新しい分子、geometry、DF表現のどれを選ぶかは、主claimの一般化軸で決める。

固定configuration transferと、固定protocolによる新instance再最適化は別の実験として呼ぶ。1サイズを増やしただけで漸近scalingを主張しない。

full QPE／chemical accuracy／fault-toleranceまでをこの第一成果の自動的な完成条件にしない。そこを主張する場合だけ、全round・state preparation・alias・synthesis等の別会計を追加する。

## 13. Pilotのscopeと資源管理

本書は科学実行を許可しない。BM-0ではコード・文献・保存textの範囲で具体化し、必要な次の計算だけを提案する。

科学pilotのbudgetは、過去BF-1の4000 cells／4hを無条件に流用しない。toy dimension、candidate数、q/r/K、actual wrapper数を決め、構造上の最悪数から上限を示す。wall/CPU予算は利用可能環境に結び付けて結果前に固定する。

推奨する初期scopeは二つのdeterministic tiers、固定R、m=1/2/4、4つの事前設計toy familiesまで。動作しない場合にtime、精度、familyをその場で追加しない。有限範囲に結論を出せないときはINCONCLUSIVEと理由を保存する。

重要な違いは、**計算数の上限よりも、何の未解決事項を判別するための上限なのか**を先に書くことである。test数・hash数・gate数を研究の完成率にしない。

## 14. 継続・縮小・終了条件

### 継続に値する場合

- 有効なnative sequenceを具体的に作れる。
- DF-specific評価または構成が、適切な既知対照と同じ情報・探索機会でも非自明な利益を持つ。
- accuracy、normalization、実装costのどこが利益を作るか説明できる。
- oracleの正解を毎回使わない運用形を明記できる、またはoracle-assistedという限定を正しく受け入れる。
- 少数toyだけでなくdevelopment／未使用条件に進める反証可能なclaimがある。

### 縮小・終了する場合

- 新規部分がcoalescing／SPRINT／一般誤差評価の直接適用だけである。
- rate変更の利益がnested overhead、basis再構成、RTE統計負担で消える。
- normを超える情報を用いても、既知の同情報対照と同じ判断・費用になる。
- 誤差の安価な評価が弱すぎ、毎回exact多体系参照なしでは動かないのに、I1 methodをclaimしたい状態になる。
- 閾値やscopeを何度も結果後に変更しないと有意な利得を作れない。

『splitが変わらなければ失敗』『finiteモデルがleadingより必ず優れなければ失敗』という旧条件は持ち込まない。新方法が同じsplitで回路を減らせば価値があり得る。

### negative resultを論文にする条件

単に『H4で効かなかった』では弱い。広い構造条件で失敗原因を特定する、既存実用法の前提が破れる範囲を示す、または明示したmethod classの限界を与える必要がある。そうした成果がなければ内部ノートで閉じることが適切であり、二本目の論文を必ず作る前提にしない。

## 15. 論文としての着地点

### 最小着地点：限定された構成・適用研究

DF-nativeな少数のfrequency-reduction構成について、共通taskと強い対照を用い、有効・不利領域を具体的に説明できる。独立methodの新規性が弱ければ、application／technical scopeに限定する。Aとは『方式固定の資源会計』対『実行列を変える構成比較』という差を維持する。

### 推奨目標：DF-specific schedule construction

> DF入力から計算可能な構造量を使い、実行頻度の変更による追加誤差と実装削減を評価する手順を構成し、その手順が作るcontrolled simulationを、既知coalescing／near-integrable対照より有用にできることを示す。

必要な最終要素は、方法の完全な仕様、取得情報と古典計算量、意味論／誤差の裏付け、同条件actual resource比較、未使用条件と失敗範囲である。

独立した新定理は唯一の道ではない。新しい実行可能な構成と広がりのある実証でも成立し得る。一方、既知要素の名前をつないだframeworkだけを寄与にしない。

### 発展着地点

schedule設計の可証な改善域、系サイズ／構造パラメータに対するcost則、幅広い化学系、最終energy推定への接続。ただしこれらを最小着地点と混同しない。

### 想定本文

1. 既知coalescing／hybridと、残る具体的な設計問題。
2. DF-native scheduleと情報モデル。
3. 内部誤差／cut誤差、追加誤差評価または構造診断。
4. 実行列作成手順とclassical cost。
5. model controls、development、actual compiled比較。
6. 独立条件と失敗領域。
7. 限界とscope。

BF-Aは動機・設計履歴として補足へ置く。新方法の正しさをBF-Aから証明したようには書かない。

## 16. 他の研究B候補との比較

| 方針 | 長所 | 新規性・実装上の問題 | 今回の推奨 |
|---|---|---|---|
| B-F条件追加 | 既存コードを再利用しやすい | 現pilotの主仮説には利得なし。別の具体機構なしの追加は弱い | 閉じる |
| B-S抽出分布変更 | 平均を保った別algorithmを作れる余地 | 一般cost-aware ISは既知、現標本のheadroomは小さい | 保留 |
| 単純multirate三層化 | current primitivesに近い | W1/W2/W3の重複が直接的 | これ自体をnew algorithmとして採択しない |
| **B-M：DF構造からの頻度削減構成** | 誤差と実装の両方へ接続できる | 特有の評価・構成を実際に作る必要。一般optimizerならR3の反復 | **構成可能性を調べる第一候補** |
| THRIFT／interaction picture | scale separationを活用 | 必要な部分Hamiltonianを安く実装できるかが先。既知hybridあり | 具体的なcheap subsolverが示せた場合の別案 |
| Trotter誤差のLCU補償 | 物理residual以外を補正できる | 補償法は既知。DF特有の改善が必要 | 自動移行しない |
| MPF・randomized ordering | 有力な既知algorithm class | 単なる差し替え比較は先行例が多い | 別の機構・scopeが必要 |

ここでB-Mを第一候補にするのは、既存資産と研究目的との接続が明確だからであり、新規性・性能の優越を既に確認したからではない。

## 17. A/B運用と保存

Aの固定原稿と証拠へBの新条件を戻さない。Bのold source、authorization、INCONCLUSIVE、R0 BF-Aを改変しない。

新しいB文書は`docs/tracks/algorithm_codesign/`、B固有コードは`src/trottertracks/algorithm_codesign/`に置く。実験は別namespaceと新identity。過去のconsumed registryを消さない。

新runを設計する際は、科学結果保存と補助表示の失敗を分離する。各candidate定義、arm origin、phase、primary結果を段階的にatomic保存し、cross-scoreのserialization failureでprimary全体を失わないようにする。標準Python型への正規化、round-trip test、synthetic end-to-end保存を先に確認する。

この保存改善は工学上の必要事項であり、研究Bの科学的新規性ではない。新runの仕様へ入れるもので、元BF-1を再実行する理由にはしない。

## 18. 直近にCodexへ反映する内容

ユーザーがこの方針を採用した場合、次の範囲へ落とす。

```text
基準handoff: 0da4d18acf3f5d32d1bc32c9661b667885bcf5f2
Bの証拠: 6d2645a09440f50e5b869ef42a1b73a1b625a1af
Aの参照: 4c23453c541700c6a41ba71fc5ec9323b53858d6

1. B-Fは現行条件でcloseする文書を追加。old statusやR0を上書きしない。
2. coalescing、SPRINT、Composite、Practical Estimation of Trotter Errorとの
   claim対応を本再設計の本文locatorに沿って整理する。
3. B-Mを『既知multirateの採用』ではなく、DF-specificな頻度削減構成の候補として記載する。
4. native A/B/R list、m=1とflatの差、cut/internal error、finite-RTE挿入を数学仕様へ落とす。
5. 計算が必要なDF量と、既存text/JSONで得られる量を分け、無い値をmissingにする。
6. 4つのmodel controlと、同情報・同構成・同budgetのbaselineを固定したpilot案を作る。
7. 承認された科学計算はまだない。NPZ、signal、compile、BF再実行へ進まない。
8. どの具体的技術差が残り、どこが既知法の直接利用かを報告してSTOP。
```

この段階で既知文献との重複が閉じ、実行可能な比較案ができたら、次は小型の構成試験へ進む。新規性を文章だけで完全証明するまで待つ運用にも、抽象的なauditを何巡もする運用にもしない。

## 19. 参照資料

### リポジトリ

[R1] B recovery SHA：`docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md`。

[R2] B recovery SHA：`docs/tracks/algorithm_codesign/bf0_prior_art_claim_matrix.md` §6。B-S sample-level diagnostic。

[R3] B recovery SHA：`docs/research_direction_pd_s1_posthoc.md`。

[R4] B recovery SHA：`docs/research/r3_prior_art_and_minimal_contract.md` §7。

[R5] handoff SHA：`docs/tracks/algorithm_codesign/inputs/fr_theorem_prior_art_review_71169d8.md`。これは過去レビューの公開snapshotで、今回その全証明を再検査したという意味ではない。

[R6] handoff SHA：`docs/tracks/algorithm_codesign/research_redesign_handoff_20261005.md`。P-A/P-B/P-C、構造結果、snapshot出典の入口。

[R7] A SHA：`docs/research/track_a_post_pm2_claim_evidence_map.md`。原稿scope、M1/PM1/M2/PM2の主張。

[R8] A SHA：`docs/research/pr2_post_m2_evidence_attribution.md`。同一candidate domain、same-R、欠測の分解、既存実装。

### 一次文献と今回の重点確認

[W1] D. Poulin et al., The Trotter Step Size Required for Accurate Quantum Simulation of Quantum Chemistry, arXiv:1406.4920; Quantum Information and Computation 15, 361–384 (2015), DOI 10.26421/QIC15.5-6-1。重点：§VI、Eq.(26)、§VI.A、結論のcoalescing＋sum-of-squares記述。arXiv本文を取得した。図表の数値を本計画へ転記していない。

[W2] P. A. M. Casares et al., Theory and practice of Trotter product formulas for quantum chemistry, arXiv:2606.30741v1 (2026)。重点：Fig.1（PDF画像確認）、§III.B–C、§IV。異なる次数と異なる更新頻度を同一methodとはせず、architecture-level overlapとして扱った。

[W3] M. Hagan and N. Wiebe, Composite Quantum Simulations, Quantum 7, 1181 (2023), DOI 10.22331/q-2023-11-14-1181, arXiv:2206.06409。重点：framework、coalescingとの関係、partition、高次Trotter/qDRIFT、discussion。diamond-distance channelと平均振幅のscopeを区別。

[W4] W. Maxwell et al., Practical Estimation of Trotter Error for Hamiltonian Simulation, arXiv:2606.30738v1 (2026)。重点：§III.B–C、§IV.A、compact BCHとgrouped commutator評価。asymptotic spectral解析を有限Tのcoherent-signal保証へ無条件転用しない。

[W5] Efficient and practical Hamiltonian simulation from time-dependent product formulas, Nature Communications (2025), DOI 10.1038/s41467-025-57580-5。THRIFTの必要primitive・fast-forwardable H0と実装scope。全algorithmのDF実装は本計画では未評価。

[W6] D. Cugini, T. A. Atif, Y. Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495 (2026)。whole-circuit costを考えたimportance sampling。B-Sの経験診断と母集団の保証を区別。

[W7] P. Zeng, J. Sun, L. Jiang, Q. Zhao, Simple and High-Precision Hamiltonian Simulation by Compensating Trotter Error with Linear Combination of Unitary Operations, PRX Quantum 6, 010359 (2025)。今回は方法のscope-level比較であり、全定理のDF転用可能性を審査したものではない。

[W8] S. Zhuk, N. F. Robertson, S. Bravyi, Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation, Physical Review Research 6, 033309 (2024), arXiv:2306.12569。動的MPF最適化という既知方向の確認。今回の主計画には組み込まない。

関連するpartition error、optimized PF、fermionic low-rank implementationの既存文献は、repoのBF-0/R3 matrixとW2/W4の引用網から追跡する。最終new-method claimを作る段階では、その具体式と最接近式を再照合する。

---

**まとめ**：B-Fは限定negativeとして閉じる。multi-rateの一般発想を新規と扱う提案は修正する。次は、DF構造から実行頻度削減を判断し実行列を構成する具体的な技術を、既知法と比較して作れるかを研究する。その技術差が作れない場合は、無理に第二論文へせずscopeを縮小または終了する。
