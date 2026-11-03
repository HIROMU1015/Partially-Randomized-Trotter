# PRリソース評価：論文の位置づけ・主張・必要検証に関するGPT科学レビュー

- レビュー日：2026-10-19（JST）
- 担当：GPT
- 区分：Level 1。論文化方針、先行研究との差別化、中心的主張、追加検証の必要性を評価。
- 開始承認：ユーザーの「レビューを開始して　論文として立ち位置についてよく考えて」。
- 対象：`HIROMU1015/Partially-Randomized-Trotter` のTrack A／PR資源評価。
- 主な証拠基点：最終技術監査 `bf8c19057fd82c900c6234152e152e79cb52e707`、新A総費用比較 `64c85f7a833b5950ef2ccae623708b73081edcd6`。
- 補足更新：H6高次PF独立確認の完了結果 `6e197973d4d4183d4e27b1b8cc95cb02f311bb92`、H8独立確認の承認済み起動snapshot `d21647d07bc86343b292b1e1154ba05f887b1593`。
- 実施：公開報告・静的監査・Claim–Evidence Matrix・添付Claudeレビューの読解、必要な固定GitHub資料の取得、外部一次文献の調査、数理的・科学的考察。
- 未実施：新signal、SCF/DF/state生成、sampling、回路生成・compile、保存値の新規再会計・再ランキング・fit・bootstrap、実科学コードの実行、GitHub変更、既存job操作。
- 本書の判断は研究戦略の提案であり、研究方針の正式採択、新規数値解析の実行承認、既存契約・grant・STOPの変更ではない。

本文では、【保存事実】、【文献事実】、【数理】、【判断】、【未確認】を区別する。出典[R1]等と[L1]等は末尾に示す。数値表は保存報告の転記であり、本レビューで再計算した結果ではない。

---

## 0. 最終判断

**条件を明記した数値比較論文として執筆を進めることを推奨する。ただし、新PRアルゴリズム、PRの一般的優位性、精度窓の普遍的法則、耐故障計算の資源削減を示す論文としては位置づけない。**

最も適切な中心課題は次である。

> 同じ有限時間Hadamard-test signalを、同じ宣言精度・shot会計で推定するとき、部分ランダム化と決定論的打切りの相対compiled資源評価は、両側の候補探索範囲とbias会計によって、どの程度変わるか。

これは「PRが勝つか」だけを問うのではなく、**精度を揃えた資源比較の結論を、どの実装・調整・会計条件の下で読めるかを定量的に調べる研究**である。

判断を三つに分ける。

| 判断対象 | 結論 |
|---|---|
| 既存結果に基づく原稿着手 | GO。Methods・既存Results・Limitationsを執筆できる |
| 新しい大規模探索・分子拡張・希少event大量sampling | 現時点の第一優先ではない |
| 投稿版データと主張の最終固定 | まだ完了とはしない。図表用の原値結合・必要な定量機構表を一回の限定処理で完成し、採用する承認済み確認runの終端結果を接続する |

**「執筆開始に十分」と「現在のまま投稿用の科学的主張がすべて完成」は異なる。** 一方、未確認のL2、非Clifford数、H6非相殺full-domainをすべて解消しなければ論文化できない、という判断でもない。

今回の重要な修正は、先行研究に打切り比較が存在することを踏まえて、新規性の言い方を狭く正確にすることである。

---

## 1. 何を評価した研究なのか

### 1.1 対象と指標

【保存事実】主panelは、linear H4/H6/H8、1 Å、STO-3G、保存DF Hamiltonian・状態、T=0.8、状態準備費用P=0である。H4はlegacy固定rank12、H6/H8はtol-only系列rank19/28。主指標はQiskit 1.3.0、`rz/sx/x/cx`、opt1、seed17、backend/couplingなしで得たfull Hadamard wrapperのRZ命令数と、保存された解析的shot数を結合した費用である。[R1–R3]

推定対象は

\[
z(T)=\langle\psi|e^{-iH_{\mathrm{DF}}T}|\psi\rangle
\]

という複素signalであり、現在のepsilonはこのsignalの許容誤差を表す。元の電子HamiltonianとのDF誤差、energy推定の全round、実際の量子状態準備、回転合成・誤り訂正・routingの費用まで含めたものではない。[R1,R2]

【判断】最も適切な分野は、量子Hamiltonian simulation／resource estimationの**task-specific numerical benchmark**である。量子化学Hamiltonianを対象にしているが、主たる新知見は分子の化学的性質ではない。

### 1.2 小さい系を扱う意味と限界

【判断】H4/H6/H8の小ささは、精密なreferenceとの照合、bias会計の分解、compiler出力の確認が可能という利点を持つ。大規模系への優位性を示すためのサイズ範囲ではない。

正確なreferenceを古典計算で用意して設定を選ぶこと自体は、数値benchmarkとして正当である。しかし、そのreferenceを実用段階でも安価に得られるという主張や、referenceによる調整費用を含むend-to-end優位性にはならない。「oracle-calibrated benchmark」であることをMethodsに明記すべきである。

H4/H6/H8はDF政策も同一ではないので、単純に三点を結んで分子サイズのscaling lawを主張してはいけない。

---

## 2. 先行研究との位置づけ：今回もっとも重要な再評価

### 2.1 「既存PR研究は打切りを考慮していない」は採用しない

【文献事実】GüntherらのPR論文の公開arXiv v2、付録D.2は、PRで乱択する小さい項を除外した場合のenergy誤差と、保持項数を変えた打切り誤差を検討している。その箇所はTrotter誤差を含めない検討であることも明記している。[L1]

したがって、「PRと決定論的打切りの比較を初めて考えた」「先行研究はfull PFしか比較しなかった」とする新規性説明は不適切である。

【判断】本研究の差別化候補は、**打切り誤差だけの比較から、同一signalの総誤差・shot負担・実compiled単価まで接続した比較へ進めたこと**である。これは先行研究の問題設定を精密化する方向であり、その大規模energy推定結果を反証する研究ではない。

### 2.2 近い文献との対応

| 一次文献 | 重なる部分 | 本研究で明確に分ける点 |
|---|---|---|
| Güntherら：PR、arXiv:2503.05647v2 | PR、Coulomb tensor factorization、phase-estimation資源、打切り誤差の検討 | 本研究は小さい保存DF系の単一signalを同一会計で比較。付録D.2にないjointなTrotter/shot/compiled比較を具体化することが差別化候補 |
| Hagan–Wiebe：Composite Quantum Simulations、Quantum 7,1181 | Trotter-SuzukiとqDRIFTの合成、分割、時間・精度に依存する費用trade-off | 本研究の有限RTE・signal精度・compiled費用とは目的関数が異なる。合成原理や「中間領域で有利」という一般的発想は新規性にしない |
| Kanasugiら：arXiv:2603.22778v2 | PRを使ったphase-estimation資源、打切りを持つ決定論対照 | 同論文は小係数総重量の閾値、S2+qDRIFT、early-FTでの資源を扱う。本研究のsignal/RTE/DF/compiled-RZと数値比を直接比較しない |
| Sugisakiら：arXiv:2608.04481 | 水素鎖、Hamiltonian打切り、gate-resource benchmark | 同研究のgate比較は一段のfirst-order Trotter・回路最適化なし。本研究はmatched signal accuracyとshot込みcompiled費用を扱う |

文献事実は[L1–L4]を参照。Sugisakiらは2026年8月のpreprintとして扱い、査読済みの定理・scaling認定とはしない。

【判断】「水素鎖で打切りにより回路が安くなる」「量子化学にPRを適用する」「Trotterと乱択を混ぜる」だけでは独立した新規性は弱い。以下の結合に研究価値を置く。

- 同一のtarget signalとerror/shot契約。
- 両方式について保存された候補domainの段階的な拡張。
- compiled費用、normalization、bias headroomを区別する定量説明。
- 実測／model、探索／確認、欠測／不適格を区別した再利用可能な比較データ。

この組合せと完全に同一の先行benchmarkが存在しないことまでは、今回の重点調査で証明していない。「first」を付けず、具体的な差分を述べる方針が適切である。

### 2.3 新規性の強さの評価

【判断】

| 論文の名目 | 現在の評価 |
|---|---|
| 新しいPRアルゴリズムの提案 | その主張を支える新構成・理論は今回のTrack Aでは未確立 |
| PRの一般的優位性・大規模量子化学の優位 | 対象・指標・保証が不足 |
| 精密な数値比較／資源benchmark | 成立を目指せる。主要な方向 |
| 資源比較の方法論的研究 | 成立候補。ただし、単なる注意事項ではなく、定量的な比較・反例・再現可能な成果物が必要 |
| 厳密な希少event費用上界の論文 | L2・数値結合が未完で、現在の主成果にできない |
| ソフトウェア論文 | コード量だけでは不足。独立に再利用できるAPI・例・公開範囲の整備が別に必要 |

「対照を強化すると最小費用が下がる」という自明な事実そのものを、新規性の中心にしない。

---

## 3. 保存結果から支持される中心的成果

### 3.1 現在の主な点結果

【保存事実】通常signed bias、allowance倍率1、各系の強化決定論集合に対するPR削減率。正はPRの保存点費用が小さい。[R2]

| 系・cohort | eps=.05 | .01 | .005 | .001 | .0005 |
|---|---:|---:|---:|---:|---:|
| H6 旧PR n32 | -13.2901% | 30.8905% | 34.9653% | 31.1677% | 10.3257% |
| H6 新A探索n8 | -76.8195% | 15.4763% | 29.2796% | 30.8980% | 9.9766% |
| H8 旧PR n32 | -8.0230% | 33.2373% | 31.4549% | 23.1282% | -50.8125% |
| H8 新A探索n8 | 3.1084% | 31.8387% | 29.9926% | 17.9519% | 3.1178% |

旧n32と新n8を合算した表ではない。各cohort内で選ばれた保存点の転記であり、母平均の最良や、cohortをまたぐ選択後の同時保証ではない。

【判断】H6/H8のeps=.01/.005/.001について旧PRの点削減が残ることは使える。ただし「新Aも一律20–35%」「その間の全連続epsilonで成立」「中間だから統計的に保証された」とはしない。

H8両端は新Aで点の符号が変わった。これは、旧PRだけの精度範囲から「PRそのものが端で必ず不利」とする結論を撤回する根拠になる。一方、新PRの母平均優位を確定する証拠ではない。

### 3.2 中心成果を三つにまとめる

【判断】論文の中心は次の三つとする。

**C-A：比較domainを変えると、保存された相対費用の大きさと一部の順位が変わる。**

単に最小値が下がることではなく、どのprefix・次数・q/r scheduleに置き換わり、どの誤差・回路要因でその変化が起きたかを示す。現在の証拠は点比較であり、domain外の最適性には進めない。

**C-B：回路単価だけでは、同じsignal精度での測定込み費用を予測できない。**

実際の会計を、回路側とshot側へ明確に分ける。ただし、費用の積・和の恒等式自体は既知であり、新規性は具体的な量的trade-offと順位の変化にある。正式な機構図には保存値の一回の集計が残る。

**C-C：signed biasと経験的非相殺会計で、資源比較の結論が変わる。**

H8は主たる定量例にできる。H6は旧B0I12候補の成分欠測を除いたsubset値に限り、full-domainの非相殺優位を主張しない。

### 3.3 使わない中心命題

以下は現時点で採用しない。

- PRは最良の決定論的Hamiltonian simulationを上回る。
- PRは中間精度の連続した窓でのみ有利である。
- H4/H6/H8の両端はすべて同等である。
- 中間precisionのwinnerは探索domainの内部にあるから頑健である。
- PRの1回路は常に決定論側より小さい。
- 小さい標本SEにより、未観測の全eventを含む平均費用が保証された。
- Cliffordを除いた回転数やFT T-resourceでも同じ結論になる。

これらの制限は[R1–R5]と後述の数理的区別による。

---

## 4. 論文の論理を整えるための数理的整理

### 4.1 比較対象の拡張は、それ自体では発見ではない

【数理】同一の目的関数と適格性の下で、候補集合SがS'の部分集合なら、

\[
\min_{j\in S'}G_j\le \min_{j\in S}G_j.
\]

したがって「より多く探索したら、見つかった対照が安くなった」だけでは弱い。研究として必要なのは、費用改善の規模、候補の構造、誤差会計との相互作用、実際に順位が変わるケースを示すことである。

また、保存費用がmodelや有限標本平均なら、この最小値もその推定値に対する最小である。未確定候補を勝手に落とすと、集合全体の最小とは異なる。

候補数を両側で同数にすれば公平になるわけではない。まず同じtarget・精度・費用範囲・情報利用を揃え、その後に各方式で何を探索したかを示す。未探索自由度は限界として残す。

### 4.2 正確な測定込み会計と機構分解

方法m、軸a∈{real,imag}について、保存shotをN_(m,a)、trajectory iのcompiled費用をC_(m,a,i)とすると、

\[
G_{m,i}=N_{m,\mathrm{real}}C_{m,\mathrm{real},i}
       +N_{m,\mathrm{imag}}C_{m,\mathrm{imag},i},
\qquad
\widehat G_m=\frac1{n_m}\sum_iG_{m,i}.
\]

これは現行のpaired費用会計である。[R2,R3,R6]

【数理】全shot数N_mとshot重み付き実効単価を

\[
N_m=\sum_aN_{m,a},\qquad C_m^{\mathrm{eff}}=\widehat G_m/N_m
\]

と定義すれば、

\[
\frac{\widehat G_{\mathrm{PR}}}{\widehat G_{\mathrm{PF}}}
=
\frac{N_{\mathrm{PR}}}{N_{\mathrm{PF}}}
\frac{C_{\mathrm{PR}}^{\mathrm{eff}}}{C_{\mathrm{PF}}^{\mathrm{eff}}}
\]

は整数丸めを含めて成立する。ただし実効単価の重みは方式ごとに異なる。図では軸別のNとCも示し、この定義を隠さない。

共通の軸別alphaで、ceil・最小shot等を無視できる範囲なら、

\[
\frac{G_{\mathrm{PR},a}}{G_{\mathrm{PF},a}}
\approx
\frac{\bar C_{\mathrm{PR},a}}{\bar C_{\mathrm{PF},a}}
\frac{B_{\mathrm{PR},a}^{2}}{B_{\mathrm{PF},a}^{2}}
\left(\frac{h_{\mathrm{PF},a}}{h_{\mathrm{PR},a}}\right)^2,
\qquad h_{m,a}=\varepsilon/\sqrt2-b_{m,a}-u_{m,a}.
\]

ここでBはshot式に用いる上界、bは会計で採用する非負のbias量である。signed成分そのものを負の許容幅として差し引かない。

**二軸を単純平均した回路比と、一つのheadroom比を掛けた式は一般には正確な分解ではない。** 既存レビューの説明用推定を、そのまま実測機構の図にしない。

【判断】この正確な保存値結合を本文の機構として用いることは有用。ただし、式そのものを新定理として売らない。また、N・C・B・hが同時に変化するwinner比較から、単一パラメータの純粋な因果効果を認定しない。

### 4.3 共通Hoeffding規則と、実際の最小shot数の違い

【数理】理想Hadamard測定の補正済み1-shot量がX=B_corr Y、Y∈{−1,+1}、E[X]=z_aであれば、

\[
\operatorname{Var}(X)=B_{\mathrm{corr}}^2-z_a^2.
\]

Hoeffding会計は範囲上界を用いるため、この実分散を直接最適化した必要shot数ではない。PR/PFに同じ形式の十分条件を適用しても、十分条件の余裕が両方法で同じ倍率になるとは限らない。

【判断】「同じHoeffding budgetの下での比較」は正当な比較仕様として採用できる。しかし「測定戦略を最適化した真の最小資源」を評価したとは書かない。Bernstein等へ変えれば必ずPRが有利になる、または不利になるという方向も、この式だけからは決まらない。

この点を明記すれば、alpha検証を繰り返したり、今から別shot規則を主研究に追加したりすることを投稿の必須条件にはしなくてよい。

### 4.4 signed相殺と選択バイアスを分ける

【数理】同じ対象に対する正確な分解がd_total=d_discard+d_PFなら、|d_total|と|d_discard|+|d_PF|は違う。前者の相殺は、その時刻・状態で実在する効果であり、それだけで不正な比較ではない。

【判断】exact referenceでその相殺を利用できる候補を選ぶことは、理想化されたoracle調整である。一方、統計ノイズにより費用の低い候補が過大選択される現象は統計的winner's curseである。両者を一語で同一視しない。

両会計を並べるのはよいが、その差を信頼区間や真値が必ず入る帯として描かない。経験的非相殺も、保存reference・数値allowance・成分分解の正当性を超える厳密証明書ではない。

---

## 5. Claudeレビューから採用すること・採用しないこと

添付された包括レビューと追加差分レビューは、研究の絞り込みと条件開示に役立つ。一方、最新監査・一次文献により以下を修正する。

| Claudeの論点 | 本レビューの判断 |
|---|---|
| PR一般優位ではなく、対照・会計を主題にする | 採用。ただし単なる最小値の単調性を新規性にしない |
| 両端ではPFが勝つ | H8旧候補の保存結果に限る。新Aでは点の符号が変わった |
| 両端は「ほぼ同等」で固定し、どんな確認結果でも結論を維持 | 採用しない。標本上約3%差という記述と、母平均の同等性は別。結果を見て記述を更新する |
| 中間winnerがdomain内部だから頑健 | 採用しない。最新B監査に境界例がある |
| r8は未探索 | 訂正する。新A登録にH4/H6/H8のr8候補がある。ただし完全なr格子ではない |
| PRの1回路は常に小さい | 採用しない。軸別実測で個別に示す必要がある |
| 選択楽観1.77SEは最悪上限 | 採用しない。特定の等分散正規モデルでの期待値例は、一般の確率上界ではない |
| Tier Rで中間の頑健性を閉じられる | まだ採用しない。L2・数値結合・基準平均の不確かさが残る |
| 高次PF独立確認は研究上無価値 | 採用しない。固定RTE配分での順位再現という証拠はある。ただし主Track Aの最適化比較とは別 |
| 大規模な新しい検証を既定にしない | 採用。必要性を中心claimから逆算する |

【保存事実】新AのH8 p7/K4の登録scheduleはq4:r8、q8:r8、q16:r4であり、全座標を固定したr4のq4/8/16走査ではない。H6新Aも疎集合である。したがって「qを倍にしたことだけの効果」と断定する図は避ける。[R4]

また、Qiskit sourceによりL1が確認されたことは、最終RZ数の単調減少や、任意のcompiler profileに対する同じ性質を意味しない。[R5]

---

## 6. 希少event、有限標本、model単価をどう扱えば論文化できるか

### 6.1 推定量と標本記述を混同しない

【数理】母平均の測定込み費用をμ=E[G]とすると、有限標本平均はその推定量である。候補固定前の探索と、候補固定後の確認を分けることは有用だが、それだけで未観測層の影響や全候補最適性が解消するわけではない。

有限K・有限Rの本研究で「希少eventがある」ことを、そのまま数学的に無限のtailや無限分散と呼ばない。問題は、**該当する低頻度層の費用と、利用する上界が現在の証拠に結合されていないこと**である。

【判断】記述的benchmarkとして標本平均・SE・観測された次数を示すことは可能。ただし「標本SEが小さいので、中間精度の平均優位は変わり得ない」は主張しない。bootstrapも保存標本に存在しない層を回収しない。[R7]

### 6.2 上界を追加する場合に本当に必要なもの

【数理】層sによる分解は

\[
\mu=\pi_0\mu_0+\sum_{s\ne0}\pi_s\mu_s.
\]

もし母平均の上界を主張するなら、層質量と費用の有効な評価だけでなく、**μ0を有限標本平均に置き換えた不確かさ**も扱う必要がある。希少層の構造上界だけで、通常層の推定誤差・選択効果が消えるわけではない。

ClaudeのTier Rは、経路ごとにC(ω)≤T(ω)を持つ方法の候補である。しかし最新監査は、実translatorの全native展開表・library閉包、期待値上界の評価を未完了としている。T/C≈3–4等を実測値や保証値に置き換えてはならない。[R5]

Tier Tの二変数境界は、未知費用がどの領域なら結論が変わるかを示す感度解析である。一つの軸切片が大きいことだけでは、他の未知層を含む全領域を排除できない。[R7]

【判断】本論文が「保証付きμ優位」を主張しないなら、これらをすべて証明することは必須ではない。ただし、その代わりに**全supportの頑健性を論文の結論に含めない**。追加の希少samplingを数百万本へ膨らませる方針は推奨しない。

### 6.3 PFモデル費用の論理

【保存事実】H8主panelの最新通常factor1で選択されたPF単価は直接実測。H6中間点の選択PF単価にはmodelが残る。登録集合全体については、全候補を直接compileしたわけではない。[R3,R4]

【数理】ある実測PFが固定PRより安ければ、「そのPRが登録PFすべてより安い」は否定される。逆に、ある実測PFよりPRが安いことだけでは、PRがPF集合の真の最良より安いとは言えない。modelだけの未選択候補もあり得る。

【判断】主表は実測・model・missingを明示する。追加実測を全136/184候補へ広げる必要はないが、「all-candidates-measured optimum」とは書かない。既存のmodel検証誤差を、全候補共通の数学的誤差上限としない。

### 6.4 H6非相殺の欠測

【保存事実】旧B0I12候補の`eligible:null`が旧Dで未確定一覧に入らなかったことが、最終監査で確認された。新A・alphaでは同じ未解決をfull-domainのnullとして保持している。[R1,R8]

【判断】旧Dの有限値を全domain結果とする表現は撤回する。subset結果として残すか、主表ではH6非相殺を未確定にする。論文用の抽出経路では三値の状態を明示して扱い、将来同じ検出漏れが伝播しないことを人工入力で確認する。

過去のraw表の上書きや、欠測の全回収・全再会計を、通常bias中心の論文化の必須条件にはしない。

---

## 7. 独立確認2系列の現在の扱い

### 7.1 H6高次PF：完了済み結果を保全し、対象内の補足結果にする

【保存事実】`6e197973…`では、103本・206軸の新標本で固定2比較の点順位を再現した。Yoshida4/q2はS2/q8より3.7481%低費用。Yoshida4/q4はq2より1.7507%高費用である。二つの百分率は分母が異なるため、後者をそのまま「q2はq4より1.7507%低い」とは転記しない。[R6]

単一2次eventはYoshida4/q2に5本、q4に1本観測された。未観測なのはs4・複数2次eventであり、「あらゆる希少eventがゼロだった」と一般化しない。

【判断】固定RTE32/outer、ordinary-controlのPR比較として有効な独立再現証拠である。一方、主Track Aのr2/4/8・symmetric-directional系列と直接費用を混ぜない。新しい手法全般の優位や最適なrまで含む比較の代用にはしない。追加samplingは当面不要。

### 7.2 H8新PR：未実行という以前の認識を更新する

【保存事実】今回GitHubで、`track-a-h8-new-pr-confirmation-execution-20261018`、commit `d21647d07bc86343b292b1e1154ba05f887b1593` を取得した。これは2026-10-18 23:35:44 JSTの**承認済み起動snapshot、RUNNING_NOT_COMPLETED**である。科学SOURCEは`1549b6f891a15ff48379996c34b344118ba2606c`、2候補各48本・192軸の契約である。[R9]

本レビューでは完了結果を取得していない。過去snapshotを理由に現在も必ず稼働していると断定しない。

【判断】既に承認・起動された科学jobを、後のレビュー提案だけで停止・縮小・再承認し直すことはしない。元契約に従って得られる完了またはSTOP証拠を受け取り、採用する場合は符号にかかわらず報告する。新しい標本追加・別candidateへの自動変更は行わない。

H8の点差を本文で強い結果として使うなら、この承認済み確認の終端結果を反映する意義はある。確認結果がまだない間もMethodsと既存Resultsの執筆は開始できる。

副評価ではp8/q2の固定PF p11/q1が非相殺で不適格という記録がある。通常biasの確認結果を、別PFを選び直した非相殺の確認へ移送しない。[R9]

---

## 8. 投稿前の必要作業と、不要な拡張

### 8.1 推奨路線に必須の完成作業

**一つの「論文用保存データ統合・機構分析」パッケージにまとめる。A〜Eの静的監査をもう一巡させない。**

| 作業 | 理由 | 科学計算の別承認 |
|---|---|---|
| 採用図表の行・元commit・cohort・metric・単価区分・欠測を結合する | 読者が同じ表を再現できることが必要 | 単純転記・属性付与は文書/技術作業。新最小化や新derived値は別扱い |
| 保存N・軸別C・B・bias/headroomから定量機構表を作る | 「なぜ変わるか」を推定値でなく実記録に接続する。本レビューが推奨する機構claimには必要 | 必要。保存値の限定数値解析として一回固定 |
| 旧Dのsubset/full-domainを区別した抽出・表示にする | 既知の`eligible:null`検出漏れを論文へ持ち込まない | 人工テスト・新表示の静的修正で済む範囲と、新再会計を分離 |
| 採用するH8承認済み確認runの終端状態を接続する | 実行済み/未完了/失敗を隠さず、独立確認の強さを正確に示す | 元承認範囲の回収。新samplingを追加しない |
| 文献との差分表と再現可能な公開データ入口を作る | 既知の打切り検討との差分、第三者の数値追跡を成立させる | 新科学計算不要 |

定量機構表を本文から完全に外す路線なら、その新計算は論理的には必須でない。ただし、その場合は貢献が記述的な比較結果に寄り、論文の説得力も下がる。本レビューは機構表を作る路線を推奨する。

### 8.2 中心claimに依存する任意の検証

| 検証 | 必須になる主張 | 現在の推奨 |
|---|---|---|
| L2閉包＋Tier R数値上界＋通常層の推定不確かさ | 全supportを含むμの上界・頑健な母平均優位 | 現論文では保留し、保証しない旨を明記 |
| final回路の非Clifford角再集計 | 非Clifford指標でも結論が維持される | 任意。compile前QPYをfinalと読み替えない |
| H6旧12候補のsigned分解取得・再会計 | 全登録集合の非相殺比較 | 通常bias中心なら必須でない |
| model-only PFの追加直接compile | 全候補実測での最良・model非依存順位 | 全格子実測は不要。主張に応じて対象を限定 |
| 新しい独立確認・追加標本 | 特定の追加候補のselection-independentな再現性 | 既存2系列以外を自動追加しない |
| q32、PF q16、高次PRのr最適化、新geometry/H9以上 | domain外の最適性や一般化、別アルゴリズム主張 | 今回の論文化の必須条件にしない |
| 別shot規則、RPE全round、状態準備、FT synthesis | end-to-end・最適な測定戦略・FT資源 | 別研究 |

「不要」とは、別の主張を追う場合にも価値がないという意味ではない。今回選ぶ論文の目的には必須でないという意味である。

### 8.3 完了と凍結の基準

- 科学条件と参照する結果cutoffが固定されている。
- 主図表が元の保存点へ再現可能に結合され、新規計算した列は別結果として記録される。
- 中心claimを支える機構表を作り、作れない箇所を推定で補っていない。
- 既知欠測が主表に明示され、H6全domain非相殺が有限値に見えない。
- 探索n8・旧確認union・H6高次PF確認・H8確認の役割が混ざっていない。
- 希少eventや非Cliffordの未保証を、保証済みの表現で隠していない。
- 採用する実行済み確認の結果・STOPを含めたfreezeになっている。

この条件が揃えば、追加candidateを増やすことより、原稿を完成させる。各技術ファイルの更新ごとに新しいLevel 1レビューへ戻す必要はない。

---

## 9. 推奨する原稿の構成と表現

### 9.1 推奨タイトル

**Search-space and error-budget dependence of compiled resource estimates for partially randomized Hamiltonian simulation**

日本語の趣旨：

**部分ランダム化Hamiltonian simulationのcompiled資源評価における、探索範囲と誤差予算への依存性**

別案：

**Accuracy-matched compiled-resource benchmarks of partially randomized and truncated product formulas**

タイトルに「optimal」「guaranteed advantage」「fault-tolerant advantage」「universal precision window」は用いない。

### 9.2 推奨する中心的主張の文案

> 保存された小規模水素鎖のDF Hamiltonianについて、有限時間Hadamard-test signalを同一の宣言精度とshot会計で推定するcompiled資源を比較した。決定論的打切り対照とPR候補の探索範囲をそれぞれ拡張すると、相対費用の大きさだけでなく一部条件の点順位も変化した。登録された中間精度点ではPRの低い点費用が残る一方、端点の結論は候補集合・bias会計・有限標本に依存する。回路単価とshot負担を区別し、各結論の実測／model・探索／確認・欠測の範囲を明示する。

これは「PRが勝つこと」を先に決めた文章ではない。新しい承認済み確認結果が得られた場合は、その結果を反映して更新する。

### 9.3 主図を絞る

| 図表 | 役割 | 注意 |
|---|---|---|
| 主図1：両側の探索拡張と相対費用 | 対照強化だけでなくPR側の拡張も並べる | 新n8と旧n32を別系列で表示。両者の最小包絡はさらに選択した量であると明示 |
| 主図2：軸別または厳密実効単価による回路/shot分解 | 費用変化を具体的に説明 | 保存実単価とshotを用いる。レビュー中の線形モデル推定は使わない |
| 主図3：通常signed／経験的非相殺の比較 | 会計への依存性 | confidence bandにしない。H6 full-domain欠測を見える形にする |
| 主表：物理系・候補domain・cohort・compiler・費用来歴 | 再現性と公平性を読者が判断できるようにする | domainの境界属性だけで大域最適性や因果を認定しない |

H4はDF政策と対照subsetが異なる補助ケース。H6/H8を中心にする。H6高次PF、理論count/compiled差、alpha検証は必要な分だけ補足へ。betaの旧RPE監査、Track B、QC系列、実行運用の全履歴を本文へ詰め込まない。

### 9.4 読者に残す一般化可能な内容

本研究で一般化するのは削減率そのものではなく、次の比較手順である。

1. target observableと精度の意味を先に固定する。
2. 双方の誤差調整・候補domain・oracle情報を開示する。
3. 回路単価とshotを分け、同じ会計で結合する。
4. 推定値・実測値・model値・欠測を区別する。
5. winnerの点結果と、選択後の統計的保証を区別する。

これは一つ一つが新原理という意味ではない。この手順を分子DF回路に具体化した定量結果と、再利用できるデータが論文の実質的な成果になる。

---

## 10. 投稿先をどう考えるか

【判断】最初に分野・貢献の種類を固定し、投稿先に合わせて実験を際限なく増やさない。

- **Quantum Information Processing**：量子計算のアルゴリズム・protocolを扱うscopeに、精密な数値比較研究として合わせやすい候補。これは採択容易性を意味しない。[J1]
- **Physical Review A**：量子情報・量子scienceのspecificな有意義な貢献を要求する。機構の定量説明と、既存PR/打切り研究を超えて得られる知見を強く示せるなら検討対象。ただし現在の監査資料の量だけではその水準を判断できない。[J2]
- **Quantum**：数値論文も対象だが、非常に大きな技術的・概念的貢献を要求し、正しくてもincrementalな仕事は基準未満と明記される。現在のTrack Aをそのまま高い確率で通る案件とは評価しない。候補数を増やすだけでこの問題は解決しない。[J3]

したがって、本レビューでは**QIP等に向く焦点の絞られた比較論文を基本路線とし、初稿の定量的説明の強さを見てPRAを検討する**。特定誌への採択、査読者の評価、引用数を保証しない。

新しい理論、実用selector、FT合成まで含む大きな主張を目標にするなら、別の研究契約と目的が必要であり、今回の論文をそのために無期限延期する方針は採らない。

---

## 11. 前のGPT判断からの変更

1. 「データは概ね揃っている」は、原稿着手について維持する。ただし、投稿版の定量機構図が未完成であることを明確にした。
2. 「PR側を拡張したら精度窓が変わった」は、保存点の記述として維持する。連続窓や母平均での符号変化としては採用しない。
3. 先行研究に最適化した打切り検討がほぼない、という説明は厳密化する。少なくともPR原論文D.2に明示的な打切り誤差検討があり、関連水素鎖benchmarkも存在する。
4. H8独立確認を未実行・今から採否判断とした以前の状態を更新する。承認済み起動snapshotを確認した。レビューの推奨で既存runを停止しない。
5. H6高次PF独立確認の結果は無価値ではない。3.7481%とq4側増1.7507%の固定条件の再現として保全する。主Track Aの全最適化比較へは移送しない。
6. 厳密なrare保証や非Clifford数を必須化して技術監査を繰り返す方針は採らない。論文でそれらを主張しないなら、明示的な限界で区切る。

---

## 12. 次の担当・承認・STOP

**ユーザー**：本書の「同一signal精度での比較domain・会計依存性を中心にした数値比較論文」という着地点を採択するか判断する。

**Codex**：採択後、論文用データの転記・状態表示・元行結合・最小再現手順と、必要な定量機構表の保存値計算を一括scopeへする。後者の数値処理は、対象・元入力・出力・受入基準を固定して別途実行承認を得る。新samplingや全格子再探索へ拡張しない。

**既存のH8承認済みrun**：元契約どおりの終端証拠を受け取る。現在の未取得結果を推測せず、既存jobへ手を加えない。H6完了系列のSTOPは保持する。

**GPT**：新規の中心claim変更、証拠と主張の実質的矛盾、未知の科学的問題が出た場合に限り再判断する。ファイル整理や人工テストPASSのたびに包括レビューを繰り返す必要はない。

今回のレビューは完了。新しい科学実行のgrant発行、既存結果の変更、論文の正式採択を代行していない。

---

## 13. 資料・読取範囲・限界

### 13.1 主要なGitHub資料

各URLは固定commitを指す。取得した報告内容の評価であり、private原本の全再現や作者の全hash検査をGPTが再実施したという意味ではない。

- **[R1] 最終技術監査**：`bf8c19057fd82c900c6234152e152e79cb52e707`。統合報告を取得。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/docs/research/track_a_final_technical_audit_20261019.md
- **[R2] 新A_COST総費用比較**：`64c85f7a833b5950ef2ccae623708b73081edcd6`。主報告を取得。数値はその保存表を転記。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64c85f7a833b5950ef2ccae623708b73081edcd6/docs/research/track_a_pr_new_a_total_cost_results_20261018.md
- **[R3] Claim–Evidence Matrix**：`bf8c190…`。主張表と主要CE記述を取得。科学的正式採択ではない点を区別。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/E_claim_evidence/claim_evidence_matrix.md
- **[R4] B domain監査**：`bf8c190…`。登録集合・主panel・境界の定義と例を取得。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/B_domain/report.md
- **[R5] A希少event監査**：`bf8c190…`。L1/L2・compiler経路・controlled RZZ・Tier Rの限界を取得。公式Qiskitの全sourceを独立に再監査したものではない。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md
- **[R6] H6高次PF独立確認完了**：`6e197973d4d4183d4e27b1b8cc95cb02f311bb92`。結果・2比較・event観測を取得。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e197973d4d4183d4e27b1b8cc95cb02f311bb92/docs/research/track_a_h6_higher_pf_confirmation_run01_results_20261019.md
- **[R7] H8 rare/selection既存診断**：`cf8e93eef50333bef41f6d75803f6b819d679343`。会話で実取得済みの内容と最新監査から継承。今回全raw・bootstrapを再実行していない。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cf8e93eef50333bef41f6d75803f6b819d679343/docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md
- **[R8] H6非相殺来歴**：`bf8c190…`。最終統合・E・Bで結合された技術所見を使用。旧D全行の数値再検算は未実施。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/D_h6_lineage/report.md
- **[R9] H8独立確認の承認済み起動snapshot**：`d21647d07bc86343b292b1e1154ba05f887b1593`。branchを取得して固定commitを解決し、報告を取得。完了結果は未取得。
  - https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d21647d07bc86343b292b1e1154ba05f887b1593/docs/research/track_a_h8_new_pr_confirmation_run01_20261018.md

主な周辺証拠：A_COST取得`fc6bd039…`、高次PF統合`1fc58aa3…`、理論/compiled互換修正`6e2b7eb6…`、alpha感度`343e0c59…`、beta追補`a8ae65b1…`。今回、過去の全branch・全private原本を読み直していない。

### 13.2 添付レビューと運用資料

- `レビューpr(3).md`：前回Claude包括レビュー。入力として使用。
- `レビューpr_2.md`：Claude追加差分レビュー。特に§1–7、資料取得範囲、付録を検討。
- `research_project_instructions_unified_2026-10-17.md`：共通運用正本。実際に参照。

Claudeレビューは独立した提案・計算例であり、SOURCEや実測結果の代替ではない。Claude自身が未取得としていた原本、数値上界、compiler部品の確認範囲は本レビューで自動的に埋まらない。

### 13.3 外部一次文献

- **[L1]** J. Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2（公開PDFの表示は2026年7月改訂）。本文の推定対象・factorizationと付録D.2/Eの関連箇所を重点読解。図表の数値を独立に再解析していない。D.2のPDF screenshotは取得失敗したため、打切り議論は取得した本文テキストに基づく。
  - https://arxiv.org/abs/2503.05647
  - https://arxiv.org/pdf/2503.05647
- **[L2]** M. Hagan and N. Wiebe, *Composite Quantum Simulations*, Quantum 7, 1181 (2023), arXiv:2206.06409v3。Introduction、main results、partitionとcost-crossoverの議論を重点読解。全証明を再証明していない。
  - https://quantum-journal.org/papers/q-2023-11-14-1181/
  - https://arxiv.org/abs/2206.06409
- **[L3]** S. Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2。推定対象、PRの種類、V.C.1の対照打切り条件を重点読解し、該当PDFページも確認。物理資源表の全再検証はしていない。
  - https://arxiv.org/abs/2603.22778
- **[L4]** K. Sugisaki et al., *Impact of molecular orbital localization on quantum computational resources for Hamiltonian simulation: A benchmark study of hydrogen chain systems*, arXiv:2608.04481。Abstractと方法・III.3のgate-count条件を重点読解。preprintとして引用し、そのscaling fitの妥当性を本レビューでは評価していない。
  - https://arxiv.org/abs/2608.04481
  - https://arxiv.org/html/2608.04481
- **参考のみ**：Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741。abstract範囲を確認。高度なPFが別にあることの参考であり、本文の最先端性能を独立評価したものではない。
  - https://arxiv.org/abs/2606.30741

本調査は関連する代表的文献への重点調査であり、2025–2026年の全Hamiltonian sparsification／truncation文献の網羅的surveyではない。したがって唯一性・最初の研究であることは認定しない。

### 13.4 投稿先の公式情報

- **[J1]** Quantum Information Processing, Aims and scope.
  - https://link.springer.com/journal/11128/aims-and-scope
- **[J2]** Physical Review A, About / scope / acceptance criteria.
  - https://journals.aps.org/pra/about
- **[J3]** Quantum, Guidelines for authors / editorial criteria.
  - https://quantum-journal.org/instructions/authors/

投稿先の判断は、これらのscope・基準と現在の証拠を対応させたGPTの推奨である。編集者の事前判定や採択予測ではない。

---

## 14. 最終結論

**論文化の着地点は、PRの優位性を一般的に証明することではなく、同じ有限時間signal精度での比較結果が、候補探索・bias会計・compiled単価とshotの組合せにどう依存するかを、保存DF量子化学系で定量的に示すことに置く。**

主張を弱めるだけの論文にはしない。両側の比較domainの拡張、具体的な費用差と順位の変化、実値に基づく機構表、第三者が読める比較データを成果として提示する。

必要なのは、もう一巡の包括監査や候補大量追加ではなく、**文献との差分を正し、採用する保存値を一回の論文用解析で確定し、原稿を完成させること**である。希少eventの厳密μ保証・非Clifford/FT資源・H6全domain非相殺を必須にするのは、それらを中心claimとして採択する場合に限る。

本書の提案を正式に採用するかはユーザーが判断する。既存科学job・SOURCE・結果・grant・ledger・marker・mandatory STOPは変更しない。
