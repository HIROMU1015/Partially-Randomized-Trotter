# Track B：SP-1後の研究方針・RQ・新規性・着地点の再設計

作成日：2026-10-06 JST  
主たる結果参照：`9d2bb1fa439748b02084bd9fbc9b10a705328f8a`  
リポジトリ：`HIROMU1015/Partially-Randomized-Trotter`  
位置付け：**研究計画案。既存の結果・契約・STOPの変更でも、新しい実行authorizationでもない。**

## 0. 推奨する結論

SP-1の登録条件でD-only/R-onlyのmaterial gainがなかったことは保持する。ただし、これを合成placement全体のno-goや、実際のRTEとPAIの二層乱数による失敗とは解釈しない。

研究Bの主RQを、狭いrole-maskの勝敗から、次へ置き直す。

> **有限精度のcoherent-signal推定に対し、何を再現する合成か、どの単位でまとめるか、どこへ合成誤差を配分するかをそろえたとき、ランダム時間発展の実装資源を削減できるか。その削減を実行可能な構成と適用条件として示せるか。**

この中の第一の構成候補は、**短いfinite-RTEブロックの補正後平均作用素を、位相を保持した低T回路の線形結合へ直接変換する方法**とする。

ただし、次は新規性にしない。

- first momentを対象にすること。
- unitaryの線形結合をsamplingすること。
- gatewise分解よりblockwise分解の方が自由度を持つこと。
- l1最小化、cost×second moment、重複atomの係数結合。
- 同一gateへPAIを重ねる代わりに別の既知分解を使うことだけ。

いずれにも直接的な先行研究または初等的な説明がある。したがって、現時点の評価は**具体化して比較する価値がある候補だが、独立した新規性は未確定**である。「joint RTE–PAI」という名称だけで新手法採択とはしない。

直近の作業は、合成対象・比較対照・小型ブロック構成を一つの研究仕様にまとめること。その後にだけ、結果前に限定した小型検証を検討する。SP-1のn/catalogue/角度追加、旧実行の再試行、H4本格計算への自動進行はしない。

---

## 1. 証拠と推論を分ける

### 1.1 今回の確認範囲

保存済みのSP-1結果報告、固定sourceの契約・会計kernel、Track Aのclaim/evidence map、過去のBF/BM判定、および関連する一次文献を参照した。[R1–R6]

今回行っていないもの：SP-1の再実行、raw全rowの独立再計算、行列simulation、新しいtrajectory、合成器呼出し、最適化solver、新しいHamiltonian生成、test suite、GitHubへの書込み。

本文中の追加数式は、既存の定義からの代数的整理である。新しい実測結果や、新規性を認定した定理とは区別する。

### 1.2 SP-1の主要結果

結果は`SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`。48 mask rows／96 axesが完了し、42 rows適格、6 rowsはモデルshot-cap超過。技術的failure、numeric-inconclusive、bias-budget exhaustionは記録されていない。[R1]

C templateのNONE比：

| n | D-only | R-only | DR |
|---:|---:|---:|---:|
| 8 | 0.99858037 | 1.0258429 | 0.01856084 |
| 16 | 1.3194975 | 4.2817191 | 0.10260741 |
| 32 | 2.3065429 | 75.369137 | 3.1606213 |
| 64 | 7.0805204 | shot cap | shot cap |

D-only/R-onlyの5% material gainは0。DRはn=8/16でgain、n=32でloss。A/Bも登録点n=16から32の間でgainからlossへ反転した。[R1]

この結果のscopeは次である。

- 2-qubit synthetic wrapper、system |0>、ancilla |+>。
- Cの外側乱数はwrapperごとの符号coinであり、**actual finite RTEではない**。
- 外側weightは1。
- D/Rラベルにはgate数・角度・符号・乱数の役割が重なっている。
- primitiveはnative error 1e-6で合成した保存列。一つのexact pi/4 catalogue。
- 指標は、保存primitive T数の加法和とBernstein十分shot数の積。
- primitive間の完全な回路最適化や、必要最小shot数を評価したものではない。[R1,R2]

### 1.3 以前の解釈を修正する

**修正A：二層RTE×PAIの失敗を実証したとはしない。**
SP-1にactual RTEのcutoff、normalization、event長分布は入っていない。Cの正負coinでjoint pairの角度は入れ替わるが、絶対角度の多重集合は同じである。signがchannel診断に影響することと、primaryのcost/momentに意味のある外側不均一性があることは別である。

**修正B：selective placement全体を閉じない。**
閉じるのは登録四maskに対する限定的な利得仮説。gate別の部分集合、精度配分、別の回路辞書などは未評価である。一方、それらを実行理由なしに追加することも勧めない。

**修正C：shot capは原理的な必要shot数の下界ではない。**
保守的十分条件から求めた数がcapを超えた、というモデル上の非適格である。技術failureとも区別する。

**修正D：交互Pauliなら最適compilerでも長い、とはしない。**
同じgeneratorの隣接融合を防いでも、1 system qubitのunitary列全体はSU(2)としてまとめられる。SP-1の固定局所合成policyには正当な意味があるが、そのn比例costを全合成法の下界には使えない。適切なphase付き全体再合成は小系の強いcontrolになる。

**修正E：SP-1の正式結果は変更しない。**
これらは研究的解釈の修正であり、既存の契約・数字・classificationの再分類ではない。

---

## 2. SP-1の現象を、まず既知の仕組みで説明する

C,n=16の保存会計は次の通りである。[R1]

| mask | 1-shot期待T費用 | 二次モーメント V2 | 十分shots/axis | NONE比 |
|---|---:|---:|---:|---:|
| NONE | 2160 | 1 | 13,523 | 1 |
| D | 1633.071 | 1.755142 | 23,601 | 1.319498 |
| R | 534.2374 | 17.62027 | 234,105 | 4.281719 |
| DR | 7.308022 | 30.92607 | 410,115 | 0.1026074 |

single-maskは通常合成の高い費用を残したまま、追加の測定負担を払う。DRではその高い費用をほぼ全て置き換えるため、さらに多い測定負担を払っても得する。

これを説明するため、bias/range/ceilを一旦固定し、gate集合Sへの変更を

\[
J(S)=\left(\prod_{i\in S}m_i\right)
\left(C_0-\sum_{i\in S}s_i\right)
\]

とする。m_i>=1はweight二次モーメント倍率、s_i>=0は1-shot費用の削減とする。残余費用をC(S)>0と書けば、

\[
\frac{J(S\cup\{i\})}{J(S)}
=m_i\left(1-\frac{s_i}{C(S)}\right).
\]

追加が有利になる条件は

\[
\frac{s_i}{C(S)}>1-\frac1{m_i}.
\]

他のgateの費用が下がると同じs_iが全体へ占める割合は増える。そのため「個別には不利、両方なら有利」は、独立積の会計だけでも起こり得る。

これは説明用の一般代数であり、SP-1の全数値を再計算したものではない。新しいcouplingやjoint分解の存在を実証するものでもない。PAI/Sparse PSで知られる積normalizationの機構と矛盾しない。[W1,W2]

---

## 3. 新規性監査：今回の最接近研究

確認日は2026-10-06。次は重点箇所を読んだscoped auditであり、全引用網を網羅した不存在証明ではない。

| 文献 | 本文で確認した範囲 | 既知として除外するclaim | 次計画での役割 |
|---|---|---|---|
| Koczor, Sparse Probabilistic Synthesis [W1] | §II、§III.1/III.4、Appendix B | 低T回路辞書、channel線形結合、l1最適化、回路全体のsampling overhead/crossover | 同じ辞書で行う一般最適化、exact/approximate分解の強い対照 |
| TE-PAI [W2] | 本文のcost議論、Appendix A/B | gate-level補間、時間発展、unitary分解とsuperoperator分解の違い | 条件を揃えたknown synthesis/simulation対照 |
| Granet–Dreyer [W3] | Resultsのsmall-angle/large-angle式、amplitude、continuous-time構成、背景Hamiltonian | unitary first momentの確率的補間、回転角とshot交換、時間発展への適用 | 単一primitiveのfirst-moment案を新規としないための対照 |
| GüntherらPR [W4] | coherent Hadamard／average operatorの議論、RTE構成、Appendix E | first-moment要件、RTE、DF、合成向けroundingとresidual移送 | 物理的なfinite-RTEと強い合成付きPR対照 |
| Cugini–Atif–Subaşı [W5] | §II、Theorem 1 | costと重み二次モーメントの共同最適化、最適importance sampling | 目的関数／sampling最適化を新定理としない |
| Morisaki–Sano–Akibue [W6] | Problem 1.1、§4のprobabilistic synthesis | 正の確率混合による単一qubit channelの近似、定義した最大branch T数の最適化 | 追加符号weightなしの有限精度対照。total T×shots最適性とは区別 |
| Campbell [W7] | Abstractのscope | unitaryの正の混合によりcoherent errorを弱める考え | [W6]の背景。今回全証明を独立確認したとはしない |
| Structure-Aware Variance Reduction [W8] | shot-noise caveat、構造別分散の議論 | count/ordering層別化等の一般構想 | 大きいtrajectory variance改善をquantum shot改善へ転用しない |

### 3.1 重要な判定

「joint分解ならgammaが小さくなる」は、新しい方法の定義になっていない。辞書を増やして最適化すれば、古い解を含む限り目的は悪化しない。それ自体は一般の最適化・三角不等式である。

first momentを対象にすることも新しくない。PRはHadamardで平均operatorが必要だと説明し、Granet–Dreyerはunitaryの線形結合を直接使い、TE-PAIのAppendix Aはそのsuperoperator法との違いを明記している。[W2–W4]

したがって、**jointという一般構想の新規性リスクは高い**。前回の「最有力・新規性リスク中」という評価は撤回し、以下の具体的な構成・取得費用・適用範囲に分解して評価する。

### 3.2 独立した新規性候補

次を実際に示せる場合に限り、方法としての差を検討する。

> 短いfinite-RTEブロックの構造を利用し、全trajectoryや巨大channelを列挙しないで、位相を保持した実行可能な回路集合とsampling係数を生成する。有限精度と強い既知対照をそろえても、量子資源・古典取得費用・適用範囲のいずれかに有用な差を示す。

これが標準LCU/Sparse PSをそのまま適用したものに還元される場合、方法の新規性は主張しない。再現可能な設計・適用研究として何が残るかを別に判定する。

---

## 4. 主RQ、副RQ、Track Aとの違い

### 仮題

**有限精度coherent-signal推定のためのブロック合成と資源設計**

### 主RQ

> 同じ有限精度のcoherent-signal taskに対し、gateごとのchannel近似／確率的補間と、短い時間発展ブロックの平均作用素を直接合成する方法を比較すると、どの条件で合成費用・測定負担・古典設計費用の釣合いが改善するか。

### 副RQ

1. **再現対象の違い**：full channelの近似と、全入力状態に対するcoherent first momentの近似では、許される回路表現と資源がどう違うか。
2. **まとめる単位の違い**：単一gate、単一RTE microblock、少数の隣接blockで、費用・normalization・古典取得費用はどう変わるか。
3. **有限精度の違い**：厳密QPDだけでなく、許されたbiasを使う正の混合や通常合成を入れると、どの利益が残るか。

全てを大きいgridへするのではない。最初に変更する軸は「同じ局所targetの合成単位と表現」。精度配分・辞書の自由度は対照とそろえる。

### Track Aとの境界

Aは固定DF/PF/RTE/shot/compiled-RZ policy内の残差処理比較。次Bは実行可能な表現と合成誤差・非Clifford資源・測定負担の設計である。[R4]

Aの原稿完成をBの成功待ちにしない。Aの成功をBの新規性・改善へ転用しない。Bが既知法の適用研究となる場合も、自動的にAと同じだとは扱わず、追加知見と論文の主claimで独立性を判断する。

---

## 5. 第一構成候補：短いfinite-RTE平均作用素の直接合成

### 5.1 対象を定義する

固定された短いblockについて、既存の補正後平均を

\[
M=\sum_\omega p_\omega b_\omega U_\omega
\]

とする。入力rhoのcoherent targetは

\[
z_M=\operatorname{Tr}(\rho M).
\]

現canonical RTEの一部ではb_omega=Bが共通である。有限化したMとexact evolutionの差は、別の既存近似誤差として保存する。

例えば現在のK=2規約では、補正後一microstep平均はTaylor三次に対応する。signed時間t、分割rなら、理想化した参照式は

\[
M_R(t)=\bigl[P_3(-itR/r)\bigr]^r.
\]

実際のadapterではscalar、DF規約、negative time、順序を既存sourceへ合わせる。KをTaylor次数と同じだと取り違えない。複数finite blockを時間和の一polynomialへ置換しない。

### 5.2 直接作るもの

実装可能な位相付きunitary V_jと係数a_jを使い、

\[
\widetilde M=\sum_j a_jV_j,
\qquad \|\widetilde M-M\|\le\delta_{\rm block}
\]

を満たすようにする。

係数は、必要なら実部／虚部と±1/±iのphase-bearing atomsへ分解し、real a_jで表す。一般の複素位相を回路へ吸収する場合は、そのcontrol-phase費用を数える。unitaryのprojective同値だけで実装を同一視しない。

q_j>0でjをsampleし、V_jのcontrolled Hadamard testの±1 outcomeをY_jとすると、

\[
X=\frac{a_j}{q_j}Y_j
\]

の平均は対応するRe/Im Tr(rho Mtilde)になる。全rhoに対するoperator residualを使えば、特定の正解stateだけに合わせる設計にならない。

このsampling-LCUの原理自体は既知である。[W3,W4]

### 5.3 channel用schemaを流用しない

SP-1のGate recordはtrace-preserving channel QPDとしてsum g=1を要求する。[R3]

Mは一般にunitaryでもtrace-preserving channelでもないため、**sum a=1を要求しない別schema**が必要である。旧kernelのこの条件を削ってSP-1へ上書きするのではなく、別namespace・型・意味論を作る。

### 5.4 小さいblockを使う理由と限界

全分子のMをdense matrixとして構築してから最適化すると、量子計算が必要な規模で使う設計法にはならない。

最初は少数Pauliの閉じた代数、またはconstant-sizeのlogical subblockへ限定する。係数多項式の積・和はPauli表現等で結合し、完全なtrajectory集合を明示列挙しない構成を検討する。

ただし、Pauli積の結合や小型algebraを使うことだけでは新規性にならない。辞書生成・最適化・samplingを含む取得費用がどう抑えられるかを比較する。

DF fragmentは一般に多くのmodeへ作用する。小support toyで動くことを、全DF fragmentへ安価に適用できる証拠としない。共通Gaussian basisを外へ出せる場合でも、そのbasis/inverse費用を数え、異なるbasis間では同じ圧縮を仮定しない。

---

## 6. 計算前に分かる制約：jointなら何でも改善するわけではない

以下は研究設計のための基本導出であり、独立した新規定理とはしない。

### 6.1 full scaled channelでは、元のnormalizationを任意には消せない

B>0、sum p=1として、ancilla込みの

\[
\mathcal T=B\sum_\omega p_\omega\mathcal C(U_\omega)
\]

を、trace-preserving channels Vcal_jの実係数線形結合

\[
\mathcal T=\sum_j a_j\mathcal V_j
\]

で**exactに**再現するなら、traceを取るだけで

\[
\sum_j a_j=B,\qquad \gamma=\sum_j|a_j|\ge B.
\]

したがって、この制約を保ったjoint channel QPDは、追加PAI overheadを削れる可能性はあっても、元のBを理由なく消せるわけではない。

これはTP辞書でfull scaled channelを一致させる場合の主張。trace-nonincreasing instruments、成功確率条件付き、近似許容、別targetへはそのまま使わない。

### 6.2 first momentを対象にすると、別の下界になる

M= sum a_j V_j、各V_jがunitaryなら、

\[
\|M\|\le\sum_j|a_j|=\gamma.
\]

ここではgamma>=Bは一般には要求されない。しかし、gammaの小さい表現がcheapな辞書内に存在すること、効率よく見つかること、controlled costを減らせることは別問題である。

full-channel対照より自由な問題を解いて安くなった場合、**制約緩和の寄与**と、**提案手順の効率の寄与**を分ける。

### 6.3 同じatomの係数結合は、強い対照にも与える

独立合成から得た同一atomの係数d_omega,jを結合すると、

\[
a_j=\sum_\omega d_{\omega,j},\qquad
\sum_j|a_j|\le\sum_{\omega,j}|d_{\omega,j}|.
\]

この三角不等式だけをjoint法の発明とはしない。equal-library最適化、同じatom merging、同じphase扱いをbaselineへ与える。

### 6.4 単一rotationのfirst-moment補間は既知control

P^2=I、R_P(theta)=exp(-i theta P/2)、0<=theta<=Delta<2piなら、

\[
R_P(\theta)
=\frac{\sin[(\Delta-\theta)/2]}{\sin(\Delta/2)}I
 +\frac{\sin(\theta/2)}{\sin(\Delta/2)}R_P(\Delta).
\]

これは2次元のI/P空間の線形代数で、Granet–Dreyerのsmall-angle/large-angle表現と同じ系統である。[W3]

3-notch channel PAIと異なるnormalizationになっても、それだけを新方法としない。controlled R_P(Delta)の費用・相対位相を含める。

---

## 7. 資源会計：Tだけが0になる抜け道を防ぐ

### 7.1 推定量の二次モーメントとrange

real a_j、±1 outcomeなら

\[
V_2=\sum_j\frac{a_j^2}{q_j},\qquad
M_{\max}=\max_j\frac{|a_j|}{q_j},\qquad
\overline C=\sum_jq_jC_j.
\]

canonical q_j=|a_j|/gammaではV2=gamma^2、Mmax=gamma。cost-aware分布を使う場合は、zero-cost atomを含む特異caseとrangeの変化も戻す。

C_j>0の理想化された二次モーメント目的では、Cauchy–Schwarzから

\[
\min_q V_2\overline C
=\left(\sum_j|a_j|\sqrt{C_j}\right)^2,
\qquad q_j\propto |a_j|/\sqrt{C_j}.
\]

これは既知importance-samplingの結果であり、exactなminimum shotsでも新しい目的関数でもない。[W5]

### 7.2 共通finite-confidence会計

各軸で

\[
s_a=\epsilon_a-b_{{\rm PF/RTE},a}-b_{{\rm block},a}
-b_{{\rm implementation},a}-u_a>0
\]

を要求する。Bernstein等の同じ十分規則を全armへ用いる。SP-1と異なる規則を採る場合は、新研究の結果前契約として全armへ同時に適用し、旧結果は変えない。

局所M_jをMtilde_jへ置換したwhole productのerrorは、norm productを含むtelescopingで伝播する。finite M_jはunitaryとは限らず、block誤差を単純加算するだけの無条件保証を使わない。

単一blockのoperator residualをI2で測っただけなら、実用methodのa priori保証とは別に表示する。小型matrixでの独立interval residual検査は、係数を生成したoptimizerとは分離する。

### 7.3 なぜT-count単独では危険か

任意の小さいMはPauli展開できる。Pauliまたは±i PauliをHadamardでsampleする部分回路はCliffordで実装でき、**blockのT数だけなら0**になることがある。その代わり、係数総量とshot数、Clifford gate数、準備・basis費用が大きくなり得る。

従って「joint blockでT=0になったから勝ち」は不適切である。

少なくとも

\[
(G_T,\;N_{\rm shots},\;G_{\rm Clifford/CX},\;W_{\rm workspace},\;C_{\rm classical})
\]

を保存する。T-based scalar比較を行うなら、共通の状態準備・basis・周辺回路を入れ、実際の文脈でTがどこに必要かを明示する。架空の準備費用を勝つように選ばない。必要なら準備費用Pを感度として表示する。

raw Pauli-LCUは弱いから除外するのではなく、この抜け道とsamplingの競合を確認するmandatory baselineとする。

### 7.4 block実装費用を実際に数える

primitive T数の和を使う初期近似は、その名前で報告する。block法だけ境界融合を許すのではなく、通常合成とPAIにも同じreduction policyを与える。full compiled claimには、実際に連結・最適化されたgate列が必要である。

prepared state、basis change/inverse、control、scalar phase、測定前回転、初期化・workspace、辞書構築とsamplingの古典費用を別欄へ記録する。

---

## 8. 公平な対照と寄与分解

### 8.1 必須の対照

| 対照 | 目的 | 限界の扱い |
|---|---|---|
| 現行RTE＋通常合成 | 元実装との接続 | targetを揃え、必要なら合成精度を公平に調整 |
| gatewise PAI/Sparse PS | 独立積の既知対照 | 同じcatalogue/辞書、phase、optimizer予算、許容bias |
| 正のprobabilistic mixture | signed overheadなしの有限精度対照 | native joint Pauli rotationまたはfull controlled blockへ正しく適用。system channel後のcontrolは不可 |
| blockwise full-channel Sparse PS | 単にblockにした効果と、target制約の違いを分離 | exponential一般solverは小型oracle baselineとして扱う |
| 同じoperator辞書の標準LCU最適化 | 特殊手順の独立した差を測る | 同一atom結合、同一情報、同一許容残差・取得予算 |
| 単純Pauli-LCU | T=0部分回路とshot/準備負担を確認 | T単独の結論を防ぐ |
| controlled blockの直接再合成 | 小さいunitary列を一括合成する強い対照 | nonunitary平均Mの実装とは区別する |
| PR Appendix Eのrounding-to-residual | 化学taskでの既知共同設計対照 | 最初のsynthetic primitiveへ無理に入れず、DF優位claim前には必要 |

### 8.2 比較は二層にする

**層A：方法の原因を分ける比較。**
同じ小型M、同じ辞書、同じ許容残差、同じcontrol/phase/costで、表現・合成単位だけを変える。

**層B：task全体の競争力。**
同じexact H/T/state-access、epsilon/alphaに対し、各方式に許したパラメータ調整を公平に与える。固定の古いq/r/Kを弱い対照として使わない。

層Aでの勝利を、そのまま全方式の最良を超えたという層Bの主張へ移さない。

### 8.3 精度配分を明示する

SP-1は1e-6のprimitive列とepsilon=.05のsignal比較だった。正式なSP-1結果に問題があるという意味ではないが、新研究の強いbaselineには、同じ全体誤差予算の中で合成精度を選ぶ機会を与える。

exact QPDとfinite-bias通常合成を比較する際、片方だけexactnessを成果として要求するのでなく、最終taskの許容誤差をそろえる。正の混合に関する[W6]の最適性は最大branch Tを目的とする単一qubit channel問題であり、whole-wrapper shots×Tの最適性へ拡張しない。

---

## 9. 次の検証を一つの有限なwork packageとして設計する

今回は新しい科学実行を認可しない。以下は実行前に詳細を固定する提案である。

### Step 1：target・構成・対照を一枚の仕様へ

成果物は次の三点に絞る。

1. 平均作用素target、実装可能なphase-aware辞書、係数/sampling/error/合成費用を定義した数学・実装仕様。
2. 上記最接近文献とのclaim表、どこが既知で何が具体的な候補差か。
3. 小型検証のcanonical入力、全armの自由度、予算、判定、保存・停止を定めた提案。

文献の一般論が既知と分かった後も、同じ文献監査だけを繰り返さない。具体的な小型constructionが書ければ、独立した実行指示を得て試験する。完成した新定理を小型試験の唯一の条件にはしない。

### Step 2：小型の構造検査と有限precision比較

最初のsystemは1–2 qubit相当の短block。one ancillaを含む意味論を評価する。full molecular matrixは不要。

**control family案**：

- SP-1型の符号coin：外側weight1。新しいRTE効果と誤認しないcontrol。
- 一つのexact/Clifford生成子：不要なsamplingやweightが増えないcontrol。
- 非自明な可換二項のfinite-RTE平均：積・和とphaseを確認。
- 非可換二項のfinite-RTE平均：小さいblockで表現差を比較。
- 二つのblockと非可換boundary：単一signalに合っただけでなく、独立blockとして接続できるか。

主要performance比較は二項RTEの可換／非可換とboundary。単一gateの線形補間を主成果にしない。

最初のdomain案は、三つのperformance family × 二つの係数比 × 二つのdimensionless時間で最大12 target blocksとし、controlを別枠に明記する。係数比・時間・合成誤差・辞書のexact gate列は、結果を見ずに固定する。これらの数は提案上限であり、許可された入力ではない。

primitive/libraryの数を先に有限化し、標準LCU/Sparse solverで比較できる小さい範囲にする。T capを後から増やして勝つatomを足さない。block spanから外れるtargetではinfeasibleとして記録する。

全blockについて、operator residual、phase、trace制約、係数norm、sampling range、moment、実gate列、classical生成costを保存する。full-channel residualは診断として保存できるが、first-moment taskと混同しない。

### Step 3：一つの実装文脈へ接続する

小型比較で役立つ差が残った場合に限り、既知developmentから一つの実DF/RTE文脈を選ぶ。これは新しい必要な角度取得・block取得であり、SP-1の再実行や既存rawから存在しない角度を推測する作業ではない。

小型blockの圧縮がDF fragment全体へそのまま使えるかを確認する。basis/prep/contextが支配してgainが消えれば、その境界を結果とする。理想小型回路の利益だけでDF advantageへ進まない。

### Step 4：固定した手順の独立評価

新しい係数生成・block選択手順を結果前に固定し、未使用条件へ適用する。exact全体signalを使って未知条件ごとに最良を選ぶものはoracle-assisted benchmarkであり、oracle-free設計とはしない。

H4 1.30 Åは既に開封済み。独立条件として再使用しない。独立評価が必要かは最終claimに合わせる。限定method/control記録なら、広い一般化を主張しない代わりにここを省ける。

---

## 10. 各結果から何を判断するか

| 結果 | 解釈・次の扱い |
|---|---|
| gatewiseの積より小さいgammaだけ得た | 既知QPDの自由度・atom結合で説明できる可能性。単独では新規性にしない |
| 単一rotationのfirst-moment補間が安い | 既知の原始mechanism。実装controlとして保持 |
| 全体channelとMのtarget差だけで改善 | coherent taskに必要な制約の違いとして説明。方法自体の効率差とは分離 |
| 同じM/辞書の標準LCUも同じ解 | 正しいことは自然。提案の取得時間・記憶・再利用・適用可能性に独立差があるかを見る |
| 小型dense LPだけで利益、規模拡大の手順なし | 小型oracle benchmarkに限定。大系向けアルゴリズムとはしない |
| 正の混合やtask-tuned通常合成で利益が消える | cheap signed interpolationの限定性として閉じる。対照を弱めない |
| block Tは0だがshots/Clifford/prepが過大 | T-only見かけの勝利。総resource/vectorでは不利と記録 |
| 強い対照後も、実装可能な手順とresource差が残る | 方法／実装研究として次段へ。独立条件とclassical costを確認 |
| 勝利はないが再利用可能な適用限界・failure mechanismが残る | 定量的設計研究の着地点を検討。自動的にnegative論文になるわけではない |
| 既知一般論の再現だけで追加知見もない | 技術記録として閉じる。次algorithmを自動で追加しない |

old5%/10%を新比較へ機械的に流用しない。新比較のmodel誤差・numerics・用途に即したmaterialityを結果前に定める。

また、別条件の実験そのものをpositive-result huntingとはしない。新しい明確な仮説と結果前の対象固定があり、不利な結果も保存するなら正当な発展である。禁止するのは、結果を見ながら支持が出るまで条件を選び、過程を隠すこと。

---

## 11. 論文としての着地点

### 11.1 最小の完結点：研究記録と比較仕様

SP-0.5/SP-1の限定結果、full-channel／first-momentの区別、誤差と資源会計、強いbaseline、小型の正しさ検査を残す。これは研究資産として有用だが、それだけで独立論文の十分性を保証しない。

### 11.2 推奨する最初の目標：設計・実装研究

> **有限精度coherent-signal taskにおいて、独立gate合成と短block平均作用素合成の資源差を、同じ精度・同じ実装制約・強い既知対照のもとで切り分ける。どの構造・精度・準備費用でどちらを使うべきかを、再現可能な構成と適用条件として示す。**

必要なのは、表現を変えると数値が安くなる一例だけではない。利益の原因、失敗範囲、classical取得費用、実装文脈との接続を説明する。

新しい一般定理がなくても、十分に独立した実装・設計知見なら研究として成立し得る。既知法と同じscoreだからという理由だけで棄却しない。

### 11.3 方法論文へ強める条件

有限RTE blockの構造から、指数的なfull dictionary/trajectory列挙を避ける構成、誤差保証、または実用的な取得費用の差を示し、同情報・同自由度の既知対照に対して利益が残ること。

全分子dense matrixや各instanceのexact ground truthを必要とするものを、この着地点へ無理に含めない。

### 11.4 Negativeの場合

「数例で勝てなかった」だけでは論文の結論として弱い。明示した表現classの制約、精度緩和後に消える利益、準備費用で必ず打ち消される限定範囲など、他者が利用できる非自明な知見が残るかを評価する。

新しいhard no-goを示していないなら、一般的な不可能性やquantum advantage否定へ広げない。

### 11.5 後のenergy/QPEへの発展

現taskは有限時間coherent signal。chemical accuracy、full QPE/RPE、低overlap状態、実機noise、物理量子ビット・runtimeは別の達成要件である。最初の論文の完成を全てに依存させない。

---

## 12. 想定する本文と図

1. **Taskと現行SP evidence**：旧結果のscope、何を再設計したか。
2. **合成対象の定義**：full channel／first moment、phase、有限誤差。
3. **構成法と既知対照**：辞書、sampling、生成cost、composability。
4. **構造control**：primitive、可換／非可換、boundary、zero-T。
5. **資源結果**：task-tuned precision、signed/positive mixtures、prep/context。
6. **適用条件と限界**：生成費用、差が消える条件、独立評価。

主図候補：

- 同じtargetに対するcost–second moment–bias frontier。
- 再現対象とblock幅による差のablation。
- Tだけでなくshots/Cliffordを含むPareto図。
- fixed primitive precisionとtask-tuned precisionの比較。
- DF文脈へ接続できた場合のみ、そのbasis/prep/context分解。

全図を先に必須化しない。目的のないparameter sweepや、figureを埋めるための計算は行わない。

---

## 13. Codexへ渡す直近の範囲案

以下は利用者が採用した場合の技術仕様化作業案であり、この文書自体は実行指示ではない。

```text
SP-1 result commit 9d2bb1fa439748b02084bd9fbc9b10a705328f8a を固定参照し、
研究Bの次の仕様を、既存B worktreeとは分離してdocs-onlyで準備してください。

1. SP-1がactual finite RTEでないこと、登録D/R maskの限定結果、十分shot/cost policyを記録する。
   原result・分類・markerを変更しない。
2. 次RQを「有限精度coherent-signalのblock合成と資源設計」とし、
   短いfinite-RTE平均作用素Mの直接合成を未実証の構成候補として記載する。
3. full scaled channel、first moment、unitary per-sampleのtarget/schemaを分ける。
   channelのsum g=1条件をoperator LCUへ流用しない。
4. Granet–Dreyer、PR、TE-PAI、Sparse PS、positive probabilistic synthesis、resource-optimal ISを
   同じtarget・精度・workspaceへそろえた比較表にする。
5. 既知の単一rotation補間と標準LCU最適化を対照に残す。
   gamma低下だけ、同一score、T=0 blockだけでGO/STOPを決めない。
6. 最大12程度の小型performance target案と必要controlを作る。
   係数・時間・dictionary・T cap・合成precision・全誤差予算・solver/budgetは未承認案として明示する。
7. matrix/trajectory/library/solver/synthesisの新実行はしない。
   新しいコード作成が必要なら、対象APIと取得情報を設計書へ書くまでに留める。
8. B-F/B-Mのclosure、SP-0.5/SP-1、Track Aの原稿・source・statusを保持する。

成果物は数学・実装仕様、claim/対照表、有限pilot提案の三点を一つの入口から辿れるようにする。
完成した新規性やsource review済み実装を宣言せず、利用者reviewへ戻してSTOPする。
```

実行承認後のtechnical testsとscientific pilotは区別する。一方、小型の正しさ検査まで過剰に文書化し続けて進めない運用も避ける。次回は具体的なtarget・辞書・比較法の採否をまとめて判断する。

---

## 14. 参照資料と読取りscope

### Repository

[R1] `9d2bb1fa439748b02084bd9fbc9b10a705328f8a`、
`docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md`。
SP-1 result summary、scope、全mask比、C n16の分解、failure/STOP。

[R2] `0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`、
`docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md`。
SP-1 target、role、controlled lowering、finite confidence、固定precision。

[R3] 同source、
`src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_accounting.py`。
Gateのsum g=1、population moment/cost/bias、Bernstein sufficient count。

[R4] `4c23453c541700c6a41ba71fc5ec9323b53858d6`、
`docs/research/track_a_post_pm2_claim_evidence_map.md`。
Track Aの採用claim、誤差・費用、post-hoc/transferの境界。

[R5] `d55de044b8e956ba6292209a94bb081014dfdae2`、
`docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md`。
既に会話で取得された一般再帰導出と同情報対照の同値性。今回は記号scriptを再実行していない。

[R6] `6d2645a09440f50e5b869ef42a1b73a1b625a1af`、
`docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md`。
既に会話で取得されたBF-A／原INCONCLUSIVEの二層記録。再replayしていない。

### Primary literature

[W1] Bálint Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*,
PRX Quantum 5, 040352 (2024), arXiv:2402.15550v2。
§II、§III.1/III.4、Appendix B。全文HTMLの該当箇所を確認。

[W2] *TE-PAI: Exact Time Evolution by Sampling Random Circuits*,
arXiv:2410.16850v2。本文とAppendix A/B。unitary-matrix補間との比較を確認。

[W3] Etienne Granet and Henrik Dreyer,
*Hamiltonian dynamics on digital quantum computers without discretization error*,
npj Quantum Information 10, 82 (2024), DOI:10.1038/s41534-024-00877-y。
arXiv:2308.03694は旧題にContinuousを含む。
出版社本文のResults、small-angle式(1)–(2)、amplitude、背景Hamiltonianの扱いを確認。

[W4] Günther et al., *Phase estimation with partially randomized time evolution*,
arXiv:2503.05647、PRX Quantum 7, 020332 (2026), DOI:10.1103/ynxb-p2xq。
44-page公開PDFのcoherent first-moment/RTE節とAppendix Eを確認。
versionless PDFの取得記録をv2 immutable bytesの再hash検証とは呼ばない。

[W5] D. Cugini, T. A. Atif, Y. Subaşı,
*Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*,
arXiv:2603.13495v1。§II、Theorem 1、objectiveのscopeを確認。

[W6] Morisaki, Sano, Akibue,
*Optimal ancilla-free Clifford+T synthesis for general single-qubit unitaries*,
arXiv:2510.05816v1。Problem 1.1と§4を確認。
最適性の対象はsingle-qubit channel errorと最大branch T-countであり、whole-task資源ではない。

[W7] Earl Campbell, *Shorter gate sequences for quantum computing by mixing unitaries*,
Physical Review A 95, 042306 (2017), arXiv:1612.02689。
今回の独立確認は公開abstractのscope。詳細の対照設計には[W6]の本文を使用。

[W8] *Structure-Aware Variance Reduction for Randomized Quantum Simulation*,
arXiv:2606.23544v1。shot-noiseを含まない改善値の注意と構造別分散の本文を確認。

### 新規性監査の最終評価

既知の広い構想との重複は明確。特定の有限RTE blockを実用的に生成・合成する手順の優位は未実証。
本書は不存在証明、査読・採択可能性の保証、未実行pilotの性能予測ではない。
同時に、一般論が既知であることだけで有用な具体的構成・設計研究を否定するものでもない。
