# Track B：G5後の研究方針レビュー
## 固定辞書の終了判断、保持する理論成果、非列挙return集約RTEの検討案

- 作成日：2026-10-10 JST
- レビュー開始承認：利用者の「レビューを開始して」
- 対象：Partially Randomized Trotter / Track B。PR内部の時間発展・乱択構成の改善を扱う。別チャットのHamiltonian前処理・PR外アルゴリズム研究は移し込まない。
- 固定証拠：`HIROMU1015/Partially-Randomized-Trotter`、branch `track-b-g5-fixed-dictionary-closure-20261010`、commit `23adf61f9342abff20f059be69089e89f66f7797`
- 主資料：G5 handoff [R1]、claim/evidence map [R2]、static access inventory [R3]。必要なsource [R4–R5]、既存研究記録、一次文献も参照した。
- 本書の判断：**現固定toy・固定辞書の実用優位実験は閉じたまま維持する。既存成果は限定理論・機構の記録として残す。次は、全event表を前提にしないreturn集約と生成法を、独立反証に進める一つの数理候補として選ぶ。新しい実用的主methodや性能実験の採択ではない。**
- 次の担当：Codex。後述G6は本書が提案する新しい検討段階名であり、既存の実行済みstatusではない。

---

## 0. このレビューで区別する四種類の内容

1. **既存の証拠**：リポジトリの固定報告・証明・保存値が支持すること。
2. **今回の科学的判断**：その証拠から、何を継続・終了・保留するか。
3. **今回の新しい数理導出**：旧G5には含まれない、GPTがこのレビューで構成した案と証明の検討。
4. **外部の既知結果**：論文の明示的な内容。新規性の不存在・存在を検索だけで断定しない。

以下の新構成は、一般式の導出と小さい形式代数の自己検算まで行った段階である。独立認証、有限bit samplerの実装、native合成費用の検証、既知法との非同値性の確定は行っていない。仮説・入力条件・保証候補を明示して次の反証へ渡す。

## 1. 最終判断と理由

G5は前回の終了判断を弱める結果ではなく、その適用範囲をより正確に確定した結果である。固定辞書に対する閉じた比較ができた以上、同じtoyのparameter、precision、proposalを増やしてCTSに勝つ点を探すことは次の主作業にしない。

一方、G5のaccess監査は別の不足を明確にした。現実装は、全eventを列挙した費用・誤差表を入力として評価とsamplingを設計している。R0の係数生成がO(m)であることから、この表の取得やsamplingが大系でも安いとは言えない。[R1–R5]

この不足に対し、単に「一般I0を研究する」「ISを高速化する」という抽象的な継続方針では不十分である。本レビューでは、**同じ演算子へ戻るTaylor寄与を次数をまたいで集約し、その集約係数を必要なwordだけ計算して乱択回路を生成する**という具体的な候補を検討した。

その候補には、有限平均の恒等式、短いstepでの係数非負性、全word列挙を避ける係数計算、既知ordinary RTEを使うproposal上界、未知の全正規化を重みに使わない推定量が得られた。これらが独立に成立するか、既存法の直接的な書換えではないか、角度合成・有限bit化で費用が破綻しないかを確認する情報価値はある。

**決定は、次の数理候補を限定的な独立監査へ進めるところまでである。G5の閉じた実験主線の再開、CTSより有利という採択、新分子・DF実験の認可ではない。**

## 2. G5から残る確定事項

### 2.1 終了した比較

対象は2-qubit distinct-basis controlled finite P3、既知development x=1/4、p=(3/4,1/4)、固定三precision、指定Bernstein sufficient-shot policyである。G5は6頂点classとその指定digital拡張の下界を認証した。[R1]

| 資源 | 指定classの保守的下界 | 同一の保存CTS law |
|---|---:|---:|
| T | 171,032,500 | 164,669,575.769880 |
| CX | 4,532,500 | 3,712,301.362350 |
| 総1Q（左はnative部分からの下界） | 453,250,000 | 431,125,042.097893 |

三座標は同じCTS lawの値であり、座標別winnerを合成したものではない。workspaceは同じ1、CTSのshots/axisは970447である。[R1]

### 2.2 この結果が閉じたもの・閉じていないもの

閉じたのは、固定辞書、固定IID内部label、非負degree matching、指定precision配分とfull-support event IS、指定L1誤差課金、同じ予算規則の範囲における実用優位の探索である。

閉じていないのは、任意LCU、新しいdictionary、別のconfidence・variance・stratification、真の物理的最小shot数、whole-circuit最適compile、一般involution、DF、PR＋QPE全体である。旧residual/tolerance-only K3全体の定理でもない。[R1–R2]

したがって、G5を「PR内部のアルゴリズム改善は不可能」と読み替えない。一方、比較条件を少し変更しただけの同じ探索を、一般化という名前で再開もしない。

### 2.3 保存すべき正の成果

R0の非負adjacent-degree familyと全奇数次数の限定normalization最適性、G1の固定P3構造、G4-Aの限定B2 digital classに対する有限law分離は残る。G5のCTS優越は、それらの数式を反証していない。[R2]

最も正確な既存成果のまとめは、次である。

> 同じ有限Taylor平均について、normalization最小化、完成済みensembleの混合、sampling最適化、dictionary自体の変更は異なる。固定実装・誤差・予算規則の下では、dictionary内の構成変更に厳密な価値が生じても、より強い既知dictionaryがそのclass全体を上回り得る。

この定理・構成例・限界を一つの理論／mechanism記録として整理する価値はある。ただし、これだけで独立論文に十分、またはpublication priorityが確定した、とは判断しない。

## 3. Access監査から判明した本質的な問題

### 3.1 I0という名前と、実際の入力は別

現sourceは、明示p=(3/4,1/4)、名前付きQ、既知V、具体的なcontrolled lowering、取得済み合成列を利用する。任意のblack-box Qに適用できる一般APIではない。[R3–R5]

一般involutionで数式が成立することと、そのinvolutionの制御・任意角rotation・誤差証明を安価に取得できることは別である。次の研究では、入力の列挙、classical preprocessing、量子oracleやnative回路記述を明示する必要がある。

### 3.2 O(m)の係数生成は、全体の取得費用ではない

R0のO(m)は(x,m)から係数を生成する算術操作数である。p/Q取得、word生成、合成列、費用・誤差評価、full-support proposal生成を含まない。[R2–R3]

現`events()`はL^k又はL^(k+1)のtupleを列挙し、現ISは全eventの費用から分布を作る。この実装の列挙を、その研究問題に不可避の計算量と主張することもできない。ordinary RTEの「次数を選び、IID labelを順に引く」という生成は全word表を必要としない。[R3–R4]

**新しい生成法の対照は、古い全列挙evaluatorではなく、既に非列挙で動くordinary RTE生成法でなければならない。**

### 3.3 「直前のbasisだけ」で正確な費用を集計できるとは限らない

`reduce_indices()`と`native.simplify()`はstack型の相殺である。[R4–R5]

形式例としてABBAは完全に消える。prefix ABとCBはいずれも最後がBだが、BAを続けた後の簡約結果はそれぞれ空列とCAで異なる。直前のlabelだけを状態にした集計では、この区別を保持できない。

したがって、finite-stateな費用評価を使うなら、相殺を制限した別compiler契約、stackを保持した厳密処理、又は安全な上界であることを明示する。近似集計を現在の完全相殺と同じものとして使わない。

## 4. 次の研究候補を比較する

| 候補 | 評価 | 今回の扱い |
|---|---|---|
| 同toy・同辞書をさらに最適化 | G5が指定classを閉じた。再探索の根拠がない | 不採択 |
| I0条件の下でJ1を別入力へ移す | どの情報が実際に制約されるか、取得費用が未定義 | そのままの移送は保留 |
| resource-optimal ISを非列挙実装する | 技術的に重要だが、IS最適分布そのものは既知。全cost取得を仮定すると循環する | 単独の主methodにはしない |
| return/cancellationを有限Taylor係数の段階で集約し、非列挙生成する | G5の固定degree classとは異なる構造。具体的な恒等式・生成法を検討できる | **第一の数理検討候補** |
| 大規模DF／PR wrapperに直行 | native角度・control・取得費用・研究差分が未確定 | 未認可 |
| R0–G5の限定理論記録を整える | 正負の証拠を再利用可能にできる | 保持。次の候補の成功条件にはしない |

この選択は、新しい候補が既知法より安いことを確認したためではない。現状の不足に対して具体的な数式と反証課題を作れたためである。

## 5. 一次文献との関係

### 5.1 既知IS

Cugini–Atif–Subaşıは固定protocolについて、正のcost Cに対しq∝p/√Cが二次モーメントと平均costの積を最小化することを示している。固定された実装のbiasは再重み付けで変わらない。[P1]

従ってISを使うことだけでは独立したmethod deltaにならない。さらに、独立成分・加法costという特定条件では、構成が長くなるほど相対的なISの利益が小さくなることも同論文で論じられている。その仮定を満たさない相殺付き回路へ、その結論を自動転用しない。[P1]

### 5.2 CTS

CTSはPauli coefficient collectionとreal／imaginary partの処理を使う強い対照である。文献にはMarkov samplingによる部分展開もあるため、常に全wordの巨大展開が必須であると仮定しない。[P2]

G5 toyではPauli情報を実際に小さく取得できた。将来その情報取得が不利なcontextを扱う場合も、RA側の取得費用と同じ基準で比較する。CTSに架空のclassical penaltyを課さない。

### 5.3 低次数吸収・相殺

Zhao–Yuanのmodified Taylor／LCUの構成では、identityや低次数へ戻る寄与の吸収が扱われる。[P3] R0に保存されたknown returnも既知原理として扱われている。[R2]

従って、以下で使うQ_i^2=I、低次数へのreturn、Euler化自体は新規性の主張にしない。特にm=3の恒等式だけを新methodとして採択しない。

### 5.4 辞書最適化・古典数学

Sparse Probabilistic Synthesisは、候補libraryを用いた確率／quasiprobability設計と凸最適化を扱う。主要なprocess／channel表現と本研究のcoherent operator meanの違いは保持するが、一般的な最適化原理は既知である。[P4]

自由積上のrandom walkのGreen functionは古典的な研究対象である。[P5] 以下のfirst-passage式を量子研究の新しい数学として主張しない。本書では、その有限次数係数をRTE生成へ接続する方法を新たに検討する。

### 5.5 調査の到達範囲

本調査は、指定された近接手法と一次資料を確認したものであり、網羅的な不存在・優先性証明ではない。提案構成全体が既知法の直接specializationかは独立監査を残す。参照版と取得できなかった資料は末尾に明記する。

## 6. 次の主RQ候補

> 明示された有限のinvolution記述と確率から、同じ有限Taylor first operator meanを保ち、同一labelのreturnを係数段階で集約したRTEを、全word・全Pauli係数・全native費用表を先に列挙せず生成できるか。その古典取得、有限bit誤差、制御・合成費用まで含めた方法に、既知の生成法に対する独立した価値があるか。

当面の理論課題は「CTSに必ず勝つこと」ではない。**相殺を利用することと、相殺後の分布を安く生成できることを一緒に扱えるか**である。

このRQに対する具体的な構成案を次節以降に示す。「非列挙return集約RTE」は本書での検討用の呼称であり、実証済みの新アルゴリズム名ではない。

## 7. 提案の入力・保証範囲

以下は新しい数理候補の仮定であり、旧G5の条件を変更するものではない。

- 有限の明示確率表p_i>0、Σ_i p_i=1。p_i=0のlabelは除く。まず有理数pを対象にする。
- Q_i†=Q_i、Q_i²=I。異なるQ_i間の可換性、Pauli closure、anticommutation、追加のword同一性は使わない。
- R=Σ_i p_i Q_i、奇数m、M=P_m(-iσxR)、σ=±1。
- 非負性の十分条件として**0<x≤1**を使う。x=0はidentityを直接扱う。この範囲外を否定する定理ではない。
- controlled Q_iとexp(-iφQ_i)の実装・strict phase・errorを取得できることは、係数生成とは別の実装契約として扱う。Q oracleからrotationが無料で得られるとは仮定しない。
- taskはfirst operator moment又はそのcoherent測定。system channel、状態限定古典解への置換、独立multi-blockでないsample再利用は含めない。

mとLに関する算術操作の見積りは後述するが、native gate費用や全bit complexityまで確定したとはしない。

## 8. 同じ演算子へ戻る係数を集約する

### 8.1 Reduced word

label列から隣接するiiを削除し続けた語uをreduced wordと呼ぶ。これはQ_i²=Iだけを利用する形式簡約である。

u=(i_1,…,i_l)に対し、Q(u)=Q_{i_1}…Q_{i_l}と定義する。これは**演算子の左から右の記法**であり、回路の作用時間順序は右から左である。空語ではQ(∅)=I。

同じ物理演算子に写る別のreduced wordがあっても、ここでは追加集約しない。この保守的な形式分類でも平均恒等式は成立する。

P_n(u)を、n個のIID labelがuへreduceされる確率とする。すると

\[
R^n=\sum_u P_n(u)Q(u).
\]

### 8.2 有限Taylorの集約係数

l=|u|とし、

\[
a_u(x)=\sum_{\substack{l\le n\le m\\n-l\text{ even}}}
(-1)^{(n-l)/2}\frac{x^n}{n!}P_n(u)
\]

を定義する。位相を次数lへまとめると、

\[
\boxed{P_m(-i\sigma xR)=\sum_u(-i\sigma)^{|u|}a_u(x)Q(u).}
\]

これは有限和の並べ替えによる厳密な恒等式である。

旧RA-RTEは形式degreeごとの正の係数配分を主な自由度にしていた。ここでは高い次数から低いwordへ戻る**符号付きのTaylor寄与そのものを先に合算する**。従ってG5の固定7-prototype classと同じ問題の名称変更ではない。

## 9. 短いstepで集約係数が非負になることの導出

ここからは本レビューの導出であり、独立認証前である。

\[
\chi=\sum_i p_i^2.
\]

n≥|u|とする。uへreduceする長さn+2のraw wordには、少なくとも一組の隣接iiがある。その一組を削除すると、uへreduceする長さnのwordになる。

逆に長さnのwordのn+1個の隙間へiiを挿入すると、長さn+2のwordを生成する。この列挙は重複を許すので上界を与え、挿入labelの重みの和はχである。従って

\[
\boxed{P_{n+2}(u)\le(n+1)\chi P_n(u).}
\]

Taylorの絶対項比は

\[
\frac{x^{n+2}P_{n+2}(u)/(n+2)!}{x^nP_n(u)/n!}
\le\frac{\chi x^2}{n+2}.
\]

0<x≤1では右辺は高々1/2で、交代和の絶対項は減少する。P_l(u)=p(u):=∏_t p_{i_t}から

\[
\boxed{
0<\frac{x^l}{l!}p(u)\left(1-\frac{\chi x^2}{l+2}\right)
\le a_u(x)\le\frac{x^l}{l!}p(u).
}
\]

末端で一項しかない場合も、この弱い下界は成立する。

この結果の意味は、有限Taylorをwordごとに集約すると一般に符号問題が生じ得るところ、明示したshort-step領域では、位相(-iσ)^lを取り出した残りを非負係数として扱える、ということである。

大きいx、任意の別の多項式、他のword関係まで同じ非負性を主張しない。

## 10. P_n(u)を全word列挙なしで計算する案

### 10.1 形式母関数

zを形式変数とし、各label iについて

\[
F_i(z)=\frac{zp_i}{1-z\sum_{j\ne i}p_jF_j(z)},\qquad F_i(0)=0,
\]

\[
G(z)=\frac1{1-z\sum_i p_iF_i(z)}
\]

をm次で打ち切って計算する。

提案する係数同定式は

\[
\boxed{\sum_{n\ge0}P_n(u)z^n=G(z)\prod_{t=1}^{|u|}F_{i_t}(z).}
\]

### 10.2 式の理由

Q_i²=Iのみの形式語は、Z2群の自由積の辺label付き木で表せる。隣接頂点へのfirst passageについて、直接その辺を渡るか、別label方向へ出て戻ってから渡るかを分けるとF_iの式になる。

原点へのreturnを同じように分けるとGの式になる。reduced pathに沿った各辺の初到達と、その後のreturnを分けると、指定uの確率母関数が積で得られる。

これは既知の自由積random walkの考え方に基づく本書の再導出である。引用文献[P5]が本書のRTE構成や非負性・samplingまで記載しているとは主張しない。

式は**形式級数**として使う。L=1,2のrecurrentな場合を、transienceや解析収束の仮定で排除する必要はない。有限次数mだけを扱う。

### 10.3 算術量の見積り

単純な係数再帰で、F_iの前計算は保守的にO(L²m²)有理算術操作、保持する級数係数はO(Lm)。Gの計算はO(Lm²)である。

指定uの積を逐次convolutionするならO(|u|m²)≤O(m³)、同じparentからL個の子への係数を得る処理は追加O(Lm²)で実施する案がある。

**これは算術操作数の候補評価である。** 入力有理数のbit長、必要な最終精度、係数相殺、sqrt・angleの精度、finite-bit random generation、native synthesisの時間は別に評価する。L非依存や全資源polylog改善を主張しない。

また、checkerは小さいwordを全列挙して式と比較したが、それは検算用であり、上記生成アルゴリズムが全wordを必要とするという意味ではない。

## 11. 集約wordをRTE型unitaryへ戻す

### 11.1 Pure-word LCUだけで止めない

集約wordをそのまま全部pure unitaryとしてsampleすると、一般にIと一次項を合わせたnormalizationが1+O(x)となり、ordinary RTEの1+O(x²)より悪化し得る。従って、集約後の奇数・偶数語を再びrotationへpairする。

### 11.2 偶数parentと奇数children

偶数長のreduced uに対し、その左にlabel iを付けてもreduceされない子iuを考える。u≠∅ではi≠i_1、空語では全iを許す。

\[
s_u=\sum_{i:iu\text{ reduced}}a_{iu},\qquad
d_u=\sqrt{a_u^2+s_u^2},\qquad
\phi_u=\operatorname{atan2}(s_u,a_u).
\]

s_u>0の場合、iをa_iu/s_uでsampleし、

\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_uQ_i}Q(u)
\]

とする。すると

\[
d_u\mathbb E_i[V_{u,i}]
=(-i\sigma)^{|u|}
\left[a_uI-i\sigma\sum_i a_{iu}Q_i\right]Q(u).
\]

すべての奇数reduced wordは一つの偶数suffix parentに属するので、

\[
\boxed{\sum_{u:\,|u|\,\mathrm{even}}d_u\mathbb E_i[V_{u,i}]=P_m(-i\sigma xR).}
\]

s_u=0ならpure Q(u)を使う。

**奇数word全体がHermitian involutionであるとは仮定していない。** rotationの軸は常に元のQ_iであり、後ろに一般word Q(u)を掛ける。演算子順序を交換しない。controlled相対位相も保持する。

## 12. Ordinary RTEをproposal上界に使う

\[
t_l=x^l/l!,\qquad b_l=\sqrt{t_l^2+t_{l+1}^2},\qquad
B_{\rm ord}=\sum_{l=0,2,\ldots,m-1}b_l.
\]

集約正規化をB_new=Σ_even u d_uとする。ただし、この全和を事前に安価に計算できるとは仮定しない。

第9節の上界から

\[
a_u\le t_l p(u),\qquad
s_u\le t_{l+1}p(u)\sum_{i:iu\mathrm{reduced}}p_i
\le t_{l+1}p(u).
\]

従って

\[
\boxed{d_u\le b_l p(u),\qquad B_{\rm new}\le B_{\rm ord}.}
\]

最後の不等式は、各長さlのreduced word確率の和が1以下であることから得る。

ここで保証候補となるのは**weight normalizationの非悪化**である。Aの最適値、CTSのnormalization、native T/CX/1Q、古典取得費用に対する優位性ではない。

## 13. 未知の全normalizerを使わない生成・推定

### 13.1 理想実数算術での一試行

1. 偶数lをb_l/B_ordでsampleする。
2. 長さlのraw IID label列uを生成する。
3. raw uがreducedでなければ、量子回路を実行せず数値0を返す。
4. reducedならa_u、s_u、d_uを局所計算し、確率A_u=d_u/[b_l p(u)]で受理する。棄却なら0を返す。
5. 受理された場合、子iをa_iu/s_uでsampleし、V_u,iのcoherent測定を一回行う。
6. 受理時の±1測定値Yに対しX=B_ord Yを返す。棄却時はX=0とする。

第3節のraw uを**reduceして同じp(u)を当てはめてはいけない**。集約済み係数には高次数returnを既に含めており、proposalの確率を正しく扱う必要がある。

### 13.2 平均・normalization

受理されたparent uの無条件確率はd_u/B_ordなので、上記Xは各axisで同じ有限平均の期待値を保つ。

\[
\mathbb EX=\operatorname{Re/Im}\langle\psi|P_m(-i\sigma xR)|\psi\rangle
\]

という形で使える。前後に固定された演算子を含む線形coherent taskへの適用は、その測定構成に合わせて確認する。

**平均を取る分母は全試行数であり、受理数だけで割らない。** 受理例だけを平均すると未知normalizerが再び必要になる。

この方式は、量子計算のpostselection成功だけを条件に正規化するchannel手法ではない。量子実行の前に生成した零寄与も含む古典的なweighted estimatorである。棄却試行の古典費用は残る。

### 13.3 受理率と重み

\[
Z=\Pr(\mathrm{accepted})=B_{\rm new}/B_{\rm ord},
\]

\[
\mathbb E[X^2]=B_{\rm ord}^2Z=B_{\rm ord}B_{\rm new},\qquad |X|\le B_{\rm ord}.
\]

さらにa_empty≥1−χx²/2、B_ord≤e^xより

\[
Z\ge(1-\chi x^2/2)e^{-x}\ge\frac1{2e}\qquad(0<x\le1).
\]

これは一block・提案されたshort-step領域での理想的な受理率下界候補である。多block全体の受理率やnative gate優位を意味しない。

### 13.4 未知Zをshot budgetへ無料で入れない

残りのaxis誤差予算をs>0、信頼係数をellとすると、同型のBernstein十分条件は

\[
N\ge\ell\left(\frac{2B_{\rm ord}^2Z}{s^2}+\frac{4B_{\rm ord}}{3s}\right).
\]

真のZを知っているという条件の下で、整数丸めを除く期待量子実行数NZは

\[
\ell\left(\frac{2B_{\rm new}^2}{s^2}+\frac{4B_{\rm new}}{3s}\right)
\]

となる。これは集約ensembleを直接canonical sampleできた場合の同型予算式に一致する。

**ただしZを全word和から計算できると仮定してはならない。** 運用上はZ≤1の安全な上界で予算を組むか、独立した古典的確認から上界Z_+を作り、その誤り確率・取得費用を分けて計上する。未知normalizerを推定量の重みに代入する設計とは区別する。

この節の一致は理想modelの説明であり、実際の有限bit実装で同じ費用を達成したという成果ではない。

## 14. 小次数で何が変わるか

m=3では、既知の一側returnに対し、すべての隣接identity returnを形式的にまとめた結果が簡潔に書ける。

\[
R^3=\sum_i p_i(2\chi-p_i^2)Q_i
+\sum_{i\ne j,\,j\ne k}p_ip_jp_kQ_iQ_jQ_k.
\]

i=j又はj=kのreturnをまとめ、三つとも同じ場合の二重計上を除いた恒等式である。

低次数部分は

\[
a_\emptyset=1-\chi x^2/2,\qquad
a_i=p_i\left[x-\frac{x^3}{6}(2\chi-p_i^2)\right].
\]

長さ2のparent u=(j,k)、j≠kでは

\[
a_{jk}=\frac{x^2}{2}p_jp_k,\qquad
s_{jk}=\frac{x^3}{6}(1-p_j)p_jp_k,
\]

なので角度はatan[x(1−p_j)/3]になる。低次数の共通角一つと、高次数側の高々L種類の角度を持つ。

このm=3の式は、原理を確認する例である。低次数吸収自体は既知研究[P3]と近く、この式単独の新規性を主張しない。known one-sided returnからの追加normalization差は短いxで高次の効果となり得る一方、角度の種類が増える。そのため、これだけを合成実験へ急いで渡すことは勧めない。

新しい候補として評価すべきなのは、**一般有限次数に対する集約、問い合わせ可能な係数、非負性、正規化上界、正しく重み付けされた生成手順**の組である。

## 15. この案の重大な弱点・未解決事項

### 15.1 角度の種類とon-demand合成

φ_uは一般にはwordに依存する。全wordを列挙しない係数計算があっても、量子実行のたびに新しい角度を合成する費用が大きければ実用的ではない。キャッシュを用いても、異なる角度の数が増える可能性がある。

ここは実装の細部ではなく、方法の価値を決める論点である。有限個の入力primitiveのnative記述を使えることと、多数のφ_uの厳密な誤差・合成費用を事前に取得できることは分ける。

### 15.2 Finite-bit化

a_uは有理数でもd_u、b_l、φ_u、受理確率には平方根・角度が現れる。理想実数probabilityを、有限bit random numberと厳密補正weightへ変換する方法は未実装である。

有限bit lawが理想lawに近いだけでは、exact finite mean保存とは言えない。誤差を係数L1・operator biasへ安全に戻すか、別のexact weightingを使う必要がある。重みのrange、denominator、停止確率、有限時間capも同時に扱う。

### 15.3 全native費用表を必要としない誤差評価

一案として、各発生eventをその場で検査し、taskのaxis-bias単位へ変換した一様上界ε_implを保証すれば、係数weighted biasはB_new ε_impl≤B_ord ε_implで押さえられる。これは全event表なしに使える安全な上界候補である。

ただし保守性が強すぎる可能性がある。また合成・係数・angle・controlled phaseの全誤差を含めなければならない。source-dependentな係数2等を、異なるtaskへ無断で移さない。

### 15.4 Compiler費用

新構成の短いwordが安いとは限らない。basis、controlled phase、元の相殺規則、追加rotation、state preparationを含める。相殺のために履歴stackが必要な点は維持する。

評価時に正確な全event平均が取得できないなら、証明付き上界又は誤差幅を使う。その幅によって勝敗が未判定になることを許す。幅を落として点値の勝敗を作らない。

### 15.5 利益が小さくなる領域

χ=Σp_i²が小さい分散した分布では、同一labelのreturn自体が稀になる。反対に少数labelへ集中した場合には、CTS等の情報取得も簡単かもしれない。

これは一般的な優越・劣位の定理ではないが、結果後に有利なpだけを選ぶことを避けるべき理由になる。将来の構造contrastは、この機構から結果前に選ぶ。

### 15.6 Multi-block・PR全体

一blockの受理率下界、normalization非悪化、有限平均は、PR全体の同時改善を保証しない。複数blockでのsampling独立性、零試行の扱い、weight積、共通wrapper費用、Taylor誤差、PF誤差、QPE精度を再契約する必要がある。

G5が閉じたtoyを、新構成の成功例を探すdevelopment探索に再利用しない。方法の意味を確認する形式的な恒等式チェックと、量子資源の性能評価は分ける。

## 16. 自己検算の実施記録

今回、リポジトリcodeをimportせず、Python標準ライブラリのFractionで形式語を列挙する小さいcheckerを作った。これは新構成の式を検算するためのもの。量子Hamiltonian行列、native回路、合成、旧登録データの再最適化は使っていない。

| p | x | m | checkerのraw語数 | reduced語数 | 母関数係数の一致検査 |
|---|---|---:|---:|---:|---:|
| (2/3,1/3) | 1/3 | 3 | 15 | 7 | 28 |
| (2/3,1/3) | 3/5 | 7 | 255 | 15 | 120 |
| (1/2,1/3,1/6) | 2/3 | 5 | 364 | 94 | 564 |
| (1/2,1/3,1/6) | 1 | 7 | 3280 | 382 | 3056 |
| (1) | 1 | 7 | 8 | 2 | 16 |

合計：raw3922語、母関数係数3784件、挿入不等式176件、parent上界169件。係数非負性、pairingによる形式平均の再構成、m=3のreturn係数も一致した。

これらは**GPTの自己検算**であり、Codexの独立検証でも、一般定理の代用でもない。一般式の根拠は第8–13節の導出にあり、その正しさを次段階で独立に反証する。G5の14 testsへ加算しない。

## 17. 新規性を判定するための比較表

| 構成要素 | 既知／今回の位置付け | 独立監査で問うこと |
|---|---|---|
| Q_i²=Iによるreturn、低次数吸収 | 既知原理 | Zhao–Yuan、RTE/modified Taylorの直接specializationか |
| Iと一次unitary寄与のEuler pairing | 既知原理 | conditional childrenへの適用で何が追加されるか |
| 自由積のfirst-passage／Green function | 古典数学として既知 | 本書の有限係数評価と計算量が正しいか。既存量子法に同じ生成器があるか |
| 受理棄却・零試行・再重み付け | 一般手法として既知 | 未知normalizerを使わず、finite mean・range・停止費用が本当に閉じるか |
| 集約係数のshort-step非負性 | 今回の導出候補 | 前提・一般証明・直接既知性・反例 |
| 全表なしの集約RTE生成と費用保証 | 今回のmethod候補 | 有限bit・native角度取得まで含めて独立した方法上の差が残るか |

既知の部品を使ったから無価値とも、新しい組合せだから自動的に新規とも判断しない。**入力、生成手順、必要情報、計算量、保証が既存方法とどこまで同じか**を照合する。

本レビューでは、この組全体の優先性を確定していない。新規性が未確定でも数学的な反証へ進める理由はあるが、性能pilotや論文の主method採択とは切り離す。

## 18. 研究としての着地点

### 18.1 既存成果の記録

R0–G5は、限定構成定理、費用に応じる配分、固定policy内の分離、より強い既知dictionaryによるclass-wide優越を一つの記録にまとめる。

論文の仮の中心は「finite-mean representationの限定最適性と資源設計の限界」である。実用上最良のJ1という主張にはしない。単独投稿の十分性は近接文献との差と一般的な含意に依存するため、現時点では未確定。

### 18.2 新候補が成立した場合

最も強い目標は、次の四点を結ぶ方法研究である。

\[
\text{有限次数の相殺を保つ構成}
+\text{全表を作らない生成}
+\text{有限bit・control・error保証}
+\text{古典取得と量子費用の公平な比較}.
\]

normalizationだけの小差や一toyのwinner表ではなく、どの情報があれば構成でき、何を前計算しなくてよいかを、実際の操作数と制御回路で示す。

### 18.3 成立しなかった場合

既知構成との同値、必要normalizerや新angle取得の負担、finite-bit制御の不成立が本質的なら、この候補を縮小・停止する。次のtoyやseedを追加して救済しない。

その場合も既存R0–G5の成果は消えない。理論記録を確定し、別の具体的な未解決問題が提示された段階でTrack Bの次方針を再設計する。

## 19. 次のCodex作業：G6の独立反証と実装可能性判断

G6は本レビューで提案する新段階名。**数理候補の採択可否を判断する資料を作る作業**であり、実用methodの採択・新性能実験ではない。

### 19.1 一つにまとめて任せる内容

**数学の独立反証**：集約恒等式、P_{n+2}不等式、short-step非負性、形式母関数、parent pairing、ordinary envelope、零試行推定量、未知normalizerとbudgetの区別を独立に再導出する。有限テストだけで一般命題を置き換えない。

**近接既知法との同値性**：modified Taylor／identity absorption、既知RTE、CTSとMarkov構成、自由積random walk、一般のrejection/LCU構成と、入力・law・重み・処理量をそろえて比較する。今回の式が単なる既知法の記法変更なら、その対応を示す。

**非列挙性・有限bit・実装契約**：L,m,入力bit長に対する処理量、保持するstate、sqrt/angleの精度、制御oracle、有限bit probability、worst-case rangeを監査する。特に全wordの費用表や未知normalizerを密かに要求していないか、新角度数が実用性を失わせないかを明記する。

小さい形式代数・独立に選ぶoff-domain semantics testsと、必要な限定prototypeは許す。形式モデル上のテスト入力は新しい性能比較条件の採択ではない。詳細なscript構成、数値表現、テスト設計、合理的な資源上限はCodexの技術裁量に任せる。

### 19.2 進めないもの

旧G5 toyでの新資源スコア探索、同辞書のv4復活、旧runのretry、新しいnative angleの合成campaign、DF/分子入力、PR/QPE総費用実験、外部サービスへの量子実行は行わない。

旧source・contract・result・marker・STOP・Track Aは保持する。本候補の作業を別の探索的数理記録として区別する。失敗を理由に、前提やbaselineを無断変更しない。

### 19.3 欲しい出力と分岐

| G6で判明すること | 次の扱い |
|---|---|
| 一般式が成立し、非列挙・finite-bit化の現実的な構成と既知法との差が残る | GPTが、その残った一点を検証する最小性能／実装実証を設計する |
| 数式は正しいが、既知手順との独立差がない | 新method候補として閉じる。既存記録へ位置付けを追記 |
| 数学に反例・不足前提がある | どの命題が不成立かを戻す。Codexで別algorithmへ自動変更しない |
| 全normalizer・全角度・全cost取得で非列挙性や費用が崩れる | 実用methodは保留／停止候補。単なる係数算術の速さを成功としない |
| 有限bit・control・biasが未閉鎖 | 未判定箇所と、その解消に必要な情報だけを示す。性能勝利を推定で埋めない |

全作業後mandatory STOP。次の重要な研究判断はGPT側へ戻す。「次の段階は？」だけでは新レビューを自動実行せず、利用者の開始承認を得る。

## 20. 前回から維持・変更した判断

**維持**：固定toy・同辞書の実用優位実験は終了。G5で閉じた比較を繰り返さない。R0/G1/G4-Aの理論的成果を保持する。大規模DFや全面v4は未認可。

**今回具体化**：access監査をもう一回行うのではなく、その不足に対して、係数を集約した非列挙生成の具体案を一つ提示し、次の独立反証対象を定めた。

**まだ採択していないもの**：新構成の独立新規性、finite-bit/native実装の成功、CTSに対する優位、大系の古典取得優位、論文の独立投稿採否。

従って、次担当をCodexへ渡すのは、研究判断を省略するためではない。本レビューで次の問い・数理案・反証すべき点を具体化したためである。

## 21. 実施範囲と未確認事項

今回行ったのは固定GitHub資料・sourceの読み取り、一次文献の検索と確認、数理導出、小さい形式語の有理算術自己検算、研究計画と本Markdownの作成である。

リポジトリ変更・push、旧研究runner・test再実行、登録LP、native synthesis、分子・DF、量子測定は行っていない。全G5 artifactを別環境でcold replayしたとは言わない。G5の独立性・数値認証の評価は固定証明／監査資料の範囲で行った。

文献確認ではCTS出版本文を読んだが、補足PDFの取得は失敗した。元PRについてarXiv履歴でv2（2026-07-10）の存在を確認した一方、本文として取得できたPDFはv1（2025-03-10）であり、v2全体を精読したとは扱わない。Aomoto–Katoは出版社の抄録・書誌情報を確認した範囲であり、本文の全定理との同一性監査は未実施である。

**このレビューの結論は、旧実験の閉鎖を維持し、次の具体的な構成案を独立反証へ進めること。方法の成功や優先性は、その結果から改めて判断する。**

---

## 参考資料・一次文献

### 固定リポジトリ証拠

- [R1] [G5 results / handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)
- [R2] [G5 claim/evidence map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_claim_evidence_map_20261010.md)
- [R3] [G5 static access inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_static_access_inventory_20261010.md)
- [R4] [Fixed model.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/src/trottertracks/algorithm_codesign/rte_reallocation/model.py)
- [R5] [Fixed native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/src/trottertracks/algorithm_codesign/rte_reallocation/native.py)
- [R6] [R0 independent proof](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md)
- [R7] [G4-A independent proof](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_A_independent_proof_20261009.md)
- [R8] [G4 CTS contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a221588f42ef3e58f63373915de95607a4ca36be/docs/tracks/algorithm_codesign/g4_B_matched_CTS_contract_20261009.md)
- [R9] [G4 review adopted before G5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/research/track_b_G4_scientific_review_20261010.md)

### 外部一次資料

- [P1] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1, 2026-03-13. [本文](https://arxiv.org/html/2603.13495v1). 主にTheorem 1、bias保存、独立加法costの節を確認。
- [P2] J. Peetz, S. E. Smart, P. Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026). [出版本文](https://www.nature.com/articles/s41534-025-01168-w). CTS構成とMarkov samplingへの言及を確認。補足PDF全文は今回未取得。
- [P3] Q. Zhao, X. Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021). [出版社](https://quantum-journal.org/papers/q-2021-08-31-534/), [arXiv](https://arxiv.org/abs/2103.07988). modified Taylorの低次数吸収に関する本文抽出を確認。画像・数値表には依拠しない。
- [P4] B. Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (2024-12-31). [出版社](https://doi.org/10.1103/PRXQuantum.5.040352). libraryと確率的process表現・凸最適化を既知原理として扱う。
- [P5] K. Aomoto, Y. Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988). [出版社・抄録](https://aif.centre-mersenne.org/articles/10.5802/aif.1123/). 古典的な背景の確認であり、本書の具体的RTE式の掲載を主張しない。
- [P6] *Phase estimation with partially randomized time evolution*, arXiv:2503.05647. [版履歴](https://arxiv.org/abs/2503.05647). 取得本文はv1、v2は存在確認のみ。involution構造と既知RTEの基礎を参照。
- [P7] K. Wan, M. Berta, E. T. Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, Phys. Rev. Lett. 129, 030503 (2022). [出版社](https://doi.org/10.1103/PhysRevLett.129.030503), [arXiv](https://arxiv.org/abs/2110.12071). RTE系の既知基礎として比較対象に含める。

### 自己検算の再現性

本文§16のcheckerはFraction、itertools、整数factorialのみで、raw語をreduceした確率表と、形式母関数の再帰係数を比較する。
使用したp,x,mは§16にすべて記載した。checkerと結果のSHA256は本書末尾の実行後記録へ追記する。

#### 自己検算ファイルのidentity

- `formal_return_selfcheck.py`：SHA256 `dd1186579690b716b85451799f191712396dfbc798f287066a2c7539068de756`、3,386 bytes。
- `formal_return_selfcheck.json`：SHA256 `3895eb77b1e5455bd2160945074c0b80a3eab0aa874cb52d6d9ea244c7e2421b`、2,018 bytes。

これらは本レビュー内のモデルローカル自己検算であり、固定GitHub証拠や独立認証として扱わない。
