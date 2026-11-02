# Track B G7 科学的研究レビュー
## 条件付き期待T改善の意味、非列挙native取得の未閉鎖点、次の研究判断

- 作成日：2026-10-10 JST
- 開始承認：利用者の「レビューを開始して」。本書はG7の科学的レビューであり、次の実験の実行記録ではない。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 結果branch：`track-b-g7-budget-control-economics-20261010`
- 結果commit：`a689694080f4b7600fe67cf77d841d1cbbf04503`
- 実行source S：`ab2549f41b3546fb3940342a2162dd9ee93699c4`
- 前段：G6 `28cfabb1d47e0e1824bce1e154a9c289738fa2b9`
- 判断：**条件付き方法候補として限定継続。ただし主method採択・一般的なnative優位・分子移送は保留。次は非列挙native取得と実装誤差を含む契約の限定確認。**
- 上記は本レビューの推奨であり、GitHubのG7結果statusを書き換えるものではない。

## 0. 結論の要点

G7は、full returnの数理的normalization改善を、取得済みRz列と明示されたcontrolled-Q query会計へ接続した。P5の既知development入力では、ordinary、partial-return+tail、P3 closed-form+tailの三対照より、固定予算規則における条件付き期待T費用が小さい。Rz部分だけでなく、各labelの期待provider呼出し数と期待preparation回数の差も負であり、同じexact-provider会計内では未知の非負provider費用を選んで作った勝利ではない。[R1–R3]

ただし、最も近い対照partial-return+tailに対するRz部分の差は約0.522%である。全試行数、worst-case quantum-call cap、古典生成費用、angle取得費用は増えている。さらに、productionのword生成は非列挙だが、native angle cacheの事前取得には小support列挙を用いている。従ってG7は「全工程が非列挙で安い」「現実のprovider誤差込みで有利」という実証ではない。[R1,R5,R6]

P3でfullがclosed-formに劣ることは、同じ三次全集約表現を、既知normalizerを持つcanonical samplerとzero-fill samplerで実現した違いと整合する。一方、P3とP5ではpとxも変わるため、今回の二点から次数だけによる優劣の転換を結論してはならない。

次の中心的な問いは、「さらに多数の入力で勝つか」ではなく、**全event/angle表を作らず、有限誤差のprovider・局所合成・古典処理まで含む実行手順にしたとき、低次数集約対照後の追加価値を保てるか**である。これを閉じる一つの限定作業をCodexへ委ねる。旧G5の固定辞書閉鎖、旧STOP・markerは維持する。

---

## 1. 確認資料・監査の範囲

本レビューでは、G7 handoff、数学・実行契約、resultの関連区間、resource summary、key inventory、manifest、focused-test記録、保存値監査、generator/provider/reference/runner sourceをGitHubから取得して読んだ。branch refが指定結果commitに一致することも確認した。[R1–R11]

結果JSONは約200 KBあり、取得した関連区間でruntime/provider契約、取得sequence、P5 fullの資源記録、affine差の符号などを照合した。全JSONをこの環境にコピーし、全有理数・全sequenceを一から再計算したわけではない。18 testsの再実行、24列の再合成、strict matrix guardの再実行、旧runnerの起動、全52 source hashと893保護pathの独立再ハッシュは行っていない。それらについては公開されたsourceと監査記録を根拠にする。

今回行った追加計算は、表示用CSVの比率・因数分解と、G7の同一P5入力に対する受理回数上界の数理的自己検算である。後者は新しいレビュー導出であり、G7実行結果や独立認証済みcertificateへ転記しない。実分子、量子行列、量子sampling、LP、合成、追加angleの取得は0。リポジトリ変更も0。

コンテナからの直接ネットワーク取得はDNS制約で成功しなかったが、GitHub connectorによる資料の取得は成功した。これは未pushや証拠欠損ではない。過去のレビュー付属selfcheckがCodexへ渡っていなかったことはhandoffに記録されているが、G7には独自source/test/resultがあり、今回の判断に不可欠な証拠欠落とは扱わない。[R1]

### G7の証拠の種類

| 種類 | 今回存在するもの | 存在しないもの |
|---|---|---|
| 理論・意味論 | 同finite mean、U、digital moment/range上界、control構成 | 任意のprovider実装での最適性・普遍的優位 |
| native取得 | 24 Rz key、actual adjoint、strict誤差guard | provider本体の分子native回路、whole-circuit最適compile |
| 費用比較 | 8登録rowの条件付き期待資源とaffine差 | 全sampling/precision/dictionaryを最適化した下界比較 |
| runtime | 一回のcold取得、128固定bitstream試行のCPU | 大規模・実運用のthroughput、外部再現 |
| 研究上の一般性 | 二つの既知development入力 | 未使用条件へのtransfer・PR/QPE全体 |

---

## 2. G7が実際に比較した問題

対象は

\[
M=P_m(-ixR)=\sum_{n=0}^m\frac{(-ixR)^n}{n!},\qquad
R=\sum_i p_iQ_i,\quad Q_i^\dagger=Q_i,\quad Q_i^2=I.
\]

同一入力内では全方式が同じ有限多項式を対象にする。exponentialへのTaylor remainder、最終QPE精度、分子geometry、DF rankは今回の比較対象外である。[R2]

| 入力 | p | x | m |
|---|---|---|---:|
| P3_control | (3/7,4/7) | 2/5 | 3 |
| P5_general_order | (1/5,3/10,1/2) | 5/7 | 5 |

四方式はordinary、partial-return+ordinary tail、P3全集約closed-form+ordinary tail、full returnである。G6から既知の形式入力を使うため、今回の費用取得自体が結果前固定であっても、independent held-outではない。

Re/Im各axisの許容誤差は1/200。8 rows×2 axesのfailure allocationは各1/320、和が1/20である。complex誤差はsqrt(2)/200<1/100。同じ保守的axis biasを全方式へ課金し、残り統計余裕は

\[
s=0.004987999993999988
\]

と固定した。G7は実際の測定結果から精度を確認したのではなく、この確率モデルと誤差上界に基づく十分予算を算出した。[R1,R2]

### 予算の式

\[
N=\left\lceil\ell\left(\frac{2m_{2,+}}{s^2}+\frac{4W_+}{3s}\right)\right\rceil,
\qquad \ell\ge\ln640.
\]

Nは一axis当たりの**全試行数**。full returnでは棄却も一試行に数え、棄却時に量子回路・readoutは実行しない。

\[
m_{2,+}^{\rm full}=\kappa B_o^+U^+,
\quad
m_{2,+}^{\rm canonical}=\kappa(B_a^+)^2,
\quad
\kappa=\frac{(1+\rho)^2}{(1-\eta)^{m+2}}.
\]

G7のeta=rho=10^-12、root bits256、probability bits160。fullの真のglobal normalizerやreference varianceを予算に代入していない。小support参照で計算したdigital momentは、予算上界との照合用である。[R2,R5,R6]

---

## 3. 結果と解釈

### 3.1 保存費用

以下は[R3]の表示値。T_RzはRz合成列のT/T†のみで、provider/preparation Tを含めた総native費用と呼ばない。

| 入力 | 方式 | N/axis | 期待量子calls（2 axes） | T_Rz（2 axes） |
|---|---|---:|---:|---:|
| P3 | ordinary | 698,195 | 1,396,390 | 195,494,600 |
| P3 | partial+tail | 603,994 | 1,207,988 | 169,118,320 |
| P3 | closed P3+tail | 602,596 | 1,205,192 | 168,550,838.34 |
| P3 | full | 648,863 | 1,205,480.03 | 168,591,120.79 |
| P5 | ordinary | 1,174,526 | 2,349,052 | 321,659,736.67 |
| P5 | partial+tail | 894,664 | 1,789,328 | 238,505,280.45 |
| P5 | closed P3+tail | 880,312 | 1,760,624 | 246,677,716.21 |
| P5 | full | 1,014,322 | 1,740,102.13 | 237,260,558.66 |

### 3.2 P5の最も近い対照はpartial+tail

fullのRz費用低下は、ordinary比約26.239%、partial比約0.522%、closed P3比約3.818%である。

ordinary比の大きい数字だけを代表にすると、既知低次数returnで得られる利益を新しい一般生成法の利益へ混ぜる。主要な比較は、少なくともpartialとclosed P3の両方を含めるべきである。normalizationだけなら強い対照と、実T列まで含むと強い対照が一致しないことにも注意する。

P5 full/partialを因数分解すると、

\[
\frac{T_{{\rm Rz},f}}{T_{{\rm Rz},p}}
=
\underbrace{\frac{K_f}{K_p}}_{0.97248919}
\underbrace{\frac{T_{{\rm Rz},f}/K_f}{T_{{\rm Rz},p}/K_p}}_{1.02292259}
=0.99478116.
\]

ここでKは二axis合計の期待accepted quantum calls。fullは約2.751%少ない回路を実行するが、一accepted回路のRz T費用は約2.292%高い。その差引きが約0.522%の純Rz削減である。これは表示値からの記述的因数分解であり、counterfactualな因果寄与率ではない。

同じ比較のprovider呼出し合計は約3.496%減る。一方、全試行数は約13.375%増える。従って、量子側の削減と古典側の増加が同時に起きている。[R3]

### 3.3 一段強い結果：未知provider費用への係数ごとの優位

方式aの条件付き期待Tを

\[
G_a=T_{{\rm Rz},a}+\sum_i A_{a,i}\tau_i+K_a\tau_{\rm prep}
\]

と書く。tau_iはlabel iのexact controlled-Q provider一回のT費用、tau_prepは同一のpreparation一回のT費用。G7はこれらを非負の未知量として残している。

P5 full-minus-partialは表示値で

\[
\Delta G\simeq
-1,244,721.786
-13,302.957\tau_0
-33,675.029\tau_1
-94,749.834\tau_2
-49,225.871\tau_{\rm prep}.
\]

各係数の符号は原resultのexact fractionsで負として保存され、runnerの`contrasts`もその条件を判定している。[R1,R4,R6]

よって、**同じexact-provider・同じ誤差予算・同じ加算会計を維持する限り**、任意の非負tauに対してfullの期待Tは小さい。providerを0と置いて勝ちを作った結果ではない。ordinaryおよびclosed P3対照に対しても、G7のintercept/各provider/preparation差は全て負。

ただし、以下は出ない。

- providerの実装誤差を変えて再予算化しても必ず有利。
- 異なるcontrol接続・routing・cachingで同じtauが使える。
- whole-circuitで異なる相殺を許しても順位不変。
- 基準側のsampling/precisionをさらに最適化しても優位。
- 古典費用を含めたwall-clockまたは金銭費用も小さい。

また0.522%はtau=0というendpointの**Rz-only比率**であって、全tauについての総T削減率ではない。総比率は、正の分母項を重みとする係数比の加重平均として変わる。

### 3.4 小さいがゼロでないmargin

固定G7予算、provider追加費用を除くRz差だけを見ると、fullだけに一accepted回路当たりk Tの追加費用を課す損益分岐は

\[
k=\frac{1,244,721.786}{1,740,102.129}\simeq0.7153\ T.
\]

これは実際に未知の追加ゲートが0.7153 Tあるという推定でも、CPUをTへ換算した値でもない。**登録外の方式固有費用に対するRz-only marginが小さい**という感度表示である。provider呼出しの削減が実物理回路で大きければ、許される追加費用はさらに大きくなる。これらを一つの恣意的な加重和へ混ぜない。

---

## 4. P3 negative controlと次数の解釈

P3ではfullとclosed P3が使うangle keyは同じ三つ、参照eventは同じ四つである。[R7] G6の閉形式では同じ全集約representationを直接canonical samplingできるため、未知normalizer回避のzero-fillを導入する理由は弱い。

G7のfull/closed P3は、期待quantum calls、T_Rz、provider callsの合計で、いずれも約1.00023899倍。取得列の違いで負けたというより、同じ構成を少し保守的な予算・proposalで実現した費用差と整合する。これは一般生成法を全ての小次数へ無条件に使わない理由になる。

ただしP3とP5では、mだけでなくp、L、xも異なる。G7は「m=3からm=5へ変えれば勝敗が反転する」という因果実験ではない。正確な記述は「二つの固定入力のうちP5で登録対照への条件付き優位、P3でclosed対照への非優位が得られた」である。

P5にも直接的な低次数展開・集約を特化実装する余地はある。Green再帰は一般mへの統一的処理法であり、固定m=5専用の最速古典法だとはまだ言えない。次の独立入力を増やす前に、一般生成器として何を保証するかを明確にする必要がある。

---

## 5. 非列挙性は、どの段階まで成立したか

### 5.1 sourceで分離されているもの

`FullReturnGenerator.sample`はorder→raw labels→reduced判定→local packet→accept→childという順序で進む。全parent表、global B_new、global cost tableを読む実装ではない。[R5]

一方、`inventory()`は`reference_events()`を列挙し、全ratioの集合を作る。runnerはその集合の24 keysを取得してからproduction traceを測定する。P5 fullの参照は31 parents/63 events、角度10 keysである。[R6,R7]

したがって、成立したものは「局所event/proposalの非列挙生成」であり、「入力から実行可能なnative circuitまでの全工程の非列挙取得」ではない。G7はこの限界を明記しており、実装上の隠蔽や既存結果の誤りとは扱わない。

### 5.2 angle complexityの実際

P5のpositive angle数はordinary3、partial3、closed P3 5、full10。fullのevent数63は対照の273/264/258より少ないが、異なる角度は増える。[R7]

ゆえに「supportが小さいから合成も少ない」とは言えない。費用は少なくとも、unique angleのcold取得、local coefficient/angle query、cache lookup/miss、bounded cacheのメモリ、provider circuit生成に分けるべきである。

同じp・x・mを多数shotで再利用できる場合はangle cacheが償却される。一方、PR/QPEではtime/step/contextが変わり得るため、今回の10 keyを一度取得するだけで済むとは限らない。大きいL,mでのangle種類数やcache hit率も未検証。

### 5.3 CPUデータの正しい扱い

P5 production128 trialsのCPUは、ordinary約0.00418 s、partial約0.00737 s、closed P3約0.01769 s、full約0.03061 s。fullはそれぞれ約7.33/4.15/1.73倍である。cold unique-angle CPUもpartial約0.126 s、closed約0.213 s、full約0.423 sだった。[R3]

これらは一回の短い固定bitstream測定で、trace serialization/hashも同じ測定区間に入る。timer分解能、Python実装、cache状態の影響を受ける。実運用で何秒遅い、または何倍遅いと断定してはいけない。少なくとも、古典費用が無料または低下したという証拠ではない。

核となる比較は

\[
\text{setup cost}+2N\,\text{per-trial classical work}
\quad\text{と}\quad
\text{quantum resource vector}
\]

を分離して示す形である。ハードウェアが未指定なのにCPU秒とT countへ恣意的な交換比率を与えない。

---

## 6. 期待量、worst-case、high-probability capを区別する

G7のP5 fullでは二axisの全試行数M=2,028,644、期待量子calls約1,740,102。closed P3は1,760,624回を全て実行する。従ってfullは期待量で約1.166%少ないが、hard capは約15.223%大きい。[R3]

この差を「期待値だから使えない」と切り捨てる必要はない。利用者のresource制約が期待費用、確率付き予算、絶対上限のどれかで結論が変わる。逆に、期待値だけから固定の量子実行予算を保証したとも言えない。

### 6.1 今回の補足導出：受理回数の確率的上界

以下はレビューでの新しい数学検討で、G7登録contractを変更しない。

独立ビット列に基づくM回の同一law試行なら、受理回数AはBinomial(M,z)である。digital proposalの相対誤差とenvelopeから保守的に

\[
z\le\bar z=(1+\eta)^{m+2}\frac{U^+}{B_o^-}
\]

とできる。これはG7のsupportを全列挙して得た真の受理率を、運用上無料で利用するものではない。

P5の同じp・x・mについて、empty-parent質量を有理式から計算し、60桁外向きsqrt区間でUとB_oを囲った。新しい角度や費用は取得していない。

\[
B_o\simeq1.50209305394,\quad U\simeq1.29675587482,
\quad\bar z\simeq0.86329929523<0.864.
\]

v=M×0.864と置き、Bernstein tailより

\[
\Pr\left(A>v+\sqrt{2vt}+\frac{2t}{3}\right)\le e^{-t}.
\]

t=7ならe^-7<1/1000であり、切上げた受理回数上限は**1,757,707**。これは同じG7のNを固定した確率的な参考上界で、closed P3の1,760,624より小さい。

ただし次を厳守する。

1. hard cap2,028,644は変わらない。
2. 1,757,707を超えたら打ち切る実行をG7は実施していない。
3. 資源失敗確率1/1000は、新たな予算項である。元のfamilywise推定失敗1/20に足せば同じ1/20保証ではない。将来採用するならfailure allocationとNを再固定する。
4. これは回路呼出し数の上界で、T costの上界ではない。event別Tとprovider呼出しが違うため、総Tには別の有界cost確率変数の評価が要る。
5. 固定SHA256 traceの頻度から確率保証を推定しない。独立uniform bitという実装契約に基づく。

よって、hard capの増加だけでmethodを止めるのは早いが、期待量の優位だけでdeadline最適とも言えない。

### 6.2 自己検算に用いたP5 root式

mu_k=sum p_i^k、chi=mu_2として、

\[
a_\emptyset=1-\frac{\chi x^2}{2}+\frac{(2\chi^2-\mu_4)x^4}{24},
\]
\[
s_\emptyset=x-\frac{(2\chi-\mu_3)x^3}{6}
+\frac{(5\chi^2-4\chi\mu_3+2\mu_5-2\mu_4)x^5}{120}.
\]

これはG6の形式母関数を五次まで展開したレビュー用式であり、新しい研究法の新規性として扱わない。自己検算コードは理想root式とG7のU式を利用するが、repository moduleをimportせず、samplingもしない。結果は独立認証前。

---

## 7. exact-providerから現実のproviderへ移す際の条件

G7の「任意の非負provider T費用で優位」は、exact providerかつ同じ予算が前提である。実providerの誤差を0に固定したまま、一般分子で有限Tの実装が達成されたことにはならない。[R2,R8]

provider呼出しrのstrict operator errorをdelta_r、event eの呼出し回数をn_(e,i)とするなら、telescopingによる十分なevent誤差は

\[
\delta_e\le2\epsilon_{\rm Rz}+\sum_i n_{e,i}\delta_i
\]

である。actual adjointを使うこと、control位相を保持すること、helperも含めた同じ全演算子を比較することが必要。平均のbiasは、proposalで重み付けした呼出し数ではなく、target coefficientで重み付けした

\[
\sum_e|\widetilde\alpha_e|\delta_e
\]

から上界を作る。従って期待provider callsが少ないことだけでは、provider biasが小さいと証明できない。

G7と同じ一様delta上界を全labelへ課し、親長さ<=m-1・helperの2 callsを使うなら、max calls<=m+1。norm<=3を使った共通の保守的axis bias候補は

\[
b_\delta\le2\{3\rho+3(1+\rho)[2\epsilon_{\rm Rz}+(m+1)\delta]\}.
\]

この式は、追加したprovider仮定に対するレビューの拡張案であり、G7の結果ではない。s_delta=1/200-b_deltaを再計算し、Nも再予算化する。sourceのproviderが満たすdeltaが未取得なら、条件付きformulaとして残す。

同じ事前登録input・lawを維持した再予算化と、新しい物理providerの選択・実行は別の作業である。後者にはそのproviderを選ぶ研究上の根拠、同providerに適用可能な強い既知対照、取得上限が必要。

---

## 8. 強い対照と新規性

### 8.1 G7の比較で主張できる範囲

G7の正の主張は、登録された三つの生成・予算方式に対する条件付き期待Tの改善である。対照全体の最適性は示していない。とりわけ、G7のcanonical基準に任意のcost-aware IS、全precision最適化、別の五次specializationを与えた比較ではない。

これはG7を無効にしない。G7の目的は低次数対照を戻した最小implementation economicsであり、その問いには答えている。ただし、これを「既知の全手法より優れたresource-optimal法」と呼ばない。

### 8.2 文献との対応

| 先行研究 | 今回確認したこと | 残る差候補 |
|---|---|---|
| Wan–Berta–Campbell [L1] | Algorithm2は次数とIID labelから直接生成する。全word表は不要 | 全returnを係数段階で集約する構成と、その追加取得費用 |
| Aomoto–Kato [L2] | free-product Green multiplierと再帰は既知の数学 | その再帰自体ではなく有限P_m・positive coefficients・回路生成・予算への接続 |
| Zhao–Yuan [L3] | 高次数寄与の低次数・identityへの吸収が既知 | 全形式returnを非列挙で保持する具体的algorithmの差 |
| Peetz–Smart–Narang [L4] | Pauli collection/Euler化と、全展開を避けるMarkov/layeringがある | 利用できる情報とnative circuitを揃えたcoherent first-moment比較 |
| Cugini–Atif–Subaşı [L5] | cost×二次モーメントのIS最適化、bias保存、ZeroFill/Discard原理が既知 | representationを変更する部分と、その取得・予算を含む価値 |
| 原PR論文 [L6] | 最新v2/出版本では分子・phase estimation全体の資源を扱う | G7はその一部のfinite polynomial taskであり、総PR優位ではない |

「同じ結合手順が見つからない」だけで新規性を確定しない。一方、部品が既知だから価値がないとも言えない。候補論文では少なくとも、利用可能入力、出力law、正しさ、算術/bit量、control前提、予算/取得法、既知構成との一対一の対応を明示する必要がある。

本レビューではWan Algorithm2の該当PDFページを画像で確認した。Aomoto–KatoとZhao–Yuanは関連本文の抽出テキストを参照したが、本環境でそのページのscreenshot取得は失敗したため、式の版面を独立に画像確認したとは言わない。CTSは出版本文の該当節を確認。補足全文の新たな独立再読を実施したとはしない。G6の文献監査は過去の確認として区別して参照する。

### 8.3 CTSをいつ戻すか

現在のG7は形式的なcontrolled-Q query modelであり、特定のQ_iのPauli記述が指定されていない。そのまま同じnative costでCTSを計算することはできない。しかし、実際にPauli/明示basisを持つproviderを選ぶ段階では、同taskのCTSやPauli簡約を外してはならない。

G5のtoyでCTSが勝ったことを、この新しい形式P5へ転写しない。同時に、G7でCTSが未評価であることを「CTSより優れている」と解釈しない。具体providerへ進む前に比較できる情報・費用を固定する。

---

## 9. 研究としての価値と着地点

### 9.1 現時点で支持する主張

「固定short-step有限Taylor targetについて、形式returnを集約し、全coefficient tableやB_newなしで生成し、取得可能な上界Uから有限bit予算を作る方法がある。既知の小development入力では、追加古典費用を伴いつつ、登録低次数対照より条件付き期待Tが小さい具体例がある。」

これはG6より一段進んだ主張である。抽象的normalization改善だけでなく、実際のRz列とprovider呼出しvectorへ接続したためである。

### 9.2 まだ支持しない主張

「全入力で有利」「五次以上で常に有利」「全工程非列挙で高速」「現実のprovider誤差込みで有利」「PR/QPE総費用が削減」「CTS等を含む最良法」「新規性・独立論文の成立」は未確定。

### 9.3 推奨する主RQ

> 同じfinite Taylor first operator momentについて、全returnを保持する局所生成は、入力・角度・control・統計予算を実際に取得する工程を含めても、安価な低次数集約法にはない資源上の利点を持つか。その利点と不利域を、明示された情報accessと費用条件で説明できるか。

「P5の勝率を増やす」ではなく、**数理上の生成可能性が実行手順として閉じるか**を主問にする。

完成形の第一候補は条件付きalgorithm/implementation研究であり、分子advantage論文ではない。必要な部品は一般構成・finite-bit保証・局所取得計算量・control契約・低次数/強い既知対照・有限資源の例である。これが実装側で閉じなければ、構成と制約を示す理論ノートとして区切る。ノートだけで投稿に十分かは、独立したmethod deltaの評価を経て決める。

---

## 10. 次の作業候補の比較

| 候補 | 判断 | 理由 |
|---|---|---|
| 別p/x/次数を多数追加し勝率を調べる | 今は採択しない | pipelineと実装誤差が閉じていない。結果後の勝つ入力探索になり得る |
| 大きいDF/分子で直接検証 | 今は採択しない | controlled provider・角度取得・CTS情報モデルを飛ばす |
| 同じ数理証明をもう一度全面再実装 | 優先しない | G6/G7ですでに別実装の証拠がある。必要箇所のみ認証する |
| 固定table全体を巨大LPで最適化 | 採択しない | 旧v4を復活させる理由がない。非列挙という中心課題から離れる |
| 非列挙native取得＋有限provider誤差・費用感度を閉じる | **次の第一候補** | G7から残った最重要のgapに直接答える |
| 条件付き構成ノートとして整理 | 並行して維持 | G6/G7成果の範囲を残し、成功していないclaimを切り分ける |

---

## 11. 次の担当とG8の範囲

**次の担当：Codex。** 次stageの名称を便宜上G8とする。目的は一つ、

> 「全event/angle表を生成器に与えない実行経路と、有限provider誤差を含む予算契約を作り、登録低次数対照後の追加価値が失われる要因を判別する」

である。本レビューを根拠に旧runを再起動してはならない。新作業は別のsource/scope/resultとして保存する。

### 11.1 保存値・式から先に閉じる項目

- G7のaffine期待費用、N/K/hard-capの区別、Rz/accepted factorizationを整合確認する。表示比率をexact sign証明の代わりに使わない。
- 本書§6の受理回数上界と§7のprovider誤差の十分式を独立検討する。必要なら反例を返す。高確率resource capを採用するなら推定failureと資源failureを別に事前配分し、予算を再計算する。元G7のfailure規則を遡及変更しない。
- 既存small supportで可能なcost-aware対照の診断を、追加自由度の帰属を確認する補助として検討する。全表を使った解析で得た改善を非列挙runtimeの証拠にはしない。G7のpoint comparisonは保存する。
- その段階で利益が消えるか、ある条件でしか残らない場合は、その内容を隠さず報告する。G7の正の限定結果を後から否定・再分類しない。

### 11.2 一つのend-to-end取得経路

- G7で固定したP3/P5入力と四方式を維持し、現時点では新しいp、x、m、分子を追加しない。development区分を維持する。
- productionから全parent/reference event表、事前列挙されたnative key表、global B_newへの依存を外す。local queryで得たsymbolic tangentから、必要なstrict Rz sequenceをon-demandに得るか、別の一様合成保証を使用する。
- cold/ warm cache、cache上限、同一tangentの再利用、保守的error、停止上限を全方式で対称に定義する。fullだけに無制限cacheを与え、対照だけをcoldにしない。
- 古典生成・local root/query・angle construction・合成miss・provider description・cache memoryを分けて保存する。whole-circuitやprovider実費が未取得なら、それを未取得のまま残す。
- 独立の小support参照を正しさ照合に使うことはよいが、productionへその表を注入しない。参照計算の費用は別記する。
- finite-provider拡張はまず明示されたerror parameter契約として行う。実際のproviderを新たに選ぶ場合は、その情報access、同じtaskで使える既知方法、error/cost取得範囲を先に固定する。既存契約から決まらない物理的選択を勝手に埋めない。

### 11.3 実行上の裁量と停止

実装ファイル構成、必要bit数、テストやprofilingの技術設計、合理的なcall/wall/CPU/RSS/output上限はCodexへ委ねる。関連修正はまとめて行い、細かなテストごとにGPTへ戻さない。

ただし、新しいdictionary、samplingの科学的意味、同じfinite target、primary指標、比較対象を実質的に変える必要がある場合は停止し、変更理由をGPTへ戻す。error不足を新しい精度gridで救済し続けない。

新規key取得が必要なら、その手順・上限・失敗時扱いを新source/scopeに結果前固定する。native missの試行を際限なく繰り返さない。旧source、contract、authorization、result、consumed marker、STOP、Track A、rootの未commit資料を保護し、必要な新資料だけstage/commit/pushする。

**今回採択しないもの**：独立held-out performance、DF/分子、PR/QPE全体、全面v4、無制限key/seed/precision探索、実機測定campaign。

G8はproof/contract/on-demand pathwayの一括確認であり、研究の主method採択をCodexが代行する段階ではない。

---

## 12. G8後のGPT判断

### 継続を支持する場合

必要情報がlocalに取得でき、finite-provider誤差を払う予算が成立し、低次数対照を同等に扱っても費用上の追加価値または明確な有効条件が残る。この場合に初めて、具体provider・未使用構造に対する一つの検証を設計する。Pauli情報を使える場合にはCTS等の強い対照を戻す。

### 縮小する場合

期待Tの差は保存されるが、native取得が全表や無制限cacheに依存する、古典費用・provider誤差の契約が現実的でない、または安価な対照の追加自由度で差が消える。この場合、全return生成の一般性を主methodとして押さず、理論・構成上の結果と不利域を記録する。

### 判別不能の場合

上界の保守性、取得失敗、特定provider未指定を分離する。「不明」から自動で大きい実験へ進まない。どの不足を解消すれば判断が変わるか、その情報価値が追加費用に見合うかを判断する。

割合の大きさだけでGO/STOPにしない。結果後に0.522%へ合わせた成功閾値を作らない。independent試験を行う場合のmaterialityは、目的taskと許容trade-offから結果前に決める。

---

## 13. 前回G6レビューからの判断変更

G6では「局所生成と認証予算をnative Rz/provider会計へ接続する価値があるか」を検証するG7を提案した。G7は、その限定目的に対してP5の肯定例とP3の非肯定例を返した。

従って、G6時点より継続根拠は強い。しかし、これを主method採択と同じにしない。次に必要なのは、数学的な恒等式確認の反復ではなく、**key取得まで含む非列挙性と、exact oracleを離れた誤差契約**を確定することになった。

旧G5の固定toy・固定6頂点dictionaryの閉鎖は不変。G7のfull-return候補はそのclass外の構成であり、旧結果の救済・再分類ではない。

---

## 14. 参考資料・source locator

以下のGitHub資料は、特記がない限り結果commit `a689694080f4b7600fe67cf77d841d1cbbf04503` に固定した。本文[R#]はこの一覧に対応する。

- [R1] [G7 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)
- [R2] [数学・実行契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md)
- [R3] [Resource summary CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_result/2026-10-10/v1/resource_summary.csv)
- [R4] [原result JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_result/2026-10-10/v1/result_v1.json)
- [R5] [Generator source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/src/trottertracks/algorithm_codesign/g7_generator.py)
- [R6] [Runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/scripts/tracks/algorithm_codesign/g7_budget_control_economics.py) ／ [Reference source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/src/trottertracks/algorithm_codesign/g7_reference.py)
- [R7] [24 key inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_preparation/2026-10-10/synthesis_key_inventory_v1.json)
- [R8] [Provider IR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/src/trottertracks/algorithm_codesign/g7_provider.py)
- [R9] [18 focused tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_preparation/2026-10-10/focused_tests.json)
- [R10] [保存値監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_result/2026-10-10/v1/saved_output_audit.json)
- [R11] [Evidence manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_result/2026-10-10/v1/evidence_manifest_v1.json)

### 外部一次文献

- [L1] Wan, Berta, Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, [arXiv:2110.12071](https://arxiv.org/abs/2110.12071), Appendix C / Algorithm2。今回そのページを画像で確認した。
- [L2] Aomoto, Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988), [primary PDF](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf), §1 / Lemma1.1。今回関連抽出テキストまで。
- [L3] Zhao, Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021), [arXiv:2103.07988](https://arxiv.org/abs/2103.07988), §4.2。今回関連抽出テキストまで。
- [L4] Peetz, Smart, Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026), [DOI:10.1038/s41534-025-01168-w](https://www.nature.com/articles/s41534-025-01168-w), Methods: Convex Taylor sampling / Markov sampling。
- [L5] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, [arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1), Theorem1、§III、§IV。固定protocolでのIS、bias、ZeroFill/Discard。
- [L6] Günther et al., *Phase estimation with partially randomized time evolution*, [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2), revised 2026-07-10、PRX Quantum 7, 020332 (2026), DOI 10.1103/ynxb-p2xq。今回書誌・研究対象を確認。44頁の独立全面精読ではない。

これらの照合は、完全なcitation-network novelty searchや優先性の不存在証明ではない。

## 15. 添付自己検算の説明

`g7_review_work/review_selfcheck.py` は表示CSVの比率と、§6のP5受理回数上界を計算する。repository codeをimportしない。60桁sqrt enclosuresとFractionを使用し、exp(7)>1000は正の有理Taylor部分和で検査した。

`g7_review_work/g7_summary.csv` は[R3]の表示用数値を手元へ転記したもの。元の全result/全cost fractionの代用ではない。主resultのexact signはGitHubの原記録に従う。

`review_selfcheck.json` はこのレビュー計算の出力。新しい独立研究結果やG7の再実行ではない。今後Codexが使う場合は入力位置・hashを確認し、検算を独立結果として重複加算しない。
