# Track B：Parent allocationの継続価値と限定native比較の科学的設計レビュー

- 作成日：2026-10-14（JST）
- 開始承認：ユーザーの「レビューを開始して　丁寧に考えて」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 最新証拠：`bf1366ac5832136ab8038d160e9a0285da69e79e`
- 技術source：`c31e40482cbf2147f75e1730b179966337949f30`
- 前回GPT方針原本：`4d80b45693a1ea7c14d7cbf9035e7358788953bd`
- 対象：研究継続の価値、近接先行研究との関係、Decision Packetの7領域、次にCodexへ委ねる範囲。
- 対象外：論文原稿・投稿可能性の採点、Track A/N3の変更、基底状態情報を使う新研究への転換。
- レビュー状態：**完了。限定native機構比較の科学設計を採択する。ただし性能優位・独立新規性は未認定であり、計算実行は未承認。**
- 運用：2026-10-12版共通指示。旧source/result/authorization/消費済みmarker/closure/mandatory STOPは維持する。

本書で「採択」と記す条件は今回GPTが選んだ新しい科学設計であり、Codex報告に既に決定事項として載っていた条件ではない。本書は実行authorizationではない。登録入力の係数・角度・新しい配分・shot予算・資源表を本レビューでは生成していない。

---

## 0. 最終判断

**Parent allocationを一つの限定的なnative費用比較へ進める価値がある。追加の一般的な準備・文献棚卸しは、今の次段階として要求しない。**

ただし、研究を「新しいEuler恒等式の発見」「新しい一般IS最適化定理」として進めない。任意identity分割とEuler式、固定辞書のcost×moment最適化、固定正価格に対する凸KKTは既知の原理の適用である。一方、今回の具体的な配分変更について、角度依存のnative価格、finite-bit、controlled位相、実装bias、有限confidenceを戻した資源効果はまだ取得されていない。[I1–I4]

次の問いを採択する。

> 同じ有限P5と同じ物理的信号精度を対象に、共通角pairingをchild別配分へ変更したとき、既存pairingにも同じsampling最適化機会を与えてなお、十分測定予算に基づく期待native総T費用が低下するか。その効果はordinaryでも生じるか、closed P5のreturn集約と組み合わせた場合に追加価値が残るか。

この問いに答えるには実際の新角度の合成価格が必要であり、今回までの形式監査を反復しても代替できない。技術準備へ費やした過去の労力そのものは、実験を続ける理由にしない。次の取得が比較的小さいroot部分へ限定でき、positive/negativeのどちらでも具体的な機構仮説を評価できることを理由に選ぶ。

**選択：限定比較へ進む。非選択：広い連続探索、高次数・分子・長時間への同時拡張、SPS全体に対する優越認定、別系列への無根拠な全面転換。**

---

## 1. 取得した証拠と確認の範囲

### 1.1 今回直接確認した内部資料

- 最新の先行研究対応報告：全体のA/B/C/D分類、Euler/ISの対応、SPSのprojected/signed embedding、R0/G1/G5との差、残された未確認事項。[I1]
- Decision Packet：native/cache/guard、runtime、内部候補、root-local/global IS、7領域と科学実行への依存関係。[I2]
- source manifestと、`parent_allocation.py`の係数・端点・proposal生成、`parent_allocation_native.py`のcache bridge・strict guard・native費用会計・sequence順序、既存`numeric.py`のstrict guardとsynthesis入口。[I3–I6]
- 元のGPT方向レビューの採択範囲と既知性・比較原則。[I7]
- D3の表示用原CSVの一部。数値の再集計やランキング更新はしていない。[I8]

GitHub branch `track-b-parent-allocation-prior-art-correspondence-20261014` のremote HEADが`bf1366ac...`であることを確認した。`c31e404...`から`bf1366ac...`へのcompareは3 commits aheadであり、返された変更一覧は文書・索引・receipt等で、algorithm sourceの変更を含まなかった。

今回のsource読取りは関連関数の確認であり、全実装を行単位で完全認証したという意味ではない。13件、62件、26件＋回帰6件という過去の試験はCodexの保存証拠として扱う。GPTは再実行していない。試験群の重複を無視して独立な総試験数へ加算しない。

### 1.2 外部一次資料

GPTが今回直接確認できた本文は、Resource-Optimal Importance Sampling v1のTheorems 1–2、Sparse Probabilistic Synthesis v2のprocess定義・Statements 1–2・design matrix/最適化、SCU/CTSの出版社HTMLの理論・Taylor構成・費用比較の記述である。[P1–P3]

指定SCU arXiv v2とPTSC v2のHTMLには取得エラーがあった。PTSC、Wan、SCU補足の指定式との詳細対応は、本文取得版・式位置を保存したCodexの最新対応報告を継承する。GPTが今回5論文の全本文を独立再読したとはしない。出版社版をarXiv v2とbyte同一と扱わず、版をまたいで式番号を流用しない。

今回、PDFの図表から新しい情報を転記していない。文献の網羅的な不存在証明や全後続実装のpriority認定はしていない。現在の限定比較を設計するための主要内部証拠は取得できている。

### 1.3 実行しなかったこと

新しいHamiltonian・登録provider行列・回路・trajectory・合成・LP・配分最適化・十分N・資源表の生成、既存runner/testsの再実行、GitHub書込み、authorization/marker作成は行っていない。数式の検討、文書・sourceの読取り、本文ファイルの作成を実施した。

---

## 2. 研究全体から見た今回の位置付け

以前の方針レビューは、Return aggregationを勝者に固定せず、回路費用に適したpairingを調べることを第一候補にした。その時点でEuler式やIS原理自体を新発見とは扱っていない。[I7]

今回の近接文献照合は、その既知性をより具体的に確認したものであり、前回の平均保存式を否定する反例ではない。全candidateが既存実装と同じだという結論でも、性能negativeでもない。[I1]

過去の研究から引き継ぐ判断は次のとおりである。これは本レビューで各旧系列を再認証した一覧ではなく、既存レビューが整理した範囲の継承である。[I7]

| 系列 | 今回へ引き継ぐ意味 | 今回しない読み替え |
|---|---|---|
| G系列・D1–D3 | 同じ有限meanでも、係数質量・生成情報・合成価格・測定予算が別々に効く | P5が普遍的に最良、局所改善率が全PR改善率 |
| R1/SP | 回転価格や合成配置の利益と、sampling/bias負担を一体で戻す必要 | 異なるfixtureの改善率の積 |
| G5 | 固定辞書・shares・policy内の強い対照を落とさない | class外に出れば勝てる、全PRの不可能性 |
| BF/BM | 既知対照と同じ情報・最適化機会を与える | 条件を増やせば自動的に新手法になる |
| DF whole・Gaussian・時間相関 | 数理的構成、既知手順との関係、実装利益を分ける | 全てが同じ種類の性能敗北、今回自動再開 |

同じ科学的疑問が残っていることを理由に、技術監査を際限なく増やす段階ではない。今回の新しい作業は、未取得であるnative価格と共通費用会計を、事前に閉じた比較へ戻すことである。

---

## 3. 何を新規性・改良性と呼べるか

### 3.1 平均保存式は既知原理の直接適用

対象parentは

\[
M_u=\zeta\left(aI-i\sigma\sum_i b_iQ_i\right)Q(u),\quad Q_i^2=I,
\]

である。\(z_i\ge0,\sum_i z_i=a\)に対して

\[
c_i=\sqrt{z_i^2+b_i^2},\quad
\phi_i=\operatorname{atan2}(b_i,z_i),\quad
c_i e^{-i\sigma\phi_iQ_i}=z_iI-i\sigma b_iQ_i.
\]

したがって平均保存は、identityの線形分割とEuler式をchildごとに適用して足すことで得られる。[I1] 本書はこの式の優先性を新たに主張しない。

### 3.2 それでも調べる改良余地は残る

共通角では\(z_i^0=ab_i/s\), \(s=\sum_i b_i\)。非common配分では回転角・係数・実装するunitaryが変わる。固定unitary辞書のproposalだけを変えるISとは、この自由度が異なる。[I1,P1]

ただし「自由度が違う」だけでは有効性の証拠にならない。調べるべき対象は、**その自由度を取得・生成・実装する費用を払った後の効果**である。一般枠組みに含まれることと、何も改良できないことは同義ではない。逆に完全同値を見つけられなかったことを、新規性の証明にしない。

### 3.3 SPSとの境界

SPSはprocessを辞書列にして線形制約・係数最適化を行う。[P2] 同じfirst moment \(M=\sum c_iV_i\)でも、\(\sum c_i\mathrm{Ad}_{C(V_i)}\)の全blockは同じとは限らない。off-diagonalの投影や、\((\mathrm{Ad}_{C(V)}-\mathrm{Ad}_{C(-V)})/2\)によるliftを使えば、同じ線形枠組みで表せるというのが最新の照合結果である。[I1]

よってSPSを「channelだから無関係」と除外しない。一方、mean-compatibleな辞書、control・位相、情報取得、費用を定義していない架空の最適SPSを、無料で使える対照にしない。

今回の結論は「Parent allocationはSPSより新しい／優れる」ではない。最初の取得では、既に実装されたpaired表現に対する、特定の配分変更の有効性を判断する。

---

## 4. 数理的な期待と、その限界

### 4.1 Normalizationは増える側にしか動かない

\[
L(z)=\sum_i\sqrt{z_i^2+b_i^2}\ge\sqrt{a^2+s^2}=L(z^0).
\]

したがってこのfamilyの狙いはnormalizationの改善ではない。回転費用の低下が、係数質量・weight・bias等の負担を上回るか、という交換である。[I2]

### 4.2 固定単価モデルは候補を作る根拠だが、native結果ではない

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2}
\]

の固定正価格・正b・正aに対するKKT解は

\[
z_i^K=\frac{b_i\lambda_K}{\sqrt{C_i-\lambda_K^2}},\quad
\sum_i z_i^K=a,
\]

で、一意の内部解となる。等単価ならcommonへ戻る。これは既知IS proxyの代入と標準凸最適化であり、新一般定理ではない。[I1,I2]

実際には\(C_i=C_i(\phi_i,\epsilon,\text{tool/provider/control})\)。新角度で価格が変わり、T数は不連続なので、旧common価格をKKT式へ入れた点は**surrogate候補**にすぎない。実native最適配分ではない。

頂点しか測定していないnegativeから内部配分を排除することはできない。一方、少数の内部候補を加えても連続family全体を排除したことにはならない。

### 4.3 端点と内部点は異なる機構を調べる

- 頂点：一childにparent質量を集中し、他childをphase付きpure-Qへする。回転を消す利益と比較的大きいnormalization負担を調べる。
- 内部候補：全childの回転を残し、費用に応じた小さい／中程度の係数移動を調べる。

この二つを一つの「配分が成功／失敗」のラベルへ潰さない。端点でもproviderのCQや位相処理の費用は残り、回転が消えることを回路費用0としない。[I2,I4]

---

## 5. 継続・別方向・停止の比較

| 選択肢 | 今回の判断と理由 |
|---|---|
| **限定Parent allocation比較** | **採択**。未取得の角度価格を取得すれば具体的な機構を判定でき、入力・mean・変更箇所を固定できる |
| full return/P5の高次数・入力拡大 | 今回は非採択。単なる適用範囲拡大では、現在のpairing変更の有効性に答えない |
| known ISだけを主研究にする | ISは公平な対照として使う。既知式の再実装だけを新しい主たる成果にしない |
| SPSの新しい大規模辞書最適化 | 今回は非採択。比較可能性は認めるが、新しいtarget embedding・辞書取得・solverを同時に立ち上げるのは別の問い |
| 物理的相殺・時間相関・whole補正等への全面転換 | 新しい具体構成や証拠が今回得られたわけではない。旧closureを解除する理由にしない |
| Parent allocationを未測定のまま終了 | 既知恒等式への包含だけでは、具体的な実装利益を否定できない。今回の小さい限定取得は先に行う価値がある |

この選択はReturn aggregationを最終主線へ永久固定する判断ではない。ordinary側で同じ変更が効き、P5が追加利益を持たなければ、その結果に従って配分改善として整理する。

---

## 6. 採択する比較：二つの中心表現と三つの参照対照

### 6.1 中心比較

ordinaryとclosed P5のrootに、**同じ候補生成規則・同じproposal候補・同じ精度・同じ費用取得機会**を与える。非rootの理想構成は保持する。

この比較は、配分変更が既存pairingより有利か、その効果がfull return集約に依存するかを調べるためのもの。

### 6.2 参照対照も新しい共通ledgerへ戻す

既存のpartial-return-tail、closed P3+tail、matched CTSを、既存表現のまま参照対照に加える。配分の新adapterは今回追加しないが、proposal最適化機会と精度・confidence・cache規則は同じにする。既存の係数・IR・合成列を固定出所付きで再利用する。

理由はordinaryが最も強い対照ではなく、既存D3の近接方式も見ないと、ordinaryに対する改善だけを実用的な前進と誤読し得るためである。[I8,I7]

**限定を明示する。** partial/P3+tailにも任意配分を新たに与えた全面比較ではない。したがって中心比較から、全表現を同等に最適化した後にもRAが必要だとは結論しない。matched CTSも保存された特殊化であり、全CTSの最適実装ではない。

SPSの新辞書・solverは今回の必須取得へ含めない。SPSと無関係だからではなく、今回の主張を既存paired実装と保存された参照方式に対する限定的な改良性に留めるためである。独立新規性や全既知法への優位を主張する段階では比較を再検討する。

---

## 7. 配分候補を固定する

各中心表現に、最大次の6候補を与える（L=3）。

1. common-angle \(z^0\)。
2. 全L頂点 \(z^{V_j}=ae_j\)。
3. common-angle rootの各childの**完全なevent単価** \(C_i^0\) を固定proxyとしてKKTから作る有理候補 \(\widehat z^K\)。
4. \(z^{H}=(z^0+\widehat z^K)/2\)。

計数はcommon 1＋頂点3＋内部2＝6、二表現で最大12。KKTと中間点を選ぶのは、価格に基づく内部方向と、その半分の移動を比較するためであり、native価格の滑らかさを仮定した最適探索ではない。

### 候補生成で許す情報

KKTのproxyは、既存common角のvalidated native event単価だけ。新候補の価格を見てzを再選択しない。tool/precision・旧cacheの出所を先に固定し、各表現に同じ規則を用いる。候補生成に使う費用には、rootの回転以外のprovider・controlled位相等も含める。

### 有理化とfreeze

KKT解は一般に無理数なので、exact rationalの配分へ変換する。非負性、exact総和、決定的な丸め、残差配分、固定解像度、tie/失敗処理を新sourceで結果前にfreezeする。これらの数値実装の細部はCodexが設計するが、新しいnative価格を見た後に解像度・候補数を変更しない。理想KKT点との距離を保存し、「exact optimumを実装」とは呼ばない。

commonへ一致する場合や重複候補はその事実を保存し、別候補で穴埋めしない。KKTの正a/正b/正C仮定が外れる場合、無断でfloorを入れず適用外を明示する。内部候補を取得できなかった場合は内部機構のnegativeを出さない。数値実装上の失敗は、数学的な適用外とも性能negativeとも区別する。

### 新合成keyの上限

commonと参照方式の旧cacheが同じ契約でreuseできる前提で、新root positive keyは

\[
2L(1+J)=2\cdot3\cdot(1+2)=18
\]

以下。端点はexact処理、正負回転は同じ保存sequenceのadjointを使う。これはrootの静的上限であって、総worker call・guard・全入力数の上限ではない。旧cacheが不足した場合に未登録の新合成を追加しない。不足を報告し、新規取得範囲の変更が必要なら別途扱う。

登録coefficients/角度/inventoryの生成は別途承認された実行内で行う。候補生成規則と入力refは実行前にfreezeし、実行中の候補値は新価格取得前に保存・hash固定する。

---

## 8. Proposalと正則化の採択

### 8.1 主比較はcanonical＋global IS系

各中心候補と三つの参照対照へ、次を同じ規則で与える。

- canonical。
- 全positive-coefficient eventでT単価が正の場合のglobal解析IS。
- global正則化ISを3種類。

root-local ISは実装機能として保持するが、今回の必須比較へは追加しない。conditionalなcost×momentの最適性と、restを含むglobal最適性が異なることは既に判明しているため、ここで別のroot-constrained optimizerまで設計しない。[I2]

### 8.2 正則化を具体的に定める

五つの既存表現（ordinary、closed P5、partial、P3+tail、matched CTS）の基準canonical lawに対する、期待event T単価の最大値を\(C_*\)とする。これを既存cacheから新しい共通規則で確定し、全候補へ同じscaleとして公開する。

\[
\kappa\in\{C_*/16,\ C_*,\ 16C_*\},\qquad
q_e^{(\kappa)}\propto\frac{\widetilde c_e}{\sqrt{C_{e,T}+\kappa}}.
\]

この集合は**今回選ぶ有限の感度確認用policy**であり、理論的に最適な正則化値ではない。価格scaleを共有し、候補ごとに後から都合のよいfloorを足すことを避ける。新配分の価格を見て\(C_*\)を更新しない。

\(\kappa\)はsampling用parameterであり、actual charged Tに加えない。KKTの乗数\(\lambda_K\)とも別物。3値を使っても新角度の合成数は増えず、古典会計・law準備の件数だけが増える。

T=0のeventが混在すれば、正費用用の無正則化式は不適用と明示する。eventをsupportから削らず、canonicalと正則化を残す。\(C_*=0\)ならこのscaleは定義に使えないため、別のfloorを自動導入せず、canonicalを保持してT基準の退化を報告する。そこでprimaryをCXへ変更して救済しない。

### 8.3 最大law数と主張の限界

中心12＋参照3＝最大15構成、各最大5 proposalで**最大75 law**。重複・適用外があれば記録して減らし、補充しない。

解析ISはcost×second momentの解である。有限confidenceのrange・ceil・bias・hまで最適にするとの主張はしない。[P1] 3つの正則化とcanonicalを比較した結果も、全proposalの最適値ではない。

### 8.4 情報アクセス

主比較は小規模I1、すなわち固定入力の有限event表と必要なnative価格を両表現へ同じ条件で利用可能とする。global IS用の全表の取得・走査・初期化・online費用を記録する。root係数取得がO(L)個で済むことを、global ISやP5全samplingがO(L)であることへ読み替えない。

---

## 9. 物理target・精度・confidence

### 9.1 入力

旧草案の既知development入力を採択する。新held-outとは呼ばない。

\[
p=(1/5,3/10,1/2),\quad R=\sum_i p_iQ_i,\quad x=\tau=5/7,
\]
\[
Q_0=Z_0,\quad V_1=R_{XX01}(\pi/4),\quad
V_2=R_{XX12}(\pi/4)R_{ZZ01}(\pi/4),\quad Q_i=V_i^\dagger Z_iV_i.
\]

\(R_P(\theta)=\exp(-i\theta P/2)\)、\(\sigma=+1\)、\(\rho=|000\rangle\langle000|\)、r=1、finite target \(M=P_5(-ixR)\)、physical target \(U=e^{-ixR}\)。controlled first operator momentのRe/Imを同じtaskとして比較する。

この設計は厳密基底状態を入力に要求せず、厳密基底状態でないことも利用しない。真の信号・固有状態重み・exact ground情報を配分選択やshot削減へ使わない。

### 9.2 精度

各axisの最大絶対誤差を\(\epsilon_{m ax}=1/200\)とする。両axisを満たせば複素誤差は\(\sqrt2/200<1/100\)。primitive synthesis precisionは\(10^{-6}\)、root/probability bits K=256/H=160、\(\rho=\eta=10^{-12}\)を旧草案から採択する。atan/guardの精度は固定した旧runtime・guard規則へbindし、結果を見て上げない。

\(\|R\|\le1\)は正のnormalized係数とinvolutionから従う。Hermitian実時間の積分剰余によって

\[
\delta_{\rm Tay}\le |x|^6/6!
\]

を全方式共通に用いる。位相や行列normの規約が異なる保証を混ぜない。

### 9.3 旧草案から変更するconfidenceの意味

**新ledgerでは「選択した一つの信号推定taskに対するfailure probability 0.05」を採択する。** Re/Imに各\(\alpha_{m ax}=0.025=1/40\)を与える。比較law数では割らない。

これは今回の科学的選択であり、旧草案の「全候補の全axisが同時成功するfamilywise保証」を黙って引き継いだものではない。

理由：今回の費用表は測定値から推定したランキングではなく、固定係数・sequence・解析予算から計算するforecastである。将来実行するlawを、量子測定値を使わずに選ぶなら、各選択lawのfailure上界\(\alpha\)から、選択後taskにも同じ上界を与えられる。75種類の未実行の反事実的task全ての同時成功を要求する必要はない。

形式的には、量子outcomeと独立な取得情報Dからs(D)を選び、各sが\(\Pr(E_s\mid D)\le\alpha\)を満たせば、\(\Pr(E_{s(D)})=\mathbb E_D[\Pr(E_{s(D)}\mid D)]\le\alpha\)。信号測定を見てlaw選択する設計へ変更する場合はこの議論を使えず、別の選択後保証が必要。

全新候補・全対照をこの同じ新alphaで会計し直す。旧D1/D3の保存Nをそのまま比較したり、旧結果を書き換えたりしない。将来全75推定値の同時保証を求めるなら別taskとしてfamilywise配分が必要である。

CPU超過やcache欠損はここでの統計alphaと別であり、成功結果を出さずtechnical stopとする。実行完遂確率や量子総費用のhigh-probability上界を今回のalphaから保証したとはしない。

---

## 10. 誤差・weight・十分測定予算の共通会計

各eventのideal係数をc、digital係数を\(\widetilde c\)、実dyadic proposalをqとする。全positive supportでq>0、\(w=\widetilde c/q\)を守る。

\[
\nu=\sum_e\widetilde c_e^2/q_e,\qquad
W=\max_e\widetilde c_e/q_e,\qquad
\overline C_j=\sum_e q_e C_{e,j}.
\]

Hadamardの一回outcome \(Y\in\{-1,+1\}\)では\(X=wY\)なので二次momentは\(\nu\)、分散は\(\nu-|\mathbb E X|^2\le\nu\)。本比較は状態依存の正確な分散をoracleで与えず、この上界を使う。

係数誤差と実装誤差は

\[
e_{\rm rep}\le\sum_e|\widetilde c_e-c_e|+
\sum_e\widetilde c_e\delta_e
\]

へ戻す。\(\delta_e\)は同じexact ideal eventへのstrict native誤差上界。sourceのRZ guardがexact atan targetとの差を既に含む場合、同じangle丸め誤差を重ねて課金しない。別のangle近似を追加した場合だけその分を別途加える。[I4–I6]

qで補償している確率丸めはmean biasへ二重計上しない。proposalを変えただけなら同じdigital実装meanを保つが、配分でc/Vを変えれば実装biasも変わり得る。[P1,I1]

旧比較で用いた保守的axis変換係数2を全方式で保持し、

\[
\beta_s=2(\delta_{\rm Tay}+e_{{\rm rep},s}),\qquad
\Delta_s=\epsilon_{\rm ax}-\beta_s
\]

とする。この係数を片側だけ改善しない。\(\Delta_s\le0\)ならそのlawは精度不適格。guard未取得による不明とは区別する。

\[
N_s=\left\lceil\log(2/\alpha_{\rm ax})
\left(2\nu_s/\Delta_s^2+4W_s/(3\Delta_s)\right)\right\rceil.
\]

実lawの\(\nu,W\)と認証済み\(\Delta\)から計算する。sqrt/logの区間とceilの安全な扱いをCodexで実装する。policyの十分Nであって、最小必要shots・実測量子shotsではない。

未使用のbias余裕は、全候補で同じ式により統計幅へ戻す。この結果の差を「angle価格だけの改善」とは呼ばず、N・単価・bias寄与を分離して報告する。

---

## 11. Primaryと費用の意味

Primaryは、h=0における**十分Nに基づく期待logical T費用**。

\[
G_{s,T}(0)=2N_s\overline C_{s,T}.
\]

これは準備費用が実際に無料という主張ではなく、共通費用を除いた切片としての比較である。準備・読出し単価\(h_j\ge0\)については

\[
G_{s,j}(h_j)=2N_s(\overline C_{s,j}+h_j)
\]

を併記し、既存の固定law集合の下で順位が残る領域・反転点を示す。hごとに新proposalを無断最適化しない。

CX、native1Q、workspace、N、最大weight、最大event費用、classical setup/online、合成取得費用は副次・説明指標。Tはnative1Qに含まれるため合算しない。未知の古典費用を0にせず、秒とTを換算規則なしに足さない。

費用は共通のliteral簡約規則と、保存されたClifford+T sequenceの加法計数に基づく。whole-circuit compiler最適性、fault-tolerant実時間、一般分子PR/QPEの総費用ではない。

期待費用とhard費用を区別する。希望した確率での総量子費用capは今回の主張ではない。最大event費用から得る自明な上限等は補助情報として扱えるが、新しいtail保証を実装済みとしない。

---

## 12. 比較結果の判定規則

### 12.1 「候補集合を増やしたから最小値が下がった」を成果にしない

表現sのcommon対照を

\[
B_s=\min_{q\in\mathcal Q_s(z^0)}G_{s,z^0,q,T}(0),
\]

非common候補を

\[
A_s=\min_{z\in\mathcal Z_s\setminus\{z^0\},\ q\in\mathcal Q_s(z)}G_{s,z,q,T}(0)
\]

と別に定義する。取得・精度適格な候補のみを比較し、欠測と適用外を明示する。

新候補集合にcommonを含めた最小値は集合包含だけでnon-worseとなるので、その事実をpositiveとしない。**異なる実装の非common候補について、\(A_s<B_s\)が数値区間を越えて示されること**を配分利益の条件とする。intervalが重なる場合は符号未確定。

正確な差、比率、bias、N、単価、取得費用を報告し、最低5%/10%等の新しいmateriality閾値は設けない。小差を一般的頑健性とも無価値とも即断しない。

### 12.2 二段階の意味を分ける

1. **機構の比較**：ordinaryとP5のそれぞれで、同じproposal機会を与えたcommonより非commonが良いか。
2. **既存実装集合での有用性**：partial/P3+tail/CTSを含む参照対照の最良値にも届くか。

第一がpositiveでも第二がnegativeなら、配分の局所改善はあるが、現参照集合の最良方式を更新していない。この二つを一つの科学的GOにしない。

同じ候補生成規則と同じproposalの固定比較も保存する。最小値比較だけでは、違いが配分・ISのどちらから生じたか分からない。

### 12.3 結果別の次判断

| 結果 | 次に判断する内容 |
|---|---|
| 非commonがcommon＋同じISを下回り、参照集合にも競争力がある | 有限条件での改良として支持。どの機構を独立条件へ移すかをGPTで検討 |
| ordinaryにもP5にも配分利益 | RA専用機構としない。配分・表現設計として進める候補 |
| ordinary＋配分がP5側より良い | P5を救済する条件探索へ進まない |
| 共通角＋ISで利益が説明される | 配分固有の追加価値を小さく評価する |
| 配分は効くがpartial/P3+tail/CTSが依然低い | 機構positiveと実装集合での非更新を両方保持。自動的な分子拡張はしない |
| 頂点のみ良い | 回転をexact endpointへ置換する機構として解釈。内部KKTのnative有効性の証明ではない |
| 内部のみ良い | 頂点だけでは検出できなかった条件付き利益として解釈。native最適点の発見とはしない |
| 全有限候補で利益がない | 今回のcandidate-generation policyの不支持。連続familyのno-goではない。追加探索は新しい根拠がある場合だけ再検討 |
| 欠測・guard/保存/数値failure | technical inconclusive。成功prefixだけで勝敗を作らない |

positive/negativeを受けた継続・縮小・終了の最終科学判断は結果取得後に行う。今回の実験採択は、その結果に対する包括的GOではない。

---

## 13. 7領域の採択一覧

| 領域 | 今回決めた内容 |
|---|---|
| 1 比較方式 | 中心ordinary/closed P5。参照partial/P3+tail/matched CTS。SPS新最適化なし、全既知法優越を主張しない |
| 2 配分候補 | 中心各common＋全3頂点＋KKT surrogate有理点＋半移動点。最大12中心構成。候補追加・価格probe反復なし |
| 3 Proposal | canonical、適用可能なglobal解析IS、global正則化3値。root-localの別最適化は追加しない |
| 4 正則化/zero cost | 共有common価格scale C_*、κ={C_*/16,C_*,16C_*}。actual費用へ加算しない。zeroの解析式は不適用 |
| 5 精度/confidence/budget | 既知3-qubit、r1、P5、x5/7、正時間。axis誤差1/200、選択taskのfailure .05、axis1/40。primitive1e-6、K256/H160、rho/eta1e-12。新root key最大18、各一回、shot cap1e9/axis |
| 6 Primary | 十分N×期待Tの切片。h領域とCX/1Q/workspace/古典費用を併記。実機時間・最小shotsではない |
| 7 解釈 | 配分利益・RA差・IS差・参照集合更新を分離。有限candidateのnegativeを全familyへ拡張しない。priority未認定 |

CPU/RSS/output/log/guard件数などの**技術的な実行上限**は、上記科学的取得範囲からCodexが既存runtimeの根拠を使って具体化し、実行承認前のcontractへ固定する。未採択の旧「6合成・600 CPU秒」を新計画へそのまま転記しない。wallの上限は既存個別運用どおりデフォルトなしとし、tool polling timeoutと区別する。

ここで科学条件を決めたことと、実行資源contractが完成したことは別である。技術上限を決めるために新しい科学的比較を自動実行しない。

---

## 14. Cache・native validation・実行範囲

旧cacheは1e-6の指定取得源に固定する。D3の1e-7価格と恣意的に混ぜたり、旧/newの安い方をkeyごとに採用したりしない。tool/version/options/seed/package/runtime/guard/strict phaseと原bytesの対応を新inventoryへbindする。

現native accountingはguard再確認済みentryだけを受け付ける。[I4] この契約を維持し、**再利用するunique旧sequenceも新sourceの固定guardで一度確認する**設計とする。これは新合成ではないが登録演算子評価なので、実行承認の範囲へ明示的に含める。metadataだけでguard済みflagを立てない。

新合成はunique root key最大18、各一回、retry0。inverseは実sequenceのadjoint。endpointはexact記述。旧列と新列に同じstrict判定を与える。sequenceに含まれるWを落とさず、projective一致へ緩めない。

採択providerへ接続した新root eventのcontrolled作用は、native記述と独立したideal定義を照合する。非rootの既存証拠はreuseし、全D2 pairや元D3 runnerの再実行は不要。必要な登録matrix/guard件数はsource inventoryで結果前に数え、追加scopeとして実行承認へ記載する。今回のレビュー内では実施していない。

全取得が揃う前に最終winnerを出さない。費用0のvalid eventとmissing priceを区別し、失敗時も取得済みraw・resource receipt・原因を保存する。全結果のclassificationは新namespaceへ保存し、旧terminalを更新しない。

---

## 15. 取得費用・古典費用・適用範囲

今回のI1比較でglobal ISが使えるのは、有限event/価格表を両者へ同じ条件で与えるからである。一般involutionの非列挙生成で全cost表が安価に手に入るとの主張ではない。

少なくとも次を別々に残す。

- root係数・有理候補作成のsetup。
- finite-bit法則、全表IS法則の初期化。
- sampling/lookupのonline実装と、測定した場合のその条件・回数。
- 旧価格情報の取得元とreuse。
- 今回の新合成・guardの取得費用。
- 比較実験全体の共有取得費用と、選んだ一方式を使う際の取得費用の区別。

T数と古典秒を無次元で合算しない。未測定項目は未測定とする。共有cacheへ入った新angleを、候補側には課金し対照には無料、またはその逆といった非対称な扱いにしない。

このpilotで示せるのは、特定provider・tool・precision・有限候補・共通policyにおけるlogical資源効果である。独立入力への一般化、全compiler最適性、実機時間、分子DFアクセス、deterministic PR部分、長時間、QPE/RPEまで含む総優位は別の未検証事項として残る。

---

## 16. 次の担当・作業単位・承認境界

**次の担当はCodex。上記で採択した科学設計を、一つの新しい実行契約・sourceへ実装する。**

一般的なDecision Packetや先行研究対応表をもう一度作る作業ではない。必要な変更は、有限候補生成とfreeze、参照方式の新ledger接続、global proposal群、誤差・N会計、同一cache政策、実provider guard、inventory/worker/保存とresource guards、launch gateである。

数値丸め・interval/certificationの実装、schema、保存分割、process管理、登録外のfocused tests、独立技術監査はCodexへ一括して委ねる。既存source・新仕様・source manifestの対応を固定し、原本Markdownは必要に応じて改変せず保存する。

**準備中に登録入力の候補値・新価格・資源表を生成しない。** 登録入力を使うcandidate generation、旧列guard、新合成、native評価、十分N・比較表の生成は、固定source/contractに対するユーザーの別途明示実行承認後に限る。

順序は以下。

1. 今回：GPTの科学設計レビュー完了。
2. Codex：採択条件に沿う実行source・contract・登録外検証・根拠付き技術capsを一括準備。
3. ユーザー：凍結された範囲の一回実行を別途承認。
4. Codex：同一の認可済みrunで候補freeze→必要価格取得→全law会計→監査・公開→mandatory STOP。
5. 結果に重要な科学的判断が必要になった時点で、ユーザーのレビュー開始承認後にGPTが解釈する。

通常のsource改訂・test PASSだけを理由として、同じ科学設計のGPT詳細レビューを繰り返さない。既に別途固定された独立source-review条件が存在するなら維持する。今回新たに、全工程へGPT承認を追加することはしない。

### GPTへ戻す必要がある変更

- 指定targetやcontrolled first-moment意味論が実装できない。
- 必須対照・precision・candidate/proposal・alpha意味論を変えないと比較が成立しない。
- cache不足等により新合成scopeが上限を超える。
- 近接既知手順との完全対応等、研究目的に影響する新証拠が出る。
- 重要なnative結果が揃い、拡張・縮小・終了を決める。

小さい数値処理・保存・互換性修正は、採択条件を変えない限りCodexで解決する。

---

## 17. 確かさと残る不確実性

高い確かさがあるのは、平均保存が既知Euler式の直接適用であること、commonのfamily内norm最小性、固定正価格KKTとそのnativeへの移送限界、root-localとglobalの区別、現在の技術sourceが科学実行sourceではないこと。

限定比較を次に行う価値があるという判断の確信は中程度。実際のT利益、価格に対する頑健性、特定表現への依存、追加取得費用、強い辞書最適化を与えた場合の関係は未確定である。

**本レビューは「もう少し資料を集めてから考える」という判断ではない。どの限定比較を行うかを選んだ。** その科学設計をCodexが実行仕様へ落とし、別途承認された取得から結果を得る段階へ移る。ただし研究主線の恒久的な確定、独立新規性、科学実行の無条件許可にはならない。

---

## 参照資料

### 内部固定証拠

- [I1] Parent allocation近接先行研究との厳密対応。`bf1366ac5832136ab8038d160e9a0285da69e79e`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf1366ac5832136ab8038d160e9a0285da69e79e/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md
- [I2] 科学設計Decision Packet。同commit。今回§1–6を中心に分割読取り。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf1366ac5832136ab8038d160e9a0285da69e79e/docs/tracks/algorithm_codesign/parent_allocation_scientific_design_decision_packet_20261014.md
- [I3] Source manifest。同commit。現在の科学契約・実行認可がfalseであることと保存試験identity。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf1366ac5832136ab8038d160e9a0285da69e79e/artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/source_manifest.json
- [I4] Native/cache/guard source。`c31e40482cbf2147f75e1730b179966337949f30`。関連部分を読取り。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31e40482cbf2147f75e1730b179966337949f30/src/trottertracks/algorithm_codesign/parent_allocation_native.py
- [I5] Parent adapter/proposal source。同source commit、主に1–165行を読取り。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31e40482cbf2147f75e1730b179966337949f30/src/trottertracks/algorithm_codesign/parent_allocation.py
- [I6] 既存strict RZ guardとsynthesis入口。同source commit。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31e40482cbf2147f75e1730b179966337949f30/src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py
- [I7] 前回GPT方向レビュー。`4d80b45693a1ea7c14d7cbf9035e7358788953bd`。GitHubの結論と、会話添付原本の既知性・比較・STOP条件を確認。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md
- [I8] D3表示用CSV。`f0df9a3b36bc3540f0c83941871fa0d3110ddbf7`。今回冒頭15行を確認。丸め表示でありexactな再認証ではない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/resource_rows_display_v2.csv
- [I9] 技術監査原本。今回はI1/I2の対応箇所と会話内に取得済みの本文を継承。旧草案・入力条件の参照。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c12558d682f6aef3cfc68e8d75082269f2282ac8/docs/tracks/algorithm_codesign/parent_allocation_technical_audit_20261014.md
- [I10] 旧最小比較草案。上記commit。本レビューの新採択条件と区別する。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c12558d682f6aef3cfc68e8d75082269f2282ac8/docs/tracks/algorithm_codesign/parent_allocation_minimal_comparison_draft_20261014.md

### 外部一次資料

- [P1] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。今回§II Theorem 1と§III Theorem 2の本文を確認。CLT/net-costと有限confidence policyを区別。
  https://arxiv.org/html/2603.13495v1
- [P2] Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, arXiv:2402.15550v2。今回§II.1–II.3、Statements 1–2、Eqs.(1)–(6)の関連本文を確認。
  https://arxiv.org/html/2402.15550v2
- [P3] Peetz, Smart, Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026)。今回は出版社HTMLの理論、Theorem 1、CTS・費用記述を確認。指定arXiv v2の再取得には失敗し、補足式対応はI1を継承。
  https://www.nature.com/articles/s41534-025-01168-w
- [P4] Wan–Berta–Campbell、arXiv:2110.12071v2、およびZeng–Sun–Jiang–Zhao、arXiv:2212.04566v2。今回GPTの独立全本文再読とはせず、I1の取得版・式locator・技術対応を使用。
  https://arxiv.org/abs/2110.12071v2
  https://arxiv.org/abs/2212.04566v2

本書は科学的判断の記録であり、各原資料の数値・実行証拠・source manifestを置換しない。
