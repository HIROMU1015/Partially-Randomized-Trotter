# Track B G6 科学的研究レビュー
## 非列挙return集約の評価、既知対照との追加価値、予算上界と実装費用の次検証

- 作成日：2026-10-10 JST
- レビュー開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-b-g6-return-generator-audit-20261010`
- 固定result commit：`28cfabb1d47e0e1824bce1e154a9c289738fa2b9`
- 直前基点G5：`23adf61f9342abff20f059be69089e89f66f7797`
- 本レビューの推奨：**限定継続。次は予算と費用を明示した最小実装実証。主method採択・一般資源優位・独立新規性は未確定。**
- 旧固定toy／同辞書の実用優位実験主線は閉鎖を維持する。G6を旧J1実験の救済としない。

## 0. 読み方と今回の証拠境界

本書では、次の三種類を分ける。

**資料上の結果**：G6の独立数学監査、finite-bit/access監査、prior-art表、source、test、result、manifestに記載され、今回読み取った内容。[R1–R7]

**本レビューの新しい導出**：三次での既知partial-return対照との展開比較、全normalizerを計算しない上界U、digital二次モーメント上界、三次閉形式対照とordinary tailの比較、controlled-Qからの費用を明示した回転実現。これらを既にG6で認証された結果とは扱わない。

**GPTの科学的判断**：何を次に検証する価値があるか、どのclaimを留保するか、論文として何を目指すか。repositoryの実行statusを書き換えるものではない。

今回、固定branchのremote HEADは指定commitと一致した。主要文書、prototype、test、raw result、manifestが取得可能であり、必須資料の未pushを理由とする停止は必要ない。[R7,R8] 一方、一般Qのnative provider、角度合成費用、実taskの総資源が未検証であることは、資料公開の不備ではなく研究上の未解決事項である。

今回の独立性は限定的である。証明・sourceを読み、別の局所算術自己検算を行ったが、G6のtechnical runnerをcold replayしたわけではない。全hashを別cloneから再計算したとも、native circuitや量子measurementを実行したとも主張しない。自己検算のscriptと出力は同梱する。

---

## 1. 最終判断

**G6候補は、主methodとして採択せず、実装費用を伴う成立性を判別する候補として一段だけ継続する。**

理由は、既知Green関数を言い換えただけの部分と、それを有限Taylorの全return集約・parent pairing・normalizer不要の局所生成へ接続した部分を分けると、後者には具体的な数理構成と有限bitの到達点が残るためである。[R1–R4] ただし、その接続が先行研究から直接得られるか、十分非自明か、実装費用後にも有用かは未確定である。[R4]

次の問いは、単にB_newがB_ordより小さいかではない。この点はG6の条件内で既に示されている。次に判別すべきなのは、

> **全returnを局所的に集約するための古典計算、可変角度、control、有限bit誤差、認証可能なshot予算を支払っても、非列挙ordinaryと安価な低次数return対照を上回る採用理由があるか。**

である。

今回の検討から、特に二点が重要になった。

1. **三次では、ordinaryからの大きいnormalization改善の主項は、既知の低次数returnですでに得られる。全集約の追加差は短時間でO(x^4)である。**
2. **未知受理率の上界を1のまま使うzero-fill予算では、normalizationがより小さくても、既知returnより量子呼出しが多くなる場合がある。**

一方、G6の局所係数から、全wordを列挙せずにB_newの上界Uを計算できる。この新しい予算上界を独立確認し、明示的なcontrol/rotation accessと組み合わせる小さい実装実証には情報価値がある。

**次担当はCodex。G7として予算・access・局所角度費用の検査を一つにまとめ、終了後にGPTへ戻す。** 同じ数学監査を繰り返すこと、大規模DFに直行すること、新しいp/seedを増やして勝つ例を探すことは推奨しない。

---

## 2. G6で何が確認されたか

### 2.1 到達点

G6の技術statusは `G6_IDEAL_IDENTITIES_VERIFIED_NATIVE_AND_METHOD_VALUE_UNRESOLVED`、raw resultは `G6_FORMAL_CHECKS_PASS` である。[R1,R6]

| 層 | G6で確認されたこと | まだ含まれないこと |
|---|---|---|
| 有限平均 | 同じfinite P_mの全形式return集約、phase、word順序 | exponential／PR＋QPE全体の誤差 |
| 非負性 | 有限正rational p、odd m、0<x<=1の十分条件 | x>1全域 |
| 係数取得 | free-product Green式による非列挙の局所query | 新しいrandom-walk理論としての優先性 |
| pairing | 偶数suffixと奇数childrenを単一Q_iのrotationで実現 | 奇数word全体をinvolutionとして扱うこと |
| envelope | B_new<=B_ord、局所accept、normalizer不要のzero-fill | native総資源が減ること |
| finite-bit | dyadic proposal、rational weight、明示biasとrangeの局所packet | 有限bitで元のirrational meanを誤差ゼロ実現すること |
| native | symbolic tangentと回路時間順序まで | controlled回路、gate synthesis、費用・実行 |
| 新規性 | 既知componentsと候補差を整理 | 新method成立・投稿優先性 |

16 focused tests、raw形式語3,965、母関数係数3,794、positive語503、signed formal mean12、digital event weight63等の照合が報告されている。[R6] これらは一般証明の補助であって、有限点一致だけを一般命題の証明とはしない。

科学実行、native synthesis、quantum matrix/circuit/measurement、LP、DF/GPUは0。prototypeのrandom drawsも0である。[R6] 実行済みsamplerによる資源測定と誤認しない。

### 2.2 Source確認

`return_aggregation.py` は、全reduced-word宇宙やB_newを列挙しない。first-passage seriesを前計算し、要求されたparentとchildrenの係数を局所計算する。`digital_parent` は固定bitで証明できるpacketを返すが、native circuitを作らない。[R5]

確率の丸めを補正weightで相殺し、係数の近似誤差は残す、という分離は適切である。`pi * W = alpha_tilde` の有理数恒等式は、`alpha_tilde = alpha_ideal` を意味しない。[R3,R5]

sourceの読解とG6証明の範囲では、今回の判断を覆す明確な恒等式の誤りは特定しなかった。ただし、これは全sourceや全将来入力を新たに独立認証したという宣言ではない。

---

## 3. 新規性についての評価

### 3.1 既知と認める部分

| 部品 | 外部一次資料／G6監査の位置付け | 本レビューの扱い |
|---|---|---|
| free-product Green multiplier | Aomoto–Kato 1988、§1とLemma 1.1 | Z2特殊化自体は新規性にしない |
| 高次数項をidentity／低次数へ吸収 | Zhao–Yuan modified Taylor | return absorption原理や三次式だけの新規性は主張しない |
| ordinary RTEのEuler pairing | Wan–Berta–Campbell、Algorithm 2／Appendix C | 既知ordinaryはorderとIID labelで非列挙生成できる |
| CTSのPauli集約と層別生成 | Peetz–Smart–Narang、本文とG6の補足監査 | 全Pauli展開evaluatorだけを対照にしない |
| cost-aware IS | Cugini–Atif–Subaşı Theorem 1 | sampling最適化だけの貢献にしない |
| zero-fillとnormalization | 同論文IV.1–IV.2、一般的なimportance weighting | 全試行平均と受理のみ平均を区別する |

今回、WanらのAlgorithm 2をページ画像でも確認した。ordinary samplerは次数を引き、その後IID labelsを引く。したがってG6の古典速度を評価する相手は、全列挙した古いpilot evaluatorではなく、streaming ordinaryである。[P3]

Aomoto–Kato PDFは関連する抽出本文を参照できたが、ページ画像取得は失敗した。G6の明示した対応式を読み、既知free-product式との関係を評価したものの、全スペクトル定理を再証明していない。[P1,R4]

CTS本文のMarkov/layering記載を確認した。補足PDFのこのターンでの独立取得は成功していない。G6が補足の関連箇所を取得・比較したことはG6の監査報告に基づく事実として扱い、自分が同じ全文を取得したとはしない。[P4,R4]

### 3.2 zero-fillの違いを落とさない

Cuginiらのerror-detectionのZeroFillは、量子実行後にerror flagが得られる設定を扱う。G6は局所係数から**量子実行前**に棄却し、棄却試行では量子回路を走らせない。この違いは費用上重要である。[P5,R2]

ただし、零寄与も分母へ含めることや、受理のみ平均にはnormalizerが必要なことは既知の重み付け原理である。前処理で棄却するという語だけを新規性にしない。

### 3.3 残る研究候補

残るのは、

> 固定finite P_mの全形式returnを保ったまま、全係数表・未知normalizerを要求せず、有限bitとcontrol/角度の費用を明示して生成できる構成。

である。G6の確認範囲で全結合lawの同一記載が特定されていないことは、候補を検討する理由にはなるが、priorityや非自明性の証明ではない。[R4]

逆に、各部品が既知であることだけから、全結合した方法に価値がないと結論することもできない。構成の保証、必要情報、計算量、強い対照に対する具体的な追加価値を一つの主張へできるかが判断点になる。

---

## 4. 数学的な対象を固定する

以下でB_oはordinary finite RTE、B_nはG6の全形式return集約後のnormalizerを表す。添字nは次数ではなくnewの略である。

\[
R=\sum_{i=1}^L p_iQ_i,\quad Q_i^2=I,\quad
M=P_m(-i\sigma xR),\quad t_l=x^l/l!.
\]

G6の仮定は有限の正の有理p、odd m、0<x<=1、sigma=±1。x=0はidentityとして別扱い。異なるQ_iの追加関係は用いない。[R2]

uをreduced word、l=|u|、P_n(u)をraw IID wordがuへ戻る確率とすると、

\[
a_u=\sum_{n=l,l+2,\ldots,m}(-1)^{(n-l)/2}t_nP_n(u),
\quad M=\sum_u(-i\sigma)^{|u|}a_uQ(u).
\]

偶数parent uについて、

\[
s_u=\sum_{i:\,iu\,\mathrm{reduced}}a_{iu},\quad
 d_u=\sqrt{a_u^2+s_u^2},\quad
 b_l=\sqrt{t_l^2+t_{l+1}^2},
\]

\[
B_n=\sum_{u\,\mathrm{even,reduced}}d_u,\qquad
B_o=\sum_{l\,\mathrm{even}}b_l.
\]

G6はd_u<=b_l p(u)、B_n<=B_oを示している。[R2]

この不等式は、ordinaryに対するweight normalizationの非悪化である。任意の実装費用、任意のsampling、既知returnやCTSに対する非悪化ではない。

---

## 5. 新しい評価①：三次では既知returnの後に何が残るか

本節はG6の三次式と、既存R0/G3で使ったpartial-return式からの**本レビューの導出**である。[R2,R9]

\[
\chi=\sum_i p_i^2,\quad \mu_3=\sum_i p_i^3,\quad \mu_4=\sum_i p_i^4.
\]

m=3では、

\[
a_\emptyset=1-\chi x^2/2,
\qquad s_\emptyset=x-(2\chi-\mu_3)x^3/6,
\]

\[
B_n=\sqrt{(1-\chi x^2/2)^2+[x-(2\chi-\mu_3)x^3/6]^2}
+\frac{x^2}{2}\sum_jp_j(1-p_j)
 \sqrt{1+x^2(1-p_j)^2/9}.
\]

比較する既知partial-returnは、R^2=chi I+Dを使い、P3の低次数側へそのreturnを吸収した構成である。

\[
B_r=\sqrt{(1-\chi x^2/2)^2+(x-\chi x^3/6)^2}
+(1-\chi)\frac{x^2}{2}\sqrt{1+x^2/9}.
\]

**B_rは全既知方法の最適値ではない。** これより強い低次数吸収やPauli集約が可能な場合がある。この一つの対照を超えただけで実用上最良とは言えない。

### 5.1 短時間展開

\[
B_o=1+x^2-\frac7{72}x^4+O(x^6),
\]

\[
B_r=1+(1-\chi)x^2+
\left(-\frac7{72}+\frac\chi{18}\right)x^4+O(x^6),
\]

\[
B_n=1+(1-\chi)x^2+
\left(-\frac7{72}-\frac\chi6+\frac{\mu_3}4-\frac{\mu_4}{36}\right)x^4+O(x^6).
\]

従って、

\[
\boxed{
B_r-B_n=\frac{8\chi-9\mu_3+\mu_4}{36}x^4+O(x^6)
=\frac{\sum_i p_i^2(1-p_i)(8-p_i)}{36}x^4+O(x^6).
}
\]

L>=2、全p_i>0なら係数は正。L=1では係数0であり、三次では両者が同じ一つのscaled rotationになる。

**ordinaryからのO(chi x^2)改善の主項はpartial returnでも得られる。全集約の追加部分を評価するには、O(x^4)の差とその取得費用を比較する必要がある。**

これはm=3の短時間展開であり、一般mに同じ係数を転用しない。またnative TのO(x^4)定理ではない。gate synthesisの費用は角度について滑らかとは限らない。

### 5.2 三次にはGreen関数なしの閉形式対照がある

m=3の偶数parent jk (j!=k)について、

\[
d_{jk}=\frac{x^2}{2}p_jp_k\sqrt{1+x^2(1-p_j)^2/9},
\qquad \tan\phi_{jk}=x(1-p_j)/3.
\]

kを足すとp_j(1-p_j)になり、normalizerは上記のO(L)和で計算できる。parent先頭j、k!=j、child i!=jを条件付き分布から引けるため、L^2個のparent表やGreen級数は不要である。

これは、**G6の三次式から直接作れる閉形式対照**である。特定の既知論文がこの全lawを同じ形で記載していると確認したわけではない。低次数吸収原理は既知であり、三次だけの優位を一般Green-kernelの価値として主張するのは弱い、という比較上の指摘である。

一般奇数m>=5には、

\[
P_m(-ixR)=P_3(-ixR)+\sum_{l=4,6,\ldots,m-1}
\left[(-ixR)^l/l!+(-ixR)^{l+1}/(l+1)!\right]
\]

を使い、**全P3集約＋ordinary高次数tail**という同target対照を作れる。そのnormalizerは

\[
B_{3+\mathrm{tail}}=B_n^{(3)}+\sum_{l=4,6,\ldots,m-1}b_l.
\]

この対照も本レビューで明示的に構成したもので、旧実験のbaselineを書き換えるものではない。次の一般次数候補の比較に用いる。

---

## 6. 新しい評価②：zero-fillの予算が利点を消す場合

G6の理想zero-fillは

\[
Z=B_n/B_o,\quad X=B_o\mathbf1_{\rm accept}Y,
\quad E[X^2]=B_oB_n,\quad |X|\le B_o.
\]

全試行Nに対するBernstein十分条件は

\[
N\ge\ell\left(2B_o^2Z_+/s^2+4B_o/(3s)\right),\quad Z_+\ge Z.
\]

未知Zを推定せずZ_+=1とすることは安全だが、最も鋭い予算ではない。[R2]

### 6.1 比較を透明にする診断

次の計算だけでは、**同じ統計余裕s、受理回路一回の費用が同じ、二次モーメント項が支配的、range・ceil・bit誤差を除外**という仮定を明示する。これは原因を切り分ける算術診断であり、native性能の予測結果ではない。

Z_+=1のzero-fillで期待量子呼出しは

\[
E[K_q]\simeq 2\ell B_oB_n/s^2.
\]

既知partial-returnのcanonical samplerでは

\[
N_r\simeq2\ell B_r^2/s^2.
\]

従って三次の比は

\[
\boxed{\frac{E[K_q]}{N_r}\simeq\frac{B_oB_n}{B_r^2}
=1+\chi x^2+O(x^4).}
\]

短時間でB_n<B_rでも、この保守的予算では量子呼出しが増え得る。G6の恒等式の誤りではなく、**使える上界と予算設計の問題**である。

期待量子呼出しNZはhard capではない。実行上限を保証するなら、最大Nを使うか、受理数の確率上界とそのfailure allocationを別に組む必要がある。期待値をそのまま最大実行回数としない。

---

## 7. 本レビューの新しい導出：全normalizerを計算しない上界U

### 7.1 raw-reduced質量の安価な計算

R_lを、長さlのraw IID wordが**隣接同一labelを含まない確率**とする。これは、長さlのwordをreduceした後の長さ分布ではない。

末尾iで終わるその質量をv_{l,i}とすると、

\[
v_{1,i}=p_i,\quad R_l=\sum_i v_{l,i},\quad
v_{l+1,i}=p_i(R_l-v_{l,i}),\qquad R_0=1.
\]

全l<=mのR_lはO(Lm)の有理算術で計算できる。

### 7.2 rootを正確に扱い、残りはenvelopeで上から押さえる

G6のempty-parent local queryでd_emptyを計算し、

\[
\boxed{
U=d_\emptyset+\sum_{l=2,4,\ldots,m-1}b_lR_l
}
\]

と置く。

各lでsum_(|u|=l,reduced) p(u)=R_lと、d_u<=b_l p(u)から、

\[
\boxed{B_n\le U\le B_o.}
\]

このUはB_nの全parent和を計算しない。追加費用はroot local queryとO(Lm) recurrenceであり、G6の前計算を含めてO(Lm^2)規模の有理算術で構成できる。平方根を上向き区間にするbit費用は別途数える。

m>=3、x>0ならd_empty<b_0なのでU<B_o。m=1はU=B_n=B_o。L=1ではU=B_nだが、これは単一involutionの特殊ケースである。

m=3、L>=2では、上記closed formから

\[
B_n<U<B_r<B_o
\]

を示せる。rootのs_emptyがpartial-return側より小さく、tailでは1-p_j<1を使う。大きい一般mでU<B_rのような命題を自動的に主張しない。

### 7.3 理想shot予算

Z_+=U/B_oを使えば、

\[
\boxed{N_+=\left\lceil\ell\left(2B_oU/s^2+4B_o/(3s)\right)\right\rceil}
\]

という安全な全試行数を構成できる。実際にはU、B_o、logの上向き区間とsの下向き区間で評価する。Uのpoint値を上界と呼ばない。

これは全wordをsampleしてZを推定する方法ではないため、別のZ推定failure probabilityや全normalizerを無料で使う必要はない。

### 7.4 有限bitへの二次モーメント上界

G6のrelative-proposal boundと係数近似boundを仮定する。

\[
\pi_{u,i}\ge(1-\eta)^{m+2}\alpha_{u,i}/B_o,
\qquad 0\le\widetilde\alpha_{u,i}\le(1+\rho)\alpha_{u,i}.
\]

すると、

\[
\begin{aligned}
E[\widetilde X^2]
&=\sum_{u,i}\frac{\widetilde\alpha_{u,i}^2}{\pi_{u,i}}\\
&\le \frac{(1+\rho)^2}{(1-\eta)^{m+2}}B_o\sum_{u,i}\alpha_{u,i}\\
&\le\boxed{\kappa B_oU},\qquad
\kappa=\frac{(1+\rho)^2}{(1-\eta)^{m+2}}.
\end{aligned}
\]

rangeは

\[
W_+=\frac{(1+\rho)B_o}{(1-\eta)^{m+2}},
\]

係数biasはrho B_oである。[R3の仮定を用いた本レビューの新しい帰結]

native error等を戻した残りs_eff>0に対して、

\[
N\ge\ell\left(2\kappa B_oU/s_{eff}^2+4W_+/(3s_{eff})\right)
\]

を評価する経路がある。**digital varianceが理想B_oB_nと等しいとはしない。** またtilde acceptance率をideal Zにそのまま置き換えない。

この上界の実装・前提チェックはG7の独立監査対象である。G6の既存certificateを後から拡張認証済みと記述しない。

---

## 8. 保存した自己検算と、その解釈

### 8.1 三次の具体例

G6の既知形式fixture p=(3/7,4/7)、x=2/5、m=3において、本レビューの別実装算術は次を返した。旧G5のp=(3/4,1/4)を再採点していない。

| 量 | 表示値 |
|---|---:|
| B_o | 1.1577409398291492 |
| 既知partial-return B_r | 1.0766944539949335 |
| 全集約B_n | 1.0754454989248340 |
| 非列挙上界U | 1.0757035203450838 |
| B_o B_n / B_r^2 | 1.0740261244357017 |
| U B_n / B_r^2 | 0.9979207292855750 |

最後の二つは§6.1のequal-event-cost／equal-s／variance-leading診断である。前者は約7.40%増、後者は約0.208%減に対応する。**実際のT countの増減率ではない。**

さらに強い、G6三次式を直接使うclosed-form対照B_n自体と比べると、U B_n/B_n^2=U/B_n>1である。三次にGreen generatorの優位を要求する設計は、直接同じ構成を生成する対照を見落とす可能性がある。

### 8.2 五次では三次closed formと分けて比較する

同じくG6で既に使用された形式fixture p=(1/5,3/10,1/2)、x=5/7、m=5では、

| 量 | 表示値 |
|---|---:|
| 全集約B_n | 1.288444557675… |
| 上界U | 1.296755874824… |
| 三次全集約＋ordinary tail | 1.3001972810223033 |
| U B_n / B_(3+tail)^2 | 0.9883379066690582 |

この診断には約1.17%の余地があるが、取得費用・native角度・bit誤差を戻す前の値でしかない。これは「新手法が1.17%速い」という結果ではなく、次の費用検査で超えてはいけない追加負担が小さいことを示す目安である。

この一例を見て成功閾値を逆算しない。既知G6 fixtureを再利用したdevelopment解釈であり、独立held-outではない。

### 8.3 自己検算の範囲

6つのG6形式fixture、合計503 reduced wordsについて、次を確認した。

- 形式wordの左乗算分布oracleと、同期power-series fixed pointによるGreen係数の一致。
- raw-reduced質量R_lのO(Lm)再帰と小さいoracleの一致。
- B_n<=U<=B_oの有理区間照合。
- 三次normalizerの閉形式、quartic係数、比較診断。
- 三次closed-form＋ordinary高次数tailの同target normalizer。

B_nの全parent和を使うのは、この小さい**自己検算oracle側**だけである。非列挙生成器の実行方法として採用していない。

平方根は192-bit dyadic interval、係数はFraction、有理数の符号で確認した。数値表示にfloatを使っても、それを保証判定に使っていない。native synthesis、quantum matrix/circuit/measurement、LP、RNG、ネットワーク、repository code importは0。

この自己検算はGPT内の別実装であって、Codex・別機関による独立認証ではない。G6の16 testsへ加算しない。script作成・確認過程で複数回実行したもので、登録one-shot runではない。

---

## 9. native accessはどう閉じるか

### 9.1 現G6にないもの

一般Q_iのcontrol、可変角度rotation、workspace、strict error、native costは未実装である。[R3] これを「Q_i oracleがあるから無料」と扱ってはならない。

実装契約としては少なくとも二種類ある。

| 契約 | 実現方法 | 費用として残すもの |
|---|---|---|
| 明示gate-description Q_i=V_i^dag P_i V_i | V_i、controlled-Pauli rotation、V_i^dag | basis回路の費用・誤差、rotation合成、phase |
| controlled-Q_i provider | helperにeigenvalueをcomputeしてRz、uncompute | controlled-Q_i二回、CRz、helper1、H gates、provider error |

どちらも新たに提案するaccess契約であり、G6がこのnative実装を済ませたことにはしない。

### 9.2 controlled-Qが与えられる場合の具体的構成

補助aを|0>にし、

\[
E_i=H_a C_a(Q_i)H_a,
\qquad \Pi_\pm=(I\pm Q_i)/2.
\]

このとき

\[
E_i|0\rangle|\psi\rangle
=|0\rangle\Pi_+|\psi\rangle+|1\rangle\Pi_-|\psi\rangle.
\]

従って

\[
E_i^\dag R_z(2\sigma\phi)_a E_i
\]

は、helperを|0>へ戻してe^(-i sigma phi Q_i)を実現する。外側のHadamard control cで条件付ける場合は、中央だけをC_c Rzにすればよい。c=0ではcomputeとuncomputeが相殺する。二重controlled-Qを無料で要求する必要はない。

必要な操作は、controlled-Q_i二回、H四回、controlled-Rz一回、追加helper一つ。parentのwordには別にcontrolled-Qを|u|回使い、parent phaseも外側controlへ付ける。

このcompute–rotate–uncomputeは標準的な構成原理であり、新規性とはしない。目的は、必要accessと隠れた費用を明示することである。

G5のworkspace1と比較するなら、こちらは外側Hadamard controlに加えhelper1を必要とする。workspaceを据え置いたと記述しない。明示V_i契約ならhelperを避けられる可能性があるが、その場合はV_i費用を戻す。

### 9.3 誤差

unitaryな近似computeと、その**実際のadjoint**をuncomputeへ用いる。理想Eとの差delta_E、中央回転との差delta_Rがstrict normで抑えられるなら、組立誤差は2delta_E+delta_R以下である。parent wordの誤差は別に足す。

平均operatorへの係数誤差・native誤差と、Hadamard signalへの変換係数はtaskに合わせて固定する。旧規約のfactor2を片側のmethodだけ落として比較しない。global phase最小化後のerrorをcontrolled phaseのstrict certificateとして使わない。

### 9.4 query modelとnative modelを分離する

controlled-Qを入力providerとして扱うなら、最初に得られるのは

`controlled-Q呼出し数 + 実合成した可変Rz費用 + workspace + 古典処理量`

という資源vectorである。providerの実T/CX/errorが未指定なら、分子実装の総T費用は確定しない。

これは有効な条件付き理論・実装モデルにはなるが、Pauli/DFを実装した実験と同じではない。全QをPauliとして指定した場合は、より直接的なPauli rotationやCTSも使える。弱いoracle実装を対照だけに強制しない。

---

## 10. 何を費用に入れるか

候補の総量を次のように分解して報告する。

- setup：pの読込・sampler準備、Green系列、normalizer上界、共通primitive準備。
- 各古典trial：label生成、raw-reduced判定、local parent/child query、root/bit処理。
- 受理時：新しいangleの合成／cache lookup、controlled event、state preparation、Hadamard/readout。
- 後処理：補正weightの高精度演算、zero試行を含む集計。

CPU秒とT gatesを無根拠に足してscalar scoreにしない。まず別資源vectorとして示す。cold acquisitionとwarm-cache executionを分け、cacheを全word表へ育てて非列挙性を失っていないかも報告する。

角度はparent histogramとfirst labelに依存し得る。m=3ならO(L)種類へまとめられるが、一般L,mでO(m)種類に収まる保証はない。[R3] 避けたはずの全係数表を、全angle合成表として前払いしていないかを確認する必要がある。

「normalizationが少し小さい」ことと、「古典queryが多項式」なことだけでは、上記全費用の有利性は導かれない。

---

## 11. 次の科学的問いと着地点

### 11.1 主RQ

> **明示的なpと、費用・誤差を伴うcontrolled involution／rotation accessが与えられる場合、全returnを保持する局所生成は、streaming ordinaryおよび安価な低次数集約対照と比べ、認証可能な有限精度予算と実装費用の交換に独立した価値を持つか。**

未知normalizerを予算に使わないこと、全word/cost表を要求しないことを特徴として維持する。ただしそれを満たすだけで新規性確定とはしない。

### 11.2 強い完成形

一般mでの構成・非負性・生成計算量・finite-bit保証に加え、明示accessでの費用上界または成立域を示す方法研究。単なる固有のtoy勝敗表ではなく、どの条件で局所query費用を支払う意味があるかを説明する。

### 11.3 限定的な完成形

新規性やnative利益が小さくても、有限Taylorと既知Green関数の接続、nonnegative envelope、normalizerなしの有限bit生成を正確に整理した構成ノートとして残す余地はある。ただし、これだけで独立論文に十分とは断言しない。

### 11.4 旧R0–G5との関係

旧成果と新候補を、無理に一つの成功物語へまとめない。旧固定degree classがCTSに劣ったことは保持する。新候補はsigned returnをsampling前に集約する別classであるが、classを変えた事実そのものは新規性・性能の証拠ではない。

---

## 12. 推奨する次のCodex作業：G7を一つの限定実装実証にする

### 12.1 G7の役割

G6の一般恒等式を再び一式検査するだけの作業にはしない。G7は、**今回の予算上界をfinite-bitへ接続し、controlled可変回転の隠れた費用が残る余地をどれだけ消すかを判別する作業**とする。

本レビューでは、科学的採択の前にこの一つの実装実証へ進むことを推奨する。独立条件の性能campaign、全PR/QPE、分子・DF、旧固定toy再開は認めない。

### 12.2 まとめて任せる内容

**A. 予算・強い小対照の独立確認**

U、raw-reduced質量の再帰、digital second-moment上界、三次展開、三次closed-form＋ordinary tailの同target性を独立に確認する。B_nを小さいoracleで算出した値は検算に限り、実運用のshot予算へ流用しない。

**B. control・有限bit・局所angleの結合実装**

primaryはcontrolled-Qのgate-description/providerを明示し、§9のcompute–rotate–uncomputeを用いて必要呼出し・helper・strict errorを数える構成にする。providerがV_i^dag P_i V_iとして利用可能ならその直接実装も別契約で明示できるが、方法間で片側だけ有利な実装を使わない。

providerの物理費用が未指定の場合、全結果をquery/native-Rz/CPUの条件付きvectorとして報告する。一般分子の総native改善と呼ばない。provider自体の新規探索をこの作業へ混ぜない。

finite-bit root/proposal/weight、実際のRNGまたは決定的bitstream interface、受理前のzero、controlled phase、uniform error、十分なNを接続する。実量子measurementは必要ない。小さいsynthetic matrixでのsemantic確認は可能だが、実験信号をtruthとしてbudgetを下げない。

**C. 小さい既知development入力での費用診断**

三次はG6のp=(3/7,4/7),x=2/5を意味論・closed-form対照に用い、一般次数の最小判別は既存G6のp=(1/5,3/10,1/2),x=5/7,m=5とする。これらは旧G5性能toyでも新held-outでもなく、G6から使っているdevelopment入力である。

m=5,L=3のreduced even parentは1+6+24=31個。この小さいsupportを別の検算・費用参照側で列挙することは許容できるが、production generatorが全31表を必須入力として読む設計へ置換しない。列挙oracleの時間を、非列挙生成器の時間と混ぜない。

比較には少なくとも、streaming ordinary、既存のpartial-return構成＋必要なordinary tail、三次closed-form全集約＋ordinary tail、G6全集約を含める。すべて同じfinite P_m、control、残りaccuracy、error semantics、初期化/readout規則で扱う。三次だけのknown absorption効果を一般Green生成の勝利として数えない。

可変Rz synthesisを行う場合は、上の有限parentと対照が要求するsymbolic angleを先にdeduplicateしてkey一覧を固定する。precisionは共通error契約から先に決め、同じbackend・strict phase条件で一回取得する。angle、p、x、m、seed、precisionを結果後に追加して改善を探さない。

これはlimited implementation-economicsであり、CTS/全known法に対する最終比較ではない。実Pauli contextで優位を主張する段階には、literal CTSおよび利用可能なlayered/nonenumerative手法との同task比較を戻す。

### 12.3 技術裁量と境界

Code分割、interval precision、unit tests、resource上限、error allocationの技術的詳細は、上記の意味論を維持してCodexがまとめて設計・結果前固定してよい。技術手順ごとのGPT承認は不要。

ただし、研究対象・主要対照・主metric・独立性を変える必要が生じたら、その科学的変更を行わずGPTへ返す。あるsource correctionがsource-bound one-shotの後なら、旧markerを使って再実行せず経緯を保存する。

G7のsynthesis/key数・wall/CPU/RSS/output上限は、固定key inventoryと既存環境を踏まえてlaunch前に記録する。新しいruntime導入やbackend探索を無制限に進めない。完了・失敗いずれでもmandatory STOP。旧G5閉鎖とG6 markerを保持する。

### 12.4 何をもって次のGPTへ戻すか

| G7の結果 | GPTでの次の判断 |
|---|---|
| 正しいfinite-bit予算・accessが実装でき、低次数対照後にも費用上の余地が残る | 方法上の差と実費を再評価し、初めて独立なnative条件の必要性を検討 |
| ordinaryには勝つが三次closed form等の対照で説明し切れる | 一般Green-generatorの主claimを縮小。native条件を増やして救済しない |
| 角度取得・control・古典queryが利点を消す | この実装routeの限界として記録。別providerを都合よく追加しない |
| 数学・bit保証に反例／不整合 | 反例と適用範囲を返す。次の性能取得を停止 |
| 証明／費用上界が保守的で判別不能 | 真の非改善と認証不足を分け、追加作業の情報価値をGPTで判断 |

新たな5%/10%の成功閾値を旧結果から逆算しない。厳密な符号、実装上の不確かさ、他資源、取得負担を別に示す。T-primaryを維持し、補助1Qだけを結果後に主要metricへ格上げしない。

---

## 13. 採らない選択肢

**母関数が既知なので直ちに全候補を終了する**：採らない。G6が接続した有限mean・局所生成・finite-bitは具体的な方法候補であり、既知部品の組合せだけでは価値を否定できない。

**G6が16 tests PASSなのでnative／分子へそのままGO**：採らない。既知return対照、unknown-normalizer budget、control・angle費用が未閉鎖である。

**旧固定toyのJ1探索を再開する**：採らない。G5のclass-wide閉鎖を保持する。

**任意のpやbasisで勝つ例を探す**：採らない。まず既存形式入力に対する方式上の不足を閉じる。新入力を選ぶなら、次レビューでstructural contrastとその理由を決める。

**全word表を取得してから高速lookupする方法を非列挙として評価する**：採らない。取得費用を省けばG5で指摘された問題に戻る。

---

## 14. 今回実施したこと・していないこと

### 実施

固定GitHub資料の取得、branch HEAD一致確認、数学・source・testsの読解、一次文献関連箇所の確認、6個の既存G6形式fixtureでの独立な局所自己検算、三次の展開計算、予算上界の導出、研究方針比較、本Markdownと再現資料の作成。

### 未実施

repositoryの変更、G6 runner／旧scienceの再実行、native synthesis、新しいquantum matrix／circuit／measurement、LP、DF・分子、PR/QPE総資源評価、独立held-out、全先行文献の不存在証明。

G6が取得したCTS補足全文を、このレビューで独立に全取得したとは言わない。Aomoto–Katoのページ画像は取得失敗しており、抽出本文とG6の対応監査を使った。WanらAlgorithm 2はページ画像で確認できた。この取得範囲の違いは、新規性が未確定であることと合わせて保持する。

---

## 15. 再現資料

同梱directory：`g6_review_reproducibility/`

- `g6_review_selfcheck.py`：stdlibのみの局所形式算術。repository codeをimportしない。
- `g6_review_selfcheck.json`：有理区間、6 fixtures、三次係数・Uの照合、明示的な三次＋tail対照。
- `selfcheck_console.txt`：最終scriptのconsole出力。
- `README.md`：実行方法と証拠境界。

script SHA256：`a28ee6e768826f2694943f21495f9bd26e0e2fa5ebaa75affab852fc27d4bfe3`

result SHA256：`3a4ed36214e34670a8224a0589be2587176148f2c655d3efd57a9c396f3c2334`

これらは本レビューの自己検算であり、G6の原結果を置換しない。Codexへ渡す際には、独立導出・実装を求め、期待値をコピーしてテストを通すことを目的にしない。

---

## 16. 参考資料・一次文献

### Repository資料（すべて固定commitのURL）

- [R1] [G6 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)
- [R2] [G6 independent mathematical audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)
- [R3] [G6 finite-bit and access audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)
- [R4] [G6 prior art and method delta](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)
- [R5] [Prototype source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py)
- [R6] [G6 raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/artifacts/track_b_g6_return_generator_audit/2026-10-10/result_v1.json) ／ [tests source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/tests/tracks/algorithm_codesign/test_g6_return_aggregation.py)
- [R7] [G6 manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/artifacts/track_b_g6_return_generator_audit/2026-10-10/evidence_manifest_v1.json)
- [R8] [G6 branch ref（照合時点のlive endpoint）](https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/git/ref/heads/track-b-g6-return-generator-audit-20261010)
- [R9] [R0 independent proof：既知partial-return](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md)
- [R10] [G5 reviewのrepository snapshot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/research/track_b_G5_research_direction_review_20261010.md)

### 一次文献

- [P1] K. Aomoto and Y. Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988). [本文PDF](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)。関連§1の抽出本文、G6の式対応を参照。
- [P2] Q. Zhao and X. Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021). [arXiv:2103.07988](https://arxiv.org/abs/2103.07988) ／ [本文PDF](https://arxiv.org/pdf/2103.07988)。modified Taylorの低次数吸収、§4.2。
- [P3] K. Wan, M. Berta, E. T. Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, Phys. Rev. Lett. 129, 030503 (2022). [arXiv:2110.12071](https://arxiv.org/abs/2110.12071) ／ [本文PDF](https://arxiv.org/pdf/2110.12071)。Algorithm 2、Appendix C。
- [P4] J. Peetz, S. E. Smart, P. Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026). [出版本文](https://www.nature.com/articles/s41534-025-01168-w)。本文のCTS／Markov記載とG6補足監査。補足の独立全文取得は今回未完。
- [P5] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 (2026-03-13). [本文](https://arxiv.org/html/2603.13495v1)。Theorem 1とZeroFill/Discard。原論文のerror-detection flagとG6のpre-quantum rejectionは異なる設定。
- [P6] *Phase estimation with partially randomized time evolution*, [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2)。今回は版情報を確認し、詳細対応はG6の関連節監査に依拠する。v2全文を今回精読したとはしない。

---

## 17. 結論の再掲

**旧G5の実験主線は閉じたままにする。G6候補は、既知Green関数の再発見ではなく、有限平均を保つ局所生成の候補として限定継続する。**

ただし、ordinaryに対するnormalization減少だけを成功条件にしない。三次で直接生成できる全return対照、未知normalizerを使わない予算、有限bitによる増分、control・angle・古典query費用を一つの比較へ戻す。

次のG7はその一点を判別する限定実装実証であり、主method採択や独立論文の成立を先取りしない。結果が出た後、GPTが継続・理論ノート化・停止を判断する。
