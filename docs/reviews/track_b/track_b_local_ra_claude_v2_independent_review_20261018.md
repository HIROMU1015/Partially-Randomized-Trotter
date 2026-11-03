# Track B：局所RA生成・有限bit構成の独立科学レビュー
## Claude v2の漸近解析、zero-fill費用会計、Pauli対照、論文化判断の検証

- 作成日：2026-10-18（JST）
- 作成者：GPTによる独立科学レビュー
- 開始承認：ユーザーの「レビューを開始して」。直前に合意した5論点の範囲。
- 状態：**レビュー完了。研究方針の正式採択・科学実行承認ではない。**
- 主対象：Claude v2 `レビューinner_2.md`。v1は履歴として保持する。
- 適用共通指示：`research_project_instructions_unified_2026-10-17.md`（本文を実際に確認）。
- 実施：添付原本の読解、固定GitHub資料・SOURCEの静的読解、関連一次文献の対象箇所確認、既存式の紙上検算、条件付きの論理・費用比較、研究判断。
- 未実施：研究SOURCEのimport/実行/変更、Hamiltonian・回路生成、simulation、compile/T合成、保存科学値の再集計・再ランキング、実分子χ_Rの算出、数値最適化、GitHub書込み、契約・grant・marker・claimの作成/消費。
- ローカルで実施した管理処理：受領Markdownの存在・bytes・SHA256確認、本レビューMarkdownの作成・検査。外部PDFの直接取得を一度試みたがDNS失敗。科学計算ではない。

## 0. 結論

**現在の保存証拠だけでTrack B単独の新アルゴリズム論文を投稿できるとは認定しない。追加の分子走査・native計算も現時点では推奨しない。一方、Claude v2をそのまま採択して「独自部分の資源価値が漸近的に否定された」と研究を区切ることには反対する。**

理由は、Claudeの理想的な二次momentの展開自体は整合するが、それを期待量子費用へ移す段階に重要な会計の欠落があるためである。G6のzero-fillでは、拒否は量子回路を実行する前に判明する。したがって、全古典試行数Nと、量子回路の実行回数NZ^rは異なる。さらに、全normalizerを求めずに利用できる上界Uが既にG6レビューとG7 SOURCEに存在する。これらを戻すと、「G6の漸近的量子費用削減は最大χ/2」という一般化は導けない。[G6-M] [G6-R] [G7-S] [COST]

実務上の当面保留はClaudeの選択肢Cに近いが、科学的根拠と主張の強さは変える。

| 対象 | 本レビューの判断 |
|---|---|
| G6の有限Taylor mean、正値性、局所係数query、finite-bitの基本整合性 | 対象仮定の下で保持。今回の論点に関わる核を紙上確認 |
| fixed m・fixed pでの理想二次momentの短step主項 | 保持 |
| m=3のpartial−fullのx^4係数 | 紙上再導出が既存式と一致。保持 |
| 二次moment下界だけからの総量子費用NO-GO | 採用しない |
| zero-fillの総量子費用は漸近的にχ/2までしか改善しない | 一般命題として不成立。採用する予算上界・課金位置等に依存 |
| Pauli-awareの二次係数質量比較 | 係数の集約・重複処理と上界/等式を修正した限定命題として保持 |
| 固定頂点の多stepへの懸念 | 保持。ただし任意proposalでpure確率O(x)とはいえず、総費用には追加条件が必要 |
| 方向Aの独立アルゴリズム貢献 | 現時点では証拠不足。全体が既知法だという完全同値も未証明 |
| 方向Bの既存資源診断 | 有用な限定結果。新しい一般的選択則は未確立 |
| 現時点の単独投稿・追加大規模実験 | 保留を推奨。数学的に論文化不能と確定したわけではない |

**実務的な推奨が似ていても、誤ったNO-GOを成果として固定しないことが重要である。** 本書はClaude原本や既存SOURCE・結果を改稿・置換しない。

## 1. 証拠、取得範囲、確かさ

### 1.1 受領原本のidentity

管理上のSHA256を今回ローカルで確認した。

| 原本 | bytes | SHA256 |
|---|---:|---|
| Claude v2 `レビューinner_2.md` | 63,362 | `027a7940da027c43394240bcbadbd4e3facdbfdff4e0b9c8c1add4e980a827b7` |
| Claude v1 `レビューinner(3).md` | 60,381 | `078547a373c2085cc2c15a1ccaee8b8f188ab2cf33a85b80146b9d340162319a` |
| 共通運用指示2026-10-17 | 38,191 | `d2ebdac4b0e44e871c65f240f317e6b5d3c3460f27d92f4d798992da90bf422f` |

主対象はv2であり、v1の原本も変更していない。Claudeが記載する過去の取得・検算履歴を、GPT自身がすべて再実行したとは扱わない。

### 1.2 今回直接取得した主なGitHub資料

| ID | 固定資料 | 実際の確認範囲 |
|---|---|---|
| G6-M | G6独立数学監査、carrier `bc649c592c8c2b625b49094b85c27e380faa3cf4` | 全文。特に§7–8の受理率、全試行分母、NZ会計 |
| G6-F | G6 finite-bit/access監査、同carrier | 全文 |
| G6-S | `return_aggregation.py`、`28cfabb1d47e0e1824bce1e154a9c289738fa2b9` | 全文静的読解。実行なし |
| G7-S | `g7_generator.py`、`ab2549f41b3546fb3940342a2162dd9ee93699c4` | 1–150行。上界U、quartic helper、canonical partial/P3+tailの定義 |
| G9-S | `g9_p5.py`、`0ef2b92738750a3c0d187743dd4b7c9b927db802` | 取得された全文。group構成・sampling・reference-only oracleの区別 |
| G6-R | 既存GPT G6レビュー、`101bf3426fcd04cc0c01c0d202d0916268765d97` | 冒頭、§5–7と周辺。m=3展開、保守予算、既存U |
| COST | Track B資源構造資料、同commit | §3.1–4。特に受理済みfull circuitだけに量子費用を課す会計 |
| S4-R | S4科学レビュー完成版、`e3086f48c5598c8bb31b0587e807ca6c1b3d2754` | §4.1–4.6の関連箇所。特に多step下界の限定とIS。全rawの再監査ではない |
| S4-C | Parent allocation正式closure、`101bf3426fcd04cc0c01c0d202d0916268765d97` | 正式比較集合、約14%/19%、準備費用、多step、D＋E判断の記述 |
| CLAIM | 最新Paper Claim Sheet、`aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14` | 全文。A/Bの中心候補と未解決事項 |

GitHub connectorによる固定refの取得である。新しいcloneで全ファイルのraw SHA256を再計算した、remote全branchの最新状態を再監査した、全テストをcold replayした、という主張はしない。SOURCE/実行結果/公開carrierのSHAを混同しない。

Claudeへの旧依頼文にあったG6 carrierの `8b2` / `8c2` の誤記は、本レビューでは正しい `bc649c592c8c2b625b49094b85c27e380faa3cf4` を使った。

### 1.3 外部一次資料と取得限界

- WBC：arXivの非version指定PDFを取得し、表示ヘッダからv2を確認。Appendix C、(C4)–(C5)、Algorithm 2をPDF画像でも確認した。[WBC]
- Resource-Optimal IS：`2603.13495v1` HTMLを取得。§IIのcost×second-momentとCauchy–Schwarz、§IVのZeroFill/Discard会計を確認した。[IS]
- SCU：`2407.21095v2` HTMLの理論枠組み・channelとcoherent cross termを確認。全Supplementや全変種の再認証ではない。[SCU]
- SPS：`2402.15550v2` HTMLのsigned decomposition、process列、L1/実装費用の枠組みを確認。[SPS]
- Zhao–Yuan：arXiv非version指定PDFの§4.2周辺を取得し、掲載版のページ画像で式(28)–(29)等を確認。version固定bytesの一致までは未確認。[ZY]
- Aomoto–Kato：Numdamの一次PDFのparsed本文でpp.62–65と乗法性・multiplierの位置を確認。PDF screenshotの取得は失敗し、壊れた記号列を根拠に細部の新しい対応証明を行っていない。自由積再帰自体は紙上で確認した。[AK]
- PR：指定v2 URLは取得できなかった。非version指定PDFのparsed本文とscreenshotには版・式番号の不一致があったため、同じv2の検証と扱わない。screenshotで確認できたのは**v1**（2025-03-07表示）のp.13、p.20等である。このv1の§VII.Bに、異なる軌道基底のinvolutionを乱択に用いる一般化が明記されていることは確認した。Claudeが読んだv2の全内容・Appendix番号まで独立照合済みとはしない。[PR-v1]

本レビューは網羅的priority searchではない。未取得資料が存在することを既知法の不存在や候補の新規性の証拠にしない。以下の主要な費用訂正は、外部PR v2の取得に依存せず、固定されたG6/G7/COST自身の式で成立する。

### 1.4 根拠の区分

本文の「紙上確認」「本レビューの推論」は既存式を論理的に検算・比較したものである。既存数値の再集計、追加benchmark、SOURCE実行ではない。保存された改善率は原資料の表示値の引用にとどめた。

確かさは定性的に表す。任意の75%/90%を査読通過確率・定理の真偽確率として割り当てない。

## 2. G6の核となる数学は保持できる

### 2.1 対象と不変量

\[
M=P_m(-i\sigma xR),\qquad R=\sum_i p_iQ_i,\quad
p_i>0,\quad\sum_i p_i=1,\quad Q_i^\dagger=Q_i,\ Q_i^2=I.
\]

mは奇数、0<x≤1。形式語の簡約は隣接同labelの除去のみ。異なるlabel間の関係やPauli反交換を使わず、Q(u)の順序を保持する。

有限和をreduced word uごとにまとめると

\[
a_u=\sum_{n\ge|u|,\ n-|u|\text{ even}}^m
 (-1)^{(n-|u|)/2}\frac{x^n}{n!}P_n(u),\quad
M=\sum_u(-i\sigma)^{|u|}a_uQ(u).
\]

この恒等式と位相は紙上で確認できる。奇数語Q(u)そのものをinvolutionとする仮定は不要であり、そのように扱ってはいけない。[G6-M] [G6-S]

### 2.2 正値性・Green再帰・parent pairing

隣接pairを決定的に一つ除去し、挿入位置とlabelを記録するcoverにより、

\[
P_{n+2}(u)\le(n+1)\chi P_n(u),\qquad \chi=\sum_i p_i^2
\]

を得る。従って隣接する交代和の項の比はχx²/(n+2)≤1/2となり、

\[
\frac{x^{|u|}}{|u|!}p(u)\left(1-\frac{\chi x^2}{|u|+2}\right)
\le a_u\le \frac{x^{|u|}}{|u|!}p(u)
\]

が成立する。x>1には拡張しない。

自由積の木上の初到達分解から

\[
F_i=zp_i+zF_i\sum_{j\ne i}p_jF_j,
\qquad G=\frac1{1-z\sum_i p_iF_i},
\qquad \sum_nP_n(u)z^n=G\prod_{i\in u}F_i
\]

を得る。SOURCEは切断級数の係数比較を実装している。ここは既知Green構造の特殊化と整合するが、原論文がfinite Taylor sampler全体をそのまま提供するという意味ではない。[G6-M] [G6-S] [AK]

偶数parentについてs_u=Σ_i a_iu、d_u=√(a_u²+s_u²)、tanφ=s_u/a_uとすれば、一involutionのEuler式でparentと全childのmeanを再構成できる。これはWBCなどのpairingとの近接性が強い。[G6-M] [WBC]

### 2.3 受理率の主語を修正する

G6の証明が直接与える

\[
A_u\ge\frac1{2\sqrt2}
\]

は、**有効なreduced parentが提案された条件付きの受理率**である。raw語自体が非reducedならその前にzeroとなる。

全試行の受理率はZ=B_full/B_ordで、原監査が別に示す安全な下界はZ≥1/(2e)。Claude v2 §8.2の狭い命題で「受理率≥1/(2√2)」とだけ書くと、この二つが混ざる。全試行Zにも同じ数値を主張するなら別の根拠が必要である。本レビューはその全体下界が偽だと数値反例を出したのではなく、**引用されている証明の対象が条件付きA_uである**と訂正する。[G6-M]

### 2.4 有限bitとP5

実proposal π、近似係数c̃、weight c̃/πについてπW=c̃という有理等式は成立する。一方、元のirrational係数と回転を実装したmeanへの誤差は別で、係数bias、確率近似からのrange/moment上界、角度・native誤差を会計する必要がある。これらの整合はG6-F/G6-Sで確認した。finite-bitで元のMへ無条件にexact一致するとはいわない。

P5のroot・長さ2・長さ4の群、L²+1以下の群数、Green kernelを呼ばない専用経路はG9 SOURCEから確認できる。全P5係数を独立oracleで再実行したわけではない。群構成のO(L²)と全samplingの費用は別であり、dyadic分布生成のsortingや再生成、線形group scan、bit演算等は別会計である。[G9-S]

これらを「既知要素を含むから価値ゼロ」とは判定しない。ただし実装の正しさだけで独立新規性を認定もしない。

## 3. 命題Rのうち保持できる部分

### 3.1 理想・固定入力の短step展開

固定したp、L、奇数m≥3でx→0とする。B_oをordinary、B_fをfull、B_pを同じfinite targetへ接続したpartialの係数質量とすると、

\[
B_o=1+x^2+O(x^4),\qquad
B_f=1+(1-\chi)x^2+O(x^4),\qquad
B_p=1+(1-\chi)x^2+O(x^4).
\]

ordinaryはb_0+b_2+O(x⁴)、return-awareはrootの[1+(1−χ)x²/2]+長さ2の[(1−χ)x²/2]+O(x⁴)から得る。P5、P3+tailはそれぞれの定義域で同じ二次主項を持つ。異なるTaylor次数のtargetを同じものと扱った比較ではない。[G6-R] [G7-S]

r回の事象選択が独立で、最後に一回のHadamard測定Y=±1を行うとき、weight積の二乗の期待値が積になる。r回の量子測定結果Y_iを独立に測って掛けるという別protocolではない。

c=rx²を固定した極限では、理想二次momentの対数は

\[
\log\nu_o^{(r)}=2c+O(cx^2),
\]
\[
\log\nu_{f,\mathrm{can}}^{(r)}=2(1-\chi)c+O(cx^2),
\]
\[
\log\nu_{f,\mathrm{zf}}^{(r)}=(2-\chi)c+O(cx^2).
\]

**この表自体は整合する。** ただし全試行一件当たりの二次momentの表であって、量子gate総費用の表ではない。

### 3.2 m=3の四次係数

μ_k=Σ_i p_i^kとおく。既存SOURCEと閉形式を紙上展開すると、

\[
B_o=1+x^2-\frac7{72}x^4+O(x^6),
\]
\[
B_p=1+(1-\chi)x^2+\left(-\frac7{72}+\frac\chi{18}\right)x^4+O(x^6),
\]
\[
B_f=1+(1-\chi)x^2+
\left(-\frac7{72}-\frac\chi6+\frac{\mu_3}4-\frac{\mu_4}{36}\right)x^4+O(x^6).
\]

従って

\[
B_p-B_f=\frac{8\chi-9\mu_3+\mu_4}{36}x^4+O(x^6)
=\frac{\sum_i p_i^2(1-p_i)(8-p_i)}{36}x^4+O(x^6).
\]

L≥2、p_i>0で係数は正、上界は2χ/9。この係数についてClaude v2の追記は既存G6レビューと整合する。**数値SOURCE・symbolic algebraプログラムは実行していない。**[G6-R] [G7-S]

一般固定mでも主項一致からO(x⁴)は得られるが、m・L・分布を同時に動かす一様なO(χx⁴)定理へそのまま拡大しない。特にm依存定数、χ→1の退化、p_minとbit費用は別問題である。

### 3.3 極限の意味

c=rx²>0固定なら、正規化した総時間T=rx=c/xは無限大へ向かう。固定総時間Tでr→∞とするとc=T²/r→0であり、同じ極限ではない。

固定mで理想finite Taylorを使う場合、\|R\|≤1より一stepのremainderはO(x^{m+1})で、積のtelescopingからr-step誤差はr O(x^{m+1}) exp(r O(x^{m+1}))程度に抑えられる。c固定ならこれはO(c x^{m-1})へ消える。これは理想Taylor誤差の話で、native誤差の保証ではない。

またfinite-bitの相対誤差η,ρを固定したままr→∞とはできない。例えば既存moment係数κ=(1+ρ)²/(1−η)^{m+2}をr回掛けるとκ^rが入る。主項比較へ移すにはrρ、r(m+2)ηなどの制御とnative biasの全体配分が必要である。**理想式の漸近論を、固定96/160/256bit等の実装性能定理と同一視しない。**

## 4. 最重要訂正：zero-fillの二次momentの代価と期待量子費用を分ける

### 4.1 量子回路を実行する前に棄却できる

G6-M §8は、raw語が非reduced、またはacceptance coinで拒否となった場合、量子測定をせずにX=0を記録する。r-stepの乱択経路も、状態に依存しない古典選択を先に行い、どこか一stepでzeroなら量子回路全体を実行しない会計が可能である。これは今回新しく提案した実装ではなく、COST §3.3が既に採用する `Z^r` 付きの会計である。[G6-M] [COST]

理想一stepの受理事象eの非正規化係数をc_e>0、B=Σ_e c_e、B_oをordinary envelopeとする。G6で当該事象が発生する全試行確率はc_e/B_oであり、

\[
Z=B/B_o,
\qquad
\Pr(e\mid\mathrm{accept})=c_e/B.
\]

したがって、**受理条件付きの事象分布は、同じfull ensembleを正準抽出する分布と同じ**である。r独立stepでも、全受理条件付きの経路分布は正準fullの積分布と一致する。

全受理経路で一回だけ実行する量子回路の条件付き平均費用をD_rとする。同じ理想事象、同じcontrolled実装と費用規則を比較する。準備・読出しも全受理後だけに実行するならD_rへ含める。一方、古典proposal、局所query、棄却、cold synthesisの費用は別に記録する。

\[
\nu_{\rm zf}=(B_oB)^r,
\quad A_r=Z^r,
\quad \overline C_{\rm zf}=Z^rD_r;
\qquad
\nu_{\rm can}=B^{2r},\quad \overline C_{\rm can}=D_r.
\]

このとき二次momentを用いるleading費用proxyでは

\[
\boxed{
\nu_{\rm zf}\,\overline C_{\rm zf}
=(B_oB)^r(B/B_o)^rD_r
=B^{2r}D_r
=\nu_{\rm can}\,\overline C_{\rm can}.
}
\]

**Claudeの `exp(χc)` は全古典試行一件当たりの二次momentの比である。量子回路を実行しない試行まで同じ量子費用を課せば、その比を費用の不利と誤認する。**

これは理想の同ensemble比較である。真のZが未知なのにそれを使ってNを選ぶという科学実行を承認・実施したわけではない。未知Zと正当な予算選択の問題は、次節で既存Uを用いて分離する。

### 4.2 全normalizerを求めない上界Uが、既に存在する

G6-R §7とG7-Sの `upper_normalizer` は、

\[
U=d_\emptyset+\sum_{\substack{2\le l\le m-1\\l\text{ even}}}b_l q_l,
\quad b_l=\sqrt{t_l^2+t_{l+1}^2},
\quad q_l=\Pr(\text{長さ}l\text{のIID語が既にreduced})
\]

を使用する。全reduced parentのd_uを足すのではない。rootの局所係数と、raw-reduced質量の再帰を使う。これによりB≤U≤B_oが得られる。[G6-R] [G7-S]

固定p,mの小xでは、q_2=1−χ、b_2=x²/2+O(x⁴)、d_empty=1+(1−χ)x²/2+O(x⁴)なので、

\[
\boxed{U=1+(1-\chi)x^2+O(x^4),\quad U/B=1+O(x^4).}
\]

Uは量子回路を走らせずに計算可能な既存の認証用上界である。本レビューではその式とSOURCEを読んだだけで、新しい入力でUを生成していない。

### 4.3 Bernstein予算へ戻した比較

一axisについて、残余統計誤差s>0、\(\ell=\log(2/\alpha)\) を共通とし、理想回路・理想係数を比較する。G6の既存Bernstein規則をr独立stepへ接続した連続値の十分予算は、

\[
\overline N_{\rm zf}(U)=\ell\left[
\frac{2(B_oU)^r}{s^2}+\frac{4B_o^r}{3s}\right].
\]

実際の整数予算はこのceilであり、期待量子費用はN_zf Z^r D_rである。ceilを外した表式は

\[
\boxed{
\overline G_{\rm zf}^{(Q)}(U)
=\ell D_r\left[
\frac{2(BU)^r}{s^2}+\frac{4B^r}{3s}\right].
}
\]

同じfull ensembleの正準抽出については

\[
\boxed{
\overline G_{\rm can}^{(Q)}
=\ell D_r\left[
\frac{2B^{2r}}{s^2}+\frac{4B^r}{3s}\right].
}
\]

したがって、moment項の余分な因子は **(U/B)^r** であって、常に(B_o/B)^rではない。既存Uならc=rx²固定で

\[
(U/B)^r=\exp[O(rx^4)]=\exp[O(cx^2)]\longrightarrow1.
\]

未知Zを無料で取得しなくても、既存Uの**二次の主項の鋭さ**で、この理想的な期待量子費用の主要項を正準fullと揃えられる。range項も上記のとおり同じである。

ここでのUは式で定義した理想上界である。実際の外向き数値上界U_+を使う場合は、r(U_+−U)/Bの寄与も制御する必要がある。有限bitの相対誤差と同様、固定bit数のままでこの極限を実装したと主張しない。必要な精度取得の古典費用は別会計である。

一方、粗いU=B_o、すなわちZ_+=1を使うと、moment費用の主項は(B_oB)^rになる。この特定の予算policyと同じ受理回路単価モデルなら、Claudeのχ/2型の削減係数に対応する。**それを全G6/G7、あるいはnormalizer不要生成一般の上限として扱ってはいけない。**

ceilによる差はzero-fill側で最大Z^rD_r未満、正準側でD_r未満の加算となる。固定誤差でNが有限のままの極限に、整数予算比まで厳密に1と断定しない。ここで一致を述べたのは連続Bernstein予算の主要項であり、ceil・硬い実行上限の同値ではない。

### 4.4 この訂正から言えること・言えないこと

| 支持される訂正 | 依然として支持されない主張 |
|---|---|
| moment比をそのまま期待量子費用比にするのは誤り | 一般fullがpartial/CTSに勝つ |
| 既存Uを使うと、理想的な期待量子費用でχ/2という普遍的上限は導けない | 現存digital SOURCEがr→∞で同じ比を達成した |
| quantum-callと古典全試行数は分離が必要 | 古典query、有限bit、角度取得の費用が無料 |
| normalizer取得の不足と、ensembleの本質的な価値を区別すべき | G9/G10の固定native負結果を撤回する |
| 既存G6-M/G6-Rの注意書きが今回の論理に直接効く | この訂正そのものが新しい論文貢献である |

全受理経路の古典生成までN_zf回必要なら、古典費用は同じにならない。誤差を伴うdigital係数・proposalでは、実際のacceptanceと条件付きlawを使い、係数・native biasを戻す。実回路で棄却が事前に決まらない場合には、棄却までの量子費用も加算する。期待費用をそのまま高確率hard capにしてもいけない。

## 5. Cauchy–Schwarz下界は正しいが、最適総費用を固定しない

固定したunitary辞書、非負係数c_e、full-support proposal q_eについて、

\[
\nu(q)=\sum_e\frac{c_e^2}{q_e}\ge\left(\sum_ec_e\right)^2
\]

は正しい。Hadamard outcomeの二乗が1なので、これはweight二次momentをそのまま表す。ただし異なる提案分布では回路費用も変わる。

正の単価C_eを用いた既知のcost-aware ISの式は

\[
\boxed{
\left(\sum_e\frac{c_e^2}{q_e}\right)
\left(\sum_eq_eC_e\right)
\ge\left(\sum_ec_e\sqrt{C_e}\right)^2,
\qquad q_e\propto c_e/\sqrt{C_e}.
}
\]

これは本レビューで新たに提案した最適化原理ではなく、一次文献の固定protocolに関するCauchy–Schwarzの結果である。[IS]

従って、fullとpartialの最小二次momentが同じ主項を持つことは、**同じ費用単価規約の下でのcanonical比較には有力な制約**だが、価格・proposal・実装・前処理を含む最適費用の完全一致を意味しない。C=0の事象や有限confidenceのrange項、ceil、biasを含む場合には、さらに別の注意が必要である。

本レビューはIS論文のerror-detection後のZeroFillを、G6の古典的な事前棄却と同じ実装だとは扱わない。共通するのは、実際の試行費用とweightを併記する会計原則である。G6で量子費用を省ける根拠はG6-M/COST自身にある。

**保持する限定結論**：固定m,p、理想canonical law、同じleadingの実行単価という範囲では、fullがpartialより低いnormalizationを持つ追加利益は高次であり、固定c極限の主要項には現れない。**採用しない拡張**：従って任意のsampler・native実装・有限step領域・情報取得条件でもfullの独立価値はない、という結論。

## 6. Pauli-aware比較：係数上界は有用だが、総費用NO-GOではない

### 6.1 同じPauli frameと係数集約を明示する

単一の共通Pauli frameで、重複・符号違いの同一Pauliを予めまとめ、scalarを分離した非identity signed Pauli Q_iを考える。この追加条件の下で、

\[
R^2=\chi I+\sum_{i<j}p_ip_j\{Q_i,Q_j\}.
\]

反交換対の寄与は0。可換対は符号付きPauli積になる。ここで、

\[
\gamma_2:=\left\|\operatorname{PauliCoeff}(R^2-\chi I)\right\|_1,
\quad
w_{\rm comm}:=\sum_{i\ne j,[Q_i,Q_j]=0}p_ip_j
\]

と置くと、正確な関係は

\[
\boxed{\gamma_2\le w_{\rm comm}\le1-\chi.}
\]

異なる可換pairが同じPauliを逆符号で生成し得るため、集約後のL1がw_commに**常に等しい**というClaudeの表現は修正が必要である。これは比較の方向を逆転させる訂正ではなく、Pauli収集による相殺をさらに認める訂正である。

rootのidentity/imaginary部分のpairingと、収集した非identity実部を同じfinite polynomialへ正しく接続するなら、その係数質量の二次主項は

\[
\frac{1-\chi+\gamma_2}{2}
\le\frac{1-\chi+w_{\rm comm}}2
\le1-\chi
\]

となる。反交換対に正の重みがある場合には、形式free-wordだけを使う場合より鋭くなる余地がある。二次以外のfinite target残差、phase、符号、sampling法を捨てた比較にしない。[ZY] [SCU]

### 6.2 係数質量と資源会計

この式は**理想係数質量のx²係数**である。Pauli対の判定・収集・係数取得には費用があり、直接pair比較ならO(L²)個の対とPauli label処理が必要である。実際のメモリ、有限bit、controlled実装、集約で変わる角度・単価まで無料ではない。

従って、Pauli-awareを強い関連対照に含める理由はあるが、同target・同情報・取得費・native費用を閉じずに「全資源で最強の対照」と断定しない。係数比較が良くても高価な生成器で負け得るという点は、RA側にも対照側にも同じように適用する。

### 6.3 「PRでは非Pauliは対象外」という除外理由は使わない

Claudeはv2 Appendix Eの構成を根拠に乱択部をPauliと扱う。今回その指定v2本文の完全な独立再取得には成功していない。一方、**画像で確認したPR v1 p.13には、乱択公式に必要なのはP_i²=Iであり、各項を異なる軌道基底で表した \(\widetilde P_i=(U^{(i)})^\dagger P_iU^{(i)}\) を利用できる旨が明記されている**。DF表現も例示されている。[PR-v1]

したがって、特定のベンチマークや実装が共通Pauli展開を使うことと、PRの理論がその表現だけに限られることを混同しない。異なるframeのQ_i同士へ、単一frameのPauli反交換テストをそのまま適用することもできない。

これはTrack Bの非Pauli分子実装が完成したという主張ではない。**非Pauliの具体的な回路・古典取得費と利点は依然として未確立**であり、一般性だけで新規性や実用性を肯定しない。

## 7. χ、L、「挟み撃ち」、E0の位置付け

### 7.1 χ≥1/Lは小ささを証明しない

Cauchy–Schwarzのχ≥1/Lは下界である。Lが大きいからχが小さいとは結論できない。紙上の分布例

\[
p_1=\frac12,\qquad p_i=\frac1{2(L-1)}\ (i\ge2)
\]

では

\[
\chi=\frac14+\frac1{4(L-1)}\longrightarrow\frac14.
\]

これは論理的な反例で、分子入力生成・科学数値走査ではない。実PRの分割がこうした分布を排除するかは、別途実際の分割契約から確認する必要がある。

逆にχ≤p_maxという上界は使えるが、乱択側の正規化後p_maxが十分小さいことは今回取得されていない。「大きい項を決定論側へ移す傾向」は特定の分割policyの話であり、全許可入力に対する定理ではない。

また、reduced語数はLだけでなくmにも依存する。固定L≥3でもmを増やせば語数は指数的に増えるため、局所queryの意味が大Lの場合にしか生じないという論理も成り立たない。ただし、mを増やすことが実際に得かは別である。

### 7.2 χはHamiltonian固有の物理不変量ではない

χは指定された係数表・分解・正規化の指標である。同じQの係数を複数labelに分割する表現変更を許せばHを変えずχを変えられる。これは同じ固定入力に対する操作ではなく、**χに基づく一般主張には入力のcanonical化・重複処理・分割policyが必要**だという注意である。

χ_Rを取得すること自体は、採択された乱択分割を記述する補助診断として意味がある。しかし以下は導けない。

- χ_RだけからRA独立構成の新規性を判定すること。
- χ_R/2をG6/G7の普遍的な期待量子資源削減上限とすること。
- χ_R×「乱択部の費用割合」を、準備・決定論部分・confidenceへの波及を含む総費用改善上限とすること。
- 固定syntheticの合成価格差から、普遍的な0.05の採否閾値を設定すること。

特にshot数が変わると準備・読出し・決定論部分も繰り返す回数が変わり得るため、固定した費用割合を後から掛けるだけのAmdahl型計算を無条件に使わない。合成列の離散的価格差は、そのまま統計的な5%ノイズ水準を意味しない。

**E0は現在の結論に必須ではない。** 今回は計算しない。将来実施する場合も、一般NO-GOの判定器ではなく、表現・分割・費用契約付きの診断とする。

## 8. Parent allocation：懸念は保持し、総費用推論の条件を補う

### 8.1 保存されたpositiveを消さない

S4は固定3-system-qubit development synthetic、r=1、h_T=0、同じfinite P5 taskでのbest-to-best比較である。ordinary約14%、P5約19%のT低下、N/CX増加、proposalの違いを保持する。これを多step全PRの利益へ拡張せず、逆に今回の漸近推論を理由にr=1の保存結果を無効とも扱わない。[S4-C]

### 8.2 係数質量の下界は保持できる

同じchild jにrootのparent質量を与える固定頂点で、s=1−p_j>0とすると、

\[
B_V(x)=1+sx+O(x^2),\qquad B_C(x)=1+O(x^2).
\]

独立stepの固定辞書でweightを掛けるなら、

\[
\nu_V^{(r)}\ge B_V(x)^{2r}
=\exp[2sT+O(rx^2)],\quad T=rx.
\]

これは多stepへ移す際の強い懸念である。固定Tでr→∞ならexp(2sT)、長時間T→∞・rx²固定ならexp(2sT+O(1))という違いも明記する。[S4-R]

### 8.3 pure係数O(x)から、任意proposalのpure確率O(x)は従わない

canonicalなら確率は係数に比例する。しかしISではqを変えるため、pure係数がO(x)という事実だけからそのsampling確率をO(x)とはできない。

既存S4レビュー自身が、安価pureへsampling確率を大きく移した場合の役割を明記している。これを採用した比較について、突然canonical確率の直観で総費用を判定してはいけない。[S4-R]

また、「回転を含む事象のT単価に正の下界がある」だけでは、commonと頂点での単価の相対変化や、全pathのコンパイル相殺を制御できない。回路を数えるモデル、価格上界、同じ誤差配分、proposalの許可範囲まで必要である。

### 8.4 条件を明示すれば、長時間の不利はcost proxyでも論じられる

以下は既知Cauchy–Schwarzを用いた**本レビューの条件付き紙上比較**である。新しいアルゴリズムの提案や科学計算ではない。

仮定：固定したr-stepのlabelled product辞書を用い、異なるpath間の係数再集約は行わない。path係数はc_path=Π c_e。費用は各回転の加法的下界を持ち、回転を含むstep数Kに対してC_path≥g_min(x)K、g_min>0とする。回路相殺やbatch合成でこの下界が壊れる場合には適用しない。

同じ頂点辞書の係数正準分布を補助的に用いると、一stepの非pure確率P_rot=1−O(x)。これは**実際に選ぶ任意proposal qの確率ではない**。

任意のfull-path proposalに対して、known cost×second-momentの下界から

\[
NC_V(q)\ge\left(\sum_{\rm path}c_{\rm path}\sqrt{C_{\rm path}}\right)^2
\ge g_{\min}B_V^{2r}\left(E_{\rm can}\sqrt K\right)^2.
\]

0≤K≤rなので√K≥K/√r、従って

\[
\boxed{NC_V(q)\ge g_{\min}\,r\,P_{\rm rot}^2B_V^{2r}.}
\]

commonの合法なcanonical実装にNC_C≤g_max(x) r B_C^{2r}という上界があり、g_max/g_minがTに対してsubexponentialなら、長時間・rx²固定の下で頂点のこのcost proxyはcommonより不利になる。

これはClaudeの結論に近い**条件付きの懸念**を支える。しかし必要なのは、labelled辞書、コストの加法的下界、commonの上界、整合した価格・精度の条件である。上式は二次momentを使う費用proxyの比較で、必要最小shotsの下界でも、あらゆる誤差・状態・compilerを含む実PRの総T費用定理でもない。

したがって、現行固定頂点をそのまま長時間へ移す提案は慎重に扱うべきだが、「すべての配分・重要度sampling・多step回路最適化に対する一般NO-GO」とはしない。現在のD＋E方針と整合する限定である。

## 9. 新規性と論文化可能性の独立判断

### 9.1 科学的正しさ、新規性、重要性、実用性は別の軸

G6の基礎構成が正しいこと、Green/Euler/ISの近接原理が既知であること、固定native実験で一般fullがclosed対照に勝たなかったことは、それぞれ支持される。一方、これらの組合せから自動的に全手順の完全同値、独立新規性の不存在、単独論文化不能までは導けない。[G6-M] [CLAIM]

特に次を新しい必須条件にしない。

- 新アルゴリズム論文なら必ず既知法に対する漸近次数改善が必要であること。
- 既知の数学部品を一つも使ってはいけないこと。
- finite-bit論文にはbit上界と下界の完全一致が必須であること。
- 限定的な構成・負結果・資源評価は原著論文になり得ないこと。

これらは本件で自動採用すべき科学基準ではない。有限領域の非自明な改善、一般的な構成保証、情報アクセスや生成費用の差、明確な限界の定理も貢献候補となり得る。ただし、**その候補が本件で既に十分に立証されたとは認定しない**。

### 9.2 今回の訂正は、Aを肯定する証拠ではない

zero-fillの費用訂正とUは既存G6資料に既にある。従って、本レビューがその見落としを修正したことは、Track Bに新たな論文の中心結果が生まれたことを意味しない。

今回保持する構成候補は、固定finite Taylor targetでの正値性、局所係数query、未知global normalizerを必要としない生成、有限bitの明示会計の結合である。この全結合を、同じ情報と最適化を許した強い既知法と同費用で対応させる証拠は、なお不十分である。non-enumerativeという語だけではordinary/CTSとの差にならず、piecewiseな既知性だけでも全構成の独立差がないとは証明されない。[CLAIM] [WBC] [SCU] [SPS] [IS]

方向BのD3/S4は、限られた条件の診断と事例として価値を保つ。しかし本件で独立した一般selectorを得た証拠はない。診断研究としての論文化可能性を論理的に排除しない一方、現在の書類をまとめれば査読原著として十分という保証もしない。[CLAIM] [S4-C]

### 9.3 結論の強さ

| 判断 | 本レビューの立場 | 確かさ |
|---|---|---|
| 現在の資料だけで新アルゴリズム原著の投稿GO | 認定しない | 高：決定的な方法差/価値の支持が不足 |
| 全構成が既知法と完全同費用で同値 | 認定しない | 未確定 |
| 一般full固有のleading normalization改善が固定短step極限で消える | 限定して保持 | 高：固定p,m、理想canonical量 |
| zero-fillに必ずχ/2の総量子費用上限がある | 採用しない | 高：既存Uと事前棄却会計で反論できる |
| 現在の構成にあらゆる資源価値がない | 認定しない | 未確定。証明範囲を越える |
| 当面、単独投稿と追加大規模実験を見送る | 推奨 | 現存証拠に対する研究上の判断 |
| 限定構成ノートの可能性 | 残すが、成立を認定しない | 方法差・学術的重要性は未確定 |

実務上はClaudeのCに近い。しかし科学的分類としては、**「Aの資源価値を否定済み」ではなく、既存のA_INSUFFICIENT_EVIDENCEにとどめる**のが適切である。Bも既存のB_DESCRIPTIVE_RESULTS_ONLYを、本レビューだけで新規方法論へ格上げしない。

## 10. Claude v2へ必要な修正

本表は修正を指示する技術的・科学的根拠であり、Claudeの原本を既に変更したという意味ではない。

| ID | 重要度 | 該当節 | 必要な修正 |
|---|---|---|---|
| F1 | 結論を左右する | §7.1(i)、§0.4、E0 | 全試行Nと量子実行NZ^rを分け、既存Uを戻す。χ/2を普遍的期待量子費用上限にしない |
| F2 | 結論を左右する | §7.1(iii)、§8.1–8.3 | 二次moment下界から最適総費用・独立新規性・論文化不能へ飛躍しない |
| F3 | 重要 | §7.1全体 | 固定m,pの理想極限、Tの増大、有限bit/native precisionとのjoint limitを区別 |
| F4 | 重要 | §7.4 | Pauli集約L1はγ₂≤w_comm。理想二次係数と総費用を区別し、無条件に最強対照としない |
| F5 | 重要 | §6.1、§7.4 | PRベンチマークの表現と理論の許容表現を分ける。原論文v1の異なるframeのinvolution一般化を無視しない。v2照合範囲を正直に限定 |
| F6 | 重要 | §7.6 | pure係数O(x)と任意ISのpure確率を区別。下界の総費用への含意には価格・labelled辞書・相殺なし等の条件を加える |
| F7 | 重要 | §7.3 | χ≥1/Lから低χを導かない。大mでも局所生成が意味を持つこと、分割policyの仮説性を明示 |
| F8 | 重要 | §11 E0/E1 | 5%の未採択閾値を科学的停止基準としない。未修正の§7を確定NO-GOとして技術ノートへ追記しない |
| F9 | 中 | §8.2 | 有効parent条件付きA_uの下界と、全試行Zの下界を区別 |
| F10 | 解釈 | §5.2、§8、結論 | 初等性や漸近改善の欠如だけで論文化不能とはしない。現在の証拠不足として研究投資判断を記録 |

F1/F2が最重要であり、全てを細かな実装修正タスクへ分割する必要はない。今回発見したのは主として**レビューの会計・推論の問題**であり、現行SOURCEを変更すべき不具合を認定したわけではない。

## 11. 推奨する次の作業と承認境界

### 11.1 現時点で実行しないこと

追加分子、χ_Rの計算、provider/seed/precision/r走査、Parent allocation再最適化、native compile、全70 lawの新包絡線、科学SOURCE再実行を行わない。今回の訂正だけで旧実験の実行済みclaimを再利用しない。

ClaudeのE1（限界節追記）を、v2の§7をそのまま「確定した構造的NO-GO」として貼り付ける作業にはしない。E0を使うにしても一般採否閾値ではない。E2等の新しい理論整備も、今回自動的に開始するテーマではない。

### 11.2 最小限の実務

ユーザーが本レビューを受けた文書整備を指示した場合は、Codexが一束で次を行えばよい。

1. Claude v1/v2と本GPTレビューを原本のまま保全し、互いの参照と変更点を固定する。
2. G6-M/G6-R/G7-S/COSTに存在するzero-fill会計・Uと、Claude v2の推論との対応を、保存資料への固定リンクで記録する。
3. 研究上の状態を「現時点の単独論文化は保留。一般的資源NO-GOや全方法既知を認定したものではない」と正確に区別する。正式な方針採択はユーザー判断とする。
4. 現存の技術ノートへ反映する場合は、固定p,m・理想量・期待量子費用の範囲を明記し、SOURCE/実験結果を変更しない。

これは新しい方法差調査や科学再解析を反復する提案ではなく、**今回確認した限定と訂正を失わないための記録**である。同じ論点の全面レビューを無目的に繰り返す必要はない。

論文化へさらに投資する方針をユーザーが別途選ぶ場合にのみ、残る構成主張のどこを追加で立証するか決める。新しい構成を自動探索して救済することや、「不足があるから全方向を再調査する」ことは推奨しない。

### 11.3 保持する研究状態

- Track B：2026-10-15 D＋Eを保持。限定成果を保存、現行数値探索は一区切り、mandatory STOP。
- 研究④の新テーマ選定：当面保留を保持。④全体CLOSEDとはしない。
- A2 NOT_ADOPTED、現実用化経路への追加投資見送り、A1 CLOSED、K1-F未承認・未実行。
- 他研究①②③、既存SOURCE/raw/certificate/契約/レビュー/authorization/grant/ledger/marker/消費済みclaimは変更しない。

本レビューは科学的判断の記録である。正式採択、GitHub更新、新科学計算には、それぞれ必要な指示・承認を別途受ける。

## 12. 固定参照一覧

### GitHub資料

- [G6-M] G6独立数学監査。carrier `bc649c592c8c2b625b49094b85c27e380faa3cf4`。特に§7–8。
- [G6-F] G6 finite-bit/access監査。同carrier。
- [G6-S] `return_aggregation.py`。`28cfabb1d47e0e1824bce1e154a9c289738fa2b9`。
- [G7-S] `g7_generator.py`。`ab2549f41b3546fb3940342a2162dd9ee93699c4`。本レビューは1–150行。
- [G9-S] `g9_p5.py`。`0ef2b92738750a3c0d187743dd4b7c9b927db802`。
- [G6-R] 既存GPT G6科学レビュー。`101bf3426fcd04cc0c01c0d202d0916268765d97`。§5–7。
- [COST] Track B資源構造とボトルネック。`101bf3426fcd04cc0c01c0d202d0916268765d97`。§3–4。
- [S4-R] Parent allocation S4科学レビュー完成版。`e3086f48c5598c8bb31b0587e807ca6c1b3d2754`。§4の関連部分。
- [S4-C] Parent allocation現行探索の正式closure。`101bf3426fcd04cc0c01c0d202d0916268765d97`。
- [CLAIM] Paper Claim Sheet。`aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14`。

### 外部一次文献

- [WBC] K. Wan, M. Berta, E. T. Campbell, *A randomized quantum algorithm for statistical phase estimation*, arXiv:2110.12071v2。Appendix Cを本文・画像で確認。
- [IS] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。§II、§IV。
- [SCU] J. Peetz, S. E. Smart, P. Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, arXiv:2407.21095v2。取得したHTMLの対象箇所に限定。
- [SPS] B. Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, arXiv:2402.15550v2。取得したHTMLの対象箇所に限定。
- [ZY] Q. Zhao, X. Yuan, *Exploiting anticommutation in Hamiltonian simulation*, arXiv:2103.07988の取得PDF・§4.2周辺。固定v2 bytesと独立一致したとはしない。
- [AK] K. Aomoto, Y. Kato, *Green functions and spectra on free products of cyclic groups*, Ann. Inst. Fourier 38(1), 59–85 (1988)。Numdam本文、画像照合未完の限定を保持。
- [PR-v1] J. Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647。**今回画像で確認した版はv1**。p.13に異なる軌道frameのinvolution/DF乱択一般化。指定v2の独立本文照合とは区別する。

[G6-M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[G6-F]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G6-S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[G7-S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G9-S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[G6-R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
[COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md
[S4-R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[S4-C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[CLAIM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/paper_claim_sheet.md
[WBC]: https://arxiv.org/abs/2110.12071v2
[IS]: https://arxiv.org/html/2603.13495v1
[SCU]: https://arxiv.org/html/2407.21095v2
[SPS]: https://arxiv.org/html/2402.15550v2
[ZY]: https://arxiv.org/abs/2103.07988
[AK]: https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf
[PR-v1]: https://arxiv.org/abs/2503.05647v1

## 13. 完了記録

重要科学レビューを完了した。Claude v2の結論を全面追認せず、zero-fill・U・費用proxy・Pauli・多step・χの各論点を独立に検討した。**単独論文化の当面保留は推奨するが、資源価値の一般NO-GOとしての終了理由は採択しない。**

本Markdown以外に研究SOURCEや結果を変更していない。原本のhash検査と本書の形式検査は管理処理であり、科学SOURCEの実行・科学数値再解析ではない。新規科学計算、実験、新しい研究方向・契約の採択、GitHub書込みはいずれも行っていない。
