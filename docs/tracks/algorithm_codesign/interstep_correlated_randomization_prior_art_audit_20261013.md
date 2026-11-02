# Track B：時間方向相関の一次資料・式・資源会計の限定照合

作業日：2026-10-13 JST。区分：文献取得、静的source照合、手計算による解析。

**取得不足だったPRHS本文・同梱補足を取得した。二ステップの高次化例は既知係数へ対応する。現PRを改善する追加の具体的生成手順は、この照合では確立していない。** qSHIFT v1のfactor 2表示差はv3の対応例で修正されているが、v3の履歴係数表示には別の不整合が残る。実測shot・controlled first moment・有限bit・生成費用を含む移送は未認証。性能negativeとも新手法GOともしない。

入力は[GPT方向レビュー原本](../../research/track_b_interstep_correlated_randomization_direction_review_20261013.md)の§11。原本は37,358 bytes、SHA256 `1eaa4aa0a0a8841552f0b86263571be4800e059f245f3e8fefe80d39e28274cd`。原本の「PRHS本文未取得」等を改稿せず、本報告を取得後の追記とする。

独立branch `track-b-interstep-correlation-prior-art-audit-20261013`、base `7c3d59139febfc94749b7c2ffc74b4f0a882f539`。既存の休止記録を基点とし、Track A・N3・直前の基底変換費用診断から別の科学値を採用していない。

## 1. 取得・版・確認位置

| 一次資料 | 取得した版と確認位置 | 今回の扱い |
|---|---|---|
| Davide Cugini, *Pathwise Random Hamiltonian Simulation* | [2608.29756v1 PDF](https://arxiv.org/pdf/2608.29756v1)、[HTML](https://arxiv.org/html/2608.29756v1)。2026-08-30、14 PDF pages。main Eqs.(2)–(8)、Supplement A/S1–S28、B/S29–S34、C/S35–S36 | 本文・同梱補足取得済み。formal係数、norm、costの関連箇所を確認。全定理の独立再証明や分子結果の再現はしていない |
| Sangjin Lee, Sangkook Choi, qSHIFT | [2604.26263v1 PDF](https://arxiv.org/pdf/2604.26263v1)、[HTML](https://arxiv.org/html/2604.26263v1)。arXiv version date 2026-04-29、21 PDF pages。§III、main Eq.(12)/PDF p.4、B、C.2/PDF p.16 | 指定された旧表示を確認。論文のtitle-page日付とversion日付を同一視しない |
| 同著者、改訂版 *Adaptive sampling based quantum simulation* | [2604.26263v3 PDF](https://arxiv.org/pdf/2604.26263v3)、[HTML](https://arxiv.org/html/2604.26263v3)。version date 2026-09-02、title-page 2026-09-03、27 PDF pages。III.1/Algorithm 1、Eq.(17)/p.6、B.2 Eqs.(B10)–(B17)/p.18、B.3、C.2/p.22 | v1と区別して照合。arXiv abstract表示の旧題名も、取得PDFの題名へ無言で置換しない |
| Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms* | [2603.13495v1](https://arxiv.org/html/2603.13495v1)、§II/Theorem 1/Eqs.(9)–(13)、II.1、III/Theorem 2 | 全経路costを使うISと、bias不変の対照 |

取得物のbytes/SHA256・URL・ライセンス表示・確認sectionは[取得記録](../../../artifacts/track_b_interstep_correlation_prior_art_audit/2026-10-13/paper_retrieval_metadata_v1.json)。PRHS PDFへのweb取得の一回のcache missと、その後のHTTP取得成功を区別する。PDF/HTML本文、抽出text、紙面画像は一時領域でのみ扱い、repositoryへ転載していない。ほかのP1/P5/P6/P7の全文監査は今回反復していない。

## 2. 版による表示差を無言で修正しない

qSHIFT v1のEq.(12)はoff-diagonalを `h1*h2/lambda^2`、C.2はその2倍と表示する。v3 Eq.(17)は2倍となりC.2と一致する。PDF紙面とHTMLを照合した。[v1 PDF p.4/16](https://arxiv.org/pdf/2604.26263v1)、[v3 PDF p.6/22](https://arxiv.org/pdf/2604.26263v3)

ただしv3 PDF p.18のEq.(B16)には、直前のEq.(B15)にある末尾のhistory factorがない。以下の左辺は**紙面の字義的表示**、右辺は**直前の再帰からの独立展開**であり、訂正済み実装として採用した式ではない。`mu_i=h_i/lambda`、`c_ij,n_i`は同頁の履歴countとする。

\[
\begin{aligned}
\text{表示 B16: }a_{ij}^{(p)}&=2p^2\mu_i\mu_j-c_{ij}-(2p\mu_i-n_i)-\delta_{ij}p\mu_j,\\
\text{B10--B15から: }a_{ij}^{(p)}&=2p^2\mu_i\mu_j-c_{ij}-(2p\mu_i-n_i)n_j-\delta_{ij}p\mu_i.
\end{aligned}
\]

`p=1`、empty history、`i!=j`を代入するだけで、前者は `2*mu_i*mu_j-2*mu_i`、後者は `2*mu_i*mu_j`となる。後者がmain Eq.(17)と整合する。この限定した不整合はPDF画像でも確認した。論文全体の保証を否定する根拠にはせず、将来の実装には字義的B16と再帰のどちらを採るか明示した確認が必要とする。[v3 B.2](https://arxiv.org/html/2604.26263v3#A2.SS2)

PRHS p.2はsigned `q_M`を説明しながらsampling probabilityを `q_M/||q_M||_1` と表示する。負係数には正の確率でない。以下の会計では数学的に必要な `|q_M|/Gamma` とsign weightを分離する。これは原文の字義的表記を訂正版と偽らず、正のsamplingを定義するための整合条件として扱う。[PRHS PDF p.2](https://arxiv.org/pdf/2608.29756v1)

## 3. 同じ会計へ戻す対応表

記号：`omega`は順序と符号を含む全path、`Y`は一回の±1測定、`C(omega)`はそのpathの実装・準備・readout費用。原PRは `A_P=B*sum P(omega)V_omega` とする。経路依存の既存weightなら下表の `B*P/q` を `W0*P/q` に置き換える。

| 構成 | joint law・正の確率・weight | 全体mean／保証target | 実測M2・range／classical access | finite-RTEへの対応と残る条件 |
|---|---|---|---|---|
| 現paired finite RTE | orderを先に選び、componentをIID選択。event coefficient / `B_K` が正の確率。event phaseをunitaryに保持し、全occurrenceのnormalization積が`B` | normalization補正後はdegree `K+1`の有限Taylor積。coherent first momentであり、exact evolutionとの差は別bias | canonical `M2=B^2, range=B`。保存sourceのsamplerは古典event選択で、quantum shotではない | D挿入、application order、signed time、controlled相対位相を保持。ソース参照のみで新eventは生成しない |
| 旧meanを保つ全経路IS | `q(omega)>0` on support(P)、`W=B*P/q`。path同士のshotは独立 | `E_q[W V]=A_P`。同じpath実装なら実装biasも不変 | `M2=B^2 sum P^2/q`、`range=B max P/q`。全path likelihoodとcostの取得・proposal生成を会計 | 非加法な全path cost自体は新規性ではない。最適lawの認識と安価なsamplingの構成は別問題 |
| PRHS型のsigned correlated block | 固定block係数`a(s)`、`Gamma=sum|a|`、正の確率`|a|/Gamma`、weight `Gamma*sign(a)`。N block独立ならweight積 | 本文はLiouvillian/diamond-distance、補足S2は形式的wordのM次一致。signed平均は一般に一つのCPTP circuitではない | 単一±1測定なら `M2=Gamma^(2N), range=Gamma^N`。閉形式係数と効率的normalizer/sampler取得を同一視しない | 既存有限Taylor平均を自動保持しない。controlled first momentには別の移送証明・biasと実装会計が必要 |
| qSHIFT型history-adaptive block | historyごとに`a_p(s|h)`、`Z_p=sum|a_p|`、正の確率`|a_p|/Z_p`。joint lawは条件付き確率の積、`W=prod Z_p*sign(a_p)` | nested-commutatorによるobservable evolutionのorder matching。v3は三角再帰＋共通sparse map、history更新 | `M2=E_joint[prod Z_p^2]`、`range=max_path prod Z_p`。一定ZのN乗と勝手に置換しない | v3の表示問題、全path誤差、有限bit、phase-preserving controlと現在のfinite polynomialへの差を閉じる必要 |

PRHSのformal uniquenessはS2のword条件に対するものとして読む。物理operatorの特殊関係を使う全sampler、全representation、最小総資源についての一意性へ拡張しない。main Eq.(8)/Supplement S35のη cost modelを、実測native gate費用と有限confidenceの一般保証へ置き換えない。[PRHS A–C](https://arxiv.org/html/2608.29756v1)

qSHIFT v3 §III Eq.(7)はobservable operatorの条件付きmatching、Eq.(8)はobservable期待値差の絶対値に関するorder記述で、線形累積を仮定する。coherent unitaryのoperator-norm certificateではない。Eq.(16)/B.3の確率変数はpathごとの量子期待値。累積weightを確認できるが、その二乗を一回の±1 outcomeの二乗へ代入できない。[qSHIFT III/III.1/B.3](https://arxiv.org/html/2604.26263v3)

全経路ISの対照は、§IIの全circuit parameter vectorとTheorem 1。II.1のadditive/independent asymptoticだけを一般pathの限界へ広げない。Theorem 2は、proposalだけを変え同じpath実装を使う場合のbias不変を示す。[IS II–III](https://arxiv.org/html/2603.13495v1)

両signed方式の関連箇所は理想real係数の構成であり、今回PRで必要な丸め後確率・zero判定・mean補償・weight rangeの有限bit認証を取得したとはしない。`lambda_i`の符号を正のmassへ吸収する際も、generatorの符号を明示する必要がある。

## 4. 入力レビューの解析例を独立に確認

以下は入力レビュー§3–9の式を展開したもの。外部論文の全定理の再証明でも、新しいregistered numerical resultでもない。

### 4.1 marginal保存はmean保存ではない

`U_i=c I-i s Q_i`、`Q_i^2=I`、`R=sum p_i Q_i`、stationary persistence `K_eta(j|i)=(1-eta)p_j+eta delta_ij`なら

\[
\mathbb E_\eta[U_2U_1]=(1-\eta)(cI-isR)^2+\eta\sum_i p_iU_i^2,
\quad
\mathbb E_\eta[U_2U_1]-(cI-isR)^2=-\eta s^2(I-R^2).
\]

従ってlabel marginal一致だけで元targetを同じと呼ばない。これは二step・involution・D挿入なしの例であり、全相関法へのno-goではない。

全経路補正は `sum q*(B*P/q)*V=B*sum P*V`。uniform 2-state、長さm、`0<=eta<1`なら、各遷移の一致／不一致を数えると

\[
\sum_\omega P(\omega)^2/q_\eta(\omega)=(1-\eta^2)^{-(m-1)},\quad
\max P/q_\eta=(1-\eta)^{-(m-1)},\quad
\mathbb E_{q_\eta}\#\text{switch}=(m-1)(1-\eta)/2.
\]

switch減少だけでは総資源改善にならない。D境界や別basisではswitch数をactual gate costへ直結しない。

### 4.2 有界weightとcost最適lawの意味

`q=(1-lambda)P+lambda r`、`0<=lambda<1`なら、pointwise `P/q<=1/(1-lambda)`。従って `sum P^2/q<=1/(1-lambda)`。全相関に指数増大が必須という結論は出ない。

正のfinite cost `C`ではCauchy–Schwarzにより

\[
(\mathbb E_q C)\sum P^2/q\ \ge\ (\mathbb E_P\sqrt C)^2,
\qquad q^*\propto P/\sqrt C.
\]

これは入力レビューの既知IS対応の確認。zero-cost supportがあればfloorまたは境界処理が必要であり、表示lawをそのまま採らない。range・bias・古典生成・integer shotを含む最適性は、このcost×M2等式だけでは保証しない。

### 4.3 二ステップ高次化は既知係数へ還元する

`F(h)=sum p_i exp(-i h H_i)`、`Hbar=sum p_i H_i`なら

\[
2F(h)^2-F(2h)=I-i2h\bar H-2h^2\bar H^2+O(h^3).
\]

長さ2のsigned lawは `a_ij=2p_i p_j-delta_ij p_i`、両marginalがp、全massが1。PRHSのM=2 run係数は、異なる2labelで `2p_i p_j`、同labelで `2p_i^2-p_i`へ減り、qSHIFTの初回二term例の修正表示とも一致する。この対応は追加のmethod deltaを示さない。チャネル側とunitary側へ同じformal式を代入できることと、既存finite-RTE／full PRの同じbias保証を得ることは区別する。

uniform L>=2なら `Gamma=sum|a|=3-4/L`。独立b blockの正のsamplingを実装したと仮定するとweight rangeは `Gamma^b`、±1測定のM2は `Gamma^(2b)`。さらに元RTE weightを独立に重ねる設計なら、そのweightも積に戻す。実装回路を構築した結果ではない。

### 4.4 実測shotとcoherent保証への必要条件

測定はaxisごとに `Z=W(omega)Y`、`Y^2=1`。path期待値を `mu(omega)` とすると

\[
\mathbb E Z^2=\mathbb E W^2,
\quad
\operatorname{Var}(Z)=\mathbb E[W^2(1-\mu^2)]+\operatorname{Var}(W\mu).
\]

exact path expectationのvarianceは右辺第二項だけであり、第一項を落とさない。古典historyを共有したfine/coarse測定にも同じ条件付き分散項が残る。

独立whole-path shots、`|Z|<=Rmax`、`E Z^2<=M2`、残る統計幅s>0に対する一つの保守的Bernstein sufficient shotsは

\[
n\ge\left\lceil\log(2/\alpha)\left(2M_2/s^2+4R_{\max}/(3s)\right)\right\rceil.
\]

ここでsはfinite-RTE/PF・変更mean・有限bit・synthesis/control biasを別枠で戻してから定義する。新accuracy/alpha/thresholdは固定していない。総費用は `n E C` と入力／sampler acquisition費用を合わせる。query数やbasis transition数のみで優位としない。

ordinary channel平均の一致は `sum a U` のcoherent first momentの一致を意味しない。`U`と`-U`は同じconjugation channelでも、制御後の相対位相は異なる。ancillaを含むliftを使う場合は、対応するgenerator、target norm、signed時間、D interleaving、全block remainderを再導出する必要がある。signed／finite polynomial平均には一般のCPTP contractionを無条件に使わない。これらを閉じずに外部論文の局所error式をPR全体のcertificateへ移していない。

## 5. 保存sourceとの照合と未取得部分

固定参照は `321cd8c3066194fe98741cbebbcdba1515d1deaa`。

- [rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py)：1594–1750のorder/component law、phase、application順、basis metadata、古典samplerの区別。1751–1799の位相付きmean、1854–1904のdegree K+1 polynomialとnormalizationを静的に確認。
- [hierarchical_cost_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/hierarchical_cost_validation.md)：1–62のshort列境界補正と、prepare/readout/shotを含まないcompiled-cost scopeを確認。このIID条件で較正された係数を、新proposalの期待費用へ無条件に流用しない。数値結果の再集計はしていない。

blob bytes/SHA256は[source identity](../../../artifacts/track_b_interstep_correlation_prior_art_audit/2026-10-13/source_identity_v1.json)。sourceをimportせず、event enumeration/sampling、行列作成、NPZアクセス、science testsを行っていない。

| 比較項目 | 確認できた対応 | まだ必要なもの |
|---|---|---|
| 旧finite meanを保持する相関 | 全経路ISでexactに保持できる | full-support proposalを安価に生成する具体手順、likelihood、range、同実装path cost |
| 高次signed block | 二step例が既知係数に対応 | 元finite Taylor対照と同accuracyのcoherent bias／normalization比較、履歴表示の解消 |
| 利用可能情報 | 既知lawはterm weightsとordered productsを使う。ISはcostにも依存 | DF関係、frame/commutator、compile oracle等の追加accessを使うなら取得・検証費用を明示 |
| compilerとのinteraction | ordered native costは一般に非加法 | correlated lawに対する費用証拠、D境界/controlを含む同実装条件 |
| 有限confidence | signed weightの実測M2/rangeを戻せる | rounded law、zero-support、制御位相、実装error、全pathの独立性の認証 |
| 追加の具体差 | 今回確認した単純式・再重み付けだけでは確立しない | 最も近い既知対照との差が一文で言える生成手順・保証・費用機構 |

## 6. 完了状態と戻す判断

照合結果は **`KNOWN_PROCEDURE_MAPPING`**。PRHS必須本文の取得不足は解消。v3表示不整合とPRへの保証移送は未解決だが、これは資料未pushや未取得とは異なる。**`ADDITIONAL_CONCRETE_METHOD_DELTA_NOT_ESTABLISHED`** とし、具体的candidateが揃っていないため最小pilotを立案・採択していない。

新実装0、scientific runner/tests0、行列・trajectory・sampling・compile・T synthesis・LP・DF・分子/NPZ・GPU処理0。既存result/source/contract/authorization/marker/STOP変更0。新authorization/marker0。文献取得・紙面render・stdlibのbytes/hash照合だけを実行した。実験的性能値は取得していない。

Track Bの自動的な実験立上げ休止、generic whole/Gaussian/G系列等の既存判断は維持する。[休止記録](track_b_research_pause_record_20261013.md)。時間相関一般の有用性／無用性、Track B全体の研究価値、新規性成立を判定したものではない。次に具体的な方法差・保証・同task費用を備えた案を採択するかはGPT側の研究判断。その場合も科学実行は別途の明示認可が必要であり、本報告は認可を与えない。

取得・式・会計の限定照合を一束で完了し、**mandatory STOP**。
