# 元Track A・仮説I：共通中心化と量子平方回避の技術検証

2026-11-10 JST。状態：**中心化の条件付きmass差・平方register省略を確認／人工38テストPASS／総費用は未評価**。

今回確認したA（数学的なmass差）とB（同じ反射のworkspace削減）は成立する。
C（同じenergy推定条件で強い方式よりshot加重総費用が小さい）は未確認。
仮説Iは比較primitiveとして保持し、研究主線への採択は保留する、という技術的提案を返す。
前回の「identity保持ではmass一致」はそのまま成立し、今回の中心化結果と矛盾しない。

## 1. 範囲・継承・SOURCE

- 基点：[66335f2812f5d8fb1a9b38171180951825fd8dfa][base]。
  同じ研究目的の連続作業として、cleanな既存branch
  `codex/track-a-threshold-reflection-tech-20261110` とその専用worktreeを継続した。
- [前回REPORT][previous]は変更しない。今回の追補とコード変更は後続commitに保存する。
  古い固定commitのSOURCE・入力・結果・grant・ledger・marker・STOPを改変しない。
- ユーザー提供の今回の指示、および仮説定義§4を照合した。仮説定義の現ローカル配置は
  `docs/research/planning/track_a/track_a_new_algorithm_hypotheses_20261110.md`。
  元rootからの移動はroot checkoutの配置表で確認した。worktreeへ無関係な整理変更は持ち込まない。
- `track_a_research_direction_gpt_review_20261110.md` と正式共通指示2026-11-09版の原文は、
  今回の既知repository／添付path検索では未取得。読了とは扱わず、今回の添付に明記された条件を適用する。
- 科学計算、保存科学値の新再解析、実分子load、実Hamiltonian行列、statevector／密度行列、
  signal、native科学回路build／compile、価格MCは実行していない。新しい詳細科学レビューでもない。
- 以下の数値は**人工整数／有理数の恒等式と抽象gate列の静的個数**。
  分子、geometry、DF rank、split、delta windowに関する科学性能値ではない。
  共通科学API、runner、validation manifest、採用済み研究方針は変更しない。

実装：[df_threshold_reflection.py](../../../src/trottertracks/df_threshold_reflection.py)。
検査：[test_df_threshold_reflection.py](../../../tests/test_df_threshold_reflection.py)。

## 2. 共通scalar中心化

### 2.1 同じ残差を作る操作

fragment和をH、thresholdのcenter和をC、指定した**共通**scalarをσとする。
比較する残差は全方式で同じ `G=H-σI`、外側は `exp(-isσ)` とする。

thresholdは以前の格子と反射を変えず、

\[
G=(C-\sigma)I+\sum_\ell\Lambda_\ell\mathbb E R_{\ell,J},\qquad
\lambda_{\rm th}(\sigma)=|C-\sigma|+\sum_\ell\Lambda_\ell.
\]

Pauli係数をcollect後に `H=d_I I+sum_P d_P P` とした場合は、

\[
G=(d_I-\sigma)I+\sum_{P\ne I}d_PP,\qquad
\lambda_{\rm P}(\sigma)=|d_I-\sigma|+\sum_{P\ne I}|d_P|.
\]

同一frameのI/Z/ZZ、collected global Pauli、thresholdすべてに同じσを適用する。
多fragmentのscalarは先に正確にcollectする。異なるframeを無条件に同一Pauliとしてcollectしない。
ここでの恒等式は係数screeningなしの共通exact targetについてのもの。
保存factorのexact積と、floatで既に丸めた展開係数を別々の方式へ渡して同一Hと呼ばない。

既存 `df_rte_tail.py:extract_df_tail` の `extract_identity_phase` は、
既存の係数screening後に**その辞書に残るidentity係数**を全て外出しする処理である。
Pauliのd_IとthresholdのCは一般に違うため、両方で同名policyを指定しても共通中心化にはならない。

今回の `ThresholdTail(..., scalar_shift=σ)` は指定σを外へ出し、C−σをtailへ残す。
既存policyとの曖昧な併用は拒否する。引数省略時は従来動作を保つ。

### 2.2 単一rank-one平方での成立条件

独立constantなし、full-Fock、単一fragment、

\[
F=\lambda(\sum_i\eta_i n_i)^2,\quad \lambda>0,\quad
P=\sum_i\max(\eta_i,0),\ N=-\sum_i\min(\eta_i,0)
\]

を考える。Lは `||F||=λ max(P²,N²)`、dはI/Z/ZZ辞書のidentity係数で、

\[
d=\frac\lambda4[(\sum_i\eta_i)^2+\sum_i\eta_i^2].
\]

前回のidentity保持massは両方L。今回、両方から同じσ=L/2を外出しすると、

\[
\lambda_{\rm th}=L/2,\qquad
\lambda_{I/Z/ZZ}=|d-L/2|+L-d.
\]

従ってd<L/2で厳密な差がある。なお、この純粋rank-one条件では実際にd≤L/2である。
Q=Ση_i²≤P²+N²なので、P≥Nの場合
`4d/λ≤2(P²+N²−PN)≤2P²`。PとNの交換も同様。
非零での等号は、一つの非零weightだけ、または正負各一つで絶対値が等しい場合。
この条件では中心化後もmass差が消える。L=0も自明な等号。

人工exact例（λ=1、full-Fock、単一frame。全方式から同じσを外出し）：

| η | σ | L | d | threshold mass | I/Z/ZZ mass |
|---|---:|---:|---:|---:|---:|
| (1) | 1/2 | 1 | 1/2 | 1/2 | 1/2 |
| (1,−1) | 1/2 | 1 | 1/2 | 1/2 | 1/2 |
| (1,1) | 2 | 4 | 3/2 | 2 | 3 |
| (2,−1) | 2 | 4 | 3/2 | 2 | 3 |
| (1,1,1) | 9/2 | 9 | 3 | 9/2 | 15/2 |
| (1,1) | 3/2 | 4 | 3/2 | 5/2 | 5/2 |

最後の行はPauli自身のidentityを外出しする共通σ=dを使うと、mass差が消える例。
一般の共通σでは
`λ_th=L/2+|L/2−σ|, λ_P=L−d+|d−σ|`。
d<L/2なら、σ≤dで一致、d<σ<L/2で差2(σ−d)、σ≥L/2で差L−2d。
従って「同じscalar処理なら常に差がない」も「中心化すれば常にstrict改善」も誤りである。

λ<0では両方式から共通σ=−L/2を外出しし、L=|λ|max(P²,N²)、dをidentity係数の絶対値とすると同じmass式になる。
独立constant追加、固定N sector、一般number-CDF、多fragmentにこの式を無条件に拡張しない。

### 2.3 強い辞書再構成への機会

単一のoccupation frameを変えずに全I/Z/ZZをcollectしても、Pauli基底の一意性から上の係数は変わらない。
一方、global Pauliの多fragment collectには相殺があり、thresholdのfragment別mass和を下回り得る。
人工例 `F+(−F)=0` は、global collectならmass=0だが、別fragmentのままのthresholdは正massを残す。
現実装はこの相殺や同値predicateの圧縮を新しく探索しない。

σ=L/2でthreshold massが `||F−σI||=L/2` に達するのは既知の反射分解の性質であり、
すべての合法な辞書再構成に対して新たな優越性があるわけではない。
対照にも同じΞ／しきい値辞書への再構成を許せば同じmassを得られる。
辞書変更を許さないI/Z/ZZとの条件付き差は残るが、それを独立した新sampling lawとは呼ばない。

各法が**独自に**scalarを最適化すると、Pauliはσ=dでL−d、thresholdはσ=L/2でL/2。
この比較は異なる残差・finite approximantの比較であり、同一finite meanの比較から分ける。

## 3. finite RTE、controlled位相、energy原点

### 3.1 理想補正平均が一致する条件

同じG、microstep時間s、非負の偶数cutoff K、独立かつ正確なatom／order law、
同じ積順序、正確なinvolutionと反射回転を仮定する。
[既存RTE][rte]のpaired lawはn=0,2,…,Kについて

\[
w_n(x)=\frac{|x|^n}{n!}\sqrt{1+(x/(n+1))^2},\quad
\mathcal B_K(x)=\sum_n w_n(x),\quad x=\lambda_{\rm dict}s.
\]

既存eventはproductsの後にrotationを適用し、位相は `(-1)^(n/2)`。
回転のcos/sinとw_nの平方根が相殺するため、

\[
\mathcal B_K\,\mathbb E[V]
=\sum_{n=0,2,\ldots,K}\frac{(-isG)^n}{n!}
 \left(I-\frac{isG}{n+1}\right)
=P_{K+1}(-isG).
\]

Kは**偶数event cutoff**で、補正後の多項式はdegree K+1。
辞書massが違っても右辺は同じ。外側の共通 `exp(-isσ)`、
同じ決定論backbone、同じmicrostep数を戻した補正平均も一致する。

**未補正の物理平均**は `P_(K+1)/B_K` なので辞書間で一致しない。
s≠0ならmass低下はB_Kを厳密に下げる（各非負weightが|x|に単調）。
atom確率 `|c_a|/λ_dict`、order確率 `w_n/B_K`、角度 `atan(λ_dict*s/(n+1))` は変わる。
同じGの辞書変更でも、event lawを古いまま使ってよいわけではない。

有限bit生成、float変換、角度合成、有限retry／失敗処理を入れた実装ではこの理想等式を再資格する必要がある。
今回の非零order検査はmockと形式的な非可換wordの有理数代数だけであり、
新しい科学的finite-RTE評価やsignalを計算していない。

### 3.2 approximantが変わる比較

σ自体を変えると一般に

\[
e^{-is\sigma_1}P_{K+1}[-is(H-\sigma_1I)]
\ne e^{-is\sigma_2}P_{K+1}[-is(H-\sigma_2I)].
\]

厳密な指数は同じHに戻るが、有限近似のbias・半径・保証は変わる。
identity保持の旧finite target、辞書ごとの自然なidentity抽出、今回の共通中心化を混ぜない。
同じG・s・Kの比較では実際の補正後PF／truncation biasは同じであり、massに基づく保守上界だけは変わり得る。

### 3.3 位相とalias

外出しσの位相はcontrol上の `exp(-isσ)`。
例えば `sink.phase(Angle(radians=-exact(s)*σ), (control,))` として保持する。
`emit_rte_event` は残差のeventだけを出力するため、この外側scalar位相は呼出側の責任である。
既存の実行wrapperへ自動接続したとは扱わない。
共通scalarを全interrogationの合計時間Tで一度へ融合することも、全方式に同じ条件で許す。
反射のcontrol単独π位相、負identity符号、paired-order位相は従来どおり別に一度ずつ処理する。

残差のenergyはE'=E−σ。位相は `−T(E−σ) mod 2π` で、復元energyにはσを戻す。
元のphaseへ戻すなら複素信号へ既知因子 `exp(-iTσ)` を掛ける。
aliasの事前energy区間はσだけ平行移動し、幅は変わらない。
中心を変えてもalias問題が自動的に解消するわけではなく、各RPE roundの分枝・測定軸・誤差条件を合わせる必要がある。
既存RPEのstate、energy精度、成功確率、shot表は今回再資格していない。

## 4. 平方を作らない同一predicate

従来のexact gridはそのまま使う。整数charge q、B=M、比較J∈1,…,Bとする。

| grid | 古典的閾値 | 同じpredicate |
|---|---|---|
| λ>0、z=q² | h=1+isqrt(J−1)=ceil(sqrt(J)) | [q≥h] XOR [q≤−h] |
| λ<0、z=B−q² | r=isqrt(B−J) | [q≥−r] XOR [q≥r+1] |

正側の二区間はh≥1により互いに素。負側の区間は包含関係なのでXORが差集合になる。
負側は `−r≤q≤r` を表し、J=Bならq=0だけを選ぶ。
正側のJ=k²では|q|=kを含む。J=k²+1ではk+1以上になる。
`math.isqrt` は任意長整数のexact平方根であり、float sqrtや四捨五入は使わない。

k-bit two's-complement chargeの最上位bitをXで反転すると、
u=q+2^(k−1)の符号なし順序になる。このuに既存 `_compare` を二回使う。
判定bitは同じflagへXORし、chargeは各比較で復元される。
閾値がregister範囲の外なら比較は定数true／falseへ置換する。

1. Gaussian逆変換。
2. 従来のoccupation制御定数加算でchargeを一度評価。
3. sign-bitを反転、二つの比較を同じflagへXOR、sign-bitを戻す。
4. 従来どおり反射の二位相、又は反射回転の二位相。
5. 比較列を実際の逆順でuncompute。
6. 連続する同一fragmentならchargeを保持して次のJを処理。
7. chargeの逆列、Gaussian正変換。

**value、abs、第二flag、独立carry registerを追加しない。**
旧方式の `w_z+k+1` workspaceから `k+1` へ減る。system nと外側control 1は別。
MCXのnative分解に必要な補助bitはこの論理workspaceに含めない。
一般二次grid（charge_weightsなし）は従来方式のまま。scalar-only atomには算術自体がない。

## 5. 実装差分と静的操作数

### 5.1 後方互換性・再利用

- 既存 `ThresholdGrid`、保存factorのexact target、全Jの範囲とuniform生成、fragment重みは不変更。
  中心化を指定すれば残差identityとmassは数学的に変わるが、平方回避オプションはそれらを変えない。
- `_add_constant`、`_compare`、`Instruction`、`Angle`、frame provider境界、
  中央phase、run共有、event schema／bind処理を再利用。
- 追加は `ThresholdTail.scalar_shift/sampled_scalar`、`_charge_predicate` と
  `rank_one_method="interval"`。既定値は `"square"`。
  `reflection_plan`、`rte_event_plans`、`emit_rte_event` から明示的に選べる。
- 基底hash、実adjointの呼出順、異幅run間の外側control、identity符号の二重計上防止を保持。
  native backendや新しいGaussian合成器は実装していない。

### 5.2 構造式

C_qはcharge評価、Sはorigin＋charge-bit平方のvalue蓄積、
D_w(J)は旧value比較、I_k(J)は今回の二区間比較の抽象X／MCX数とする。
各atomをuncomputeする同一fragmentの長さmのrunでは、

| 構成 | X／MCX | workspace | 中央位相 | Gaussian |
|---|---|---|---|---|
| 旧square | 2C_q+2S+2Σ_j D_w(J_j) | w_z+k+1 | 2m | 2 |
| 新interval | 2C_q+2Σ_j I_k(J_j) | k+1 | 2m | 2 |

frameなしならGaussianは0。新方式のX／MCX減少は
`S+ΣD_w > ΣI_k` の場合に限る。Sはrun中で既に一回へ共有されているため、
一atomでの削減量をm倍してrunの利得としてはいけない。

既存ladderの定数加算数は `A_w(c)=sum_(set bits b of |c|, b<w)(w−b)`、
`D_w(J)=A_(w+1)(−J)+1+A_w(J)`。
新区間はsign反転2個、範囲内の比較二回、正側ならflag反転1個（境界で定数化）を使う。
新最大control数は高々k。Gaussian providerの内部操作とnative分解のcontrol数は別。

### 5.3 人工入力の静的個数

全行はcontrolledな単一threshold反射。λ=±1、独立constantなし。
frame IDを持たせたためGaussianは全て2回、中央phaseは全て2個。
workspaceからsystemと外側controlは除外。左がsquare、右がinterval。

| 整数weight v | sign(λ) | J | k / w_z | X／MCX 旧→新 | 最大control 旧→新 | workspace 旧→新 |
|---|---:|---:|---:|---:|---:|---:|
| (1) | + | 1 | 2 / 1 | 14→36 | 2→2 | 4→3 |
| (1) | − | 1 | 2 / 1 | 16→34 | 2→2 | 4→3 |
| (1,−2) | + | 1 | 2 / 3 | 32→38 | 3→2 | 6→3 |
| (1,−2) | − | 1 | 2 / 3 | 34→22 | 3→2 | 6→3 |
| (1,−2) | + | 4 | 2 / 3 | 24→24 | 3→2 | 6→3 |
| (1,−2) | − | 4 | 2 / 3 | 26→36 | 3→2 | 6→3 |
| (1,1,1) | + | 5 | 3 / 4 | 66→68 | 4→3 | 8→4 |
| (1,1,1) | − | 5 | 3 / 4 | 76→66 | 4→3 | 8→4 |
| (3,−5,7) | + | 23 | 5 / 7 | 226→134 | 7→5 | 13→6 |
| (3,−5,7) | − | 23 | 5 / 7 | 242→114 | 7→5 | 13→6 |
| (31,−17,5) | + | 257 | 7 / 11 | 398→178 | 11→7 | 19→8 |
| (31,−17,5) | − | 257 | 7 / 11 | 420→158 | 11→7 | 19→8 |

共有run例：v=(3,−5,7)、λ=+1、回路順J=(24,25,23)、最後がmock角0.125のrotation。
square→intervalでX／MCXは324→286、最大controlは7→5、workspaceは13→6。
両方Gaussian2回・中央phase6個。これは既にrun共有を認めた比較である。

**小さい入力で演算数の悪化・同数もある。** workspace削減を無条件なgate数削減と呼ばない。
上表は期待event単価、native CX／T／RZ数、depth、速度、削減率の科学benchmarkではない。
Jや符号で費用が変わるため、将来の期待価格は正しいatom／order lawで戻す必要がある。

再現する静的集計の核（表の各weight、sign、Jへ適用）：

```python
from trottertracks import df_threshold_reflection as th
f = th.ThresholdFragment(
    "f", th.ThresholdGrid.from_df_square((3, -5, 7), 1), "frame", "mock"
)
a = th.ThresholdAtom(f, 23)
for method in ("square", "interval"):
    p = th.reflection_plan(a, rank_one_method=method)
    c = p.structural_counts()
    print(method, sum(c["x_by_control_arity"].values()),
          max(c["x_by_control_arity"], default=0),
          len(p.value) + len(p.charge) + int(p.flag is not None))
```

## 6. 強い決定論対照にも平方回避を許す

決定論的 `exp(-isλu²q²)` も、q²を量子registerへ格納する必要はない。
two's-complementのbit weightをa_bとすると、

\[
q^2=\sum_b a_b^2 x_b+2\sum_{b<c}a_ba_c x_bx_c.
\]

charge評価後、そのbit／bit対へ `−sλu²a_b²`、`−2sλu²a_ba_c` の位相を付ければよい。
旧N2のcharge-bit対位相と同じ構造である。chargeをuncomputeし、Gaussianを戻す。
workspaceはk、抽象位相は高々k(k+1)/2、二区間比較もJ生成もRTE normalizationも不要。
外側controlと定数phaseは同じ条件で保持する。これは設計式の静的比較で、native実装を新しく実行していない。

occupationのZ／ZZ位相列なら、そもそもchargeもvalueも不要。
従って、量子平方registerの省略をthreshold方式だけの独占利益にはしない。
thresholdの二区間predicateは少数の中央phaseで済む一方、その都度比較のMCXを払う。
bit対位相との費用差は角度精度・多重位相合成・frame共有を同条件にして判断する。

## 7. 人工検証

専用の既存一時venv：Python 3.12.3、NumPy 2.5.4（既存rte.pyの単独import用）。
共有環境の設定変更はない。新moduleはstdlibだけで動く。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /tmp/track-a-threshold-tech-20261110-venv/bin/python \
  -m unittest discover -s tests -p test_df_threshold_reflection.py -v
```

結果：**38 tests PASS、FAIL／ERROR／skipなし**。旧28件を保持し、10件追加。
初回は新テストが別クラスのsetUpClass実行順に依存して1 ERRORとなった。
既存rte.pyの単独loaderを共有し、クラス順序依存を除いて全件を再実行した。
科学条件変更、許容誤差緩和、失敗caseの除外はしていない。

追加確認：

- 共通σの前後でexact operator恒等式。残差identityの正負、scalar-only、zero、例外。
- 同じσをthresholdとI/Z/ZZ・collected diagonal Pauliへ適用し、同一残差を確認。
  中心σでのstrict／等号、σ=dでの差消失、fragment間collectでの相殺。
- 異なるmassの二辞書に対し、K=0/2/4のpaired numeratorを形式的な非可換wordの有理数代数で比較。
  独立なdegree K+1の展開とも一致。unitary／statevector／signalは作らない。
- 1〜5bitの全signed register値、全J=1,…,2^(2k−2)、両符号、
  flag初期0/1のXOR、charge復元、比較の逆列。
- 2000bit境界整数を含むisqrt、平方境界、q=0、最小負数、閾値のregister外定数化。
- 小さい人工chargeの全J、正負係数、反射／負角回転、control有無で新旧の絶対位相が一致。
  valueを省略し、全workspaceがzeroへ戻る。
- mixed-widthの異fragment、負係数、残差identity、同fragment連続eventの共有、
  Gaussian ID/hash・実逆列のdispatch、固定外側control、event metadata不変更。
- default square、一般二次、scalarの旧経路を保持。不正methodを拒否。

追加の後方互換照合では、固定commit 66335f281のmoduleを一時fileへ読み出し、
weights=(1)、(1,−2)、(1,−2,3)、λ=±1、全J、反射／負角回転、control有無の
計168件で、既定gate計画の全dataclass内容が固定版と完全一致した。
旧identity policyのscalar・massとgridも一致。これは人工データの技術比較であり、保存科学結果の再解析ではない。

phase sinkは一つの古典bit列とexact位相だけを追うtruth table。
mock Gaussianは符号付きorbital swapであり、実DF frameの科学的同値性評価ではない。
古い科学test全体や科学runnerは再実行しない。人工PASSを科学的採択、native価格、immutable CIとして扱わない。

## 8. 元Track Aとして残る価値と次段階

| 区分 | 今回の到達点 | 限界 |
|---|---|---|
| A：数学 | 同じ共通σ、同じ残差で、指定I/Z/ZZ辞書よりstrictに小さいmassを得る条件を確認 | 既知Ξ分解の辞書効果。全合法辞書・全fragmentに対する新優越性ではない |
| B：技術 | 同じ反射・event lawでvalue registerを省略。大きめの人工例では抽象bit gateも減少 | 小さい例でgate増加あり。native／Gaussian／期待価格は未評価 |
| 共通改善 | charge共有、frame共有、平方register回避 | 強い決定論対照にも許す。sampling専用の新機構ではない |
| samplingで残る交換 | 辞書massの低下で物理平均の減衰を減らせる可能性と、比較・frame費用の交換 | shot加重総費用、PF bias、成功条件は未確認 |
| C：全体 | 同一finite補正平均で比較する経路が具体化 | 科学的優位性・独立新規性・正式GOは認定しない |

同じG・s・Kと成功条件を固定すれば、B低下と期待1-shot費用の交換を測る経路はある。
例えばr個の同じmicrostep、同じ範囲型の補正推定器で他条件が同じなら、
`(B_th/B_P)^(2r) * (C_th/C_P)` が一つの比較量になる。
CはGaussian・算術・uncompute・control・共通backbone・測定を含む一shot全scopeであり、
上のbit-gate表をそのままCへ代入しない。全roundでの予算配分・PF bias・state条件も別途必要。

技術的提案は **「比較primitiveとして保持し、研究主線は保留」**。
前回未確認だった中心化の条件付きmass差と平方回避は具体化したため、
限定検証の候補資料にはなるが、既知要素を超えるsampling固有の新機構や総費用利益は未確定。
新しい自由度・分子grid・最適化探索を追加して自動継続しない。

次の担当は**ユーザー／GPT**：この到達点で研究主線を保留するか、限定科学検証を別途選ぶか判断する。
もし後者を選ぶなら、最小範囲は次のように固定することを提案する（**未承認・未実行**）。

1. 対象：一つの明示されたrank-one DF fragment／短いinterrogation。
   同じ保存factorのexact target又は同じ丸め済み係数targetを全方式で統一する。
   full-Fockかsectorか、frame、state、時間、精度、成功条件を指定する。
2. 比較：共通σ=sgn(λ)L/2のthreshold／DF I/Z/ZZ／collected Pauli。
   強い決定論側には共通中心化、Z/ZZ、charge-bit位相、合法なbasis共有を許す。
   独自最適scalarの比較は異なるapproximantとして別枠にする。
3. 最小実行：固定G・s・Kでfinite-bit補正平均とcontrolled位相を資格化し、
   事前に指定したnative scope・角度誤差でwhole circuit費用と必要shotを接続する。
   旧SOURCEのRTE schema／order式・Gaussian providerは再利用候補だが、旧価格・state・科学承認は移送しない。
4. STOP：指定した一条件の同値性・費用比較の報告で停止。
   不一致時はその場で停止し、追加geometry、分子grid、長RPE、cutoff再探索へ進まない。
   ここでの計画は科学実行や詳細科学レビューの許可ではない。

**Codex**：今回の実装・人工検査・報告・必要索引をcommit／pushし、remote照合して完了。
最終報告後mandatory STOP。新科学計算・数値再解析・native科学費用検証・詳細科学レビュー・正式採択へ自動進行しない。

[base]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/66335f2812f5d8fb1a9b38171180951825fd8dfa
[previous]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66335f2812f5d8fb1a9b38171180951825fd8dfa/docs/research/track_a_threshold_reflection_technical_20261110/REPORT.md
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66335f2812f5d8fb1a9b38171180951825fd8dfa/src/trotterlib/rte.py
