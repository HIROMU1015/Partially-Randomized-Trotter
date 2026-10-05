# 有限RTE平均振幅演算子の位相・半径解析：研究成果・技術整理

- 固定日: 2026-09-27 (JST)
- 基点: `16d4482da6f445877c68c0d613f1d9ccdcf2f26b`
- 状態: `RESEARCH_TECHNICAL_RECORD_COMPLETE_T3_UNRESOLVED_NO_NEW_COMPUTE`
- 文書種別: 研究成果の技術記録。投稿原稿または論文化の開始を意味しない
- 数値判断: FR-R1b の `MECHANISM_ONLY_NO_PRACTICAL_GO` を維持する
- 理論監査: [定理・先行研究レビュー](../../fr_theorem_prior_art_review_71169d8.md)を補助資料として採択
- 計算方針: 研究成果の技術整理を完了。新しい数値計算を開始しない

## 0. この文書の役割と結論

本書は、FR-R1b 後の研究を「候補探索」から「一つの成果の完成」へ切り替えるための正本である。既存の数値結果を増やすことではなく、有限 RTE が作る平均信号について、何が既知で、どの直接帰結まで成立し、何が未解決かを固定する。

定理・先行研究レビューの結果、現在具体化されている内容は新手法の主定理としてではなく、平均振幅演算子の意味、位相・半径境界、成立域、反例を統合した `TECHNICAL_NOTE` として閉じる。これは将来の有限 RTE 制約付き到達可能性定理を否定する判断ではないが、未定義の T3 を理由に計算系列を継続しない。

FR-R1b では、位相方向の改善機構は確認されたが、固定予算で一方向に資源優位となる witness は得られなかった。この停止判断を変更せず、FR-R2、H4/H12、長時間 RPE、compiler 総 cost、大規模 grid へは自動的に進まない。

## 1. 研究課題と固定する成果

### 1.1 研究課題

> finite paired-Taylor RTE の平均信号に含まれる位相不変な正の scalar を物理的な減衰と分離したとき、利用可能な演算子・状態情報の層ごとに、位相誤差と信号半径をどこまで保証でき、どこから先は保証できないか。

### 1.2 研究成果として固定する内容

> 有限RTEの平均振幅演算子を正scalarと方向依存補正へ分解し、norm-disk baselineのsharpness、追加構造で得られる既知の改善、finite-RTE固有の未解決範囲を、同じ意味規約の下で区分する。

scalar分離、weak-value表現、q-numerical-range幾何、telescoping norm bound、局所Taylor remainderを
独立した新規定理とはしない。technical noteの価値は、これらを有限RTEの正しい対象、成立域、
数値上の負の結果とともに統合し、未解決の到達可能性を達成済み主張から分離することに置く。

## 2. 対象と記号

### 2.1 finite paired-Taylor RTE

次数 K+1 の打切り多項式を `P_(K+1)(-i τ h) = Σ_(n=0)^(K+1) (-i τ h)^n / n!` とする。対象は、独立にsampleされたunitary列の期待値として得られる**平均振幅演算子** `M=E[U_ω]` と、その指定状態に対する複素振幅である。random-unitary channel `Φ(X)=E[U_ω X U_ω†]` とは区別する。主な既存数値証拠は K=2、すなわち P3 に対するものである。

### 2.2 scalar 分離

既存FR契約と同じく、sampled-unitary平均の正規化を 𝔅、補正後numeratorから除く正scalarを Γ_c とする。

`A_mean = A_corr/𝔅`, `A_corr = Γ_c Â_corr`, `Â_corr = U(I+Q)`, `𝔅>0`, `Γ_c>0`。

参照信号と観測信号を `z0 = <ψ|U|ψ>`、`zhat_corr = <ψ|Â_corr|ψ>`、
`z_corr = Γ_c zhat_corr`、`z_obs = (Γ_c/𝔅) zhat_corr` とする。Γ_c と 1/𝔅 は正なので
位相を変えないが、物理的半径、shot 数、実行 cost には係数 Γ_c/𝔅 が影響する。

### 2.3 比較境界

- U は、同じ outer PF と split を保ったまま randomized tail の有限 RTE だけを exact exponential に置き換えた参照である。
- outer PF bias、H_D 内部近似、state-preparation error は、明示的に追加しない限り本書の finite-RTE 誤差へ混ぜない。
- sampling は独立な fresh draw を基本契約とし、reuse や correlated sampling は別問題とする。
- 位相保証には必ず `|z0| >= ρ_min > 0` のような signal-floor 条件を置く。

## 3. 主張の階層

| ID | 位置づけ | 内容 | 現在状態 |
|---|---|---|---|
| C1 | technical note・基礎 | 正 scalar と方向依存補正の分離、および phase/radius certificate の正しい定義 | 既知結果と初等的導出の統合。独立した新規性は主張しない |
| C2 | technical note・限界整理 | norm-diskとの比較条件、一般norm情報でのsharpness、追加構造を使う既知幾何、未解決なfinite-RTE到達可能性を区分する | 一般情報だけでの一様strict改善には反例。finite-RTE制約付き最適性は未解決 |
| C3 | 不実施 | certificate を用いた finite-RTE parameter/resource design | FR-R1bの実用GO不通過と理論監査結果を受けて開始しない |

この原稿で扱う理論情報は、凍結済みFR-R0の実装・certificate層`I0/I1/I2`と混同しないよう、次の`J0/J1/J2`に分ける。両者は用途が異なり、一対一対応を仮定しない。

- J0: `||Q||` と ρ_min だけ。
- J1: Q の Hermitian/anti-Hermitian 成分、符号、sector、交換・対称性など、状態非依存の追加構造。
- J2: 対象状態または状態族に関する directional expectation、moment、部分空間情報。

C2 の「改善」は、比較対象と同じ情報層だけを用いて証明しなければならない。J2 の情報を用いた bound を J0 baseline と比較して強さを主張する場合は、情報取得 cost と適用条件を別に示す。

## 4. 出発点として固定する既知・検算済み事実

### 4.1 正 scalar の除去

𝔅>0、Γ_c>0 なら `arg(z_obs) = arg(zhat_corr)`、
`|z_obs| = (Γ_c/𝔅)|zhat_corr|` である。したがって phase mechanism と physical attenuation は
分離して論じる。ただし resource comparison では Γ_c/𝔅 を戻して両者を再び統合する。

### 4.2 weak-value / q-numerical-range 表現

`z0 = ρ exp(iφ) != 0` とし、`|u> = |ψ>`、`|v> = exp(iφ) U†|ψ>` と置くと、
`<v|u> = ρ` であり、`zhat_corr/z0 = 1 + <v|Q|u>/ρ` となる。

固定 overlap の q-numerical range を

`W_ρ(Q) = {<y|Q|x> : ||x||=||y||=1, <y|x>=ρ}`

と定義すれば、実際の補正比は `1 + W_ρ(Q)/ρ` に含まれる。これは containment であって、W_ρ(Q) の全点が有限 RTE により実現されるとは主張しない。また、この書換え自体は新規性主張に使わない。

### 4.3 norm-disk baseline

`ω = (zhat_corr-z0)/z0` とし、`|ω| <= η < 1` なら、標準的な disk geometry から
`|arg(1+ω)| <= asin(η)`、`1-η <= |1+ω| <= 1+η` が得られる。FR certificate は、
この baseline と同じ情報で方向制約を証明できる場合にだけ「より強い」と呼ぶ。物理半径へ戻す際は
Γ_c/𝔅を掛ける。

### 4.4 P3 の局所漸近展開

既存契約の K=2 規約では、scalar Hamiltonian h に対して

`log P3(-ix) = -ix - x^4/24 - i x^5/30 + x^6/72 + O(x^7)`。

したがって x=Th/r について

`P3(-iTh/r)^r = exp(-iTh) [1 - T^4 h^4/(24r^3) - iT^5 h^5/(30r^4) + O(r^-5)]`。

これは scalar recentering により r^-3 の共通 radial drift を除き、phase-relevant term がより高次に現れ得ることを示す出発点である。固定されたbounded Hermitian familyに対する局所一様remainderはfunctional calculusと通常のTaylor remainderの直接帰結として扱う。一般の非可換interleaving、負時間、任意のfinite-RTE列まで含む一様評価は別の未解決部分である。

### 4.5 二固有値の検算例

等重みの二固有値 a,b で `cos((a-b)T/2) > 0` の branch を選ぶと、共通 radial scalar を分離する前の先頭位相項には、規約に応じた符号を除いて

`T^4(a^4-b^4) tan((a-b)T/2) / (48r^3)`

が現れる。この例は mechanism の検算であり、一般保証ではない。係数、branch、remainder を独立に再証明してから定理として用いる。

### 4.6 signal floor の必要性

`|z0| -> 0` では、絶対 operator error が小さくても相対位相は不安定になり得る。したがって signal floor のない一様な相対位相保証は主張しない。near-zero signal は失敗条件として明示する。

## 5. 先行研究との差分監査

| 文献・概念 | 既知の内容 | 本研究との関係 | 現在判定 |
|---|---|---|---|
| Günther et al., partially randomized time evolution | partial randomization と位相推定の資源評価 | 有限 RTE の出発点。単なる再実装は新規でない | `KNOWN_NOT_NOVEL` |
| Wan–Berta–Campbell, randomized statistical phase estimation | randomized experiment による phase estimation と統計保証 | 観測モデル・shot accounting の近接領域 | `CLOSE_NOT_DIRECT` |
| Gu et al., noise-resilient phase estimation with randomized compiling | Hermitian Kraus operator を持つchannel perturbationの一次 phase invariance | 対象はchannelであり、本書の平均振幅演算子へ移すには別の議論が必要 | `KNOWN_RELATED_DIFFERENT_OBJECT` |
| Ogawa et al., operational formulation of weak values | 小変換に対する postselection amplitude の感度としての weak value | 4.2 の比表現と近い。表現自体は新規でない | `KNOWN_NOT_NOVEL` |
| Li ほか、q-numerical range | fixed overlap matrix element の値域と幾何 | certificate の自然な既知言語 | `KNOWN_NOT_NOVEL` |
| Yi–Crosson / Li, product formula spectral or QPE error | PF の spectral error、gap、residual に基づく解析 | outer PF と位相推定の誤差解析に近い | `CLOSE_NOT_DIRECT` |
| Casares et al., SPRINT | Symmetry-Protected Randomized near-Integrable Trotter。randomized PFのspectral shift、damping、重み変化を解析 | 位相と減衰の区別は既知。finite paired-Taylorの全域certificateとは同一でない | `KNOWN_RELATED_NOT_IDENTICAL` |
| Hu–Jin, nonunitary amplitude/phase analysis | 非ユニタリ発展の amplitude/phase 構造 | finite RTE の非ユニタリ平均との重複可能性 | `OPEN_SCOPE_AUDIT` |

検索で同一表現が見つからないことは新規性の証拠にしない。レビューで全文を取得できなかった文献は未確認と明記し、未取得を「同じ定理がない」根拠にしない。現在具体化されたC1/C2の大部分は既知結果または直接帰結としてtechnical noteへ置く。

## 6. 証明義務

| ID | 証明・監査項目 | 完了条件 | 現在状態 |
|---|---|---|---|
| T0 | semantics と branch の固定 | K,r,T,𝔅,Γ_c,U,Q,arg の規約を一意にし、負時間を含む | 契約・監査済み。平均振幅演算子とchannelを区別 |
| T1 | 順序を保つ有限RTEのoperator bound | telescopingでfull product/orderを保つnorm remainderとphase/radius boundを導く | `DIRECT_COROLLARY`。新しいorder-sensitive sharp boundではない |
| T2a | 登録FR式の比較条件 | norm-diskより小さくなる明示的不等式を導く | `DIRECT_COROLLARY` |
| T2b | J0での一様strict improvement | normとsignal floorだけでnorm-diskを一様に改善する | `FALSE_OR_COUNTEREXAMPLE`。rank-one構成がdiskを達成する |
| T2c | 追加構造によるstrictness・最適性 | J1/J2またはfinite-RTE構造が許す改善とsharpnessを区分する | 一般q-range幾何へ還元できる部分は既知。finite-RTE固有部分は`UNRESOLVED` |
| T3 | finite-RTE制約付き限界・最適性 | 情報写像、許容列、scalar固定規則を定義し、同情報の到達可能集合で上下界を一致させる | `UNRESOLVED`。現状は定理自体が未定義で、達成済みcontributionに含めない |
| T4a | P3局所remainder | bounded Hermitian familyと二固有値例で一様局所剰余を与える | `DIRECT_COROLLARY` |
| T4b | 一般finite-RTE内部実現 | 非可換interleaving、負時間、一般のN/r依存を含めてQの実現性と一様剰余を与える | `UNRESOLVED`。T3の到達可能性を含む |
| T5 | certificate 入力の取得可能性 | 必要情報、古典計算/測定 cost、abstention 条件を明示 | 未完 |
| T6 | controlled implementation と減衰 | normalization、success/amplitude、controlled relative phase を整合 | 一部検証済み、一般化未完 |

数値実験の pass は、これらの証明義務を埋めない。とくに8個のstrict-gain witnessは、凍結済み比較で得た**数値計算上の存在候補**であり、解析的な存在証明、一般十分条件、T3の限界定理ではない。また`OPT`は登録区間内の数値最適化ラベルであり、全scalarに対する数学的大域最適性を意味しない。

## 7. 解析結果・反例・未解決範囲

この節が研究成果の数理的な本体である。`DIRECT_COROLLARY`は既知結果または初等的な恒等式から
同じ仮定の下で導けること、`FALSE_OR_COUNTEREXAMPLE`は記載した一般主張に反例があること、
`UNRESOLVED`は命題の対象または達成可能性が未確定であることを表す。

### 7.1 T1：順序を保つ有限非可換積のbound

有限個のunitary `U_j` と局所corrected factor

\[
A_j=U_j(I+D_j)
\]

を考える。scalar再中心化を行う場合、`D_j`は再中心化後の局所誤差とする。deterministic factorは
`D_j=0`として同じ列に含められる。積とprefixを

\[
U=U_N\cdots U_1,\qquad
A=A_N\cdots A_1,\qquad
V_{j-1}=U_{j-1}\cdots U_1
\]

とし、

\[
\widetilde D_j=V_{j-1}^{\dagger}D_jV_{j-1}
\]

と定める。因子の順序を変えずに移動すると

\[
U^\dagger A
=(I+\widetilde D_N)\cdots(I+\widetilde D_1)
=:I+Q .
\]

従って

\[
Q=\sum_{j=1}^{N}\widetilde D_j+R.
\]

`\|D_j\|\le e_j`ならunitary共役によって`\|\widetilde D_j\|\le e_j`であり、積の二次以上を
submultiplicativityで評価して

\[
\|R\|\le R_2
:=\prod_{j=1}^{N}(1+e_j)-1-\sum_{j=1}^{N}e_j,
\]

\[
\|Q\|\le E
:=\prod_{j=1}^{N}(1+e_j)-1
\]

を得る。交換可能性は仮定していない。このboundは任意の順序付き非可換列に適用できる一方、
順序依存の相殺を利用するsharp boundではない。

`D_j=F_j+iG_j`をHermitian分解とし、

\[
\|F_j\|\le a_j,\qquad
\|G_j\|\le b_j,\qquad
a=\sum_j a_j,\qquad b=\sum_j b_j
\]

とする。`z_0=\langle\psi|U|\psi\rangle`、`\rho=|z_0|\ge\rho_0>0`に対し

\[
|\eta\rangle=(U^\dagger-z_0^*I)|\psi\rangle
\]

と置けば、

\[
\langle\psi|\eta\rangle=0,\qquad
\|\eta\|=\sqrt{1-\rho^2}
\]

であり、

\[
\frac{\langle\psi|A|\psi\rangle}{z_0}
=1+\langle\psi|Q|\psi\rangle
+\frac{\langle\eta|Q|\psi\rangle}{z_0}.
\]

`\kappa_0=\sqrt{1-\rho_0^2}/\rho_0`とすると最後の項の絶対値は`\kappa_0E`以下である。
従って

\[
L=1-a-R_2-\kappa_0E,\qquad
Y=b+R_2+\kappa_0E
\]

と置き、`L>0`なら

\[
|\Delta\phi|
\le \arctan\frac{Y}{L},
\]

\[
|z_{\rm obs}|
\ge \frac{\Gamma_c}{\mathcal B}\rho_0L.
\]

これはT1の成立部分であり、分類は`DIRECT_COROLLARY`である。位相boundと物理半径boundは、
正scalar `\Gamma_c/\mathcal B`を位相から除き、半径へ戻す同じ規約で記述されている。

### 7.2 T2a：登録FR boundがnorm-diskより小さくなる条件

J0のnorm-diskでは`E<\rho_0`のとき

\[
|\Delta\phi|
\le \arcsin\frac{E}{\rho_0}
=\arctan\frac{E}{\sqrt{\rho_0^2-E^2}},
\]

\[
|z_{\rm obs}|
\ge\frac{\Gamma_c}{\mathcal B}(\rho_0-E).
\]

従って、7.1節のFR位相boundがstrictに小さくなる条件は、`L>0`の下で

\[
\boxed{
\frac{Y}{L}
<
\frac{E}{\sqrt{\rho_0^2-E^2}}
}
\]

である。半径下界もstrictに改善するには、これとは別に

\[
\boxed{
\rho_0L>\rho_0-E
}
\]

が必要になる。位相改善と半径改善は同値ではない。

例えば単一Hermitian相対誤差について`a=e=E`、`b=R_2=0`、`\rho_0=4/5`とすると、
`\kappa_0=3/4`で

\[
L=1-\frac{7E}{4},\qquad Y=\frac{3E}{4}.
\]

十分小さい`E`では位相boundはnorm-diskより小さいが、

\[
\rho_0L=\frac45-\frac{7E}{5}
<\frac45-E
\]

なのでFRの半径下界は小さい。従って「位相がstrictに改善すればphase/radiusの両方が改善する」
という主張は採用しない。T2aは二つの既存boundの比較であり`DIRECT_COROLLARY`である。

### 7.3 T2b：J0だけではnorm-diskがsharpである反例

一般演算子クラスとして、全てのunitary `U`、unit vector `|u\rangle`、`\|Q\|\le E`、
`|\langle u|U|u\rangle|\ge\rho_0`を許し、`0<E<\rho_0\le1`とする。

`\langle v|u\rangle=\rho_0`を満たすunit vectorを選び、`U^\dagger|u\rangle=|v\rangle`となる
unitaryを取る。任意の`|\zeta|\le E`に対して

\[
Q=\zeta|v\rangle\langle u|
\]

とすれば、

\[
\|Q\|=|\zeta|,\qquad
z_0=\langle u|U|u\rangle=\rho_0,
\]

\[
\frac{\widehat z}{z_0}
=1+\frac{\zeta}{\rho_0}.
\]

従って、J0が許す補正比はnorm-disk全体を実際に含む。`\eta=E/\rho_0`として

\[
\zeta=\rho_0
\left(
-\eta^2+i\eta\sqrt{1-\eta^2}
\right)
\]

を取れば位相は`\arcsin\eta`に等しく、`\zeta=-E`なら半径`\rho_0-E`を達成する。

従って、normとsignal floorだけを知る一般演算子クラスでは、norm-diskより一様に小さい位相上界も、
一様に大きい半径下界も保証できない。J0での一様strict improvementは
`FALSE_OR_COUNTEREXAMPLE`である。

この反例は一般のrank-one `Q`を使う。これがfinite paired-Taylor RTEから生成可能とは示していない。
従って「全finite-RTEでもnorm-diskがsharp」という結論には使わない。

また、`\rho_0=\epsilon^2`、`\zeta=i\epsilon`とすれば`\|Q\|\to0`でも位相は
`\arctan(1/\epsilon)`へ動く。signal floorなしの一様相対位相保証が成立しないことも分かる。

### 7.4 T2c：追加構造による改善と既知幾何

Hermitian二固有値演算子

\[
Q=cI+dZ
\]

を考える。fixed-overlap q-numerical rangeの既知の楕円定理、または二次元での直接計算により、
`W_\rho(Q)`の中心は`\rho c`、実軸半径は`d`、虚軸半径は
`d\sqrt{1-\rho^2}`となる。

`c=0`、`0<d<\rho\le1`なら、補正比`1+W_\rho(Q)/\rho`の楕円から得られる位相上界は

\[
\beta_{\rm ell}
=
\arctan
\frac{d\sqrt{1-\rho^2}}{\sqrt{\rho^2-d^2}}.
\]

一方、norm-diskは

\[
\beta_{\rm disk}
=
\arcsin\frac{d}{\rho}
=
\arctan
\frac{d}{\sqrt{\rho^2-d^2}}.
\]

`0<\rho<1`では`\beta_{\rm ell}<\beta_{\rm disk}`であり、`\rho=1`では楕円が実軸上に退化して
位相上界は0になる。追加のHermitian構造がstrict改善を与え得ることは正しいが、一般機構は既知の
q-numerical-range幾何で説明される。

任意の非可換finite-RTE full `Q`はHermitianまたは二固有値とは限らない。この結果を一般の
finite-RTE積のsharpnessへ移すことはできず、その部分は`UNRESOLVED`である。

### 7.5 T4a：bounded Hermitian P3反復の一様局所展開

Hermitian `h`、`\|h\|\le H_*`、`|T|\le T_*`を固定し、
`M=T_*H_*`、`r\ge\max(1,4M)`とする。`P_3`について`|z|\le1/2`では

\[
|P_3(-iz)-1|
\le\frac12+\frac18+\frac1{48}
=\frac{31}{48}<1.
\]

従って原点から連続なlog branchが存在する。`f(z)=\log P_3(-iz)+iz`はこのdiskで解析的かつ有界で、
Cauchy評価により`|z|\le1/4`で7次以上の剰余を定数倍の`|z|^7`で一様に抑えられる。
spectral theoremでscalar評価をoperator normへ移すと

\[
r\log P_3(-iTh/r)
=
-iTh
-\frac{T^4h^4}{24r^3}
-i\frac{T^5h^5}{30r^4}
+\frac{T^6h^6}{72r^5}
+O_M(r^{-6}).
\]

全て同じ`h`の関数なのでこの段階では可換であり、指数を戻して

\[
\boxed{
P_3(-iTh/r)^r
=
e^{-iTh}
\left[
I-\frac{T^4h^4}{24r^3}
-i\frac{T^5h^5}{30r^4}
+O_M(r^{-5})
\right]
}
\]

を得る。`|z_0|\ge\rho_0>0`を固定し、相対補正がbranch内に入る十分大きい`r`では、
`\operatorname{Arg}(1+w)=\operatorname{Im}w+O(|w|^2)`も一様に使える。

ここで`h^4`は一般にはscalarでない。正scalar再中心化で除けるのは共通成分だけであり、
重ね合わせ状態では`r^{-3}`のradial非一様性が位相へ移り得る。この局所一様展開は
Taylor解析とfunctional calculusの`DIRECT_COROLLARY`である。

異なる`h_j`を含む非可換積ではlogを加算しない。局所remainderを個別に評価し、7.1節の順序付き積で
合成する必要がある。因子数`N`が`r`とともに増える場合は`\sum_j e_j`、`R_2`、
正scalar `\gamma_j`が0から離れる条件を別に追う。この一般部分はT4bに属する。

### 7.6 T4aのfinite-RTE内部例：一般superpositionの先頭位相項

`h|a\rangle=a|a\rangle`、`h|b\rangle=b|b\rangle`、

\[
|\psi\rangle
=\frac{|a\rangle+|b\rangle}{\sqrt2},
\qquad
d=\frac{(a-b)T}{2},
\qquad
\cos d>0
\]

とする。参照信号は

\[
z_0=e^{-i(a+b)T/2}\cos d.
\]

7.5節の展開から、negative-exponent規約でのsigned phaseは

\[
\boxed{
\operatorname{Arg}\frac{z_{\rm corr}}{z_0}
=
\frac{T^4(a^4-b^4)\tan d}{48r^3}
-\frac{T^5(a^5+b^5)}{60r^4}
+O(r^{-5})
}
\]

となる。絶対位相誤差はこの式の絶対値であり、signed coefficientと区別する。
uniformな主張では`|a|,|b|,|T|`を有界にし、`\cos d\ge\rho_0>0`を置く。

paired-Taylor class内の具体例として

\[
h=\frac{Z\otimes I+I\otimes Z}{2}
\]

を取り、固有値1の`|00\rangle`と固有値0の`|01\rangle`の等重み重ね合わせを使う。
deterministic interleavingをidentityとすれば、`r`回のfinite Taylor列そのものである。
`0<T<\pi`では

\[
\operatorname{Arg}\frac{z_{\rm corr}}{z_0}
=
\frac{T^4\tan(T/2)}{48r^3}
-\frac{T^5}{60r^4}
+O(r^{-5}).
\]

従って、単一固有値で位相誤差が`r^{-4}`でも、一般superpositionのfinite-RTE信号は
`r^{-3}`になり得る。正scalarで割ってもこの位相項は消えない。

これはfinite-RTE内部で成立する限界例だが、全finite-RTEの最適性やnorm-diskのsharpnessは示さない。

### 7.7 T3/T4b：未解決問題を再開可能な形で保存する

finite-RTE制約付きの情報最適性を主張するには、少なくとも

\[
\mathcal A(i)
=
\{x\in\mathcal C_{\rm RTE}:\mathcal I(x)=i\},
\]

\[
\Theta^*(i)
=
\sup_{x\in\mathcal A(i)}
\left|
\operatorname{Arg}
\frac{\widehat z(x)}{z_0(x)}
\right|
\]

を定義する必要がある。`x`には以下を含める。

- 許されるHamiltonian分解と局所operator class
- cutoff、時間、反復数、係数符号
- deterministic interleavingとsampling独立性
- 入力状態またはstate promise
- normalizationと正scalarの固定規則
- signal floorとbranch規則

`\mathcal I(x)`には、正確な値か上界か、局所情報かfull-product情報か、取得費用を含めるかを明記する。

上からのcertificateだけでは最適性にならない。同じ情報値`i`を共有するfinite-RTE列が上界を達成する、
または任意に近づくことが必要である。7.3節の一般rank-one構成とq-numerical range全体は、
そのfinite-RTE達成可能性を自動的に保証しない。

現時点では`\mathcal C_{\rm RTE}`と`\mathcal I`を固定していないため、T3は`UNRESOLVED`である。
非可換interleaving、負時間、因子数の増加を含む一般一様remainderと内部実現T4bも同じ未解決範囲に
属する。これは「補題を一つ追加すれば完成する状態」ではなく、新しい研究を開始するなら先に問題設定を
作る段階である。


## 8. 既存証拠とその限界

| 検証 | 固定された結果 | この原稿での用途 | してはいけない外挿 |
|---|---|---|---|
| FR1 | phase/amplitude 分離の基本 mechanism と stop 条件を確認 | 記号・semantics の検算 | 一般定理、H4/H12 資源優位 |
| FR-R1a | 既存 artifact の事後解析で scalar recentering と directional geometry を確認 | C1/T0 の経験的裏付け | blind prediction、実用 GO |
| FR-R1b | 非一様 4x4、20 条件・61 状態、610 records、227 applicable、soundness failure 0、strict FR-gain 8、one-sided fixed-budget witness 0 | C2の数値候補と、C3を開始しない根拠 | 解析的存在証明、普遍的改善、resource advantage |

### 8.1 証拠の参照先とprovenance

| 証拠 | 文書 | machine-readable artifact | 固定status |
|---|---|---|---|
| FR-R1a | [正scalar事後再解析](../fr_revision_fr1a_posthoc.md) | [JSON](../../artifacts/fr_revision_fr1a_posthoc/2026-09-26/fr_revision_fr1a_posthoc_v1.json) | `POSTHOC_SCALAR_EXPLAINS_OLD_GAIN` |
| FR-R1b | [非一様4×4結果](../fr_revision_nonuniform.md) | [JSON](../../artifacts/fr_revision_nonuniform/2026-09-27/fr_revision_nonuniform_r1b_v1.json) | `MECHANISM_ONLY_NO_PRACTICAL_GO` |

FR-R1bのresult fingerprintは
`affac0ae8132450ccb2de3512b6a463a3f9d7b1ac8a6f12cc38303ac75e891d4`、
文書記載のfile SHA-256は
`e2a6f9326951fe67e979022dc733704e0c036d3342ab3cb317c5f77b885aeebc`である。
保存報告では専用test `5 passed`、FR系列関連test `13 passed`、全suite
`628 passed, 2 skipped, 4 warnings`で失敗0だった。

これらは保存時のprovenanceであり、今回の文書整理でraw bytesからhashを再生成したことや、
testを再実行したことを意味しない。対象は登録した4×4 toy条件であり、H4/H12、一般化学系、
長RPE、compiled総costへ外挿しない。


現時点の決定は `MECHANISM_ONLY_NO_PRACTICAL_GO` である。既存 artifact から、少なくとも次の図表だけを再利用候補とする。

1. scalar recentering 前後の phase/radius error。
2. norm-disk と FR directional certificate の幅の比較。
3. strict-gain witness と abstention/failure 条件の対比。
4. fixed-budget で one-sided witness が 0 だった負の結果。

これらのために新しい grid は実行しない。

## 9. 研究上の結論

1. 平均振幅演算子、正scalar、物理半径の意味関係と、順序を保つ積boundは整合している。
2. J0、すなわち`||Q||`とsignal floorだけを使う一般演算子クラスでは、norm-diskはsharpであり、
   一様なstrict改善はできない。
3. Hermitian性や二固有値構造を使えばstrictな位相改善は可能だが、その一般機構は既知の
   q-numerical-range幾何へ還元される。
4. bounded Hermitian P3反復の局所展開と一様剰余、一般superpositionで`r^-3`位相項が残り得る例は
   成立するが、通常のTaylor展開とfunctional calculusの直接帰結である。
5. finite-RTE制約付き到達可能集合でのsharpな最適性は、情報写像と許容クラスが未定義のため未解決である。
6. FR-R1bは機構の非空性を数値的に支持したが、固定予算の設定差は0で、実用GOは得られていない。

以上から、本成果は新手法の主定理ではなく`TECHNICAL_NOTE`相当の研究記録として確定する。
T3を未解決として保存するが、この未定義項目を理由に数値研究を継続しない。

## 10. 条件付き応用C3の境界（今回は不実施）

C3を将来の別研究として行う場合は、C1/C2の独立した差分を先に確定し、比較では次をすべて固定する。

- 同じ target time、推定 task、state promise、誤差 budget。
- 同じ K,r 候補集合と、実行可能性条件。
- baseline は同じ入力情報を使う強い norm/geometry certificate。
- deterministic bias、signal radius、shot 数、一実行あたり cost を分離してから統合する。
- oracle、guaranteed、measured performance を混同しない。

たとえば Re/Im を独立な ±1 outcome で測り、各軸に N shots を与えて Hoeffding で評価する。
半径下界 v>0、許容位相誤差 ε、deterministic bias b<ε、かつ `0 < ε-b <= π/2` の下で、
各座標誤差を `v sin(ε-b)/sqrt(2)` 以下に抑える保守的な十分条件は

`N >= 4 log(4/α) / [v^2 sin^2(ε-b)]`

のような bound を使用できる。ただしこれは一つの保守的な十分条件であり、最適測定・最適推定器の主張ではない。certificate 入力の取得 cost が利益を上回る場合、C3 の実用性は主張しない。

## 11. 追加証拠

今回の`TECHNICAL_NOTE`分類を閉じるための追加計算は行わない。図表が必要なら既存artifactだけを
再利用し、FR-R1bのfingerprintとgateを変更しない。T3またはT4bを将来の別研究として再開する場合は、
問い、情報写像、許容finite-RTE列、scalar規則、評価量、GO/STOPを新しい事前契約へ固定する。
H4/H12、FR-R2、長RPE、full compiled cost、多数のPF family、広いdelta/r/L_D gridは本noteに含めない。

## 12. 研究記録として固定したもの

- 対象を平均振幅演算子へ固定し、random-unitary channelと区別した。
- T1、T2a、T4aの導出、T2bの反例、T2c/T3/T4bの未解決範囲を記録した。
- 数値証拠を解析的証明から分離し、FR-R1bの`MECHANISM_ONLY_NO_PRACTICAL_GO`を保存した。
- 文献監査で確認できなかった全文は未確認として残し、不在証明に使わない。
- 新しいrunner、計算artifact、検証IDを追加しない。

将来T3を再開する場合は、情報写像、許容finite-RTE列、scalar規則、達成可能性を別の事前契約で定義する。

## 13. 明示的に主張しないこと

- H4/H12 や一般化学系への一般性。
- 最終 RPE 総 cost、実機優位性、noise/backend robustness。
- finite RTE が常に norm baseline より強いこと。
- 高い signal radius を新しく認証できること。
- q-numerical range または weak-value 表現そのものの新規性。
- `C_use` を厳密上界とみなすこと。
- 既存 FR-R1b の mechanism-only 結果を実用的 GO と読み替えること。

## 参考文献と監査資料

- Günther et al., “Phase Estimation with Partially Randomized Time Evolution,” *PRX Quantum* 7, 020332 (2026), <https://doi.org/10.1103/ynxb-p2xq>.
- Wan, Berta, and Campbell, “Randomized Quantum Algorithm for Statistical Phase Estimation,” <https://arxiv.org/abs/2110.12071>.
- Gu et al., “Noise-resilient phase estimation with randomized compiling,” <https://arxiv.org/abs/2208.04100>.
- Ogawa et al., “Operational formulation of weak values without probe systems,” <https://arxiv.org/abs/1912.10222>.
- Chi-Kwong Li, “q-numerical ranges of normal and convex matrices,” <https://doi.org/10.1080/03081089808818538>.
- Yi and Crosson, “Spectral analysis of product formulas for quantum simulation,” <https://doi.org/10.1038/s41534-022-00548-w>.
- Li, “Some Error Analysis for the Quantum Phase Estimation Algorithms,” <https://arxiv.org/abs/2111.10430>.
- Casares et al., “Theory and practice of Trotter product formulas for quantum chemistry” (SPRINT: Symmetry-Protected Randomized near-Integrable Trotter), <https://arxiv.org/abs/2606.30741>.
- Hu and Jin, “Nonunitary amplitude-phase analysis,” <https://arxiv.org/abs/2602.09575>.
- Li, Poon, and Sze, “Elliptical range theorems for generalized numerical ranges of quadratic operators,” author manuscript, <https://cklixx.people.wm.edu/quadra.pdf>.
- [`finite-RTE研究：定理単位の先行研究監査と理論主張の独立レビュー`](../../fr_theorem_prior_art_review_71169d8.md).
