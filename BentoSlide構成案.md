# BentoSlide構成案

## 1. 制作依頼の要約

### 資料タイトル案

**DF部分ランダム化時間発展のコスト評価に向けた研究方針と現在地**

副題を付ける場合は、次を使用する。

> Product Formula・有限Randomized Taylor Expansion・RPEを接続するための実装と近似検証

### 資料の目的

共同研究者に対して、次の3点が一続きの話として分かる資料を作成する。

1. PR+DFプロジェクト全体として、最終的に何を評価・最適化したいのか。
2. そのために、どの変数を選び、どの量を式から計算し、総コストへ接続するのか。
3. 現在はどこまで実装・検証できており、次に何を進めるのか。

この資料は意思決定を求める会議資料ではなく、主に「現在はこの考え方で研究を進めている」
ことを共有する進捗説明資料とする。最後に確認事項を並べるのではなく、今後の実装・検証の
順序を明確に示す。

### 想定聴衆

- 量子位相推定、Product Formula、Double Factorizationの基本は知っている共同研究者。
- 決定論的PFについて、Qiskitで回路を基底状態へ作用させ、摂動的にエネルギー誤差を
  評価できることは共有済み。
- 今回採用した摂動評価式の由来や論文の解説は不要。

### 全体のトーン

- 研究発表らしい簡潔で落ち着いたデザイン。
- 「理論として定義済み」「local検証済み」「実装のみ」「未実装」「最終結論ではない」を
  混同しない。
- 結論を先に置き、式はその結論を支える最小限のものをスライド内へ入れる。
- 補足スライドは原則作らない。細かいgrid、全artifact名、全acceptance条件は本文に
  詰め込まず、必要なら発表者ノートまたは参照元として保持する。

### Cの説明に関する重要な指示

- Cの検証は、現在採用しているfull-$H$基底状態に対する摂動評価法を用いた検証結果へ
  完全に置き換える。
- スライド上では、この式を「採用した摂動評価法」または
  「full-$H$基底状態を用いる摂動評価」と呼ぶ。
- `D6`という名称、式の導出、論文の紹介、以前使用した別の摂動式との比較は説明しない。
- Cの検証で示すべき主張は、摂動評価から得た$C_{\mathrm{partial}}$が、直接求めた
  $C_{\mathrm{PF,eig}}$を小規模系で2%以内に再現したことである。
- survival phaseと支配分枝の細かな診断や、H3の別の2.076%条件は本資料では扱わない。
  今回のスライドで説明する検証対象はC推定法の妥当性である。

### 用語上の指示

- `finite RTE`または「有限RTE」を初めて使うスライドでは、
  **Randomized Taylor Expansionを有限Taylor cutoff $K_m$で打ち切って実装するもの**
  と明示する。
- `RTE`、`attenuation`、`compiled cost`などを無説明のまま初出させない。
- `C_{\mathrm{use}}`は厳密上界ではなく、実際に評価した$\delta$窓内の経験的上包絡とする。
- QPE/RPEの統計誤差はPFの係数$C$へ混ぜず、別の誤差項として示す。
- 単一の主固有位相が得られることはEvaluation側と同じ前提として置き、その説明だけで
  1枚のスライドを使わない。

### 推奨表示仕様

- 16:9、1280×720を前提とする。
- 想定枚数は26枚程度。枚数を減らす場合でも、変数導出、実装状況、C検証、有限RTE検証、
  今後の計画は削らない。
- 数式は編集可能なnative要素として作る。
- フロー図、依存関係図、比較表は編集可能なHTML/CSS/SVG図形で作る。
- 実験結果を装飾目的で画像化しない。数値表またはsource-derivedなnative chartにする。
- 色は、決定論側を濃紺、ランダム側を橙、誤差・検証を青緑、未実装を灰色で統一するとよい。

## 2. 資料全体のストーリー

資料全体は次の順序で構成する。

1. **研究全体の目的**：部分ランダム化が有利になる条件を、最終的にはcompiled期待総コストで決める。
2. **先行研究との差**：先行研究の解析的コスト式を、有限・離散・compiled回路のモデルへ拡張する。
3. **評価モデル**：$L_D$、$\delta_{\mathrm{time}}$、$r_m$、$K_m$、誤差・失敗確率配分が何を決めるかを式で示す。
4. **最適化の流れ**：外側候補ごとに内側最適化を行うことを、架空の数値例で具体化する。
5. **現在の実装**：実装済みの回路・検証・cost providerと、まだ接続していない部分を示す。
6. **現在の検証**：最終コストの前に、Cの推定と有限RTE近似が使えるかを独立に確かめたことを示す。
7. **次の段階**：random event回路のcompiled期待コスト、誤差・失敗確率配分、長round接続を順に進める。

中心メッセージは次の一文とする。

> 解析的には有望な部分ランダム化を、実装可能な有限回路とRPEの必要shot数まで接続し、
> 最終的に分割・時間刻み・RTE条件をcompiled総コストで比較できる形へ進めている。

## 3. セクション構成

| Section | スライド | 役割 |
|---|---:|---|
| A. 研究目的と先行研究との差 | 1--4 | 研究全体の狙いと新規性を共有する |
| B. PR+DF評価モデル | 5--8 | Hamiltonian分割、時間発展、最終コストを定義する |
| C. 変数の導出と依存関係 | 9--14 | 各変数が何から計算されるかを丁寧に示す |
| D. 最適化の具体例 | 15--17 | 入れ子型最適化を架空の数値で直感化する |
| E. 現在の実装 | 18--19 | 実装構造と現在地を示す |
| F. 近似入力の検証 | 20--24 | Cと有限RTEについて、何を調べ、何が分かったかを示す |
| G. 現在の結論と次の作業 | 25--26 | 主張できる範囲と今後の順序を示す |

---

## 4. スライド別の詳細構成

### Slide 1. タイトル

**タイトル**

> DF部分ランダム化時間発展のコスト評価に向けた研究方針と現在地

**副題**

> Product Formula・有限RTE・RPEを接続するための実装と近似検証

**入れる内容**

- 発表者名、日付。
- 右下などに小さく「研究方針・実装状況・近似検証の共有」と入れる。

**見せ方**

- $H=H_D+H_R$を中心に、`PF → finite RTE → RPE → compiled total cost`へつながる
  細い一本道を背景に置く。
- タイトルスライドでは数値や詳細な回路図を出さない。

**このスライドで言うこと**

- 最終コスト結果の報告ではなく、そこへ向けた研究設計と現在地の共有である。

---

### Slide 2. 研究全体の目的と現在の焦点

**タイトル**

> 最終目標は「部分ランダム化を総コストで選ぶ」こと

**主張**

部分ランダム化の良し悪しを、PF誤差やランダムtailの大きさだけでなく、必要shot数と
compiled回路コストを含めて判断する。

**必須内容**

最終目的関数を中央に大きく示す。

$$
G_{\mathrm{total}}
=
\sum_{m=0}^{M}\sum_{b\in\{c,s\}}
N_{m,b}\,
\mathbb E\!\left[C_{m,b}^{\mathrm{no\text{-}prep}}\right]
$$

- $N_{m,b}$：RPE round $m$、cosine/sine軸$b$の必要量子shot数。
- $\mathbb E[C_{m,b}^{\mathrm{no\text{-}prep}}]$：状態準備を除いた1 interrogationの
  期待compiled cost。
- 比較したいもの：分割$L_D$、時間刻み$\delta_{\mathrm{time}}$、有限RTE条件、
  誤差・失敗確率配分。

**現在の焦点を下段で示す**

`最終総コストは未評価 → 現在は、その入力となる近似と実装経路を検証中`

**見せ方**

- 左に「最終目標」、右に「現在地」の二段階ロードマップ。
- この段階ではGPUや特定の分子サイズの話を入れない。

**参照元**

- `docs/research/研究概要・現状.md` 3節
- `docs/research/研究目的・研究課題.md` 1節

---

### Slide 3. 先行研究が与えた解析的コストモデル

**タイトル**

> 先行研究：決定論回路・ランダム回転・信号減衰を解析式で統合

**主張**

先行研究は、部分ランダム化時間発展を単一ancilla位相推定へ接続し、決定論部分、
ランダム部分、attenuationによるshot増加を一つの解析式で評価した。

**必須内容**

二次PFに対する代表的な解析式を示す。

$$
G_{\mathrm{PR}}
=
\sum_{m=0}^{M}2N_m
\left[
G_{\mathrm{det}}N_{\mathrm{stage}}L_D2^{m-1}
+
G_{\mathrm{rand}}\kappa\lambda_R^2\delta^2 2^{2m}
\right]
$$

$$
N_m=e^{2/\kappa}[11+4(M-m)]
$$

**式の横に短く説明する**

- 決定論側：$L_D$とround長に応じて増える。
- ランダム側：$\lambda_R^2$とround長に応じて増える。
- $\kappa$：回路長とattenuationによるshot増加のtrade-offを調整する。

**重要な位置付け**

- この式はscalingとscreeningに有用。
- random eventごとに完全回路をcompileした期待値ではない。
- 先行研究が$\delta$を無視したのではなく、解析的RMSE配分から$\delta$を選んでいる。

**見せ方**

- 式を中央、その下に「deterministic」「random」「shot amplification」の3色分解。
- 論文の長い背景説明ではなく、何が解析的に得られているかに集中する。

**参照元**

- `docs/research/先行研究と未解決点.md` 3.2--3.5節

---

### Slide 4. 本研究で追加するもの

**タイトル**

> 本研究：解析的resource modelを有限・離散・compiled評価へ拡張

**主張**

新規性は先行研究の式を否定することではなく、実装可能な有限回路で必要になる要素を
追加し、最終的なexpected compiled costへ接続することである。

**比較表**

| 項目 | 先行研究の主評価 | 本研究で追加する評価 |
|---|---|---|
| RTE | 無限Taylor次数 | 有限cutoff $K_m$とTaylor残差 |
| RTE step | 共通$\kappa$から解析的に設定 | roundごとの整数$r_m,K_m$を比較 |
| 回路cost | 平均primitive cost×回転数 | event分布上のcompiled回路cost期待値 |
| shot数 | 解析的な固定schedule | attenuation・位相予算・$\alpha_{m,b}$から導出 |
| 誤差 | PFとRPEのRMSE配分 | PF・有限RTE・統計位相誤差を明示的に配分 |
| $\delta$ | 連続的な解析選択 | RPE round遷移を含む離散候補比較 |

**中央メッセージ**

$$
\text{解析的resource model}
\longrightarrow
\text{有限・離散・compiled期待総コストmodel}
$$

**見せ方**

- 左右比較またはbefore/afterのbento grid。
- 「何が違うか」が一目で分かることを優先する。

**参照元**

- `docs/research/先行研究と未解決点.md` 4節、5節

---

### Slide 5. DF Hamiltonianの部分ランダム化

**タイトル**

> 固定したDF fragment列を$L_D$で二分する

**主張**

$L_D$を増やすとランダムtailは軽くなるが、決定論回路が重くなるため、$\lambda_R$だけを
最小化しても総コスト最適にはならない。

**必須式**

$$
H=H_D(L_D)+H_R(L_D)
$$

$$
H_D=H_{\mathrm{1body}}+\sum_{\ell=1}^{L_D}H_\ell,
\qquad
H_R=\sum_{\ell>L_D}H_\ell
$$

$$
\lambda_R=\sum_j|h_j^{(L_D)}|
$$

**図**

- rank順に並んだDF fragment列を横棒で描く。
- 左側の$L_D$個を濃紺の`deterministic`、右側を橙の`random tail`にする。
- $L_D$を右へ動かしたとき、`deterministic cost ↑`、`random burden ↓`を矢印で示す。

**下段の結論**

$$
\lambda_R\text{が小さい}
\ \not\Rightarrow\
G_{\mathrm{total}}\text{が小さい}
$$

**参照元**

- `docs/research/研究概要・現状.md` 2節
- `docs/research/研究方法・解析手順.md` 3.1節

---

### Slide 6. partial-$S_2$と有限RTEの役割

**タイトル**

> 外側PFと、中央のランダムtail実装を分けて評価する

**主張**

PFの分割誤差と、$e^{-iH_R\delta}$を有限回路で実装する誤差を混ぜずに評価する。

**必須式**

$$
U_{\mathrm{partial}}(\delta)
=S_D^{\mathrm{rev}}(\delta/2)
e^{-iH_R\delta}
S_D(\delta/2)
$$

**有限RTEの初出定義**

> 有限RTE：Randomized Taylor Expansionを有限Taylor cutoff $K_m$で打ち切り、
> event unitaryを確率的に実装する方法。

**三層の区別**

1. `outer PF`：$H_D$のsymmetric sweepと厳密な$e^{-iH_R\delta}$の分割誤差。
2. `finite RTE`：中央の$e^{-iH_R\delta}$を有限event回路で近似する誤差とattenuation。
3. `RPE statistics`：減衰したcosine/sine信号を有限shotで推定する誤差。

**図**

- `S_D forward half | e^{-iH_Rδ} | S_D reverse half`の回路ブロック。
- 中央ブロックだけを拡大し、`exact tail（PF基準）`と`finite RTE（実装）`へ分岐。

**参照元**

- `docs/research/研究概要・現状.md` 2節
- `docs/research/先行研究と未解決点.md` 3.1節、4.1節

---

### Slide 7. 最終的に数える回路scope

**タイトル**

> 最終コストはtime-evolution blockだけではない

**主張**

最終目的は、状態準備を除いたRPE interrogation全体の1 shot costと必要shot数の積である。

**必須内容**

3段階のscopeを図示する。

| 段階 | 含むもの | 現在の扱い |
|---|---|---|
| time-evolution subcircuit | controlled partial-$S_2$ evolution | 主な直接compile対象 |
| 1 Hadamard interrogation | ancilla、axis change、測定を追加 | short roundで中間評価可能 |
| full RPE | 全round・全軸・全shot | 最終目標、未評価 |

**注意書き**

- 状態準備は研究範囲外として除外する。
- 「候補間で相殺される」とは主張しない。shot数が違えば相殺しないためである。

**図**

- 同心円または入れ子の3カード。
- 最終scopeほど右へ広がる構成。

**参照元**

- `docs/research/研究目的・研究課題.md` 2.2節、2.4節
- `docs/research/研究方法・解析手順.md` 6節

---

### Slide 8. 固定条件・探索変数・派生量

**タイトル**

> 何を固定し、何を選び、何を計算するか

**必須表**

| 区分 | 量 | 扱い |
|---|---|---|
| 固定入力 | 分子、basis、DF表現・rank・fragment順、PF次数$p$、$\epsilon_E$、compiler条件 | 現段階では固定 |
| 外側探索 | $L_D,\delta_{\mathrm{time}}$ | 候補を列挙して最後に比較 |
| round内探索 | $r_m,K_m$ | 各外側候補・各roundで整数探索 |
| 配分変数 | $\overline\beta_{\mathrm{PF},m},\overline\beta_{\mathrm{RTE},m},\overline\beta_{\mathrm{stat},m},\alpha_{m,b}$ | 現在は事前schedule、将来は比較・最適化 |
| 派生量 | $H_D,H_R,\lambda_R,M,\tau_m,\epsilon_{Z,m},A_m,N_{m,b},\mathbb E[C]$ | 上の入力・変数から式で計算 |

**主張**

派生量を探索変数のように自由に置かない。例えば$N_{m,b}$は、attenuation、信号半径、
統計位相予算、失敗確率から計算される。

**見せ方**

- `fixed / choose / derive`の3列を色分け。

**参照元**

- `docs/research/研究方法・解析手順.md` 1節

---

### Slide 9. 外側変数から何が決まるか

**タイトル**

> $L_D$はHamiltonian分割を、$\delta_{\mathrm{time}}$はRPE時間構造を決める

**左側：$L_D$から計算する量**

$$
L_D
\longrightarrow
H_D,H_R,\lambda_R,
\{p_j\},\text{deterministic circuit},C_{\mathrm{use}}
$$

- $p_j=|h_j|/\lambda_R$はrandom component確率。
- $C_{\mathrm{use}}$はその分割に対するPF energy-bias係数。

**右側：$\delta_{\mathrm{time}}$から計算する量**

$$
q_m=2^m,
\qquad
t_m=q_m\delta_{\mathrm{time}}
$$

$$
M=
\left\lceil
\log_2\frac{\beta_{\mathrm{RPE}}}
{\delta_{\mathrm{time}}\epsilon_E}
\right\rceil
$$

二次PFのenergy biasを

$$
\epsilon_{\mathrm{PF}}
=C_{\mathrm{use}}\delta_{\mathrm{time}}^2
$$

とすれば、round位相誤差は

$$
\beta_{\mathrm{PF},m}^{\mathrm{ub}}
\le t_m\epsilon_{\mathrm{PF}}
$$

となる。

**主張**

$\delta_{\mathrm{time}}$はPF誤差だけでなく、round数$M$と各roundの時間を同時に変える。

**参照元**

- `docs/research/研究方法・解析手順.md` 3節、5.1節

---

### Slide 10. $(r_m,K_m)$から有限RTE誤差を計算する

**タイトル**

> round内候補$(r_m,K_m)$は誤差とattenuationを同時に変える

**必須式**

1 short stepの無次元時間：

$$
\tau_m=\frac{\lambda_R\delta_{\mathrm{time}}}{r_m}
$$

Taylor残差からround全体の信号誤差上界：

$$
\epsilon_{Z,m}
\le
\left(1+R_{K_m}(\tau_m)\right)^{q_mr_m}-1
$$

有限normalizationとattenuation：

$$
B_{K_m}(\tau_m)
=\sum_{0\le n\le K_m,\ n:\mathrm{even}}a_n(\tau_m)
$$

$$
A_m^{\mathrm{att}}
=B_{K_m}(\tau_m)^{-q_mr_m}
$$

**必須説明**

- $r_m$を増やすと$\tau_m$は小さくなるが、short-step数は増える。
- $K_m$を増やすとTaylor残差は減るが、event回路は重くなり得る。
- $\epsilon_{Z,m}$は位相・半径を変え得る誤差、$A_m^{\mathrm{att}}$は既知の正の信号減衰。
  二つは同じ量ではない。

**図**

- $(r_m,K_m)$から`error`、`attenuation`、`event distribution`、`circuit cost`へ
  四方向に分岐する図。

**参照元**

- `docs/research/研究方法・解析手順.md` 4節

---

### Slide 11. PF・有限RTE・統計誤差を位相で配分する

**タイトル**

> 異なる誤差をRPE roundの位相ずれへ換算する

**必須制約**

$$
\overline\beta_{\mathrm{PF},m}
+\overline\beta_{\mathrm{RTE},m}
+\overline\beta_{\mathrm{stat},m}
\le\beta_{\mathrm{RPE}}
$$

各実誤差上界が配分を満たすことも別途確認する。

$$
\beta_{j,m}^{\mathrm{ub}}
\le\overline\beta_{j,m},
\qquad
j\in\{\mathrm{PF,RTE,stat}\}
$$

有限RTEの信号誤差から位相上界への換算例：

$$
\beta_{\mathrm{RTE},m}^{\mathrm{ub}}
\le
\arcsin\!\left(
\frac{\epsilon_{Z,m}}{\rho_{\star,m,\mathrm{lb}}}
\right)
$$

**暫定値は小さな注記として示す**

近似検証で使用した一例：

$$
\beta_{\mathrm{RPE}}=0.40,
\quad
(\overline\beta_{\mathrm{PF}},
\overline\beta_{\mathrm{RTE}},
\overline\beta_{\mathrm{stat}})
=(0.08,0.08,0.24)\ \mathrm{rad}
$$

これは最終最適配分ではない。

**見せ方**

- 0.40 radの横棒をPF/RTE/statの3区画に分ける。
- 「allocated budget」と「calculated upper bound」を異なる線種で示す。

**参照元**

- `docs/research/研究方法・解析手順.md` 5.1節、5.4節

---

### Slide 12. attenuationと失敗確率からshot数を導く

**タイトル**

> 回路を短くしても、信号が減衰すればshot数が増える

**必須式**

保守的な観測半径下界：

$$
\rho_{m,\mathrm{lb}}^{\mathrm{obs}}
=A_m^{\mathrm{att}}
(\rho_{\star,m,\mathrm{lb}}-\epsilon_{Z,m})
$$

統計位相予算から各座標の許容誤差：

$$
\epsilon_{\mathrm{coord},m}
=
\frac{\rho_{m,\mathrm{lb}}^{\mathrm{obs}}}{\sqrt2}
\sin\overline\beta_{\mathrm{stat},m}
$$

Hoeffding型shot上界：

$$
N_{m,b}
=
\left\lceil
\frac{2}{\epsilon_{\mathrm{coord},m,b}^{2}}
\log\frac{2}{\alpha_{m,b}}
\right\rceil
$$

総失敗確率制約：

$$
\sum_{m=0}^{M}
(\alpha_{m,c}+\alpha_{m,s})
\le\alpha_{\mathrm{tot}}
$$

**主張**

- attenuationが小さい、または失敗確率$\alpha_{m,b}$を厳しくすると、必要shot数が増える。
- random circuitの量子shotごとにfresh IID trajectoryを使う。
- expected compiled costを推定する古典Monte Carlo標本数$S_{\mathrm{MC}}$は、量子shot数
  $N_{m,b}$とは別物で、掛け合わせない。

**図**

`$(r,K)$ → attenuation/radius → coordinate tolerance → quantum shots`の縦フロー。

**参照元**

- `docs/research/研究方法・解析手順.md` 5.2--5.4節

---

### Slide 13. random event回路のexpected compiled cost

**タイトル**

> 1 shot costはrandom event分布上で平均する

**必須式**

event列$\omega$の確率を$p_m(\omega)$とすると、

$$
\mathbb E[C_{m,b}]
=\sum_\omega p_m(\omega)C_{m,b}(\omega)
$$

列挙できない場合は、古典Monte Carloで

$$
\widehat C_{m,b}
=\frac1{S_{\mathrm{MC}}}
\sum_{s=1}^{S_{\mathrm{MC}}}
C_{m,b}(\omega_s)
$$

と標準誤差を求める。

**なぜprimitive加算では不十分か**

$$
C_{\mathrm{comp}}(U_1U_2)
\neq
C_{\mathrm{comp}}(U_1)+C_{\mathrm{comp}}(U_2)
$$

- basis transformの共有
- 隣接回転のfusion/cancellation
- controlled回路化
- routingとbackend依存性

**見せ方**

- 3種類程度のevent回路カードと確率を並べ、compile後の異なるcostをweighted averageする図。
- 実データに見える架空の棒グラフは使わない。

**参照元**

- `docs/research/先行研究と未解決点.md` 4.2節
- `docs/research/研究方法・解析手順.md` 6節

---

### Slide 14. 変数依存関係の全体像

**タイトル**

> 探索変数から総コストまでの依存関係

**このスライドの役割**

単なる矢印図ではなく、各主要量がどの式から求まるかを同じ図の中に入れる。

**推奨図**

4列のsource-derived native flowchartにする。

#### 列1：外側候補

- $L_D$
- $\delta_{\mathrm{time}}$

#### 列2：Hamiltonian・RPE構造

- $H=H_D+H_R$
- $\lambda_R=\sum_j|h_j|$
- $q_m=2^m$
- $t_m=q_m\delta_{\mathrm{time}}$
- $M=\lceil\log_2[\beta_{\mathrm{RPE}}/(\delta_{\mathrm{time}}\epsilon_E)]\rceil$
- $\epsilon_{\mathrm{PF}}=C_{\mathrm{use}}\delta_{\mathrm{time}}^2$

#### 列3：round内候補と派生量

- 選ぶ量：$r_m,K_m,\overline\beta_{j,m},\alpha_{m,b}$
- $\tau_m=\lambda_R\delta_{\mathrm{time}}/r_m$
- $\epsilon_{Z,m}\le(1+R_{K_m})^{q_mr_m}-1$
- $A_m=B_{K_m}^{-q_mr_m}$
- $N_{m,b}=N(\epsilon_{\mathrm{coord},m,b},\alpha_{m,b})$

#### 列4：cost

- $\mathbb E[C_{m,b}]=\sum_\omega p_m(\omega)C_{m,b}(\omega)$
- $G_m=\sum_bN_{m,b}\mathbb E[C_{m,b}]$
- $G_{\mathrm{total}}=\sum_mG_m$

**強調する矢印**

- $L_D$は決定論回路と$\lambda_R$の両方へ効く。
- $(r_m,K_m)$は1 shot costだけでなく$N_{m,b}$にも効く。
- $\alpha_{m,b}$は全roundの総和制約でround間を結合する。

**参照元**

- `docs/research/研究方法・解析手順.md` 2節
- `docs/research/研究方法・解析手順_補足QA.md` Step 10

---

### Slide 15. 入れ子型最適化

**タイトル**

> 外側候補を仮固定し、内側最適化後の総コストを比較する

**必須式**

$$
G_{\mathrm{opt}}(L_D,\delta_{\mathrm{time}})
=
\min_{\{r_m,K_m,\overline\beta_{j,m},\alpha_{m,b}\}}
G_{\mathrm{total}}
$$

$$
(L_D^*,\delta_{\mathrm{time}}^*)
=\arg\min_{L_D,\delta_{\mathrm{time}}}
G_{\mathrm{opt}}(L_D,\delta_{\mathrm{time}})
$$

**手順**

1. $(L_D,\delta_{\mathrm{time}})$を1候補だけ仮固定。
2. 全roundで誤差・確率制約を満たす$(r_m,K_m)$を列挙。
3. 1 shot expected costと必要shot数を掛け、round・軸で合計。
4. 配分scheduleも比較して内側最小値を得る。
5. 外側候補を変えて繰り返す。

**誤解防止**

- $L_D$と$\delta_{\mathrm{time}}$を先に最終決定するわけではない。
- 現行実装では配分自体の完全最適化と外側探索は未統合。

**図**

- 外側loopと、round別内側loopの二重ループ図。

**参照元**

- `docs/research/研究方法・解析手順.md` 7節
- `docs/research/研究方法・解析手順_補足QA.md` Q1

---

### Slide 16. 架空例：round内で$(r_m,K_m)$を選ぶ

**タイトル**

> 架空例：最小shotではなく、shot×1回路costで選ぶ

**冒頭に必ず表示**

> 以下は変数依存を説明するための架空の無次元値であり、実験結果ではない。

**前提**

- $L_D=3$、$\delta_{\mathrm{time}}=0.10$、$\lambda_R=1.2$
- round $m=2$、$q_2=4$
- $\overline\beta_{\mathrm{RTE},2}=0.020$

**候補表**

| $(r_2,K_2)$ | RTE位相上界 | 判定 | $N_{2,c}=N_{2,s}$ | 仮の$(\widehat C_c,\widehat C_s)$ | $G_2$ |
|---|---:|:---:|---:|---:|---:|
| $(1,0)$ | 0.030331 | 不可 | — | — | — |
| $(2,0)$ | 0.014788 | 可 | 157 | (64, 66) | 20,410 |
| $(1,2)$ | $3.54\times10^{-5}$ | 可 | 166 | (54, 56) | **18,260** |
| $(3,0)$ | 0.009773 | 可 | 154 | (80, 84) | 25,256 |

**結論**

$$
(r_2^*,K_2^*)=(1,2)
$$

- $(3,0)$はshot数が最少でも回路が重い。
- $(1,2)$はshot数がやや多いが、1回路costが小さくround総コストが最小。

**見せ方**

- 表の右端$G_2$を強調する。
- 量子shotと古典Monte Carlo標本を同じものとして描かない。

**参照元**

- `docs/research/研究方法・解析手順_補足QA.md` Step 0--7

---

### Slide 17. 架空例：外側候補を比較する

**タイトル**

> 架空例：$\lambda_R$最小や最小$\delta$が自動的な最適解ではない

**冒頭注記**

> 全roundの内側最適化が完了したと仮定した架空値。

**比較表**

| $L_D$ | $\delta_{\mathrm{time}}$ | $\lambda_R$ | $M$ | $G_{\mathrm{opt}}$ |
|---:|---:|---:|---:|---:|
| 2 | 0.10 | 2.0 | 3 | 64,800 |
| 3 | 0.10 | 1.2 | 3 | **59,360** |
| 4 | 0.10 | 0.5 | 3 | 61,200 |
| 3 | 0.05 | 1.2 | 4 | 60,500 |

**結論**

$$
(L_D^*,\delta_{\mathrm{time}}^*)=(3,0.10)
$$

**説明**

- $L_D=4$は$\lambda_R$が最小だが、決定論回路増加まで含めると最小costではない。
- $\delta=0.05$はPF誤差を抑える一方、round数$M$が増えて総コストが上がる。
- このような比較を実データで行うことが最終目標。

**参照元**

- `docs/research/研究方法・解析手順_補足QA.md` Step 8--9

---

### Slide 18. 現在の実装構造

**タイトル**

> 物理モデルから回路・検証・cost artifactまでを段階的に実装

**主張**

最終探索を一つの巨大な処理として作るのではなく、物理近似、回路生成、cost評価、
resource accountingを交換可能な層として実装している。

**推奨アーキテクチャ図**

1. `Chemistry / DF layer`
   - H-chain生成、sector、DF fragment、rank・threshold。
2. `Partition / PF layer`
   - $L_D$による$H_D/H_R$分割、partial-$S_2$ step、controlled evolution。
3. `finite-RTE layer`
   - finite order分布、random event、exact列挙／古典Monte Carlo。
4. `Signal / validation layer`
   - PF係数、演算子・信号・位相・半径・attenuation検証。
5. `Compiled-cost layer`
   - event回路構築、transpile、scope別metric、標準誤差。
6. `Accounting / artifacts`
   - roundごとのshot数・cost集計、schema、fingerprint、provenance。

**実装上の具体例**

- Qiskit回路で決定論half stepを作用。
- 小規模基準ではsector内の行列指数・固有値計算を使用。
- PF演算子を構築しない経路では`expm_multiply`で$H_R$を状態へ直接作用。
- 同じHamiltonian、eigensystem、symbolic tailをbatch内で再利用して高速化。
- 成果物はsource hashとSHA-256 fingerprintを持つJSONとして保存。

**見せ方**

- 6層の縦パイプライン。各層に代表ファイル名を小さく添えてもよい。

**参照元**

- `docs/research/数値実験・評価計画.md` 6節
- `docs/pf_c_system_size_validation.md`
- `docs/finite_rte_signal_validation.md`

---

### Slide 19. 現在の実装段階

**タイトル**

> 現在は「最終コスト前の入力検証」まで進んでいる

**必須status表**

| 項目 | 状態 | 現在の位置付け |
|---|---|---|
| partial-$S_2$ one-step／短い反復回路 | 実装済み | 回路生成・小規模検証に使用 |
| finite-RTE event回路・分布 | 実装済み | exact/MC評価が可能 |
| PF係数$C$の小規模・state-action検証 | local検証済み | 今回のC入力として採用 |
| finite-RTE信号・上界検証 | H4でlocal検証済み | 近似式の妥当性確認 |
| short-round accounting | $q_m\le4$で実装 | 小規模なround比較に使用 |
| medium-$q$ benchmark / affine proxy | fit・holdout実装済み | accountingとは未接続 |
| random eventの最終scope期待cost | 部分実装・要検証 | 次の主要課題 |
| 誤差・失敗確率配分の最適化 | 未実装 | 当面は複数schedule比較 |
| $(L_D,\delta)$外側探索 | 未統合 | 上記入力を固めた後に実装 |
| full RPE総コスト | 未評価 | 最終段階 |

**ロードマップ表示**

`回路実装 → 近似検証［現在］ → expected cost検証 → 配分・長round接続 → 外側探索 → 最終総コスト`

**重要な注意**

実装済みの回路やproxyが存在しても、全round総コストまで接続・検証済みとは言わない。

**参照元**

- `docs/research/数値実験・評価計画.md` 6節
- `docs/research/研究概要・現状.md` 3節、8節

---

### Slide 20. なぜ最終コストの前に近似を検証するか

**タイトル**

> 総コストへ入れる前に、各近似を独立に検証する

**主張**

最終コストが得られても、入力近似のどこが正しいか分からなければ結果を解釈できない。
現在はコスト計算を急ぐより、次の入力を順番に検証している。

**3段階カード**

1. **PF係数$C$** — 検証済み
   - 安価なscreening係数と、候補で直接使う係数の役割を分ける。
   - 大規模系向けの摂動評価が小規模基準を再現するか確認。
2. **有限RTE誤差・attenuation** — 小規模検証済み
   - Taylor残差上界が実演算子・信号・位相誤差を覆うか確認。
   - $(r_m,K_m)$で予算内へ戻せるか確認。
3. **random回路cost・配分** — 次に検証
   - expected compiled costをexact列挙／Monte Carloで安定に推定できるか。
   - $\beta$・$\alpha$配分がshot数と最適候補をどう変えるか。

**見せ方**

- `validated / validated locally / next`の進捗付き3カード。

---

### Slide 21. Cの検証：何を確かめたかったか

**タイトル**

> Cの検証：大規模系向け摂動評価でPF係数を求められるか

**このスライドで答える問い**

> PF演算子全体を構築・対角化しなくても、full-$H$基底状態へpartial-$S_2$を作用させる
> 摂動評価から、コスト評価に使う$C$を求められるか。

**比較した二つの量**

1. **小規模基準** $C_{\mathrm{PF,eig}}$
   - partial-$S_2$演算子をsector内で直接構築。
   - full-$H$基底状態に対応する主固有位相のenergy biasから$C$を求める。
2. **scalable推定** $C_{\mathrm{partial}}$
   - full-$H$基底状態へ同じpartial-$S_2$をstate-actionで作用。
   - 採用した摂動評価法からenergy biasと$C$を求める。

**検証時の演算モデル**

$$
U_{\mathrm{partial}}(\delta)
=S_D^{\mathrm{rev}}(\delta/2)
e^{-iH_R\delta}
S_D(\delta/2)
$$

- 中央の$e^{-iH_R\delta}$は直接作用し、有限RTE誤差を入れない。
- したがって、ここで調べるのは外側PFの$C$であり、有限RTE検証とは分離されている。

**検証scope**

- H4、DF rank 12、全$L_D=0,\ldots,11$。
- H2--H5では直接固有位相との系サイズ比較。
- H2--H5でstate-actionとdense参照を照合し、H6までstate-actionを拡張。
- H-chain、原子間距離1.0 Å、STO-3G。
- 事前の一致基準は相対2%。

**明示的に入れない内容**

- 採用した摂動式そのもの。
- D6という名称や論文の説明。
- 以前の摂動式との比較。
- 固有位相の複数分枝に関する長い説明。

**図**

- 同じ$U_{\mathrm{partial}}$から、左の`direct diagonalization`と右の`state action + perturbation`
  に分岐し、最後に$C$を比較する図。

**参照元**

- `docs/pf_delta_validation.md`
- `docs/pf_c_system_size_validation.md`
- `artifacts/pf_delta_validation/*_v5.json`
- `artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json`

---

### Slide 22. Cの検証結果

**タイトル**

> 摂動評価のCは、小規模基準を2%以内で再現した

**上段：H2--H5の系サイズ比較**

| 系 | qubit | DF rank | $L_D$ | $C_{\mathrm{PF,eig}}$ | $C_{\mathrm{partial}}$ | 相対差 |
|---:|---:|---:|---:|---:|---:|---:|
| H2 | 4 | 3 | 1 | 0.00328942 | 0.00331879 | 0.893% |
| H3 | 6 | 5 | 2 | 0.00498346 | 0.00504048 | 1.144% |
| H4 | 8 | 7 | 3 | 0.01340628 | 0.01349787 | 0.683% |
| H5 | 10 | 9 | 4 | 0.01187331 | 0.01192681 | 0.451% |

全4系で事前の2%基準を通過。

**下段：追加結果を3カードで示す**

1. **H4全分割**
   - DF rank 12、$L_D=0,\ldots,11$。
   - $C_{\mathrm{partial}}$と、同じ有効点で求めた$C_{\mathrm{PF,eig}}$のfit差は最大0.288%。
   - 全12分割が2%条件を通過。
2. **state-action実装**
   - H2--H5でdense参照との差は最大$7.94\times10^{-15}$。
   - PF演算子を構築しない経路が数値的に一致。
3. **H6**
   - 12 qubit、DF rank 11、$L_D=5$、$\delta=0.250$--0.256。
   - 実行窓内の経験的上包絡$C_{\mathrm{use}}=0.02086663$。
   - H6ではPF演算子を対角化していない。

**この結果から採用する方針**

- $C_D$：広い$L_D$を安価に絞るscreening専用。
- $C_{\mathrm{partial}}$：候補ごとにstate-actionで再計算し、コスト評価へ使う。
- $C_{\mathrm{use}}$：実行した$\delta$窓内のpoint coefficient最大値。厳密上界ではない。

**言わないこと**

- H4/H6から大規模系の$C$を外挿できるとは言わない。
- 最終コストまで検証済みとは言わない。

**見せ方**

- H2--H5はpaired dot plotにしてもよいが、値が読み取れる表を必ず残す。
- `within 2%`を中央の結論として強調する。

**参照元**

- `docs/pf_c_system_size_validation.md` 結果節
- `docs/research/研究概要・現状.md` 6節

---

### Slide 23. 有限RTE検証：何を確かめたかったか

**タイトル**

> 有限RTE検証：解析上界を実際の信号評価に使えるか

**このスライドで答える問い**

> finite Taylor残差から計算する$\epsilon_{Z,m}$、attenuation、位相上界が、
> 実際の有限RTE平均演算子と複素信号の誤差を正しく覆うか。

**基準と比較対象**

- 基準：中央に直接$e^{-iH_R\delta}$を作用したpartial-$S_2$。
- 比較：finite cutoff $K_m$とshort-step数$r_m$から作る有限RTEの期待演算子。

**検証した4項目**

1. normalization補正後の演算子誤差がTaylor残差上界以下か。
2. full-$H$基底状態などの複素信号誤差が同じ上界以下か。
3. attenuationと保守的信号半径下界が整合するか。
4. 信号誤差から換算した位相上界が実位相誤差を覆うか。

**何を知りたい検証か**

- 大規模系の最適化で、毎回random trajectoryを全列挙せず解析上界を使えるか。
- ある$(r_m,K_m)$が誤差配分を超えたとき、$r_m$または$K_m$を変えて実行可能に戻せるか。
- ただし、どの候補が最小costかはこの検証だけでは決めない。

**代表scope**

- H4 linear chain、距離1.0 Å、STO-3G、DF rank 12。
- 分割感度：$L_D=0,1,3,6$。
- 時間刻み感度：$L_D=3$で複数$\delta,q,r,K$を評価。

**図**

- `exact tail partial-S2`と`finite-RTE expected operator`を並べ、
  operator → signal → radius → phaseの4段階で比較する図。

**参照元**

- `docs/finite_rte_signal_validation.md`
- `artifacts/finite_rte_signal_validation/`

---

### Slide 24. 有限RTE検証結果

**タイトル**

> 解析上界は検証gridの実誤差を覆い、$(r,K)$変更で予算内へ戻せた

**上段：分割感度**

各$L_D$で48条件を評価し、演算子・信号・半径・適用可能な位相上界はすべて成立。

| $L_D$ | $\lambda_R$ | 最小attenuation | 暫定RTE位相予算を満たす点 |
|---:|---:|---:|---:|
| 0 | 13.7237 | 0.01421 | 32 / 48 |
| 1 | 2.86296 | 0.73062 | 45 / 48 |
| 3 | 0.583270 | 0.98651 | 48 / 48 |
| 6 | 0.0220789 | 0.999981 | 48 / 48 |

**下段：$L_D=3$の$\delta$感度**

- 375条件中374条件が暫定$\overline\beta_{\mathrm{RTE}}=0.08$ rad内。
- 予算超過は$\delta=0.4,q=4,r=1,K=0$の1条件のみ。
- 同じ$\delta,q$でも、$r=2,K=0$なら上界0.0581 rad、
  $r=1,K=2$なら$5.18\times10^{-4}$ radとなり予算内へ戻る。

**この結果の意味**

- 検証範囲では、有限Taylor残差から作った上界をcandidate rejectionに使える。
- $K:0\to2$の効果が大きい例が確認できた。
- ただし$r=2,K=0$と$r=1,K=2$のどちらが低costかは、compiled cost評価前には決めない。

**注意**

- 小規模H4のlocal検証であり、全系・全roundの保証ではない。
- 最終コスト評価は行っていない。

**見せ方**

- 左に分割表、右に375条件の`374 pass / 1 fail → parameter changeでpass`という図。

**参照元**

- `docs/finite_rte_signal_validation.md` 結果節

---

### Slide 25. 現時点で言えること／まだ言えないこと

**タイトル**

> 近似入力の実装経路は確認できたが、総コスト結論はまだ先

**左：現時点で言えること**

- $C_D$をscreeningに使い、候補では$C_{\mathrm{partial}}$を再計算する二段階方針を採用できる。
- 採用した摂動評価による$C$は、小規模基準を2%以内で再現した。
- PF演算子を構築しないstate-action経路をH6まで実行できた。
- H4の指定gridでは、有限RTEの解析上界が実演算子・信号・位相誤差を覆った。
- $(r_m,K_m)$変更により有限RTE誤差予算を満たす候補を作れる。

**右：まだ言えないこと**

- $C_{\mathrm{use}}$が全$\delta$に対する厳密上界であること。
- 有限RTE上界が全分子・全サイズ・全roundで十分tightであること。
- random eventのexpected compiled costが最終scopeで検証済みであること。
- 誤差・失敗確率配分が最適化済みであること。
- 部分ランダム化が決定論PFより低いfull RPE総コストになること。

**下段メッセージ**

> 現在得たのは「最終コストへ入れる近似を使える見込み」であり、
> 「最終的な優位性の結論」ではない。

**見せ方**

- 緑系の`established locally`と灰色の`not yet`の二列。

---

### Slide 26. 次に進める実装・検証

**タイトル**

> 次はrandom回路costとshot数を接続し、外側探索へ近づける

**優先順を時系列で示す**

1. **Cを入力として固定**
   - 今回検証した摂動評価を候補ごとの$C_{\mathrm{partial}}$計算に使用する。
   - より大きい系では、screeningで残した$L_D$候補だけをstate-action評価する。
2. **有限RTE検証を拡張**
   - 系サイズ、$q_m$、$\delta$範囲を増やし、上界のtightnessとparameter感度を確認する。
3. **random eventのexpected compiled costを検証**
   - 列挙可能な小規模event空間でexact期待値を作る。
   - Monte Carlo推定値・標準誤差・停止条件をexact値と比較する。
   - time-evolution scopeからHadamard interrogation scopeへの差を確認する。
4. **誤差・失敗確率scheduleを比較**
   - まず等配分と複数の非一様scheduleを比較する。
   - $\beta$・$\alpha$の変更が$(r_m,K_m)$、attenuation、$N_{m,b}$、$G_m$へ与える影響を調べる。
5. **medium/long round costをaccountingへ接続**
   - short-round直接評価とholdout検証済みproxyの適用範囲を明確化する。
6. **最後に外側探索と総コスト比較**
   - 各$(L_D,\delta_{\mathrm{time}})$で内側最小値を求める。
   - 同一Hamiltonian、精度、成功確率、制御方式、compiler条件で決定論PFと比較する。

**強調する直近課題**

> 直近の中心は「random回路の期待compiled cost」と「誤差・失敗確率配分がshot数へ与える影響」の検証。

**締めの一文**

> 近似を一つずつ検証しながら、部分ランダム化を総コストで比較できる評価系へ段階的に接続する。

**参照元**

- `docs/research/研究概要・現状.md` 8節
- `docs/research/数値実験・評価計画.md` 6節

---

## 5. 全体のvisual plan

BentoSlide側では、次のvisualを優先してnative要素で作成する。

| Slide | visual種別 | 内容 | 注意 |
|---:|---|---|---|
| 1 | native flow | PF → finite RTE → RPE → total cost | 装飾的にしすぎない |
| 2 | roadmap | 最終目標と現在地 | H12/GPUを出さない |
| 3 | equation decomposition | 先行研究cost式のdet/random/shot分解 | 式を省略しない |
| 4 | comparison grid | 先行研究と本研究の差 | before/afterを明確に |
| 5 | partition diagram | DF fragment列と$L_D$ | det=濃紺、random=橙 |
| 6 | circuit block | partial-$S_2$とexact/finite tail | finite RTEを初出定義 |
| 7 | nested scope | evolution / interrogation / full RPE | 状態準備除外を明示 |
| 8 | classification table | fixed / choose / derive | 色分けを統一 |
| 9 | two-column derivation | $L_D$依存と$\delta$依存 | 各式を近くに置く |
| 10 | dependency fan-out | $(r,K)$から4つの影響 | errorとattenuationを混ぜない |
| 11 | phase budget bar | PF/RTE/stat配分 | 暫定値と最終値を区別 |
| 12 | vertical flow | attenuationからshot数 | MC標本との違いを注記 |
| 13 | weighted circuits | event別compile costの期待値 | 架空データplotを作らない |
| 14 | full dependency graph | 変数と導出式の全体像 | 本資料の中心図 |
| 15 | nested loops | outer/inner optimization | 外側を先に確定しない |
| 16 | decision table | 架空のround内比較 | $G_2$最小を強調 |
| 17 | outer comparison | 架空の外側比較 | fictitious表示を大きく |
| 18 | architecture | 実装の6層 | local file参照を添える |
| 19 | status matrix | 実装済み／検証済み／未実装 | final cost未評価を明示 |
| 20 | three validation cards | C / finite RTE / cost・allocation | 進捗を色分け |
| 21 | two-path comparison | direct基準とstate-action摂動 | D6の式・論文説明を出さない |
| 22 | result table/cards | C検証結果 | 2%以内を主メッセージに |
| 23 | validation pipeline | operator→signal→radius→phase | 何を調べるかを中心に |
| 24 | pass/fail recovery | 374/375と$(r,K)$変更 | cost結論ではないと注記 |
| 25 | two-column boundary | 言える／言えない | local evidenceを明示 |
| 26 | phased roadmap | 次の6段階 | 直近2課題を強調 |

生成画像は不要。研究上の数値、式、回路構造および結果をAI生成画像で表現しない。

## 6. source利用方針

BentoSlide側の資料作成AIは、数値や式を新たに推測せず、次の順にリポジトリを確認する。

1. `docs/research/研究概要・現状.md`
2. `docs/research/研究目的・研究課題.md`
3. `docs/research/先行研究と未解決点.md`
4. `docs/research/研究方法・解析手順.md`
5. `docs/research/研究方法・解析手順_補足QA.md`
6. `docs/research/数値実験・評価計画.md`
7. `docs/pf_c_system_size_validation.md`
8. `docs/pf_delta_validation.md`
9. `docs/finite_rte_signal_validation.md`
10. `VALIDATION_STATUS.md`と`artifacts/validation_manifest.json`

数値を採用する場合は、対応するmachine-readable artifactも確認する。

- CのH4全$L_D$検証：`artifacts/pf_delta_validation/*_v5.json`
- CのH2--H6検証：`artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json`
- 有限RTE検証：`artifacts/finite_rte_signal_validation/`

旧schemaのartifactが同じdirectoryに残っている場合は、現行文書が指す上記成果物を使う。
staleな旧DF screening結果やmachine-readable evidenceのないUWC表は、現行の科学的結果として
資料へ載せない。

## 7. 最終チェック項目

HTML生成前に次を確認する。

- [ ] Slide 2はPR+DF研究全体の目的を示し、GPUやH12の話から始めていない。
- [ ] finite RTEを初出時に定義している。
- [ ] 先行研究の解析的コスト式と、本研究の有限・compiled評価との差を具体的に説明している。
- [ ] 変数依存関係だけでなく、各主要量の導出式をSlide 9--14に入れている。
- [ ] 架空の最適化例を実験結果と誤認しない表示にしている。
- [ ] 現在の実装方法と実装段階を別々に説明している。
- [ ] C検証は現在採用した摂動評価結果へ置き換わっている。
- [ ] C検証でD6の式、D6論文、以前の摂動式を説明していない。
- [ ] C検証の目的が「scalableな摂動評価で基準Cを再現できるか」と明示されている。
- [ ] 有限RTE検証で、検証条件の列挙より「何を確かめたいか」を詳しく説明している。
- [ ] Cと有限RTEの検証を、最終総コスト評価と混同していない。
- [ ] $C_D$を厳密上界または最終Cとして扱っていない。
- [ ] $C_{\mathrm{use}}$を実行窓外まで保証する厳密上界と呼んでいない。
- [ ] full RPE総コストや部分ランダム化の優位性を確定結果として示していない。
- [ ] 最後が質問一覧ではなく、次に行う実装・検証のロードマップになっている。
- [ ] 本編だけで説明が完結し、補足スライドへ重要式を追い出していない。
