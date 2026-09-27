# 有限RTEの信号位相誤差：研究主張・証明義務・完成原稿契約

- 固定日: 2026-09-27 (JST)
- 基点: `16d4482da6f445877c68c0d613f1d9ccdcf2f26b`
- 状態: `DRAFT_CLAIM_INTEGRATION_NOVELTY_AND_PROOF_AUDIT_OPEN`
- 数値判断: FR-R1b の `MECHANISM_ONLY_NO_PRACTICAL_GO` を維持する
- 計算方針: 本契約の監査が終わるまで、新しい数値計算を開始しない

## 0. この文書の役割

本書は、FR-R1b 後の研究を「候補探索」から「一つの成果の完成」へ切り替えるための正本である。既存の数値結果を増やすことではなく、有限 RTE が作る平均信号について、何が既知で、何を新しく証明し、どの証拠で支えるかを固定する。

FR-R1b では、位相方向の改善機構は確認されたが、固定予算で一方向に資源優位となる witness は得られなかった。この停止判断を変更せず、FR-R2、H4/H12、長時間 RPE、compiler 総 cost、大規模 grid へは自動的に進まない。

## 1. 研究課題と候補主張

### 1.1 主研究課題

> finite paired-Taylor RTE の平均信号に含まれる位相不変な正の scalar を物理的な減衰と分離したとき、利用可能な演算子・状態情報の層ごとに、位相誤差と信号半径をどこまで保証でき、どこから先は保証できないか。

### 1.2 候補 contribution

> 有限 RTE の平均信号を正 scalar と方向依存補正へ分解し、同じ入力情報だけを用いる norm-disk baseline に対して厳密に強い phase/radius certificate が成立する十分条件と、その情報層では改善不能となる限界条件を与える。

これは現時点では候補であり、新規性を確定した主張ではない。scalar 分離、weak-value 表現、q-numerical-range の幾何、既知の norm bound を並べるだけでは独立した成果とみなさない。

## 2. 対象と記号

### 2.1 finite paired-Taylor RTE

次数 K+1 の打切り多項式を `P_(K+1)(-i τ h) = Σ_(n=0)^(K+1) (-i τ h)^n / n!` とする。paired sampling を平均した有限 RTE channel/operator を対象とし、主な既存数値証拠は K=2、すなわち P3 に対するものである。

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
| C1 | 中核・基礎 | 正 scalar と方向依存補正の分離、および phase/radius certificate の正しい定義 | 局所代数は確認済み。一般形の証明監査が必要 |
| C2 | 中核・新規性候補 | 同じ情報層で norm-disk baseline より強くなる十分条件と、改善不能となる限界条件 | 未完成。最重要の証明義務 |
| C3 | 条件付き応用 | certificate を用いた finite-RTE parameter/resource design | C1/C2 が残る場合だけ実施。主成果の必須条件ではない |

利用可能情報は少なくとも次の三層に分ける。

- I0: `||Q||` と ρ_min だけ。
- I1: Q の Hermitian/anti-Hermitian 成分、符号、sector、交換・対称性など、状態非依存の追加構造。
- I2: 対象状態または状態族に関する directional expectation、moment、部分空間情報。

C2 の「改善」は、比較対象と同じ情報層だけを用いて証明しなければならない。I2 の情報を用いた bound を I0 baseline と比較して強さを主張する場合は、情報取得 cost と適用条件を別に示す。

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

これは scalar recentering により r^-3 の共通 radial drift を除き、phase-relevant term がより高次に現れ得ることを示す出発点である。一般 operator 版の一様 remainder は未証明である。

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
| Gu et al., noise-resilient phase estimation with randomized compiling | Hermitian Kraus operator を持つ perturbation の一次 phase invariance | scalar/Hermitian 方向の位相不変性に直接近い | `OPEN_FULL_TEXT_REDUCTION_AUDIT` |
| Ogawa et al., operational formulation of weak values | 小変換に対する postselection amplitude の感度としての weak value | 4.2 の比表現と近い。表現自体は新規でない | `KNOWN_NOT_NOVEL` |
| Li ほか、q-numerical range | fixed overlap matrix element の値域と幾何 | certificate の自然な既知言語 | `KNOWN_NOT_NOVEL` |
| Yi–Crosson / Li, product formula spectral or QPE error | PF の spectral error、gap、residual に基づく解析 | outer PF と位相推定の誤差解析に近い | `CLOSE_NOT_DIRECT` |
| Casares et al., SPRINT | signal-processing 型の resource-efficient phase inference | 半径と位相の共同利用に近い | `OPEN_SCOPE_AUDIT` |
| Hu–Jin, nonunitary amplitude/phase analysis | 非ユニタリ発展の amplitude/phase 構造 | finite RTE の非ユニタリ平均との重複可能性 | `OPEN_SCOPE_AUDIT` |

検索で同一表現が見つからないことは新規性の証拠にしない。特に Gu et al. と q-numerical-range 文献について、定理の仮定・結論を全文で照合し、C2 が既知結果の直接系でないことを確認する。

## 6. 証明義務

| ID | 証明・監査項目 | 完了条件 | 現在状態 |
|---|---|---|---|
| T0 | semantics と branch の固定 | K,r,T,𝔅,Γ_c,U,Q,arg の規約を一意にし、負時間を含む | 契約済み、原稿監査待ち |
| T1 | 非可換な有限 RTE の一般 bound | full product/order を保つ operator bound と適用 domain を証明 | 未完 |
| T2 | 同一情報での strict improvement | I0/I1/I2 ごとに norm disk より厳密に強い十分条件を証明 | 未完 |
| T3 | 情報層ごとの限界・最適性 | 同じ情報を共有する識別不能例、または改善不能定理を構成 | 未完・最有力の新規性候補 |
| T4 | finite-RTE 内部実現と remainder | paired Taylor の Q が仮定を満たす条件と一様 remainder を証明 | 未完 |
| T5 | certificate 入力の取得可能性 | 必要情報、古典計算/測定 cost、abstention 条件を明示 | 未完 |
| T6 | controlled implementation と減衰 | normalization、success/amplitude、controlled relative phase を整合 | 一部検証済み、一般化未完 |

数値実験の pass は、これらの証明義務を埋めない。とくに 8 個の strict-gain witness は T2 の存在例であり、一般十分条件や T3 の限界定理ではない。

## 7. 既存証拠とその限界

| 検証 | 固定された結果 | この原稿での用途 | してはいけない外挿 |
|---|---|---|---|
| FR1 | phase/amplitude 分離の基本 mechanism と stop 条件を確認 | 記号・semantics の検算 | 一般定理、H4/H12 資源優位 |
| FR-R1a | 既存 artifact の事後解析で scalar recentering と directional geometry を確認 | C1/T0 の経験的裏付け | blind prediction、実用 GO |
| FR-R1b | 非一様 4x4、20 条件・61 状態、610 records、227 applicable、soundness failure 0、strict FR-gain 8、one-sided fixed-budget witness 0 | C2 の存在例と C3 を保留する根拠 | 普遍的改善、resource advantage |

現時点の決定は `MECHANISM_ONLY_NO_PRACTICAL_GO` である。既存 artifact から、少なくとも次の図表だけを再利用候補とする。

1. scalar recentering 前後の phase/radius error。
2. norm-disk と FR directional certificate の幅の比較。
3. strict-gain witness と abstention/failure 条件の対比。
4. fixed-budget で one-sided witness が 0 だった負の結果。

これらのために新しい grid は実行しない。

## 8. 完成判定

次のいずれか一つへ収束させる。

### `PROCEED_THEORY`

- 先行研究監査後も C2/T3 の独立した差分が残る。
- T1–T4 のうち主張に必要なものを証明できる。
- 既存 FR-R1b 証拠が定理の非空性・failure mode を支える。

### `TECHNICAL_NOTE`

- 数式統合、実装 semantics、negative result には価値があるが、主定理が既知結果の直接系である。
- 新規手法としてではなく、有限 RTE の正しい解析・診断ノートとして閉じる。

### `STOP_NEW_METHOD`

- C1/C2 が既知理論へ直接還元され、独立した theorem、limitation、または診断原理が残らない。
- 追加の数値 grid で延命しない。

### `ONE_OPEN_ITEM`

- 上記判定に必要な未解決点が一つだけ残る場合、その一点だけを事前登録して解く。
- 新しい gate 名や探索系列を増やさない。

## 9. C3を行う場合の条件付き応用契約

C3 は C1/C2 の新規性が残る場合だけ行う。比較では次をすべて固定する。

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

## 10. 許される追加証拠

原則として既存 artifact で完成させる。追加が必要なら、以下を最大一件ずつに制限する。

1. T4 の証明だけでは branch/remainder の成立域が確定しない場合の、事前登録済み finite-RTE 内部例。
2. C3 を実際に主張する場合の、独立な小規模 application 一件。

いずれも、問い、入力、評価量、GO/STOP を先に固定する。H4/H12、FR-R2、長 RPE、full compiled cost、多数の PF family、広い delta/r/L_D grid は含めない。

## 11. 直ちに行う作業

1. Gu et al.、weak-value、q-numerical-range、finite-RTE 原論文の全文定理を照合する。
2. T1–T4 を引用から独立に証明し、成立域と反例を同じ原稿へ書く。
3. 既存 artifact から第7節の最小図表を抽出する。
4. `PROCEED_THEORY`、`TECHNICAL_NOTE`、`STOP_NEW_METHOD`、`ONE_OPEN_ITEM` の一つを採択する。

この順序が完了するまで、新しい runner、計算 artifact、検証 ID を追加しない。

## 12. 明示的に主張しないこと

- H4/H12 や一般化学系への一般性。
- 最終 RPE 総 cost、実機優位性、noise/backend robustness。
- finite RTE が常に norm baseline より強いこと。
- 高い signal radius を新しく認証できること。
- q-numerical range または weak-value 表現そのものの新規性。
- `C_use` を厳密上界とみなすこと。
- 既存 FR-R1b の mechanism-only 結果を実用的 GO と読み替えること。

## 参考文献監査入口

- Günther et al., “Phase Estimation with Partially Randomized Time Evolution,” *PRX Quantum* 7, 020332 (2026), <https://doi.org/10.1103/ynxb-p2xq>.
- Wan, Berta, and Campbell, “Randomized Quantum Algorithm for Statistical Phase Estimation,” <https://arxiv.org/abs/2110.12071>.
- Gu et al., “Noise-resilient phase estimation with randomized compiling,” <https://arxiv.org/abs/2208.04100>.
- Ogawa et al., “Operational formulation of weak values without probe systems,” <https://arxiv.org/abs/1912.10222>.
- Chi-Kwong Li, “The q-numerical range of a matrix,” <https://doi.org/10.1080/03081089808818538>.
- Yi and Crosson, “Spectral analysis of product formulas for quantum simulation,” <https://doi.org/10.1038/s41534-022-00548-w>.
- Li, “Theory of Quantum Simulation with Product Formulas,” <https://arxiv.org/abs/2111.10430>.
- Casares et al., “SPRINT,” <https://arxiv.org/abs/2606.30741>.
- Hu and Jin, “Nonunitary amplitude-phase analysis,” <https://arxiv.org/abs/2602.09575>.
