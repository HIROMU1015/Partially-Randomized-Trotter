# 研究方針の再設計：有限RTEの位相・振幅誤差を分離する

作成日：2026-09-26
対象：HIROMU1015/Partially-Randomized-Trotter
確認branch：all-r-coherent-opt2-reoptimization
確認commit：4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172
文書の位置付け：研究方針・着地点・段階的検証の提案。既存のSTOP判定を変更しない。FR-0の数式・先行研究・比較契約は
[`docs/research/finite_rte_phase_amplitude_contract.md`](docs/research/finite_rte_phase_amplitude_contract.md)以下へ固定した。FR-1は2×2 fixed gridで完了し、結果は
[`docs/finite_rte_phase_amplitude_validation.md`](docs/finite_rte_phase_amplitude_validation.md)に記録した。FR-2以降と回路compileはまだ行っていない。

## 0. 結論

**推奨する次の主題候補は、「B2をB4へ近づける選択器」ではなく、「有限RTEの打切りが位相と振幅へ与える影響を分離し、位相推定に必要な近似精度を判断する方法」である。**

仮題：**有限ランダムTaylor時間発展における位相誤差と振幅変形の分離――部分ランダム化位相推定の誤差評価と精度設計**。

計画の主軸はこの一件へ絞る。ただし、新規性・実用上の利益が実証済みという意味での正式採択ではない。最初の小さな解析・機構検証が通った段階で本研究として採択する。

狙う成果は、有限RTEを使った非可換な時間発展列について、(i) 位相bias、(ii) 追加の振幅変形、(iii) 既知normalizationによる減衰を分け、従来のノルム型上界より適切な受理条件または設計則を与えること。PFの名前を変えること、低次公式を勝たせること、PRを決定論より常に安くすることは目標にしない。

P-A/P-B/P-C/P-D/R3-Sの停止結果は保存する。旧S2、H12、長RPE、広いprefix掃引は自動再開しない。今回の案に必要な一部の小系演算子評価は旧S2と技術的に重なるが、対象仮説・対照・停止点を新たに固定し、旧計画全体は復活させない。

---

## 1. まず訂正すべき、これまでの研究の読み方

### 1.1 STOPが否定したのは、実施した仮説と範囲である

P-Aは区間分割固有の追加効果、P-Bは現H4 gridでのenergy-only/signal-aware選択差、P-Cは特定H4 familyに対する予測・追跡法、P-DはS1の固定候補集合での有限補正の選択上の利益、R3-Sは現時点で具体化できた方法差分について停止した。[Repo1][Repo2][Repo3]

これらを「回路合成研究は無意味」「信号誤差は常に小さい」「構造間相殺は研究できない」「有限RTEには未解決点がない」という一般的否定へ拡張しない。

### 1.2 最適近傍でB2/B4が一致するだけでは、新しい機構とは言えない

最適近傍はB4でfeasibleな候補を基に定義されている。その集合内でB2受理/B4不適格が0という事実は、集合の作り方からも生じる。予測誤差が小さいことや選択regret=0には意味があるが、それだけで普遍的な「最適近傍で近似が自動的に正確になる原理」は主張できない。

全候補の45件の不一致も、B2とB4というモデル同士の受理差であり、実際の物理的誤採用45件を意味しない。B4は詳細モデル参照であって、真の信号位相を直接測った値ではない。[Repo2]

### 1.3 簡単に計算できる量の予測を、研究の主目的にしない

有限cutoffのB_K、整数配分、Taylor残差は、既存の実装で直接計算できる。これらの閉形式・有限和を用いてB2/B4差を再計算するだけでは、独立した方法論として弱い。

安価な判別式を研究するなら、「直接求めると本当に高価な何を避けるのか」を明示する。今回なら、全体系の有限RTE平均演算子・信号を直接計算せずに、位相biasと信号半径を拘束することが候補になる。

### 1.4 nested/nativeの13倍差は、まだ新しい現象の証拠ではない

事後解析では、差の中心は内部deterministic component actionsである。さらに誤差受理規則・融合の数え方・内部substepなどの影響がある。[Repo2]

したがって、この比だけを新しい研究テーマの根拠にせず、既知の内部精度配分・native構成を対照にする必要がある。今回の主計画には組み込まない。

### 1.5 「定理がまだない」と「研究してはいけない」は違う

新しい定理・方法が完成していないこと自体は、研究を始めない理由にはならない。一方、既存の言葉を並べただけで重い計算を始めるのも避ける。

最初に必要なのは、具体的な数理対象、既知研究との未解決な差、反証できる予測、限定した初期検証である。本書では、その出発点となる位相・振幅の分解と初等的な上界の導出を6節に記す。新しい枠組み全体の保証を完成させてからでなければ探索してはいけない、という条件は置かない。

---

## 2. 候補の選び直し

| 候補 | 現時点の評価 | 扱い |
|---|---|---|
| B2/B4の差が小さい領域を学習する | 定義済み有限量の比較や一般multi-fidelity法へ還元されやすい | 主題にしない。必要時の補助解析 |
| nested/nativeの差を調べる | 既知構成・内部精度・費用の取り方で説明される可能性が高い | 本研究の主軸にせず保留 |
| compiled costの局所予測を理論化する | compiler classを固定すれば別研究になり得るが、現状では新しい構成・保証をまだ持たない | 自動的な次候補にしない |
| **有限RTEの実位相・振幅誤差を分離する** | 直接の物理量を対象にでき、既存の多項式・state-action基盤を再利用できる。数理的な出発点も具体的 | **最初に検討する一件** |

この選択は、部分ランダム化の現在の実装を必ず最後まで完成させるためではない。目的量を「全設計の総コスト順位」から「推定される位相の誤差と、その推定に残る信号」へ変える。

---

## 3. 推奨する研究RQと、主張の範囲

### 主RQ

> 有限paired Taylor RTEを非可換な決定論的時間発展列へ挿入したとき、打切り誤差のどの部分が信号の位相を動かし、どの部分が振幅を変えるか。両者を分離することで、実際の信号を全行列で計算せず、位相推定に使える近似精度を従来より適切に判断できるか。

### 副RQ1：蓄積と破綻条件

> 単一tailにおける位相誤差の高次性は、非可換な挿入、負時間、反復、不完全な対象状態でもどこまで残るか。何がその改善を壊すか。

### 副RQ2：設計への効果

> 同じ位相精度・統計成功条件に対して、K・rを選ぶとき、位相専用評価の利益は、振幅減少によるshot増加を含めても残るか。

独立RQを増やすのではなく、主RQの正当性→適用範囲→資源への含意という順序にする。

### 初期scope

- Hamiltonianは時間非依存でHermitian。
- 元の確率分布を変える新samplerは導入しない。
- まずK=2のpaired有限Taylor RTEを主対象とし、K=0を既知対照、K=4を移送・拡張対照にする。
- scalar/identity phaseは正確に別処理し、無視しない。
- RTE以外の部分はunitaryな参照列として固定する。内部H_Dを近似するときも、それ自体は同じPF unitaryとして参照列へ含める。
- 比較する有限RTE配置を変える際、Hamiltonian、基準PF、入力状態、物理時間、control規約をそろえる。
- 状態準備法の開発、noise、FT合成、新しい高次PF探索は初期scope外。

---

## 4. 先行研究との境界

| 一次研究 | 既知の内容 | 本案に残る候補差分／注意 |
|---|---|---|
| Günther et al., PR phase estimation [Lit1] | RTE、qDRIFTの信号、部分ランダム化、絶対tail時間・sampling負担 | RTEの位相と振幅を区別する発想は既知。有限Kの非可換な挿入に対する、状態条件を明示した定量的位相評価が対象 |
| Wan–Berta–Campbell [Lit2] | Taylor由来のrandomized LCU、統計的位相推定、samplingとgate費用の交換関係 | 有限打切り、係数の正規化、LCU平均は新規性ではない |
| Yi and Crosson, spectral PF analysis [Lit3] | unitary PFの固有値・固有ベクトル誤差を分離する解析 | 今回はfinite-RTE平均が一般に非unitary。位相／振幅／参照状態ずれを含む信号評価が必要 |
| Casares et al., SPRINT [Lit4] | near-integrability、factorization、randomization、位相・スペクトル広がりの解析 | 位相と減衰の区別自体を新規としない。Appendix Fの平均信号解析と数式レベルで照合する |
| phase-lag / dissipationの数値解析 [Lit5] | Taylor/RK系の分散誤差と振幅誤差は別次数になり得る | scalar Taylor展開だけでは研究にならない。非可換列、入力状態条件、測定資源への接続が必要 |
| certified multi-fidelity / best-arm identification [Lit6][Lit7] | fidelityと費用を使った選択・認証 | 本案はoptimizerを新規化しない。必要なら、導いた誤差情報の利用先として既知手法を使う |
| LiのQPE誤差解析 [Lit8] | 不完全な入力状態、近似unitary、random unitaryをresidual・gap・concentrationから解析 | finite-RTE平均信号のphase/radius分離と同一ではないが、状態情報を使う主張の直接対照にする |
| Hu--JinのAmplitude-Phase Separation [Lit9] | 一般のnonunitary生成子をcoherent/dissipative部分へ分けるalgorithmic framework | 本案はfinite-RTE推定量の局所相対誤差を扱う限定問題であり、generic nonunitary simulationやAPSという名称を主張しない |

**現在の確認で「世界初」は主張しない。** 具体的な最終上界と採用条件ができた時点で、[Lit1]のqDRIFT/RTE Appendix、[Lit4]の平均信号解析、[Lit3][Lit8]のstate-specific bound、[Lit5]のdispersion解析、[Lit9]のnonunitary分解へ還元されないかを照合する。本研究では`APS`を手法名・略称として使わない。

一つの初等的不等式を導けただけで独立論文が成立すると決めない。既知の最も近い結果より、仮定、計算量、非可換性の扱い、許される近似設定、または設計上の効果のどこが変わるかを示す。

---

## 5. 数理的な出発点：有限RTEはどんな平均演算子を作るか

### 5.1 記号

H_R = λh、||h||≤1、短時間s、τ=λsとする。Kは偶数cutoffであり、現行実装のpaired規約では通常Taylor次数K+1まで残す。[Repo4][Repo5]

\[
P_{K+1}(-i\tau h)
=\sum_{n=0}^{K+1}\frac{(-i\tau h)^n}{n!}.
\]

有限分布の正規化を

\[
B_K(\tau)=\sum_{\substack{n=0\\n\;\mathrm{even}}}^{K}
\frac{|\tau|^n}{n!}\sqrt{1+\frac{\tau^2}{(n+1)^2}}
\]

と書く。独立にsampleする現行paired規約に対して、1短stepの平均は

\[
\mathbb E[U_\omega(s)]
=\frac{P_{K+1}(-isH_R)}{B_K(\lambda s)}.
\]

この等式は既存RTE構成から得られる基礎式であり、新規成果ではない。実装の積順・符号・identity処理は、既存referenceと独立に照合する。

RTE短stepが複数あり、各stepを独立にsampleする場合、全体の正の既知normalizationを
\(\mathcal B=\prod_\nu B_{K_\nu}(\tau_\nu)\)とする。既存実装の用語に合わせ、Taylor分子を積み上げた
**normalization-corrected operator**を\(A_{\rm corr}\)、実際のsampled-unitary平均を
\(A_{\rm mean}=A_{\rm corr}/\mathcal B\)、tailをexact exponentialに戻したunitary参照列を\(U\)とする。このとき、

\[
z_0=\langle\psi|U|\psi\rangle,\quad
z_{\rm corr}=\langle\psi|A_{\rm corr}|\psi\rangle,\quad
z_{\mathrm{obs}}=\langle\psi|A_{\rm mean}|\psi\rangle
=z_{\rm corr}/\mathcal B.
\]

\(\mathcal B>0\)なのでargは変えない。ただし\(|z_{\rm corr}|\)は一般に\(|z_0|\)と同じではない。**既知normalizationによる減衰と、有限打切りによる振幅変形を分ける。**

独立サンプリングなら平均の積へ置ける。固定した1 trajectoryをq回再利用する場合は\(\mathbb E[U_\omega^q]\neq(\mathbb E U_\omega)^q\)が一般的なので、本式をそのまま使わない。

### 5.2 K=2のscalar例

\[
P_3(-ix)=1-ix-\frac{x^2}{2}+i\frac{x^3}{6}.
\]

x=0から連続な対数を選び、|x|が小さい範囲で、

\[
\log P_3(-ix)
=-ix-\frac{x^4}{24}-i\frac{x^5}{30}+O(x^6).
\]

したがって、

\[
\log|P_3(-ix)|=-\frac{x^4}{24}+O(x^6),
\qquad
\arg P_3(-ix)+x=-\frac{x^5}{30}+O(x^7).
\]

つまり、近似演算子全体の先頭誤差は4次でも、scalar位相の先頭誤差は5次になり得る。正確には\(|P_3(-ix)|^2=1-x^4/12+x^6/36\)であり、全xで単純な減衰と仮定しない。

これは既知のphase-lag/dissipation型の現象であり、**この展開だけを新規性にしない**。研究の難所は、異なるbasisの非可換unitaryの間へこれを挿入し、長く反復し、入力状態が参照列の厳密固有状態でもないときである。

---

## 6. 解析の具体案：相対誤差のHermitian/anti-Hermitian成分を残す

以下は本書で整理した初等的な解析出発点である。数学的な導出と、未実施の数値検証・新規性監査を区別する。新定理としての優先権は主張しない。

### 6.1 一般のunitary参照列

\[
U=U_N\cdots U_1,\qquad A_{\rm corr}=A_N\cdots A_1,
\qquad A_j=U_j(I+D_j).
\]

RTE以外の同じunitary factorはD_j=0としてよい。各局所相対誤差を

\[
D_j=F_j+iG_j,\quad
F_j=(D_j+D_j^\dagger)/2,\quad
G_j=(D_j-D_j^\dagger)/(2i)
\]

と分ける。F_j,G_jはHermitian。

検証済みの上界を\(a_j\geq\|F_j\|\)、\(b_j\geq\|G_j\|\)、\(e_j\geq\|D_j\|\)とする。保守的にはe_j=a_j+b_jを使える。

\[
a=\sum_j a_j,\quad b=\sum_j b_j,\quad
s=\sum_j e_j,\quad
R_2=\prod_j(1+e_j)-1-s.
\]

全て非負で、R_2は二次以上の積をまとめた上界。

prefix W_{j-1}=U_{j-1}\cdots U_1を使えば、

\[
U^\dagger A_{\rm corr}=\prod_{j=N}^{1}(I+\widetilde D_j),\qquad
\widetilde D_j=W_{j-1}^\dagger D_jW_{j-1}.
\]

したがって、

\[
U^\dagger A_{\rm corr}=I+\sum_j\widetilde F_j+i\sum_j\widetilde G_j+R,
\qquad \|R\|\leq R_2.
\]

非可換性を無視していない。非可換な高次積もRへ含める。ただしこの扱いは保守的なので、必要なら後でcommutator構造を残した上界へ改良する。

### 6.2 入力状態と信号の条件

正規化した|ψ⟩について\(z_0=\langle\psi|U|\psi\rangle\)、\(\rho=|z_0|>0\)とする。

\[
\kappa=\frac{\sqrt{1-\rho^2}}{\rho}.
\]

\(U^\dagger|\psi\rangle=z_0^*|\psi\rangle+|\eta\rangle\)、\(\|\eta\|=\sqrt{1-\rho^2}\)より、

\[
\frac{z_{\rm corr}}{z_0}
=1+\langle\psi|(U^\dagger A_{\rm corr}-I)|\psi\rangle
+\frac{\langle\eta|(U^\dagger A_{\rm corr}-I)|\psi\rangle}{z_0}.
\]

Hermitian成分の期待値が実数であることを使うと、

\[
L=1-a-R_2-\kappa(s+R_2),
\qquad
Y=b+R_2+\kappa(s+R_2)
\]

に対して、L>0なら

\[
|\arg z_{\rm corr}-\arg z_0|_{\mathrm{principal}}
\leq\arctan(Y/L),
\]

\[
|z_{\mathrm{obs}}|\geq\frac{\rho L}{\mathcal B}.
\]

証明は、\(z_{\rm corr}/z_0\)の実部をLで下から、虚部の絶対値をYで上から抑えるだけである。L>0が位相枝の局所的な安全条件にもなる。L≤0は「物理的に失敗」ではなく、この上界では認証できないという意味。

### 6.3 この式が示す、検証可能な機構

参照列Uの固有状態を入力する場合、ρ=1、κ=0である。このときF_jの一次項は位相へ直接寄与しない。G_jと二次以上の積が位相を動かす。

K=2の短stepでは、小さいη=|s|\|H_R\|について概ね

\[
a_j=O(\eta^4),\qquad b_j=O(\eta^5).
\]

N個の同程度の誤差因子、Lが1から大きく離れない領域なら、位相上界は概念的に

\[
O(N\eta^5+N^2\eta^8+\kappa N\eta^4)
\]

となる。通常のノルム和O(Nη^4)とは違う項が支配し得る。

一般の偶数Kでは、一次のradial成分がO(η^{K+2})、tangential成分がO(η^{K+3})となる構造を確認する。単一stepと固定総時間での累積次数は混同しない。例えばN∝1/ηとなる比較では指数が一つ変わる。

この式は、必ず改善するという保証ではない。状態条件κ、非可換な高次積R_2、参照信号のゼロ近傍、normalizationの増大が改善を消す。**改善と破綻を同じ式で説明すること**を目指す。

### 6.4 cheapに使えるかが最大の実装上の関門

a_j,b_jは、Hermitian H_Rに対するscalar関数

\[
e^{ix}P_{K+1}(-ix)-1
\]

の実部・虚部を\(|x|\leq |s|\Lambda_R\)、\(\|H_R\|\leq\Lambda_R\)で抑えて得られる。λ_RをΛ_Rとして使うことは安全だが、緩い可能性がある。有限gridの最大値を厳密supremumとは呼ばず、解析的余項または検証付き区間評価を用いる。

ρをdense referenceから与えた場合、上界の機構診断には使えるが、スケーラブルな受理方法を実装したとは言わない。実用版には、独立に正当化されたρ下界が必要である。

例として、参照PFの一つの固有分枝への重みw_0が保証できれば、すべての反復qで\(\rho_q\geq\max(0,2w_0-1)\)を使える。ただしこの不等式と固有重み診断は既知であり、新規性ではない。w_0の推定・認証費用を別途計上する。[Lit3]

また、対象のexact H固有状態に対して参照Uの作用誤差をεで正当に抑えられるなら\(\rho\geq\max(0,1-\epsilon)\)は使えるが、緩さにより今回の利益が消えるかもしれない。

**ρ下界を安価に用意できない場合、参照値を使った理論診断と実用選択則を分け、実用化を達成したことにはしない。**

### 6.5 研究としてさらに必要なもの

上記の代数だけでは最小着地点に届かない。少なくとも次を確認する。

1. 既存のphase-lag・state-specific・平均信号解析から、同じ仮定・入力・計算量の結論が既に得られるか。
2. a_j,b_j,ρ下界を、真の答えを知った後でなく事前に用意できるか。
3. 非可換な実用候補で有効な範囲があり、通常の良いノルム上界を超えるか。
4. 許されるK/rや必要shotを合わせた資源・適用時間の改善につながるか。

---

## 7. 着地点：最小成果と目標成果

### 7.1 最小の独立した着地点

> 明示したfinite-RTE／unitary参照列／入力状態条件について、位相と振幅を分離した誤差評価と、その有効・無効条件を与える。既知の適切な対照より、非可換な未使用条件の位相精度・信号半径を有用に予測できることを示す。

成果物は、命題または正当化された診断法、仮定と限界、反例・破綻機構、再現可能な小系数値検証である。非可換な列への適用・定量的な改善が既知結果に完全に含まれるなら、その再導出だけでは独立成果としない。

### 7.2 目標の着地点

> 同じ物理時間、位相誤差許容量、信頼度、入力状態、費用定義に対し、位相・振幅評価を使ってK/rを選び、従来のノルム型受理条件と比べ、正しい位相推定に必要な資源または古典評価負荷を改善する。

資源の比較は、rが小さいだけで終えない。振幅低下に伴うshot増を含め、同じタスクで比較する。PF名が変わらなくても、同じPF内で近似設定を適切に選べればよい。

### 7.3 発展先（必須ではない）

負時間を含む高次PF、複数cutoff配置、一般のpolynomial approximation、近似入力状態、長時間の非正規効果、複数RPE roundへの接続。新たな定理や有用性がないまま、適用対象だけを増やさない。

### 7.4 最小成果に届かない結果

- K=2のscalar位相が5次だった、という観察だけ。
- dense全行列で真の信号を求め、その値に後から合う係数を作っただけ。
- B4が棄却した一条件を、誤差保証を緩めて通しただけ。
- 参照状態が厳密固有状態である場合だけの例を、一般の位相推定へ拡張しただけ。
- phase boundは小さいが、追加減衰でタスクcostが大きくなった事実を隠す。

負の結果でも、適用域のsharpな制約や反例によって、何を仮定しないと改善不能かを説明できれば成果候補になる。ただし、数点で効果が出なかったことは一般no-goではない。

---

## 8. 比較対照と評価対象

| 対照 | 内容 | 用途 |
|---|---|---|
| REF | 同じunitary参照PF、tail exact | RTEだけによる位相・振幅誤差の基準 |
| OLD-NORM | 現行Taylor残差→複素信号誤差→asin位相上界 | 現行受理法との比較 |
| STRONG-NORM | spectral/scalar構造を使って改善した演算子ノルム余項。ただしphase/amplitudeは分離しない | 弱いboundだけに勝つ問題を防ぐ |
| SCALAR/EIGEN | 単一tailまたは可換系で既知の位相・振幅関数を正確計算 | 既知現象の再現と実装確認 |
| PROPOSED-REF | 新分解にdenseで得たρを与える | メカニズム診断。実用選択則とは呼ばない |
| PROPOSED-AVAILABLE | 独立に保証・検証されたρ下界とnorm情報のみを与える | 本研究の実用性・費用を判定する主候補 |

同一candidateに対するboundの比較と、各boundから設計を選ばせる比較を分ける。まず前者で機構を確認し、後者では同じ候補集合・共通詳細参照・同じタスクを使う。

### 三層の誤差を分ける

1. 物理target \(e^{-iHT}\) とunitary参照PFの差：PF誤差。
2. unitary参照PFと有限RTE normalization-corrected operator \(A_{\rm corr}\)の差、および\(A_{\rm mean}=A_{\rm corr}/\mathcal B\)の半径：今回の主対象。
3. 有限RTE平均信号と有限shot推定の差：統計誤差。

本研究の主boundは2を扱う。最終タスクでは1・2・3の同じ位相単位への接続を別途行う。2だけ改善して全エネルギー誤差を達成したとしない。

---

## 9. 検証の進め方：一つの仮説を段階的に試す

実行フェーズ名はFR-0～FR-4とする。旧P-D S0/S1/S2やカタログIDとは別の検証である。FR-0とFR-1は完了し、FR-2以降は未実施である。

### FR-0：式と比較契約の固定

**目的**：6節の出発点が実装・既知研究に整合し、調べるべき差が具体的かを確認する。

**作業**：
- finite平均＝polynomial/Bの等式を、現行reference、符号、identity処理、sampling独立性まで確認する。
- K=0/2/4のscalar実虚部と残差を整理する。
- 6節の積分解・R_2・ρ条件を独立に検算する。
- [Lit1–5]の最も近い式と、同一結論への還元可能性を照合する。
- 何を新しく求めるかを「非可換列」「利用可能な状態情報」「保証する量」「費用」の四項で固定する。

**成果物**：仮定・命題・証明または未証明箇所・prior-art対応表・FR-1実行契約。

**停止条件**：scalarの既知phase-lag以外に差が残らない、または対象タスクが未定義のまま。証明に穴が見つかっても直ちに一般no-goとはせず、修正可能な仮説かをこの段階で判断する。

**しないこと**：新しい分子、Qiskit compile、広いPF family、旧S2一式。

### FR-1：非可換toyでの機構・反例試験

**目的**：非可換な挿入後も位相と振幅を分けることに意味があり、その破綻要因を式が捉えるか。

**最小対象案**：2×2または4×4 Hermitian系。一つの主familyを\(H_D=aZ\)、\(H_R=b(\cos\vartheta Z+\sin\vartheta X)\)とし、可換端点と非可換条件を含める。identity部分を別に追加する意味論対照も用意する。値は実行前に固定し、結果に合わせて角度・状態を入れ替えない。

**比較**：
- exact reference列の固有状態。
- 同じHの基底状態。
- 対象外成分を一定量混ぜた状態。
- 可換／非可換、正時間／負時間、短い対称列／対称性を外した対照。

全部を直積にせず、主比較と一因子変更の対照へ分ける。主K=2、K=0既知対照。K=4は事前に選ぶ独立チェックだけ。

**観測**：実位相差、振幅比、normalization、a,b,R_2,κ,L、OLD/STRONG-NORMと提案bound、誤受理、数値残差。q一定の局所次数とT一定の反復次数を別に解析する。

**GO**：独立な非可換例で、導出した条件内のcoverageが保たれ、radial一次項を位相へ数えない利益またはその破れを説明できる。新規性・実用性の確定ではなく、FR-2へ進む根拠とする。

**STOP/縮小**：可換scalar例でしか利益がない、非可換性・状態条件を戻すとboundが常に空、既知の強い対照で全て説明できる。反例の原因が特定できる場合は、対象classを一度だけ明示的に縮小して再設計する。都合のよいgrid追加はしない。

**最初の共有点**：FR-0＋FR-1まで。ここで一度必ず結果と差分を確認する。

### FR-2：既存H4基盤への接続と、安価な入力情報の検査

**目的**：toyだけでなくDF/PFの既存経路で、真の有限平均信号の位相・振幅を予測できるか。

**範囲**：既存H4 snapshotをdevelopmentとして用い、一つのunitary PFと一つの分割から始める。K/rの少数設定を、従来boundの余裕がある点・境界付近・認証不能点に分けて結果前に選ぶ。旧S1の全308候補を再計算しない。

**方法**：既存finite-Taylor momentとsector state-actionを再利用する。unitary参照の全行列固有分解や有限平均の直接行列は、小系のground truthだけに使う。提案法の入力へreference回答を流用しない。

**最大の判定**：PROPOSED-REFでは改善してもPROPOSED-AVAILABLEでは改善しない場合、その差を明示する。状態情報を得る費用が直接参照に近ければ、「安価な認証法」の主張をしない。

**GO**：利用可能な状態情報を使っても有用なphase/radius予測が残る。または、どの追加情報が必要かを定量的に限定できる。

**STOP**：全行列の真値を与えないと利益が出ず、別の独立した解析成果もない。有限誤差が既存タスクで常に無関係なほど小さい場合も、資源改善方向へ自動進行しない。

### FR-3：未使用条件での確認

**目的**：調整した規則の外挿ではなく、固定した式・定数・受理方法の有効域を確認する。

実行前にsnapshot/path/hashの既参照一覧を作る。未使用と確認できた小系またはgeometryの一件を主holdoutとし、さらに状態条件または非可換性を変える一件を破綻側対照として固定する。H4の過去のstretch条件は既参照なのでblindとは呼ばない。

式・normの求め方・ρ下界・数値許容差・失敗時処理を固定する。holdoutで調整したらdevelopmentへ戻し、別の新規holdoutが必要になったことを記録する。

**GO**：宣言した条件ではphase/radiusの予測・coverageが保たれ、条件外では無理に安全と判定しない。有限個の成功を普遍的証明に昇格しない。

**STOP/縮小**：状態・系を変えるたびに定数の事後調整が必要、または適用域を計算前に判定できない。

### FR-4：位相タスクでK/r設計と資源を比較

**目的**：小さいphase boundが、実際に意味のある設計変更へつながるか。

同じ物理時間T、同じ入力状態、同じ基準PF、同じ目標位相精度・信頼度にする。PF誤差、有限RTE誤差、統計誤差の予算規則も共通にする。最初からPF次数・分割・DF表現まで共同最適化しない。

候補は有限のK/r集合とし、全方法へ同じ選択機会を与える。各法で選ばれた設定を、共通のfinite平均referenceと同じ測定モデルで評価する。

正の信号半径下界を\(\underline\rho_{\rm obs}\)、統計位相余裕をβ_statとすると、同じ濃度不等式の下で、shot数の比較には概ね

\[
N\propto\frac{\log(1/\alpha)}{\underline\rho_{\rm obs}^2\sin^2\beta_{\rm stat}}
\]

を用いる。定数・cos/sin配分は既存の同じ規約を両法へ用いる。Nのモデル予測と、有限shot模擬で検証した値を区別する。

costはまず同一component-actionモデルで比較し、設計差を変え得る代表設定だけcompileする。実機費用・FT費用を主張する段階になって初めて該当合成を加える。

**GO**：正しい精度・信頼度を維持し、必要work、shot×cost、またはreference評価費用のいずれかに説明可能な改善がある。PF名の変更やlog Bの削減率だけで判断しない。

**代替着地点**：資源改善は小さくても、非可換な信号誤差の新しい有効域・破綻条件が厳密に残れば理論・診断研究としてまとめる。何も残らなければ主題化を止める。

---

## 10. 事前登録に必要な最小項目

数値を任意に増やすのではなく、最初のFR-1の前に以下だけ固定する。

| 項目 | 固定する内容 |
|---|---|
| 対象命題 | どの位相・振幅bound、どの仮定を検査するか |
| 入力class | H_D/H_R family、状態、正負時間、非可換対照 |
| 比較単位 | 1短step、1外側step、固定Tのどれか。別々に記録 |
| baseline | OLD-NORMだけでなくSTRONG-NORMと既知scalar対照 |
| 計算条件 | 精度、norm上界の求め方、phase枝、zero-signal処理 |
| 設計と評価の分離 | dense referenceが提案法の入力へ漏れない規則 |
| 主判定 | bound coverage、適用域、機構予測、必要情報の費用 |
| 数値許容差 | machine precision、演算子残差、signal conditioningを基に事前設定 |
| 実用差の閾値 | FR-4前に、タスクと費用不確かさから固定。旧5%を科学的定数として流用しない |
| 失敗時 | 原因不明の違反なら停止。技術修正は旧版保存・差分記録。都合のよい条件変更を隠さない |
| 終了範囲 | FR-1終了後にレビュー。FR-2以降は自動起動しない |

経験的に違反0というだけでconfidence付きの普遍保証とは呼ばない。数学的上界を名乗る部分は証明・仮定・数値評価の安全性が別途必要。経験的診断として出す部分はcoverageと失敗をそのまま示す。

---

## 11. 成果物と図の構成

### 11.1 最低限の保存量

- source commit、library版本、snapshotと入力Hamiltonianのhash。
- H_D/H_Rとidentity分離、basis/sector、参照unitaryの定義。
- K、各signed時間、r、全factor数、反復回数、総時間、sampling独立性。
- \(z_0\)、\(z_{\rm corr}\)、\(z_{\rm obs}\)、phase差、radius、PF phaseとの分離。
- 各a_j,b_j,e_j、a,b,s,R_2、ρ入力の由来、κ、L,Y、normalization。
- 従来bound、新bound、真値、適用不可の理由。
- 開発／既参照／未使用holdoutの区分。
- 解析proxy cost、実compile cost、古典前処理costを別々に保存。

新ファイル例（提案であり、現存を仮定しない）：

```text
docs/research/finite_rte_phase_amplitude_contract.md
docs/research/finite_rte_phase_amplitude_prior_art.md
docs/research/finite_rte_phase_amplitude_fr1_preregistration.md
docs/finite_rte_phase_amplitude_results.md
artifacts/finite_rte_phase_amplitude/<date>/...
```

既存の`rte.py`、finite-RTE signal reference、sector actionを再利用する。最初から汎用executorや新たな大規模階層を作らない。

### 11.2 論文または修士論文の中心図

1. scalar校正：同じTaylor cutoffでノルム・位相・振幅の次数が違うこと。既知背景として配置。
2. 非可換性と状態条件：a,b,R_2,κのどれがphaseを支配するか。
3. 従来bound／強い対照／提案bound対真の位相・半径。改善だけでなく認証不能点も表示。
4. 事前に利用可能な入力とdense oracle入力との差。cheap診断の現実性を表示。
5. 未使用条件でのcoverageと破綻。H4 trainingの図だけにしない。
6. 同じタスクでのK/r選択・shot×cost、または得られた適用条件の定量的意味。

最初の図だけ完成しても研究が閉じたとはしない。反対に、全RPE・H12を加えるまで閉じられない設計にもしない。

---

## 12. 進行判断のまとめ

| 結果 | 次の扱い |
|---|---|
| 一次radial項の分離が非可換列でも有用、独立な状態条件で使える | FR-2/3へ進み、本研究候補へ昇格 |
| 理論式は正しいがρ情報が高価すぎる | 条件付き理論結果と実用設計を分ける。cheap化に明確な道筋がなければ資源最適化主張は止める |
| 新しいboundが既知結果の直接系で、追加の適用条件・利益もない | 新規方法として主題化しない |
| phaseは改善するがamplitude低下でcost利益が消える | どの状態／時間域で起きるかを整理。非自明な限界が残る場合のみ診断・理論として完結 |
| 一条件で誤差が小さいが機構を説明できない | 条件を無制限に増やさない。数値floor、branch、state等の検算を先に行う |
| 数値または数学的反例で主仮説が崩れる | 旧B2/B4 grid探索へ自動で戻らず、何が欠けたかを明示して停止 |

研究方向を変更するのは、事前に指定した関門を越えられない理由が判明したときにする。1件のpilot不通過をその分野全体の不可能性へ拡張しない。

---

## 13. 今すぐ行うこと／保留すること

**現在の停止点**：FR-1ではG0/G1/G3/G4を通過したが、利用可能な$\underline\rho=0.8$でのG2は不通過だった。真の$\rho$を使う場合だけ改善したため`GO_FR2_MECHANISM_ONLY`とし、FR-2は開始しない。次は大規模計算でなく、oracleなしの高$\rho$認証または$\kappa s$項を縮める境界に独立差分があるかを再設計する。

**保留**：H12、長RPE回路の一体compile、PF全次数探索、全prefix掃引、新しいDF最適化、新しいsampling分布、noise/backend、旧S2の一括再開。

この案を本研究に採る判断は、前の候補より必ず成功すると保証することではない。前の失敗から、(i) 目的量を物理信号へ戻す、(ii) scalar既知結果と非可換列の差を先に示す、(iii) 位相と振幅を同時に評価する、(iv) 高価なoracle情報を隠して使わない、という具体的な改善をした計画である。

---

## 14. 確認した出典と確認範囲

### リポジトリ一次資料（上記commit固定）

- [Repo1: PROJECT_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172/PROJECT_MAP.md)
- [Repo2: P-D S1事後再解析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172/docs/research_direction_pd_s1_posthoc.md)
- [Repo3: R3 prior-art and minimal contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172/docs/research/r3_prior_art_and_minimal_contract.md)
- [Repo4: rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172/src/trotterlib/rte.py)（paired cutoff、分布、残差合成を確認）
- [Repo5: finite-RTE signal validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172/docs/finite_rte_signal_validation.md)（有限平均をmomentで直接計算する既存経路、位相上界の保守性を確認）
- [研究概要・現状](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b28bc64fa99ef0d7b7aa1f01b47294d05a2d172/docs/research/研究概要・現状.md)

### 公開一次文献

- [Lit1: Günther et al., Phase estimation with partially randomized time evolution](https://arxiv.org/abs/2503.05647)。PRX Quantum 7, 020332 (2026)。公開PDFのAppendix AのqDRIFT/RTE/partial randomizationを確認。PDF更新で式番号が変わるので本書は節名で参照する。
- [Lit2: Wan, Berta, Campbell, Randomized Quantum Algorithm for Statistical Phase Estimation](https://arxiv.org/abs/2110.12071)。Phys. Rev. Lett. 129, 030503 (2022)。randomized LCU、平均信号、sample/circuit交換関係を確認。
- [Lit3: Yi and Crosson, Spectral analysis of product formulas for quantum simulation](https://www.nature.com/articles/s41534-022-00548-w)。npj Quantum Information 8, 37 (2022)、Changhao Yi and Elizabeth Crosson。固有状態・固有値・gap条件の関係を確認。
- [Lit4: Casares et al., Theory and practice of Trotter product formulas for quantum chemistry](https://arxiv.org/abs/2606.30741)。SPRINT/GRADE、公開PDF v1、Section IIIとAppendix Fのrandomized平均信号・spectral broadeningの議論を確認。
- [Lit5: Van der Houwen and Sommeijer, Phase-Lag Analysis of Implicit Runge–Kutta Methods](https://doi.org/10.1137/0726012)、および[Papakostas and Tsitouras, High Phase-Lag-Order Runge–Kutta and Nyström Pairs](https://doi.org/10.1137/S1064827597315509)。位相・振幅の異なる次数が古典数値解析で既知であることを確認。全定理との同値性監査はFR-0の対象。
- [Lit6: de Montbrun and Gerchinovitz, Certified Multi-Fidelity Zeroth-Order Optimization](https://arxiv.org/abs/2308.00978)。certified optimizationの一般目的を確認。
- [Lit7: Poiani et al., Optimal Multi-Fidelity Best-Arm Identification](https://arxiv.org/abs/2406.03033)。multi-fidelity selectionの一般目的と費用理論を確認。
- [Lit8: Li, Some Error Analysis for the Quantum Phase Estimation Algorithms](https://arxiv.org/abs/2111.10430)。J. Phys. A 55, 325303 (2022)。不完全な状態、近似unitary、random unitaryに対するresidual・gap・concentration解析を確認。
- [Lit9: Hu and Jin, Quantum Simulation of Non-Unitary Dynamics via Amplitude-Phase Separation](https://arxiv.org/abs/2602.09575)。v2 (2026)。一般nonunitary生成子のCartesian分解とalgorithmic frameworkを確認し、本案とは対象・目的・名称を分離する。

本書の5～6節は既知構成を出発点にした数式整理と独自の導出案であり、これを公開文献の未確認定理として引用していない。新しい命題の優先権、広い系での有効性、資源優位性は未確定。計画した検証を既に実行したとは記載しない。
