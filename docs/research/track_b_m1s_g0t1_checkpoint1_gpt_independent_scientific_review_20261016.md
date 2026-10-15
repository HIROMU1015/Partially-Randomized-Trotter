# Track B／M1-S G0-T1：Checkpoint 1 — T1a解析証明のGPT独立科学レビュー

**レビュー日**：2026年10月16日（JST）  
**担当**：GPT-6  
**レビュー種別・重要度**：重要科学レビュー／Level 1（新しい理論的保証の科学的受理、既知性、部分乱択への移送価値を判断）  
**実施権限**：ユーザーの「レビューを行って」という明示承認に基づく。解析的な紙上検算、固定GitHub資料の読解、関連する公開文献の限定的照合、研究上の推奨まで。新規の科学数値計算・行列数値評価・シミュレーション・compile、T1bの新規証明、GitHub変更は未承認。  
**状態**：`GPT_CHECKPOINT1_REVIEW_COMPLETE_T1A_MATHEMATICS_ACCEPTED_WITH_FIXED_DOMAIN_T1B_LIMITED_STUDY_RECOMMENDED_NOT_AUTHORIZED_MANDATORY_STOP`

> **重要な権限上の区別**：本書は、提出されたT1a証明を**その明示された数学的範囲で科学的に評価**し、次の研究に関する推奨を与える。ユーザーに代わるT1b実施承認、研究契約変更、著者の原本・分類・authorization・markerの書換え、Checkpoint 2やGate 0へのGOではない。既存のmandatory STOPは解除しない。

---

## 0. 最終判断

### 0.1 Checkpoint 1の結論

**① 数学的受理：T1aの主定理を、固定有限次元・有限減衰予算・採択済みC-sec比較モデルという明示された適用域に限って、独立の紙上検算上「妥当」と判定する。確信度：高。**

確認した中心評価は、

\[
\boxed{
|z_N-z_N^{\mathrm{sec}}|
\le \sqrt n\left[
18\tau|\theta|+\tau b_{\mathrm{near}}(\eta)
+2|\theta|\kappa_{\mathrm{far}}(\eta)\bigl(1+\tau(b+b_0)\bigr)
\right],\quad
\tau=N\theta^2\le D_{\mathrm{cap}},\quad |\theta|\le1 .}
\tag{CP1-1}
\]

著者証明の§§3–6について、分散型散逸性、HS収縮、局所Taylor剰余の定数18、非可換な反復のtelescoping、near/farの振動積分の部分積分、HSから対象信号への\(\sqrt n\)の変換を順に検算した。**この範囲では、結論を無効にする代数的欠陥は確認されなかった**。後述の強い制約を留保した受理であり、機械検証済み・数値再現済みという意味ではない。

**② 科学的価値：T1a単独の独立新規性を認定しない。確信度：中〜高。** 本証明は、分岐相関した非CPTPな*coherent first moment*の明確な適用例と、半径損失を許して位相を扱う保証の仕組みとして有用である。一方、技術的骨格は有限次元の分散型散逸性、Taylor評価、Duhamel、secular平均化、oscillatory integrationという確立した理論手段である。現在の\(\sqrt n\)および逆周波数gapの依存を持つ評価を、そのまま「量子化学で資源改善を保証する新定理」や独立の強い論文claimと認定できない。

**③ 研究進行：正式契約A′の次の段階である、限定的なT1b（部分乱択への移送可能性の検証）へ進むことを**科学的に推奨**する。確信度：中。** 成功を予告するものではない。T1aだけで研究を閉じるより、T1bで\(H_D\ne0\)の先導減衰・位相・dressingと利用可能性を判定する方が、現在の研究課題に直接答える。ただしT1b開始には**ユーザーの別途明示的な新規理論実施承認が必須**であり、本レビューはその承認を与えない。

**④ 未認証：T1bの移送、\(\rho^{(H)}_\infty\)の成立、分子系での有用性、RPEのRMS energy error・shot数、native総費用改善、新規アルゴリズムの発明、学術的新規性の確定は、いずれも未達である。**

### 0.2 判断一覧

| 論点 | GPT独立判定 | 強さ・制約 |
|---|---|---|
| 正しい対象law／C-secへの一致 | 合格 | 採択済み共有±、Pauli/involution、IID、HS超演算子空間に限定 |
| \(\mathcal D=-\frac12\mathbb E(\mathcal J_j-\mathcal L)^2\le0\) | 合格 | HS上の自己随伴作用素としての負半定値 |
| \(\|\Phi_\theta-e^{\mathcal G_\theta}\|_{2\to2}\le18|\theta|^3\) | 合格 | 実\(|\theta|\le1\)、保守的な定数。\(\mathcal G_\theta=i\theta\mathcal L+\theta^2\mathcal D\) |
| 任意\(N\)の局所剰余累積 | 合格 | 両方の作用素がHS収縮。\(18N|\theta|^3\) |
| Near/far平均化と\(\kappa_{\rm far}\) | 合格 | 固定有限スペクトル。小gap時は大きくなり得る |
| \(\sqrt n\)係数を含む複素信号上界 | 合格 | state-specificな低費用保証ではない |
| 位相保証 | 条件付き合格 | 参照信号のfloorと\(E<r_*\)が必要。単一信号の主値に限る |
| 単一pair-sum blockでの\(V_k+v_{kk}\) | 条件付き合格 | 当該blockが真に一次元となるときだけ |
| 非可換二準位例 | 解析的整合性を確認 | 非空の保証域の例であり、分子系の実測や資源有利性ではない |
| 同lawのchannelとの比較 | 意味論として妥当 | 比較targetが異なることの証拠。既存channel法に対する普遍的優越ではない |
| T1a単独の新規性 | 未認定 | 一般平均化法の具体化に近い可能性が大きい |
| T1bへ進む研究価値 | **限定的進行を推奨** | ユーザーのT1b理論実施承認が必要、mandatory STOP維持 |

---

## 1. 対象研究とこれまでの決定

M1-Sは、Hadamard型位相信号の二枝で、各乱択gateの**labelを共有し、同じ時系列で符号だけ反転する**方向制御型乱択時間発展である。このbranch間の相関により、平均信号は一般のCPTP平均channelそのものではなく、\(X\mapsto\mathbb E[A_jXA_j]\)というcoherent写像で表現される。

これまでの決定では、方法の外形であるdirectional control、qDRIFT、mirror/palindromic word、共有乱数それ自体を新しい発明として主張しない。M1-Sの新規性候補は、**有限減衰を許容する長時間coherent信号保証**が部分乱択へ実際に移り、位相推定と最終資源に意味を持つかという点にある。

ユーザーが正式採択したG0-T1／案A′は、

\[
\mathrm{T1a}\ \longrightarrow\ \mathrm{Checkpoint\ 1}\ \longrightarrow\
\mathrm{T1b}\ \longrightarrow\ \mathrm{Checkpoint\ 2}
\longrightarrow\ (\mathrm{Gate\ 0を検討})
\]

である。T1aは固定有限次元の完全乱択\(h=\sum_jp_jP_j\)、孤立した\(h\)の固有状態、C-sec主比較モデル、複素信号の絶対誤差と一様\(N\theta^2\)窓に限定される。T1bは、exact deterministic halfと中央Pauli-tailを組み合わせる、固定\(r\)・小step系列に限る。親のParent allocation D＋E、既存のSOURCE/result/marker等の保護、T1a後の非自動GOという制約は変更しない。

本Checkpointでは、**定理の正しさ**、**数学的な保証が何を意味するか**、**先行研究との実質差**、**T1bへ持ち運べる価値**を別々に評価する。

---

## 2. 固定原資料と取得範囲

Repository：`HIROMU1015/Partially-Randomized-Trotter`。**2026-10-16のT1a固定commit** `4e7b18dd2aa14c127bce9e248c63996c62a055cc`、branch `track-b-m1s-g0t1-t1a-theory-20261016`。GitHub接続により文書を取得し、branch HEADが同SHAであることを確認した。

| 原資料 | 固定先 | 今回の扱い |
|---|---|---|
| **T1a証明本文** | [主定理・全証明（4e7b18dd）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md) | **全文§§1–12を取得・確認**。主定理・各補題・非可換例を紙上検算。Git blob `65e5a16898fccdbce8df3df78730c2e736acf54a` |
| **Checkpoint 1資料** | [Decision Packet（同SHA）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_checkpoint1_decision_packet_20261016.md) | **全文§§1–7を取得**。主張と未達・STOPを照合。Git blob `434d1d16de291c0d64ce546640d7a2ed47a49fad` |
| **解析claim inventory** | [解析claim一覧（同SHA）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/analytic_claim_inventory.json) | 固定claim 7件と`PENDING`科学受理状態を確認 |
| **Evidence index** | [証拠索引（同SHA）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/evidence_index.json) | 数値0件、文書のhash/byte記録、停止状態を確認 |
| **実施承認記録** | [T1a authorization（同SHA）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/user_authorization_record.json) | T1a解析研究と資料公開の承認。T1b・新規科学計算は含まれない |
| **公開receipt** | [T1a publication receipt（同SHA）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/publication_receipt.json) | Payload公開後の再取得・旧成果保護のCodex記録。本文の科学的正しさの証拠ではない |
| **S0正式契約** | [373b8e4b固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md) | §§5–17の対象・受理条件・非自動進行を照合 |
| **直前GPT採択レビュー** | [27df63e固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/27df63ede3513ed3982ea4c341823a3f18fa2b83/docs/research/track_b_m1s_g0t1_s0_formal_adoption_gpt_scientific_review_20261016.md) | C-sec採択とT1a受理・T1b scalingの由来。独立検証の代用にはしない |
| **G0-pre文献監査** | [8b7666d固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md) | 近接8論文とのlaw/対象/保証の区別を確認。8論文すべての原PDFを今回は再精読していない |

Codexの索引は証明本文20,153 bytes、SHA256 `6d777495d8e737fd271ed7e239e122bde2fc50772a58c120c1c8a3f67afcc58f`、Checkpoint文書10,939 bytes、SHA256 `240d97238d2535f3d1f310facfc78c90b2c8d82334ddfb9ded105b41717a4b3c`と記録する。本レビューではGitHub固定refの原文を実際に読んだが、**全16ファイルの独立した再hash検査・旧worktreeの現地再検査は行っていない**。監査receiptと科学的受理は別である。

外部研究は、[Günther et al., PRX Quantum 7, 020332 (2026)](https://doi.org/10.1103/ynxb-p2xq)、[Kivlichan–Granade–Wiebe, *Phase estimation with randomized Hamiltonians* (2019)](https://arxiv.org/abs/1907.10070)、[Chen–Huang–Kueng–Tropp, *Concentration for random product formulas*](https://doi.org/10.1103/PRXQuantum.2.040305)、[Campbell, *A random compiler for fast Hamiltonian simulation*](https://arxiv.org/abs/1811.08017)、[Davies, *Markovian master equations* (1974)](https://link.springer.com/article/10.1007/BF01608389)の公開書誌・概要・既存G0-pre調査との対応を確認した。**今回これらの全定理・全付録を独立に再証明したわけではない**。同等結果の不存在は断定しない。

---

## 3. T1aの主張と前提を精密に再記述

Hilbert空間\(\mathcal H\)の次元を\(n\)、Pauli/Hermitian involutionを\(P_j=P_j^\dagger\)、\(P_j^2=I\)、確率を\(p_j\ge0\)、\(\sum_jp_j=1\)とする。

\[
h=\mathbb E_jP_j,\qquad A_j=e^{i\theta P_j},\qquad
\Phi_\theta(X)=\mathbb E_j[A_jXA_j].
\]

\(\Phi_\theta\)は**通常のチャネル\(\mathbb E[A_jXA_j^\dagger]\)ではない**。ただし、\(X\mapsto A_jXA_j\)はHilbert–Schmidt (HS)空間上のunitaryである。この違いが証明の安定性の核心となる。

\[
\mathcal J_j(X)=P_jX+XP_j,\quad
\mathcal L=\mathbb E_j\mathcal J_j,\quad
\mathcal D=-\tfrac12(\mathbb E_j\mathcal J_j^2-\mathcal L^2).
\]

\(\mathcal L\)の周波数集合は、\(h\)の固有値\(e_a\)からなる\(\Omega=\{e_a+e_b\}\)。直交射影\(\Pi_\omega\)を用い、C-secを

\[
\mathcal D_0=\sum_{\omega\in\Omega}\Pi_\omega\mathcal D\Pi_\omega,
\qquad
z_N^{\rm sec}=\langle k|e^{iN\theta\mathcal L}e^{\tau\mathcal D_0}(I)|k\rangle
\]

とする。\(z_N=\langle k|\Phi_\theta^N(I)|k\rangle\)、\(\tau=N\theta^2\)。\(b=\|\mathcal D\|_{2\to2}\)、\(b_0=\|\mathcal D_0\|_{2\to2}\)。異周波数blockについて

\[
\begin{aligned}
b_{\rm near}(\eta)&=\left\|
\sum_{0<|\omega-\nu|\le\eta}\Pi_\omega\mathcal D\Pi_\nu\right\|_{2\to2},\\
\kappa_{\rm far}(\eta)&=\sum_{|\omega-\nu|>\eta}
\frac{\|\Pi_\omega\mathcal D\Pi_\nu\|_{2\to2}}{|\omega-\nu|}.
\end{aligned}
\]

\(\eta\)は**証明上のnear/far分析用cutoff**であり、正式に採択したC-secの周波数blockそのものを変更しない。この区別は文書§2.2で適切に扱われている。

**注意すべき保証の量化**：\(\forall N\in\mathbb Z_{\ge0}\)、\(|\theta|\le1\)、\(N\theta^2\le D_{\rm cap}\)における**固定モデルの一様上界**である。\(n\)、Hamiltonian family、gap、sampling lawを同時に変える場合の一様性は**証明されていない**。

---

## 4. 各補題の独立紙上検算

### 4.1 散逸性とHSノルムの安定性：**正しい**

\(P_j\)がHermitianであるため、\(\mathcal J_j\)はHS上の自己随伴作用素で、\(\|\mathcal J_j\|_{2\to2}\le2\)。平均\(\mathcal L\)も自己随伴で同じ上限を満たす。

\[
\mathbb E(\mathcal J_j-\mathcal L)^2
=\mathbb E\mathcal J_j^2-\mathcal L^2.
\]

**非可換な\(\mathcal J_j\)についても**、\(\mathbb E(\mathcal J_j\mathcal L)=\mathcal L^2=\mathbb E(\mathcal L\mathcal J_j)\)なので等式は成立する。したがって

\[
\mathcal D=-\frac12\mathbb E(\mathcal J_j-\mathcal L)^2\preceq0.
\]

\(\mathcal J_j^2\preceq4I\)と分散の非負性から、\(-2I\preceq\mathcal D\preceq0\)、\(b\le2\)。\(\mathcal D_0\)も直交pinchingにより負半定値であり\(b_0\le b\)。\(\mathcal L\)と\(\mathcal D_0\)は可換。

\(\Phi_\theta=\mathbb E e^{i\theta\mathcal J_j}\)はHS-unitary作用素の凸結合なので収縮する。\(\mathcal G_\theta=i\theta\mathcal L+\theta^2\mathcal D\)について

\[
\mathrm{Re}\langle X,\mathcal G_\theta X\rangle_2
=\theta^2\langle X,\mathcal D X\rangle_2\le0,
\]

よって\(e^{t\mathcal G_\theta}\)も\(t\ge0\)で収縮する。**CPTP性、\(\Phi\)の正規性、左右固有vectorの良条件性を仮定する必要がない**という著者の主張は正しい。

### 4.2 一step局所剰余の定数18：**計算上整合する**

\(|\theta|\le1\)とすると\(\|\mathcal G_\theta\|\le2|\theta|+2\theta^2\le4|\theta|\)。\(\Phi_\theta=\mathbb E e^{i\theta\mathcal J_j}\)の二次Taylor剰余は、\(e^{it\theta\mathcal J_j}\)のunitary性から\(8|\theta|^3/6\)以下。

\(e^{\mathcal G_\theta}\)の二次Taylor剰余も、\(e^{t\mathcal G_\theta}\)の収縮性から\(\|\mathcal G_\theta\|^3/6\le64|\theta|^3/6\)以下になる。二次までの係数は

\[
\mathcal D-\frac12\mathcal L^2
=-\frac12\mathbb E\mathcal J_j^2
\]

により一致する。\(\mathcal G_\theta^2/2\)の三次・四次の残余は

\[
\frac{i\theta^3}{2}(\mathcal L\mathcal D+\mathcal D\mathcal L)
+\frac{\theta^4}{2}\mathcal D^2
\]

で、normは\(4|\theta|^3+2|\theta|^4\le6|\theta|^3\)。したがって

\[
\|\Phi_\theta-e^{\mathcal G_\theta}\|_{2\to2}
\le\left(\frac86+\frac{64}6+6\right)|\theta|^3
=18|\theta|^3.
\]

定数18は**鋭いという主張ではない**。その正しさは、任意\(\theta\)・無限次元・diamond normでの上界を意味しない。

### 4.3 任意Nへの累積：**正しい**

非可換な作用素についてもtelescoping identity

\[
\Phi_\theta^N-e^{N\mathcal G_\theta}
=\sum_{m=0}^{N-1}\Phi_\theta^{N-1-m}
(\Phi_\theta-e^{\mathcal G_\theta})e^{m\mathcal G_\theta}
\]

は恒等的に成立する。\(\Phi_\theta\)と\(e^{\mathcal G_\theta}\)が共に収縮するので

\[
\|\Phi_\theta^N-e^{N\mathcal G_\theta}\|_{2\to2}
\le18N|\theta|^3=18\tau|\theta|.
\]

ここが、以前のレビューで指摘した**可換例にすら残る局所高次誤差**を実際に閉じた箇所である。固定\(N\)の形式級数を長時間へ無条件に延長したものではない。

### 4.4 Near/far平均化：**積分・部分積分とも整合する**

連続解析変数\(t\ge0\)で

\[
Y(t)=e^{-it\theta\mathcal L}e^{t\mathcal G_\theta},
\quad\mathcal D(t)=e^{-it\theta\mathcal L}\mathcal De^{it\theta\mathcal L},
\quad Z(t)=e^{t\theta^2\mathcal D_0}
\]

とする。\(Y'=\theta^2\mathcal D(t)Y\)、\(Z'=\theta^2\mathcal D_0Z\)。収縮性を使い

\[
Y(t)-Z(t)=\theta^2\int_0^tZ(t-u)
\bigl[\mathcal D_{\rm near}(u)+\mathcal D_{\rm far}(u)\bigr]Y(u)\,du.
\]

Near項は\(t\theta^2 b_{\rm near}\)で制御できる。Far項の原始関数\(B(t)=\int_0^t\mathcal D_{\rm far}(u)\,du\)は、異周波数の因子\(e^{i\theta t(\nu-\omega)}\)により

\[
\|B(t)\|\le\frac{2\kappa_{\rm far}(\eta)}{|\theta|}.
\]

部分積分した\(\int_0^tZ(t-u)B'(u)Y(u)du\)には、境界項\(B(t)Y(t)\)のほか、\(+\theta^2\int Z\mathcal D_0 B Y\)と\(-\theta^2\int ZB\mathcal D(u)Y\)が現れる。**二番目にはnear成分を含む全\(\mathcal D(u)\)が必要**であり、原文はその点を正しく守っている。

したがって

\[
\|Y(t)-Z(t)\|\le t\theta^2 b_{\rm near}
+2|\theta|\kappa_{\rm far}\bigl[1+t\theta^2(b+b_0)\bigr].
\]

\(t=N\)とし、§4.3のtelescoping boundを足して原文の(T1a)を得る。\(\theta=0\)は別に厳密一致として扱われている。**積分の振動因子、\(\theta\)の分母、部分積分の符号、収縮性に基づく定数について致命的な取り落としは確認できない**。

### 4.5 HSノルムから対象信号への変換：**正しいが保守的**

正規化された\(|k\rangle\)について、functional\(X\mapsto\langle k|X|k\rangle\)のHS dual normは1。入力恒等作用素\(I\)のHS normは\(\sqrt n\)。よって

\[
|\langle k|(\Phi_\theta^N-e^{iN\theta\mathcal L}e^{\tau\mathcal D_0})(I)|k\rangle|
\le\sqrt n\,\|\Phi_\theta^N-e^{iN\theta\mathcal L}e^{\tau\mathcal D_0}\|_{2\to2}.
\]

固定モデルの\(\eta=0\)では、異なる周波数間の有限個の分母は非零であるため\(\kappa_{\rm far}(0)<\infty\)。\(b_{\rm near}(0)=0\)なので、\(\sup_{N\theta^2\le D_{\rm cap}}|z_N-z_N^{\rm sec}|=O(|\theta|)\)を得る。これは本提出の「全N一様」の意味と一致する。

ただし\(\sqrt n\)と\(\kappa\)は**global HS norm／全spectral block**に依存し、状態\(|k\rangle\)に本当に必要なcouplingだけを数える保証ではない。多量子ビットでは\(n\)がHilbert空間次元として増える。**式として小さくなる極限が存在することと、実用的なgate数で非空の保証が得られることは全く別**である。

---

## 5. 位相系、pair-sum係数、非可換参照例

### 5.1 Signal-floor付き位相系：**正しいがRPE保証ではない**

参照信号\(|z_N^{\rm sec}|\ge r_*>0\)、複素誤差上界\(E<r_*\)のとき、\(w=z_N/z_N^{\rm sec}\)は\(|w-1|\le E/r_*<1\)。複素平面の円板が原点を含まないことから

\[
|z_N|\ge r_*-E,
\qquad
\left|\operatorname{Arg}\frac{z_N}{z_N^{\rm sec}}\right|
\le\arcsin\frac{E}{r_*}.
\]

証明§7の幾何学的論法は正しい。ただし単一複素信号の**主値の相対位相差**の保証であり、多roundのphase unwrapping、shot分散、初期重なり、RMS energy errorは含まない。これを「化学エネルギー推定が既に保証された」と読んではならない。

### 5.2 一次元pair-sum blockでの減衰係数：**成立条件が重要**

追加条件として、\(2e_k=e_a+e_b\)となる組が\((k,k)\)に限られるなら、周波数\(2e_k\)のHS固有部分空間は\(E_{kk}=|k\rangle\langle k|\)の一次元である。このとき

\[
\langle E_{kk},\mathcal D(E_{kk})\rangle_2
=-(1-e_k^2)-\operatorname{Var}_j\langle k|P_j|k\rangle
=-(V_k+v_{kk}).
\]

よってC-sec参照信号は

\[
z_N^{\rm sec}=e^{2iN\theta e_k}e^{-\tau(V_k+v_{kk})}.
\]

**原文の証明はこの十分条件を明示している**。一方、\(h\)の単純固有値であることだけでは、別の\((a,b)\)で\(e_a+e_b=2e_k\)になるpair-sum共鳴を排除できない。真の信号の有限\(N,\theta\)の減衰率、イベント比\(\rho=(V+v)/(2V)\)、化学sector外の共鳴を直ちに保証するものではない。

### 5.3 非可換二準位例：**式は整合するが効果主張は限定される**

\(P_1=Z\)、\(P_2=X\)、\(p_1=p_2=1/2\)では\(h=(X+Z)/2\)。その固有基底で\(h=\widetilde Z/\sqrt2\)と置けば、\(P_{1,2}=(\widetilde Z\pm\widetilde X)/\sqrt2\)。原文の

\[
\mathcal C(A)=\tfrac12\widetilde X A\widetilde X,
\quad\mathcal D(A)=-\tfrac12A-\tfrac12\widetilde X A\widetilde X
\]

は分散式に整合する。二つの対角modeの非共鳴couplingから\(b=b_0=1\)、\(\kappa_{\rm far}(0)=1/(2\sqrt2)\)。\(V_\pm=1/2\)、\(v_{\pm\pm}=0\)。したがって

\[
z_N^{\rm sec}=e^{\pm i\sqrt2N\theta}e^{-\tau/2},\quad
|z_N-z_N^{\rm sec}|\le|\theta|
\bigl(18\sqrt2D_{\rm cap}+1+2D_{\rm cap}\bigr)
\]

となる。\(|z_N^{\rm sec}|\ge e^{-D_{\rm cap}/2}\)なので、十分小さい非零\(\theta\)に**解析的な非空位相保証域**があるという論理は成立する。

また、\(\tau\to\tau_*>0\)では、理想単位半径信号との差は\(1-e^{-\tau_*/2}\)に近づく一方、参照信号との相対位相誤差は0へ向かう。**理想unitary channelに近いことと、減衰を会計した位相が正確であることは異なる**という良い教育的・理論的例になっている。

ただし、\(\theta\)が極端に小さくなることで初めて非空となるdomainかもしれない。**固定\(D_{\rm cap}\)で数学的に非空**であることは、実機向けの許容gate数・必要shot数・量子化学での効率を証明しない。

### 5.4 同じlawのchannel witness：**異なる保証targetの区別として妥当**

\(U_j=\mathrm{diag}(A_j^\dagger,A_j)\)による平均CPTP mapを構成すると、その枝間off-diagonal blockは\(\Phi_\theta\)の作用に一致する。初期ancilla\(|+\rangle\)とsystem\(|k\rangle\)からN回反復したancilla coherenceは\(z_N/2\)。\(\operatorname{tr}(\Phi(X)Y)=\operatorname{tr}(X\Phi(Y))\)という**traceの双線形恒等式**を用いた原文§9.1の導出は正しい（HS随伴と混同しないこと）。

理想controlled unitaryに対する出力ancillaのtrace-norm差は\(|z_N-u_N|\)となるため、**同一lawの全channelを理想unitary channelと比べたdiamond norm**は少なくともこの量である。これは\(\tau_*>0\)で減衰による距離が残る例を与える。

しかし、理想channelへの小さいdiamond距離を要求する手法は、**そもそも位相信号の減衰を許容して資源と交換する設計とは評価対象が異なる**。この下界から「既知のすべてのchannel/RPE解析より優れている」や「総費用が小さい」という結論は出ない。通常の単独枝qDRIFT channelを同じ\(\Phi\)と見なしていない点も妥当である。

---

## 6. 数学的受理の限界：研究への接続前に残る問題

1. **次元・gap一様性がない。** \(\sqrt n\)および\(\kappa_{\rm far}\)は固定分子には有限でも、系サイズ・近縮退・対称sectorの変化に伴い大きくなり得る。化学的精度と実用的な回路深さの両立については無言である。
2. **\(\kappa\)は物理的なstate-specific excitation weightではない。** 各block normの総和という保守的な解析量である。実分子に適用するときに、その計算可能性・情報アクセス・有用な上界かを別途評価する必要がある。
3. **Signal-floorは一般には存在を証明していない。** 明示した条件付き幾何学的系と非可換二準位例にはfloorがあるが、多準位・pair-sum共鳴による相殺ではC-sec信号が零近傍になり得る。
4. **減衰・位相に関する結果とRPEの全体保証は異なる。** Shot noise、round間correlation、複数周波数のspectral weight、phase unwrapping、ground-state overlapは未検証。
5. **完全乱択の\(h\)固有stateは、部分乱択の全\(H\)固有stateではない。** 後段でそのまま\(V_k,v_{kk}\)を置換できない。\(\rho^{(H)}_\infty\)は未証明。
6. **状態・対象lawの選択は既に契約で制限されている。** \(\Phi^N\)の作用素norm評価自体は、固定モデルについてより一般の入力にも適用できるが、この事実だけで一般guiding-state RPEまで正式scopeを拡張しない。
7. **Analytic exampleは独立の数値検証ではない。** 元の非可換二準位モデルの紙上整合例である。新たな実分子数値・別系へのtransfer testは行われていない。
8. **上界の保守性と正しさを混同しない。** 定数18や\(\sqrt n\)は証明された上界であって、必要イベント数や実際のphase biasの鋭い予測式ではない。

以上は**正しい限定定理の重要な限界**であり、現時点でT1aの数学的受理を否定する欠陥ではない。ただし、これを除いた一般保証や実用保証の受理は拒否する。

---

## 7. 先行研究と新規性の評価

### 7.1 直接の既知部分

| 先行研究・理論 | 既知の中心的内容 | T1aとの関係・今回認められる差分 |
|---|---|---|
| Campbell 2019／qDRIFT | ランダム指数gate、平均unitary/channel誤差の解析 | Pauli lawとrandom product自体は既知。共有±というcoherent branch targetの解析は別の条件を要する |
| Günther et al. 2026／PR RTE | 部分乱択RTE、位相推定、対称PFの制御と資源会計 | M1-SはPRの論文記述を反証するものではない。T1aは\(H_D=0\)の補題なので、PRのpartial RTEを既に改良した証拠ではない |
| Kivlichan–Granade–Wiebe 2019 | ランダムHamiltonianのphase estimation、gapや状態分散条件 | State-dependent phaseの既知結果が存在する。\(V_k,v_{kk}\)の存在だけで新規性を主張しない。今回のgate共有first momentの保証とはモデル・観測対象が異なる |
| Childs–Ostrander–Su／randomized symmetric PF | 順序の乱択・対称word、平均化による誤差制御 | 回文構造・共有順序の新規性はない。今回のIID lawと信号対象との同一性は未証明 |
| Chen–Huang–Kueng–Tropp 2021 | Rand PFの単一実現・高確率集中、channel誤差 | T1aは単一実現の集中ではなく、branch間相関を持つ**無条件first moment**の保証。比較している保証の型が異なる |
| Davies／Kato等の平均化・摂動法 | 有限次元・弱結合・振動平均・スペクトル分離を使う標準理論 | T1aの証明機構は標準的方法に近い。これらの一般方法を発明したという主張は不適切 |

G0-preではMLMCによる乱数共有・coherent branch配置、SCUのcross term、COSのpalindrome等についても既存知見として報告されている。**乱数共有・coherent測定・回文が既知でないという主張は退ける。** 本レビューでは8論文の全本文を独立に精読したわけではないため、同等の定理が存在しないという排他的な新規性認定も行わない。

### 7.2 T1aの科学的意義と限界

- **数学的な価値**：非CPTPの平均coherent写像をHS-unitary ensembleとして扱い、長時間\(N\theta^2=O(1)\)で局所三次累積とsecular平均化を共に明示する、追跡可能な補題を得たことは有意義である。前回S0レビューで要求した抜けた局所誤差を、実際の定数付きで閉じた。
- **方法としての新規性**：低い。Proof techniqueは教科書的／標準的な有限次元作用素平均化の組合せである。
- **独立定理としての新規性**：現時点では未認定。係数付きの明示式が文献にそのまま見当たらないことだけでは、論文の主貢献として新しいとは断定できない。
- **PR研究の中での情報価値**：**中程度**。完全乱択でcoherent位相信号を減衰付きで制御できると確認できたため、非可換決定論部を含むと同じ機構が消えるか残るか、という次の問いに数学的に進む土台が整った。
- **実分子／資源価値**：**未確立**。\(\sqrt n\)、\(\kappa\)、signal floor、分子stateと費用の実際の対応が残る。

独立論文の核とする候補は**「部分乱択へ移送したcoherent phase／attenuationの保証」**であり、T1aの\(O(|\theta|)\)という一般形だけではない。T1bが失敗した場合、T1aは内部理論記録・限定的technical noteとして扱うのが妥当で、無条件に大型論文テーマを正当化しない。

---

## 8. Checkpoint 1：T1bに進む科学的合理性

### 8.1 進行により新しく分かること

T1aは、**決定論部のない完全乱択**について、\(\Phi_\theta^N(I)\)を固定C-secで近似する一様保証を得た。しかし、部分乱択では既存の厳密恒等式

\[
\Psi(X)=D\Phi_\theta^r(DXD)D,
\qquad D=e^{i\alpha H_D},
\]

だけでは\(\Psi\)の有効mode、長時間の減衰・位相、および\(\rho^{(H)}\)を制御できない。**T1bはこの差を埋められるか**という新しい情報を与える。

期待できる具体的な情報は、(i)\(H_D\ne0\)を入れたときHSの安定性と二次共分散の機構が利用可能か、(ii)全\(H\)固有stateを基準とする有限減衰のsecular解析が閉じるか、(iii)dressing・小分母・outer symmetric-step biasが先導係数をどの程度変えるか、(iv)\(\rho^{(H)}_\infty\)の自然な適用域が存在するか、である。これはT1aの単純な再証明ではなく、研究の価値を左右する未解決の問いである。

### 8.2 T1aから**自動的に**移せないこと

T1aの\(\mathcal J_j(X)=P_jX+XP_j\)と\(\mathcal D=-\frac12\mathbb E(\mathcal J_j-\mathcal L)^2\)は**single-event average**の構造である。T1bの一stepには複数乱択gateと決定論halfを含む。従って、同じ定数18、同じ\(V_k+v_{kk}\)、同じ近共鳴gap、同じsignal floorが保たれるとは仮定できない。

具体的に必要なのは、少なくとも次の理論課題である。

1. **Step-level mapの正確な小step展開**：採択済み\(\Psi\)の順序・時間・\(r\)を保ち、一次の全\(H\)生成子、二次の散逸成分、反Hermitianな位相・分割補正を区別する。平均mapの存在だけで散逸率を同一視しない。
2. **非可換な長時間累積**：T1aの収縮・telescoping・interaction pictureを使用できるか、\(\delta_{\rm rel}\to0\)、固定\(r\)、\(s\delta_{\rm rel}^2=O(1)\)で誤差定数と周波数分離を制御する。
3. **Outer Strang-type splitting bias**：exactな\(H_D\) halfであっても、\(H_D\)と\(H_R\)が非可換なら分割誤差は消えない。長時間の累積\(O(s\delta_{\rm rel}^3)\)候補を平均化の\(O(\delta_{\rm rel})\)と区別する。**原理的な項の列挙であり、T1bの新しい定理を今回証明したわけではない。**
4. **全\(H\)固有mode／dressing／alias**：\(h_R\)の固有stateで計算した完全乱択の係数を、全\(H\)固有stateへそのまま代入しない。\(DXD\ne DXD^\dagger\)、有効modeの左右weight、比の分母、周波数aliasとpair-sumを扱う。
5. **対照モデルとの同一task性**：ordinary、独立±、gate共有±、既存PR-RTEで同じ理想物理時間と情報アクセスを固定し、**減衰係数の比**と**同精度総費用**を別々に記録する。

### 8.3 研究の早期STOP条件

T1bに進む場合も、次の場合に成果を誤認せず段階を区切ることを推奨する。

- 正しい\(\Psi\)から得られる一様評価が、\(\sqrt n\)や逆gapの大きさにより**実際に適用したい分子系列で空になる**：理論上の成立と化学資源価値を分離し、初期scopeを再評価する。
- 有効modeやdressingの定義・分母が自然な条件で閉じない：\(\rho^{(H)}\)というratio claimは限定／断念し、絶対複素信号保証だけの価値を別に検討する。
- 既存の平均化／ランダムHamiltonian phase理論の完全な直接系であり、**部分乱択固有の定量的帰結が無い**：独立論文の優先度を下げ、technical note化／M2などへの資源移行を検討する。
- 実機上の実装可能性・価格が未取得：未認証\(\rho\)や旧データの費用を使って数値GOを出さない。T1b理論とGate 0・Q4/Q5は段階的に扱う。

**結論：T1aの数学的基盤が成立したことは、限定T1bの研究に進むための必要な理由を与えるが、科学的新規性や資源利益を既に保証したものではない。** T1bを一回の限定理論研究として実施する価値はある。研究を自動的に拡大する根拠にはならない。

---

## 9. 正式なCheckpoint 1科学的推奨

### 9.1 判定ラベルと確信度

| 段階 | レビュー判断 | 確信度 |
|---|---|---|
| **T1a数学的受理** | **`ACCEPTED_WITH_FIXED_DOMAIN`** | **高**：主補題・主上界を紙上で検算 |
| T1aの実用保証 | `NOT_ESTABLISHED` | 高：分子とRPE・shotを未検証 |
| T1a単独の独立新規性 | `NOT_CERTIFIED` | 中〜高：一般平均化技術への依存、先行研究の非網羅性 |
| **Checkpoint 1研究進行** | **`RECOMMEND_LIMITED_T1B_THEORY_AFTER_SEPARATE_AUTHORIZATION`** | **中**：T1bに未解決の中心問題がある |
| T1b移送・改善比 | `UNPROVED` | 高：T1a対象外 |
| 研究全体の方向転換・終了 | `NOT_ADOPTED` | 中：現段階でM1-Sを全面STOPする根拠はない |
| 実施承認 | `NOT_GRANTED` | 確定：本レビューは承認行為ではない |
| mandatory STOP | `MAINTAINED` | 確定 |

### 9.2 不採用とした判断

**「T1aの全N一様定理が完成したから、M1-Sの方法が新規でRPE資源改善を保証した」**という判断は不採用。数学的対象・観測量・適用域が異なる。

**「定理の道具が標準的だからT1bは不要」**という判断も不採用。部分乱択への移送可否こそ研究上の未解決事項であり、適切に限定した理論検証の情報価値が残る。

**「\(\sqrt n\)や小gapによる保守性だけでT1aを数学的に不受理」**という判断は不採用。数学的上界としての成立性と、物理系で強い保証として役立つかは別の質問である。

**「未承認のH4／H6数値、RPE/compileでT1b前の不確実性を解消する」**という手順は不採用。現在の理論contractと科学計算承認を混同する。

---

## 10. 次の担当、具体的な範囲、別途承認

**次の直接の担当：ユーザー（T1bの実施承認判断）。承認された場合の技術・理論主担当：Codex。重要な科学的判断：必要な節目でGPT。**

推奨する次の一作業は、**正式S0契約§§8–10に基づくT1bの限定的な解析的移送検証**である。最初から一般DF event、実装PF、RPE、native compile、全Fock分子数値へ拡大しない。

研究実施を別途承認する場合は、既存正式契約の次を守る。

1. **入力固定**：S0 `373b8e4b...`、T1a `4e7b18dd...`、このCheckpoint 1レビュー原本（公開される場合は固定commit）を参照する。証明本文・既存結論を上書きしない。
2. **対象law**：exact deterministic half＋中央Pauli-tail、gate共有±、固定\(r\ge1\)、\(\delta_{\rm rel}\to0\)、\(s\delta_{\rm rel}^2\)有界の解析系列。同じ物理\(T_{\rm rel}\)で比較。
3. **提出物**：\(\Psi\)の有効生成子、散逸／位相・outer-step biasの分離、全\(H\)modeでの適用条件、dressing・小分母・alias、\(\rho^{(H)}\)が成立／修正／適用外／未解決の区別。**正しい有限次元条件付き部分結果や理論的障害・反例も記録する**。
4. **早期STOP**：根拠のない係数移送、既存定理の単純な再表示、定数が空になる保証、非認証proxyの事後的採用、必要条件が満たせない事例を区別。曖昧な数値GOを作らない。
5. **禁止**：新規科学数値計算、分子H4/H6の期待値や行列・スペクトルの数値評価、simulation/sampling、回路生成・compile・資源forecast、Q4/Q5、一般DF event、T1bからCheckpoint 2／Gate 0への自動進行。これらは今回も未承認。
6. **証拠・停止**：原資料・定式化・証明／不成立条件・監査材料を公開し、mandatory STOP。**Checkpoint 2の重要GPTレビューは、その時点のユーザーの別途開始承認を得てから実施**する。

T1bの**新規理論実施はこのレビューの範囲外**であり、上記は次の研究契約に基づく推奨でしかない。次のユーザー依頼が「次の段階は？」であれば必要な承認事項を示して止まり、明示的なT1b実施承認があれば既定scopeでCodexへ移す。

---

## 11. 本レビューの証拠強度と未実施事項

- **独立に検算した**：HS分散恒等式、散逸性・収縮、局所定数18、非可換telescoping、near/farの積分・部分積分、\(\sqrt n\)因子、floor付き位相系、単一pair-sum係数、非可換二準位モデルの表示とchannel witnessの数学的意味。
- **原資料で確認した**：T1aの固定law、解析定理・提出範囲、著者の`PROVED_WITH_DOMAIN`、証拠索引とpublication receipt、S0のCheckpoint条件、T1b未承認・mandatory STOP。
- **限定的に照合した**：PR、KGW、Campbell、COS、Chen等の先行研究の書誌・概要と、既存G0-preの対象／保証比較。**世界初性・先行全証明の再現・網羅的新規性調査は未実施**。
- **実施していない**：新たな数値行列評価、科学simulation、アルゴリズムの実装、外部の形式証明assistantによる機械検証、保存データの再集計、T1bの新定理構築、T1bの科学的GO、分子上の資源評価、repositoryやauthorization／markerの変更。
- **未確立**：非自明な分子familyに対する規模一様性、常に正のsignal floor、部分乱択\(\rho^{(H)}\)、RPE/shot・total resource、独立の学術的新規性。

### 11.1 最終的な研究判断

**T1aの解析保証は、固定有限次元・有限減衰予算・C-secの条件下で数学的に受理できる。** しかしその結果単独では量子化学の資源改善も独立論文の核も確立しない。**次に必要な科学的情報は、採択済みT1bの部分乱択への移送の可否である**。したがって、**ユーザーの別途明示承認を前提としてT1bの限定的理論研究へ進むことを推奨する**。実行許可は付与せず、研究全体の既存決定とmandatory STOPを維持する。

---

## 12. 参照先

### GitHub固定資料

- T1a proof: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md>
- Checkpoint 1 Decision Packet: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_checkpoint1_decision_packet_20261016.md>
- T1a analytic claims: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/analytic_claim_inventory.json>
- T1a evidence index: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/evidence_index.json>
- S0 formal contract: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md>
- G0-pre literature audit: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md>

### 外部研究（今回の確認範囲は§2を参照）

- J. Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332 (2026), <https://doi.org/10.1103/ynxb-p2xq>.
- I. D. Kivlichan, C. E. Granade, N. Wiebe, *Phase Estimation with Randomized Hamiltonians* (2019), <https://arxiv.org/abs/1907.10070>.
- E. Campbell, *A Random Compiler for Fast Hamiltonian Simulation*, Phys. Rev. Lett. **123**, 070503 (2019), <https://arxiv.org/abs/1811.08017>.
- C.-F. Chen, H.-Y. Huang, R. Kueng, J. A. Tropp, *Concentration for Random Product Formulas*, PRX Quantum **2**, 040305 (2021), <https://doi.org/10.1103/PRXQuantum.2.040305>.
- E. B. Davies, *Markovian Master Equations*, Commun. Math. Phys. **39**, 91–110 (1974), <https://link.springer.com/article/10.1007/BF01608389>.

---

## 13. 最終監査

- [x] ユーザーの今回のGPTレビュー開始承認を確認。
- [x] 固定T1a証明・Checkpoint 1・解析claim・監査index・承認・receiptをGitHub接続で実際に取得。
- [x] 正式S0と前回GPTの対象・比較モデルを照合。
- [x] 主補題・主定理・位相系・解析例を紙上で独立検算。
- [x] 数学的受理と規模一様性・RPE・総費用・論文新規性を厳密に分離。
- [x] 継続・縮小・technical note・STOPの選択肢を検討。
- [x] 新規科学数値計算・シミュレーション・回路生成・compile・未承認T1b新証明・GitHub変更なし。
- [x] Checkpoint 1の科学的推奨は、ユーザーによるT1b実行承認の代替ではない。
- [x] mandatory STOP、Parent allocation D＋E、既存SOURCE/result/authorization/markerを維持。
- [x] レビュー完成版のMarkdownの存在・内容を確認し、チャットではリンクのみを提出する。
