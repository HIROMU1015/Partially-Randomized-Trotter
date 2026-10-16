# Track B／M1-S G0-T1：Checkpoint 2 GPT独立科学レビュー
## T1bの数理的受理・Claude新補題の検証・等イベント比較・Gate 0の科学的採否・研究の着地点

- **レビュー日**：2026-10-16（JST）
- **担当**：GPT-6
- **重要度**：Level 1（高）。新しい科学的保証と研究継続／縮小／論文化の判断
- **開始承認**：ユーザー「レビューを開始して」（本Checkpoint 2への明示的承認）
- **実施範囲**：固定GitHub原資料、ユーザー提供Claudeレビューの独立読解、紙上の数理・論理検証、限定した一次文献照合、次の科学的作業の推奨、Markdown記録
- **明確に未承認**：新規の科学数値計算、Hamiltonian／行列の数値評価・対角化、サンプリング、simulation、回路生成・compile、保存結果の新規科学再解析、Gate 0／E0、T1c・Q4・Q5・一般DF eventの新規研究、GitHubの改変
- **状態**：`GPT_CHECKPOINT2_SCIENTIFIC_REVIEW_COMPLETED_T1B_ACCEPTED_WITH_FIXED_DOMAIN_JENSEN_FLOOR_ACCEPTED_EQUAL_EVENT_COMPARISON_CONDITIONAL_CLAUDE_GLOBAL_STOP_RULE_NOT_ACCEPTED_LIMITED_GATE0_PREREG_RECOMMENDED_NOT_AUTHORIZED_MANDATORY_STOP`

> **権限上の結論**：本書はGPTの独立した科学的評価・推奨である。Claudeの推奨をユーザーによる正式採択へ変更しない。T1bの著者原本・過去の科学SOURCE／結果／classification／authorization／markerを書き換えず、**mandatory STOPを維持する**。科学数値計算は0件。論文・研究契約・新規計算の正式な採択はユーザーが別途行う。

---

## 0. 要旨と最終判断

### 0.1 数学的な受理

**T1b主定理「固定有限次元・固定整数 \(r\)・小step \(\delta\to0\)・有限減衰予算で、全 \(H\) のC-sec複素信号に全 \(s\) 一様な \(O(|\delta|)\) 誤差で近づく」について、提示された適用域に限定して数学的に受理する。確信度：高（紙上検証）。**

著者の正確な二枝平均写像、二次生成子と散逸性、局所三次剰余、非可換なtelescoping、near/far振動平均化、Riesz射影を使う条件付きmode分離、先導減衰係数、有限stepの位相補正、outer splittingの会計について、中心結論を破壊する代数的欠陥を確認しなかった。ただし、形式証明・機械検証・数値再現は行っていない。

**Claudeの追加主張であるJensen型signal floorは、全 \(H\) の対象固有値 \(E_k\) が非縮退なら、pair-sum共鳴があってもC-sec参照信号に対して成立すると独立に確認した。確信度：高。** 重要な強化であり、(SP)は単一指数減衰の**等式**には十分だが、下界・参照信号の理想位相保持には必要ない。ただし、有限stepの真の平均信号への保証にはT1bの誤差上界が別途必要である。

### 0.2 Claudeレビューとの主要な一致・不一致

| 論点 | GPT独立判定 | 確信度・制限 |
|---|---|---|
| T1bの一様C-sec主定理 | **受理** `ACCEPTED_WITH_FIXED_DOMAIN` | 高。固定model・固定\(r\)、逆pair-sum gap／\(\sqrt n\)依存を持つ |
| 先導\(W_k+v_k\)、三次位相・dressing、(SP)下の\(\rho\) | **受理**、限定条件を維持 | 高。全\(H\)固有状態、指定law、正の分母が必要 |
| 非縮退\(E_k\)で共鳴を許すJensen floor | **受理** | 高。C-sec上の下界と厳密な参照位相であり、真の信号には誤差控除が必要 |
| 同一**乱択イベント数**でのshared／independent／ordinary係数比較 | **代数的に受理、資源解釈は条件付き** | 高／中。制御gate単価、実装、固定stepでの\(r\)変更に注意 |
| 三準位共鳴例の単一rate反例 | **受理（限定）** | 高。符号違いの同一involutionを重複させたlawであり、非冗長なPauli分解一般の反例ではない |
| Claude §20・付録A.9の「\(\rho\)だけで全資源節約率に上限を与え、STOP可能」 | **一般的なSTOP規則としては不採用** | 高。shot数・contrast・bias・native単価が共通であるという追加仮定が欠ける |
| H4中心の目的限定Gate 0 | **事前登録・入力source確認の後に実施を条件付き推奨** | 中。新規科学計算は別承認。\(\rho\)はheadroom診断に留める |
| T1c「有限\(\alpha\)／固定\(\delta\)・\(r\)変更」の追加理論 | **重要な候補だが自動開始しない** | 中。小step理論と実際のPR運用との差を埋める必要が生じた場合 |
| 論文化の着地点 | **限定理論成果を確保し、拡張可能性はGate 0等で決める** | 中。独立新規性と量子化学資源優位は未認証 |

### 0.3 研究方針の最終推奨

**推奨する基本線は「C：T1a＋T1bの限定理論成果として保全」＋「対象と判断基準を先に固定した、少数系のGate 0 headroom診断を条件付きで検討」である。** Claudeと方向性は近いが、**Claudeの\(\rho\)しきい値による総資源no-go／自動STOPの論理は採用しない**。Gate 0での肯定的／否定的な\(\rho\)は、採択済み小stepモデルの**oracle診断**であり、RPEのshot・実装単価・固定step運用の総費用を直接決めない。

新しい学術的主張の中心候補は、「**Pauli/involution-tail、branch-shared ±、exact deterministic halfという明示的lawにおける、減衰を許容したcoherent平均信号の長時間secular保証、共鳴時の正のfloor、係数比の成立域、位相補正の分離**」まで。これを「PR一般の資源半減」「RPE全体の改善保証」「新しい回路構成の発明」へ拡張しない。

---

## 1. 研究全体での位置付け

本研究は部分ランダム化Trotter法（PR）を用いる量子化学のエネルギー推定で、精度を維持しつつ量子資源を減らすことを中心目的とする。M1-Sは、正負Hadamard分岐の同じgate位置でPauli labelを共有し、各gateの回転符号を反転させたとき、平均coherent信号を通常の独立乱択と異なる写像で制御できるかを調べる限定理論課題である。

採択済みG0-T1／A′は「T1a完全乱択 → Checkpoint 1 → T1b部分乱択 → Checkpoint 2 → 必要ならGate 0」という順序。T1aは前回GPTが固定有限系・有限減衰領域で限定数学的受理した。T1bは今回Codexが提出し、Claudeが独立レビューを完了した。**Checkpoint 2において研究として継続する価値は、単に定理が正しいかではなく、PRの実用条件と強いbaselineへ意味のある形で接続できるか**によって決まる。

Parent allocation D＋E、他Trackとの責任分界、SOURCE・過去結果・authorization・mandatory STOPは今回変えない。

## 2. 固定原資料・取得状況・証拠の水準

### 2.1 GitHub固定資料

Repository：`HIROMU1015/Partially-Randomized-Trotter`。

| 固定原資料 | Commit／参照 | 今回の取得・扱い |
|---|---|---|
| **T1b主定理・全証明** | [`54336028…` 証明本文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md) | GitHubから取得、§§1–17の必要数式を照合。Git blob `60a43265eebe2c9086f38ace38076170eb91b9df` |
| **T1b Checkpoint 2 Packet** | [同commitのDecision Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_checkpoint2_decision_packet_20261016.md) | 取得、科学的に未認証と著者自ら区別する箇所を確認。Git blob `dfe9e6479c2bf027f10e4c9869c6e96f06758153` |
| **T1b analytic claim inventory** | [claim一覧](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/analytic_claim_inventory.json) | 取得。著者区分`PROVED_WITH_DOMAIN_PLUS_COUNTEREXAMPLE_IN_DEFINED_CLASS`を確認 |
| **T1b evidence index** | [証拠索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/evidence_index.json) | 取得。新科学数値0、独立受理pending、mandatory STOPを確認 |
| T1b実施承認 | [authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/user_authorization_record.json) | 取得。許可はT1b解析研究であり今回のGate 0ではない |
| **S0正式契約** | [`373b8e4b…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md) | 取得。§§8–17の対象・固定\(r\)系列・STOPを確認 |
| T1a証明 | [`4e7b18dd…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md) | 取得。T1aの平均化構造を確認 |
| Checkpoint 1 GPT原本 | [`4899608e…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4899608ea222777e64e706fda2b7e54b84c86247/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md) | 取得。前回の適用域付き受理・T1b推奨を引継ぎ |
| G0-pre 文献・protocol監査 | [`8b7666d0…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md) | 取得。既知word／law／費用の区別を確認 |

### 2.2 Claude独立レビュー

ユーザー提供ファイル：`レビューinner(2).md`。原本SHA256：`c4fdf64d1acd4b21a7abfd42b3c069495ce167f0e64ec1526f53afdf9af7f379`。レビュー日2026-10-16。Claude status：`CLAUDE_CHECKPOINT2_REVIEW_COMPLETED_T1B_ACCEPTED_WITH_DOMAIN_RECOMMEND_C_PLUS_LIMITED_GATE0_NOT_ADOPTED`。§§0–28、付録A.1–A.9等を確認した。**Claudeの「独立検証」とGPTによる本レビューの検証は別の証拠階層**であり、Claudeの受理を数学的真実の根拠として代用しない。

### 2.3 外部文献の今回の独立照合範囲

- [Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332 (2026)](https://doi.org/10.1103/ynxb-p2xq)：出版社の公開書誌・abstractを確認。PRが部分乱択・single-ancilla phase estimation・量子化学資源評価を扱うことは一次情報に基づく。全文の全定理を今回再証明していない。
- [Campbell, *A Random Compiler for Fast Hamiltonian Simulation*, PRL **123**, 070503 (2019)](https://doi.org/10.1103/PhysRevLett.123.070503)：公開abstractを確認。Pauli係数の重みに応じた乱択qDRIFT自体は既知。
- [Yi–Crosson, *Spectral analysis of product formulas for quantum simulation*, npj Quantum Information **8**, 37 (2022)](https://www.nature.com/articles/s41534-022-00548-w)：公開本文・abstractで有効Hamiltonianのスペクトル摂動、状態・位相分離、QPEへの接続の先行性を確認。今回の非CP分岐相関写像との同一性は主張しない。
- [Morisaki–Fujii, *Optimized Randomized Hamiltonian Simulation via Average-Error Analysis*, arXiv:2609.36694 (2026-09-29)](https://arxiv.org/abs/2609.36694)：公開abstractおよび公開要約を確認。乱択生成子の分散、状態／信号error、sampling lawの最適化に近接する。**Claudeの主要比較一覧には含まれておらず、論文化前の追加の一次本文照合対象とする**。非CPのbranch-shared C-secと同じ定理であるとの断定はしない。
- [Chen–Huang–Kueng–Tropp, *Concentration for Random Product Formulas*, PRX Quantum **2**, 040305 (2021)](https://doi.org/10.1103/PRXQuantum.2.040305)、[Kivlichan–Granade–Wiebe, *Phase estimation with randomized Hamiltonians*](https://arxiv.org/abs/1907.10070)、[Childs–Ostrander–Su, *Faster Quantum Simulation by Randomization*](https://arxiv.org/abs/1805.08385)は既存G0-preの比較を参照。今回それぞれの全文全証明を独立に読み直したものではない。

**制限**：文献調査は非網羅。形式証明、フルソース実装の実行、全18公開ファイルの独立byte再hash、親dirty worktreeの現地監査はしていない。Codexの監査receiptと今回の数学的受理を混同しない。

---

## 3. 平均写像・散逸性・局所剰余の独立検証

### 3.1 分岐相関の厳密写像

\[
H=H_D+H_R,\quad H_R=\lambda h,\quad h=\sum_jp_jP_j,\quad
P_j^\dagger=P_j,\ P_j^2=I,\quad \sum_jp_j=1.
\]
\[
\theta=\frac{\lambda\delta}{2r},\quad D=e^{i\delta H_D/4},\quad
\Phi_\theta(X)=\mathbb E_j[e^{i\theta P_j}Xe^{i\theta P_j}].
\]

右端先作用、同じ時系列labelを両枝で共有、各branchでgate符号を反転、位置間・step間IIDとすると、\(G_-^\dagger XG_+\)の入れ子構造を各位置から平均できるため

\[
\boxed{\Psi_\delta(X)=D\Phi_\theta^r(DXD)D}\tag{3.1}
\]

となる。これは**共通label**と**IID**の使い方が明確な式で、\(DXD^\dagger\)というCPTP共役ではない。HS上では二側unitary乗算はnorm保存。従ってその平均\(\Psi_\delta\)はHS contractionである。相対時間\(T_{\rm rel}=s\delta\)と各branchの理想半時間\(s\delta/2\)は整合する。**受理**。

### 3.2 二次生成子の係数

HS上で \(J_X(Y)=XY+YX\) とし、\(J_j=J_{P_j}\)、\(L_R=\mathbb EJ_j\)、\(J_D=J_{H_D}\)。

\[
B=J_D/4,\quad C=\lambda L_R/2,\quad A=2B+C=J_H/2,
\]
\[
D_R=-\tfrac12\mathbb E(J_j-L_R)^2\preceq0,
\qquad\beta=\frac{\lambda^2}{4r},\quad Q=\beta D_R\preceq0.
\]

各superoperatorはHS自己随伴なので分散恒等式が成立する。\(\Phi_\theta^r\)の二次係数は\(Q-C^2/2\)、左右の\(e^{i\delta B}\)を掛けた二次係数は交差\(BC+CB\)を含めて\(Q-A^2/2\)。従って

\[
\Psi_\delta=I+i\delta A+\delta^2(Q-A^2/2)+O(\delta^3).
\]

ここで二次logに交差commutatorが残らないことは**対称配置の性質**であり、一般に\([H_D,H_R]=0\)を仮定したからではない。\(Q\)は負半定値、\(i\delta A+\delta^2Q\)は散逸的なので\(F_\delta=e^{i\delta A+\delta^2Q}\)も収縮。係数と符号は整合する。

### 3.3 明示的な局所剰余

\(d=\|H_D\|\)、\(m=d+\lambda\)、\(a=\|A\|\)、\(q=\|Q\|\)とし、固定\(\delta_0>0\)、\(|\delta|\le\delta_0\)で、著者は

\[
K=\frac{m^3}{6}+\frac{(a+\delta_0q)^3}{6}+aq+\frac{\delta_0q^2}{2}
\]

として\(\|\Psi_\delta-F_\delta\|_{2\to2}\le K|\delta|^3\)を示した。

各実現のunitary積を\(\delta\)で三回微分すると、Leibnizのmultinomial評価はgenerator norm和\(2\|B\|+r|\lambda/(2r)|\max\|J_j\|\le m\)の三乗で制御できる。\(F_\delta\)側のTaylor剰余は収縮性と\(\|i\delta A+\delta^2Q\|\le|\delta|(a+\delta_0q)\)から得られる。三次・四次交差の残りは\(aq+\delta_0q^2/2\)。**局所定数の作り方を独立に点検し、整合していると判断する**。T1aの定数18の無条件転用ではない。

## 4. 全\(s\)一様C-sec主定理の検証

全\(H\)のHS pair-sum作用素\(A\)の周波数は\(\omega=(E_a+E_b)/2\)。直交射影\(\Pi_\omega\)を用い、\(Q_0=\sum_\omega\Pi_\omega Q\Pi_\omega\)、\(u=s\delta^2\)とする。

\[
z_s^{H,\rm sec}=\langle k|e^{is\delta A}e^{uQ_0}(I)|k\rangle.
\]

\(Q\)の厳密に異なる周波数間のcouplingをcutoff \(\eta\)でnear/farに分け、\(q_n\)と\(\kappa_f\)を定義したとき、著者の定理は

\[
\boxed{|z_s-z_s^{H,\rm sec}|\le\sqrt n\left[Ku|\delta|+u q_n+2|\delta|\kappa_f\{1+u(q+q_0)\}\right].}\tag{4.1}
\]

**独立検証の要点**：

1. \(\Psi_\delta\)、\(F_\delta\)が収縮するため、非可換なtelescopingにより\(\|\Psi_\delta^s-F_\delta^s\|\le sK|\delta|^3=Ku|\delta|\)。
2. Interaction picture \(Y(t)=e^{-it\delta A}e^{t(i\delta A+\delta^2Q)}\)は\(Y'=\delta^2Q(t)Y\)、\(Q(t)=e^{-it\delta A}Qe^{it\delta A}\)を満たす。\(Z(t)=e^{t\delta^2Q_0}\)も収縮。
3. Duhamel中のnear項は\(u q_n\)。Far項のprimitiveのnormは\(2\kappa_f/|\delta|\)以下。部分積分では境界項のほか\(+\delta^2ZQ_0VY\)と\(-\delta^2ZVQ(t)Y\)の二項が生じる。後者には**nearも含む全\(Q(t)\)**を使用し、著者は落としていない。
4. 入力\(I\)のHS normは\(\sqrt n\)、出力functional \(X\mapsto\langle k|X|k\rangle\)のHS dual normは1。従って(4.1)が導ける。
5. 固定有限\(H,\lambda,r\)では\(\eta=0\)とでき、異周波数差は有限集合上の非零数。\(q_n=0\)、\(\kappa_f<\infty\)、\(u\le D_{\rm cap}/\beta\)なので**全整数\(s\ge0\)に一様な\(O(|\delta|)\)**を得る。

**判定：`ACCEPTED_WITH_FIXED_DOMAIN`（高）**。ただし\(\sqrt n\)、\(\kappa_f\)（逆pair-sum gapを含む）、\(1/\beta\)を持つので、Hilbert次元、近共鳴、\(r\)を同時に増やすfamilyで同じ上界が機能する保証はない。分子で具体的な定数が非空になる、とは言えない。

---

## 5. 有効mode・位相・dressingの独立検証

### 5.1 (SP)の意味とRiesz射影

対象\(E_k\)について「\((E_a+E_b)/2=E_k\)になるpairが\((k,k)\)だけ」という条件(SP)を加え、\(e_0=|k\rangle\langle k|\)をHS norm1の単純modeとする。\(iA\)とのpair-sum gap \(g>0\)を定義し、\(M_\delta=(\Psi_\delta-I)/\delta\)を摂動対象にすれば、\(\|M_\delta-iA\|\le|\delta|T_*\)となる。\(|\delta|T_*<g/8\)で中心\(iE_k\)、半径\(g/2\)のRiesz contourを使えば、resolvent恒等式から

\[
\|P_\delta-P_0\|\le 4|\delta|T_*/g.
\]

階数保存・非直交射影の大きさ・左右modeを使った重みを含める条件は整合する。特に、\(c_\delta=(e_0,P_\delta I)\)とすると\(|c_\delta-1|\le\sqrt n p_\delta\)、かつ\(P_\delta\)が\(\Psi_\delta\)と可換であることから

\[
\left|z_s-c_\delta\lambda_\delta^s\right|\le\sqrt n p_\delta\quad(\text{全 }s)
\]

を導ける。**固有値だけで信号を近似したのではなく、他modeの重みを直接制御している点を評価する。** ただし、(SP)と\(g>0\)は一般分子・sector内groundで自動成立しない。

### 5.2 先導減衰の\(W\)、\(v\)は全\(H\)状態で正しい

\[
W_k=1-\langle k|h^2|k\rangle,\qquad
v_k=\operatorname{Var}_j\langle k|P_j|k\rangle.
\]

\(\Delta_j=(P_j-h)|k\rangle\)と置くと、\(W_k=\mathbb E_j\|\Delta_j\|^2\)、\(v_k=\mathbb E_j|\langle k|\Delta_j\rangle|^2\)。直交射影の不等式から\(0\le v_k\le W_k\)が**全Hの任意の正規化状態**で成立する。\(W_k\)を\(1-\langle h\rangle_k^2\)へ置き換えてはいけない。

\(Q=\beta D_R\)について\((e_0,Qe_0)=-\beta(W_k+v_k)=:-\gamma_{\rm sh}\)。単純pair-sum条件(SP)ならC-sec信号は

\[
z_s^{H,\rm sec}=e^{iT_{\rm rel}E_k}e^{-u\gamma_{\rm sh}}.
\]

有限stepでのlog固有値誤差が\(O(|\delta|^3)\)、累積位相が\(O(u|\delta|)\)、相対rate誤差には\(\gamma_{\rm sh}>0\)の下界が必要という区分を受理する。\(W_k=0\)なら\(v_k=0\)で、\(\rho\)は0/0となり定義できない。

### 5.3 三次位相の三成分

著者は

\[
\log\Psi_\delta=i\delta A+\delta^2Q+i\delta^3T+O(\delta^4),
\]

\[
T=rc^3T_R+K_{\rm sp},\quad
K_{\rm sp}=\frac{[B,[B,C]]+[C,[B,C]]}{6}
\]

を導いている。対称BCH\(\log(e^{X/2}e^Ye^{X/2})\)の三次係数と\(X=2i\delta B\)、\(Y=i\delta C+\delta^2Q+\cdots\)を代入すると、\(K_{\rm sp}\)の係数・符号は一致する。有限角tailの\(T_R\)も\(\log\Phi\)の三次項から導ける。

(SP)下で、\(\log\lambda_\delta=i\delta E_k-\delta^2\gamma_{\rm sh}+i\delta^3\chi_k+O(\delta^4)\)、

\[
\chi_k=(e_0,Te_0)+\sum_{\omega\ne E_k}\frac{\|\Pi_\omega Qe_0\|_2^2}{\omega-E_k}
\]

の第二項は\(iA\)の通常の二次固有値摂動を\(Q\)で評価した結果として符号が整合する。対象が全Hilbert空間の**非縮退最低状態**なら分母は正、mode mixingによる寄与単独は非負。ただし**全位相biasの符号**はouterとtailの項を含むため一般には決められない。\(\chi_k\)は解析係数であり、四次以上の明示的な有限step一様剰余とは別である。

### 5.4 Dressing

\(D=e^{i\alpha H_D}\)、\(\alpha=\delta/4\)、\(H|k\rangle=E_k|k\rangle\)から、

\[
\|D|k\rangle-e^{i\alpha\langle H_D\rangle_k}|k\rangle\|
\le|\alpha|\,\sigma_k(H_R).
\]

一方で真の有効modeの距離は\(O(|\delta|T_*/g)\)、真のrate絶対誤差は\(O(|\delta|)\)、相対誤差は正のrate／ordinary分母に依存する。状態の近さをmode／rateの相対誤差に変換していないことを確認した。**固定\(\alpha\ne0\)の別系列へこのdressing消失を移してはいけない**。

---

## 6. Claude付録A.8：共鳴blockでのJensen floorを独立検証

これは今回のClaudeレビューから得られた**最も重要な数学的補足**の一つであり、単なる引用ではなく以下を独立に検証した。

**前提**：全\(H\)の対象固有値\(E_k\)が非縮退。\(E_a+E_b=2E_k\)となる他の\((a,b)\)は許す。\(e_0=E_{kk}\)、\(Q_0\preceq0\)。

**Step 1：入力恒等作用素の該当周波数成分。** \(I=\sum_aE_{aa}\)で、\(E_{aa}\)のpair-sum周波数は\(E_a\)。\(E_k\)が非縮退なら\(\Pi_{E_k}I=E_{kk}=e_0\)。**Pair-sum共鳴で生じるoff-diagonal \(E_{ab}\) は、この入力\(I\)には含まれない**。

**Step 2：該当blockの生成子。** \(Q_{E_k}=\Pi_{E_k}Q\Pi_{E_k}\)はHS自己随伴で負半定値。\(e^{is\delta A}\)はこのblock上で\(e^{is\delta E_k}\)倍である。従って

\[
z_s^{H,\rm sec}=e^{is\delta E_k}\,m(u),\qquad
m(u)=(e_0,e^{uQ_{E_k}}e_0).
\]

**Step 3：非負性とJensen。** \(Q_{E_k}=\sum_\ell\mu_\ell|x_\ell)(x_\ell|\)、\(\mu_\ell\le0\)とし、\(w_\ell=|(x_\ell,e_0)|^2\)とすると、\(w_\ell\ge0,\sum_\ell w_\ell=1\)。よって

\[
m(u)=\sum_\ell w_\ell e^{u\mu_\ell},\qquad
\sum_\ell w_\ell\mu_\ell=(e_0,Qe_0)=-\gamma_{\rm sh}.
\]

凸関数\(e^x\)のJensenから

\[
\boxed{e^{-u\gamma_{\rm sh}}\le m(u)\le1,\qquad
\operatorname{Arg}(z_s^{H,\rm sec})=s\delta E_k\pmod{2\pi}.}\tag{6.1}
\]

**判定：成立（高）**。重要な点は、これが**一つの指数\(m(u)=e^{-u\gamma_{\rm sh}}\)を要求しない**こと。共鳴した複数の減衰指数の非負weightを許容し、その重ね合わせに対して下界を得る。

**Step 4：真の平均信号への限定的移送。** T1bの複素誤差右辺を\(E_{\rm sig}\)とする。\(E_{\rm sig}<e^{-u\gamma_{\rm sh}}\)なら

\[
|z_s|\ge e^{-u\gamma_{\rm sh}}-E_{\rm sig}>0,
\quad\left|\operatorname{Arg}(z_s/z_s^{H,\rm sec})\right|
\le\arcsin\!\left(E_{\rm sig}e^{u\gamma_{\rm sh}}\right).
\]

従って固定有限modelにおける十分小さい\(|\delta|\)の非空領域では、(SP)なしで位相保証へ進める。ただし系サイズ／small gapにより\(E_{\rm sig}\)が実用的に小さくなるとは保証されない。初期状態が一般のguiding stateである場合や、対象エネルギーが縮退する場合の同じ式は**本証明の対象外**。

**成果の分類**：`JENSEN_REFERENCE_FLOOR_ACCEPTED_WITH_NONDEGENERATE_EIGENVALUE`。これはT1b定理1の訂正でなく、適用域を強める**新しい系**。Codexの元の「共鳴ではfloor未保証」という記述については、非縮退対象であることを条件に新しいaddendumが妥当である。元SOURCEの黙示的な上書きは不可。

## 7. 共鳴反例の射程

ClaudeとCodexの三準位モデルでは、\(H=\operatorname{diag}(-1,0,1)\)、孤立した中央固有状態、非零の乱択tail、\([H_D,H_R]\ne0\)、pair-sum共鳴がある。該当blockの散逸作用素の二つの固有減衰係数に対応し、

\[
m(u)=\tfrac23e^{-\beta u W/2}+\tfrac13e^{-2\beta uW}\ne e^{-\beta u W}
\]

（一般の正の\(u,W\)について）となる。重み\(2/3,1/3\)、係数\(1/2,2\)と初期導関数の一致を紙上で確認した。Jensenによる\(m(u)\ge e^{-\beta uW}\)とは矛盾しない。**否定されるのは「非縮退Ekだけから単一指数rateが等式として成立する」という一般化であり、定理1のblock保証、非負floor、参照位相は否定されない。**

ただし、この具体例は\(P\)と\(-P\)という符号違いの同じinvolutionを別々にsamplingし、確率\((1\pm\xi)/2\)を与える表現を利用している。Hamiltonianを同一Pauli軸で非冗長に集約し直すと、元のstochastic lawは変わり、同じ減衰が残るとは限らない。**非冗長なPauli-string lawや既存DF-native実装に対する一般的な反例としては認定しない**。数学的な存在例としては有効。

---

## 8. Claudeの等イベント比較：代数的には正しいが資源意味論を限定

### 8.1 必須の公平比較

同じ理想\(H\)、分割、状態、相対時間\(T_{\rm rel}=s\delta\)としたとき、著者の先導係数は次のとおり：

| 方式 | 乱択label／gate位置数（1 step） | 固定枝数の変数 | 先導減衰 |
|---|---|---|---|
| Gate共有± | 共有された\(r\)位置 | \(r\) | \(\beta(W+v)\) |
| 独立± | 各枝\(r\)で合計\(2r\)位置 | \(r\)／branch | \(\beta W\) |
| Ordinary | 片枝\(r\)位置（相対角は\(2\theta\)） | \(r\) | \(2\beta W\) |

\(\beta=\lambda^2/(4r)\)。同じ\(r\)だけを見ると\(\gamma_{\rm sh}\ge\gamma_{\rm ind}\)であり、これは**同じ枝あたりのサンプル数**の比較。総位置数\(R\)を同じにする概念的比較では、shared\(r=R\)、independent各枝\(r=R/2\)、ordinary\(r=R\)なので、形式的なC-secの先導減衰は

\[
\gamma_{\rm sh}(R)=\frac{\lambda^2}{4R}(W+v),\quad
\gamma_{\rm ind}(R)=\gamma_{\rm ord}(R)=\frac{\lambda^2}{2R}W.
\]

従って\(W>0\)なら

\[
\boxed{\rho_\infty^{(H)}=\frac{W+v}{2W}\in[\tfrac12,1].}\tag{8.1}
\]

**判定**：係数の算術と同位置共有lawの分散構造は正しい（高）。(SP)なら単一rateの等式、\(E_k\)非縮退・共鳴を許す場合には§6のJensenによる**参照visibility下界としての優位性**がある（条件付き）。

### 8.2 「必要イベント数比」への格上げの条件

ある一定の減衰予算を達成するため\(R\)を調節する*先導C-secモデル*では、\(\gamma\propto1/R\)から(8.1)が必要位置数の比の**漸近的な設計指標**となる。ただし：

1. **各\(r\)を固定した\(\delta\to0\)での定理**を、\(\delta\)固定のまま\(r\to\infty\)へ一様に交換する根拠はT1bにはない。有限の二つの\(r\)を比較する小step漸近として読む。小予算で\(r\)自体が増加する運用の保証は別途必要。
2. \(R\)は**理想化した乱択gate位置数**であり、directional controlled \(Z_c\otimes P\)と、片枝だけのcontrolled \(P\)、および異なる枝のmultiplexingの**native T／RZ／CX／depth単価**は等しいとは限らない。\(R\)は回路深さや総費用と同義でない。
3. 独立±の二枝は異なるlabelを持ち、同じハードウェア上の制御多重化、並列化、compiler最適化でゲート価格が変わり得る。
4. (SP)または§6のfloorだけでは**一般のguiding state／多round RPE／shot数／精度**の比較にならない。
5. 整数\(R\)、単位演算価格、\(\lambda\)・sampling lawの同一性、既存PR-RTE（Taylor系のイベント）とのアクセスモデルの差を固定する必要がある。

**したがって**「同イベント数でsharedが独立±よりC-sec先導減衰において有利」は適切だが、「量子化学でsharedが独立±よりRPE総費用で有利」は**未認証**。Claude §12.2後半のnative単価に関する楽観的な見通しは仮説として扱う。

### 8.3 係数の物理的意味

\(W=\mathbb E\|\Delta_j\|^2\)、\(v=\mathbb E|\langle k|\Delta_j\rangle|^2\)なので、\(W-v\)は\(|k\rangle\)に直交するfluctuation成分の平均二乗normである。

\[
\rho=1-\frac{W-v}{2W}.
\]

これは「全分散を一律に半減」ではない。**直交成分に関係する寄与だけがordinaryとの係数比較において半減し、状態に平行な変動\(v\)は残る**という限定した機構の説明が妥当である。\(W=0\)なら0/0で、\(\rho\)を強引に1/2とは置かない。

---

## 9. Claude付録A.9の総資源STOP不等式：一般的には成立しない

**この節はCheckpoint 2で研究判断を変える最重要の注意点**である。

### 9.1 成立する狭い条件

各候補分割\(j\)について、**同じshot schedule、同じ精度配分、同じ決定論費用、同じイベント単価、同じ比較対象、同じ分割候補集合**が確定しており、さらに共有方式の単一試行コストだけが

\[
C_j^{\rm shared}=D_j+\rho_jR_j,\quad
C_j^{\rm ordinary}=D_j+R_j
\]

と変わると**仮定**する。\(0\le\rho_j\le1\)なら

\[
\frac{C_j^{\rm shared}}{C_j^{\rm ordinary}}
=1-(1-\rho_j)f_{R,j},\qquad
f_{R,j}=\frac{R_j}{D_j+R_j}.
\tag{9.1}
\]

\(\rho_j\ge\rho_{\min}\)が全候補で成立するとき、\(C_j^{\rm shared}\ge\rho_{\min}C_j^{\rm ordinary}\)。同じ候補集合なら最小化しても\(\min_j C_j^{\rm shared}\ge\rho_{\min}\min_j C_j^{\rm ordinary}\)。**この限定された固定shot・同価格の単一実行コストモデルでは、Claudeの代数不等式は正しい**。

### 9.2 なぜ一般のRPE総費用no-goにはならないか

Claude付録A.9は「shotやbiasの追加費用は削減率を減らす方向にしか働かない」と述べるが、**これは一般には正当化されていない**。

**数学的な反論（新規数値計算ではない記号的考察）**：固定の位相測定精度で、干渉振幅\(m\)が小さくなると必要測定反復が概ね\(N_{\rm shot}\propto m^{-2}\)に増えるという典型的な測定モデルを考える。同じ対象でsharedのvisibilityが\(m_{\rm sh}>m_{\rm ord}>0\)なら、単一試行コスト比を\(c_{\rm sh}/c_{\rm ord}\)として総費用比は

\[
\frac{C^{\rm total}_{\rm sh}}{C^{\rm total}_{\rm ord}}
=\frac{c_{\rm sh}}{c_{\rm ord}}
\left(\frac{m_{\rm ord}}{m_{\rm sh}}\right)^2.
\tag{9.2}
\]

すなわち**可視度の改善が必要shot数も減らす場合、総費用の削減率は単一試行の\(1-\rho\)より大きくなり得る**。その逆に、directional制御のgate単価増大・準備・phase-bias補正で効果が相殺され得る。したがって「\(\rho\ge0.85\)なら総資源削減15%以上は不可能」といった包括的なSTOPは、\(\rho\)だけからは導けない。

更に、\(\rho\)が小さい場合でも\(f_R\)が小さければ、式(9.1)の**同shot・同価格**モデルですら総利得は小さい。\(\rho\)単体は資源見積りではなく、特定のattenuation headroomである。

**判定**：Claudeの**条件付き費用不等式は代数的に受理するが、未検証のshot／bias／単価へ一般化した資源STOP規則は受理しない**。固定\(S^*\)、\(\rho\)閾値、\(\alpha\sigma\)閾値だけで全M1-Sの正式STOPを自動化しない。研究の継続を判断するには、条件を明示した別の資源モデル、あるいは限定screeningという位置付けが必要である。

### 9.3 適切なGate 0での利用法

\(\rho\)は**small-step・oracle-state・Pauli-tailという契約における改善余地の診断**として使う。高い\(\rho\)を理由に、この一つの減衰機構の**研究優先度を下げる**ことは可能だが、native制御単価やshotの効果を未検証のまま「研究全体の改善不可能性」を証明したことにはしない。低い\(\rho\)も、有限stepのPR実装やRPEの成功を保証しない。

---

## 10. 実用の極限・Pauli/DFのモデル差・Gate 0の情報価値

### 10.1 T1bの極限はPR一般の運用保証ではない

T1bの主系列は **\(r\ge1\)を固定し、\(\delta\to0\)、\(u=s\delta^2\)を有限管理**するもので、非自明な減衰領域では\(T_{\rm rel}=s\delta\)は長くなる。この数学的結果は成立する。一方、実際のPR設計では有限のTrotter step、乱択部のイベント数、角度、round精度をjointで選ぶ。**固定\(\delta\)で\(r\)を増やす経路はT1bの受理域から外れる**。ただし、全PR文献の運用が必ず「\(\delta\)固定・\(r\to\infty\)」だという普遍的説明までは固定資料で証明されていない。重要なのは、**採択した漸近と、実際に最適化するパラメータ経路の不一致を残したまま資源claimへ飛べない**という点である。

有限\(\alpha=\delta/4\)における決定論的なFloquet／Strang参照のmodeと、\(1/r\)で小さくなる乱択散逸の同時解析は、T1cとしての研究価値がある。ただしClaudeの「半stepだけdressedした状態の分散式」は未証明の予想であり、正式な減衰式にはしない。実施にはS0初期scope外の**新しい理論研究契約と別途ユーザー承認**が必要。

### 10.2 DF nativeとの相違

現理論はPauli／Hermitian involutionのIID law。既存DF-nativeのPR eventは一般にinvolutionではなく、決定論blockの実装・normalizer・rotation価格も異なる。H4の既存DF prefixをPauli-tailへ展開すれば形式的にこの理論の対象を作れる可能性はあるが、**元のDF-native PRと同じ確率分布・\(\lambda_R\)・イベント価格・資源優位性を検証したことにはならない**。Gate 0の科学的対象は「Pauli-tailに置換した別方式のheadroom診断」と明示する必要がある。DF-nativeへ一般化するには新しい契約が必要。

またPauli項は符号を吸収して**同じPauli文字列を非冗長に集約**し、identity項、零係数、符号、正規化係数の処理を事前固定する。重複表現やsplitの恣意的な選び方で\(W,v,\rho\)を操作しない。

### 10.3 実分子で最初に確認すべき情報と、不要な計算

H4／STO-3G程度の小系は候補となるが、指定された保存NPZ・Hamiltonian・分割の**正本とSHAが取得可能か**をCodexの技術preflightで確認するまでは、特定のgeometry・rank・保存hashを最終採択しない。対象\(N,S_z\)のsector groundは、全Fock空間の最低状態・(SP)・全pair-sum非共鳴と同義ではない。Pauli-tailがsectorを保存しなければ、対象の存在域・端点性・alias/near couplingを**全到達空間**で扱う。

Gate 0の科学的に安価な順序としては、**事前固定した少数の分割**について (i) Hamiltonian再構成・Pauli law／normalizerの整合、(ii) 厳密対象stateの可用性と情報oracle区分、(iii) \(W,v,\rho\)と\(W=0\)例外、(iv) \(\alpha\sigma_k(H_R)\)という小step／有限step差の警告量、(v) 余力があれば全space pair-sum gapとcoupling support、を提案する。**上界(4.1)の非空性を主目標にして大きな全spaceスペクトル計算を増やすことは推奨しない**。必要な各科学量の数値生成・再解析は、今回の承認に含まれていない。

Gate 0から得るべき答えは「この事前登録したPauli-tail／oracleのsmall-step\(\rho\)機構について、実分子候補にheadroomがあるか」であり、「PR-RTE一般の化学精度資源優位を実証したか」ではない。

---

## 11. 独立新規性・先行研究・論文化の評価

### 11.1 既知の枠組み

qDRIFTのPauli確率サンプリング、RTEと部分乱択QPE、secular／Davies型平均化、Kato型Riesz射影、BCHによるStrang分割、状態依存の有効Hamiltonianスペクトル解析はすべて既知の枠組みである。従って証明技法やdirectional gate共有の外形を新発明とするのは不適切。

一方、**branch-shared coherent first momentという対象を固定し**、non-CPTPであるにもかかわらずHS空間で散逸性を示し、全Hのsecular近似に長時間一様な誤差を与え、\(W,v\)と共鳴の違い、位相の三成分を定量化することは、この研究に固有の適用結果として区別できる。Jensen floorによる「共鳴でも振幅下界・理想参照位相が保たれる」系は、簡潔だが強い整理の価値がある。

### 11.2 強い先行研究比較がまだ必要な点

今回公開abstractを確認した2026年9月のMorisaki–Fujii論文は、ランダム生成子の分散と**信号誤差**・sampling ruleの最適化を扱う。これはClaudeの比較一覧より近い対抗概念であり、論文化する前に**full proofの対象・入力分布（Haar平均か対象固有stateか）、評価norm、時間域、branch間law、平均channel／first moment、gate価格**をCodexへ調査させ、差分を明確にすべきである。またYi–Crosson 2022の状態・有効スペクトル／QPE phase分析は、T1b三次位相の一般的な着想の先行として考慮する。

**新規性評価**：数学的手法の新規性は低、対象を固定した保証としての個別性は低〜中、科学的価値は現時点では中以下。論文として採択可能な「独立の新定理」であるか、応用の有用性が実際に高いかは未確定。網羅的なpriority claimは行わない。

### 11.3 主張できる限定的な中心claim

> Pauli／Hermitian involution-tailを持つexact deterministic-half＋branch-shared ±のcoherent平均信号に対し、固定有限次元・固定r・小step・有限減衰予算で全Hに基づくC-sec信号への全step一様な誤差上界を与える。全Hの対象固有値が非縮退なら、pair-sum共鳴を許してもsecular参照の振幅はJensen下界を持ち、位相は理想の相対位相と一致する。単純blockでは先導減衰係数比\(\rho\)が等式で成立し、三次位相はouter／tail finite-angle／mode-mixingに分けられる。

**このclaimが言わないこと**：任意のguiding state・RPE statistical RMS、一般DF-native、全次元／gap一様な保証、実装イベント価格、エネルギー推定総コスト半減、新しいunitary wordの発明。論文化は学術的差分と実分子での情報価値が確認された後に強く主張するのが妥当。

---

## 12. 代替方針と研究資源配分

| 選択 | 得られる情報・価値 | 主なリスク | 今回の推奨 |
|---|---|---|---|
| **C：T1a＋T1bを限定理論成果として整備** | 今確実な数学的成果、Jensen系、共鳴の限界、先行差分を保全 | 単独の新規性は限定的 | **基本線として推奨**。実作業は採択後Codexが整理 |
| **A：目的限定Gate 0** | 小系で\(W,v,\rho\)に改善余地があるか、Pauli-law／full-space条件が実際に成立するかを判定 | Oracle情報・固定\(r\)理論・DFとnative費用の差、偽陰性／偽陽性 | **科学的に条件付き推奨**。事前登録・資料検査と別の数値計算承認が必要 |
| **B：T1cの固定\(\delta\)／有限\(\alpha\)理論を先行** | PR運用とのギャップを直接閉じる | 理論が難しく、headroomゼロなら無駄 | 現時点の第一優先にはしない。Gate 0が弱い情報しか与えない場合は順序を再検討 |
| **D：研究方向の再定義（DF-native／guiding state／Q4）** | 元PRの回路／RPE価値へ近づく | 別law・別実験／数学契約が必要、研究範囲が拡大する | 現段階で自動移行せず、Gate 0と新規性照合後に判断 |
| **E：M1-Sをtechnical noteで閉じて他Track優先** | 研究資源の節約、成立成果の保全 | 安価なheadroom情報を得ずに可能性を捨てる | Gate 0の情報価値が低い、資料が整わない、または資源が逼迫する場合に合理的 |

**総合推奨の確信度：中**。Claudeの「Cを確保＋目的限定Gate 0」の大筋に同意するが、Gate 0の**判定ルール・実測対象・停止範囲**はClaude案から変更する必要がある。したがって本レビューだけでGate 0の科学的計算GOを出さない。

---

## 13. Gate 0を行う場合の科学的設計条件（実行ではない）

### 13.1 事前登録すべき最小契約

1. **対象の正本**：対象H4のgeometry／基底／全Hamiltonian、対象電子数と\(S_z\)、既存NPZ・Hハッシュ、エネルギー参照の版、必要なら対象の全Fock／到達空間を固定。過去の別geometry・rank・分割を混ぜない。
2. **候補law**：既存の分割から結果を見ずに選んだ少数の\(H_D/H_R\)。Pauli-tailに展開する場合は符号集約済み非冗長Pauli表、identityとnormalizer、\(\lambda_R>0\)、reconstruction toleranceを事前固定。DF-nativeとの差を宣言する。
3. **主評価指標**：\(W_k\)、\(v_k\)、\(\rho=(W+v)/(2W)\)と\(W=0\)の扱い。値は**oracle headroom指標**であって価格・RPE保証ではない。
4. **補助診断**：\(\sigma_k(H_R)\)、事前に固定した参照\(\alpha\)に対する\(|\alpha|\sigma\)、必要性が明らかな範囲の全space端点性・near pair-sum coupling／alias。取得費用・使途が不明な診断は計算しない。
5. **比較対象**：ordinaryとindependent ±は同ideal task・同law・相対時間・概念的イベント予算で比較。Native gate単価は既存資料にある場合に限り別会計。旧DF-native PRを同じ方式として混ぜない。
6. **解釈**：\(\rho\)が小さくてもGate 0は資源優位のGOを与えず、\(\rho\)が大きくても全資源no-goを認定しない。固定step幅の運用差とshot／phase-biasが残ることを明記。
7. **判断を変える条件**：ユーザーが事前に採択した「このlaw・baseline・実用価格／必要削減量に対して次段階に資源を投じる意義があるか」という基準を用いる。\(0.7,0.85,0.1\)等の数値は**Claudeの例示であり正式閾値ではない**。本レビューで閾値を新採択しない。
8. **失敗時STOP**：入力SHA不一致、Hamiltonian reconstruction不一致、lawの非正規化、対象state・sectorが未確定、Pauli-tailとDF-nativeの混同、未承認の数値拡張。研究縮小・終了の正式判断はGPT／ユーザーへ戻す。

### 13.2 既存資料の技術preflightと科学計算の境界

資料所在・版・既存の実装可用性・read-onlyのsource整合・候補分割の一覧整理はCodexで技術的に対応できる。**ただし、新しい固有状態、\(\langle h^2\rangle\)、\(W,v,\rho\)、gap、\(\alpha\sigma\)などの値の取得・再解析は新規科学計算に該当し、別の明示的な実行承認が必要**。分割の後付け最適化、事後的閾値変更、実験数の自動拡大、retryや未承認source変更をしない。

### 13.3 Gate 0の結果で何を変えるか

- **小step headroomが明確で、finite-stepのdressing警告が小さい**：固定幅・\(r\)変更を扱うT1cまたは、限定した実用モデルでの検証に進む研究価値を再判定。自動実施はしない。
- **headroomはあるが\(|\alpha|\sigma\)が大きい**：\(\rho\)を運用改善の見込みとして解釈せず、T1cの理論的必要性を優先検討。
- **headroomが乏しい**：この特定Pauli-lawの分散削減機構は優先度を下げる。ただしnative単価・shots・別分割による改善を未検証なら一般的no-goを出さず、technical noteで閉じる選択肢を研究判断へ戻す。
- **入力・モデル前提が満たせない**：同じ結果へ無断の別法で到達せず、実験の科学的意図の再設計を提案してSTOP。

---

## 14. 判定一覧・確信度

| 科学的命題 | GPT判定 | 確信度 | 補足 |
|---|---|---|---|
| Exact step mapのfactorization | `ACCEPTED_WITH_DOMAIN` | 高 | 位置間／step間IID、branch共有・同順符号反転 |
| \(Q=\beta D_R\preceq0\)、二次係数、局所\(K\) | `ACCEPTED_WITH_DOMAIN` | 高 | HS normと固定\(r\) |
| T1b一様C-sec主定理 | **`ACCEPTED_WITH_FIXED_DOMAIN`** | **高** | 全\(s\)、固定有限model、\(\beta u\le D_{\rm cap}\) |
| Riesz mode／weight／残余 | `ACCEPTED_WITH_DOMAIN` | 高 | (SP)、gap／\(\delta\)条件 |
| 三次位相係数とouter会計 | `ACCEPTED_WITH_DOMAIN` | 中〜高 | 解析係数、一般finite-stepの高次明示剰余ではない |
| 先導\(\gamma_{\rm sh}=\beta(W+v)\) | `ACCEPTED_WITH_DOMAIN` | 高 | (SP)下のscalar式、一般では期待減衰rateとしての値 |
| Claude Jensen floor（共鳴あり） | **`ACCEPTED_WITH_NONDEGENERATE_E_k`** | **高** | C-sec。真の信号は\(E_{\rm sig}\)控除 |
| (SP)下の\(\rho\)等式 | `ACCEPTED_WITH_DOMAIN` | 高 | \(W>0\)、定義済ordinary、先導係数比 |
| 非縮退・共鳴のfloor優位 | `ACCEPTED_AS_REFERENCE_BOUND` | 高 | 実信号・必要ショットへのそのまま転用不可 |
| 同イベント数のshared/independent/ordinary比較 | `ALGEBRA_ACCEPTED_COST_NOT_CERTIFIED` | 高／中 | 位置数でありnative circuit priceでない |
| 三準位共鳴例 | `COUNTEREXAMPLE_ACCEPTED_IN_DEFINED_CLASS` | 高 | ±同一involution重複law、Pauli-onlyには未拡張 |
| Claudeの\(\rho\)総資源STOP | **`REJECT_UNQUALIFIED_TOTAL_RESOURCE_STOP`** | **高** | 同shot・同単価の限定代数式としては成立 |
| H4での実用優位・RPE・native費用 | `NOT_ESTABLISHED` | 高 | 数値未実施、モデル差・oracle・費用未確定 |
| 独立論文の十分な新規性 | `NOT_CERTIFIED` | 中 | 近接先行との差分追加確認が必要 |
| Gate 0を実施すること | `CONDITIONAL_RECOMMENDATION_NOT_AUTHORIZED` | 中 | 新事前登録・別計算承認が必要 |
| M1-Sの全体正式GO／終了 | `NOT_ADOPTED` | 確定 | ユーザーの意思決定 |

---

## 15. 次の担当・承認・mandatory STOP

**直後の担当はユーザー**：本レビューの科学的推奨を採用するか判断する。採用する場合に限り、Codexへ（i）T1a＋T1b＋Jensen補足を**原本非改変のaddendum／technical note草稿**として整理、（ii）限定Gate 0の**読取り中心のsource preflightと実験事前登録案**を依頼できる。ただしどちらも本文作成やGitHub更新を含む作業範囲についてユーザーの指示を受ける。

**次のCodex作業として条件付き推奨**：新規数値計算を行わず、固定SOURCE・既存H4候補・Pauli-tail再構成の入力所在、非冗長化の契約、比較baselineの定義、Gate 0の計算を開始した場合の必要な科学項目を一覧化。提出後はmandatory STOP。正式Gate 0契約の主要指標・STOP閾値・比較条件を、ユーザーの承認なくCodexが科学採択しない。

**新規数値計算（別途明示承認必須）**：H4/H6の\(W,v,\rho\)、新たなHamiltonian／Pauli行列評価、gap・coupling・\(\alpha\sigma\)、保存済みデータからの新しい科学量生成、sampling、compile、Q4/Q5。**今回実施件数0**。

**次の重要GPTレビュー**：Gate 0の科学的な結果が得られ、T1cへの進行・研究縮小・論文化の中心claimの判断が必要なとき。ただしユーザーの明示的なレビュー開始承認を再度要求する。T1c実施には固定\(\delta\)系列を対象にする新研究契約と明示承認が必要。既存S0・T1a・T1b・Claude/GPT原本・Parent allocation D＋E・marker・STOPを維持する。

### 結語

**数学的成果は成立したが、そのままPRの実用的な資源改善には到達していない**。ClaudeのJensen補足はC-sec位相保証の見通しを確かに改善し、等イベントの\(\rho\)は概念的に意味がある。一方、特定の\(\rho\)閾値だけを根拠に総資源のno-goを宣言することはできず、非冗長Pauli-law／DF-nativeの差と有限stepのdressingが大きな未解決事項である。**限定理論成果を先に確保し、必要資料の事前登録の後、別途承認された最小のheadroom検証で研究の次の投資価値を判断する**。今回のレビューはこれを推奨するのみで、科学的なGate 0実施を認可しない。

---

## 16. 参照・監査

### 固定GitHub資料

1. [T1b証明本文・全17節（commit `54336028edd7ef759f8a0c73bd0a96ade13f47f9`）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)
2. [T1b Checkpoint 2 Decision Packet（同commit）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_checkpoint2_decision_packet_20261016.md)
3. [T1b claim inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/analytic_claim_inventory.json)
4. [T1b evidence index](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/evidence_index.json)
5. [S0正式研究契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md)
6. [T1a証明](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md)
7. [Checkpoint 1 GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4899608ea222777e64e706fda2b7e54b84c86247/docs/research/track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md)
8. [G0-pre文献監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md)

### Claude・外部研究

- ユーザー提供Claudeレビュー原本：`レビューinner(2).md`。§§0–28、付録A.8/A.9を重点検討。SHA256 `c4fdf64d1acd4b21a7abfd42b3c069495ce167f0e64ec1526f53afdf9af7f379`。
- [Günther et al., PRX Quantum **7**, 020332 (2026)](https://doi.org/10.1103/ynxb-p2xq)。公開abstract確認。
- [Campbell, PRL **123**, 070503 (2019)](https://doi.org/10.1103/PhysRevLett.123.070503)。公開abstract確認。
- [Yi & Crosson, npj Quantum Information **8**, 37 (2022)](https://www.nature.com/articles/s41534-022-00548-w)。公開本文とabstractの関連節確認。
- [Morisaki & Fujii, arXiv:2609.36694 (2026)](https://arxiv.org/abs/2609.36694)。公開abstract・要約確認。全文の数学的同値性は未検証。
- [Kivlichan, Granade & Wiebe, arXiv:1907.10070](https://arxiv.org/abs/1907.10070)。既存G0-pre監査を参照。
- [Chen et al., PRX Quantum **2**, 040305 (2021)](https://doi.org/10.1103/PRXQuantum.2.040305)。既存G0-pre監査を参照。

### レビュー実施監査

- [x] 本Checkpoint 2の明示的なGPTレビュー開始承認を確認。
- [x] 固定GitHub原資料の実取得を確認し、著者主張と科学的受理を区別。
- [x] 主要数学的補題・主定理・適用域を紙上で検討。
- [x] Claude新規主張（Jensen、同イベント比較、資源STOP）を独立に評価。
- [x] 成立する等式・下界・必要条件・適用域・資源上の仮説を分離。
- [x] 共鳴反例と非冗長lawとの差を評価。
- [x] Gate 0／T1c／technical note／研究終了という代替案を比較。
- [x] 限定した外部先行研究を照合。未読の全本文を読了扱いしていない。
- [x] 新規科学数値計算・行列数値・sim・sampling・compile・GitHub変更を行っていない。
- [x] S0、T1a/T1b、Parent allocation D＋E、既存結果・SOURCE・authorization・marker、mandatory STOPを保持。
- [x] 完成版Markdownの存在・内容を確認した後、チャットにはそのダウンロードリンクのみを提出する。
