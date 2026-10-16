# Track B／M1-S Gate 0 v5：独立照合後のGPT科学レビュー

- **レビュー日**：2026-10-16（JST）
- **レビュー担当**：GPT-6
- **区分**：Level 2（既存T1a/T1b理論と採択Gate 0科学scope内での、独立数値結果の科学的解釈・次段階選定）
- **開始承認**：ユーザーの本会話における「レビューを開始して」
- **状態・要約**：`GATE0_ORACLE_HEADROOM_SUPPORTED_BY_INDEPENDENT_AVAILABLE_SCOPE_REPRODUCTION / FULL_COMPARATOR_SCOPE_UNDETERMINED / T1B_SINGLE_RATE_DOMAIN_NOT_CERTIFIED / FINITE_STEP_RPE_RESOURCE_GO_NOT_ESTABLISHED / LIMITED_FOLLOWUP_PREREG_RECOMMENDED_NOT_AUTHORIZED / MANDATORY_STOP`
- **今回実施**：GitHub固定SOURCE・報告・保存JSON・既存理論・先行レビューの読解、紙上の恒等式と論理条件の検証、関連公開論文の照合、科学的な主張範囲と次の最小検証の判断。
- **今回実施していないこと**：新しい科学的数値計算、保存値の数値再解析、固有値・gap・pair-sumの数値評価、simulation、sampling、回路生成・compile、Gate 0再実行、契約変更、GitHubへの書込み、新しい実施authorization／grant／markerの発行。

## 0. 結論

**科学的に受理できる内容**：H4 linear 1.00 Å／STO-3G／保存DF rank12／固定生成順prefix G3/G6/G9／符号集約済み非冗長Pauli-tail／保存sector-ground数値oracleに関して、W・v・ρが取得され、独立したoccupation／CAR・厳密整数／有理数経路で**照合可能範囲は全3分割PASS**となった。この事実は、採択された代数的headroom指標に実在の分子モデル上で非自明な値があることを支持する。

**ただしT1bの物理的な単一減衰率比、finite-stepのvisibility/phase改善、RPE shotやnative総資源の改善は未認定**である。保存状態は数値的sector-ground oracleであり、T1bの全Hにおける正確な非縮退固有状態、pair-sum単純条件(SP)、全到達空間のスペクトルgap・mode weightは検証していない。

したがって、今回の結論は **`LIMITED_POSITIVE_HEADROOM_DIAGNOSTIC`** であり、研究全体の`GO`、新アルゴリズムの独立新規性、あるいは資源優位の実証ではない。現行結果は限定理論ノート・後続実験の出発証拠として保存し、さらに研究資源を投じる場合は**既存3分割のまま、finite-stepのcoherent複素信号・位相bias・visibilityをfairな対照と比較する最小pilotの事前契約**を先行させる。pilotの科学数値実行は別途ユーザーの明示承認が必要である。

## 1. 固定証拠・取得した原資料

Repository: [`HIROMU1015/Partially-Randomized-Trotter`](https://github.com/HIROMU1015/Partially-Randomized-Trotter)。レビューでは以下を固定commitでGitHubから直接取得した。

| 資料 | 固定commit／用途 |
|---|---|
| [独立one-shot結果・STOP報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/docs/tracks/algorithm_codesign/m1s_independent_one_shot_results_and_stop_20261016.md) | `cc2f451324b7292ae50cef94ff794d46f225d892`。独立照合・適用範囲・監査 |
| [独立raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc2f451324b7292ae50cef94ff794d46f225d892/artifacts/track_b_m1s_independent_reference_result/2026-10-16/v1_one_shot/result.json) | 同commit。SHA256 `32c73f57a1b3fada6bc185311bd56007628849222ad77e4a33e8383a2d0085ac` |
| [v5 Gate 0結果・STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/docs/tracks/algorithm_codesign/m1s_gate0_v5_one_shot_results_and_stop_20261016.md) | `cdcb7f6f151df217960611f502eb1bac8be63f58` |
| [v5 raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json) | 同commit。SHA256 `8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5` |
| [独立検証契約（採択済み）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3ec5721bda8ac449f61b77eb28b66dfcc5f0e8c8/docs/tracks/algorithm_codesign/m1s_gate0_v5_independent_correctness_contract_adopted_20261016.md) | `3ec5721bda8ac449f61b77eb28b66dfcc5f0e8c8` |
| [独立検証SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4787cf90b053d251bf6392988af35516bdbbe663/src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1) | `4787cf90b053d251bf6392988af35516bdbbe663` |
| [v5 SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5) | `69de022dc8142f703423b1ce0bd42f05e123954e` |
| [v5 tail局所上界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5a185cc0c3aae0e477bfc52b93061e86feb7dec/docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md) | `b5a185cc0c3aae0e477bfc52b93061e86feb7dec` |
| [v5上界の既存GPT独立レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/docs/research/track_b_m1s_gate0_v5_gpt_independent_mathematical_review_20261016.md) | 条件付きモデル下での数式の受理・未認証項目 |
| [T1b理論原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md) | `54336028edd7ef759f8a0c73bd0a96ade13f47f9` |
| [T1a理論原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e7b18dd2aa14c127bce9e248c63996c62a055cc/docs/tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md) | `4e7b18dd2aa14c127bce9e248c63996c62a055cc` |
| [Checkpoint 2 GPT科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md) | `62554ee127a2bec1206367cf1754df2ff49fa000` |
| [S0正式理論契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md) | `373b8e4b9831f4d00bdc00ffc741ac61e03f0841` |

これらは固定入力・結果と論理的な出典である。報告にある別ユーザーturnの実行承認・one-shot消費・既存STOPを本レビューにより変更しない。過去のGate 0 v4は`TECHNICAL_STOP`、W/v/ρ=nullであり、v5は新たに採択された局所誤差モデルで実施した**同development条件に対する事後的な改訂評価**である。結果の独立照合も同一入力の別演算経路であって、held-out実験ではない。

## 2. 研究対象と算術結果

### 2.1 比較対象の定義

保存H4 Hamiltonianは8 spin-orbital、DF rank12、4電子の特定sectorに対応する数値状態を持つ。one-body・constantはHD側、DF blockは保存生成順prefix G3/G6/G9、tailは残余blockで構成する。tailはJW変換後に**同じPauli文字列の係数を符号付きで集約してから絶対値を取る**。identityはHRからHDへ移し、approximate coefficient cutoffは0。元のDF-native event lawではなく、別の非冗長Pauli lawである。

\[
H=H_D+H_R,\quad H_R=\lambda_R h,\quad
h=\sum_jp_jP_j,\quad
\lambda_R=\sum_{P\neq I}|c_P|,\quad p_P=|c_P|/\lambda_R.
\]

\(P_j\)は係数符号を吸収したHermitian involution、\(P_j^2=I\)。保存vectorのnorm²で期待値を割るという固定仕様で\(W,v,\rho\)を計算する。

### 2.2 保存値（表示は原本からの引用）

| HD prefix | \(\lambda_R\) (Ha) | \(W\) | \(v\) | \(\rho\) |
|---|---:|---:|---:|---:|
| G3 | 0.8137790393725155 | 0.9625288642498284 | 0.13226039984301002 | 0.5687046408452855 |
| G6 | 0.026294457645549787 | 0.38153576790356514 | 0.1400352453142982 | 0.6835152259560796 |
| G9 | 5.47382434715216e-7 | 0.9753534991954069 | 0.24871838546505062 | 0.6275016625614326 |

v5原runは到達1,239 gateすべてPASSし、\(W/v\)の採択要求 \(2^{-24}\)、\(\rho\)の分解能 \(2^{-12}\)を満たした。採択したgamma/BLAS/fsumモデルの下での**engineering区間**はraw reportの該当表を参照。外向き丸め等を用いても、platform-level rigorous interval arithmeticや真の固有状態誤差のcertificateではない。

### 2.3 独立実装の証拠と欠測境界

独立検証のSOURCEは保存binary64 bit値を有理数targetとして扱い、既存JW係数helperとは異なるoccupation wedge/CAR・全Pauli trace・整数演算・FWHT・Fractionにより係数、HR、状態量を算定した。全3splitの独立referenceを生成してhash固定した後にだけv5 comparatorを読み、結果との照合を行うという契約である。

独立one-shotは**照合可能範囲PASS**、support欠落・余分な語・符号反転は各0件。報告された絶対差はG3でW約3.8647e-17・v約3.0066e-17・ρ約1.7203e-17、G6でW約5.9335e-17・v約4.5105e-17・ρ約9.1407e-17、G9でW約6.8064e-17・v約6.8509e-17・ρ約1.1499e-17。これは保存されたexact rational差の**表示値の転記**であり、今回再評価していない。

同独立実装は内部でfull-H/HD/HRの厳密な代数整合も検査した。しかし**元v5結果には比較すべきfull-H行列とHD非identity行列が保存されていない**ため、元v5実行のそれらの行列値と独立referenceとの**直接一致は未判定**。全要求scopeは`UNDETERMINED`であり、`PASS`への格上げは禁止する。これは、available scopeの計算値の独立再現を否定するものではない。保存されていない元行列を後付け再生成して元runの直接比較と呼ばない。

独立計算の「厳密」は**保存binary64入力の値から演算仕様が正しく実装された場合の代数的厳密性**であり、CPython整数／有理数の形式検証・hardware認証・量子化学の真の物理Hへの誤差保証ではない。固定条件を変更した独立geometry・別分子での再現性は評価していない。

## 3. 独立した数学的解釈

### 3.1 \(0\le v\le W\) とstrict headroomの意味

規格化された任意の状態\(|\psi\rangle\)と\(h=\sum_jp_jP_j\)を用いる。\(\Delta_j=(P_j-h)|\psi\rangle\)と置けば、各\(P_j\)がHermitian involutionであるため

\[
\begin{aligned}
\mathbb E_j\|\Delta_j\|^2
&=\mathbb E_j\langle\psi|(P_j-h)^2|\psi\rangle\\
&=1-\langle h^2\rangle_\psi=:W,\\
\mathbb E_j|\langle\psi|\Delta_j\rangle|^2
&=\operatorname{Var}_j\langle\psi|P_j|\psi\rangle=:v.
\end{aligned}
\]

正規化状態の射影を\(\Pi_\psi=|\psi\rangle\langle\psi|\)とすると、直交分解から

\[
W-v=\mathbb E_j\|(I-\Pi_\psi)(P_j-h)|\psi\rangle\|^2\ge0.
\]

従って\(W>0\)なら

\[
\rho=\frac{W+v}{2W}=1-\frac{W-v}{2W}\in[1/2,1].
\]

**今回の3分割はいずれも\(W>v>0\)という保存値に整合しているため、非零の「状態に直交する乱択揺らぎ」を持ち、代数的な\(\rho<1\)のheadroomを持つ。** この幾何学的分散分解には\(|\psi\rangle\)が全Hのexact eigenstateであることは不要。一方、\(\rho\)を全s一様な実際の減衰率比へ解釈するには別条件が必要である。

### 3.2 T1bの単一rate比を主張できる条件

T1bで採択された主系列は、**全Hの正規化されたexact固有状態\(|k\rangle\)**、固定有限次元・固定整数\(r\)、\(\delta\to0\)、\(u=s\delta^2\)有界、\(\beta=\lambda_R^2/(4r)\)である。共有±は二枝が**同位置で同じlabel・負枝は同順sign reversal**のcoherent生成子に従う（一般のCPTP channelの散逸係数と混同しない）。

対象pair-sum周波数\(E_k\)のblockが単純である(SP)

\[
\frac{E_a+E_b}{2}=E_k\ \Longrightarrow\ (a,b)=(k,k)
\]

が成立し、適切なgap/mode条件を満たすとき、全Hに基づくsecular信号の先導係数は

\[
\gamma_{\rm sh}=\beta(W+v),\qquad
\gamma_{\rm ord}=2\beta W,\qquad
\rho=\gamma_{\rm sh}/\gamma_{\rm ord}.
\]

等しい**概念的な乱択位置総数**\(R\)で比較する場合には、shared \(r=R\)、ordinary \(r=R\)、independent±は各枝\(r=R/2\)とする。independentの枝あたりstep数だけを同じにする別比較と混同しない。(SP)が破れる場合、\(\rho\)は単一指数の長時間減衰率比としては**認証できない**。固定\(\delta\)で\(r\to\infty\)へ進む系列にも、固定\(r\)小step定理を自動転用しない。

今回は保存された**sector-ground数値oracle**の全256次元Rayleigh residualが小さいというgateは通ったが、全Fock空間での真の非縮退固有値、正確な固有状態との距離、(SP)、pair-sum gap、mode weight、near-resonance couplingは認証されていない。小さい残差だけで固有vectorの近さを保証するにはgap等の情報が不足する。独立referenceの精度一致はこの欠落を代替しない。また、Pauli各eventが保存電子数・spin sectorを保存するとは一般に仮定できないため、sector内スペクトルだけで条件を判定しない。

### 3.3 共鳴がある場合のJensen floorの限定的効力

既存Checkpoint 2は、全Hの対象\(E_k\)が**非縮退**なら、pair-sum共鳴が残っても該当secular blockの実倍率\(m(u)\)について

\[
e^{-u\beta(W+v)}\le m(u)\le1
\]

というJensen下界と参照位相の一致を示している。これはscalar単一rate等式より弱いが、参照visibilityの**下界**として有用である。ただし今回の対象数値oracleがこの前提を満たすことは未検証で、真のfinite-step信号へ移送するための信号誤差上界とfloorの数値的非空性も未検証である。**共鳴を認めても無条件に\(\rho\)を有限時間の単一rateにできる、という意味ではない。**

### 3.4 \(\rho\)順位・\(\lambda_R\)と実際の有用性

\(\rho\)は**同じ分割・同じPauli lawでの先導減衰係数の相対的比較**である。分割を変更すると\(\lambda_R\)、\(W\)、\(v\)、HDの費用も変わり、実際の減衰尺度には\(\beta=\lambda_R^2/(4r)\)が掛かる。特にG9は\(\lambda_R\)自体が非常に小さいため、\(\rho\)が1未満でも、有限時間における絶対的なvisibility差や実用的資源差が有意になるとは限らない。

よって「G3の\(\rho\)が最小だから、G3が最良回路・最良PR分割」「全3splitでサンプル数を\(1-\rho\)だけ削減できる」「RPE総費用が\(1/\rho\)改善する」「DF-nativeのPRにも同じ改善が生じる」とは**主張しない**。\(\rho\)の大小による後付けのsplit選択を、独立の最適化結果に読み替えない。

## 4. 研究上の意義・新規性・着地点

**実質的成果**：

1. 既存のbranch-shared coherent平均信号理論（T1a/T1b）に現れる\(W,v,\rho\)が、事前固定されたH4の複数の分割で数値的に非自明となることを確認した。
2. v4のglobal誤差予算による技術STOPを、条件付き演算モデルを維持したtail局所係数誤差の証明・SOURCE接続で解消し、全precision gateを通して主要指標を取得した。ただし、これは同じdevelopment入力でモデルを改訂した事後評価である。
3. 元SOURCEと独立したexact saved-bit入力referenceが、主要数値の正しさを強く補強する。主に**数値再現性・代数correctness・conditional headroomの実在性**についての成果である。

**未獲得の成果**：有限\(\delta\)のbranch-shared signalの改善、phase biasの制御、統計shot削減、native回路の価格比較、総資源crossover、独立分子条件へのtransfer、研究方法の独立新規性。

新規性の背景として、Günther et al., *PRX Quantum* 7, 020332 (2026)は部分乱択と位相推定・資源評価を既に扱う。Morisaki–Fujii, arXiv:2609.36694（2026-09-29）はランダム生成子の分散・平均誤差・sampling lawを扱い、Yi–Crosson, *npj Quantum Information* 8, 37 (2022)はPFの状態／固有位相に関するスペクトル摂動解析を行っている。今回のレビューは関連箇所と既存Checkpoint 2を確認したが、これらに対するT1bの**定理単位の完全な新規性判定は行っていない**。\(W,v,\rho\)の数値値だけで新しいアルゴリズムの新規性や論文化の独立性を認定しない。

**着地点**：現時点でもT1a/T1b＋限定分子headroom診断を理論・技術ノートに整理する価値がある。一方、独立の新しい主論文としての主張や量子位相推定の実用的改善は、有限時間信号・bias・visibility、必要ならnative費用との接続が成立してから検討すべきである。研究全体を全面GOまたはNO-GOとするだけの証拠はない。

## 5. 次の最小検証：判断を変えるための順序

### 第一優先：事前登録した有限stepのcoherent複素信号pilot

次に科学的に知りたいことは、**small-stepでの\(\rho<1\)という改善余地が、有限の\(\delta,r,s\)でのactual branch-shared信号品質に現れるか**である。最初から長RPE・大grid・複数分子・full compiled総費用へ進めない。

**Codexの次の技術作業（科学実行は別承認）**：

- 既存S0/T1bの**同位置label共有・同順sign reversal・exact deterministic half**を固定した、shared／ordinary／independent±のfinite-step coherent**平均信号**の数学的・SOURCE意味論を、同一理想task・同一Pauli law・同じ相対時間・同じ概念的イベント位置数で定義する。CPTP平均channelの忠実度と混同しない。
- 最初はdevelopment H4の固定3分割G3/G6/G9を維持。結果を見て有利なG3だけへ絞るときは**事後選択**と明示し、同じ集合の独立screeningという主張はしない。
- 有限stepの\(\delta\)、per-step\(r\)、反復数\(s\)、理想相対時間、入力状態、複素信号\(z_s\)、visibility\(|z_s|\)、複素信号誤差、位相差、継続／STOP判断に必要な精度・数値予算を**新しい数値結果を見る前に**最小の固定候補で事前登録する。一般の数値状態に対する理想参照は\(\langle\psi|e^{iTH}|\psi\rangle\)であり、exact eigenstateを未認証なのに\(e^{iTE_kT}\)と置かない。
- まず有限stepで可視度・位相biasの競合を評価し、結果が研究判断を変えない限りshot推定・native compilation・RPE多roundへ自動拡張しない。曖昧な位相unwrapやfloorがある場合は結果をUNDETERMINEDとし、後からwindow／枝・thresholdを選び直さない。
- 可能ならmatrix-free／再帰による平均写像の演算を検討し、無制限に巨大なsuperoperatorをdense構成することは避ける。実行費用・正しさの独立検証・停止条件を設計する。検証の価格が不要に高くなる場合には規模を縮小する案をユーザーに提案する。
- **実行ではなく**独立branchでSOURCE・固定契約候補・preflight・予算案・mandatory STOPを準備することを第一段階とする。

**研究判断の分岐**：

| 将来の限定pilotの結果 | 変え得る研究判断（自動採択ではない） |
|---|---|
| visibility改善がfinite-stepで残り、phase biasが許容内 | 限定価格・shotを含めた次stageの価値を判定。T1c理論が必要かを検討 |
| small-step\(\rho\)は良いがfinite-step効果が消える／biasが増える | v5 headroomは技術ノートへ保存。有限stepの理論（T1c）を別契約で追うか、M1-Sを縮小するか判断 |
| 共有±とordinary／independentの意味論が実装・比較上不公平 | 比較規約を修正した**新契約**を事前に採択するまでは実行しない |
| 特定分割・時間で指標が数値不安定 | 事前契約に従ってUNDETERMINED／STOP。後付けでcandidate・閾値を動かさない |

### 第二優先：T1b適用域の局所的な証拠取得（必要に応じて）

現状の全space正確な固有状態／非縮退性／(SP)／pair-sum couplingは未認証。純粋に**T1bのscalar理論式を今回のH4状態へ適用したい**なら、どの範囲のスペクトル情報が必要かをCodexに先に監査させる。全Fock非縮退groundなら(SP)の十分条件となるが、保存sector-groundだけでは足りない。full 256次元の新しい固有値・gap・coupling計算は、目的と決定の変化が明確になった場合のみ別途承認する。先に巨大な全pair-sum列挙を行うことは推奨しない。

### 第三優先：理論新規性・総資源との接続

Gate 0結果の独立再現性は**新規性判定の代わりにならない**。Checkpoint 2で指摘したGünther、Morisaki–Fujii、Yi–Crosson等の定理・signal・baseline対応について、Codexによるread-onlyの先行研究差分監査を並列化できる。有限stepの利益が確認されるまでは、native費用の大規模compileや全RPE shot数探索に直行しない。

## 6. 科学的判定と権限・STOP

| 問題 | 今回の判定 | 信頼度・証拠の限界 |
|---|---|---|
| 保存H4 oracleのW/v/ρ計算・照合 | **LIMITED_ACCEPT** | 高：原v5と独立referenceはavailable scopeで整合。full-HD comparatorは未保存 |
| 3分割の代数的strict headroom \(\rho<1\) | **SUPPORTED_WITH_FIXED_LAW** | 高：係数・状態・非冗長Pauli lawが固定。条件付き算術target |
| T1bの単一secular減衰率比としての実分子適用 | **DOMAIN_UNVERIFIED** | 全H exact state／(SP)／gap・weight不明 |
| Jensen floorの実分子保証 | **DOMAIN_UNVERIFIED** | 全Hの非縮退固有状態など未認証 |
| finite-step visibility・phase bias改善 | **NOT_TESTED** | δ／r／sの有限信号未評価 |
| RPE shots・native gate cost・DF-native転用 | **NOT_ESTABLISHED** | 理論・law・費用指標が異なる |
| M1-Sの方法新規性・独立主論文 | **NOT_ESTABLISHED** | 詳細prior-art claim比較・有限task利益不足 |
| Gate 0 v5本体・独立one-shot再実行 | **NOT_AUTHORIZED** | 両one-shot消費済み、旧marker/原結果保全 |

**GPTとして推奨する方針**：現在の分子headroomを限定的な正の成果として保存する。M1-S全体の科学的GOはまだ出さず、次の分岐を判断できるように**最小finite-step coherent signal pilotのSOURCE・契約準備をCodexに優先させる**。研究方針の最終採否と新契約の採択はユーザーが行う。本レビューはCodexの契約書改変や実行承認ではない。

今回の重要科学レビューは実施済みにつき本Markdownに記録した。ただし、新しい科学数値計算、保存値の追加数値再解析、固有値計算、simulation、native compile、authorization/grant/marker作成・GitHub書込みは**0件**である。既存のraw結果、SOURCE、契約、one-shot、`mandatory STOP`を保持する。

## 7. 文献・外部確認の範囲

- [Günther et al., “Phase Estimation with Partially Randomized Time Evolution,” *PRX Quantum* **7**, 020332 (2026-05-19)](https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq)：部分乱択を位相推定・化学ベンチマークの資源評価へ接続した先行例。
- [Morisaki and Fujii, “Optimized Randomized Hamiltonian Simulation via Average-Error Analysis,” arXiv:2609.36694 (2026-09-29)](https://arxiv.org/abs/2609.36694)：ランダム生成子の分散・平均誤差・sampling分布最適化。今回の結果との定理水準の同値性は未判定。
- [Yi and Crosson, “Spectral Analysis of Product Formulas for Quantum Simulation,” *npj Quantum Information* **8**, 37 (2022)](https://www.nature.com/articles/s41534-022-00548-w)：状態・gapを用いたPFの有効スペクトル解析。M1-Sの固有値・state依存の議論に近い。
- [Wan, Berta and Campbell, “A randomized quantum algorithm for statistical phase estimation,” arXiv:2110.12071](https://arxiv.org/abs/2110.12071)：乱択時間発展と位相推定の既知技術。

**文献監査の限界**：上記の公開抄録・掲載情報および既存Checkpoint 2の比較を確認した。外部全論文の全証明の再査読や、研究の網羅的新規性判定は今回の限定レビュー対象外。一般化した優位性・独立の初出性は認定しない。
