# M1-S統合研究サマリー：理論・保存結果・科学レビュー・現在地

2026-10-17 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。
文書整理の基点はStage S最終結果 `f2bfb8770e6858de56a63e50e68f07ff2d6464b6`、
先行研究調査は別履歴 `4301c0f87a16be45b620304c3fa935d5aa2d721a`。

**研究記録・引継ぎ資料。論文・論文草稿、新しい科学レビュー、証明、科学計算ではない。**
既存の主張と判断を出典付きで整理する。科学的な受理は過去レビューの記録、数値は固定報告の引用であり、
今回それらを再証明・再評価していない。研究終了・継続の新しい正式採択も行わない。

[証拠・結果索引](evidence_and_results_index.md)には固定commit、SOURCE、契約、raw、監査、レビューを、
[Codexハンドオフ](codex_restart_handoff.md)には再開条件と承認境界をまとめた。
最新の原本は[Claudeレビュー][C]と[GPT総合再評価][G]。両原本と[2026-10-17共通指示][OPS]を
ユーザー提供ファイルからbyte一致で新規保存した。identityは[原本記録](provided_originals.json)。

## 1. 到達点と現在の結論

M1-Sは、乱択時間発展を読むHadamard干渉の正負branchで、同じ位置のPauli labelを共有する方式である。
研究の問いは、このjoint lawが減衰と位相をどう変え、同じ推定taskで資源上の利点を持ち得るかである。

| 証拠層 | 確保されているもの | 成立しない飛躍 |
|---|---|---|
| 理論・適用域付き受理 | T1a/T1bのC-sec一様保証、条件付き先導rate、非縮退対象での共鳴floor | 任意の系サイズ・gap・r・実装・RPEへの保証ではない |
| 条件付き数値診断 | H4のGate 0、有限stepのvisibilityとphase bias | 保存oracleを認証固有状態、数値PASSを科学GOとしない |
| 独立計算経路による照合 | Gate 0の保存可能範囲PASS | 未保存full-H/HDを含む全scopeはUNDETERMINED |
| 同じ保存信号の事後解析 | γ境界、π/3限定gate、Stage Sの局所整数witness | 別分子・held-out・独立した反復実験の蓄積ではない |
| 研究上の推奨 | 最新GPTは成果保全を優先し、資源路線の自動進行を勧めない | 正式終了・継続・Stage Cの採択を代行しない |

**総回路費用改善、多round RPE成功、方法としての独立新規性、論文化は未確立。**
最新GPTは限定数学成果の受理を維持しつつ、強い新primitiveの主張を支持せず、定理statement全体のpriorityを未判定に残す。
ClaudeとGPTは成果を記録として確保する方向では概ね一致するが、還元の範囲・分散の用語・費用gateなどで相違がある（§8）。
今回ユーザーが承認したのはこの記録作成であり、外部発表や資源研究の継続・終了ではない。

## 2. 数学的定義と比較の意味

### 2.1 固定law

[T1b §2][T1B]の規約は、右端先作用、有限次元、

\[
 H=H_D+H_R,\quad H_R=\lambda_R h,\quad h=\sum_jp_jP_j,
 \quad P_j=P_j^\dagger,\ P_j^2=I,\ \sum_jp_j=1,
\]
\[
 D=e^{i\delta H_D/4},\quad A_j=e^{i\theta P_j},
 \quad\theta=\lambda_R\delta/(2r).
\]

正枝は `G_+=D A_jr ... A_j1 D`、負枝は `G_-=D† A_jr† ... A_j1† D†`。
同位置のlabelは二枝で共有、位置間・step間はIID。負枝は**同順の符号反転**であり、一般には正枝全wordの随伴と異なる。
別shotでfreshにdrawする条件は統計集中保証の条件であり、ensemble平均の恒等式そのものと区別する。

\[
 \Phi_\theta(X)=\mathbb E[A_jXA_j],\qquad
 \Psi_\delta(X)=D\Phi_\theta^r(DXD)D,\qquad
 z_s=\langle k|\Psi_\delta^s(I)|k\rangle.
\]

`DXD`や`A_j X A_j`の右側を随伴に置き換えない。通常のsystem共役channelではないが、ancillaを含めた
物理channelのHeisenberg coherence blockとして表せる。非CPなblockであること自体は新規性の根拠にならない。
理論の相対時間は `T_rel=sδ`、各branchの理想時間は絶対値 `T_rel/2`。
保存数値では数値oracleの元norm²で期待値を割り、理論上の厳密固有stateとの違いを残す。

### 2.2 三方式と公平比較

| 方式 | 一stepの乱択 | 理想相対時間を合わせた規約 | 注意 |
|---|---|---|---|
| shared ± | 二枝同位置でlabel共有 | 各枝r、角θ、deterministic quarter sandwich | 同labelでも二枝の演算とcontrol価格は残る |
| independent ± | 二枝は独立 | 各枝rなら共有法の倍の独立draw | 同総draw数Rなら各枝R/2、Rは偶数 |
| ordinary | 一枝の平均振幅 | r個、角2θ、HDのhalf sandwich | 片枝controlとdirectional controlのnative単価は未照合 |

pilotでは総概念label数を揃え、shared/ordinaryはr=R、independentは各枝r=R/2とした。
**同じ概念label数、branchあたりgate数、Pauli指数総数、native gate数、深さ、shots、総資源は別の尺度。**
PR原論文の独立RTEはTaylor eventとnormalizerを持ち、このindependent Pauli exponentialと完全同値とは認定されていない。

### 2.3 方向制御qDRIFT・部分乱択RTEとの関係

[最新GPT §§4–5][G]が整理した位置付けでは、shared eventは
`exp(-iθ Z_a⊗P_j)`として表せる。HD=0なら方向制御拡張HamiltonianのqDRIFT列である。
HD≠0のM1-S全体は、**方向制御された決定論／qDRIFT混成対称積**。
HDまで全項samplingする通常qDRIFTと同一視しない。

列優先vecでは `K_j=P_j^T⊗I+I⊗P_j` により平均写像を倍化空間の平均演算子へ写せる。
pair-sum周波数はそのスペクトルに対応する。partial caseのdeterministic sandwich、相対時間の1/2、
入力vec(I)のnorm√nは残る。この表現は既存の平均演算子枠組みへの包含であり、T1a/T1b全statementの既知定理への直接還元とは別。

PRのhalving議論は決定論対称PFの厳密な随伴関係を使う。M1-Sは乱択wordにもその関係が成立すると示したものではなく、
別のcoherent平均信号を採っている。PR論文の数学的誤りを訂正したという主張は採らない。[G §§4.2–4.3][G]

## 3. T1a：完全乱択での全step一様保証

原証明：[T1a][T1A]、受理：[Checkpoint 1][CP1]。以下は既存結果の記録。

固定有限次元n、固定hの対象固有state、Hermitian involution、共通線形角、位置間IIDを仮定する。
HS内積の演算子空間で `J_j(X)=P_jX+XP_j`、`L=E J_j` とし、

\[
 \mathcal D=-\tfrac12\mathbb E(J_j-L)^2\preceq0,
 \quad\mathcal D_0=\sum_\omega\Pi_\omega\mathcal D\Pi_\omega,
 \quad z_N^{sec}=\langle k|e^{iN\theta L}e^{\tau\mathcal D_0}(I)|k\rangle,
 \quad\tau=N\theta^2.
\]

Πはhのpair-sum周波数 `e_a+e_b` の直交射影。normはHS誘導2→2 norm。
b=||D||≤2、b0=||D0||≤b、near/farは厳密に異なる周波数だけをcutoffηで分け、
`κ_far=Σ ||Πω D Πν||/|ω−ν|` はordered pairのcoupling加重逆gap和である。

\[
 |z_N-z_N^{sec}|\le\sqrt n\left[
 18\tau|\theta|+\tau b_{near}(\eta)
 +2|\theta|\kappa_{far}(\eta)\{1+\tau(b+b_0)\}\right].
\]

対象は全整数N≥0、実|θ|≤1、τ≤有限D_cap。固定有限モデルでη=0ならnear項は0、κは有限で、全対象Nに一様なO(|θ|)。
局所高次の累積 `18τ|θ|` を省略しない。収縮性、Taylor積分剰余、telescoping、振動積分が原証明の構成。
次元非依存・縮むgap familyに一様・実用的な安価predictorという保証ではない。

参照floor r_*>0と複素誤差E<r_*があれば、真の半径はr_*−E以上、相対主値位相差はarcsin(E/r_*)以下。
これは単一信号の位相であり、branch復元や多round RPE、required shotsを保証しない。
非可換二準位の非空域は原証明§9の解析例で、今回再評価した数値例ではない。

## 4. T1b：部分乱択、減衰、中心化分散、共鳴

### 4.1 適用域と主保証

[T1b][T1B]と[Checkpoint 2][CP2]の対象は、固定有限次元・固定整数r、exact HD half、全Hの孤立対象固有state、
δ→0、u=sδ²有界、`βu≤D_cap`、`β=λ_R²/(4r)`。HD内部の実装PF、一般DF event、固定δでr→∞は対象外。

`A=J_H/2`、`Q=βD_R≤0`、`Q0=Σ Πω Q Πω`、周波数は**全H**の `(E_a+E_b)/2`。
`z_s^{H,sec}=<k|exp(isδA)exp(uQ0)(I)|k>` に対し、

\[
 |z_s-z_s^{H,sec}|\le\sqrt n\left[
 Ku|\delta|+u q_n+2|\delta|\kappa_f\{1+u(q+q_0)\}\right].
\]

q=||Q||、q0=||Q0||、qnはnear結合、κfはfar coupling加重逆gap。
KはT1b固有の局所三次剰余定数で、T1aの18を流用しない。定義は原証明§§4–5。
η=0で固定model・rの全sに一様O(|δ|)。√n、β、pair-sum gap依存を消さない。
通常のenergy gapやsector内gapをpair-sum gapと交換しない。個々のPauliがsectorを出る場合、全spaceが対象。

### 4.2 W、v、ρの正しい解釈

全Hの対象stateで

\[
 W_k=1-\langle h^2\rangle_k
 =\mathbb E\langle(P_j-h)^2\rangle_k,\quad
 v_k=\operatorname{Var}_j\langle P_j\rangle_k,\quad0\le v_k\le W_k.
\]

最新GPT§7の解釈修正を記録する：

\[
 w_k=W_k-v_k=\mathbb E\operatorname{Var}_k(P_j-h)
 =\mathbb E\operatorname{Var}_k(P_j)-\operatorname{Var}_k(h).
\]

**中心化した乱択揺らぎの量子分散**であり、生のPauli項の平均量子分散とは一般には異なる。
全Hの固有stateがhの固有stateとは限らないため、Var_k(h)=0を勝手に置かない。
この恒等式は最新レビューの既存結果を記録したもので、新しいCodexの証明ではない。

等総label数R、`β_R=λ_R²/(4R)`の先導比較は、

\[
 \gamma_{sh}=\beta_R(W+v)=\beta_R(w+2v),\qquad
 \gamma_{ord}=\gamma_{ind}=2\beta_R W=2\beta_R(w+v).
\]

W>0かつ単一pair-sum条件(SP)のscalar比較なら `ρ=(W+v)/(2W)∈[1/2,1]`。
半減するのはwの寄与であり、vは変わらない。同per-branch rならindependentは2r drawを使い、
`γ_sh≥γ_ind`である。W=0はρが0/0で適用外。ρはshotsやnative総費用の比ではなく、一般的な資源利得の最大2倍定理でもない。

### 4.3 共鳴、Jensen floor、scalar式の区別

(SP)は `(E_a+E_b)/2=E_k` のpairが(k,k)だけという十分条件。そのときC-secは
`exp(iTE_k)exp[-uβ(W+v)]` の単一指数になる。
対象E_kが非縮退なら、(SP)なしでもCheckpoint 2 §6のJensen系により、

\[
 z_s^{H,sec}=e^{iTE_k}m(u),\quad
 m(u)=\sum_\ell w_\ell e^{u\mu_\ell},\quad
 e^{-u\beta(W+v)}\le m(u)\le1.
\]

weightは非負で和1、μ≤0。参照位相は理想位相mod2π、真の信号には主定理の誤差控除が別途必要。
対象energy縮退・一般guiding state・保存sector-ground oracleへ無条件に適用しない。
T1b提出時の「一般共鳴floor未保証」は当時の履歴として保ち、後続Jensen系は[addendum草稿][ADD]に分離されている。

原証明§12の三準位例は単一指数の一般化を否定するが、Jensen下界とは両立する。
符号違いの同じinvolutionを別eventにするlawを使うため、非冗長Pauli-tail一般の反例と主張しない。

### 4.4 位相biasと有限stepの限界

T1bは条件付きRiesz射影、左右weight、mode残差、縮む固有位相gapを明示する。
三次log位相はouter splitting、tail finite-angle、Qによるmode mixingに分ける。
exact HDでもouter splittingは残る。状態dressingの近さ、modeの近さ、rate絶対差、分母付き相対差は別量。
主保証のKに含まれるouter誤差を再加算しない。

有限pilotはT=1,4固定で、δ→0ならu=Tδ→0。有限減衰を残す長時間系列と同じ解釈にはならない。
[有限stepレビュー §5][FP]と[G §8][G]は、非secular結合が固定T信号へ一次の虚部を与え得る機構を記録する。
これはH4スペクトルを使った量的な原因分解・fitではない。孤立modeの先導位相安定性と、有限時刻信号の位相biasは両立する。

## 5. 数値成果の共通scopeと証拠の限界

以下の分子結果は全て **linear H4、1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、
生成順prefix G3/G6/G9、非冗長signed Pauli-tail、保存数値sector-ground oracle** に限る。
Gate 0はδ windowなし。有限step以降は **δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1**。
108行=非zero72方式行＋zero36、非zero24cell、shared対二方式の48対照。

保存binary64入力が定義する対象であり、物理Hamiltonianの完全性、厳密ground state・overlap・全space非縮退を認証しない。
誤差区分は `CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。
IEEE、complex BLAS、fsum、trig/log、underflow等の各契約の未検証仮定を引き継ぐ。
契約採択や数値test PASSをplatform certificateにしない。

### 5.1 Gate 0 v4停止、v5完了、独立照合

v4はglobal共通係数上界が精度予算を支配しTECHNICAL_STOP、W/v/ρはnull、one-shot消費済み。
科学的negativeではない。後続原因調査・条件付き局所証明を経て、v5は
`alpha_R=γ_(713+r) S_R` を採用した。ここでγ_nは丸め誤差因子で、V1の角許容幅γとは別記号。
残差品質は絶対 `1e-8 Ha+B_res`。旧solverの受理基準で、固有state距離保証ではない。
17精度項目を満たすv5初回runを完了した。[v5結果][G0]

| split | λ_R [Ha] | W | v | ρ |
|---|---:|---:|---:|---:|
| G3 | 0.8137790393725155 | 0.9625288642498284 | 0.13226039984301002 | 0.5687046408452855 |
| G6 | 0.026294457645549787 | 0.38153576790356514 | 0.1400352453142982 | 0.6835152259560796 |
| G9 | 5.47382434715216e-7 | 0.9753534991954069 | 0.24871838546505062 | 0.6275016625614326 |

上表は既存報告の保存値転記。λ_R²を含む絶対減衰尺度を省いてρのみでG9の有用性を主張しない。
v5のSOURCE内1,239 gate PASSは独立実装照合とは別。

[独立one-shot][IND]は、occupation-bit/CAR経路から厳密整数・有理数でHamiltonianを再構成し、
`c_P=2^-8 Tr(P H_R)`で係数・support・符号・identity・λ・law・HR・state・W/v/ρを照合した。
参照hash固定後にだけv5の比較結果を読む設計。全3分割の照合可能範囲PASS。
**v5にfull-H/HD行列が保存されていないため、全scopeはUNDETERMINED**。
同じ保存入力への事後的な独立計算経路の確認であり、外部hardware・別分子・held-out再現ではない。

### 5.2 有限step複素信号

[完了報告][FINITE]では初回correctness、3分割のtwirl、全108行の精度gateを通過。
G3/G6は各16対照で絶対visibilityがresolved higher、G9の16対照は絶対差未解像。
位相の数値半径が小さいことと、方式そのもののphase biasが小さいことを区別する。

[有限step結果レビュー §3.3][FP]の既存値例：G3、R=2、δ=1/8、T=4で、
visibilityはshared 0.9556370833／ordinary 0.9233899381、principal biasは
shared −0.0078561882 rad／ordinary +0.0003549442 rad。
G6にも解像された絶対差があるが規模は小さく、実用的重要性は別問題。
32対照を独立標本の「32勝」や統計的な有意性と数えない。

### 5.3 Hamiltonianモデル差の会計

`H_full`はraw DFからの理想参照、`H_tilde_j=HD_j+HR_saved_j`は実際のsplit target。
既存v2契約は `B_j≥||H_tilde_j−H_full||_2` [Ha]、`η_j(T)=|T|B_j` [無次元] を別台帳に保存する。
Hermitian・同一時間・同じ保存state元norm²の期待値という条件で、Duhamelから複素信号差≤η。
数値丸め、モデル差、有限stepアルゴリズム誤差を混ぜない。

raw-Hに対する総biasはモデル差を既に含むため、V1/Stage Sのraw直接routeへηを再加算しない。
ηそのものはradではない。位相へ移すには正のvisibility下界・非zero・branch条件が必要で、欠ければUNDETERMINED。
モデル上界のみから純アルゴリズム誤差を厳密分離測定したとは言えず、補助H_tilde ideal信号も本系列では未評価。
旧v5の未保存行列との直接一致を、有限stepで埋めたとも主張しない。

## 6. γ依存V1、π/3 gate、Stage S

### 6.1 Task AとTask B

**Task A**は保存raw-Hの理想平均信号の円周位相を、単一時刻・共通角許容幅で読む条件付き十分shot診断。
**Task B**は特定energyを多round RPEで推定し、精度・confidence・全wrapper価格を揃えて総資源を比較する目標。
raw信号の位相が特定E*Tとは限らず、Aの達成はBの保証にならない。

### 6.2 V1の設計と全結果

採択条件はPER_METHOD_ROW、α_row=0.05、fresh-IID、Hoeffding、X/Y等配分、p=128。
`A_m^- >0`、`b_m^+ <γ<π/2` の正margin領域で、

\[
 M_m(\gamma)=A_m^-\sin(\gamma-b_m^+),\quad
 S_m(\gamma)=\frac{4\log(2/\alpha_{axis})}{M_m(\gamma)^2},\quad
 n_m=\lceil S_m\rceil,\quad N_m=2n_m,
 \quad\alpha_{axis}=\alpha_{row}/2.
\]

nは一軸、Nは両軸。**共通の一つの十分設計**であり最適必要shot数ではない。
95%は各方式行についての仮想測定統計の条件付き設計で、全72行同時保証・保存coverageの認定ではない。

[γ依存V1結果][V1]はG3/G6各16対照に内部tradeoff境界を保存した。
sharedは境界の下でmarginが小さく、上で大きい。境界区間内は未解像。
G9の形式順位ALL_COMMON_GAMMA_SHARED_HIGHERは保存しても、絶対改善未解像を維持する。
直接tradeoff `a>c>0, 0≤v<u<π/2` での境界式は、既存レビューの
`X=a cos u−c cos v`、`Y=a sin u−c sin v`、X>0なら `γ*=atan(Y/X)`。
この節のu/vはbias記号で、T1bのu=sδ²と分散v_kではない。
SOURCEは符号未解像、domain外、dominance等も分類する。今回は境界・shotsを再計算していない。

### 6.3 π/3限定gate

[π/3結果][PI3]は保存境界[L,U]と保存π区間[pL,pU]だけを比較した。
`3U<pL`はSATISFIED、`3L≥pU`はNOT_SATISFIED、それ以外UNRESOLVED。

| 層 | 対照数 | 保存判定 |
|---|---:|---|
| G3 T=1,4 | 16 | SATISFIED |
| G6 T=4 | 8 | SATISFIED |
| G6 T=1 | 8 | NOT_SATISFIED |
| G9 | 16 | 絶対改善未解像をcarry、数値照合対象外 |

閾値未解像は0。成立は、連続shared有利域と保守的decoderのγ<π/3領域との交差だけ。
RPEのstate、prior、実装誤差、全round、energy精度は確認していない。不成立もM1-S一般のNO-GOではない。

### 6.4 Stage S：開端点と整数witness

[Stage S設計レビュー][SS]の直接tradeoff条件下で、連続shot比 `R_cont=S_cmp/S_sh=(M_sh/M_cmp)^2` はγに単調増加。
cap=π/3での値は開領域のsupremumで、一般には達成されない。
`S_m`がcapへ向け厳密減少するため、一軸の片側整数は

\[
 n_m^{cap-}=\lfloor S_m(\pi/3)\rfloor+1.
\]

S(cap)が整数ならcapそのもののceilと1異なる。strict整数差が解像されれば、capより小さいγの非空開近傍に
この十分設計の削減が存在する。同数は全γでの削減不存在を意味しない。

[Stage S結果][STAGE]は主対象24対照を評価し、carry24と合わせ全48を報告した。

| 層 | 保存判定 | 対照数 |
|---|---|---:|
| G3 T=4 | CAP_EDGE_STRICT_INTEGER_WITNESS | 8 |
| G3 T=1 | CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED（cap直下同数） | 8 |
| G6 T=4 | 同上 | 8 |
| G6 T=1 | 旧π/3不成立carry、新shot未評価 | 8 |
| G9 | 旧絶対差未解像carry、新shot未評価 | 16 |

主対象の算術未解像0。G3 T4の保存整数表は以下。

| δ [Ha^-1] | R | n_shared / n_ordinary / n_independent | N_shared / N_ordinary / N_independent |
|---|---:|---|---|
| 1/8 | 2 | 26 / 28 / 28 | 52 / 56 / 56 |
| 1/16 | 2 | 25 / 26 / 26 | 50 / 52 / 52 |
| 1/8 | 4 | 25 / 26 / 26 | 50 / 52 / 52 |
| 1/16 | 4 | 24 / 25 / 25 | 48 / 50 / 50 |

8陽性は4cell×二内部対照。物理shotを実行した結果でも、8回の独立実験でもない。
具体的な外生的γ、開近傍の実用幅、全γの整数順位、強いbaseline、native費用は未評価。
Stage Sは初回正常完了・SOURCE修正/retry0、保存監査448・保全14 PASSというローカル記録。
監査は保存整数/schema/provenanceの照合で、sin/log/πやH4信号の独立再生成ではない。

## 7. 費用・RPEへ何が未接続か

[最新GPT §§10–13][G]の修正を、旧packetの単純な段階論より優先して読む。

\[
 G_m=F_m+N_m(C_m+P),\qquad
 G_c-G_s=(F_c-F_s)+N_cC_c-N_sC_s+(N_c-N_s)P.
\]

Cは全controlled wrapperの一shot費用、Pは共通per-shot状態準備、Fは固定費。同じmetric・同じ実行可能taskでのみ比較する。
**整数shot削減は総費用改善の必要条件でも十分条件でもない。** 同数でも単価が下がれば改善し得るし、shotsが減っても単価が高ければ悪化し得る。
Stage S同数を理由にStage Cの全可能性を否定しない一方、witnessからStage Cへ自動進行もしない。

共通Pはshotsが異なれば相殺されず、N_s<N_cならP増加は上式の絶対差をshared有利へ動かす。
共通固定費が比を薄める場合と混同しない。未知のP、HD実装、control、synthesis、較正価格を0に置かない。
T/CX/RZ/depthは別metric、peak workspaceは加算費用ではない。compiler標本数を量子shotsとして二重課金しない。

Task Bにはtarget state/overlap、energy prior、branch初期化、全dyadic round、raw位相→E*Tのbridge、
残余位相予算、成功条件、実装誤差・state preparationのscopeが要る。保存T=1,4だけで完全なround列は揃わない。
論理モデルの研究に必ず全実機SPAM測定が必要という普遍条件は設けず、選ぶscopeの仮定と未取得情報を区別する。
新契約・科学条件は今回作らず、採択しない。

## 8. 科学レビューの一致点・相違点・時点差

本節は原文の判断の要約であり、今回の新しい科学的判定ではない。
指定された[C]、[G]は全文取得・読解した。旧Claude `レビューinner.md`、`レビューinner(1).md`、
`レビューinner(2).md` は今回未取得で、旧GPTが引用したClaudeの判断は**二次記録**として扱う。

| 論点 | Claude原本 | 最新GPT原本 | 引継ぎでの扱い |
|---|---|---|---|
| 新primitive | 方向制御qDRIFTの自然な特殊形、新規性は弱い（§0,3） | 強い新primitiveを不支持（§12） | 大筋一致。partial全体にはHD sandwichが残るというGPT修正を明記 |
| 理論の既知性 | 標準技法への直接適用を強く評価。ただし有限減衰statement・Davies包含は未判定（§4,7） | MF小誤差boundだけから全C-sec保証は出ない。priority未確定（§5–6,12） | 「全理論が既知と証明済み」「新定理priority確立」の両方を避ける |
| 分散 | 「項内量子分散だけ半減」（§4） | P_j−hの中心化分散と定義すべき（§7） | W/v/ρの原式は保持し、w=E Var(P_j−h)を記録 |
| 分子の明示上界 | 分子では空と強く表現（§4） | 常に空という全称断定を不支持（§6.2） | 実用的定量有効性は未実証、次元/gap保守性を明記 |
| G6 | 有効な比較を実質G3と評価（§5） | 小さいG6絶対差も解像済みとして保持（§8） | 観測を消さず、重要性は未確定とする |
| Stage S | 未完了、整数shots未検証という時点（§6–8） | 8局所witnessを取り込むが実用GOにしない（§9） | 原本を訂正上書きせず、時点差として記録 |
| 状態準備 | 共通費は利得比を薄めるとの一般表現（§6.2） | per-shotと固定費を区別し一般化を修正（§10） | 上記費用式で支払い頻度を明示 |
| 次のgate | 頑健な整数削減の場合だけ費用一点判定（§8,10） | 整数削減は費用改善の必要条件でも十分条件でもない（§10–11） | 整数gateを普遍的なStage C必須条件にしない |
| 研究方針 | 原則限定ノートで区切り、PR-2等へ戻す推奨（§0,8） | 記録保全優先、資源路線自動進行なし。他研究との定量優先比較は未実施（§12–13） | 記録作成のみ今回承認。研究終了・他trackへの転換は未採択 |

旧Checkpoint 2ではClaudeのJensen floorをGPTが受理した一方、ρ単独の一般総資源STOPは不採用。
最新Claude§12はその旧STOP案を撤回し、GPTの留保に同意している。
初期の「毎shot freshでなければ平均恒等式が成立しない」、局所高次誤差を落とした候補上界、
full-H stateへのh固有state係数の単純代入は、旧レビューで修正・限定された履歴として読む。

## 9. 先行研究と未確定の還元

固定[文献調査][SURVEY]は28論文PDF＋2書誌の30レコード。全論文全証明の読了や網羅的新規性証明ではない。
今回追加の文献検索・新レビューは実施していない。調査とC/Gの比較内容を整理した。

| 先行対象 | 接点 | なお区別するもの |
|---|---|---|
| Campbell | controlled qDRIFT | 方向制御と相対時間、native lowering |
| GüntherほかPR | partial evolution、directional deterministic部、独立RTE、RPE | Taylor/normalizer、joint law、HD実装、成功条件 |
| MLMC-qDRIFT | coherent二枝の共有index | fine/coarse・同方向・異角/深さと、M1-S反対符号干渉 |
| COS | 共有permutationの対称word・mean operator | 非復元permutationとIID復元samplingのlaw |
| MF | Hermitian random exponentialの平均演算子・分散bound | 理想unitary小誤差と有限減衰C-sec参照・共鳴floor |
| Chen | random productの集中 | 単一実現とensemble first moment、共有word内の依存 |
| Abbott | coherent controlの実装依存cross block | system marginalから干渉信号は決まらない |
| Gu / Yi–Crosson | 位相mode・spectral摂動 | noisy channel／決定論unitaryと散逸平均写像、gap・norm |
| Davies型平均化 | secular/弱結合の背景 | 原論文全文未取得、特定定理の全仮定・定数への対応未確定 |

共有乱数・対称積・HS散逸・Taylor/Duhamel/Riesz・Hoeffding/円板幾何・費用×shotsの会計の技法そのものに、
強い新規性を置かないという点はC/Gで共通。特定partial lawと有限減衰一様statementの完全な直接還元、
同一既発表例の存在/不存在は未確定。未発見を不存在証明、標準技法を全成果既知の証明にしない。

## 10. 未解決事項と記録としての停止位置

理論面では固定δ・r変更の運用極限、一般DF event、HD内部PF、一般guiding state/縮退、
state-specificに実用的な定数、非冗長Pauliでの共鳴例、先行定理への完全な対応が残る。
実装・資源面ではwhole-wrapper、HD、状態準備、同じ精度・利用情報の強いbaseline、Task B全roundが未取得。
これらは[再開ハンドオフ](codex_restart_handoff.md)で、候補ごとの必要証拠・担当・承認を区別する。

元の停止・失敗・未解像・消費済one-shotを保全する。finite-stepとγ V1には、初回失敗後の
**後続のユーザー明示修正・完了承認**による別S/C/A・別試行がある。無承認自動retryと解釈しない。
独立Gate 0照合以降の保存信号解析は、独立したHamiltonian再生成を増やしたものではない。
詳細な来歴は[証拠索引](evidence_and_results_index.md)。

本資料は既存記録の統合までで、論文執筆・新規証明・解析・simulation・native compile・Stage C実装/契約作成は0。
既存科学status/manifest/契約/authorization/grant/ledger/markerは変更しない。
**公開・保全確認後mandatory STOP。研究再開、正式終了、継続方針の採択はユーザーへ返す。**

[C]: ../m1s_prior_art_novelty_and_research_value_claude_review_20261017.md
[G]: ../m1s_integrated_reassessment_claude_prior_reviews_stage_s_20261017.md
[OPS]: ../research_project_instructions_unified_2026-10-17.md
[T1A]: ../../tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[T1B]: ../../tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[CP1]: ../track_b_m1s_g0t1_checkpoint1_gpt_independent_scientific_review_20261016.md
[CP2]: ../track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[ADD]: ../../tracks/algorithm_codesign/m1s_g0t1_post_checkpoint2_theory_addendum_draft_20261016.md
[G0]: ../../tracks/algorithm_codesign/m1s_gate0_v5_one_shot_results_and_stop_20261016.md
[IND]: ../../tracks/algorithm_codesign/m1s_independent_one_shot_results_and_stop_20261016.md
[FINITE]: ../../tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md
[FP]: ../m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md
[V1]: ../../tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md
[PI3]: ../../tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md
[SS]: ../m1s_stage_s_scientific_conditions_review_20261017.md
[STAGE]: ../../tracks/algorithm_codesign/m1s_stage_s_one_shot_results_and_stop_20261017.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/README.md
