# R1.5後のRA-RTE数学設計 v1

2026-10-06。GPT側の設計・導出。**研究上の新規性認定、実装承認、R2科学実行authorizationではない。**

## 要旨

本設計は、R0の有限平均保存familyを基礎に、R1.5で明確になった「normalization・実合成費用・合成誤差によるshot予算の三者競合」を扱う。

推奨する最初の解法は、**合成候補表を固定し、Taylor係数をその候補へ配分するRA-RTE**である。連続angleの費用が滑らかだと仮定しない。canonical samplingを使う一block問題では、normalizationの逆数と抽出確率へ変数を変換すると、**固定shot数におけるBernstein十分条件、平均保存、誤差・資源制約を線形計画にできる**。

この導出は本書で与える。独立したCodex検証は未実施であり、性能向上は未測定。LP、辞書型合成、分数計画の正規化、importance sampling自体は既知である。研究候補となるのは、有限RTEの次数構造・phase-preserving実装・有限confidence会計をそろえた構成と、その限定最適性・取得費用・実装上の意義である。

---

## 0. 入力・証拠・今回行ったこと

### 0.1 固定入力

- R1.5: `af3d014d0a0cfcbbd25bb544f6544652fec92942`
  - `docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md`
  - `docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md`
- R1結果: `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`
- R1 source: `d43d64a821a0249a0dfab12a2472bd3a72fdee74`
  - `docs/tracks/algorithm_codesign/rte_reallocation_r1_native_semantics_v1.md`
  - `docs/tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md`
- R0.5: `61dd534567fda5c7348fdc688814089eb26a3561`
  - `docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md`
- R0: `672d6bc667eaa7b9ca4979b012f1530499d701b8`
  - `docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md`
- 過去STOP索引: `0da4d18acf3f5d32d1bc32c9661b667885bcf5f2`
  - `docs/tracks/algorithm_codesign/research_redesign_handoff_20261005.md`
- Track Aの論文境界: `4c23453c541700c6a41ba71fc5ec9323b53858d6`
  - `docs/research/track_a_post_pm2_claim_evidence_map.md`

### 0.2 証拠から出発する点

R1.5はR1保存値のPOSTHOC帰属であり、独立validationではない。登録されたdistinct-basis controlledの三座標task front `(G_T,G_CX,G_1Q)` にAが残るが、これは全資源・全実装・全precisionについての優位ではない。

- x=1/4, native epsilon=1e-3, A/PTSC-K0: B²比0.9945715353、N比0.9055130347、E[T]比0.9094176195、G_T比0.8234895084。
- x=1/8, native epsilon=1e-4では、Aのshot数は少し多いが、E[T]減少がこれを上回る。
- x=1/4, A:1e-4のfront残留はG_1Q座標に依存する。G_T/G_CXは同precision PTSC-K0の方が低い。
- 全primary arm/x組で、最も厳しい登録native precision=1e-6は最小G_Tではなかった。
- 以上は固定pygridsynth設定、保存strict error upper、Bernstein十分shot数、加法的native費用の関係である。真のsignal bias・最小必要shots・別compilerでの頑健性は未実証。

### 0.3 今回の操作

GitHub上の固定資料と一次文献を読み、本書の数学を導出した。正規化Bernstein式、正根、二角度の係数matchingについて記号恒等式の自己検算を行った。

R1/R1.5データの新しいresource採点、登録targetの合成、分子計算、回路生成、trajectory、R2、リポジトリ変更は行っていない。自己検算は独立再現ではない。

---

## 1. 研究方針と、以前の説明から修正する点

### 1.1 主RQ

**固定したfinite Taylor meanを保ちながら、eventの次数・rotation角・合成実装/精度・抽出法を設計し、共通の有限confidence条件下で有用な資源点を生成できるか。**

第一版は一block、canonical sampling、有限候補表へ限定する。これは解法を正しく構成するための提案scopeであり、PR全体・一般LCUの最適化問題を解いたという主張ではない。

### 1.2 採用しない説明

1. `min B`が総資源を最小化するとは考えない。
2. etaを細かく探索すれば研究になるとは考えない。
3. B²・word長などのproxyだけで候補を落とし、残りだけ合成する処理を安全とは呼ばない。proxyで劣るangleが実合成で安い可能性がある。
4. 最適解がordinary/A端点なら即STOP、とはしない。端点が最良であることを証明できる場合にも限定的な設計価値はある。ただし新しいrepresentationの実利とは区別する。
5. generic LCU/LPでも同じ答えが出ることを棄却理由にしない。同じ問題を解けば同じ答えが自然である。実際の構成・入力・保証・取得手順の差を問う。
6. 名前をresource-awareへ変更すること、候補表を使うこと、Bernsteinを採用することだけを新規性にしない。

### 1.3 成功していた過去部品は共通baselineへ戻す

P-Aのrun-level basis policyは残す。interval DPの停止は解除しない。P-D/B-Fの選択不変、FRのbound改善とdecision relevanceの違い、SPの累積overhead、BSの同一性、B-Sの固定標本ISの小headroomを区別する。

Track Aは既知残差処理の資源事例研究として独立に維持する。Bの成功をA原稿の完成条件にしない。

---

## 2. 記号と元のfamily

\[
\widehat R=\sum_\ell p_\ell Q_\ell,\quad p_\ell\ge0,\quad\sum_\ell p_\ell=1,
\quad Q_\ell^\dagger=Q_\ell,\quad Q_\ell^2=I.
\]

相互可換性、global Pauli closure、dense operatorは仮定しない。`Qを使える`だけで任意角rotationやcontrolled-Qが無料とは仮定しない。nativeの生成法は別入力である。

\[
F_k=(-i\sigma)^k\widehat R^k,\qquad
M=\sum_{k=0}^{m}t_kF_k,\qquad t_k=x^k/k!,\quad m=2d+1.
\]

元R0 family:

\[
a_k+b_{k-1}=t_k,\quad b_{-1}=b_m=0,\quad a_k,b_k\ge0.
\]

\[
c_k=\sqrt{a_k^2+b_k^2},\qquad
U_k=(-i\sigma)^k e^{-i\sigma\phi_kQ_0}Q_k\cdots Q_1,
\quad \phi_k=\operatorname{atan2}(b_k,a_k).
\]

\(\sum c_k\mathbb E U_k=M\)。wordとrotationのindexは独立にpから引く。finite meanのみの等式で、channelや一branchの同一性ではない。

R0の限定最小値は

\[
B_* =\sqrt{E_d^2+O_d^2}.
\]

これを基準として残す。任意LCU全体の下界にはしない。

---

## 3. 合成候補表による再定式化

### 3.1 一つのcolumnは「event family＋実装」

有限集合 \(\mathcal J\) を固定する。column jには次を含める。

- Taylor次数 k_j。
- ideal角 \(\phi_j\in[0,\pi/2]\)。k_j=mならphi_j=0だけ。
- time sign、phase、literal/complement lowering。
- conditional IID word law。
- 決定論的native gate実装と合成precision。
- primitive/circuit identityと検証済みerror bound。
- conditional expected resource \(C_{j,Q}\) と、その取得費用・不確かさ。

\[
V_j(\omega)=(-i\sigma)^{k_j} e^{-i\sigma\phi_jQ_0}Q_{k_j}\cdots Q_1.
\]

同じideal角でも異なる合成精度・loweringは別columnである。native費用をsmooth関数として補間しない。

### 3.2 次数行列

\[
D_{k_j,j}=\cos\phi_j,\qquad
D_{k_j+1,j}=\sin\phi_j,
\]

それ以外は0。terminal columnは \(D_{m,j}=1\) のみ。

\[
\mathbb E[V_j]=\sum_{r=0}^{m}D_{rj}F_r.
\]

したがって非負係数 \(w_j\) が

\[
\boxed{Dw=t,\quad w\ge0}
\]

を満たせば、\(\sum_jw_j\mathbb E[V_j]=M\)。次数行列は(m+1)行、各columnは最大2非zero成分である。

**これは特定Qに対する必要十分条件ではなく、一般involutionに対して保証する十分なformal coefficient matchingである。** 特定Pauli関係を使うCTSやoperator LCUは、これより広い相殺を使える。

### 3.3 意図的なfamily拡張

元のA-familyは「各次数に一つのangle」であった。本案は同じ次数へ複数angle/precisionのcolumnを許す。実行時はそのどれか一つを抽出する。

これは元familyの厳密な単なる変数変換ではなく、**有限mixed-column familyへの拡張**である。実行可能なrandomized algorithmなので、非物理的なrelaxationではない。

元の一角度/次数制約を維持するなら、その制約を明記して離散選択やmixed-integer問題に戻す。そちらに以下のLP最適性を無断で流用しない。

### 3.4 原始的な実行手順

\(B=\sum_jw_j\)、\(q_j=w_j/B\) とする。

1. jをqから選ぶ。
2. jに指定したconditional IID lawからword indexを生成する。
3. 位相を保持したcontrolled eventを実装する。
4. Hadamard outcome \(Y_a\in\{-1,1\}\) を得て \(X_a=BY_a\) とする。

理想回路では\(\mathbb E X_a\)がMのcoherent信号の該当軸に一致する。別occurrenceは独立にsampleする。回路のコンパイル結果を再利用することと、random eventをoccurrence間で共有することを区別する。

---

## 4. 候補角への分解と、限定的な完全性

任意の元familyのvector \(c(\cos\phi,\sin\phi)\) を、\(\phi_-\le\phi\le\phi_+\)、\(\Delta=\phi_+-\phi_-<\pi\) の二つのcatalogue方向へ分解できる。

\[
\lambda_-=c\frac{\sin(\phi_+-\phi)}{\sin\Delta},\qquad
\lambda_+=c\frac{\sin(\phi-\phi_-)}{\sin\Delta}.
\]

\[
\lambda_-v(\phi_-)+\lambda_+v(\phi_+)=cv(\phi),\qquad\lambda_\pm\ge0.
\]

normalization倍率は

\[
\frac{\lambda_-+\lambda_+}{c}
=\frac{\cos(\phi-(\phi_-+\phi_+)/2)}{\cos(\Delta/2)}
\le\sec(\Delta/2).
\]

全必要角を最大間隔Delta_maxで挟むcatalogueなら、任意の元representationに対し、同じfinite meanを保つcatalogue representationがあり、B増加は最大sec(Delta_max/2)倍で抑えられる。

これは既知の二次元円錐・角度補間の恒等式の本問題への適用であり、独立の新原理とはしない。**T費用・bias・最終Gに同じ近似比が成立するとは言えない。** 離散合成costの補間に関する仮定を追加していないためである。

ordinary/A/PTSC-K0の使用columnを最初から含めれば、それらのexact ideal representationは候補表内で再現できる。これは探索空間の包含であって、strictな改善の保証ではない。

またmixed-columnでも、偶奇を交換したunit vectorの和に三角不等式を適用することで\(B\ge B_*\)が保たれる。A columnsを含むなら、normalizationだけの最適値は元のB_*である。

---

## 5. 誤差とcostの入力契約

### 5.1 一blockのcoherent bias

j内のword omegaについて、phase-preserving joint unitaryのoperator誤差がdelta_(j,omega)以下なら、一般のjoint実装で

\[
d_j\ge 2\mathbb E_{\omega|j}\delta_{j,\omega}
\]

を安全なconditional observable-error boundとして使える。R1のfactor2と整合する。

\[
\beta_{\mathrm{impl}}\le\sum_jw_jd_j.
\]

共通に予約するbias \(\beta_0\) を含め、axis許容値epsilon_aに対して

\[
e=\epsilon_a-\beta_0>0,\qquad
s=e-d^Tw>0.
\]

有限Mそのものがtargetなら、exponential truncationを勝手に加算しない。exact evolutionがtaskなら、その共通biasをbeta_0等へ明示して戻す。

### 5.2 最適化するのは真のbiasではなく上界

R1.5の小さいbias upperは、真のbiasが同じ比率で小さいという測定ではない。本算法の保証は指定されたuniform boundの下での保証であり、実際の期待値誤差を最小化したという主張ではない。

### 5.3 条件付きexpected cost

\[
C_{j,Q}=\mathbb E_{\omega|j}C_Q(j,\omega),\qquad
\bar C_Q=\sum_jq_jC_{j,Q}.
\]

QはT、CX、1Q等。basis、odd extra Q、control phase、共通簡約、状態準備・readoutを計上する。

- whole-event costが保存/取得されている場合は、その同じ境界を全方式に使う。
- additive costしかない場合、actual whole-wrapper最適compileと呼ばない。
- word数が膨大な場合、C_jの取得を無料としない。
- 解析的期待値、認証上界、経験的点推定を別ラベルにする。点推定だけのLP解をresource certificateとは呼ばない。

### 5.4 取得費用

catalogueに対して取得した全合成、error certification、未採用columns、solver、sampler前処理を数える。採用columnだけの費用を報告しない。

次数行列の小ささは、native cost取得の小ささを保証しない。特にDFのsupport-union/cancellationを含む期待costの取得は未解決の実装課題である。

---

## 6. 中心の導出：固定shot数のLP

以下は本書の新しい定式化であり、R0/R1で検証済みの成果ではない。

### 6.1 元の十分条件

canonical estimatorは\(|X_a|=B\)、\(\operatorname{Var}(X_a)\le B^2\)、centered range≤2B。
\(\ell=\log(2/\alpha_a)\) とする。R1の保守的Bernstein規則は

\[
n\ge \ell\frac{2B^2+\tfrac43Bs}{s^2}.
\]

nはaxisあたりの独立測定回数であり、真の最小必要shotsではない。

### 6.2 正規化変数

\[
q_j=\frac{w_j}{B},\qquad y=\frac1B.
\]

\[
Dq=yt,\quad\mathbf1^Tq=1,\quad q\ge0.
\]

正規化したstatistical余裕は

\[
h=\frac{s}{B}=ey-d^Tq.
\]

よって

\[
\boxed{n\ge\ell\frac{2+\tfrac43h}{h^2}.}
\]

右辺はh>0で単調減少する。

\[
\kappa_n=
\frac{\tfrac43\ell+\sqrt{(\tfrac43\ell)^2+8n\ell}}{2n}
\]

と置くと、指定nで十分条件を満たすことは\(h\ge\kappa_n\)と同値である。

### 6.3 命題

有限catalogue、固定conditional law、固定d/C、canonical sampling、固定nの下で、次の実行可能集合は元の十分条件と一対一対応する。

\[
\boxed{
Dq=yt,\quad\mathbf1^Tq=1,\quad q\ge0,\quad y\ge0,
\quad ey-d^Tq\ge\kappa_n.
}
\]

最後の条件からy>0。したがってw=q/yで元の係数へ戻れる。逆に任意の元のfeasible wはこの集合へ写る。

一資源の期待費用を最小化するなら、目的は\(C_Q^Tq\)。全て線形である。

総resource budgetを指定する場合も、例えば両軸同n、共通context h_Qなら

\[
2n(C_Q^Tq+h_Q)+C_{Q,once}\le L_Q
\]

を足すだけで線形のままである。Re/Imのoverheadが異なるなら、2h_Qではなくその和を使う。

### 6.4 証明の射程

- ideal coefficient matchingをexactに保存する。
- finite synthesis bias upperとBernstein range項を落としていない。
- fixed nではLPのglobal optimumを求められる。
- ただし候補表外の角度、非canonical sampling、multi-block共同最適、真の最小shots、物理fault-tolerance resourceは対象外。
- 正規化の着想はCharnes–Cooper型の分数計画変換と共通であり、その一般手法を発明したとはしない。

### 6.5 二軸の扱い

主導出は共通n/epsilon_a/alpha_aと共通dを使った対称会計。Re/Im別のd/e/alphaを採る場合、固定(n_Re,n_Im)に対し対応する線形余裕条件を二本置けばよい。抽出qを共有するか軸別にするかは先に固定する。

---

## 7. nをどう選ぶか

### 7.1 正確な定義

1≤n≤N_maxの全整数nについてLPを解き、総資源の非劣点を取れば、その有限候補・canonical・指定十分条件における解になる。

ただし百万個のnを逐一解く実装は不要。次の幾何grid近似またはbounds付き探索を使える。

### 7.2 幾何gridの保証

任意の整数nに対し、n以上で最初のgrid値\(\hat n\)が\(\hat n/n\le r\)を満たすgridを用意する。

nでfeasibleな同じ(q,y)は\(\hat n\ge n\)でもfeasible。したがってshotに比例する全非負resourceは最大r倍となる。一度だけの非負setup costを加えてもこの上界は保たれる。

つまり、**各nでの全resource frontierを保持する理想的な解法**なら、gridはfull整数n問題のfrontをfactor rで被覆する。

実装で有限個のweighted sumだけを解いた場合、全frontを取得したとは限らない。この保証をそのまま「全Pareto保証」にしない。所定resource budgetsのfeasibility、または明示したoperating pointsについての保証として使う。

\(n_{i+1}=\lceil(1+\delta)n_i\rceil\)なら、下端n_min以上でr≤1+delta+1/n_min。cap末端も含める。

### 7.3 一資源のouter bounds

他resourceのtotal capを付けない一資源問題で、v(n)=最小per-shot costはnについて非増加。
よってn∈[n_L,n_R]の総costには\(n_Lv(n_R)\)というlower boundを使える。

**total CX cap等をn依存で課す場合、feasible setの入れ子性は失われ得るので、このmonotonic boundを流用しない。**

### 7.4 疎解の位置付け

固定n、一資源objective、exact coefficient modelで最適極点が存在するなら、LPの基本解として疎なqを選べる。
独立equalityは高々m+2本、confidence inequality一本とr本の追加linear resource inequalitiesなら、y>0を含めた基本変数計数からqのsupportは高々m+2+rとなる基本解が存在する。

これは一般LPの性質の適用である。interval residual補助変数、追加robust scenario、複雑なselection constraintを入れた場合は数え直す。solverが必ずその疎解を返すとは限らない。

---

## 8. numerical・robust版

### 8.1 exact meanと有限bit実装を分ける

理想式Dw=tはexactである。しかしsolver出力・probabilities・anglesは有限precisionで扱う。
実装した有理(q,y)について

\[
r=Dq-yt
\]

が残るなら、\(\|\widehat R\|\le1\)よりnormalized operator residualは\(\|r\|_1\)で抑えられ、補正後biasは\(\|r\|_1/y\)以下。

従って

\[
ey-d^Tq-\xi\ge\kappa_n,\qquad \xi\ge\|Dq-yt\|_1
\]

で残差を戻せる。平均保存を研究の前提にする場合、xiを自由な新しいapproximation budgetにせず、事前固定した非常に小さいnumerical budget、例えば\(\xi\le y\delta_{num}\)へ制限する。

### 8.2 interval係数

Dとtのmidpoint/radiusが既知なら、各行のworst-case residualを

\[
z_r\ge\pm(D_{mid}q-y t_{mid})_r+(D_{rad}q)_r+y(t_{rad})_r
\]

で囲める。q,y≥0を使用する。\(\xi\ge\sum_rz_r\)。有限tableのdはupper、resourceは必要に応じupperを使用する。

これは区間相関を捨てた保守的certificateで、丸めたnominal LPの最適値そのものではない。nominal/robust modelを別保存する。

### 8.3 不確かさscenario

同じideal columnsを異なる固定backend scenario sで実装する場合、各sのd^(s),C^(s)を用い、すべての余裕/予算制約を置けばLPのまま。

頑健性は登録scenario集合に対するものに限定する。R1.5には一backendしかなく、今すでにrobust性が実証されているわけではない。

### 8.4 安全なpruning

同じideal D column、同じconditional semanticsを持つ二つの実装で、一方が全resourceとerror boundで劣らなければ劣る実装を落とせる。

異なる角度・異なるD columnを、B proxyやword長だけで落とす処理にはこの保証がない。効率化のためにproxyを使うなら、ranking/取得順序に留めるか、明示的なvalid lower boundが必要。

---

## 9. dualと候補取得

一資源の基本LPをmin C^Tqとする。dualは以下で書ける（e=epsilon_axis-beta0）。

\[
\max_{u\in\mathbb R^{m+1},\zeta\in\mathbb R,\lambda\ge0}
\zeta+\lambda\kappa_n
\]

subject to

\[
D_j^Tu+\zeta\le C_j+\lambda d_j\quad\forall j,
\qquad t^Tu\ge\lambda e.
\]

uとzetaはfree、lambdaだけがnonnegative。primal/dual gapを有限table内の最適化監査に使える。

未追加column jのreduced costは

\[
C_j+\lambda d_j-D_j^Tu-\zeta.
\]

負ならcurrent dualを破る候補であり、取得の優先付けに使える。ただしdegeneracy等によりstrict improvementを保証するわけではない。

**未合成angleのC_j/d_jを知らずに「全連続angleに負のreduced costがない」と証明してはいけない。**
column generation自体も既知の最適化技術で、新規性にはしない。最初の監査/pilotは固定finite tableを対象とし、適応的column取得は別の手順・予算が確定してからにする。

---

## 10. 強いbaseline

### 10.1 必須の段階的対照

| 比較 | 目的 |
|---|---|
| ordinary / A / PTSC-K0の各precision envelope | R1.5の既知選択からどれだけ進むか |
| 同一固定representation内のfamily別precision配分 | 改善が精度配分だけで説明されないか |
| 完成済みwhole-ensembleの混合 | 単純な既存方式混合を超えるか |
| degree-column RA-RTE | Taylor係数をevent family間で組み替える利益 |
| 同じD/d/Cを与えたgeneric LP | correctness/取得時間。値の一致は失敗ではない |
| 固定ensemble＋有限confidence IS | canonical-only対照が不当に弱くないか |
| I1が利用できる範囲のcollected CTS | I0-style制約による見かけの優位を分ける |

whole-ensemble mixingでは、各元representation rのgroup係数の相対比を保ち、その全体weightだけを混ぜる。
これに対しRA-RTEは各degree/angleへ係数を再配分できる。両者を同じものとして報告しない。

元方式rのB_r、原normalized profileを持つwhole-family columnは\(D_r=t/B_r\)として書ける。
canonicalな混合は\(\sum_r q_r/B_r=y\)を満たす。元方式を選んで元weightをそのまま使う非canonical混合は、別にそのmoment/rangeを数える。

### 10.2 fixed-ensemble ISの有限confidence比較

固定representation係数w_j、固定実装d/Cならbias bound\(\beta=\beta_0+d^Tw\)はsampling確率r_jを変えても不変。

\[
V(r)=\sum_j\frac{w_j^2}{r_j},\qquad R(r)=\max_j\frac{w_j}{r_j},\quad s=\epsilon_a-\beta.
\]

固定nで

\[
2V(r)+\tfrac43sR(r)\le ns^2/\ell
\]

を満たしながらmin C^Trを解く問題は凸である。正のsupport上でr>0、sum r=1。
quadratic-over-linearとreciprocalのepigraphによりSOCPで実装可能。

ここにはC=0による1/sqrt(C)の特異性はない。zero-weight/support処理は別途固定する。jがevent familyを表すなら、この式はfamily-level ISである。word-label別のISを使う対照には、その情報access・費用取得・生成手順を新旧方式へ対称に与える。

### 10.3 既知のleading-order joint optimum

rangeとbiasを無視し、C_j>0の場合、既知ISのCauchy–Schwarz評価から

\[
\min_r(C^Tr)\sum_jw_j^2/r_j=(\sum_jw_j\sqrt{C_j})^2.
\]

従って有限Dでrepresentationも選ぶなら、leading objectiveは

\[
\min_{w\ge0,Dw=t}\sum_jw_j\sqrt{C_j}
\]

というweighted LPへ帰着する。**この帰着を新しいRA-RTEの独立発明として数えない。**
今回の中心は、R1.5で重要だったfinite bias/range/integer shotsを落とさないことである。

### 10.4 representationと非canonical samplingのjoint拡張

固定n、range上限R_bar、implementation bias予算beta_barなら、s=epsilon_a-beta0-beta_barを固定し、

\[
Dw=t,\quad w\ge0,\quad r\ge0,\quad\sum r=1,\quad d^Tw\le\bar\beta,
\quad w_j\le\bar R r_j,
\]

\[
\sum_j w_j^2/r_j\le ns^2/(2\ell)-\tfrac23\bar R s
\]

を使うSOCPが得られる。ここでR_barとbeta_barを固定したことが重要。両者まで自由にした全問題のglobal convexityは主張しない。

最初から外側gridを増やさず、canonical LPを最初の実装対象とする案。その最終優位性の主張前にはfixed-ensemble IS等へ戻す。

---

## 11. controlled構造を保つnative実装

### 11.1 R1の境界

R1は独立に合成した±angle等を使うため、approximate controlled loweringでcontrol=0が厳密identityとは限らない。そのためfactor2のjoint-unitary boundを使用している。

一blockの本LPはこのままd_jに取り込める。しかしmean contractionのmulti-block式を適用するには条件が増える。

### 11.2 共通改善としてのadjoint pairing

controlled RZのtarget側half rotationを近似Aで実装し、反対符号を**同じgate列の正確なadjoint A†**で実装すると、

\[
\mathrm{CX}\,(I\otimes A^\dagger)\,\mathrm{CX}\,(I\otimes A)
=|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes X A^\dagger X A.
\]

AがRZ(theta/2)のdelta近似なら、1 branchはRZ(theta)の高々2delta近似である。0 branchは正確にI。

conjugatorにも同じ近似列とそのadjointを対に使えばcontrolled formを保てる。
これは基本的な回路恒等式で、新規性の主張ではない。ordinary/A/PTSC/他方式へ共通に与える強いnative optionである。

独立符号合成とadjoint-paired実装ではcounts/errorが変わり得る。旧R1の列を無言で置き換えない。新しいnative policyとして比較し、旧結果は保持する。

### 11.3 finite-block mean boundを使う条件

物理回路が各branchでexactにCtrl(tilde U)の形なら、tilde M=sum w E tilde Uという平均作用素を使える。
真のM_jがcontractiveでlocal operator errorがe_j以下なら

\[
\|\widetilde M_L\cdots\widetilde M_1-M_L\cdots M_1\|
\le\prod_j(1+e_j)-1.
\]

R0のP3 contractivityは|x|≤sqrt3の範囲。全cutoffへ外挿しない。

physical blockがgeneral joint approximationの場合、上記mean式を使わず、次節のchannel boundを使う。

---

## 12. multi-block：単一blockの結果を掛けて終わらせない

独立block jのcanonical normalizationをB_jとすると、total correctionは\(\mathcal B=\prod_jB_j\)。
ideal first momentはblock順序を保った積である。mean preservationは成立するが、それだけで低資源や低誤差が保証されるわけではない。

### 12.1 一般joint実装の安全会計

ideal/approximate physical channelがCPTPなら、channel errorのtelescopeにより

\[
\beta_{tot}\le\mathcal B\sum_j\bar d_j,
\qquad \bar d_j=\sum_s q_{js}d_{js}.
\]

ここでdは各conditional physical channelのdiamond-norm upper（または合成可能性を別途証明した同等のnorm bound）として使う。固定observable一つの誤差だけではchannel telescopeへ流用できない。2deltaのjoint-unitary boundはこの条件を満たす。単一blockのweighted biasはB_j bar d_j。

normalized statistical余裕は

\[
h_{tot}=\epsilon_a\prod_jy_j-\sum_j\bar d_j
\]

（外部biasがあれば明示的に追加）。product yがあるので、全block同時の問題は一般に単一LPではない。

全体のother blocksを固定し、costがadditiveなら、一blockのcoordinate subproblemがlinearになる場合はある。しかしglobal optimumの保証にはならない。

### 12.2 controlled-formを保証できる場合

前節のmean-level誤差伝播を使い、統計負担はなお\(\mathcal B^2\)、rangeは\(\mathcal B\)を残す。
contractivityはbiasを改善するための性質で、measurement burdenを無料にするものではない。

### 12.3 circuit費用の非加法性

basis cancellationなどがblock境界を越える場合、\(\bar C_{tot}=\sum\bar C_j\)はexactでない。
共通境界状態を持つfinite-state cost model、またはwhole-wrapper評価が必要。

P-Aのrun-level policyは共通部品として維持する。旧interval-DPの独立効果が得られなかったことを、別の確証なく覆さない。

### 12.4 第一完成条件

本v1の数学的完成点は一blockの有限catalogue最適化。multi-blockはscopeを定めた二block semantic/controlから接続する。全RPEやH12を最初の必須条件へ追加しない。

---

## 13. 新規性・既知性の比較

今回の外部確認は関連一次資料の本文・該当箇所に限る。引用網全体の不存在証明ではない。

| 内容 | 位置付け |
|---|---|
| Euler pairing、identity再配分、共通角 | Zeng/PTSC等で既知。R0.5の境界を維持 |
| Taylor/Pauli係数のcollectionとCTS | Peetz–Smart–Narangの強い既知手法。Markov layeringを無視しない |
| gate dictionary＋凸最適化＋誤差/overhead交換 | Sparse Probabilistic Synthesisに直接の先行例 |
| 固定ensembleのcost×moment最適IS | Cugini–Atif–Subasiの既知結果 |
| term-dependent angleのhardware-aware調整 | Structure-Aware Variance ReductionのRemarkに既知構成 |
| 分数計画の正規化・LP/SOCP・dual | 古典最適化の既知技術 |
| R0のadjacent有限familyとclass optimum | 指定corpus内で狭い差が残った候補。priorityは未確定 |
| 本書の有限confidence canonical LPとRTE degree catalogの組 | 本書で導出した研究候補。個々の既知技術の適用以上の論文貢献になるかは未確定 |
| 複数block・DF・別compilerでの資源利益 | 未実証 |

### 13.1 前BS案との違い

BSは特定Mを構成して同じgeneric sparse/operator LCUへ渡す処理であり、独立差が未定義だった。
本案はevent/precision選択を次数matchingとして明示し、指定finite-confidence resource課題を実際に解く形にした。

ただし「汎用LCUへ渡さず小さいDを使う」だけでも新規性は確定しない。同じstructured Dを対照にも与える。
方法として意味を持つ候補は、phaseを含む実行可能性、次数構造、finite-confidence reduction、取得費用、限定保証をまとめた実用的な手順である。

### 13.2 新規性を主張する場合の命題

候補claim:

> 一般Hermitian-involutionのfinite meanを保存するdegree-local ensemble classに対し、有限の合成実装表を用いて、canonical bounded-measurement confidence条件を満たす資源設計をLP系列へ帰着し、指定表内の最適性・shot-grid近似保証・実装誤差を監査可能にする。

除外claim:

- 世界初のcost-aware quantum sampling。
- 世界初のprobabilistic synthesis。
- 全連続angle/任意LCU/全hardwareでの最適性。
- oracle-free end-to-end効率（C/d取得を無料にした場合）。
- 単一toyのPauli展開を禁止したことによるI0 advantage。

---

## 14. 過去全証拠を今回へどう使ったか

| 系列 | 今回への設計上の帰結 |
|---|---|
| P-A / DF structure | 有効だったbasis/run policyは共通化。proxyだけによる安全でないpruningを避ける |
| P-B | exact-state signal異常を仮定せず、全stateに使えるformal mean/errorを使う |
| P-C | geometry外挿による安いcost/bias予測へ依存せず、入力と実装表のidentityを固定 |
| P-D | 弱いwork baselineを作らない。設定を変えない固定mean比較と、task再最適化を分離 |
| R3 | generic最適化の名称変更ではなく、目的・constraints・solver保証を具体化 |
| FR | bound改善と実用差を区別。mean contractivityの適用条件を明示 |
| B-F | 複雑なobjectiveを足しただけでgainを期待しない。precision-only対照を置く |
| B-M | 同じ方程式/同じbackendの値一致を新手法としない。known/general solverへ同条件を与える |
| SP-0.5/SP-1 | sampling補正・位相・複数gateの累積を省略しない。actual RTEとtoy coinを混同しない |
| BS-0.5 | 全state operator targetとchannel targetを混同しない。取得情報とoracle範囲を明示 |
| R0/R0.5 | formal係数制約と既知構成との境界を継承 |
| R1/R1.5 | precisionを実装columnへ入れる。B-onlyではなくbiasとCを同時に扱う |
| Track A | 現論文の固定resource/applicability scopeを維持。新algorithmの成否をAへ逆流させない |

この表は保存成果の再分類ではなく、次の設計で得た教訓である。

---

## 15. 研究としての着地点

### 第一目標

**finite-mean-preserving resource-aware RTEの構成・限定保証・native実装研究。**

構成内容:

1. R0 familyと有限mixed-column拡張。
2. finite-confidence制約の正規化LP。
3. 合成精度とnative費用・biasの共通会計。
4. 取得費用を含む実装。
5. endpoint/precision-only/whole-mixture/IS/CTSとの比較。
6. 有効・不利条件、およびmulti-blockでの範囲。

### 最小の完結点

式・有限table最適性・certificate設計と再現可能な小型実装。性能差が小さくても理論/technical result候補として残るが、独立論文に十分とは自動認定しない。

### 強い着地点

新しいdegree配分または精度配分のjoint設計が、強い対照後にも再利用可能な資源点を生み、古典取得費用と新条件での挙動まで説明できる。

### 狭める条件

- 全利益がfixed endpointのprecision選択/混合だけで説明される。
- full candidate tableを与えた標準手順と完全に同じで、固有の保証・取得差・実装知見も残らない。
- 原結果の特定合成列だけを再選択すると得するが、固定設計手順として再現しない。
- field/multi-block contextを戻すと利益が消え、その制約についても新しい定量知見が残らない。

これらの場合はtechnical/design noteへ縮小する。内点が選ばれなかったこと、全座標strict dominanceしなかったことだけで棄却しない。

---

## 16. 次のCodex作業：一回の統合数学監査

**今回はR2科学実行を指示しない。** 資料作成のためのレビューを際限なく繰り返すのでなく、本書の数学命題・既知法との照合を一つにまとめて独立監査する。

### 読むもの

本書、R0 proof、R0.5 audit、R1 native semantics、R1.5 attribution。指定commitを保持する。

### 監査命題

1. mixed-column Dのmean preservation、終端、位相、negative time。
2. 元one-angle familyとmixed extensionの区別、standard/A/PTSC端点包含。
3. 二角度分解とsec(Delta/2) norm bound、resource boundではない点。
4. q=w/B,y=1/Bの一対一対応、kappa_n根、固定nのLP同値性。
5. finite numerical residualのbias会計。identity/channel条件を混ぜない。
6. fixed-n dual、support bound、n-grid coverageの条件。
7. fixed-ensemble ISのconvexity、joint非canonical拡張で固定すべき変数。
8. adjoint-paired controlled loweringの0 branch identityと誤差上界。
9. multi-blockでmean boundとjoint-channel boundを取り違えていないか。
10. Koczor/IS/PTSC/CTSおよび古典分数計画との差。一般技術を新規と呼んでいないか。

### 許容される最初の技術検証案

- 自由wordのexact係数照合。
- rational unit-circle columns（例として3/5,4/5等）を使うoff-domain bookkeeping。
- 小さい人工cost/error表におけるdirect formulationとnormalized LPの対応確認。
- 元式のsample数・余裕・正根の記号確認。
- 故意にphase、terminal、factor2、sum-probability、one-angle制約を壊したmutationの検出。
- boundedなLP/SOCP toy testを行う場合、科学dataではなく数学検証として明示して記録する。

実合成・新angle計測・R1再実行・R1.5再分類・分子・DF・新geometry・全RPEは行わない。

### 返す成果物

- 各命題の PROVED_UNDER_ASSUMPTIONS / COUNTEREXAMPLE / UNRESOLVED。
- 既知法とのclaim-level表。
- 採用できる最小modelと修正が必要な仮定。
- 次のsmall pilotが何を反証するのかを一つに絞った未承認案。

結果を受けたalgorithm採択、論文着地点、science scopeはGPTへ戻す。

---

## 17. その後の小型pilotの設計原則

数学監査を通過した場合にだけ契約を作る。

- R1.5はdevelopment入力。R1のwinning angle近傍だけを増やさない。
- first targetはfinite mean固定、m/split/PF/timeを同時に再探索しない。
- candidate tableは結果前の生成規則・上限を持つ。費用は未選択columnも含める。
- ordinary/A/PTSCのprecision envelopeとwhole-mixtureを公平に戻す。
- actual resource優位の主張前にはfixed-ensemble ISとI1利用可能なcontrolを入れる。
- T/CX/1Q/Nshots/workspace/classicalを保持し、一つのscalarへ後付けで潰さない。
- 同じq,yが異なるbackendでも必ず良いとは仮定しない。設計手順のtransferと係数固定transferを区別する。
- 追加backend/geometryの選択は現在の結果を救うためではなく、主張する適用範囲を検証するために固定する。
- 全outcomeでSTOPし、結果をGPTへ戻す。R2・DF・次段へ自動進行しない。

---

## 18. 主要一次文献と今回の確認範囲

1. Cugini, Atif, Subasi, **Resource-Optimal Importance Sampling for Randomized Quantum Algorithms**, arXiv:2603.13495v1.
   - https://arxiv.org/html/2603.13495v1
   - §II Eq.(7)–(13)、§III bias preservationを確認。固定protocolのsampling最適化で、underlying circuits自体を変えないというscopeを使用。
2. Bálint Koczor, **Sparse Probabilistic Synthesis of Quantum Operations**, arXiv:2402.15550v2 / PRX Quantum 5, 040352.
   - https://arxiv.org/html/2402.15550v2
   - §II.1–II.3、dictionary/convex optimization/error trade-offを確認。
3. Pei Zeng, Jinzhao Sun, Liang Jiang, Qi Zhao, **Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations**, arXiv:2212.04566v2.
   - https://arxiv.org/pdf/2212.04566v2
   - 本文§IV.B Eqs.(73)–(79)のparsed textとR0.5監査を照合。Web PDF screenshot取得はエラーだったため、新たな図版の視覚確認をしたとはしない。
4. Joseph Peetz, Scott E. Smart, Prineha Narang, **Quantum Simulation via Stochastic Combination of Unitaries**, arXiv:2407.21095v2.
   - https://arxiv.org/html/2407.21095v2
   - Supplementary Note 3のpartial expansion/layeringとnorm増加、およびCTSの資源・confidence記述を確認。
5. **Structure-Aware Variance Reduction for Unbiased Randomized Hamiltonian Simulation**, arXiv:2606.23544v1.
   - https://arxiv.org/html/2606.23544v1
   - §III.1 term-dependent TE-PAI angleのhardware-aware trade-offを確認。平均保存・cost-aware angleという一般原理を新規性としない。
6. A. Charnes and W. W. Cooper, **Programming with linear fractional functionals**, Naval Research Logistics Quarterly 9, 181–186 (1962).
   - https://doi.org/10.1002/nav.3800090303
   - 出版社書誌を確認。今回の正規化を古典分数計画と関連づける出典。全文を新たに精読したとはしない。
7. S. Zionts, **Programming with linear fractional functionals**, Naval Research Logistics Quarterly 15, 449–451 (1968).
   - https://doi.org/10.1002/nav.3800150308
   - 出版社abstractでCharnes–Cooper変換のLP化と分母符号条件を確認。

一次文献の広い枠組みが既知であることは明確。本書と同じrestricted finite-confidence RTE構成が未発表だという証明はしていない。独立監査で既知の直接帰結と判明した場合、prior artを明記したdesign/application studyへ位置付けを修正する。

---

## 結論

R1.5の後に行うべきことは、もう一度保存値を掘ることでも、ただeta gridを増やすことでもない。

**有限平均を保つ次数column、phase-preserving実装表、finite-confidenceの正規化LP**という具体的な数学設計を、まず一回の独立反証へ渡す。

今回、解法の入力・制約・変数変換・保証範囲・baseline・multi-block上の障害まで明記した。性能や新規性は未実証だが、「何をCodexが深く検証すべきか」は、抽象的な最適化候補から検証可能な数学命題へ進んだ。
