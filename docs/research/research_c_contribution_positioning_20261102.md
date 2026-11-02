# 研究C：PR-aware DFの先行研究との差分・科学的貢献候補・論文構成案

2026-11-02 JST。Codex-Bによる、固定済み科学結果と一次文献の限定調査・位置づけ資料。
共通起点は `46f1461d213ad73f8e187ef98e1491ef8b522aaa`、作業branchは
`codex/research-c-contribution-positioning-20261102`。

**現在支持される核は、固定H4条件でのPR-tail低減、候補固有の条件付き資源会計への接続、そして強決定論PFとの資源tradeoffである。** 一般J/Uの最適化、係数正則化、PRとDFの接続、attenuationを含む資源見積りは既存研究にある。研究Cで確認できる目的・実装・数値結果の差分と、独立した科学的新規性・重要性の認定は区別する。本書は後者、論文化GO/NO-GO、中心claimの正式採択、研究Cの継続／終了を決定しない。

新規科学計算、保存NPZ/QPYの数値再解析、新fit、PF/RTE・shots・nativeの再計算、optimizer／cost surrogate／科学契約の開発、GPT/Claudeの詳細科学レビューは行っていない。数値・削減率・比率・会計項は既存報告の引用であり、新しい統一集計や順位は作らない。Codex-Aの新Evidence Mapを待たず、以下の固定資料から独立に作成した。

## 1. 固定資料と参照の境界

### 1.1 必須7資料のcommit確認

各行の固定commitでGitHub公開本文を取得し、返されたblob SHAとローカルGit objectを照合した。表のcommitは**報告を収録する版**であり、各実験を実行した科学SOURCEとは別である。

| ID・資料 | 本書が引用する固定commit | 共通起点tree内の有無・役割 |
|---|---|---|
| [R1 統合実行][R1] | `aced1647b81e8d28091058cdc7b6e3e1ee54896a` | 有。同一blob。update005、DF予算、候補固有PR評価。科学SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。 |
| [R2 同一新J/U・強D][R2] | `38b8d0df39b8943178ecef611194819cff1c2169` | 有。同一blob。同じupdate005のD/PR比較。科学SOURCE `5f27539ae0ab88181d9cfcd45badb54daf855c28`。 |
| [R3 PR資源寄与分析][R3] | `2639221a59713d431bc7c157854628bbc006a1bd` | **無。別branchの公開資料を参照し、mergeしない。** 旧新PRのW・価格・shell・資格を説明。 |
| [R4 Native価格改善の技術整理][R4] | `46f1461d213ad73f8e187ef98e1491ef8b522aaa` | 有。本書の起点。既存負結果と `TECHNICAL_NO_GO_CURRENT_EVIDENCE` の射程。 |
| [R5 10/31 Evidence Map][R5] | `354a090da388c6d9718624bb4525ff3c72a22022` | 有。同一blob。既存23項目と定義・資格・履歴。11/2結果を含む新版とは扱わない。 |
| [R6 正則化比較設計][R6] | `150f904f6975868abe09966dda8ab41722e2a7cc` | 有。同一blob。目的の同値性・非同値性、公平性の境界。 |
| [R7 PR-aware DF初期設計][R7] | `47e0d406226be365951bd8666d5acae61106f500` | 有。同一blob。初出 `94d1c7b4bbed106c3991ef0a5bfd1ffde337807a` 後の10/28追記を含む版。初期未承認表示は当時の履歴。 |

R3の「同一新J/UのD結果は未収録」はその報告時点の記述である。本書の現在の比較にはR2を使用する。旧J/UのDとの比2.04191と、同一新J/Uの比1.97019を混ぜない。[R1]・[R2]・[R3]

補助的に、共通起点内の[RC2 Stage A結果][R8]、[保存一次診断][R9]、[後続RC2/PR native結果][R10]を確認した。これらの有限終点比較は存在するため、「既存手法との数値比較を全くしていない」とは記述しない。同時に、収束・強度調整・原論文solverを含む公平比較の完成とも扱わない。

[VALIDATION_STATUS][VS]と[validation manifest][VM]の該当記録は、統合点の同時一次未達、Dの `NUMERICAL_PILOT_CA_PASS`、反実lane／未資格量を区別している。本書も公開済みlocal evidenceとして扱い、immutable CI・外部独立再現へ昇格しない。古い総論文書の「DF固定」「最終総cost未実施」は従来系列の記述であり、研究Cの後続J/U最適化・条件付き会計を消去する意味には読まない。

### 1.2 一次文献の版・確認範囲

| 文献 | 確認した版 | 本書で確認・使用した箇所 |
|---|---|---|
| Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization* | [arXiv:2212.07957v3][P1]、2024-06-04、Quantum 8, 1371 | 本文A/B/C.4、式5/8/13/15/17、weighted L1/L2の本文、Appendix A式18以降・B。一般係数、正則化、solver、測定／qubitization。 |
| Rocca et al., *Reducing the runtime of fault-tolerant quantum simulations in chemistry through symmetry-compressed double factorization* | [arXiv:2403.03502v1][P2]、2024-03-06、JCTC 20, 4639–4653 | §2.1–2.4、式12/13/15/19/20/22/25–27、結果節の資源表、Appendix A/B。一般Vの前段とrank-one SCDF、shift、費用。 |
| Günther et al., *Phase estimation with partially randomized time evolution* | [arXiv:2503.05647v2][P3]、2026-07-10改訂、PRX Quantum 7, 020332 | §IV.B/V/VI.B/VII.A–B/VIII、式27–39/42–43、Theorem V.1、Appendix A.2–3/C.3/E.3、式A20–29/C14/E19–23。 |
| Cohn, Motta, Parrish, *Quantum Filter Diagonalization with Double-Factorized Hamiltonians* | [arXiv:2104.08957v1][P4]、2021-04-18、PRX Quantum 2, 040352 | §II.1.3、式14–18/23–31、§II.3式38–42と回路説明。一般CDF・二段階fit・number→Z再配置・基底間融合。 |

arXivの版付き一次資料を取得した。P1/P2/P4はHTMLの数式alttext、P3は44頁PDF本文も照合した。P3の版付きHTMLは404だったが、同v2 PDFは取得できた。P3 PDF内部の日付はJuly 13, 2026で、投稿履歴のv2改訂日とは区別する。上記関連箇所の確認であり、全付録・全引用文献・出版版との差分の精査やDF全分野の網羅レビューではない。以下で「同じ式を確認できない」とする場合も、調査範囲内の記述である。

## 2. 現在の研究Cを比較可能な形で定義する

### 2.1 変数・Hamiltonian・spin/frame規約

空間軌道p,q,r,s、frame t=0,…,7、各frameの軌道i,jを使う。研究Cは一般signed実対称Jと実直交Uを用いる通常**number CDF**であり、JにPSD・rank-one制約を課さない。ここでCDF8はouter frame数8を指す。Uの列が回転後軌道を定義し、実直交codecの列符号gaugeはdensity projectorを保持する。[R6]・[R7]・[S1]・[S2]

$$
g^\theta_{pqrs}=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj},\qquad
f=\tfrac12\lVert g-g^\theta\rVert_F^2,
$$

$$
\kappa_{pq}=h_{pq}-\tfrac12\sum_r g_{prrq},\qquad
D_t=\tfrac12\sum_{ij\sigma\tau}J^t_{ij}\widetilde n^t_{i\sigma}\widetilde n^t_{j\tau}.
$$

κは**元g**からの補正を保持し、新gθで置換しない。Nuclear scalarは一回計上する。Spinはinterleaved `a=2i+σ`、frameは元順0…7。Headはκ blockとframe0/1、tailはframe2…7という固定prefix `L_D=2` で、weight再順位付けをしない。固定splitでもhead frameのJ/Uは最適化変数である。Tail identityは正確な決定論側相対位相へ抽出する。J変更でidentity値は変わるため、固定なのは処理規約であって抽出値ではない。[R7]・[S2]・[S3]・[S4]

Spin-orbital number行列とPauli写像は

$$
K_{ab}=J_{a//2,b//2}/2,\quad
c_{Z_a}=-\tfrac12\sum_bK_{ab},\quad
c_{Z_aZ_b}=K_{ab}/2\ (a<b),\quad
c_I=(\sum_{ab}K_{ab}+\operatorname{tr}K)/4.
$$

**RC-DFの空間係数Zと研究CのJはtensor contraction上でZ=Jに対応する。K=Jのspin展開/2は別の変換である。** Uのspin liftとKのspin展開も別で、Kを `J⊗I/2` と置かない。一般signed Jの線形写像は既存SOURCEとR6の独立general-K照合を再利用した。[R6]・[S1]・[S2]

### 2.2 目的関数と独立物理評価

Tailの非identity係数を `c=T_tail J` と表す。異なるframe基底間の演算子を無条件に相殺せず、現在のatom辞書の係数massとして

$$
\lambda_R=\sum_{t\ge2,a\ne I}|c_{ta}|,\qquad
M_\epsilon=\sum_{t\ge2,a\ne I}\sqrt{c_{ta}^2+\epsilon^2},\qquad
F_{\rm PR}=f/f_*+\gamma(M_\epsilon/M_*)^2
$$

を用いる。今回のγ=.1、ε=1e-8 Ha、元scale `f*=2.1839632053132894e-9 Ha²`、`M*=2.2093719370722074 Ha` を維持する。Smooth目的には零係数slotも入り、exact sampling用λとは異なる。R7に既存の `0≤Mε−λ_R≤216ε` の構造的評価があるが、本書は実分子でその差を再評価していない。

一般対称J一frameのexact massは、既存R6の導出で

$$
\lambda_t=\sum_i\left|\sum_jJ^t_{ij}\right|
+\tfrac14\sum_i|J^t_{ii}|+\sum_{i<j}|J^t_{ij}|.
$$

**行和の符号相殺**を含むため、単純なentrywise J normと同じ定義ではない。目的のpenaltyは固定JでUに直接依存しない。Uはtensor表現、J再最適化、非可換性、PF誤差、Gaussian価格を通じて重要になる。[R6]・[R7]・[S1]

物理sector誤差は目的値とは別に、元physical Hamiltonianに対して
`sector operator norm + engineering guard ≤ CA/8 = .00019875 Ha` で判定する。これはhard constraintを内包するoptimizerでも、全探索経路の予算保持保証でもない。今回の保存受理点は独立判定を通過した。Tensor loss低下だけからDF予算PASS、厳密DF enclosureや化学精度達成を推定しない。[R1]・[R7]

### 2.3 J/Uの最適化という言葉の範囲

初期Stage Aは同じqualified x0・共通scaleからγ0/.1をffsimの共通joint L-BFGS-Bで各一回fitした。11/2統合実験は同じF_PRの下、各新Uでcold JからNewton-CG＋refinementを行い、実接空間のU方向、recentered chart、有限step、独立終端・物理判定を用いた。[R1]・[R5]・[S5]

したがって「J/U同時最適化」は両変数を調整する意味では使えるが、「同時一次収束」「新しい一般的なjoint solverの成立」を意味しない。既存CDF/RC-DFにも係数inner solveと軌道outer updateがあり、この二段階構造自体は差分候補にしない。[R6]・[P1]・[P4]

## 3. 先行研究との比較

### 3.1 研究ごとの比較表

| 研究・版・出典 | DF表現 | 最適化変数 | 目的関数 | 対象アルゴリズム | 資源評価 | 数理保証の実際の範囲 | 研究Cとの共通部分 | 条件に基づく差分 | 未確定事項 |
|---|---|---|---|---|---|---|---|---|---|
| RC-DF [P1] v3 | 一般実対称Z、直交U。XDFのrank-one制約を解除 | Z/U、外からleaf数・正則化重みを設定 | 式8のtensor loss＋式17のentrywise weighted L2。本文にweighted L1二乗も記載 | NISQ測定、FT qubitization等 | 測定variance/shot、block encoding別λ、FT資源 | 表現・目的・微分、条件付き最適化式。全分子の性能、大域解、PR-CAは保証しない | Z=J、実軌道・係数の最適化、精度対normの交換 | Cは写像後の固定tail smooth mass二乗。全frame uniform L2とは異なる | Tail/写像/normの寄与とsolver・強度の公平性、資源上の追加利益 |
| SCDF [P2] v1 | 式26の一般V＋shiftを経て、式27はV=W Wᵀ。shift後は通常rank≤2 | W/U/αᵗ、rank政策・閾値 | Tensor loss＋ρΣ abs(W_kW_l−αᵗ)。非二乗L1 | 主にqubitization QPE | λ、Toffoli、logical qubits | 既知対称性固有sectorのshift関係・分解式。資源改善は実験対象内 | 正則化とDFを結合。一般V前段もCとの比較対象 | Cは一般signed J、元κ固定、shiftなし、PR-tail、native RZ会計 | Shift補償とtarget、表現制約・solver差を含む公平比較 |
| Partial randomization/RTE [P3] v2 | PauliおよびDF、回転基底ごとのinvolution | split・時間刻み・RTE/位相推定設定、軌道基底・symmetry shift | λ低減・誤差とsampling込み費用。式E21/E22 | 単ancilla位相推定＋部分ランダムPF/RTE | 減衰、sample数、Toffoli・2qubit gate、depth交換 | Theorem V.1は仮定したpartition別PF誤差定数等の下の複雑度。数値PF推定はheuristicを含む | PR/RTE、DF接続、λ/基底最適化、samplingと価格の会計は既知 | Cは一般Jの固定tail目的を実装・物理判定し、候補固有有限RTEとcomplete P0価格へ接続 | 同じ目的は確認範囲で未確認。既存のweight集中提案との差分・重要性は未認定 |
| Compressed DF/QFD [P4] v1 | 一般対称Z、直交U、number／Z再配置 | Z/U、圧縮rank | Tensor二乗誤差、固定Uで係数解・outer軌道更新 | QFDのtime propagation | 回路深さ、Givens・Z/ZZ、測定 | 分解・微分・固定Uの線形問題。圧縮やPFで一般的な精度／費用最適性は保証しない | CDF・J/U更新、controlled回路、基底間融合は既知 | Cは固定PR-tail penaltyとRPE/条件付き資源の候補評価 | 既知の基底融合やCDF実装を方法的新規性と数えない |
| 研究C [R1]・[R2]・[R6]・[R7] | 一般signed J・実直交U、number CDF8、shiftなし、元順LD2 | 全J/U。rank・split・元κを固定 | f/f*＋.1(Mε/M*)²、sector誤差を独立判定 | Partial S2・有限K0 RTE/RPE、全frame S2対照 | exact λ、PF/RTE、A、shots、W、P0 kernel＋shellの6指標 | 保存H4の数値pilot・反実lane。固定別回路のvalidated保証は別scope | 左4行の既知要素を再利用 | Tail写像目的と、その条件付き数値結果・資源上の限界を具体化 | 同時一次収束、外部移送、最適既存法対照、母平均・全law・厳密CA |

### 3.2 正則化の同値性・非同値性

R6の比較と人工検証を再利用し、以下の**数学的分類**を保持する。

| 比較対象 | 現F_PRとの関係・適用条件 |
|---|---|
| RC-DF式17 `Σρ_tij J_tij²` | 固定entry重みのL2では一般に再現不可。行和、mass全体の二乗の交差項、tail mask、smoothingを区別する。 |
| Entrywise weighted L1 `Σw·abs(J)` | 非二乗であり、ε=0でもmass二乗とは次数が異なる。 |
| RC-DF本文のweighted L1二乗 `(Σw·abs(J))²` | **部分的に同値。** ε=0、各row内に符号相殺がない領域では、tailのdiag重み5/4・非対角独立成分重み3、head0で同じmass二乗になる。Ordered添字のoff重みは3/2。 |
| 任意signed J・現在のε>0 | 上の固定entry重みへの帰着は一般に成立しない。R6のsigned family／smooth rayの既存反例を参照。特別な制約下の一致まで否定しない。 |
| 写像TJ上のsmooth L1二乗を許す一般化正則化 | F_PRをその枠組みで書ける。広いnorm正則化という概念から独立した新原理と断定する根拠にはならない。 |
| SCDF式26/27 | `Σ abs(V−α)`／`Σ abs(WWᵀ−α)` はshift付き非二乗L1。Cのshiftなし・一般J・tail smooth mass二乗とは別。 |

RC-DFの本文は `(Σρ|Z|^p)^(2/p)`、p=1,2を記載し、主結果はuniform L2に集中する。一方、Appendix A式18は `Σρ|Z|^p` と書く。L1の二乗有無を同一視せず、比較名は式を明示する。[P1]・[R6] `COMPARABLE_DISTINCT_OBJECTIVE` はこの技術分類であり、新規性の認定ではない。特にεによる微小な非同次性だけを科学的貢献の中心にしない。

SCDFを単に「一般Jを扱わない研究」と要約するのも不正確である。式26には一般V＋shiftの前段があり、論文が採用する式27のrank-one構造と分ける。Signedなentryを持つrank-one行列と、固有値符号も含めて一般の対称Jを許すことは同じではない。Shiftは既知対称性sectorでのenergy・controlled相対位相の補償を要し、全Fock空間のtargetをそのまま保存する操作ではない。現Cのκ/scalar規約へ無断で追加できない。[P2]・[R6]

### 3.3 λの定義とPR論文との直接の重複

| λ・量 | 定義される対象 | 本書で同一視しないもの |
|---|---|---|
| Cのexact λ_R | Number fragmentの固定tailを現在のZ/ZZ atom辞書へ写した係数絶対和、identity除外 | 全Hamiltonianのnorm、smooth Mε、演算子normそのもの |
| RC-DF/SCDFのλ | 指定block encodingの全Hamiltonianの係数和。One-bodyへの再配置や分解構造も関係 | 現Cのtail λ_Rや同じnative kernel価格 |
| 測定向け量 | 状態依存varianceと測定・shot配分 | RTE normalizationやRPEの十分shots |
| RTE normalization Bとattenuation A | 指定LCU/law・時間・microstep・cutoffに依存 | 実信号の絶対値や厳密CA認定 |

P1の式6やP4の式40では、number積から生じる線形Z項をone-bodyへ再配置する。Cは現在のnumber fragment／tailの辞書とsplitを固定するため、同じ二体係数Jから始めても、再配置後の全Hamiltonian λをCのtail λへ直接転記できない。[R6]・[R7]

P3 §VII.A/C.3は、軌道基底の最適化とsymmetry shiftによるλ低減を既に行う。§VII.BはDFのouter/inner truncationをPRへ接続し、基底ごとに回転したPauli involutionも扱う。さらにweightを少数項へ集中させる軌道最適化を将来方向として述べる。したがって「PR向けに表現を変える」「DF tailをランダム化する」だけのclaimは重複が強い。[P3]

一方、確認した関連節では、**一般signed Jの固定prefix tailについて、現在のnumber由来Z/ZZ smooth mass二乗をtensor lossへ結合し、元κを保持してsector gateとcomplete P0会計まで比較した同一手順**は確認できなかった。これは限定検索上の差分候補であり、先行研究全体に存在しないという証明ではない。[R6]・[R7]

P3 Appendix A.2式A20–A29はpaired Taylor LCU、normalization、sample負担を扱い、有限次数への切詰めにも言及する。研究Cの固定cutoffと候補固有誤差評価はその具体化であり、「finite RTEを初めて導入した」とはしない。P3式E21/E22もsample数と決定論／ランダム平均gate価格を結合するため、単なるshots×価格の接続自体を新理論とは主張しない。[P3]

研究Cの既存K0式は、h/r/mを固定して

$$
\tau=\lambda_R h/r,\quad B_0(\tau)=\sqrt{1+\tau^2},\quad
A_m=B_0(\tau)^{-r2^m}=(1+\tau^2)^{-r2^m/2}.
$$

したがってλ低下はこの条件下のA改善へ結び付くが、PF誤差やcompiled価格の単調改善は含意しない。有限cutoffの補正平均operatorと、normalizationで減衰した観測平均を分け、切詰め誤差は別に評価する。無限Taylor和の理想的な不偏性を有限K0へそのまま主張しない。[R3]・[R7]

### 3.4 差分がどの水準にあるか

| 水準 | 確認できる関係 |
|---|---|
| 数学的目的 | 全frame entrywise L2とは差がある。Weighted L1二乗とは条件付き重複があり、一般化norm正則化という枠組みは共通。 |
| 変数・制約・表現 | RC-DFとは一般J/Uの表現が共通。SCDF式27とはrank-one/shift条件が異なり、式26の一般V前段との共通性も残る。 |
| 量子アルゴリズム | RC-DF/SCDFの主要な測定・qubitization用途からPR/RTEへのtask特化はあるが、PRへのDF・基底最適化の導入はP3に既存。 |
| 資源指標 | Cのcomplete P0 RZ/CX等とshell込みserial会計は、block encodingのλ・Toffoliや測定varianceと別。P3もsample数と平均gate価格を結合している。 |
| 実験的効果 | RC-DFは対象分子で測定variance/shotとλ関連資源の改善、SCDFは対象分子・水素鎖でλ/Toffoli低減、P3は自身のbenchmarkと仮定でPR資源改善を報告する。Cは§4の単一H4でtail低減・条件付きPR内部改善と対Dの主RZ不利を確認した。各論文の条件・費用単位が異なるため、これらを数値的な勝敗表にはしない。[P1]・[P2]・[P3] |

## 4. 現在の数値証拠を条件付きで固定する

### 4.1 数値表に共通する条件

以下の主比較はすべて、**保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、一般signed J/実直交U、Nα=Nβ=2・sector36、元g補正κ・核scalar一回・元frame順**に限定する。PRはLD2・partial ordered S2、δ=h=.1、s1/r128/K0、m=0…12、t_m=.1×2^m。δ=.1一条件であり、別delta窓を走査した結果ではない。Dは同じ時間設定の全9因子ordered S2/s1で、PR split・r/Kは非適用。[R1]・[R2]・[R3]

共通の目標政策はCA=.00159 Ha、DF reserve1/8、βtotal=π/6、βstat=π/12、failure=.01・axis=.01/26。各候補自身のDF ground-state oracleを仮定し、状態準備費用を除外する。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、MPFR160 declared phase、位相fitなし。Native費用はfenced serial kernel反復とinterrogationごとのancilla head/readout shell。全長回路の追加最適化やhardware runtimeではない。

旧PRはStage A γ=.1最終J/U（SHA256 `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`）、新PR/Dはupdate005（`938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`）。**60.60%の分母は旧PRであり、通常CDF γ0ではない。** 新DFのsector norm5.23337351802e-9 Ha＋guard3.22739270291e-9 Haは保存予算PASS。厳密enclosureではない。[R1]・[R2]

### 4.2 旧新PRと同一新J/UのD

全値はR1–R3の既報値を転記し、表示を丸めた。PF/RTE誤差は無次元operator2の数値評価である。

| 指標 | 旧PR | 新PR | 同一新J/UのD |
|---|---:|---:|---:|
| Exact tail λ_R [Ha] | 1.4424298855756243 | .5683498546862645 | PR tail非適用 |
| m12 PF operator誤差 | .071614997914 | .094546804594 | .094327074049 |
| m12補正RTE operator誤差 | .060945308518 | .012063601599 | RTE非適用 |
| m12 attenuation A | .716844437765 | .949629430088 | 1、定義 |
| 条件付き十分shots合計 | 16056 | 14048 | 13728 |
| PR kernel重みW | 16631888 | 10487392 | 本表では未引用 |
| Kernel RZ価格 | 14076.4375、32標本平均 | 19825.625、32標本平均 | 11026、固定kernel |
| Shell込み条件付き総RZ | 234117788135 | 207919150188 | 105532782860 |
| 総CX depth、scheduled serial sum | 本表では未引用 | 43549878473 | 60069240312 |

旧新PRはideal_exact_native lane、Dはnumerical lane。R2の**両方idealへそろえた反実laneでも同じ整数費用**で、PR/D総RZ比1.97018542062、総CX depth比.724994660275。約2倍の点差を合成誤差laneの違いだけで説明はできない。同じJ/Uをそろえる比較は既に実施済みで、未実施課題として再提案しない。[R2]

R1の保存比ではλが約60.60%低下し、PR内部総RZが約11.19%低下した。一方kernel RZ標本平均は約40.84%上昇した。R3の会計説明はW比.630559320746と価格比1.40842631525を区別する。総shots比をW比へ置換しない。Raw signal errorもm12では増え、すべての誤差が改善した結果ではない。[R1]・[R3]

### 4.3 最適化・価格・誤差の資格

| 証拠 | 支持される範囲 | 残る制約 |
|---|---|---|
| 5回のU更新 | 各受理点の物理予算PASS、F・exact λ低下。B_initial λ=.725819387316からupdate005へ更新 | 最終J∞9.5932671593e-8はMET、U₂=.658935814277はNOT_MET。`FINITE_STEP_GRID_NO_ACCEPT` は登録3stepの不成立で、局所最適・改善不能ではない。 |
| Native32/cohort | 各候補のfinite-bit/PCG64、独立非paired32 trajectoryのcomplete kernel価格・数値correctness | 32独立DF／量子shotsではない。母平均CI、ideal-law価格移送、全law supportは未資格。 |
| 校正不確かさ | 新RZ mean SE214.461468031、総RZ descriptive SE2249141484.14という保存値 | 同じ平均を13roundで共有し、SEはW倍。√13で縮小しない。新旧差の検定・有意性は本書で計算しない。 |
| Evidence lane | 実装law／全law合成の不足をnullで保持 | 保証shots・保証総費用・厳密CAは未成立。 |
| Ideal lane | 理想合成・ideal finite-RTE IIDを宣言した条件付き点推定 | 未知実装誤差の実測0ではない。有限bit価格proxyも別仮定。 |
| Assumed lane | 合成・law誤差を各.001Aと仮定する感度評価 | 仮定した値の認証ではない。 |
| 旧F2-K1/F3-K1 | 別設定s4/r16/K0の固定各1trajectory、指定targetに対するvalidated operator保証 | 現update005/r128・全law・全round・元physical Hamiltonian・CAへ移送しない。[R5] E23。 |

## 5. 科学的貢献候補と証拠の強さ

### 5.1 候補A：固定PR-tailを意識したDF目的と数値的低減

**主張候補文：**「一般signed number CDFの固定prefix tailについて、実際の非identity atom係数に基づくsmooth mass penaltyを用い、単一H4条件で物理DF誤差予算を保持したtail低減を観測した。」[R1]・[R5]・[R6]・[R7]

| 観点 | 評価 |
|---|---|
| 1. 直接支持 | 初期同x0 γ0/.1比較はexact λ34.7246%減。追加J/U更新を経た新点は旧PR比60.60%減、受理点のDF予算PASS。初期fitには時間発展δは非適用、下流条件は§4。 |
| 2. 既知の部分 | CDF変数、J/U最適化、norm penalty、重み調整、inner/outer手順、λ低減の動機。既存要素を組み合わせた実装であることは明示する。 |
| 3. Cでの条件付き知見 | Number由来の行和を含む写像、固定head/tail mask、元κと独立sector gateをそろえた具体的結果。Jを解き直しながらUも動かせたが、Uだけの効果は未分離。 |
| 4. 未検証 | 最適γ、最良λ、初期値非依存、独立分子／幾何の再現性、収束済みRC-DFに対する優位。 |
| 5. 新規性判断に不足 | Weighted L1二乗／写像後norm／tail限定という各差分の実質的効果。既知枠組みのtask適用を超える機構・利点の証拠。 |
| 6. 中心claimのリスク | 「新DF表現」「新一般solver」「RC-DFより優れる」と書くと既知性・未収束・比較不公平を招く。60.60%をγ0比とする誤読にも注意。 |
| 7. 判断を変える確認候補 | 共通の物理誤差水準・停止資格と強度政策での近接対照、必要に応じtail mask／norm／写像の対照。方法の一般性を中心に選ぶ場合だけ独立分子・初期値を確認する。実施は未採択。 |

固定tailの意味は、PRでrandomizedに残す量を直接抑えつつ、headへの負担移動とU/PF/nativeの影響を同じtaskで測れることにある。全Hamiltonianの係数normとは研究上の問いが異なる。ただし固定frameラベルやmaskは設計条件であり、それだけでは独立した科学的原理ではない。

**既存RC2比較の到達点。** R8では同x0・初期penalty寄与.1一致の全frame uniform L2がλ=2.209761871954849 Ha、旧PR-tailが1.4424298855756243 Haで、両DF予算PASSだった。R9では両保存終点が工程的一次基準を未達で、RC2にtensor-nullなL2下降成分が残る。後続R10には別の共通有限予算・未収束RC2/PR終点のnative価格比較もある。これらは有用な限定比較だが、原論文のtwo-step solver・最適ρ・誤差Pareto・収束をそろえた既存法比較ではない。R10のPR exportはupdate005とも異なり、その価格を新点へ移送しない。[R8]・[R9]・[R10]

### 5.2 候補B：DF最適化からPR量子資源への候補固有評価

**主張候補文：**「DF候補ごとにPF/RTE誤差・attenuation・十分shots・反復重みW・complete native kernel価格・shellを接続し、λやtensor誤差だけでは見えない条件付き資源変化を比較した。」[R1]・[R2]・[R3]

| 観点 | 評価 |
|---|---|
| 1. 直接支持 | 新J/Uから固有Hamiltonian・ground/alias・law・13round誤差と新native32を取得した既存実験があり、旧価格・shotsの単純移送ではない。 |
| 2. 既知の部分 | RTE normalization、attenuationによるsample増、位相推定の統計、平均gate価格とsample数の費用結合はP3にある。Hoeffding十分式やserial会計も既存手法。 |
| 3. Cでの条件付き知見 | 候補ごとに変わる誤差・非paired finite-cohort価格と、scalar・actual adjoint・X/Y・head/tail fusion・shellを同一範囲に戻した具体的診断。未資格量も資源と同時に表示。 |
| 4. 未検証 | 全law implemented error、母平均価格、厳密CA、状態準備込み・hardware runtime・別compilerの費用。 |
| 5. 新規性判断に不足 | 既知式を接続した評価実装を超え、意思決定を改善する再利用可能な方法・診断として何を示せるか。既存PRの費用モデルとの差が順位や設計判断へ及ぼす独立証拠。 |
| 6. 中心claimのリスク | 「初のend-to-end PR評価」「実装CA保証」「最終総資源評価済み」は過大。反実laneと有限価格proxyを外すと支持不能。 |
| 7. 判断を変える確認候補 | 主張を母平均順位へ強める場合はnative価格の不確かさ、保証へ強める場合はlaw／合成／bridge／CA。限定会計の記述だけなら追加計算は不要。 |

既存SOURCEの会計は

$$
C_{\rm total}=W\bar C_{\rm kernel}+S_{\rm shell},\qquad
W=\sum_{m,b}N_{m,b}2^m\quad(s=1).
$$

R3が引用する新旧Wと平均価格、shell込み総費用をそのまま使用する。R3で新shell=49168は保存総費用とWμの**既存の残差確認値**であり、独立保存fieldではない。r128はkernel価格に内包し、Wにさらに128を掛けない。[R3]・[S4]

ここでの追加情報は、既知の「減衰にshotsを払う」という原理を新発見することではなく、**特定DF更新で、どの誤差が改善／悪化し、価格増が重み減をどれだけ相殺したかを検査可能にした実例**にある。その実例が単独の論文中心となる重要性は後続判断に残す。

### 5.3 候補C：PR内部改善と強決定論PFとのcompiled-resource tradeoff

**主張候補文：**「固定H4・同一DF候補・共通精度政策とserial nativeモデルでは、PR-tail低減によるPR内部改善が強決定論S2への主RZ優位を伴わず、CX depthとのtradeoffが残った。」[R1]・[R2]・[R3]

| 観点 | 評価 |
|---|---|
| 1. 直接支持 | 同一update005でD総RZ105532782860、PR ideal207919150188、PR/D≈1.970。CX depth比≈.725。旧新PRではλ60.60%減に対し総RZ約11.19%減、単価上昇を観測。 |
| 2. 既知の部分 | Sampling overheadとdepthの交換、metricによる選択変化、Gaussian基底変換の費用はP3/P4で既知。λだけで価格を決められないこと自体を新定理とはしない。 |
| 3. Cでの条件付き知見 | 同じ新J/Uの強baselineでcandidate不一致を除き、主RZ不利と副指標の利点を同時に示す具体的観測。両ideal laneでも主RZ点差が残る。 |
| 4. 未検証 | 全PR・全PF・全δ/s/r/Kの最適性、別分子やhardwareでも同じ勝敗、母平均の有意差、価格悪化の因果機構。 |
| 5. 新規性判断に不足 | 既存の一般的tradeoffを超えた、再現する失敗条件・機構・設計則。単一小系の観測から一般則へ強める独立証拠。 |
| 6. 中心claimのリスク | 「PRは決定論に勝てない」「全資源で支配」「λ目的は無効」「PR論文を反証」は支持されない。公平な強Dでも特定実装政策の対照であり最適D一般ではない。 |
| 7. 判断を変える確認候補 | 他条件でcrossover／持続を調べる場合は未使用分子・幾何・政策、機構claimなら物理精度を保つ介入、hardware claimなら事前の価格・depth定義。現在の限定負結果の記述には不要。 |

負結果の情報価値は、(i) proxy成功とtask成功を切り分けたこと、(ii)弱いbaselineでは隠れる費用差を示したこと、(iii)新fitを続ける前にkernel価格との接続を問う根拠を与えることにある。λの低下は固定K0でのA改善という有効な情報を持ち、その成果は主RZ不利によって消えない。逆にPR内部改善からD優位は導けない。

R4の `TECHNICAL_NO_GO_CURRENT_EVIDENCE` は、新しい価格改善機構を特定できない現時点の技術準備判断である。既存surrogateのcohort内相関・第三DF移送判定不成立・native人工反例を、新しい有望機構の証拠へ読み替えない。これは研究C全体の終了や、未実装head/tail境界融合等の不可能性の証明でもない。[R4]・[R5]

## 6. 論文構成候補：いずれも未採択

### 案A：PR-tailを意識したCDF最適化の限定的実証

| 項目 | 構成候補 |
|---|---|
| 中心RQ | 物理DF誤差予算を満たす一般CDFで、PRに残るtail massをどこまで減らせるか。 |
| 想定claim | 固定tailの写像後penaltyを実装し、H4でtail低減と候補固有資源変化を観測した。 |
| 使用する既存証拠 | R5の同x0 Stage A、R6の数理比較、R8/R9の限定RC2対照、R1の追加更新・物理予算、R2の強D制約。 |
| 先行研究との対応 | P4→P1の一般CDF/正則化を土台に、P3のPR表現改善という問いを固定tailに具体化。 |
| 予想される弱点 | 数理的には既知正則化の自然な拡張との評価を受け得る。単一H4・一次未収束・目的形状とmaskの効果未分離。 |
| 必要な追加証拠 | 独立方法論・一般性を中心に置くなら、収束と強度政策を含む近接対照、未使用系・初期値。追加項目は中心claimを選んでから限定する。 |
| 現証拠だけで書ける範囲 | 方法の定義、既知法との包含／非同値性、固定有限endpointの低減、条件付き資源と失敗範囲。一般性能保証や原論文RC-DFへの勝利は書けない。 |

**章立て案：** 問いと既知正則化 → number-CDFとtail写像 → 独立物理評価 → 同x0比較と追加更新 → PF/RTE/native接続 → 強Dと未収束・汎化限界。図表候補は目的比較、候補履歴、λ/物理誤差/価格の併列表。現資料の引用で構成し、新しい図用再集計は本書では行わない。

### 案B：同一DF上のPR/PFのcompiled-resource比較

| 項目 | 構成候補 |
|---|---|
| 中心RQ | 同じ近似Hamiltonianと精度政策の下で、PRの減衰・反復・compiled kernelは強決定論PFとどの資源交換を生むか。 |
| 想定claim | 固定H4の条件付き主RZではDが低費用、PRはCX depthが低い。表現更新の利益はアルゴリズム／metricによって異なる。 |
| 使用する既存証拠 | R1/R2の同一update005、旧同J/U対照を整理したR5、R3のW/価格/shell、P0とcorrectness記録。 |
| 先行研究との対応 | P3のsample・gate価格モデルを実際のcomplete native kernelで具体化。P4の基底融合を強Dにも適用。 |
| 予想される弱点 | δ/s/r/K・compiler・serial政策が限定的。P3の大系・STO-6G/1.4 BohrやToffoli見積りと直接勝敗比較できない。全law/実CA未資格。 |
| 必要な追加証拠 | 一般tradeoffやhardware上の利益を中心に選ぶ場合のみ、他条件での比較と価格・depthの対応。母平均claimには価格の不確かさ。 |
| 現証拠だけで書ける範囲 | 単一taskの複数metric比較、同J/Uで残る点差、反実laneの違いと限界。PR一般や最適化済み全手法の順位は書けない。 |

**章立て案：** 共通taskと比較範囲 → 強native D/PR builderと合法融合 → 誤差・sampling・serial会計 → 同一新J/Uの6metric → 旧新表現の効果 → モデルとhardwareの適用限界。

### 案C：係数mass低減と総資源削減が一致しない事例

| 項目 | 構成候補 |
|---|---|
| 中心RQ | PR-tail proxyの大幅改善が、なぜ同じ比率の総費用低減やDへの優位を伴わないか。 |
| 想定claim | 固定条件ではA/RTEとWが改善する一方、PF誤差・kernel単価が悪化し、総RZ改善が限定される。記述的会計でその不一致を示す。 |
| 使用する既存証拠 | R1/R3の全13round比較と保存W/価格、R2の強D、R4/R5のsupport/rank/surrogate/native負結果。 |
| 先行研究との対応 | P1のnorm正則化、P3のsampling/gate価格、P4の基底変換費用を結び、proxy評価の実例として位置づける。 |
| 予想される弱点 | 既知の一般的注意を単一H4で例示しただけと評価され得る。価格悪化をUやswitchへ因果帰属できない。複数負結果も独立分子の数ではない。 |
| 必要な追加証拠 | 新しい機構・設計則を中心に選ぶなら、物理精度を保った操作とcomplete価格の関係、または未使用DFへの予測。R4で未資格のsurrogateを再採用しない。 |
| 現証拠だけで書ける範囲 | 既知式と観測が整合する限定事例、proxyと総費用の違い、失敗した仮説の射程。不可能性定理や価格の因果機構は書けない。 |

**章立て案：** Proxyとtaskの問い → 既知のλ/A関係 → 旧新PRのW×価格＋shell → 同J/U強D → 限定負結果と同値性を区別 → 変わり得る判断と未取得証拠。

案Aは方法差と近接対照、案Bは比較の適用範囲、案Cは機構と一般化が主要な論点になる。三案を一括採用したり、現証拠での投稿可能性を本書で決めたりしない。既知要素の使用は貢献の否定理由そのものではなく、**組合せから何を新しく検証可能にしたか、その重要性を証拠で示す必要がある**という位置づけである。

## 7. 追加確認を、変わる科学的判断に対応させる

下表は不足証拠の候補で、研究計画・実行順序・科学契約の採択ではない。現在の限定結果を正確に説明するために、全項目を完了する必要はない。

| 判断対象 | 現証拠だけで言えること | 強める場合の追加確認 | 結果が変える判断・実施境界 |
|---|---|---|---|
| 目的差の独立した利益 | 一般classで目的が異なり、有限点でtail差がある | 同じ物理誤差水準・初期化・十分な一次資格・強度政策で、uniform L2と近接weighted L1二乗等を比較。Tail mask／写像の対照は問いに必要な場合だけ | 既知正則化のtask適用を超える利益があるか。新fit等は別許可。 |
| 最適化の信頼性 | 5回の物理適格更新、終端U未収束 | 共通stationarity診断と数値精度、有限step停止・inner安定性の切分け | 有限endpoint比較から再現性ある最適化結果へ強められるか。現STOPのretryを自動開始しない。 |
| 初期値への依存 | 一つの起点・限定trajectoryで成功 | 結果前に定めた複数初期値で分布・失敗条件を見る | 方法の頑健性か、特定初期値の局所現象か。新条件の科学実行。 |
| 分子・幾何への移送 | H4/1Å/STO-3Gのみ | 未使用分子／距離、明示したrank政策・LD・誤差予算で再現を確認 | 単一事例から適用域を主張できるか。旧H2/H6通常CDFをPR-awareの移送実験と数えない。 |
| Native母平均の順位 | 各finite32の平均とdescriptive SE | 目的に必要な精度の独立校正・誤差区間・選択効果の扱い | 点順位から母平均に関する主張へ進めるか。既存標本の再解析も科学許可対象。 |
| λ→価格の機構 | W改善と単価悪化の記述的共存 | DF予算を保つ介入、候補固有law/誤差、complete P0価格を組にして確認 | U構造・basis分布等のどれを操作すべきか。新機構を特定してから範囲を選び、旧相関を因果効果にしない。 |
| 厳密CA／全law保証 | 数値pilotと反実lane、旧固定回路の限定validated保証 | 現候補の全law synthesis・implemented-law・target bridge・丸めとenergy/RPE接続 | 保証付き資源claimを支えられるか。探索的記述をまとめるための一律前提にはしない。 |
| Hardware／metric一般性 | RZではD有利、CX depthではPR有利というserialモデル | 事前に対象hardwareと価格・parallelism・routing・noise等の必要条件を定義 | 実用上どのtradeoffが重要か。副metricを結果後に主指標へ変更しない。 |

既存結果だけで可能なのは、§5の限定主張文と§6の範囲付きストーリー候補である。独立新規性の正式判断は、一次文献の重複評価と主張の価値の判断も要し、追加実験の件数だけで自動的に成立しない。重要レビューの許可と科学実行の許可も独立である。

## 8. Codex-A・統合担当への引渡しと保全

### 8.1 AのEvidence Mapと統合する項目

| 担当 | 統合する具体的項目 |
|---|---|
| 統合担当Codex | 本書R1–R7のcommit/branch境界を維持し、R3の絶対固定リンクを残す。R3が起点treeにあると書かない。 |
| 統合担当Codex | Aの最新IDと、本書の候補A/B/C・数値表・資格表を対応づける。本書の候補Bと、過去Evidence MapのClaim B等のラベルを混同しない。 |
| 統合担当Codex | 旧新PR比と同一新J/UのD/PR比を分け、60.60%の分母・δ/s/r/K・候補SHA・P0・oracle・準備除外を添える。 |
| 統合担当Codex | RC2比較は実施済みだが収束・強度・原論文solverを含む公平比較は未完了、と接続する。別RC2/PR終点をupdate005へ置換しない。 |
| 統合担当Codex | 全law/implemented-law・母平均・strict CAのnull、固定回路保証の限定範囲、既存STOPと技術NO-GOの射程を保持する。 |
| 統合担当Codex | A/B両成果がそろった後、必要な共有索引・研究概要等への最小リンク統合を担当する。本branchでは共有ファイルを変更しない。 |
| ユーザー／研究C | 独立新規性・重要性・中心claim・論文化GO/NO-GOと、判断を変える追加確認を選ぶ。本書は選択材料まで。 |

### 8.2 今回の静的確認・再利用

入口PROJECT_MAP、研究概要、関連研究資料、status/manifest、R1–R10と一次文献を参照した。以下の既存SOURCEを**読解のみ**で確認した。科学pipelineは起動していない。

| 既存資産 | 再利用した定義・照合内容 |
|---|---|
| [c_pr_aware_df.py][S1] | coefficients、smooth_mass、make_value_and_grad、components。Exact/smoothとpenaltyのU非依存。 |
| [c1_general_k_adapter.py][S2] | J→K、spin順、number表現、一般signed係数・Uの規約。 |
| [c1_stage1.py][S3] | 元g補正κ、tensor contraction、独立raw CAR因子と物理sector評価の区別。 |
| [c1_stage3_connection.py][S4] | 元順split・identity抽出・tiny非zero保持、finite RTE接続、serial費用とshell。 |
| [c_pr_integrated.py][S5] | Cold-J再solve、U更新・有限step、独立終端／物理gate、事前選定規則。 |
| [c_rcdf_comparator.py][S6] | 全ordered J²のL2、共通ffsim目的、原論文solver再現との区別。 |
| 既存R5/R6の試験・照合記録 | 目的の非同値・条件付き同値、general-K、codec、correctnessの既存証拠。今回再実行したPASS数としては扱わない。 |

新しいコードやテストは不要だった。検証は、公開固定本文／Git blobの一致、引用箇所・値・候補identity、数式・SOURCEの静的対応、Markdownリンク・差分の技術確認に限る。単一H4 NPZ/QPYもdecodeせず、raw再集計・統計再推定・新しい損益分岐計算はしない。

独立worktreeは `/tmp/prt-research-c-contribution-positioning-20261102`。本branchの変更は本Markdown一件だけとし、PROJECT_MAP、VALIDATION_STATUS、docs/README、研究概要、研究ノート、validation manifest、Codex-AのEvidence Map、科学SOURCE/raw/契約/grant/ledger/marker/STOPを編集しない。共通運用指示2026-10-25版の§8/8.1/8.4/10/10.1/10.3は依頼元workspaceの原本で確認し、起点treeにない同文書の公開リンクは捏造しない。承認証明・新grant・管理frameworkは追加しない。

文献取得と独立資料確認は並列化した。重い計算はなく、CPU/RAM/worker/wallへの任意capや共有環境・他jobの変更はない。Commit/push後にremote SHA/tree、公開Markdown、起点からの一件のみの差分を照合する。旧SOURCE/rawが変更されていないことはこのGit差分で確認し、別担当の作業領域まで一律に不変と認定する監査はしない。

**次の担当はユーザー／研究Cと統合担当Codex。** 前者が主張・重要性・追加確認の要否を判断し、後者がA/B完了後の証拠対応と共有入口への統合を行う。本Bの文献・位置づけ資料作成範囲に未完了の科学実行はない。追加fit・再解析・回路計算・詳細科学レビューへ進まず、**mandatory STOP**する。

## 出典

[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[R2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[R3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2639221a59713d431bc7c157854628bbc006a1bd/docs/research/research_c_pr_resource_contribution_analysis_20261102.md
[R4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[R5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_evidence_synthesis_and_decision_readiness_20261031.md
[R6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/150f904f6975868abe09966dda8ab41722e2a7cc/docs/research/research_c_regularization_comparator_design_20261101.md
[R7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47e0d406226be365951bd8666d5acae61106f500/docs/research/research_c_pr_aware_df_optimization_design_20261026.md
[R8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/335a7e04951e009bd6910afaecd1a3107e0d61a4/docs/research/research_c_regularization_comparator_execution_20261101.md
[R9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5fdcd81addf83e893d76c89e64f733433a4e76b3/docs/research/research_c_regularization_first_order_execution_20261101.md
[R10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/611a618674aba7146b0103cc34f156c9bc89bc16/docs/research/research_c_rc2_pr_native_execution_20261101.md
[P1]: https://arxiv.org/html/2212.07957v3
[P2]: https://arxiv.org/html/2403.03502v1
[P3]: https://arxiv.org/pdf/2503.05647v2
[P4]: https://arxiv.org/html/2104.08957v1
[VS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/VALIDATION_STATUS.md
[VM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/validation_manifest.json
[S1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c_pr_aware_df.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_general_k_adapter.py
[S3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_stage1.py
[S4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_stage3_connection.py
[S5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c_pr_integrated.py
[S6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/150f904f6975868abe09966dda8ab41722e2a7cc/src/trottertracks/c_rcdf_comparator.py
