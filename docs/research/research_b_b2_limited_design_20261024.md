# 研究B：B2 Corrected／Processed PFのPR接続 — 限定設計

2026-10-24 JST。区分：既存理論・SOURCEに基づく設計提案。新規実装・科学計算・回路生成・数値再解析は行っていない。

**提案：最初の科学検証はB2-Dとし、published YP8m8をDの半時間発展に用いる案を、同じ配置の既知PF群と比較する。** D内部誤差を減らせるかと、その改善がfull physical signalに残るかを分けて調べる。これは検証順の提案であり、YP8m8の性能勝利、新機構の正式採択、科学実行の承認ではない。B2-Fはfull列の補正位置と平均の意味論を定義できるが、現PR kernelに合うcorrectorの保証・実装accessが不足している。

## 1. 目的、参照資料、固定commit

目的は `H=H_D+H_R` のphysical coherent signalを同じ精度で近似するための資源削減可能性を、B2-DとB2-Fに分けて具体化すること。本書の式は列の配置・評価対象・成立条件を指定するもので、新しい誤差係数・改善率・科学的保証を導出したものではない。

| 資料 | 実際の確認と使用範囲 |
|---|---|
| PR repositoryの起点 | `HIROMU1015/Partially-Randomized-Trotter`、`a40a655b27da6e8528c3e6fba7d6d9158e7d3056`。[直前の限定調査][SURVEY]をGitHubから再取得。`PROJECT_MAP.md`、研究概要、関連SOURCEはこの固定版を参照 |
| PF repository | `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae`、`20d64c2fea1ed2d2188777df4f3d962ee94df486`。§7の係数・列・cost・validator・responseをGitHubから実取得して静的確認 |
| Morales et al. | [2210.15817v3、2025-01-10][MORALES]。§II.1–3、Appendix Bの多項和のS2、processed列の定義を必要範囲で確認。published Table 1/2との係数対応は固定PF SOURCE・responseを使用 |
| Bagherimehrab et al., Corrected Product Formulas | [2409.08265v4、2026-07-28][CPF]。§IIの型、§III.BのEqs.(2), (24)–(26)、§Vの合成、§VIIIの適用限界を確認。Eq.(9)の出版社紙面不整合は直前調査の確認を引き継ぎ、再監査・推測修正なし |
| 共通運用指示 | ローカルの `research_project_instructions_unified_2026-10-23.md` を実参照。既読版と同じSHA-256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` を確認 |
| 研究地図・カード・過去の終了 | [初期調査][INITIAL]と[直前調査 §§7–9][SURVEY]の固定参照を再利用。既済の先行研究監査やB-F／B-Mを再開しない |

独立worktree・branch `codex/research-b-b2-limited-design-20261024` を上記PR起点から作成し、本書だけを追加する。研究Cの**C1（圧縮DFによるHamiltonian表現改善）を研究全体の最優先**とする今回のユーザー指示を維持する。B2の静的設計はC1の完了を待たずに完了させ、C1のSOURCE・計算・検証・branchに介入しない。

## 2. 共通の記号と、現PRの出発点

scalarとtailから抽出したidentityは、既存policyに従う独立した位相として保持する。以下の列式ではその既知scalar位相を省略するが、実装・controlled比較で省略してよいという意味ではない。`H_D=Σ_j D_j` の各 `D_j` はone-body correctionまたはDF fragmentであり、和全体のexact指数は仮定しない。

| 記号 | 意味 |
|---|---|
| `δ`, `q`, `T=qδ` | PRのmacro step幅、反復数、physical時間。Dの内部substep数 `n_D`、tail分割数 `r_RTE` と区別 |
| `U_D(h)`, `U_R(δ)` | 各和全体のexact指数。将来の小規模古典参照には使えても、無料の量子oracleではない |
| `F_D(h)`, `R_D(h)` | 現SOURCEのforward／reverse D sweep。逆順で同じsigned timeを使い、互いのinverseではない |
| `V_{R,ω}(δ)` | 既存finite RTEの一occurrenceに対応する実現unitary。event順序・scalar以外のevent phaseを含む |
| `N_R`, `A_R`, `M_R` | 共通tail設定での正の有限normalization product、normalization-corrected Taylor operator、raw mean。`A_R=N_R M_R` |
| `P`, `K` | 数学的なunitary processorと反復unitary kernel。CPF論文の `log S` をkernelと呼ぶ用法とは異なる |

現PRの一stepのsystem operatorは、行列が右から作用する規約で

`W_{0,ω}(δ)=R_D(δ/2) V_{R,ω}(δ) F_D(δ/2)`

である。[df_partial_s2.py][S2]のappend順はforward、RTE、reverseであり、[有限RTE signal SOURCE][SIGNAL]も `reverse @ tail @ forward` を使う。**前後のsweepはそれぞれ独立した高精度な `U_D(δ/2)` ではない。** 既存partial-S2の名称から、standalone D-S2半stepへ読み替えない。

将来比較する全候補で、Hamiltonian snapshot・DF rank／policy・block順序・`L_D`、identity／threshold、入力状態、`T,δ,q`、tail辞書・`r_RTE,K`、phase規約を固定する。変更する量は各armの仕様として明示する。tailの `lambda_r` ranking proxyと実際のRTE L1を混同しない。[split SOURCE][SPLIT]

## 3. B2-D：D内部PFを処理する限定案

### 3.1 候補列を具体的に固定する

**第一候補は既知YP8m8のDへの接続で、新correctorの発明は行わない。** `H_D` の同じordered block列から、standaloneの対称二次列 `S_{D,2}(τ)` を定める。その実時間primitiveは個別blockのsigned指数である。Moralesの多項和S2定義と、そのS2を用いるkernel／processor列を根拠とする。[Morales §II.1–3・Appendix B][MORALES]

`h=δ/2`, `τ=h/n_D` に対し、候補のD半時間発展を

`G_D(h;n_D)=P_D(τ) K_D(τ)^{n_D} P_D(τ)†`

と定義し、PR側の一stepを

`W_{D,ω}(δ)=G_D(h;n_D) V_{R,ω}(δ) G_D(h;n_D)`

とする。**左右は同じ正方向のD半時間発展であり、`G_D` と `G_D†` の対ではない。** `n_D` はD内部だけの反復で、tailを複製しない。`P_D,K_D` は固定PF SOURCEのpublished YP8m8列を使用する候補とし、係数探索・再最適化をしない。[product_formula.py][PFSOURCE]

SOURCEのlistは回路append順に消費され、行列は最後にappendしたgateが左に来る。将来のadapterでは数学的 `P_D` と、`_get_processor_s2_sequence` のlistが作るoperatorの向きを照合する。**list名だけで `P K P†` の作用順を確定しない。** 既存validatorが使う積規約と一次式に結び付け、inverseは実現列から生成する。[pf_decomposition.py][PFDECOMP]、[PF validator][PFVALIDATOR]

これは現 `F_D/R_D` を同値な回路へ書き換える案ではなく、D/Rをouterの二分割として、D半時間発展を独立近似する**nested構成への変更案**である。現partial-S2も必ず対照に残す。対称D-S2を同じ位置に入れる対照を置くことで、配置変更の効果とYP8m8の効果を分ける。

### 3.2 何の誤差を改善したいか

| 層 | B2-Dで狙うもの／残るもの |
|---|---|
| D単独 | `G_D(h;n_D)` の `U_D(h)` へのoperator近似。published processed familyのfull列による改善を狙う。kernel単独の状態overlapを代用しない |
| D/R分割 | exact `U_D(h) U_R(δ) U_D(h)` と `U(δ)` の差は残る。D内部を高次にしても、このouter二次分割を高次にしたことにはならない |
| Full operator／signal | D近似の改善が、残差との積の中でも有効かを調べる。既存列にあった誤差相殺が変わり、full誤差が増える可能性もある。改善の保証は未成立 |
| 有限RTE | tail設定を固定するので `A_R` 自体のTaylor biasと `N_R` は同じ。新しいD sandwichを含むfull corrected operatorは異なり、そのphysical bias・位相を別に比較する |
| 統計 | 同じtail normalization／測定rangeでも、固定状態signalの半径・位相conditioning・実測varianceは変わり得る。必要shot数が同じ／減るとはまだ言えない |

Dが可換な一blockだけの場合、D内部誤差を減らす機構を識別できない。そのようなcaseはcorrectness対照に使えても、主たる改善検証にはしない。将来のfixtureでは、D内部にもD/R境界にも非可換性があることを結果取得前に確認する。

### 3.3 CPFをD内部へ使う別armの条件

既知CPFの二分割 `H_D=A_D+B_D` を利用する案も残す。ただし `A_D,B_D` の和指数が実装可能か、§Vのcorrector合成に必要なprimitiveを供給できるかを示す必要がある。現在のDF block群を任意に二群へ分けてもexact可解二分割にはならない。これを解消するための新split・oracle・commutator実装は本書では設計しない。

整合的な既知定義として、CPF Eqs.(2), (25)–(26)の通常PF2とsymplectic correctorは参照できる。この限定参照の適用範囲はexact二分割・実時間であり、D全体がgenericな多fragment和の場合へ保証を自動移送しない。主候補を多項和のS2列から作れるYP8m8とする理由の一つである。[CPF §III.B][CPF]

## 4. D/R境界、processor配置、合法な相殺

固定 `P_D=P_D(τ)`, `K_D=K_D(τ)` に対し、一つのD run内では既知processed PFの恒等式に従い、反復kernelの間の `P_D†P_D` を消せる。[Morales Eq.(27)][MORALES]

| 境界 | 許される整理 | 許されない扱い |
|---|---|---|
| 一つのD半時間発展の内部 | `n_D` 個のkernelを一組のPで包む | 各kernelごとに不要なprocessorを重ねる／反対に端点費をゼロにする |
| 隣接macro stepの `G_D(h)G_D(h)` | 同一 `τ,P_D,K_D`、同じ実現inverseなら `P_D K_D^(2n_D) P_D†` とできる | 幅を倍にした別processor `P_D(2τ)` へ置き換える。別の近似列になる |
| 同一step内の `G_D(h) V_R G_D(h)` | それぞれのD runにprocessorを保持する | 間の `P_D† V_R P_D` を無視してD両端だけへ移す。一般には残差を変える操作になる |
| 残差をまたいだ可換性 | 実際の `P_D` と全対象residual operatorの可換性が別途確立した特殊caseのみ検討可能 | `H_D` がdeterministic、tailが小さい、平均が同じという理由だけで消去 |
| 異なる幅・`n_D`・family・DF表現 | 各境界のprocessorを個別に照合する | 全てのPを同じ記号で表し自動相殺 |
| Shot／X・Y測定／RPE round | 各executionの列として費用を払う | 測定・reset・状態再準備を越えて端点を一組に償却 |

一般の非可換tailを挟む固定幅 `q` stepでは、最初と最後のD run、および隣接stepで融合したD runが残る。理想的に上記の隣接相殺ができても、**D processorは全時間発展を通じて一組だけにはならない**。この配置では `q+1` 個のD runに各一組が残る。これは列配置上の個数であり、新しいgate countや費用予測値ではない。tailがない等の特殊caseでは別に再確認する。

正負時間についても、対称なprimitive S2の負時間列、非対称processorのinverse、freshな負時間RTE sampleは異なる。`P_D†` は実際に採用したunitary列の逆順・各gateのadjointで作る。近似processor `Q_D≈P_D` を用いる場合でも、その **actual adjoint `Q_D†`** なら隣接inverse対は相殺できるが、Pへの合成biasが消えるわけではない。

## 5. B2-F：full physical evolutionを処理する限定案

### 5.1 固定deterministic processorをfull列の両端へ置く

第一の設計対象は**full operatorと固定physical入力状態のcoherent signal**である。kernelの固有位相だけを改善対象とはしない。現在のPR一stepを `W_ω`、一executionの実現列を `V_Ω=W_{ω_(q-1)}…W_{ω_0}` として、固定unitary `P_F(δ)` による候補を

`V_{F,Ω}=P_F V_Ω P_F†`

と定義する。同じPを各stepに置く場合も、隣接inverseが合法に消えるならこの端点配置になる。Pはsnapshot・split・step幅に依存してよいが、**そのexecutionのrandom realization・測定結果に依存しない固定列**とする。この条件下では

`E[V_{F,Ω}]=P_F E[V_Ω] P_F†`

であり、共通の正normalization `N_tot` があるなら、corrected mean `A_0=N_tot E[V_Ω]` に対し `A_F=P_F A_0 P_F†` となる。これは固定線形操作と平均の意味論であり、**`A_F` が元physical `U(T)` に近くなるという保証ではない**。

外側stepが独立で同じlawのときはそのordered meanを使える。一方、一つのsampleを全stepへ反復する場合や、processorをsampleごとに変える場合は別ensembleとなる。`E[W_ω^q]`、`(E[W_ω])^q`、`E[V_Ω]` を条件なしに同一視しない。[現repeated契約][REPEAT]、[finite RTE mean][RTE]

### 5.2 既知correctorを参照できる具体的範囲

Eq.(9)を用いない参照構成として、CPFの **Eq.(2)の通常PF2、Eq.(25)の `C=-λ²[A,B]/24`、Eq.(26)のsymplectic処理**を用いる。`A=H_D, B=H_R, λ=-iδ` とすれば、これはexact二分割の参照列

`J(δ)=U_D(δ/2) U_R(δ) U_D(δ/2)`

を `P_F=exp(C)` で処理し、一部のleading commutator誤差を除く既知機構の候補となる。Eq.(26)には残る二重Bの誤差項があり、単なるconjugationで一般の二次誤差が全て消えるとはしない。[CPF §III.B Eqs.(24)–(26)][CPF]

**現PRへの保証は未成立。** 現在の `R_D V_R F_D` はこのexact `J` ではなく、D内部PFと有限RTEの差がある。CPF §Vのcorrector合成もA/Bの指数から行うため、`exp(C)` を無料のcommutator oracleとして追加できない。D／R和指数をさらにPF・RTEで置換した際の保証と費用は未確定である。上記Cは既知式の候補参照であり、現PRへの正式採択・完成した回路recipeではない。

P_FにD-onlyのprocessorを使えば、そのままfull kernelに適したprocessorになるわけでもない。逆にfull列のS2をbaseにpublished YP8m8を構成する方向は、**理想exact指数の対称S2**なら既知processed構成である。しかしbase内へ有限RTEを入れると、meanはunitaryでなく、単純な負時間指定はinverseを与えず、同じorder条件をそのまま援用できない。負stageごとのRTE normalization・biasも追加されるため、今回の最初の候補にはしない。

### 5.3 Symmetric／compositeとrandom processorを分ける

| 構成 | 配置と未解決点 |
|---|---|
| Symplectic／固定processed型 | `P_F W_ω P_F†`。固定Pの端点化・平均の線形性は使えるが、改善するPとkernelの適合性が必要 |
| Symmetric CPF型 | `Q W_ω Q`。同符号のQはinverse対ではなく、interiorに残る。全列を一組で包んだ形へ機械的に変形しない |
| Composite | symplectic成分のみ端点化可能かを判定し、symmetric成分は各step内のkernelへ含める。型とstep幅に応じた合成条件が必要 |
| Randomなprocessor | `P_ω` とkernelの相関、実現inverse、raw meanとcorrected meanが変わる。固定Pの式を使わない。freshにsampleした「逆」をactual inverseとはしない。今回は新lawを設計しない |
| MPF | linear operator／coherent signalの線形結合という別方式。signed weightと推定range、またはLCU accessを必要とし、この端点型の相殺規則を使わない |

根拠：[CPF Eqs.(4)–(6)][CPF]。D／Fともsymplectic、symmetric、processedという名称だけで同一の費用・保証を与えない。

## 6. Operator、固有位相、signal、biasの評価対象

将来の比較では次の層を保存する。これは新しい誤差boundや加法的な改善率の仮定ではなく、どの差を検証するかの指定である。

| 対象 | 比較する量と用途 |
|---|---|
| Dのoperator近似 | `G_D(h)` 対 `U_D(h)`。D単独の機構を識別 |
| 理想outer分割 | `J(δ)^q` 対 `U(T)`。Dをexactにしても残るD/R biasの参照 |
| 候補のideal-tail full列 | 各候補でtailを `U_R(δ)` としたordered列。現partial-S2とnested D列を別々に定義 |
| Normalization-corrected mean | 実現eventのmeanから得る `A_a(T)`。その候補のideal-tail列とのTaylor biasと、`U(T)` へのphysical biasを区別 |
| Raw mean | `M_a=E[V_a]`。共通normalizationがある場合 `A_a=N_tot M_a`。`A_a` 自体を実現unitaryや量子channelと呼ばない |
| 実装平均 | 有限bitのlawと合成後回路からのmean。理想event lawのmeanと区別し、その差を未測定のままゼロとしない |
| 固定状態signal | 同じphysical `ψ` で `z_a(T)=⟨ψ|A_a(T)|ψ⟩`、参照は `⟨ψ|U(T)|ψ⟩`。raw signal半径と補正後半径も区別 |
| 位相／energy | signalのphaseは半径・branch条件を伴う。kernel eigenphase、full信号phase、ground-energy biasを同一視しない |
| 統計 | 理想meanとの差のbiasと、weighted測定推定量の有限shot変動を分離。range／moment、IID、失敗確率範囲と配分を特定 |

固定unitary Pによる相似変換は、反復kernelの固有値を変えない。B2-Fで固定Pを既存kernelへ付けただけの改善は、主にoperatorの固有vector対応・固定状態overlapの側で検討すべきである。独立したPR実現列には単一の共通unitary kernel spectrumがあるとは限らず、corrected meanの固有値も一般にunit circle上にない。その偏角をそのまま物理energyとはしない。

B2-Dの `G_D R G_D` はDだけを処理するため、full元列への単純な相似変換とは限らない。したがってfull列の固有位相も変わり得るが、改善する向きは未検証である。processed familyのkernelを設計することと、固定kernelへprocessorを追加することの差を保持する。[直前調査 §4][SURVEY]、[Morales §II.3/IV.3][MORALES]

**「normalization-corrected」と「corrector-applied」を分ける。** 前者はfinite RTEのraw meanをNで補正した意味、後者はP／Qを適用した意味である。Pを掛けてもfinite Taylor polynomialがexact physical evolutionになるわけではない。Pがunitaryであっても、補正後meanをunitaryとして扱わない。[rte.py][RTE]

B2-D／B2-Fは一般に従来PRとは異なるfinite targetを持つ。旧finite targetへの一致検査や既存の数値許容差を、新しいphysical biasやfinite-bit allowanceの保証として流用しない。各候補のfinite targetを実現回路のmeanから定め、その `U(T)` との差を別に評価する。

## 7. 既存SOURCEの再利用と、必要な追加実装の境界

以下の処理は全て静的確認だけで、今回import・関数呼出し・tests実行をしていない。

| 固定SOURCE・接口 | 将来再利用する部分 | 追加実装／適合性確認が必要な理由 |
|---|---|---|
| PF `product_formula.py`：published Y8m10b、YP8m8の係数／列、`_get_kernel_s2_sequence`、`_get_processor_s2_sequence`、`_get_s2_sequence` | 既知係数、kernel反復と端点processorの分離 | 現PRの同名moduleは別版。数学的積とappend順、D-onlyの時間scale、丸めを照合するadapterが必要。係数再探索は不要 |
| PF `pf_decomposition.py`：`iter_s2_sequence_steps`、`inverse_s2_sequence` | 任意S2列のterm展開、隣接同termのmerge、逆順・符号反転 | 現PRの対称compact weight専用経路へ非対称processor列を直接入れられない。DF block／basisの展開とactual inverse対応が必要 |
| PF `processed_cost.py`：`ProcessedCostComponents.total` | kernel費と明示 `processor_pair_count` を分ける会計構造 | H-chain Pauli rotation／RZ-layerの既存tableはPR native価格ではない。D run／full endpointの数え方だけを先に再利用し、値は移送しない |
| PF H2 validator・response | full processed列でoverlapを評価し、direct spectrumと区別する検証方式 | grouped Pauli H2/STO-3Gの旧taskであり、DF-PRではない。D／full physical信号への入力・参照・評価対象の接続が必要 |
| PR `df_partial_randomized_pf.py` | 固定split・block選択、Hamiltonian identity、既存D回路基盤 | `_get_w_list`／symmetric PF経路だけではprocessorの任意列を表せない。新しいDF rank・`L_D` 探索を開始しない |
| PR `df_partial_s2.py` | `prepare_df_partial_s2` のblock・tail・位相metadata、個別blockのsigned指数・Gaussian basis往復 | requestは `pf_label='2nd'` 専用、前後は一方向sweep。B2-Dのstandalone D半stepとB2-F端点の列を表す接口が必要。既存契約を上書きしない |
| PR `df_partial_s2_repeated.py` | ordered trajectory、共通step設定、seed・attenuation、既存境界共有の検証規約 | 現boundary最適化は現partial-S2用。D run融合・Pの同一性・endpoint位置を扱う接続は未実装 |
| PR `rte.py` | 既存law／event、signed time、`finite_rte_operator_moments`、heterogeneous occurrenceのnormalization記録 | B2-Dはtailを固定して再利用。B2-Fのcorrector内へRTEを入れる新列には未適合。`compose_finite_rte_occurrences` はD interleavingを実装しない |
| PR `finite_rte_signal_validation.py` | exact／ideal-tail／corrected／raw operator、同じ状態でのsignal・半径・条件付きphaseの比較 | 現 `reverse @ tail @ forward` が固定。候補D／F列を明示注入する検証接続が必要。dense exact参照は安い量子accessではない |
| PR `df_rte_qiskit.py`、dictionary `circuits.py`／`gates.py` | event／identityのancilla相対位相、`actual_adjoint`、Hadamard X/Y、phase-fitなし `diag(I,V)` 比較、既存compiler修正 | dictionaryはsynthetic fixtureのtyped wrapper。分子DFや任意processorへの汎用wrapperではない。register、basis規約、state prepを接続して確認する必要がある |

出典：[PF列][PFSOURCE]、[PF展開][PFDECOMP]、[PF費用][PFCOST]、[PF validator][PFVALIDATOR]、[PF response][PFOLD]、[PR split][SPLIT]、[PR S2][S2]、[PR反復][REPEAT]、[RTE][RTE]、[signal][SIGNAL]、[DF RTE回路][DFRTE]、[wrapper][CIRCUITS]、[phase検証][GATES]。

既存testsは再利用先を選ぶ資料である。`tests/test_df_partial_s2.py` のordered dense reference、`tests/test_df_partial_s2_repeated.py` のnontrivial basis／signed step／controlled zero branch、`tests/test_rte.py` のmean／normalization、`tests/test_finite_rte_signal_validation.py` のoperator moment比較、`tests/management/test_pr_dictionary_compile_phase.py` の位相検査を確認した。旧PASSを新B2構成のPASSと呼ばず、将来必要な差分だけを追加する。今回新検証パイプラインや管理testは作らない。

## 8. 追加回路・shot・合成費用の構造

| 費用層 | B2-D | B2-F（固定unitary P） |
|---|---|---|
| 反復kernel | 各D run内の `K_D` 反復。残差を挟むため単一long-time D計算とは違う | full kernel／PR列の各step。内側誤差やtail event費も残る |
| Processor端点 | §4で数えるD runごと。隣接run融合は同一実現inverseに限る | 一つの連続execution内なら一組。Pが変わるscheduleでは境界を再確認 |
| Symmetric corrector | 各D近似内のinterior費を含める | full step内に残る費を反復kernelへ含める |
| Controlled化 | block中心部・scalar位相を既存規約で制御。Pの実現方法を別に比較 | full controlled evolutionの相対位相を保持。Pの全gateを必ずcontrolする必要があるとは限らない |
| 状態準備・readout | 各shotの同じphysical状態準備、ancilla X/Y測定と全D endpoint | 同じ状態準備に加えprocessor、readout。processor済み入力は無料としない |
| Tail・normalization | 同じ辞書、時間、`r_RTE,K` ならtail列と `N_tot` は固定。ただしDとのnative境界費は変わり得る | Pがdeterministicなら既存tail normalizationを保持可能。P自体にRTEを入れる案にはこの会計を流用しない |
| 合成 | signed DF指数、rotation精度、basis、Pへの近似bias | corrector合成と和指数の内部近似、actual inverseと別近似逆の差 |
| Setup／online | list／template／cacheを対照にも公平に認め、setup償却範囲を明記 | 同左。Pの古典生成・合成を候補だけ無料にしない |

固定unitary Pのconjugationは、register順序を固定すれば

`(I_anc⊗P) diag(I,V) (I_anc⊗P†)=diag(I,P V P†)`

と実装上表現できる。この場合はP両端をuncontrolledに置けるが、**その量子操作費は各executionで必要**であり、zero branchのactual inverse・相対位相を確認する。symmetric型の `Q V Q` に同じ方法を使うとzero branchが一般にidentityでなくなるため、この方式を流用しない。

費用会計では、kernel項、processor pair項、tail、状態準備、readout、合成を分離して記録する。これは[既存加算的cost接口][PFCOST]の考え方を引き継ぐが、native一体compileは境界共有に依存するため、項の単純和を測定済みfull costとはしない。round・axisごとにshot数を掛ける際も、同じ物理精度・bias配分・confidence、bounded/IID条件が必要である。新しいN、cost値、crossoverは本書では計算しない。

共通normalizationを用いるdeterministicなD／F処理なら、raw Hadamard測定のrangeとnormalizationを掛けた推定量のrangeは従来と同じ形で管理できる。ただし信号半径から位相精度へ換算する条件と必要shot数は再確認する。総資源の会計は各round・axisの `N_(m,b) × E[C_full_(m,b)]` の合計とsetup費を分ける。旧検証の状態準備なしinterrogation費を、状態準備・processorを含むfull-wrapper費と直接比較しない。

## 9. 強いbaselineと既知法との差

| 必須baseline候補 | 比較する役割と適用条件 |
|---|---|
| 現PRのpartial-S2＋finite RTE | 保存SOURCEの元列。B2-Dのnested配置変更も含めた実用上の対照として残す |
| 標準2次PF | 同じD半時間発展の位置に `S_{D,2}` を使う配置対照。必要なら同じ情報で内部反復を許す |
| Yoshida4次PF | 同じD blockからの既知高次化。低い次数でも短い列が有利な可能性を排除しない |
| Published Morales8次Y8m10b | fixed published係数のnon-processed対照。legacy arXiv-v2 m8だけを相手にしない |
| Published YP8m8 | B2-Dでは主候補であると同時に既知法そのもの。kernel-onlyとfull列の診断を区別し、processor pairを数える |
| 適用可能な既知CPF | exact二分割等の条件が成立する場合に含める。不成立の場合はaccess不足と記し、保証付き実装として捏造しない |
| Full deterministic PF | 同じDF snapshotとphysical精度で適用可能な、全Hの既知決定論PF。tailを使うこと自体を利益と仮定しない |
| その他の高次PF／MPF | 必要性が生じた場合に限り別taskとして追加判断。MPFのweight／LCU費をprocessor費へ混ぜない |

全armでHamiltonian・時間・固定入力状態・physical精度・confidence・許可情報・費用scopeを揃える。D内部だけのsame-cost比較は機構診断であり、full same-accuracy比較とは別である。exact行列／ground stateは全arm共通の古典参照としてのみ用い、候補の量子access・入力準備を無料にしない。各armに同じ合法cache、回路共有、内部反復の機会を与えるが、B-F／B-Mの係数探索や旧budgetを再開しない。[初期調査][INITIAL]、[published PF response][PFOLD]

本書の候補は**既知processed PFを既存PRへ接続する適用設計**である。nested化、Pの端点配置、actual inverse、同条件cost会計だけでは独立新規性は確立しない。新しいPR固有の方法差が残るか、端点費込みの資源優位があるかは未検証である。

## 10. 数学的保証・過去のnegative・未解決事項

| 資格 | 本書で認める範囲 |
|---|---|
| 既知理論 | exact primitiveの対称S2からのpublished PF／processed構成、固定Pとinverseの相殺、相似変換の固有値保存、CPFの型ごとの条件。丸め・合成後のPR版を認証したものではない |
| 既存SOURCE | 係数、任意S2列、inverse生成、加算的cost接口、PRの位相・mean検証基盤が存在。新B2 adapterは未実装 |
| 既存ローカル検証 | 直前調査が記録する有限RTE、PF、controlled wrapperの限定証拠を保持。今回再実行・再認証なし。新B2やimmutable CI／外部再現の証拠へ格上げしない |
| 保持する限定negative | B-Fは同family・情報／評価budgetでのfinite objective固有の追加decision valueを支持しなかった。B-Mは固定形式評価の既知再帰との同値性で停止。これらをprocessed PF一般のno-goにはしないが、係数再探索・既知評価式の再実装の理由にも使わない |
| 前回からの未解決 | B2-Dのfull physical signalへの改善、B2-Fのinexact／stochastic kernelに合う補正保証、native endpoint費、信号半径と必要shot数、finite-bit実装平均との差 |

過去の対象条件・固定証拠は[直前調査 §8][SURVEY]とその固定closure／カード参照に従い、本書で数値結果を再集計・再ランキングしない。停止済みのGaussian singleton・mixed-access系列も再開しない。`C_use` を厳密上界とせず、H12係数・最終全RPE総costは未決のまま保持する。

**CPF Eq.(9)は未解決のまま。** 出版紙面で中央指数の `/2` と通常S2定義の不整合を確認した直前調査を引き継ぐ。推測訂正した式を設計契約に使わない。本書の具体的なCPF参照は、独立に整合するEq.(2)と§III.B Eqs.(24)–(26)に限定する。Prop.1の一般k保証をそのまま現PRへ適用せず、表示問題の解消とinexact accessの問題も別に扱う。[直前調査 §4.2][SURVEY]、[CPF][CPF]

## 11. 最初に検証すべき候補と、将来の最小検証案

### 11.1 B2-Dを先にする理由

| 判断軸 | B2-D：D内YP8m8 | B2-F：full補正 |
|---|---|---|
| 狙う機構 | 非可換D内部の既知PF誤差を減らす | full splittingの一部commutator誤差と固定状態overlapを改善する |
| 前提 | 個別D block指数、standalone対称S2、published列の作用順・inverse整合 | full kernelに合うP、和指数／corrector合成、有限RTE下の意味論と保証 |
| 再利用 | 係数・任意S2列・DF block・tail lawを使える | wrapper／adjointは使えるが、適合するPと合成経路が未確定 |
| 強い対照との差 | 同配置Yoshida／Y8m10bに対してprocessor短kernelと端点費の交換を検討できる | 同一kernelへのP追加では固有位相は変わらず、operator／signal側の改善を示す必要 |
| 主な失敗要因 | D/R biasが支配、D誤差との相殺消失、短D runごとの端点費、合成丸めの支配 | 不適合なcorrector、和access不足、finite-RTE biasの残存、改善してもshot／合成費に負ける |
| 最初の判断に必要な検証 | D単独とfull signalの差分を同一tailで識別可能 | Pを特定する前提・理論検討が先に必要 |

したがって、**B2-Dの「D半時間発展にpublished YP8m8を使うnested PR」を最初の限定検証候補として提案する。** 最初に問うのは、YP8m8が最良かではなく、既知高次D近似がfull physical signalへ有意な改善余地を残すかである。Yoshida／Y8m10bが同等以上なら、その既知法を採用候補として報告し、YP8m8の勝利を作るための探索を自動追加しない。B2-Fは性能劣位と判定したのではなく、現時点では前提が多く未確定という理由で後に置く。

### 11.2 別途承認後に限る最小検証の提案

次は実行契約ではない。SOURCE・入力・数値許容差・小さい登録時間窓・候補の内部反復・終了条件は、結果を見る前に別途固定する。

1. **既存の小規模非可換fixtureを一つ再利用する。** 例えば `tests/test_df_partial_s2_repeated.py` の `_nontrivial_basis_case` にある固定one-body＋3 DF factor、`L_D=2` のsynthetic入力を候補とする。分子geometry／AO basis／分子DF rankは非該当で、D内にもD/R間にも非可換性があることと適切な非自明sector／状態を確認する。可換Dの対照だけで機構を判定しない。Hamiltonianの再生成・変更は今回行わない。
2. **作用順・inverseの技術的接続を先に確認する。** published列からのD-only operator、数学的Pとappend順、actual inverse、scalar・controlled zero branchを照合する。これは将来の実装と計算を要するため未承認。零誤差や数値noiseで高次を判別できない場合は未判定とし、時間窓を結果後に拡げてfitを作らない。
3. **同一入力のoperator／mean診断を行う。** D単独、exact-D/R参照、現partial-S2、同配置S2／Yoshida4／published Y8m10b／full YP8m8を比較する。CPF armは既知access条件を満たす場合だけ追加判断する。fixed `δ,r_RTE,K,q` と同じtail lawで、ideal-tail、finite corrected mean、raw meanを分離する。最初は一stepと、隣接D融合・順序を識別できる短い反復までとし、長RPEへ進まない。
4. **同じphysical状態のX/Y信号を照合する。** operator精度が改善してもfull signalが改善しないcase、半径／位相が悪化するcaseを明示する。正の半径とbranch条件なしにphaseを合否指標へしない。exact ground stateを使うなら全arm共通の診断用で、無料の実用state preparationとはしない。
5. **改善余地が残る場合だけ、別承認の回路・統計段階を提案する。** full wrapperのraw／compiled相対位相、D run融合、endpoint費・tail境界を照合し、研究Aとfinite-bit allowance・range／IID・confidenceを確定する。その後に限り必要shot数とsame-accuracy資源を比較する。最初のoperator診断にnative資源・N・全RPEの科学callを同梱しない。

D単独改善があってもfull改善がなければ、今回の入力／窓でD改善を追う優先度を下げる資料となる。full改善があっても、より短い既知baselineに端点込みで負ければ資源GOとはしない。結果が不明瞭なら未決として返し、自動的な別geometry・rank・split・係数探索は行わない。科学的GO/STOPの数値閾値や最終採否は本書で正式採択しない。

将来の承認を分ける対象は、①最小adapterと必要な技術tests、②固定小規模operator／signal計算、③必要時だけのbuild／compile・統計資格・資源比較である。重要な理論・新規性・採否の詳細科学レビューはこれらと別の開始承認を要する。新しい計算契約ではwall・CPU・GPU・memoryの恣意的上限を既定にせず、既存実装の再利用・実資源確認・並列化を計算前に行い、共有環境や他ユーザーへの変更はしない。

## 12. 研究A・C、統括への引継ぎとSTOP

| 担当 | 次に判断・作業する内容 | 今回の状態 |
|---|---|---|
| 研究B／統括／ユーザー | B2-D先行の提案、最小fixtureと改善対象を採択するか判断。B2-Fの理論上の不足に追加検討を投資するか判断 | 提案まで。正式採択・科学GOなし |
| 研究A | tailを固定しても変わるsignal半径、weighted range／moment、finite-bit実装平均差、必要shot数の条件を整理 | 新law・新N・新統計資格を今回作らない |
| 研究C | C1圧縮DFを優先し、将来確定する表現のblock・scalar・rank／threshold・split identityをBへ渡す。processorのnative費用・phaseと共有条件を連携 | C1 SOURCE変更・再実行・共同実装・共同科学計算なし。published PFの普遍的な重み列は再利用対象だが、旧DFの誤差係数・費用値を新DFへ無検証で移送しない |
| Codex | 次の明示指示後だけ、承認されたadapter・固定検証・公開作業を担当 | 今回は本書作成・公開確認まで |

統括へ渡す短い要約：**B2-Dを最初の限定検証候補として提案。published YP8m8でD半時間発展を近似するnested PRを、同配置S2／Yoshida4／Y8m10bと現partial-S2に対して比較する。D内部改善をfull signal改善と同一視せず、残差を越えるprocessor相殺は認めない。B2-Fは固定Pのmeanとendpoint配置を定義できるが、現finite-RTE kernelへの補正保証・和access・合成が未確立。CPF Eq.(9)の紙面不整合は維持し、独立に整合する通常PF2の既知式だけを限定参照する。新規性・総資源優位は未確立。C1圧縮DFを最優先とし、B2実装・科学検証・重要レビューは別承認。**

変更は指定Markdown一件のみ。固定SOURCE・raw結果・grant・ledger・marker・科学契約・共有概要・manifest・旧mandatory STOPを保全し、他研究branchをmergeしない。文書の参照・差分確認以外のtests／実行管理処理を追加しない。新規科学計算、Hamiltonian生成、保存数値の新解析、回路build／compile、新費用・誤差係数の生成は行っていない。

専用branchへ本書だけをcommit・pushし、remote SHAと本文一致を確認した時点で **mandatory STOP**。正式な機構採択・新SOURCE実装・科学実行・詳細科学レビューへ自動進行しない。

## 参照先

[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/docs/research/research_b_b4_b2_pr_connection_static_survey_20261024.md
[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/docs/research/research_b_time_evolution_initial_survey_20261024.md
[MORALES]: https://arxiv.org/html/2210.15817v3
[CPF]: https://arxiv.org/html/2409.08265v4
[PFSOURCE]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[PFDECOMP]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/pf_decomposition.py
[PFCOST]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
[PFVALIDATOR]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py
[PFOLD]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/README.md
[SPLIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trotterlib/df_partial_randomized_pf.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trotterlib/df_partial_s2.py
[REPEAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trotterlib/df_partial_s2_repeated.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trotterlib/rte.py
[SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trotterlib/finite_rte_signal_validation.py
[DFRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trotterlib/df_rte_qiskit.py
[CIRCUITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trottertracks/pr_dictionary_validation/circuits.py
[GATES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/src/trottertracks/pr_dictionary_validation/gates.py
