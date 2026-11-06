# 研究C：RC-DF・SCDF・UWC・保存DF baselineの公平比較可能性

2026-11-06 JST。静的調査報告。**新規科学計算・保存結果の再解析・手法実装・重要科学レビューは行っていない。**

主な結論は、再配置後のZZ-only massがRC-DFのweighted entrywise L1族へ厳密に書ける一方、保存ENTRYはその重みを使っておらず、原法のsolver・精度選定・資源評価まで同じではない、ということである。SCDFはsector symmetry shift、UWCはPauli係数分布と可変head/tailを扱い、単なるJ/U optimizerの設定変更とは区別する。公開UWC実装を今回実際に取得できた。保存XDF8/CDF8は歴史的D対照として使えるが、再配置後の新規確認標本にはならない。

## 1. 調査目的・対象・取得資料

目的は比較の実行可否と不足処理の特定であり、性能順位・独立新規性・論文化GO/NO-GOの認定ではない。主agentのみで実施した。独立branch `codex/research-c-baseline-comparability-20261106`、独立worktree `partially-randomized-trotter-worktrees/research-c-baseline-comparability-20261106`、基点は `a67036a6baafc00deaa38a1b0b801d313d519ad5`。追跡対象の変更は本報告書一つに限定する。

共通運用指示2026-10-25版、projects/AGENTS.md、worktreeのAGENTS.md、PROJECT_MAP、研究概要、研究目的、VALIDATION_STATUSと中央manifestの関連状態を確認した。固定実行報告を、過去の概要内の一般記述や過去レビューの要約より優先した。中央manifestには現在のH4統合結果の登録がある。旧grouped-UWCのprose-only/missing-rawおよび旧DF screeningの無効状態は、今回見つかった著者UWC実装と別物である。

### 1.1 固定repository資料

以下は指定commitのGit objectを読み、GitHub connectorでも本文を取得した。別系列の二資料を、基点worktreeに存在すると偽らず固定commitから読んだ。

| ID | 固定commit | 取得・読解した資料 |
|---|---|---|
| H | `417fa459b823509be5c2010c5836e58955088282` | [H4統合結果][H]。旧辞書PR/ENTRY、DF・PF/RTE・RPE・native・統計 |
| Z | `d601e0ef4375c37185211aa75685ae0e582f3e64` | [固定Jの再配置mass診断][Z] |
| P | `a67036a6baafc00deaa38a1b0b801d313d519ad5` | [下流技術準備][P]と[未承認契約][PC]。SOURCE/候補/処理の参照のみ |
| L | `1a588f5491bce394ecf039ee4d2de57581e2e0b1` | [既存先行研究整理][L]。一次資料の探索入口 |
| E | `5a53d0569b70e4417765c4876616260db911e603` | [論文証拠対応][E]。歴史的な証拠資格を参照 |
| D | `417fa459b823509be5c2010c5836e58955088282` | [保存強D比較][D]。B2 CDF8/B3 XDF8の固定結果 |

上表7件は、GitHub取得本文をGit blobとして照合し、指定されたローカル固定commitのblobと全件一致した。Markdown読取と来歴照合であり、保存科学数値の再解析ではない。

Pは**その固定snapshotでは未承認・未実行の準備資料**である。別Codexの現在の実行承認・進捗・結果をこの記述から判断しない。進行中の実行worktreeの科学出力、cache、QPY、rawは読まず、優先順位を結果依存で変えていない。

Claude独立レビュー原本は今回未取得。原本の結論を確認済みと扱わず、固定SOURCEと一次文献による以下の対応を根拠にした。IDEのTrack C2資料は今回の固定SOURCE指定ではなく、別研究の実行条件を混入させていない。

### 1.2 一次文献の版と読解範囲

| ID | 実際に確認した版 | 関連節・式・付録 |
|---|---|---|
| R | [RC-DF arXiv:2212.07957v3][R] | 本文のfactorization/regularization節、式2–8、13–17、Appendix A式18–31、Bの強度選定、F/Gのnumber→Pauli・FFF。HTMLは本文見出しもAppendixと表示するため式番号で特定 |
| S | [SCDF arXiv:2403.03502v1][S] | §2.1–2.4、式19–27、§3.1、Appendix A/B。v2/v3は存在する版と扱わない |
| W | [UWC arXiv:2603.22778v2][W] | §II.2.3–4式14–18、§III式19–37、Appendix B/C/D、§VI、Data availability。予備確認のv1から、判定用はv2へ固定 |
| T | [PR arXiv:2503.05647v2][T] | 固定PDF pp.12–14の§VII.A/B、式42/43と回転frameでのrandomizationの議論。DF・norm削減・係数集中の先行関係を確認 |
| C | [CDF arXiv:2104.08957v1][C] | §II.1.3式14–31。一般core、固定Uの係数solve、outer U更新。preprintの題名はCompressedなし |

要旨のみから実装を推定していない。関連節の静的読解であり、各論文の全定理・全数値の独立再証明や再現実行ではない。RC-DF本文のL1二乗と付録の非二乗L1は区別する。

## 2. 現在の研究Cの科学条件

### 2.1 表現・目的関数

固定物理系は直鎖H4、隣接1.00 Å、STO-3G、4空間/8 spin軌道、Nα=Nβ=2、sector36。一般実対称signed core Jとframe別実直交U、number-CDF8、元順LD2である。Tensorは

\[
g_{\mathrm{DF},pqrs}=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj},\qquad
f=\tfrac12\|g-g_{\mathrm{DF}}\|_F^2.
\]

実SOURCEは `src/trottertracks/c_pr_aware_df.py:coefficients/make_value_and_grad` と `c_near_objective.py:entry_weights/mass/make_value_and_grad`。以下の式はその静的対応で、保存Jを新しく評価したものではない。

\[
F_{\rm PR}=f/f_*+\gamma(M_\epsilon/M_*)^2,
\quad M_\epsilon=\sum_{t\ge2,\ell}\sqrt{(TJ^t)_\ell^2+\epsilon^2},\quad\epsilon=10^{-8}\ \mathrm{Ha}.
\]

K_ab=J_(a//2,b//2)/2、c_Za=−Σ_b K_ab/2、c_ZaZb=K_ab/2(a<b)、identityはmassから除く。非平滑massは

\[
M=\sum_{t\ge2}\left[\sum_i|\sum_jJ^t_{ij}|+\sum_{i<j}|J^t_{ij}|+\tfrac14\sum_i|J^t_{ii}|\right].
\]

保存ENTRYは `B_eta=Σ_(t≥2,ij) w^E_ij sqrt(J_tij²+eta²)` の二乗を正則化し、ordered重みはdiag=5/4、off=3/2（unique offでは3）。eta=epsilon(2n+1)/(1.5n−.25)。PRと零点massを合わせたもので、全点で同じ平滑化ではない。強度ごとの初期massを一度だけscaleとし、終点で再正規化しない。両目的のhead Jもtensor fitの変数であり、固定head係数ではない。

Hの3start×3strength×2arm、16/64更新は有限recipeの比較。36checkpointでDF予算PASS、J一次MET/U一次NOT_METであり、最強の収束済み手法間比較ではない。物理精度は元g補正κと核scalar一回の下で、sector spectral norm＋engineering guard≤CA/8=.00019875 Haを別判定する。Tensor lossだけでこの条件を代用しない。[H4固定結果][H]

### 2.2 再配置と下流条件

`s_ti=Σ_j J_tij` と置き、tail frameの線形Zを

\[
L_t=\sum_{i\sigma}s_{ti}\widetilde n_{ti\sigma}-(\sum_i s_{ti})I
\]

としてheadへ移す。κ'=κ+Σ_tail U_t diag(s_t) U_t^T、全体scalar c'=c−Σ_tail,i s_ti。Tailの元identityとZZを保持し、線形Zだけ除く。これは**全Fock空間で同じH**を別のsplitにする恒等変換である。再配置済み同一modelから同J/U Dを作り、LD8で再配置し直さない。SOURCEは `c_h4_relocated_downstream.py:prepare/raw_generators`。[固定準備§3–4][P]

元のHではpartial ordered S2/s1、δ=h=.1、K0、m=0..12。Own-DF sector ground oracle、状態準備費用除外、CA=.00159 Ha、βtotal=π/6、βstat=π/12、failure=.01/候補、axis=.01/26。P0 Qiskit1.3.0/opt1/seed17、rz/sx/x/cx、control位相保持、fenced serial、shell込み。費用は `W×mean_kernel+shell`。rはkernelに含み、native価格標本数はRPE shotsと別である。

HのC0/64更新の旧辞書公平retuning45.47%削減はこの旧条件付きモデルの引用であり、再配置後へ移さない。Zの主3候補のM_ZZ=.472232/.380832/.381049 Haも固定mass診断の引用に限定する。新しい比率・集計・ランキングを作っていない。[H][H]、[Z][Z]

Pの保存3候補、r={8,16,32,64,128}、selection32/final64、同J/U D、主2差97.5%案は参照条件であり、本報告で採択・変更・実行しない。PR全law synthesis/finite-bit law信号誤差は未資格で、ideal/exact-native仮定付き費用とactual CAを区別する。

## 3. RC-DF比較可能性

### 3.1 数学的に一致する部分と一致しない部分

RC-DFの空間core ZはCのJと同じtensor contractionに入る。**Z_RCDF=Jであり、spin係数K=J/2ではない。** RC-DFは一般対称coreで、PSD・rank-one制約を要求しない。U=exp(X)のSO表現は、density projectorを変えない列符号gaugeを考えればCの実直交frameと対応する。原論文の式17はfull core L2、本文はweighted L1/L2の `(Σρ|J|^p)^(2/p)` 族も記す。主数値例はuniform L2で、Appendix A式18は非二乗L1/L2を記す。[R式5/8/17/18][R]

**今回の比較導出**：再配置後は行和項がtailから消えるので、任意のsigned symmetric Jについて

\[
M_{\rm ZZ}=\sum_{t\in R}\left(\sum_{i<j}|J^t_{ij}|+\tfrac14\sum_i|J^t_{ii}|\right)
=\sum_{tij}w^{ZZ}_{tij}|J^t_{ij}|,
\]

\[
w^{ZZ}_{tij}=\begin{cases}0&t\notin R,\\1/4&t\in R,\ i=j,\\1/2&t\in R,\ i\ne j.\end{cases}
\]

したがって、**非平滑のM_ZZ²は、tail maskとこの重みを選んだRC-DF本文のweighted L1二乗族の一員**である。旧Mの場合に必要だった「row内符号相殺なし」は不要となる。証拠は上式の係数対応と `c_h4_z_relocation.py:diagnose` の既存定義であり、新しい数値試験ではない。

`f/f_*+γ(M_ZZ/M_*)²` をraw lossに戻すと `f+α M_ZZ²`、α=γ f_*/M_*²。RC-DFの二乗内重みならρ=√α wZZで一致する。非二乗 `Σρ|J|` と次数が異なり、full L2とも一般に同じ関数ではない。Fidelityとのtradeoffを変えるため、「massとその二乗が単調」という理由でtensor loss込みのargminを同じとしない。

| 比較軸 | 一致に必要な条件 | 現在の差 |
|---|---|---|
| 対象frame | tailのみならhead重み0 | RC-DF主例は全frame一様、Cは元ラベルLD2以降 |
| Entry重み | ordered diag1/4、off1/2 | 保存ENTRYは5/4、3/2。diag比5とoff比3で一つのstrength倍では直らない |
| べき乗 | 全tailのL1和を一度二乗 | Appendix Aの非二乗L1、式17のentry L2と別 |
| 平滑化 | epsilon=0で上記は厳密。epsilon>0なら定義を一致させる | 現PRはmapped atomごと、ENTRYはentryごとの異なるfloor |
| Tensor fidelity | 同じg、f規約、物理DF予算を別確認 | 論文のCCSD(T) heuristic/測定MSEはsector norm予算ではない |
| 一体項・scalar | 同じ近似Hから整合して再配置 | 論文式6/13の全frame Pauli一体項と、Cのtailのみの再配置は別 |
| 変数・solver | 一般J/各frame Uは対応 | 原法のnested solveと有限recentered U/Newton-CG recipeは同一ではない |

RC-DF式13の全λ_LCUは一体項Σ|F_k|と全frameのZZ massを含む。M_ZZはその**tail二体部分と同型**であり、全λやλ_Burgを最小化したという意味ではない。線形Z再配置は既知のPauli表現・FFFの再分配自由度と関係するが、FFFの状態依存分散最適化まで実装したものではない。[R式13–15、Appendix F/G][R]

近似tensorの場合にはさらに注意が要る。Cは元g由来κを保持し、再配置補正は**候補J/Uの行和**から作る。原gから直接作ったFや論文のscalarを、そのまま新J/Uへ付けると別Hになり得る。変換前後の同一候補Hを保持してから、元物理Hへの誤差を別評価する。

### 3.2 パラメータ変更だけでできる範囲

- **既存full uniform L2/RC2目的の利用は可能**：`c_rcdf_comparator.Objective/initial_spec/make_value_and_grad/fit` は既に `f/f_*+strength Σ_all,ij J²/l2_scale` を実装する。原法のuniform L2値に対応し、既存solver接続を使える。ただし現在の固定H4候補を新しいRC2候補へ変える実行は新科学計算である。
- **再配置整合weighted L1二乗は、現公開parameterだけでは不可**：`c_near_objective.entry_weights(norb,xp)` は固定重みで、Objectiveにweight引数がない。epsilon>0必須の `pr.Objective`、固定ENTRY identityを検査する `c_near_stable.validate` もある。LDやstrengthの変更ではwZZにならない。
- **最小拡張候補**：独立のobjective定義としてtail mask/wZZ/平滑化/初期scaleを追加し、ADだけでなく `c_near_stable.Coefficients.T`、`penalty_epsilon`、`metric_chart`、`parts_gradient_J` の独立解析微分・Hessian接続まで同じ定義にする。既存ENTRY/PRの値・VERSION・保存結果を上書きしない。Tensor linear map、codec、U更新、物理admissionは再利用する。
- **原法solverの比較**：RC-DFはU更新と固定UのJ解法を交互に使い、L2はridge線形系/CG、L1は非平滑条件を扱う。既存RC2 joint fitを原法再現と呼ばない。公開CUDA-QのL2 nested implementationを先に検討でき、一からsolverを作る必要はない。

同じmask・重み・平滑化・scale・solverを使う「再配置PR」と「RC-DF型L1」が**同一目的になる場合は、二つの名称で重複fitを走らせても方法差を検証できない**。比較すべきはfull/tail、L1²/L2、原法solver/共通recipe、同じ物理精度での下流費用など、残る具体差である。独立新規性の正式判断には進まない。

## 4. SCDF比較可能性

SCDFの一般core式26は `f(U,V)+ρΣ_tkl|V_tkl−α_t|`。論文の採用式27はV_tkl=W_tk W_tlを用い、U/W/αを最適化する。未shift Vはrank-one PSD（一般には半正定値）、shift後 `V−α11ᵀ` はrank≤2で不定になり得る。一般signed JのCDF8へshiftだけ足した方式は式26系の拡張であり、採用SCDFそのものとは区別する。[S§2.2–2.4、式26/27、Appendix B][S]

粒子数shiftは `H_S=H−a1 N−a2 N²`。Frame shiftはα=Σ_t α_t、a2=α/2で、各frameのcoreを `J_t−α_t 11ᵀ` にする。Uの直交性から全frameの粒子数Nは同じである。**各sectorの固有状態は保つが、元の全Fock spectrumを同値にはしない。** 固定N0のenergyは補償scalarを戻して比較する。

### 4.1 Cのnumber規約で必要な処理（本調査の対応導出）

\[
H=c+\kappa E+\tfrac12\sum_t\sum_{ij}J^t_{ij}\widetilde E^t_{ii}\widetilde E^t_{jj}
\]

から、`J'_t=J_t−α_t 11ᵀ`、κ'=κ−a1 I、`c'=c+a1 N0+(Σα_t)N0²/2` とすれば

\[
H'=H-a_1(N-N_0)-\tfrac12(\sum_t\alpha_t)(N^2-N_0^2).
\]

H4ではN0=4でtarget sector上H'=Hだが、他sectorは異なる。これはCのnumber式の対応であり、論文のPauli一体係数fへのshiftを同じa1数値で二重適用しない。Bare h/gからshiftする場合はκ補正とidentityを改めて整合させる必要がある。SCDFのtensor fitting lossは**unshifted V=WWᵀ**の再構成に置き、shift後J'を元gへ直接fitして「精度不良」と誤判定しない。Truncationがあれば別の物理誤差として加える。

線形Z再配置は任意sectorで同じHの分割を変更する操作、SCDF symmetry shiftはtarget sector以外のHを変更する操作であり、独立の自由度である。Shiftした後で線形Z再配置することもできるが、その順序、補償scalar、head/tailを明示し、新PFを評価する必要がある。

### 4.2 全512検査を維持する具体的方法

1. Shift後の各number frame、one-body、補償scalarから**独立したshift後full-Fock CAR reference**を用意する。Raw/compiled/実inverse/control0/1/X-Yを、そのreferenceに対して従来どおり全512入力・全512出力で検査する。
2. 元物理Hとの一致は `P_sector(H'−H)P_sector` とsector外leakage/粒子数保存を別検査する。元Hへの全512一致を要求すると合法なshiftも失敗する。逆にsectorだけのraw/compiled比較へ弱めない。
3. Sector ground oracleとscalar/energy補償・aliasを結ぶ。Shiftが他sectorのエネルギー順位を変えても、full-Fock全体のgroundを新たなtargetにしない。有限PFもtarget sectorを保つか確認する。
4. 元の `prepare_candidate` の `same saved H after relocation` や保存factor identityは、そのままでは通らない。独立namespaceのshift-aware接続と新しいreference bindingが必要で、旧gate/STOPを無効化しない。

既存 `SectorCAR.one/raw_frame`、一般Kのsigned I/Z/ZZ、control・位相保持compilerは再利用可能で、rank2 coreをPFのために二frameへ分ける必然性はない。一方、論文のqubitization価格を再現するならrank2/negative-eigenvalueの追加実装費用をAppendix Bどおり数える。PFにgeneral-Jとして輸入する費用と同一ではない。

原法のW→α median→U nested最適化、rank-one変数化、one-body median shift、second-factor truncationとaccuracy選定は現C optimizerにない。著者配布物にはfitterがないため、原法比較にはこの科学実装が必要になる（§8）。一般Jで固定J/U shiftだけを試す場合はより小さい拡張だが、別名称の限定baselineとする。いずれもPF/RTE・RPE・native・価格標本は候補固有に新しく必要で、旧値を移さない。

## 5. UWC比較可能性

UWCは *Unitary Weight Concentration*。Pauli-LCUの係数を大きさ順に並べ、headは上位LD個のPauli項、tailは残り、λ_R=Σ_tail|c_l|とする。最終roundの離散目的は `min_LD[g_D LD+g_R λ_R²]`、soft版は

\[
s_\ell=\operatorname{sigmoid}((|c'_\ell|-w)/\epsilon_s),\qquad
G_{\rm soft}=g_D\sum_\ell s_\ell+g_R[\sum_\ell |c'_\ell|(1-s_\ell)]^2.
\]

境界wは離散最適境界の隣接係数の平均から更新する。変数はglobal orbital rotation、number BLISS、spin拡張BLISSである。Tensor近似fitを主損失にするCと違い、対象blockのspectrumを保つ表現変換で費用を下げる。論文はPauli-LCUに集中し、DFへの拡張を§VIの将来課題と述べる。[W式17/18/22–37、Appendix C/D、§VI][W]

| 軸 | UWC | 研究C |
|---|---|---|
| 表現族 | 単一global軌道基底のPauli-LCU | 各frameに別U、number-CDFと回転Z/ZZ atom |
| 最適化変数 | OO/BLISSの積分変換 | 独立J_t/U_tによる近似tensor fit |
| Head/tail | Pauli重み順、可変境界、soft count | 元ラベルframe prefix LD2、固定mask |
| 目的・制約 | head費用＋tail²、block spectrum保存 | f/f*＋tail surrogate²、物理誤差予算は独立admission |
| Cost model | 論文のPauli rotation/RPE・最終round近似、STAR資源へ接続 | 全13roundのPF/RTE誤差・attenuation・shots/W・P0完全kernel＋shell |

**UWC目的をJ/Uへそのまま代入するだけでは公平比較にならない。** 仮にc=TJと置いてsigmoidを追加しても、Cのheadはframe全体であり、atomごとにheadへ昇格させるbuilder/PF分割はない。回転frameのevent単価はUと隣接basis融合に依存し、Pauliの一律gD×個数では表せない。J/U近似は元Hを一般には厳密保存しないため、物理DF制約も引き続き必要である。

DF版への適用には、(a) frame/atomどちらを昇格単位にするか、(b) J/Uとshiftのどこを変数にするか、(c) 不同basisでの決定論・RTE費用、(d) fidelityとsoft loss、(e) hard境界への復元・適格判定を設計する必要がある。これは新しい科学目的/処理であり、著者Pauli-UWCと同じ手法と自動認定しない。固定LDでgDを定数にし、shiftも外したtail²だけなら§3のweighted-L1族へ縮退し得る。

### 5.1 著者コードの取得により分かったこと

[FujitsuResearch/early-ftqc-qpe-resource][WG] のcommit **`8a74108e308bd72ce1d8ea45aafac5ebb08bc32e`**を固定し、README、LICENSE、`scripts/uwc_optimization.py`、`src/soft_gate_count.py`、`molecular_hamiltonian.py`、`partially_randomized.py`、`cholesky.py`を取得・関連関数を読解した。実行・import・installは行っていない。論文内の公開予定という文言だけから「公開実装なし」とは結論しない。

- `SoftGateCountOptimizer(h1e_u,h1e_d,eri_uu,eri_dd,eri_ud,nele,Sz,xi,eps,...,mode)` → `optimize(...)` → `optimized_integrals` は5個のspin別積分とcore offsetを返す。H4の元h/gを同じspatial/chemist規約へ変換し、N=4/Sz=0、保存核scalarと返却coreを一度ずつ結ぶ必要がある。
- `find_optimal_L_and_tau`、`gate_count_soft`、`jitted_minimize`、OO/number/spin BLISSは再利用候補。JAX AD＋SciPy SLSQPの実装であり、DFのJ/U配列を直接受けるAPIではない。
- CLIは既定がqDRIFT、`--RTE`でRTEとなる。class既定mode='RTE'との差を無視しない。資源estimatorにもqdrift既定Trueがある。CのK0 finite-RTE/13roundへ費用を移すには別照合が必要。
- 係数APIは**絶対値の配列**を返し、Pauli support/符号を保持したnative circuit IRではない。`truncate=True`の外側APIと、zero slotsを含むJAX内側APIも別。Native比較にはsigned operator列、bit順、truncation予算を用意する必要がある。
- `__post_init__`のCholesky初期化はその時点で新しい分解・norm評価を行う。今回classを生成していない。固定版コードは候補basisを先頭3個に限定する経路があり、論文Appendix Dの全Cholesky候補という記述をそのまま実行済みと扱わない。
- `PartiallyRandomizedResourceEstimation` は係数からの経験的Trotter係数・独自shots・STAR価格を使い、Cの全512回路検査・実測P0価格を出力しない。C側との同条件比較用adapterが必要。

公開実装があるためOO/BLISS/SLSQPを一から再実装する理由はない。**Pauli-UWCのH4への輸入**は既存科学コード＋新しい入出力/native/契約接続、**DF版UWC**はその上に新しい科学設計、という負担を区別する。

## 6. XDF8/CDF8保存baseline

保存強Dの参照はD。物理元入力P-H4容器SHA256は `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。研究Cのh/g/scalar subset容器 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` と同じファイルbytesとは称さず、元入力→subsetの保存来歴とκ/scalar規約で対応を確認する。

| 項目 | 保存CDF8 | 保存XDF8 |
|---|---|---|
| 系・sector | H4直鎖1.00 Å/STO-3G、Nα=Nβ=2 | 同じ元P-H4、sector36 |
| 生成 | ffsim0.0.83、number CDF、max_vecs8/tol1e-8/L-BFGS-Bの保存1 fit | OpenFermion1.6.1 eighの保存元順prefix8、signed λ/Gを保持 |
| 表現 | 一般対称J/各frame U、rank8 | 保存density-square8 frame。J/Uへ変換するならsigned λと1/2規約を照合 |
| 精度 | norm＋guard=7.666215916301914e-5 Ha | 9.533901455091894e-5 Ha |
| 適格性 | 共通CA/8予算PASS | prefix1..9の保存順ではrank8がfirst-passing。全XDF族の最小rankではない |
| 候補identity | CDF export SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`、route `B2_CDF_relative` | 元packet＋prefix8＋route `B3_XDF_spin_relative`。新しいCDF checkpointのidentityと混同しない |
| 基底・fusion | legacy CDF bases＋relative Gaussian | exact interleaved spin lift＋relative Gaussian |

CDF8生成SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11` のfit後に旧raw-reference不一致でSTOPし、修正評価で保存fitを再利用した来歴を保持する。B2科学SOURCEは `880006396326a4d46f4a4030564279e92fadb9cf`、B3は `b40fce5c62c9fff97f68073cbc02182e4bbf2b9b`。総資源集計SOURCEは `1aeedd96178a3f50a15bc2701c3c332f4f391b64`、数値実装 `969c17d623d56eb645417b281deadb1be4757699`。これらはH/Pの科学SOURCEとは別である。[候補生成準備](research_c_c1_stage1_technical_preparation_20261025.md)、[prefix診断](research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)、[B2来歴](research_c_c1_h4_strong_baseline_execution_20261026.md)

保存実行は全frame ordered S2、s=1のみ、δ=.1/m0..12、n=2^m、PR split非適用。Own-DF sector ground oracle、ideal IID Born、状態準備除外、CA/DF reserve/β/failureは§2と共通。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、Gaussian `nature_jw_exact_zero_lapack_atan2_v1`、MPFR160 declared-phase、ordinary control、fenced serial、head/X/Y shell分離。全512数値Frobenius/norm-growthからの伝播はlevel Bで、丸めenclosureではない。[D][D]

既存表の引用では、CDF8/XDF8のkernel RZは11,062/10,885、shotsは13,606/13,600、総RZは102,056,444,569/100,267,421,030。**今回これらを再計算・再集計・新しいPR値との比率算定はしていない。** 新しい独立final cohortでもない。

比較資格は二層である。

- **DIRECTLY_REUSABLE**：元条件を併記した歴史的な強D資源参考値、同じ物理精度予算を通る保存候補、既知のspin/relative Gaussian改善例として引用できる。
- **CONTRACT_ALIGNMENT_REQUIRED**：再配置後のPRの同条件下流baselineとしての新比較。旧Dは再配置modelから生成されず、旧CDF8は現在のP/E候補とも異なる。H全体が同じでも一体項とframeのPF分割が違えばDのPF誤差も変わり得る。候補固有D、builder/phase/fusion、誤差laneと統計政策を揃えた接続が必要。

旧Dが安いことから、そのDFを再配置後PRで使っても安いとは推論できない。両保存DFの誤差は新しいPR/ENTRY checkpointの誤差より大きく、同予算PASSと同じ到達精度を混同しない。既存Dの数値CA資格とPRのideal/exact-native条件付き資格を、同じ保証水準であると表示しない。

## 7. 共通比較条件・交絡要因

以下は**将来比較を選ぶための候補条件**であり、新規科学契約の採択ではない。

| 比較軸 | 揃えること／区別すること |
|---|---|
| 物理系 | H4直鎖1.00 Å/STO-3G、同じ元積分・MO空間・核scalar。別論文のSTO-6G/1.4Bohr等の値を換算流用しない |
| 電子sector | Nα=Nβ=2、N=4、Sz=0。SCDF/BLISSは対象sectorの同値、OOは状態のunitary変換も結ぶ |
| 表現 | Number-CDF、XDF-square、Pauliを明記。名称の同じλ/LDでも辞書と単位を明記 |
| DF精度 | 共通物理精度予算と誤差の会計を必須にする。可能なら到達誤差帯/Pareto比較も示す。Tensor toleranceやCCSD(T) heuristicだけでは代用しない |
| Rank | 同rankは一般J内の目的ablationに有用。rank-one SCDF/XDF、Pauli-UWCとの手法全体比較には精度一致を先にし、rank/second-rankを別資源として報告 |
| 変数 | J/U、one-body、shift、head/tailの可変/固定を明示。自由度が違う比較を純粋なloss因果ablationと呼ばない |
| 目的 | fの規約、loss scale、weight、べき乗、smoothing、初期化、有限budget/一次停止、強度選択機会を結ぶ |
| 再配置 | 同じlinear-Z規則とidentity/scalar会計。全Fock恒等変換とsector shiftは分離 |
| PF/RTE | まず共通政策の比較と、各方式に同等tuning機会を与える比較を分ける。LD2はCDF frame数でありUWCのPauli head2項へ固定しない |
| RPE | 目標energy、round horizon、alias、β・failure/shots政策を共通化。論文UWCのξ/最終round近似をCのexact own-ground/13roundへ無断で移さない |
| Ground/準備 | 各方式のown-sector oracleまたは共通physical状態のどちらかを明示。状態準備除外は仮定でありshots差があれば費用は一般に相殺しない |
| Native | 同compiler profile、phase/control/実inverse、Gaussian/canonical gauge、許すfusionとfenceを揃える。異なる表現に自然なbuilderは、その費用も含めて比較 |
| 指標 | 同じ総RZ/CX/depth/sizeとshell/W。Qubitization Toffoli、STAR物理時間は別cost modelで、RZから固定倍率で推定しない |
| 統計 | Selectionと独立finalを分離、同等tuning機会、family/精度政策を事前に決める。旧標本pool禁止。候補再利用・D共有の依存を保持 |
| Correctness | full512 raw/compiled/実inverse/XYは各方式のfull referenceへ。元Hとの一致は必要に応じsectorで別確認。丸め/全law未資格はnull |
| 環境 | 固定版API/依存を分離環境で扱い、現fit/evalをupgradeしない。軽微な環境差で無意味な再計算を要求せず、数値結論に関わる差を検証 |

比較の問いも分ける。(I) 同じJ/U class・開始点・実行recipeで目的の差を見る比較、(II) 同じ物理精度/実行政策で表現の違いを見る比較、(III) 各方式を適正に調整したend-to-end比較である。(I)だけで最強原法への優位は主張できず、(III)の自由度差を(I)の目的効果へ帰属しない。RankやLDの機械的一致だけではどれも保証しない。

## 8. 実装再利用マップ

### 8.1 現repositoryのAPI（すべて基点a67036a6版、呼出し未実施）

以下の短縮pathはすべて `src/trottertracks/` の下にある。関数を「使える」とするのは処理の再利用可能性で、旧固定runnerの許可範囲を拡大できるという意味ではない。

| SOURCE/API | 入力 → 出力・主parameter | 再利用と制約 |
|---|---|---|
| `c_pr_aware_df.py:coefficients/smooth_mass/make_value_and_grad` | J/spec、g/x → atom係数/目的/gradient。rank/norb/ld/scales/gamma/epsilon | 元numberのZ+ZZ mass。ZZ-onlyへ使うならmapを限定。資源値は出ない |
| `c_near_objective.py:Objective/mass/initial_spec` | signed J、ld/loss_scale/strength → weighted entry mass/spec | 既存ENTRY固定重み。wZZ指定APIなし |
| `c_near_stable.py:Coefficients/parts_gradient_J/trajectory` | g/J/U/spec/physical → stable inner/outer trajectory | Tensor map/adjoint・packing/U更新を再利用。weight変更は解析微分・metricにも反映が必要 |
| `c_rcdf_comparator.py:Objective/full_l2/fit` | g/J/U/x0、l2_scale/strength、callbacks → J/U/OptimizeResult | full ordered L2は既存。Dedicated process factory hook。原法nested solverではない |
| `c1_ffsim_corrected_codec.py`、`c1_real_orthogonal_codec.py` | real J/Uとprepared x0 → faithful parameter chart/fit | det−1・π branch対応。外部実装のlogmを盲目的に混在させない |
| `c1_general_k_adapter.py:from_ffsim_number_arrays` | corrected κ/J/U/c、明示number flag/frame IDs → GeneralKModel | 任意signed対称core。κを再補正しない。spin非対称UWC積分は直接入力不可 |
| `c1_molecule_extension.py:SectorCAR(4).one/raw_frame` | κまたはJ/U → sector36の独立CAR行列 | SCDF sector確認の部品。`rank_assessment`は元h/g/physicalとnorm+guard、shift補償を知らないので接続変更が必要 |
| `c_h4_relocated_downstream.py:prepare/raw_generators` | 保存κ/c/J/U、ld/source/input identity → relocation/独立sectorまたはfull generators | Linear-Z恒等変換を再利用。新fit・new shiftの実装ではない |
| 同 `prepare_candidate` | candidate/physical record/reference → own-ground/Arm packet | H4/CDF8/保存3候補/DF identity/元κの固定gate。汎用baseline importerではない |
| 同 `reference_r/native_context/native_cohort/strong_d` | qualified relocated Arm、r、split/batch/prepared context → PF/RTE/価格校正/D | 既存engineを再利用可能。新候補・新representationの旧価格を再利用できない |
| `c_integrated_retuning_policy.py:select_r`、`c_near_r32_statistics.py:primary` | selection rows/bindings、独立price rows → 選択/名目的差区間 | 候補identity・n・family・r集合を新比較へ整合させる。finalをselectionへ返さない |
| `c1_h4_strong_baseline.py:prepare_xdf_basis`、`c1_h4_b3.py` | κ/λ/G/cとbasis policy → XDF NativeModel/強D route | 保存XDFのexact spin/relative Gaussianを再利用。Cのnumber-J移植と係数規約は別照合 |
| `c1_h4_strong_ca_resources.py:load_bundle/attach_evidence/propagate_round` | 保存JSON・SOURCE/contract → D packet/round誤差 | B2/B3と旧SOURCEへ固定。新baselineを既存identityへ偽装しない |

対応runnerは `scripts/run_research_c_regularization_comparator.py`、`run_research_c_near_practical.py`、`check_research_c_h4_relocated_downstream.py`、`run_research_c_c1_h4_strong_ca_resources.py`。今回は起動していない。既定preflightでも研究scopeを変える効果はない。`tests/management/test_research_c_{near_objective,near_stable,regularization_comparator,h4_relocated_downstream,c1_general_k_adapter}.py` 等に関連人工検査資産があるが、今回再試験・新testは不要で実施していない。

### 8.2 公開実装の取得状態・版・license・依存

| 系列・公開先 | 固定版・取得範囲 | License/依存、研究Cへの適合性 |
|---|---|---|
| RC-DF著者 [Zenodo 7866658](https://zenodo.org/records/7866658) | API metadata取得。`rc-df-data.tar.gz` 665,070,260 bytes。先頭directory/図/loader notebook名まで部分inventory、最初の大きなpickle payload前で読取を終えた。archive全体・fitter有無は未確認 | RecordはCC-BY-4.0。保存factor/result配布であることは論文にも記載。完全optimizerがないとは断定しない。元分子結果をH4 baselineへ転用しない |
| RC-DF型L2 [NVIDIA/cudaq-algorithms][NG] | commit `4891dbe62c0baeac4569bf608562ffee59635cca`。`python/cudaq_algorithms/double_factorization/_factorization.py`、`_backend.py`、pyprojectを取得・読解 | Apache-2.0。Python≥3.11、cudaq≥0.16,<0.17、NumPy/SciPy、optional CuPy。g→`DoubleFactorization.leaf_cores/leaf_rotations`はJ/Uへ対応。`regularization=rho`、inner_solver=lstsq/cg、outer L-BFGSのfull L2。任意wZZ/L1² APIはない。独立環境でadapter再利用候補 |
| RC-DFを参照する [PennyLaneAI/pennylane][PG] | commit `cc7894c8acefe347db6bd8f96afc727d03346d8e`。`pennylane/qchem/factorization.py`、LICENSE、pyproject取得・読解 | Apache-2.0。当該HEADのPython≥3.12、NumPy/SciPy、optional JAX/Optax。`qchem.factorize(...compressed=True,regularization='L1'/'L2')`。下記のloss実装差により、そのまま論文weighted entrywise objectiveとして採用不可 |
| CDF基盤 [qiskit-community/ffsim][FG] | v0.0.83=`558d3517385778c733f1aa226bb61b8f30f4b0f3`、`python/ffsim/linalg/double_factorized_decomposition.py`取得・読解 | Apache-2.0。既存fit環境はNumPy1.26.4/SciPy1.14.1/JAX0.4.38。`double_factorized(g,tol,max_vecs,optimize,method,options,...)`はJ/U/任意OptimizeResult、ERI APIにregularization引数なし。t2 APIの同名引数をERI用と誤認しない |
| SCDF著者 [Zenodo 10836682](https://zenodo.org/records/10836682) | v1、`scdf.tar` 42,657,280 bytes、全33項目のtar inventoryと`SCDF_tensor_factorization/load_SCDF.py`を取得・読解。数値NPZはdecodeしていない | Record CC-BY-4.0、配布archive内に汎用fitter/依存lockなし。LoaderはNumPy/SciPy expmでWt_jax/alpha/Gtflat_jaxを読む形式。H10–H80/P450/FeMoco保存結果で、H4 fit APIではない。optimizer実装はこの配布物から再利用できない |
| UWC著者 [FujitsuResearch/early-ftqc-qpe-resource][WG] | commit `8a74108e308bd72ce1d8ea45aafac5ebb08bc32e`、§5.1の7ファイルを取得・読解 | 再配布・改変可の3条項形式＋明示patent非許諾のLICENSE.txt。単純にMITと表記しない。README tested: Python3.13.11、NumPy2.3.0/SciPy1.17.0/JAX **0.91（原文表記）**、ffsim0.0.69、h5py3.15.1等。requests importはREADME一覧外。lock/可用性・数値互換性は未検証 |

**公開APIの名称だけで原法忠実性を認定しない具体例**：取得したPennyLane `_compressed_cost_fn` は `norm(residual)**2 + prefactor * jnp.linalg.norm(Zs, ord=norm_order, axis=(1,2))[0]` を実行する。Matrix ord=1/2はentrywise L1/二乗Frobeniusと異なり、`[0]`は先頭frameを選ぶ。Greedy layer追加と併用されるため、任意frame tensor重みの全entry正則化とは一致しない。これは**静的式の相違**であり、実行テストでbugを確定したとの主張ではない。名称L1/L2を根拠に現ENTRYやRC-DF式17の代替にしない。[固定SOURCE](https://github.com/PennyLaneAI/pennylane/blob/cc7894c8acefe347db6bd8f96afc727d03346d8e/pennylane/qchem/factorization.py)

CUDA-Q側のJ/U出力も元κ/cを新しく自動生成してよい意味ではない。Input tensorのchemist軸順、column convention、対称coreのoff二重計数、SO/logm初期化、inner toleranceと終点精度をC側で照合する。SCDFとUWCはsector shiftを含むため、既存「元Hとfull512同値」の契約へそのまま挿入しない。すべて既存fit/eval環境へのinstall・package変更は0。

### 8.3 追加処理の負担と限界

| 案 | 既存コードでは足りない理由・必要な差分 | 科学条件/下流への影響・負担 |
|---|---|---|
| wZZのRC-DF型L1² | 固定ENTRY重みを任意指定できない。map/平滑化/scale/解析gradient/metricを一つの定義へ揃える | 一般J/U・物理系は保持可、目的は変わる。Fit以降は新候補固有に評価。最小拡張、時間/iterationsは未測定 |
| RC-DF原法L2 | Local RC2は共通joint solver。原法nestedの比較にはCUDA-Q等の入出力/初期化adapter | solver/強度政策は変わる。物理accuracyを合わせる科学計算は別許可。新optimizerの重複実装は不要 |
| 採用SCDF | 公開archiveにfitterがない。W/α/U、median更新、shift/scalar/sector referenceを追加 | 変数族とfull-Fock Hが変わる。Rank/精度選定、native/PF/RPEを新評価。数理・検証負担は大きいが所要時間は不明 |
| Pauli-UWC輸入 | 著者optimizerは再利用可。元H4入出力、spin別積分、signed Pauli IR、OO/BLISS reference、同じnative/RPE会計が不足 | 論文資源式をC資源値へ直接移せない。Adapterと科学検証が必要。DF版はさらに新loss/head policy/native設計 |
| 保存CDF8/XDF8の整合比較 | 固定runnerは旧SOURCE/routeにbindingされ、新representationを受けない | 保存J/U/λGを再利用、必要な新reference/再配置/価格だけ独立範囲で取得。旧raw再解析も新科学処理に含む |

H4の一般CDF8はU48＋独立対称J80=128変数という規模をSOURCEから数えられる。SCDFなら同8frameでW32/U48/α8にone-body shiftが加わるが、同rankでaccuracy適格になる保証はない。これは変数数の式による規模説明で、実fitの反復数・wall time・RAM予測ではない。公開大分子例の所要時間や現H4下流の予測を、新baselineの計算時間として流用しない。今回、工数を人日やCPU時間で見積もる測定根拠はない。

## 9. Baseline比較可能性判定表

判定は目的/APIと下流資源で分ける。`DIRECTLY_REUSABLE`も新しい科学実行許可を含まない。

| Baseline | 既存手法の本質 | 研究Cとの差 | 再利用できるSOURCE | 必要な追加処理 | 主判定・補足 | 検証できる問い |
|---|---|---|---|---|---|---|
| RC-DF | 一般J/U tensor fit＋係数正則化 | 全frame/L2主体、L1族・solverの別 | local RC2、tensor/codec/stable engine、CUDA-Q L2 | wZZ L1²のobjective/backend接続、物理精度とtuning整合 | **MINIMAL_EXTENSION_REQUIRED**（tail wZZ目的）。原法solver資源比較はCONTRACT_ALIGNMENT_REQUIRED | 再配置後に既存正則化族で残る差は何か、有限recipe/強度/solverの寄与か |
| SCDF | rank-one Wとα shiftを含む圧縮 | 一般signed Jから変数族とoff-sector Hが変わる | SectorCAR/general-K/native、著者loader | fitter、shift補償、独立full/sector reference | **NEW_SCIENTIFIC_IMPLEMENTATION_REQUIRED**＋CONTRACT_ALIGNMENT_REQUIRED | 固定sector shiftが精度を保ってPF/RTEを含む総費用を下げるか |
| UWC | Pauli soft head cost＋tail²、OO/BLISS | 可変Pauli境界、global積分変換、費用式 | 著者UWCコードを取得済み、Cの会計/検査部品 | H4 adapter/signed IR/native/RPE整合。DF版は新科学目的 | **CONTRACT_ALIGNMENT_REQUIRED**（Pauli原法）。DF版はNEW_SCIENTIFIC_IMPLEMENTATION_REQUIRED | 同精度・同費用指標でPauli-UWCに対するCDF固有の利益が残るか |
| CDF8+D | 保存一般CDF＋強D S2 | 旧J/U/旧split/旧error lane | B2保存結果、CDF export、relative builder | 新比較のpolicy/精度/価格資格を整合 | **DIRECTLY_REUSABLE**（歴史的参考）。新下流主比較はCONTRACT_ALIGNMENT_REQUIRED | PR効果よりDF/決定論実装で説明できるか |
| XDF8+D | first-passing保存prefix＋spin/relative強D | square表現、旧J/U/費用SOURCE | 元λ/G、B3保存結果とbuilder | 同上。PRへ使うなら新辞書と全下流 | **DIRECTLY_REUSABLE**（歴史的参考）。新下流主比較はCONTRACT_ALIGNMENT_REQUIRED | 最適化なしの既存強DFでどこまで競争できるか |

| 限定した問い | 判定 |
|---|---|
| 非平滑M_ZZ²がweighted L1²族に入るか | **DIRECTLY_REUSABLE**：静的係数対応で確定。性能/新規性認定ではない |
| Local full-L2の目的値/APIを再利用できるか | **DIRECTLY_REUSABLE**：既存RC2コード。新fitの結果は存在すると仮定しない |
| 著者SCDF optimizerを配布archiveから再利用できるか | 取得したarchiveにはなく**NEW_SCIENTIFIC_IMPLEMENTATION_REQUIRED**。未探索の非公開実装不存在までは主張しない |
| RC-DF著者archive全体にfitterがあるか | **EVIDENCE_INSUFFICIENT**：部分inventoryのみ。CUDA-Q等の確認済み候補と混同しない |
| 公開UWCのH4/DF/native互換性が実証済みか | **EVIDENCE_INSUFFICIENT**：静的読解のみ。公開コードの存在は確認済み |

## 10. 次に追加するbaseline候補（最大3案、未採択）

進行中結果を参照せず、反論への直接性・再利用・負担から次を提案する。

1. **再配置整合のRC-DF型weighted-L1²と既存full-L2の位置付けを先に固定する。** まず§3の厳密同値を利用し、名称だけ異なる重複fitを排除する。残るfull/tail・L1²/L2・原法/共通solverのうち、何を問うか選ぶ。必要ならwZZ backendの最小拡張後、同物理精度・公平tuningの独立候補を作る。肯定/否定のどちらでも、Cの差が正則化の既知族内の選択か、下流資源への追加価値かを判別できる。H6にも同じ係数mapを再利用可。**新fit・下流は要科学実行許可。**
2. **保存XDF8/CDF8＋強Dを新比較の会計に整合する。** 生成済み候補を再利用し、まず保存値は歴史的referenceに限定する。同じPF分割を強制する比較と各表現に適正な強Dの比較を分け、どこまで既存誤差/価格を再利用できるか選ぶ。最も小さい負担で「PRより強D/既存DFで十分か」を確認できるが、保存数値の新しい結合・再解析も自動実行しない。H6では別rank/候補の資格が必要。
3. **著者Pauli-UWCを利用する、精度・native指標を揃えた比較。** 著者optimizerの再実装は不要。Cに最も近いPR資源目的の反論を直接検証できるが、signed Pauli回路、shift/OO sector reference、同じRPE・native会計へのadapterが必要で、1/2より負担が大きい。DF版UWCを先に新設するより、原法の輸入可否を確かめる方が比較対象を明確にできる。**新科学実装・科学実行は未許可。**

SCDFは重要な対照だが、公開fitter不足と変数族/sector shift/精度調整の追加負担があるため、この最小3案の先行対象には置かなかった。これは有用性・性能の否定ではなく、今回の情報価値と再利用性からの優先案である。ユーザーがshift機構を主課題にするなら3と入替えを検討できる。

| 将来H4下流の状態（結果の予測ではない） | 推奨の変化 |
|---|---|
| 再配置後もPR-aware由来表現が有利 | 1で既知L1族・有限recipe差を切り分け、2で強D競争力を確認。残る利益を原法PR目的へ問う場合3の価値が高い |
| ENTRY由来表現が有利 | 1の重み/辞書対応と2を優先。旧45.47%を新辞書へ主張しない。3を自動追加せず、主張範囲を先に判断 |
| ほぼ同等 | 不検出を同等性証明とせず、事前精度・区間を確認。1の同値性と2で単純な既存方式による説明を優先。研究判断が変わる時だけ3 |
| 不適格・STOP/未成立 | 費用を0や敗北とせず、原因と資格を先に整理。1/2/3の新科学実行を自動採択しない。必要な技術修正は別依頼、科学再実行は別許可 |

## 11. 未確認事項・最終回答・停止状態

### 11.1 未確認事項

RC-DF著者archive全内容/optimizer、Claude独立レビュー原本、外部optimizerのH4実動作と到達精度、公開UWCのdependency表記/lock・truncation/境界条件の実行互換性は未確認。SCDF archiveのfitter欠如は確認したが、世界中の公開/非公開実装の不存在証明は行っていない。PennyLaneとUWCの静的相違は修正・実行テストしていない。これらを性能判断に補完しない。

進行中H4下流結果は未取得であり、本報告の条件分岐はその結果を先取りしない。新baselineで同じDF予算が達成できるか、PF/RTE/RPE/nativeの順位がどうなるかは未評価。全law誤差・actual CA・FT費用の未資格を、論文の一般理論や別SOURCEのPASSで埋めない。

### 11.2 Q1–Q12への回答

| 質問 | 回答 |
|---|---|
| Q1 数学的一致 | 任意signed JでM_ZZ=tailのordered diag1/4・off1/2 weighted entry L1。非平滑M_ZZ²はRC-DF本文のL1²族へ厳密帰着。全frame/L2/非二乗L1/一体項/平滑化まで同じとは限らない（§3） |
| Q2 設定変更だけか | 既存full-L2対照は既存APIを使える。wZZ L1²は固定ENTRY重みなので設定変更だけでは不可、objectiveとstable解析経路の最小拡張。原法solver再現は別（§3/8） |
| Q3 SCDF導入 | W/α/Uとshift補償・sector誤差を定義し、full512はshift後referenceへ、元H同値はtarget sectorへ別接続。採用rank-one SCDFと一般J+shiftを区別（§4） |
| Q4 UWCとの差 | head cost＋可変境界＋OO/BLISSのPauli表現最適化。Cは固定frame maskの近似DF fit。既存Cへそのまま適用不可。著者コードは再利用可、DF版は新設計（§5） |
| Q5 公開実装 | RC-DF型L2 CUDA-Q/基盤ffsimを取得、PennyLaneはloss相違あり。SCDFは全archiveの結果/loaderのみ。UWC著者コードは固定commitで取得・読解済み（§8） |
| Q6 保存DFの公平範囲 | 同物理系・共通予算下の歴史的強D参考値。新再配置comparison/cohortではない。旧SOURCE/近似H/誤差lane/builderを整合しない直接資源優位は認定不可（§6） |
| Q7 揃えるもの | 同物理精度・費用/誤差資格を必須にし、共通policyと公平tuned policyを分離。同rankは目的ablation向け、異表現にrank/LDを機械固定しない（§7） |
| Q8 SOURCE再利用 | tensor/codec/J/U/stable/sector/general-K/relocation/PF/RPE/native/統計の部品は再利用可。新weight、shift/fitter、Pauli/native bridge、DF-UWC目的が追加対象（§8） |
| Q9 最小で高情報価値 | 1のRC-DF型目的対応が先。同一目的なら重複fitを不要とし、残る差と2の強Dを検証候補にする（§10） |
| Q10 結果別の推奨 | PR優位なら1→2→必要時3、ENTRY優位/ほぼ同等なら1/2を優先、未成立なら資格/原因を先に整理。実結果を読まず条件分岐のみ提示（§10） |
| Q11 非干渉 | 作業用worktreeと専用/tmp資料だけへ書込み、科学process停止/再起動・SOURCE/raw/契約/STOP・共有設定変更・既存環境installは0。外部worktreeの通常更新を凍結していない |
| Q12 次の担当 | **ユーザー/GPT**が比較する問いと優先baselineを選ぶ。**Codex**は別の具体的な実装依頼後に必要最小限のadapterを準備する。**ユーザー**の別科学実行許可後だけfit/再解析/PF/RPE/nativeへ進める。今回は判断を戻して終了 |

### 11.3 実施範囲とMANDATORY STOP

本報告の検証は、出典/API/式の静的照合、Markdownの参照・差分確認、commit/push後のremote SHA・tree・公開本文の照合である。科学testを走らせたとの意味ではない。実SCF/Hamiltonian生成/DF fit/固有値/PF/RTE/RPE/state preparation/回路build/compile/transpile/sampling、保存数値の新再解析・再集計・再ランキングはすべて0。Scientific SOURCEや新実行契約は作っていない。

公開照合のSHA/tree/本文一致の実績は最終チャットに記載する。**公開照合後MANDATORY STOP**。Baseline最適化、H2/H6/H4追加評価、正式新規性判定、論文化GO/NO-GO、重要GPT科学レビューへ自動進行しない。次は§11.2 Q12のユーザー/GPTの判断であり、新計算の承認証明書・grant発行を求めるものではない。

[H]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md
[Z]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d601e0ef4375c37185211aa75685ae0e582f3e64/docs/research/research_c_h4_relocation_execution_20261106.md
[P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a67036a6baafc00deaa38a1b0b801d313d519ad5/docs/research/research_c_h4_relocated_downstream_preparation_20261106.md
[PC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a67036a6baafc00deaa38a1b0b801d313d519ad5/docs/research/research_c_h4_relocated_downstream_contract_unapproved_20261106.json
[L]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1a588f5491bce394ecf039ee4d2de57581e2e0b1/docs/research/research_c_related_work_novelty_evidence_20261105.md
[E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a53d0569b70e4417765c4876616260db911e603/docs/research/research_c_h4_100_paper_evidence_20261106.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md
[R]: https://arxiv.org/html/2212.07957v3
[S]: https://arxiv.org/html/2403.03502v1
[W]: https://arxiv.org/html/2603.22778v2
[T]: https://arxiv.org/pdf/2503.05647v2
[C]: https://arxiv.org/html/2104.08957v1
[WG]: https://github.com/FujitsuResearch/early-ftqc-qpe-resource/tree/8a74108e308bd72ce1d8ea45aafac5ebb08bc32e
[NG]: https://github.com/NVIDIA/cudaq-algorithms/tree/4891dbe62c0baeac4569bf608562ffee59635cca
[PG]: https://github.com/PennyLaneAI/pennylane/tree/cc7894c8acefe347db6bd8f96afc727d03346d8e
[FG]: https://github.com/qiskit-community/ffsim/tree/558d3517385778c733f1aa226bb61b8f30f4b0f3
