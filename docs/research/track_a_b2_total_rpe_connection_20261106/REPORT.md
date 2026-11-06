# Track A・B2から同精度・同成功確率の総RPE費用への接続可能性

2026-11-06 JST / SERVER 2 / Codex主エージェント。**既存文書・SOURCE・保存結果の読取調査。新科学実行・数値再解析・詳細科学レビューなし。**

## 結論

**接続する実装資産はそろっており、B2の再探索や研究全体の作り直しは不要である。ただし、既存B2の費用改善を、そのまま元の物理Hamiltonianの同一energy精度・成功確率での総RPE改善とすることはできない。** 最小の不足は、元Hと状態の結合、各roundの実信号・半径、共通energy/branch/失敗配分、同条件の強PF価格である。

旧PR-2契約は明示的に**rank12 DF Hamiltonianをtargetとし、元の分子Hamiltonianをtargetにしていない**。SOURCEもDF Hamiltonianのsector ground stateを生成・保存している。後続tol-only DF10の入力生成もDF Hamiltonianを対角化している。従って、今回の「元Hの厳密基底状態を仮定・準備費用除外」は維持するが、既存B2のstateをその状態と認定しない。[旧科学契約][PR2CONTRACT]、[旧入力生成SOURCE][OLDSTATE]、[DF10入力生成SOURCE][TOLINPUT]

最初の代表は **H4 linear 1 Å/STO-3G、旧DF12、generation-prefix L_D=3、二次B2、r4/K2** を提案する。既存候補IDは `B2-rank3-q1-r4-K2`。選定理由は来歴とコード接続の明瞭さであり、優位性の期待ではない。RPEへ時間条件を拡張した結果は新しい検証として扱う。全RPEの比較条件は未採択であり、本報告は実行契約・実行許可ではない。

目標会計は、指標ごとに

\[
G_j=\sum_{m=0}^{M}\sum_{b\in\{X,Y\}}
N_{j,m,b}\,\mathbb E[C_{j,m,b}^{\mathrm{full,no\text{-}prep}}]
\]

である。X=cosine、Y=sine。古典価格MC標本数を量子shot Nへ重ねて掛けない。RZ、CX、各depth、sizeは別指標とし、depth加重和を量子hardwareのwall timeと呼ばない。

## 1. 調査範囲と固定資料

| 固定commit | 今回の用途・限界 |
|---|---|
| `731b3d6f3572c6aa7950a579b68251fec9a5218b` | [Track A研究データ索引][INDEX]。比較可能性、cohort、SOURCEへの入口。索引作成は独立実験ではない |
| `3571477093441ac4852166fe829f8ccc5db5ec4f` | [3環境統合][INTEGRATION]。旧結果・後続更新・再利用地図を区別。本報告branchのbase |
| `50ee55bc30f30838e1ac0704bc94f720914640de` | [O1 M1固定C0総費用][M1C0]。全7round・実半径・full no-prep価格会計の実行例 |
| `0daf01d51c022977a1c0a7d29dde34a1c4c82d7c` | [O1 M1 r8価格pilot][M1R8]。信号減衰と価格交換の限定負結果。終了候補・STOPは維持 |

PROJECT_MAP、研究概要、正式共通運用指示、関連の検証文書とSOURCEを読んだ。固定SOURCEはGit objectから静的に確認し、下記の保存JSONは必要なfield・構造を読取確認した。配列NPZのload、保存信号からのphase/radius再算出、raw展開、再集計・再順位付け、runner/testの起動は行っていない。数値は既存報告の引用である。

命名の衝突に注意する。旧PR-2のM1/M2はdevelopment/geometry transferであり、11/6のO1 M1/M2とは別研究である。またH8の「Phase B2」は決定論S4価格モデルの検証段階名であり、PR候補method B2そのものの検証ではない。[H8 Phase B2報告][H8PHASE]

## 2. ① 既に検証済みの事項

### 2.1 B2の科学条件と既存結果

| 項目 | 旧PR-2／matched-accuracy B2で確認した条件 |
|---|---|
| Hamiltonian | H4直線鎖、隣接1.00 Å、STO-3G、8 system qubits、N=4/Nα=Nβ=2の36次元sector。`build_df_h_d_from_molecule(...,df_rank=12)`の保存DF12。tol-only DF10と同一ではない |
| 入力identity | development NPZ SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。旧pilotの別hashと混ぜない |
| 分割 | B2のcandidate名のrank3は **L_D=3** で、Hamiltonian自体のDF rank3化ではない。one-bodyと生成順prefixを決定論、残りをcanonical finite-RTE。V1–V3ではL_D3/6/9のgeneration/weight順が一致し、B2-G/Wを重複候補として統合 |
| scalar/identity | 核scalarを含む保存constantとtailから抽出したidentityを明示的に保持。`extract_identity_phase`、coefficient_atol=0。random tailは抽出後の非identity成分。controlledでは相対位相として一度だけ計上 |
| PFと時間 | 二次partial-S2。forward deterministic half sweep → finite tail → reverse half sweep。T=.8、旧q=1/2/4/8、δ=T/q=.8/.4/.2/.1。qはこの固定T内のPF outer step数 |
| finite RTE | base r=1/2/4/8/16/32、K=2/4、一部の事前規則r64候補。short stepはδ/r、τ=λ_R δ/r。SOURCEの補正平均はdegree K+1の有限多項式。normalizationは実lawのb_K(τ)の全qr個の積 |
| 状態・参照 | DF12 Hamiltonianの保存sector ground state。`_target_signal`はDF HへのRayleigh residualを検査し、exp(-i E_DF T)を参照。元分子Hとのstate一致・energy誤差の証明ではない |
| 精度・成功 | ε_complex=.05、各軸ε/√2、α_axis=.025の十分Hoeffding shots。両軸signal推定の条件付き失敗配分であり、energy誤差や全round RPE成功率ではない |
| 費用 | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし。測定付きHadamard full wrapper、準備費用除外。主RZのshot加重費用と、CX/count/depth/sizeの計6指標 |

[M1-A結果][M1A]、[M1-B1結果][B2DEV]、[分割・入力検証][V123]、[signal SOURCE][B2SIGNAL]、[資源契約][MATCHED]

旧M1-B1はSOURCE `33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62`、結果 `8e0814e70c14ecf526444fac8a2142799610dc96`。206適格候補の6指標point ParetoにL_D3/q1/r4/K2とr8/K2が残った。r4の保存RZ workは `130,774,896.656`、analytic shotsは20,563。ただし32価格trajectoryの点推定で、r/Kの厳密winner・母平均優位ではない。[M1-B1結果][B2DEV]、[一次compile map][RAWDEV]

PM-1は近接discard L_D4/5を追加した固定時間比較、PM-2は保存値によるε=.005〜.1の302表示点の事後感度解析である。PM-2の厳しい側ではB2 q4/K4、次にq2/K4、緩い側ではq1/K2という保存点最小が報告されている。これはenergy精度の掃引でも302回の独立実験でもない。今回は再解析していない。[PM証拠対応][PMMAP]

### 2.2 correctness、移送、強baselineの証拠範囲

| 既存資産 | 再利用できる証拠 | 制限 |
|---|---|---|
| V1–V3・M1 signal | DF blockとtail再構成、符号、prefix、sector/state、raw/correctedとlog/direct normalizationの整合 | local検証。同じ保存入力・一部共有実装に依存。元Hへの一致ではない |
| 旧M2 H4 1.30 Å | SOURCE `2978e2fea672b7a1ff20cac74269ec9a610159dc`。結果前固定5構成、B2 r4/r8を含む全5件適格、`TRANSFER_SUPPORTED` | 元ε_complex=.05/T=.8での固定構成移送。held-outでの再最適化・全RPE・他サイズ保証ではない。[移送報告][B2TRANSFER] |
| AX-2B H4 v5＋補完 | full36列のsector作用、native/dense、±時間のprimitive、selected controlled branch/X/Y、独立occupation構成、S4 MP80/120参照の一致。full no-prep measured wrapperとglobal S4費用 | v5のB2はL_D6/q4/R8/r2/K2,4でありL_D3/r4の直接証明ではない。event order0/2の選択probe、共有basis/lowering、総数値allowance未認定を保持。全law/operator認証ではない。[v5][AXV5]、[補完][AXSUP] |
| 後続H4 tol-only DF10 P | L_D3/q1/r4/K2（ε_complex=.05、n1108）とL_D2/q8/r2/K4（.0005、n78）の独立価格確認。native/control/event oracleの限定検査 | 同SOURCEの独立価格cohortであり、別実装での全correctness再現ではない。保存stateはDF10のeigh由来。粗精度はPF低費用 `NOT_REPRODUCED`、細精度はPR低費用 `REPRODUCED`。旧DF12の成果へ合算しない。[DF10 P][TOLP] |
| H6/H8 tol-only系列 | DF19/28、既存候補・固定endpointのnative/shot加重RZ比較と保存移送結果 | H4 L_D3固定候補の同一条件移送ではない。H6粗精度のREPRODUCEDはPF低費用方向、H8は公開詳細監査の不足あり。母平均・全RPE・系サイズscalingへ一般化しない。[統合主表][H468] |
| 強決定論対照 | AXのglobal S4、後続COMMON588のS2/Yoshida4/m5_best・Full/B0/B0I、M1のrelative-fusion/spin-aware StrongBuilder | 旧B1 S2だけでは最適化十分といえない。COMMON588もq≤8・固定Tの有限探索域。M1のsubsteps4もそのC0だけの資格である。[共通PF採用][PFADOPT]、[strong SOURCE][STRONG] |

### 2.3 元Hとの対応について確定できること

旧DF12のstate生成は `solve_df_ground_state(hamiltonian,sector,...)`、DF10は生成したDF sector行列の `eigh(...).vectors[:,0]` である。固有状態残差が小さいことは、この**DF Hの固有状態**である証拠であり、未切断分子Hの厳密基底状態である証拠にはならない。元のAO/MO integrals、DF approximation、Hermitian projection、核scalar、state/orderの来歴を結合する必要がある。[旧state生成][OLDSTATE]、[DF10 state生成][TOLINPUT]

一方、O1 M1のH4-A loaderは元H診断用の`state`を明示的に読み、`DF_state`へfallbackしない。M1固定C0はそれを厳密基底状態と**仮定**し、DF/model誤差を別計上している。この設計と検査経路は再利用できるが、inputは別NPZであり、B2のstateやDF factorsとの同一性を認定する根拠にはならない。[M1 adapter SOURCE][M1ADAPTER]

同じ分子名・energy表示値・配列寸法だけで結合しない。元H厳密基底状態は今回の研究仮定なので新たな基底状態証明を必須化しないが、**どの元Hのどのstateを使用するか**は固定し、DF stateに黙って置換しない。

M1 C0ではPauli/DF PRとも、単回価格が低くても長roundの減衰・必要shot増加により、保存条件付き総CX平均は強DF-PFを上回った。r8限定pilotも必要単価を上回り固定候補の終了が提案された。この負結果は会計接続の参考であり、prefix付きB2の負結果でも一般no-goでもない。[M1 C0][M1C0]、[M1 r8][M1R8]

## 3. ② 不足事項：RPE接続の8要素

区分：**接続**＝既存APIの結合・最小adapterで対応可能、**数値**＝別承認後の新検証が必要、**判断**＝科学条件・主張の採択が必要。

| 要素 | 再利用する実装・既存証拠 | 今回のB2に残る不足 |
|---|---|---|
| 1 各round PF誤差 | `_apply_outer_step`、`_preparation_eigensystems`、`finite_rte_signal_validation`、既存round scheduleの構造 | **数値**：同じB2分割を元H stateへ作用させ、全使用時刻でDF exact→ideal-tail PFの複素差・位相差・半径を検査。旧C_useは経験包絡で、snapshotも異なる。q倍の無条件外挿は禁止 |
| 2 finite-RTE/normalization | `make_rte_config`、`finite_rte_operator_moments`、`finite_rte_attenuation`、旧B2有限多項式、M1独立paired-polynomial参照 | **接続＋数値**：prefixを保持した同じcanonical lawを両経路へ渡す。各roundの全Q_m r_m short stepのb_K積・raw/correctedを確認。M1 LD0のλ/Bを流用しない |
| 3 元H stateの複素信号・実半径 | M1のoriginal/DF/PF/finite/implementation分解、`observed_signal_bounds`/`account_radius_round`の会計仕様 | **判断＋数値**：元H/stateの対応を先に固定。PF有効固有状態や単位半径へ置換せず、各roundで実信号と半径を評価。旧B2のDF-state信号は入力にできない |
| 4 energy/branch/success/失敗配分 | `RPERoundSpecification`、`RPEErrorAllocation`、`required_rpe_round_horizon`、`reconstruct_rpe_phase`、既存V1/V2の条件付き18round接続 | **判断＋数値**：ε_E、energy window、符号、scalar、全round系統誤差とΣα、初期aliasとnearest branchの条件を同時固定。旧ε_complexやα_axis=.025を全roundへ複製しない |
| 5 X/Y必要shots | `account_radius_round`、`finite_rte_signal_validation._hoeffding_shots`。旧`corrected_hoeffding_shots`/`u_aware_shots`は固定時間taskの対照 | **接続＋数値**：実半径・finite error disk・統計位相予算・軸別αから新N。単位PF半径、Bだけ、旧Nのコピーは不可。量子測定samplingは解析会計の必須条件ではない |
| 6 full no-prep回路価格 | `partial_native_from_step_requests`、`build_native_hadamard_wrapper`、既存DF event/repeated builder、`transpile_and_measure_cost` | **接続＋数値**：同じprefix/event列/scalar/controlで全使用roundのX/Yをwhole compile。中央RTE価格や単一time-evolution価格をwrapper価格としない |
| 7 強決定論PF | `prepare_xdf_basis`、`StrongBuilder.sequence`、native S2/S4、既存3公式・Full/B0/B0I登録域 | **判断＋数値**：同じ元H state・DFモデル誤差・ε_E・成功率・compiler・no-prep scopeで対照も適格化。PRだけr/Kを調整し対照のstepを固定する比較をbest-to-bestとしない |
| 8 全round費用・価格不確かさ | `aggregate_registered_rounds`、既存paired metric accumulator。M1全7roundの実行例 | **接続＋数値／主張次第で判断**：全required roundの欠測なしを確認しN×E[C]を指標別集計。X/Y共分散、round間共有、価格標本誤差、未観測希少eventを区別。M1もsample SEのみ・総価格CIなし・`not_certified` |

旧round scheduleは再利用できる**方法**である。旧RPE系列のH hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` はB2 developmentの `de7a5492…` と異なる。後続V2には18roundの `CONDITIONAL_SUCCESS` が実在するため「長RPEの証拠が全くない」とはしない。同時に、そのN・半径・分枝成功率・CA/10 scheduleをB2へ移せない。[RPE技術調査][RPEBRIDGE]、[既存V2][RPEV2]

信号減衰による利益相殺は評価可能である。必要な入力は各roundのPF参照半径、corrected有限誤差、実lawのattenuation、両軸の全wrapper価格である。M1の会計は `rho_observed_lower = A*(rho_PF_lower - epsilon_corrected)` を使う。既に減衰したraw半径へ再度Aを掛けたり、raw信号に補正用Bを掛けた量を±1測定平均としたりしない。各armのNで総費用を比較する。今回、新しい相殺率・shot・総費用は計算していない。[M1会計SOURCE][M1BRIDGE]

## 4. ③ 最小接続案（実装は次段階）

| 再利用元・版 | 具体的な再利用点 | 最小差分案 |
|---|---|---|
| `2978e2f…` [旧B2 signal][B2SIGNAL]：`_prepare`、`_dense_block_operators`、`_preparation_eigensystems`、`_apply_outer_step`、`_random_signal_record` | generation-prefix、exact identity抽出、二次sweep、canonical finite polynomial、PF/finite分離 | 元H stateとoriginal/DF exact referenceを明示引数で受ける薄い新adapter。`_target_signal`のDF固有状態gateを改変して元H stateに通用したことにしない。元コード・旧結果は変更しない |
| `31cdff4…` [native DF SOURCE][NATIVE]：`partial_native_from_step_requests`、`build_partial_native`、`build_native_hadamard_wrapper`。`2978e2f…` の `make_df_partial_s2_repeated_request`/`QiskitDFPartialS2RepeatedCircuitBuilder` | 明示event列と決定論half sweep、scalar、control、X/Y wrapperの構築 | 既存preparationとround別step requestsを直接渡す。event sampler、Gaussian lowering、wrapperを新規実装しない。ordinary/directionalは検証済みnative APIで扱い、control policyとancilla相対信号の一致を確認 |
| `8a13539…` [M1会計][M1BRIDGE]：`account_radius_round`、`aggregate_registered_rounds`、`_compile_pair`/価格統計 | state/half sweepに依存しない実半径会計、欠測拒否、指標別価格、条件付きラベル | cost recordのcontext/arm/lawを新B2へ結合。`observed_signal_bounds`はM1専用schemaなのでB2 error分解を明示翻訳。数値allowanceを明記し、観測差を厳密上界へ昇格させない |
| 同M1の `prepare_matched_resources`/`build_controlled_trajectory` | 実装設計と独立検証の参照 | **直接適用不可**：固定`T_eff/V_square`・LD0・faithful tail identity。これをB2準備として偽装せず、上記prefix builderを会計へ接続する。M1 `load_h4_a_snapshot`のrank10 gateを単に緩めて旧DF12を読む修正もしない |
| `8a13539…` [StrongBuilder][STRONG]：`prepare_xdf_basis`、`StrongBuilder.sequence` | 全signed DF factors、same-basis合算、relative Gaussian fusion、analytic vacuum phase、exact spin lift | 旧B2と同じ保存factorsを引数として渡す。M1の保存baseline価格/substeps4は使わない。spin構造が適合する場合だけSPINを用い、phaseを無視したoperator alignmentで検査しない |
| 既存RPE [schedule][SCHEDULE]、[resource accounting][RESOURCE]、[branch復元][DECODER] | error/failure配分、horizon、roundごとの候補、energy/aliasに必要な構造 | 旧schedule runnerをそのまま起動しない。元HとB2の信号を入力する新driverで固定候補だけ検査し、選定・最適化は別範囲とする |

再利用するtestは、旧`test_pr2_matched_accuracy_m1_execution.py`、`test_df_partial_s2_repeated.py`、`tests/tracks/resource_applicability/test_ax2b_h4_supplement_v1.py`、M1 `test_o1_m1_resource_bridge.py`/`test_o1_m1_circuit_signals.py`/`test_o1_m1_parallel_price.py`、RPE `test_rpe_resource_accounting.py`/`test_rpe_four_round_phase_validation.py`。今回は静的な所在・対象確認だけで実行していない。次段階では既存のsynthetic恒等式とcontext拒否を再利用し、追加testはprefix非零の接続・scalar一回・raw/corrected半径の区別を中心にする。

新規framework、承認証明・grant発行システム、全環境固定、旧結果の再計算は不要である。SOURCE/input/compiler/実際の回路意味の来歴と、既存研究のone-shot/STOP条件は保つ。並列価格providerは再利用候補だがM1専用preparationを前提とするため、先にB2 serial builderとの意味・paired統計の適合を確認する。

## 5. ④ 最初の代表条件と最小科学検証案

### 5.1 代表条件の提案

**第一候補：旧DF12 developmentのL_D3・二次B2・r4/K2。** 元の固定時間anchorはT=.8/q1である。r8は既存Paretoの感度候補として記載するが、最初の検証へ自動追加しない。

選定理由は、入力hashとSOURCEが明示され、M1-A/B1→旧M2→PMまでの結果が追跡可能で、prefix/finite law/全wrapperを同じ実装から構成できるため。AXの独立occupation/MP/native検査を部品単位で再利用でき、全DF対照も同じfactorsで構成できる。ただしL_D3/r4そのもののAX全coverageと、元H stateとの対応は新検証が必要である。

**補助候補：H4 tol-only DF10 `B2_p3_q1_2nd_r4_K2`。** 独立価格n1108、native control/event oracle、より強いPF登録域への接続が利点。既存粗精度でPF低費用という結果も保持できる。一方、旧DF12からの同一入力移送でなく、元H stateへの不足は残る。M1 H4-Aと同じrank10という理由だけで結合しない。今回は両候補を再順位付けしていない。[TOLP][TOLP]、[PFADOPT][PFADOPT]

### 5.2 既存の精度・成功条件の違い

| 条件 | 意味 | 本提案での扱い |
|---|---|---|
| 旧B2 ε_complex=.05、両軸α=.025 | 一つのT=.8の補正signal task | 既存成果の引用条件として保持。energy精度へ同名転用しない |
| PM-2 ε_complex=.005〜.1 | 同じ保存値の事後感度 | 新しいenergy目標の根拠としない |
| 後続共通PF ε_complex=.05/.01/.005/.001/.0005 | tol-only・有限Tの登録比較 | 旧DF12と混合せず、比較domain設計の参照 |
| 既存RPE暫定CA/10 `.00015936001019904 Ha`、α_total=.05、β_RPE=.4 | legacy DF12 RPE系列のenergy目標。δ=.02などで後続条件付き証拠あり | **最初のenergy検証の提案値**。既存目標の再利用案であり新たな正式採択ではない。B2側のhorizon/N/scheduleは未算出 |
| O1 M1 C0 ε_E=.1 Ha、成功目標95%、β_RPE=.30、δ=.05、m0..6 | H4-A/DF10、LD0、別split・別law | 会計の実行例だけ。B2のenergy目標・round・N・価格に転用しない |

RPE時刻の案は、既存RPE系列の **δ=.02、Q_m=2^m、r_m=4/K_m=2固定** から適格性を調べるものとする。これは旧候補のT=.8/q1とは異なる科学条件であり、旧winnerの再現とは呼ばない。旧系列のm0..17というhorizonや最適r/K配列はコピーしない。元H/state・energy window・予算を固定後、既存horizon式で必要Mを確定する。δ=.01等の追加候補やround別最適化は今回の最小pilotに含めない。

### 5.3 別承認を受ける最小実行範囲（まだ開始しない）

1. **入力とtargetを結合する。** 元物理Hのintegrals/核scalar/規約、DF12 factors、元Hの保存stateの来歴を指定する。旧B2にはDF stateしか確認できていないため、元H stateを既存資産から同定できなければSTOP。新たなstate生成・solveが必要なら、その具体的範囲を含む別承認まで進めない。無料の元H厳密基底状態という仮定をDF stateで代用しない。
2. **SOURCEと条件を固定する。** 上記旧B2/AX/M1/StrongBuilderの固定版から最小adapterを準備し、新SOURCEを固定。候補はL_D3/r4/K2一つ、比較は同じDF factorsの強Full S2と既存global Yoshida4。対照の既存step候補1/2/4/8を同じenergy/成功条件で検査する有限pilotとし、連続最適化・m5追加・新prefix探索は含めない。全PFに対する最良比較とは呼ばない。
3. **全required roundの信号検証を先行する。** 元H exact→DF exact→ideal-tail PF→finite corrected/raw→実装を分離。元H stateに対し半径・phase budget・alias/branch・Σαを確認し、それぞれのarmのX/Y必要Nを求める。独立occupation/CAR・有限多項式参照とselected order0/2のcontrolled/native probeを再利用。synthetic PASSと分子correctnessは別に記録する。
4. **適格なarmだけ価格取得する。** 同じQiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし、準備除外の全X/Y wrapperを直接compileする案。PRは各round32 fresh IID価格trajectoryを初期固定案とし、旧32標本を再使用・poolしない。X/Yは同じevent列を共有、各wrapperをwhole compile。対照も同等のrelative fusion・制御・scalar条件で評価する。MC個数・seed・required round・compiler依存版を含む実行範囲は承認前に具体化し、この案だけでlaunchしない。
5. **同じtargetで全roundを集計する。** 指標ごとにN×E[C]、paired sample SE、未観測層、直接価格とモデル価格の区別を報告。最初は直接価格・条件付き平均比較を目的とし、総価格CI・母平均優越・認証済みenergy成功保証を自動的に主張しない。予算不適合ならr/K再探索せずSTOPして結果を報告する。

**GO条件：** 同じ元H/stateへの結合が明示され、各armの全使用roundでPF/finite/model/implementation誤差・正半径・branch/energy・全失敗予算が、事前採択した条件内。価格が全round/両軸/同一scopeで揃うこと。GOは条件付き費用を算出可能という意味で、B2の利益を意味しない。

**STOP条件：** 元H/stateの未結合、必要入力不在、誤差・半径・branch不適合、control/scalar不整合、非有限normalization/N、対照を同条件に適格化できない、価格欠測、または事前に定める価格精度条件不達。失敗を別候補・単位半径・旧価格で補完しない。有限MCで改善を支持しなくても、全lawの不可能性とはしない。科学実行・保存値再解析・sampling/compileの承認範囲を固定してから着手する。

## 6. ⑤ GPT詳細科学レビューの必要性

**既存APIの技術接続、SOURCE追跡、synthetic検査、条件付き平均費用pilotの準備だけなら、詳細レビューは不要。** 既存RPE V2やAXの解決済み論点を再レビューする必要もない。

ただし、今回求める**元の物理Hの同一energy精度・成功確率を保証した比較**まで主張する前には、次の限定的な科学判断が残る。これは既存B2で閉じていないため、必要な入力・前提を具体化した後の詳細レビュー対象とする。

| 未解決の判断 | 既存証拠で閉じない理由 |
|---|---|
| DF-groundをtargetとした旧B2から、元H groundに対するenergy保証へ接続する条件 | 旧契約は元分子Hを明示的に除外。DF/model/projection誤差と状態の違いを、同一energy予算へどう位置づけるかが必要。小Rayleigh residualだけでは閉じない |
| 元H stateがPF固有状態でない場合の全round共通位相・branch・成功条件 | 固定時間axis accuracyからは導かれない。既存18round V2は別snapshot・DF target・保存平均IIDモデルの `CONDITIONAL_SUCCESS`。O1 M1も `guarantee_status=not_certified` |
| 数値診断・許容誤差から、どの強さのenergy/成功claimを許すか | 既存のobserved error＋allowance、sample SE、経験C_useは厳密証明ではない。条件付きpilotと保証付き比較の区別を採択する必要がある |

レビューを開くなら、上の**target/state/model誤差と保証の接続に限定したLevel 2（共通運用指示上はGPT-6 xhigh推奨）**を想定する。新定理・研究方向変更を要求する材料は今回確認していない。モデル推奨は自動変更や開始許可ではない。ユーザーが条件付き数値pilotだけを選び、既存の十分条件をそのまま適用できると具体化できた場合は、追加の詳細レビューを必須化しない。

**今回はレビュー要否・論点の特定まで。詳細科学レビューは開始していない。**

## 7. 公開・検査範囲・mandatory STOP

成果物はこのREPORT一つ。専用branch `codex/track-a-b2-total-rpe-survey-20261106`、専用worktreeで作成し、元SOURCE・input・raw・result・grant・ledger・marker・STOPを編集しない。M1終了候補、O1 M2保留、旧M2 transfer状態を変更しない。旧文書の過去の「未実行」は後続結果と区別して引用した。

公開前の検査は文書内固定commit/pathと関数の所在、リンク、差分範囲、引用と条件の整合に限定する。GitHub connectorで指定4固定資料のREPORTと、B2 signal/DF10入力/M1会計の3 SOURCEを取得し、7件のblobがローカル固定Git objectと一致した。主要一次JSONではM1-B1のSOURCE/status、V1–V3のstatus、旧M2のSOURCE/input identity/STOP、M1 C0のstatus/STOPを読取確認した。保存統計の新集計はしていない。科学testを実行した・別担当が独立レビューしたとは表示しない。公開後にremote SHAとREPORT本文一致を確認し、最終チャットでSHAを報告する。自己SHAを埋めるための再commitや追加証拠台帳は作らない。

公開経路：ローカルGitのauthor未設定でcommitが成立せず、通常git pushもHTTPS認証未設定で失敗したため、接続済みGitHub連携で専用branchとREPORTのcommitを作成する。共有Git設定や認証設定は変更せず、公開commitをローカルworktreeにも取り込んで同一SHA・tree・本文を確認する。

**mandatory STOP：REPORTをcommit/pushし、remote SHA・公開内容一致を確認したら調査を終了する。新科学計算、再解析・再ランキング、回路build/compile、sampling、shot/総費用算出、最適化、実装変更、詳細科学レビューへ自動移行しない。**

次の担当は**ユーザー**。必要な場合だけ、第一候補・既存energy目標の採否、元H/state入力、条件付きpilotか保証付き比較かを判断する。その後**Codex**が対象SOURCE・入力・予算・round・価格標本を具体化し、ユーザーの別途科学実行承認後に実施する。詳細科学レビューを行う場合もユーザーの別途開始承認が必要。現在実行中の後続科学作業はない。

[INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/731b3d6f3572c6aa7950a579b68251fec9a5218b/docs/data_collection/2026-10-24_track_a_research_data/REPORT.md
[INTEGRATION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md
[PR2CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr2_primary_research_contract.md
[OLDSTATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/pr2_s0_s1_validation.py
[MATCHED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr2_matched_accuracy_resource_contract_v1.md
[M1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/pr2_matched_accuracy_m1_a_validation.md
[B2DEV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e0814e70c14ecf526444fac8a2142799610dc96/docs/pr2_matched_accuracy_m1_b1_result_validation.md
[RAWDEV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e0814e70c14ecf526444fac8a2142799610dc96/artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json
[V123]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr2_v1_v3_result_packet_ef86868_20260928.md
[B2SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/pr2_matched_accuracy_m1_execution.py
[B2TRANSFER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/pr2_matched_accuracy_m2_transfer_result_validation.md
[PMMAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/track_a_post_pm2_claim_evidence_map.md
[AXV5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/track_a_ax2b_h4_pilot_v5_result.md
[AXSUP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1c2ad5a2c0d704e4b4758fd4c07672d9f3fb3be8/docs/research/track_a_ax2b_h4_supplement_execution_v1.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[TOLINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84db3909639b34ec521004c3d8f7ef9b5e6ad851/scripts/h4_tol_only_input.py
[TOLP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/docs/research/track_a_h4_tol_p_completed_20261019.md
[H468]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/track_a_h468_paper_evidence_integration_20261019.md
[H8PHASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/track_a_h8_b2_completed_results_20261017.md
[PFADOPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/track_a_h246_pf_common_adoption_20261022.md
[M1C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
[M1R8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0daf01d51c022977a1c0a7d29dde34a1c4c82d7c/docs/research/track_a_o1_m1_r8_price_pilot_20261106/REPORT.md
[M1ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/o1_m1_matched_split.py
[M1BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/o1_m1_resource_bridge.py
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trottertracks/c1_h4_strong_baseline.py
[RPEBRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_rpe_energy_bridge_readiness_20261103/REPORT.md
[RPEV2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[SCHEDULE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/src/trotterlib/rpe_delta_round_schedule_validation.py
[RESOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/rpe_resource_accounting.py
[DECODER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/src/trotterlib/rpe_four_round_phase_validation.py
