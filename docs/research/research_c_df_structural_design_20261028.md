# 研究C：DF構造改良の横断調査・最小検証の準備

2026-10-28 JST。起点`64accf4d9eda87ae4bb229469ad05f3d5ac97f11`。独立branch `codex/research-c-DF-structural-design-20261028`。

**最初の一案には、headの一般Jを維持し、tailのJだけを固定した非零supportへ制約するCDFを推奨する。γ探索ではなく、最適化の実行可能集合を変更する。** 既知のsupport制約CDFの限定拡張であり、独立した新アルゴリズムの新規性を認定しない。新fitterは不要で、ffsimのloss/AD/codecとSciPyを再利用する。

今回は技術調査・人工PoC・実行条件の準備のみ。**実H4の数値decode、保存値の新集計、新fit/optimizer iteration、sampling、回路build/compileはすべて0。** 人工テスト60件＋29 subtestsがPASS。次の未承認契約は新候補1 fit・DF評価だけの最大21 logical unitsで、PF/RTE/shots/nativeへ自動接続しない。

## 1. 対象と調査の範囲

保存直鎖H4・隣接1.00Å・STO-3G、Nα=Nβ2/sector36、通常number CDF8、元frame順0..7/LD2、元g補正κとnuclear scalar一度を維持する。下流比較の優先政策はpartial ordered symmetric S2、δ=h=.1/s1/r128/K0/m0..12、各DF own-ground oracle/状態準備除外。RPE CA=.00159Ha、DF reserveCA/8=.00019875Ha、βtotalπ/6、βstatπ/12、候補failure.01/axis.01/26、同P0を使う計画である。

正式共通指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3に従う。旧SOURCE/raw/契約/grant/ledger/marker/STOPは保全し、承認証明・承認原文保存・新grantを作らない。共有workspaceのdirty変更は取り込まない。

必須6報告、現在の概要/status/manifest、関連SOURCE/tests/契約/保存JSONのstatus・来歴を確認した。NPZ/QPYはhash確認だけで、実配列・実回路を復元していない。数値は下記の既報の引用であり、再計算・再ランキングではない。履歴検索はPR-aware objectiveとsupport関連SOURCEに限定し、全branch不存在証明や網羅的priority調査はしない。外部一次資料は関連式/APIだけの技術確認で、重要GPT詳細科学レビューではない。

## 2. 既存研究・検証・今回との差分（Q1/Q2）

| 既存研究・証拠 | 評価済みの範囲と来歴 | 今回の差分／再利用 | 未検証・移送できないこと |
|---|---|---|---|
| [圧縮DF接口調査 20261024](research_c_compressed_df_interface_survey_20261024.md) | XDF/通常CDF/RCDF/SCDFの一般J・square・shift、既存ffsimのsupport指定、実装取得の制約 | 通常CDFの既存support機能を出発点とする。新規fitter不要 | 当時は実装・性能未検証。そこからnative資源削減を導かない |
| [Hamiltonian/回路調査 20261024](research_c_hamiltonian_circuit_survey_20261024.md) | C1～C5、basis共有、Gaussian/relative融合、UWC、過去singleton/区間法の負結果 | 既知融合を強baselineとして保つ。単なるgroupingや共通軌道変更を新DF法としない | P-A/PS48の条件付き結果は今回の分子/分解の証拠ではない |
| [PR-aware初期設計](research_c_pr_aware_df_optimization_design_20261026.md) §6/7 | J/U可変のsmooth tail mass目的を実装。固定U・凍結event価格weighted massは設計案のみ | 今回も同じloss/gradient/scaleを使い、hardなtail J supportを加える | 旧weighted massの再命名は新規案に数えない。単event価格和はwhole-P0価格ではない |
| [性能・機構整理](research_c_pr_aware_df_performance_mechanism_readiness_20261028.md) | γ/初期値/固定U cost-aware ablation、移送候補を設計 | γ/距離/分子/r探索を今回は行わず、構造一つを固定 | この計画自体は追加fitの実績ではない |
| [Stage A V2](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md), SOURCEe1085c95… | 同一qualified x0・R8、γ0/.1、2 fit。tensor/max-entry・sector誤差/guard・J/U・optimizerを保存。λR 2.2097618368/1.4424298856Ha、既報34.7246%減。両DF予算PASS | 初期snapshot/shared scales、通常CDF/現PR-awareを保存対照として再利用 | Gradient停留や大域最適性の証明なし。hard supportのfitは未実施 |
| [B1](research_c_pr_aware_df_stage_b1_execution_20261027.md), SOURCE7da0fdb3… | 両arm一般K/CAR・ground/alias・PF/補正RTE/Aをs4/r16/K0・各13roundで評価 | reference/atom/identity/phase-room codeを再利用 | 別J/Uやs1/r128へPF/RTE/groundを流用しない |
| [B2](research_c_pr_aware_df_stage_b2_execution_20261027.md), SOURCEa1cf0046… | 同条件の各32独立native sample、full512/adjoint/X/Y、Gaussian/whole-P0価格とmean/SE | cache/mutex/precision builder/full verifierを再利用 | 価格校正は非paired有限cohort、全law supportではない |
| [C1](research_c_pr_aware_df_stage_c1_execution_20261027.md) | 保存B1/B2を同CAの明示2仮定でshots/serial totalsへ接続。主仮定のProposed改善約83.086% | RPE/energy/費用集計を再利用 | Actual CA/全law誤差/母平均優位は未資格。旧s4/r16の順位を新政策へ転用しない |
| [M1](research_c_pr_aware_df_mechanism_execution_20261028.md), SOURCE47e0d406… | 保存shots×価格の4corner・記述的寄与。主要な記述的寄与はshots側 | λだけでなくshotとnativeを接続する評価基準 | 因果分離や構造変更の実験ではない |
| [M2](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md), SOURCEa6ffc23d… | 同J/U全frame決定論S2 s1、full512/PF13/数値CA/費用。Proposed kernel RZ11026、総RZ101825929455 | 強決定論対照を保持 | 今回新candidateの決定論結果ではない。新J/Uならその自身の対照が必要 |
| [M3 run02](research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md), SOURCE5354edec… | s1/r64、PF/RTE13、32native、471 COMPLETE。旧tuple/list schema STOPを保全 | JSON境界、occurrence独立、coherent反復/会計を再利用 | r64のprice/normalizationをr128や新DFへ掛け替えない |
| [M4](research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md), SOURCE7dbabc78… | 同J/U/s/hのPF13をidentity確認で再利用。r128 RTE/A/shots・未知価格の閾値。主仮定shots16056 | 同一現Proposedの保存結果だけを対照へ使う | 新DFはgenerators自体が異なるのでPF再利用条件を満たさない |
| [M4-Native128](research_c_pr_aware_df_M4_native128_execution_20261028.md), SOURCEdcf7ab6d… | 332 COMPLETE/32 full512。μRZ14076.4375、条件付き総RZ234117788135、M2比2.299196 | 現在の資源差と比較政策を出発点にする | 有限cohortの条件付き点推定。新candidateの価格・actual CA・母平均優位ではない |

このH4系列では、outer R、tensor残差、sector operator誤差、exact/smooth λ、PF/RTE/A、shots、native6指標、shots加重総費用まで既に評価している。**同じ保存J/Uについて「λが下がれば費用も下がるか」を再度問う実験は不要。** 一方、hard supportをtailだけに課したCDF、U間の関係を直接制御するfit、RCDF/SCDFの新H4 PR実行は、確認した現tree/限定履歴では実施済みの証拠がない。

H2/H6のrank政策・codec修正・決定論/sector/PF検証は関連資産だが、今回のH4 CDF8・LD2・PR-policyへ性能を移送しない。旧invalidated DF screening、raw欠損UWC prose、異なるrank10対照の初期比較は新しい強baselineの証拠としない。

## 3. 既知手法との重複と未解決事項（Q2/Q9）

- **XDF/通常CDF**：XDFのJは符号付きrank-one、通常CDFは一般対称J。Rとinner rankを分ける。Cohnらは既にC-DFによる短い時間発展回路を検討しており、圧縮/疎化という概念は既知である。[原論文のabstract](https://arxiv.org/abs/2104.08957)。本調査はP4 supportそのものの文献初出を主張しない。
- **ffsimの既知support機能**：v0.0.83は`diag_coulomb_indices`による対称J上三角の非零位置制限を備える。[公式SOURCE/API](https://qiskit-community.github.io/ffsim/_modules/ffsim/linalg/double_factorized_decomposition.html)。固定installed util.py/fit SOURCEをhash照合し、Uパラメータ群の後にJ上三角群が並ぶpack/unpackを確認した。既存指定は全frame共通、今回のhead-full/tail-P4はframe依存なので、既存full vectorへの固定線形注入で接続する。
- **RCDF**：残差にJ/Z係数のweighted L1/L2型正則化を加え、係数の大きさを制御する。今回のhard supportとは、有限penaltyでの縮小と許容集合の制約という違いがあるが、広い「正則化・構造化CDF」は重なる。[RCDF §B/Appendix A/B](https://arxiv.org/html/2212.07957v3)。同論文のp=1/2というpenalty次数と、既存PR目的のγ=.1という重みを混同しない。
- **RCDF実装の制約**：旧接口調査が固定PennyLane v0.45.1で確認したmatrix norm/最初のcoreの選択と論文の全entry正則化との不一致は未解消。今回別版や著者archive全体を再監査しておらず、忠実RCDF fitter確保済みとは扱わない。
- **SCDF**：rank-oneパラメータ化とfragment別symmetry shiftを共同最適化し、qubitization向けnormを抑える既知法。[SCDF §2.2–2.4](https://arxiv.org/html/2403.03502v1)。shift後のsigned-square数、N/N²補償とcontrol相対位相が必要。旧取得archiveのloaderだけでは新H4の汎用fit入口は確保できず、今回その不足を解消していない。

**新しいoptimizer/DF再構成/CAR/Gaussian合成器を作る必要はない。** まず既知support制約を研究Cのgeneral-J/固定PR政策へ最小拡張し、実際の資源価値を確認するのが合理的。独立新規性や出版上のmethod claimは、既存法との差と反証実験の結果を得てから別途判断する。

## 4. 三候補と採否（Q3/Q4/Q6）

| 案 | 数学的変更／既存γとの違い | 期待する機構と主な悪化リスク | 既知法との重複・実装・最小反証 |
|---|---|---|---|
| A：basis関係を含む目的 | J/U可変で`Fγ+η T(J,U)`。U間のdensity-basis関係とbasis確率を直接含める | 同λでもbasis遷移の構造差を識別できる。PF非可換性/DF誤差が悪化し得る。近いUがP0で安いとは限らない | Orbital optimization/cost-aware regularizationと重なる。既存fixed-U weighted massとは異なるが独立新規性未確立。JAX unpack/ADは再利用可能。まずsurrogateとwhole-P0価格の対応の検証が必要。今回は未実装・第二候補 |
| **B：tail Jの固定support（推奨）** | `min F_.1(θ)` subject to tail Jの禁止entry=0。Head Jと全Uは可変、R8固定。γを増減せず実行可能集合を変更 | 不要なoff-diagonal density couplingをfit前から持たない。Atom種類/inner構造/λとnative価格の交換を検証。DF予算やλ/PFが悪化し得る | Support制約CDF自体は既知。既存parameter injection/AD/SciPyを再利用しframe別maskだけ追加。1 fitで同CA/8を維持できるかを最初に反証できる。技術確実性と初期検証の情報価値で選ぶのであり、資源勝利確率を既知とした選定ではない |
| C：RCDF/SCDFのPR拡張 | RCDFの係数正則化、またはrank-one/shift/signed-squareをPR tail定義へ接続 | norm/fragment数低下の可能性。Gaussian費・sector外位相・finite target・λRは別問題 | 既知法の適用であり再実装を新規性としない。忠実fitter/shift補償が未解決のため初回採用を保留。解決後、同一物理予算の1条件で反証可能 |

Aの具体例として、tail frame確率`q_l=M_l/ΣM`と`D_lk=n−Σ_ij|(U_lᵀU_k)_ij|⁴`を使い、`T=(r−1)Σ_lk q_l q_k D_lk`を考えられる。これは独立eventの内部境界だけの**構造surrogate**で、head入口/出口・中央Z/ZZ・whole-P0価格は含まない。実直交Uなら0≤D≤n−1、列符号/列置換に不変。固定Jでλを変えずUを変えればTは変わり、既存のλ-only目的とは数学的に異なる。しかし現nativeはtailの隣接同一basis共有であり、単なるUの近さから削減は保証できない。近いbasisの無断mergeや小角丸めは行わない。λ=0時のqは未定義なので別分岐が必要。今回はこの案へηや新compilerを採択しない。

Bではinner rankを1にせず、負の係数/非PSD一般Jを許す。すべてのJを対角にすると再構成tensorは`Σ_li d_li u_li^⊗4`という過剰な添字対称性を持ち、tailだけでもその部分の表現力が制約されるため、初回はoff-diagonal edgeも残す。この制約の表現能力はH4で未検証であり、初期予算PASSや最適化成功を仮定しない。

## 5. 選定案Bの仕様

共通のchemist積分に対し

`gθ_pqrs=Σ_lij U_lpi U_lqi J_lij U_lrj U_lsj`,
`Fγ=.5||g−gθ||F²/f* + .1 (Mε(J)/M*)²`

を既存と同じ式で評価する。`Mε`は元の216 tail係数位置すべてを数える。強制0の項にも元のε寄与が残り、支持数の変更に合わせてlossを別物へ変更しない。ε=1e−8、f*/M*はStage Aの保存共通scaleをそのまま使用し、再scaleしない。

- Head frame0,1：J上三角10 entryすべて許す。
- Tail frame2..7：対角4＋edge(0,1),(1,2),(2,3)の上三角7 entryを許す。
- 禁止entry：(0,2),(0,3),(1,3)と対称entryは、初期化から終了まで厳密0。事後閾値・小値削除・mask探索なし。
- 全frameのUは既存real antisymmetric/expm codecで可変。Outer R8/frame順/group ID/LD2を固定し、分割・merge・sortをしない。
- Full parameter128＝rotation48＋J80。Compact110＝48＋head20＋tail42。固定線形注入`x=E y`で既存lossを呼び、gradientは`Eᵀ∇F`。新しいgradient計算器/solverを作らない。
- P4は**U列の添字graph**であり、分子の物理的軌道局在性を仮定した名称ではない。他graph/列順/初期値を結果後に探索しない。

Spin liftは`K_ab=J_(a//2,b//2)/2`、interleaved2i+σ。`n=(I−Z)/2`からZ係数は行和、ZZ係数はJ/4。P4の一般非零位置は1 frameあたりZ8＋intra-orbital ZZ4＋edge由来ZZ12で最大24、tail計144（密Jは36×6=216）。これは**構造上の位置上限**で実H4測定値ではない。符号・行和相殺・exact zeroで実active数はさらに小さくなり得る。

**144/216を回路価格削減率と読まない。K0/r128のevent数は依然128である。** RTEは全atomを毎回実行せず、basis6種類も固定のまま。分布が高価なZZへ偏る、λが増える、U/Gaussianが高価になる、headへ負担が移る可能性がある。Sparse J/low inner rank/少ないouter frame/安いnativeは別の性質である。

### 初期化と物理規約

Stage Aの保存qualified x0へmaskを一度適用し、既存ffsim unpackでJ/Uを得る。Uを再logmしない。**このprojectionは元DFを保持するcodec修正ではなく、開始時のtensor/Hを変更する。** 旧未制約初期値と同一とは言わず、初期tensor/sector誤差を記録する。初期DF予算は診断、表現・finite/support/rotationのcorrectnessがfit前gate。終了値でDF予算を判定し、初期値へのfallbackはしない。

元`κ=h−.5 Σ_r g_prrq`とnuclear scalarを固定する。新gθからκを作り直さず、maskで除いたdensity couplingをone-body/shiftへ無断移送しない。Identity係数は新Jから再抽出して正確な位相として扱い、λRからは除く。General Jをsquare専用DFHamiltonianへ格納せず、一般Kとraw CARを使う。

初期projectionのため、旧fitとの比較は同一初期点での純粋ablationではない。まず「固定した制約ansatz＋規定のprojected start」での可否を見る。性能改善が観測された後に初期点を揃えたdense control等が必要なら、その検証を別途設計する。

## 6. 再利用SOURCEと追加差分（Q5/Q6）

| 処理 | 再利用する関数・コード | 追加／適用条件 |
|---|---|---|
| Tensor objective/AD | `c_pr_aware_df.make_value_and_grad`, ffsim0.0.83 factory/JAX | 式/γ/scaleは不変。Compact固定注入とgradient gatherのみ追加 |
| Codec/最適化接続 | `c1_ffsim_corrected_codec.fit`, `c1_real_orthogonal_codec`, old qualified x0 | pack/unpack・Schur gauge・L-BFGS-B options=None・mock入口をそのまま使用。既存venv/ffsimは改変しない |
| 物理表現 | `c1_stage1.corrected_kappa`, `SectorCAR`, `rank_assessment`, `representation_gate` | 原g/c/sector36・既存guard。Tensor誤差とsector2normを別記録 |
| 一般J/spin/scalar | `c1_general_k_adapter.from_ffsim_number_arrays/general_k_coefficients`, full CAR | 非PSD/negative/general J維持、I位相保持、rank-one退化も人工確認 |
| PR target/law | `c1_stage3_connection`, `c_pr_aware_df_stage_b.prepare_arm/configure`, B1/reference | 新candidateのregistry/atoms/sign/identity/λ/ground/aliasを再作成してから使用。旧結果の移送なし。既存prepare_armはbaseline/proposedの2roleに固定されており、第三候補のtarget/role bindingは後続用の薄い接続が必要 |
| PF/RTE | `c1_stage3_reference.reference_kernels`, B1の13round処理 | 新DFではPFも新計算。M4のsaved-PF専用entryを無条件再起動しない |
| RPE/統計/energy | `c1_stage3_accuracy`, `c_pr_aware_df_stage_c`, M3 serial会計 | DF誤差[Ha]とsignal/phase誤差を区別。unknown synthesis/lawはnull、2仮定laneの結論は条件付き |
| Native/費用 | PrecisionBuilder/StrongBuilder、M3/M4-native、P0/full512/actual adjoint/X/Y、shared mutex、metrics/calibration | 同じ合法fusion/precision/P0を全armへ適用。新lawの新cohortを登録し、旧sample最大値をsupportにしない |
| 管理の最小部分 | old `Calls`, pilot `finish`/I/O | 科学call/fresh/STOPだけ再利用。承認証明frameworkは作らない |

追加は2 module・1 runner・2 testsの5 SOURCE：[support/compact adapter](../../src/trottertracks/c_df_structural_support.py)、[DF段階接続](../../src/trottertracks/c_df_structural_execution.py)、[専用runner](../../scripts/run_research_c_df_structural_support.py)、[fit人工test](../../tests/management/test_research_c_df_structural_fit.py)、[評価人工test](../../tests/management/test_research_c_df_structural_execution.py)。旧SOURCEはこの作業で変更していない。[SOURCE34](../../artifacts/research_c_df_structural_support/2026-10-28/technical/SOURCE.json)は現起点を固定する。Stage A当時のSOURCE29との比較では27件一致、2件は後続K1作業でoptional `arithmetic=None`経路を追加済みだった（`c1_molecule_extension.py`, `c1_stage3_reference.py`）。差分を静的に確認し、今回のdefault Noneの式が維持されることを確認した。旧SOURCEへ巻き戻したり、旧結果を新実行として再評価していない。

## 7. 人工PoCと回帰（Q7）

| 環境 | 内容 | 最終結果 |
|---|---|---:|
| task-local fit venv | 新6＋既存objective9＋corrected-codec15。128→110、exact mask/roundtrip、負/一般/full-inner-rank/rank-one、U det±/π、AD/finite difference、実ffsim→compact x0、実solver mock/nit0、finally復元、prepare/fit worker | 30 PASS |
| task-local eval venv | 新7＋既存pilot10。独立raw CAR/general-K full256/sector36、scalar/control相対位相の人工full512、同じs1/r128/K0のsigned RTE law・frame/group/λ接続、mock21-call E2E、DF不適格/optimizer不成功/correctnessFAIL/partialSTOP/fresh拒否、passive import/preflight/hash不一致 | 17 PASS |
| 別eval process | 既存general-Kの係数/型/spin/phase/grouping回帰 | 13 PASS＋29 subtests |

合計**60 unique PASS＋29 subtests**。新しい検証は13件で、実分子の科学性能testではない。Optimizerはmockで遮断し、実際の最適化iteration0。人工データはinvented J/U/tensorで、保存H4数値を入力していない。Full512人工検査も新candidateの実native回路検証を代替しない。

初回の人工evalで、テスト側のfull CAR/sector bit順対応漏れと、新decision JSONのNumPy bool型を検出した。既存のbit_reverse対応とPython boolへの明示変換で修正し、人工suiteを再実行した。初期ログも保持し、科学条件/allowanceは変更していない。[最終logs/verification](../../artifacts/research_c_df_structural_support/2026-10-28/technical/)。

## 8. 最小科学実験：最初はDF段階だけ（Q8/Q10）

[未承認契約](research_c_df_structural_support_contract_unapproved_20261028.json)：**新fit最大1回、最大21 logical units**。科学実行は別途チャットでの明示許可が必要。

| 項目 | 固定する第一段階 |
|---|---|
| 共通input/初期/scale | 保存StageA `prepare/input.npz`, `prepare/initial.npz`, `prepare/objective.json`、各hash固定。元P-H4来歴SHA8a9c4542… |
| 対照 | 保存通常CDFγ0、現PR-awareγ.1。新fitは候補B一件だけ。旧fitの再実行なし |
| 新候補 | R8/LD2、tail-P4、head-full、全U可変、既存F_.1/ε/scale、fixed projected x0 |
| Solver | ffsim0.0.83/JAXx64/既存L-BFGS-B/options=None。tol1e−8はffsim初期化設定で、SciPy tolを新設しない |
| 記録 | 初期/終了tensor F/maxabs、sector2norm/guard、support/inner rank診断/active係数、exact/smooth λ、full/compact gradient、raw compact optimizer x/jac、success/status/message/fun/nit/nfev/njev |
| DF適格性 | 表現・support・finite PASS、optimizer successと終了sector_error＋guard≤CA/8を独立記録。両方満たした時だけDF_ELIGIBLE_NOT_PERFORMANCE_GO |
| 不成功 | DF予算未達/optimizer不成功は登録結果として診断後STOP。Correctness FAILは依存処理を停止・partial保存。mask/rank/γ/retry/fallback変更なし |
| 出力 | fresh `artifacts/research_c_df_structural_support/2026-10-28/run01`、retry/resume0、mandatory STOP |

21=prepare4＋fit6＋evaluation11。Parameter数・内部AD評価・optimizer iterationを追加のfit枠に読み替えない。契約にcallごとの内容を列挙した。1 fitのみなのでfit同時並列は作らず、既存JAXのvectorized contraction/AD・sector36を使用し、実affinity/quota/空きRAMからresource planを記録する。CPU/RAM/worker/wall人工capなし、cache/venv/process/outputはtask-local。

原始の数値H4値をdecodeせず、fit/eval両環境のpassive preflightがPASS。SOURCE34/保存input14/旧保護1006件を照合した。SOURCE manifest SHA256 `406b5ce36cf626501b8732da8b82c321e60c9f610854c0fa8b085f982f4e7f48`、契約SHA256 `8a0c1c3387d9ced3970c44228699c7bdf76be0f45a79dcee3938a989c965bd66`。

将来の許可後に使う入口（今回は実行しない）：

```bash
.venv-eval/bin/python -B scripts/run_research_c_df_structural_support.py \
  --contract docs/research/research_c_df_structural_support_contract_unapproved_20261028.json \
  --fit-python .venv-fit/bin/python --execute
```

Task-local cache環境と-Bを親子に維持する。旧StageA/M2/M3/M4のSTOP済みrunnerを再開しない。

### 適格だった場合の後続設計：今回の21-callへ含めない

1. 新candidateの物理H/一般K/atom/identity/λ/law/own-ground/aliasを固定し、全13roundのPF・補正RTE・Aを新しく評価する。誤差とattenuationを新DF自身で測る。
2. 共通s1/r128/K0で通常CDFγ0、現Proposedγ.1、新candidateを比較する。**通常CDFγ0に保存s1/r128のPR評価・native価格はない**ので、旧s4/r16やProposed値を埋めない。ProposedのM4/M4-nativeだけは同一input/JU/SOURCE/政策のidentity確認時に再利用できる。
3. PF/RTEが登録phase-roomに入った対象だけ既存RPE/energy/2仮定scenarioへ接続する。Unknown all-law synthesis/implemented lawは0で補完せず、evidence laneは未資格のまま。仮定付きNを保証shotsと呼ばない。
4. 実価格が判断に必要なら、新baseline/新candidateの固定cohort native校正を**別契約・別許可**で行う。同じPrecisionBuilder/relative-head/同basis-tail/no cross-head-tail/P0/MPFR160/fusion機会を与え、全controlled operator/actual adjoint/X/Yを新しく確認する。Event加算proxyや144/216比でpriceを作らない。
5. `Σ N_ma E[C_ma]`で条件付き比較し、保存M2同Proposed J/U決定論を参考対照へ置く。ただしそれは新candidate自身の同J/U決定論対照ではない。PR固有の優位を主張する前には新candidateの強決定論S2 s1も必要。

M3/M4のSOURCE/input gateも旧Proposedに固定されている。新candidateを旧Proposedという名で通したり、旧hash条件を外して再起動したりしない。後続のimmutable J/U・SOURCE・call・cohortは今回未取得なので、未承認契約内でも`DESIGN_ONLY/NOT_EXECUTABLE`としcall上限はnull。**第一段階21-callのrunnerは準備完了だが、後続全段階が一括実行可能という意味ではない。** 新しいrankやfragment数を候補に有利に選ばず、今回R8固定の結果と将来の同誤差rank比較は区別する。

## 9. 残る問題と次担当（Q6/Q9/Q10）

- P4-tail ansatzがCA/8を満たせるか、初期projection/局所解がどれほど影響するかは未検証。DF予算を保証する新solverではない。NOT_METは登録した開始点/solver/条件の結果で、ansatzの数学的不可能性や最良近似を証明しない。
- Support減少がλ・PF/RTE・shots・Gaussian/whole-kernel価格・最終費用を改善するかは未検証。初回λ減少だけで科学的GOにしない。
- 物理sector予算、tensor loss、state energy、全controlled位相、有限RTE target、回路合成誤差、統計失敗率は別。Sector同値だけからfull controlled同値を主張しない。
- RCDF/SCDF制約、Aのnative surrogate対応、独立新規性、実装law/全law保証/actual CA/母平均優位は未解決。今回は追加保証や詳細GPT科学レビューを自動開始しない。

**ユーザー／研究C**：3案の比較と制約を確認し、最初の1 fit/21-callの構造仮説を採用して科学実行を別途許可するか判断する。**Codex**：その具体的許可後だけ、固定SOURCE/input/fresh outputで第一段階を実行し、DF適格性と構造を返してSTOPする。下流のPF/RTE/native/総資源はさらに結果に対応した契約が必要。

必要なSOURCE・人工証拠・契約案・本報告をcommit/pushし、remote SHA/tree・SOURCE・既存原本を照合後**mandatory STOP**。今回の依頼に新実H4科学実行は含まれず、run01は作成していない。
