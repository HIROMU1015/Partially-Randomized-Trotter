# 研究C：全law synthesis・implemented-law signal・CA資格の技術準備

2026-11-03。**技術adapter・人工検証はGO、実H4 Stage 1〜3の一括科学実行はNOT_READY。**
[未承認契約](research_c_all_law_signal_ca_contract_unapproved_20261103.json)は
`UNAPPROVED_DESIGN_NOT_EXECUTABLE`。Whole-kernel P0の有用な一様誤差と、現在候補のreference/joint-stream対応が未取得である。
値やcall上限を推測で埋めて実行可能とはしない。新しい実H4誤差計算・保存raw数値再解析・compile・sampling・shots評価は0。

正式共通指示の原本 `research_project_instructions_unified_2026-10-25.md` を実際に読み、§8、§8.4、§10を適用した。
技術調査と科学実行・重要科学レビューの許可を区別し、承認証明・ID・新grantは作らない。

## 固定対象と既存証拠

第一対象はPR endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
直鎖H4・隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36、number CDF rank8、元frame順LD2、
partial ordered symmetric S2、s1/r32/K0、δ=h=.1、RPE m0..12、CA=.00159 Ha/DF reserveCA/8。
当該DF自身のground-state oracle、状態準備費用除外、controlled full512、head/X/Y shell込みfenced serialを保持する。
P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、
MPFR160 declared phase、既存Gaussian/relative fusionのまま。ENTRY/XDF/CDFへの新評価は対象外。

| 固定証拠 | 結果commit / 科学SOURCE | 維持する資格 |
|---|---|---|
| [新PR/ENTRY共通r32](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md) | `beb8de5cd8b71f80bb69556f92299435ff431c3e` / `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` | 有限native32 full512 correctness、ideal/assumed条件付き資源。全law/actual CAはnull |
| [PR対保存XDF/CDF直接統計](research_c_pr_xdf_cdf_statistical_comparison_execution_20261103.md) | `2f1c86f161fb76e84dfa6b31f6912eb8a01ccb8f` / `e2884a071d60ff8d2ffcacc98705e2e5cfafb206` | 保存native32の事後的探索的な近似価格モデル。実装精度の証拠へ転用しない |
| [統合Evidence Map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/169e327ae50960173bb2690ab98ea92c85e7a5ed/docs/research/research_c_near_objective_xdf_evidence_integration_20261103.md) | `169e327ae50960173bb2690ab98ea92c85e7a5ed` | C01–C29/N01–N10/G01/Q01–Q02の元本文・過去判定を保持 |

Exact tail λ=.500044243591 Ha、ideal shots14,738、W12,539,808、条件付き総RZ94,288,043,542、
ideal13/13 round、保存32trajectoryのraw/compiled/actual-adjoint/X/Y PASSは保存値の引用である。
有限32件の最大残差を全law上界へ昇格させない。今回、実H4の誤差・信号・新価格統計を再計算していない。

## 調べた実装と再利用の適否

以下はbase `2f1c86f161fb76e84dfa6b31f6912eb8a01ccb8f` の実ファイル／関数を読んだ結果である。
その中の計算コードは変更していない。旧B1/B2/C2/H6の科学SOURCEは各固定報告・SOURCE manifestへ辿れる。

| 実装 | 実関数・適用範囲 |
|---|---|
| [Stage B](../../src/trottertracks/c_pr_aware_df_stage_b.py) | `prepare_arm/configure/atom_operators/build_kernel/independent_occurrence`。General-K/spin、符号・tail identity、K0 eventとCAR対応を再利用 |
| [Stage B2](../../src/trottertracks/c_pr_aware_df_stage_b2.py) | `reference_context/raw_gate/compiled_gate/adjoint_pair_gate/shell_assessment/XY_gate`。有限trajectoryのfull512照合。全wordへの一様性を含まない |
| [Stage C](../../src/trottertracks/c_pr_aware_df_stage_c.py) | `qualification/assess_round/energy`。既存evidence hookとraw-error/accuracy接続を再利用。元C1 loaderはerror_evidence=None |
| [r retuning](../../src/trottertracks/c_rte_r_retuning.py) | `normalization/assess_lanes`。r依存A、保存PF/RTEと3laneの分離。今回実データでは呼ばない |
| [Integrated native](../../src/trottertracks/c_pr_integrated_performance.py) | `native`。全kernelを構築してからwhole-kernel P0、actual inverseとX/Yを照合。旧価格32を新精度上界にしない |
| [update064 downstream](../../src/trottertracks/c_update064_r32_downstream.py) | `references/calibrate`。保存PF/RTE/referenceとraw budgetの作成経路を確認。旧candidate/rank政策の値は流用しない |
| [最新r32 adapter](../../src/trottertracks/c_near_practical_r32.py) / [runner](../../scripts/run_research_c_near_practical_r32.py) | `prepare_arm/execute/sampling/seed_manifest`。現在candidate/r32/cohort/SOURCEとPF→native→会計の結合を確認 |
| [Native fair](../../src/trottertracks/c1_h4_native_fair.py) | `controlled/decode/compile_checked`。Full512、固定basisとlayout検査。許容1e-10はcomponent gateでありoperator norm上界ではない |
| [NativeBuilder](../../src/trottertracks/c1_stage2_native.py) / [StrongBuilder](../../src/trottertracks/c1_h4_strong_baseline.py) | `sequence/actual_adjoint`、`transition/sequence`。隣接同basis共有、headのdirected relative Gaussian、scalar/phaseを保持 |
| [PrecisionBuilder](../../src/trottertracks/c1_h6_native_precision.py) / [P0 phase wrapper](../../src/trottertracks/c1_h6_compiler_precision.py) | `gaussian/_basis`、`compiled_wrapper`。MPFR160はdeclared phaseの安定化で、全gate/matrix演算の厳密包囲ではない |
| [Full synthesis](../../src/trottertracks/c1_h4_full_synthesis.py) | `segment/propagate`。両方のrepresented matrixのnorm growth、kernel反復、head/X/Yを再利用。Frobenius pilotを厳密enclosureと呼ばない |
| [Accuracy](../../src/trottertracks/c1_stage3_accuracy.py) / [deterministic](../../src/trottertracks/c1_stage3_deterministic_accuracy.py) | `RoundEvidence/assess_round/assess_energy`、`round_synthesis`。同じraw complex単位と独立shotの前提、DF一度の会計 |
| [RPE accounting](../../src/trotterlib/rpe_resource_accounting.py) | `_hoeffding_shots/evaluate_rpe_round_candidate`。新しいshots engineを作らない。今回実H4に呼ばない |
| [C2 categorical](../../src/trottertracks/c_pr_aware_df_stage_c2.py) | `choice_grid/analyze_table/tv/upper_float/synthesis_preview`。正確な有限格子TV・証拠入力代数を再利用。旧`law_round`は16×4で、現在32×1とは異なる |
| [C2 reference bridge](../../src/trottertracks/c_pr_aware_df_c2_bridge.py) | `event_mean/kernel_bridge/signal_schema`。係数・角度・normalizationと保存CARの差を分離。旧`replay/heads/kernel_bridge`のr16/s4/h1/40を現在へ無変更流用しない |
| [Coverage](../../src/trottertracks/c1_stage3_qualification.py) / [guarantee strategy](../../src/trottertracks/c_pr_aware_df_guarantee_strategy.py) | `support_size/synthesis_envelope`、`uniform_model_status/finite_claim`。Sample最大値、few-ULP、固定instanceのenclosureはuniform proofではない |
| [H6 state action](../../src/trottertracks/c1_h6_state_action.py) / [complete columns](../../src/trottertracks/c1_h6_synthesis_bound.py) / [stream](../../src/trottertracks/c1_subspace_synthesis_bound.py) | `StateActions/column_batch/ColumnStream/reference_info/repeat_bound`。全入力列/full出力/leakage込みFrobeniusの再利用候補。有限probeと完全basisを区別 |

[旧C2-Bridge実行](research_c_pr_aware_df_stage_c2_bridge_execution_20261027.md)の科学SOURCEは
`3f64842d69d54a70a5b736486b5faefa398903b8`。旧二候補/r16/s4のsector36観測であり、新PR/r32/s1のboundではない。
[旧H6 complete-sector結果](research_c_c1_h6_synthesis_bound_execution_20261026.md)も固定deterministic回路の領域限定数値評価で、
新H4の全sampling wordへ値・資格を移さない。旧科学契約のcalls/STOPを新実行へ持ち越さない。

## Stage 1：推奨するall-law synthesis経路と未完成部分

**推奨は構造的な一様評価＋有限familyの数値照合のhybrid**。216^32個のwordを全compileしない。
保存metadataのcatalog216 atomとr32から、catalog上のword数の上限は約5.04×10^74。
実finite-bit supportが減るかは別の表解析であり、微小正確率・underflowを推測で削除しない。
m12のkernel occurrence数は4,096、event数は131,072（s1）。これらはmetadataからの計算量見積もりで、誤差評価ではない。

Raw構造は9 Gaussian bases、head内部4 directed transitions、tail6 frame×最大36 support/sign、
2つのhalf-head、scalar/identity phase、head/X/Y shell。
Tailは同basis runのbasis出入りを共有し、中央のsupport/符号付き角度は順序通りに出す。
Headのrelative fusionは隣接pairに依存し、raw段階ではhead/tailを跨ぐ新しい融合を仮定しない。
`build_event_sequence` の最適化とwhole-P0の最適化を混同しない。

| 評価単位 | 必要な対応とcoverage |
|---|---|
| native primitive/event | Same exact saved-atom unitaryとのphase-sensitive operator差。全符号・support・制御・angle/reference誤差を含む |
| Gaussian/transition | Single-particle/vacuum相対診断だけでfull-Fock normを認定しない。隣接fusionの向きとphase補正を保持 |
| complete r32 kernel | すべてのwordに対する最終whole-kernel P0とraw/referenceの差。Event別compileの和だけでは不足 |
| coherent反復 | 同一・異なるkernel列のどちらでも一様operator boundからtelescoping。Law平均のM^nとの同一性には独立な新event列が別途必要 |
| actual inverse | 代数的C†への置換ではなく、実際のinverse生成結果も同じ一様epsilon/norm guardで覆う |
| X/Y signal | Head、forward、必要なactual inverse、X/Y readout、full出力を含めてraw complex信号へ伝播 |

Exact unitary積なら、各segment差の和が積のoperator差を抑える。状態の期待値差は軸ごとに≤2×状態距離、
複素XYは≤2 hypot(dX,dY)。実際のbinary64行列を扱う数値pilotでは両行列のnorm growthも必要で、
既存 `c1_h4_full_synthesis.propagate` を使う。PFはここで足さず、後のaccuracyで一度だけ扱う。
既存C2のexact rational previewは、与えられたunitary boundからの代数を検査するが、そのproof自体は検証しない。

Full512の有限trajectory residualはその1回路を覆うだけ。全lawに広げるには全wordに共通のraw/reference/compiler評価が必要。
Sector36入力に限定しても、作用・leakage・参照不変性を含む別の論証が必要で、勝手に入力領域を縮めない。
Complete-column/Frobeniusは**represented difference matrix**の数値operator上界になり得るが、
CAR/expm/state-action/減算/normの丸め包囲は別で、数学的・物理的保証を自動認定しない。

### 現P0のblocker

保存[compiler監査](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/compiler_coverage.json)を再利用し、
その関連Python source6件のhashが現在の固定venvでも一致することを確認した。実H4回路やpass managerを新実行していない。
P0はofficial definitionを空EquivalenceLibraryでbasis+uへ展開し、whole kernelにopt1を適用する。
`Optimize1qGatesDecomposition` のrun行列積・再合成はevent境界を越え、translation、phase更新、反復最適化を含む。
MPFR160、opt1、seed17、approximation_degree1は一様なoperator誤差ゼロの証明ではない。
Seed固定とmutable-state mutexは再現性・非干渉の措置で、全compiler出力の数学的決定性/誤差を保証しない。

過去の監査ではRustのparameter比較toleranceが見えても、実presetのInverseCancellation対象はparameter-freeだった。
それを任意RZの近似キャンセル誤差へ誤って足さない。一方、f64/libm/Rust再合成の一様誤差、
全runの入力領域、変形回数/loopの上限、宣言phase/referenceの囲い込みは未成立。
有限Gaussian/event残差表だけを作ればこの不足が解消するとは言えない。

共通の外側global phaseは測定期待値を変えないが、controlled baseのphaseはcontrol枝の相対位相になる。
Full controlled referenceの位相fitでこれを隠さない。Raw basisのB/ B^-1で打ち消すphaseと、
relative Gaussian境界・ancilla scalar/identity phaseを区別する。普遍的unitary距離≤2のような粗い式があっても、
有用なRPE phase roomを満たすとは限らない。今回は実H4の上界値・成立見込みを新計算していない。

## Stage 2：implemented-law signalの分解

理想確率、stored binary64確率、再正規化、NumPy CDF/searchsorted-right、53-bit格子の確率を別々に扱う。
既存 `analyze_table/choice_grid` は格子を2^53個列挙せず、CDFのceil差で各cell数と正確な有理TVを求める。
今回のadapterは旧16×4の固定乗数を使わず、N_m=32×2^mに一般化した代数だけを人工検証した。
新candidateの確率表・TV・信号差は未計算で、旧candidateの数値を流用しない。

同じexact unitary trajectory mapで |z(word)|≤1 なら、`|E_q z − E_p z|≤2 TV(p,q)`。
独立drawのcouplingならN_m個のeventについて `min(2,2 N_m eta)` というraw complex上界を設計できる。
これはideal shellによる複素matrix elementのlaw差であり、A_mを再乗除しない。実shell誤差はsynthesis側へ置く。
K0でも捨てられるorder選択drawがあり、実streamの順序と乱数消費を無視しない。

**実PCG64のjoint streamは、IID一様53-bit wordsというモデルそのものではない。**
別hash seed、単一カテゴリTV、32件のcorrectness PASSは、event/occurrence/shot間の独立性やjoint law差を証明しない。
保存native32は価格校正で、実RPEを実行したsampling scheduleの記録ではない。
一つのsample kernelをcoherentに再利用すると一般にE[U^n]≠(E[U])^nとなるため、同じlawと扱わない。
本来の独立occurrence政策と、価格校正平均を13roundの費用に共用することも別である。

橋渡しの対象を次の順に固定する。

`saved raw CAR finite-RTE reference → exact saved-atom IID mean → IID-grid same-atom map → actual joint-stream same-atom map → actual compiled/raw XY mean`。

| 加算項 | 担当する差 | 二重計上を避ける規則 |
|---|---|---|
| synthesis | 同じactual stream上のcompiled回路とexact saved-atom trajectory、basis/angle/control/phase/actual inverse/shellの差 | 有限sample最大値を使わず一様coverageが必要 |
| categorical | 理想カテゴリ確率とfinite-bit IID-gridの差 | Signed/complex unitary mapを共通にし、Aで再scaleしない |
| reference_bridge | Saved-atom平均と保存CAR raw referenceのλ、τ/φ、cos/sin/b、normalization、identity/scalar対応 | 旧PF/RTE自体の近似誤差を含めない。現在r32/s1/h.1で別途必要 |
| reference_uncertainty | Bridge/reference評価の未包含の不確かさ | Bridgeへ既に含めたものと別加算を併用しない |
| stream_model | Actual joint streamとIID-gridモデルの差 | Seed分離で0を代入しない。独立shotの前提も別に必要 |

この分解ではC1の`synthesis`欄に第一項、`law`欄に残る4項を合わせる。
後者は単なるカテゴリTVではなく、既存raw referenceへ接続する全差であることを出力component欄に残す。
PF/RTEをlawへ再加算せず、DFをraw信号誤差へ混ぜない。Observed residual、assumed bound、未取得はqualified inputにしない。

## Stage 3：既存accuracyへの接続とA/B/C/D

既存schemaは `kind=all_law_numerical_bound` または `rigorous_all_law_bound`、
`norm=raw_complex_signal`、`law_model=actual_implemented_law`、
`coverage=all_trajectories_all_occurrences_and_shells`、provenance、m0..12の`synthesis/law`を要求する。
[追加adapter](../../src/trottertracks/c_all_law_signal_ca.py)はcandidate/SOURCE/compiler/law identity、
分解した13roundのbound、joint stream・独立shot・full-P0/phase/actual-inverseの前提が揃う場合だけ、このhookへ渡す。
Schema一致はproofを検証したことを意味しない。現在の実H4 `error_evidence` はnullのままである。

既存 `RoundEvidence/assess_round` は
`delta_m=A_m*(eps_PF+eps_RTE)+eps_synthesis+eps_law`、`rho>=A_m-delta_m`、
`beta_det<=asin(delta_m/A_m)`を使い、固定beta/alpha政策から必要shotsを求める。
新しく別のphase/shots engineは作らない。合成・law budgetはこのraw単位で一度だけ足す。
PF/RTE/referenceの数値pilot資格とown-DF/alias条件は維持し、DFは`assess_energy`でHaとして一度だけ足す。

| 資格 | 意味／不足 |
|---|---|
| A ideal条件付きCA | 現在の反実仮想exact-native/ideal-lawでの十分条件。今回変更なし |
| B implemented-law数値的十分条件 | 全law raw誤差とactual stream/独立shotの前提を接続し、既存round/energy条件が通る場合。数値reference/PF/DFの資格が残る |
| C 保証付き精度 | All-law coverageに加え、reference・丸め・確率的前提・RPE理論の適用条件が必要。`kind`という文字列だけでは認定しない |
| D actual CA achieved | 明確に定義した実RPE手順を実際の量子shotsで実施した記録、outcomes、energy referenceと不確かさ、失敗率/精度評価が必要。A〜Cから自動的に導かない |

Classical state-actionのAer `shots=1`はsnapshot処理であり、量子測定やRPE実験ではない。
将来のclassical shot-level simulationで観測した場合も別のsimulation labelにし、実量子shotsなしでDをtrueにしない。
現adapterは数値的条件が通っても`actual_CA_achieved=None/guaranteed_CA=None`を維持する。
厳密signal入力が人工的に揃っても、保存PF/DF/referenceの数値資格を自動昇格しない。

最初の非PASS roundで後続round/energyを進めずpartialを保持する。
Unknown、loose bound、phase/radius不足はそのまま結果とし、r/h/CA/誤差上限/solver政策を変更して救済しない。
Solverに基づく将来の必要shots評価と、都合に合わせた事後のshots政策変更を区別する。今回どちらも実H4で行っていない。

## 計算範囲・calls・進行条件

| Stage | 計算とcallの導出 | 技術状態・進行条件 |
|---|---|---|
| 1 | Saved load1＋inventory/identity、N_ref/N_build/N_compile、full512なら512×N_verified_instance列action、coverage gate1＋13round伝播＋summary1。Family数・全P0 coverageの方法が未固定で、総数はnull | METHOD_INCOMPLETE。少数familyの観測を全wordへ広げる論証、reference/norm/phase/actual-inverse条件が揃うまで後続の成功を仮定しない |
| 2 | Current candidate table replay1、choice-grid解析1、有限atom/CAR bridge評価、actual joint-stream評価、13round law/reference伝播、summary。Bridge/streamの評価方法が未完成で、総数はnull | IID-gridだけなら条件付きpreviewまで。Actual-law/独立shotの不足を維持し、Bへ昇格しない |
| 3 | 実装済みadapterはsignal qualification1＋round_accuracy最大13＋energy1＋summary1＝**最大16 calls** | 必要なqualified packetが揃った場合のみ。Unknownなら2、初回phase不適格なら3。16は全Stage科学実行の上限ではない |
| 4 | 必要なshot-level実行・より強い包囲・モデル選択を別途設計 | OUT_OF_SCOPE。今回や前Stage完了から自動実行しない |

既存 `Calls`、H6 `column_batch`（batchingと無関係に論理column数を計上）、`resource_plan`を再利用する方針。
旧1,231/7/35/47callやnative32を新call予算へコピーしない。
新しい全Stage runnerの実装・最大calls・科学SOURCEは未固定で、現在のentryは科学実行を拒否する。
Stage1の有用な一様法の成立と、Stage2の実streamモデルの定義は、実行許可だけでは補えない不足である。

## 人工・回帰検証、SOURCE、passive preflight

追加は[adapter](../../src/trottertracks/c_all_law_signal_ca.py)、
[passive entry](../../scripts/prepare_research_c_all_law_signal_ca.py)、
[人工tests](../../tests/management/test_research_c_all_law_signal_ca.py)。既存native/PF/accuracy/統計engineは無変更。

[テスト記録](../../artifacts/research_c_all_law_signal_ca/2026-11-03/technical/tests.json)：**新62＋既存90＝152 PASS**。
128件は新adapter＋既存C1/C2/Bridge、24件はcomplete-column/norm-growth/relative Gaussian/controlled fusionの回帰。
人工identity0・単一primitive既知誤差・relative/global phase・actual inverse・Gaussian fusion・event順序・coherent反復・head/X/Y、
TV0/微小差・符号complex signal・不正確率・二重計上、coverage不足・欠損round・phase不足・numerical/rigorous/assumed・
actual CA非昇格、wrong candidate/SOURCE/compiler/law・SOURCE/input drift・fresh/retry/resume・call cap・FAIL/partial/STOPを確認した。
実H4の数値入力はadapter計算テストに渡さず、real metadataのテストはschema/hashだけ。
人工boundやmock前提は実H4のevidence laneへ注入していない。

[技術SOURCE](../../artifacts/research_c_all_law_signal_ca/2026-11-03/technical/SOURCE.json)は150file、
保存inputは16file（固定result/契約/STOP/metadataと、将来再生に必要な候補・input・factors・ground・PF/RTE reference等のhash）。
NPZはbytesのhashのみでdecodeしない。技術SOURCEには既存closureと調査した補助SOURCEを含み、すべてを実行する意味ではない。

- Branch：`codex/research-c-all-law-signal-ca-readiness-20261103`
- 独立worktree：`/tmp/prt-c-all-law-signal-ca-readiness-20261103`
- 実装固定commit：`ff80264447158c02d1f5d2f075e06f2ee8c970a4`
- 技術SOURCE SHA256：`04ce07003015734ccbe7f723f4d8634ad4111f1d6c59cd50a55cec22b2d70fa3`
- 未承認design契約SHA256：`de92d64b5952928800eb27e0e7bbaf41e82727e5efa5e3142949ffb47d70b150`
- Future fresh output：`artifacts/research_c_all_law_signal_ca/2026-11-03/run01`、現在は未作成。

[Passive preflight](../../artifacts/research_c_all_law_signal_ca/2026-11-03/technical/passive_preflight.json) PASS。
NumPy/SciPy/Qiskit/ffsim/JAX/PySCF/OpenFermion/flintのimportを禁止したprocessでもPASS。
これはdesign/SOURCE/input/旧STOPの技術整合だけで、科学実行可能性やall-law成功のPASSではない。
既定コマンドは以下の読取りだけである。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B scripts/prepare_research_c_all_law_signal_ca.py
```

`--execute`や`run(...execute_science=True)`でも現designは明示拒否し、科学call/outputを作らない。
実行可能な新契約に進む場合は未確定方法とcallsを先に固定し、別の明示科学実行許可が必要。
承認記録・ID・原文保存・grant発行は要件にしない。

## 資源・保全・残る判断・STOP

CPU32、available RAM約53GiBを確認。今回の支配処理は小さな人工行列・回路とscalar/byte検査で、
CPU/RAM/worker/wallの人工上限を設けていない。テストのみprocess-local BLAS1で重複threadを避けた。
将来のfull512検証は既存StateActions/ColumnStreamでcolumnをbatch化し、同一数学的入力を共有し、
独立familyの並列を実memory/quotaに合わせる。Full512 complex128行列は約4MiBだが、reference/差分/backend/cacheの総memoryも考慮する。
未測定のspeedupを主張せず、全word列挙を資源増量で解決しようとしない。

旧SOURCE/raw/契約/grant/ledger/marker/STOP、元dirty checkoutと未commit変更、共有venv/OS/cache/jobは保全した。
Evidence Map・研究概要・validation manifestの科学的資格は変更せず、関係する入口だけ追記した。
既知のvalidation manifest旧schema FAILの修復や全体PASS認定は対象外。
公開後にremote SHA/tree、公開本文と主要blobを照合する。
[技術STOP](../../artifacts/research_c_all_law_signal_ca/2026-11-03/technical/STOP.json)は実H4科学call0、actual CA/error_evidence=null。

残る科学的判断は、実PCG64 streamをどのrandom experimentとして定義・正当化するか、
Bの条件付き数値資格とCの厳密保証をどこまで求めるか、Dの対象protocol/量子shots/成功判定を何に固定するかである。
有用なwhole-P0一様法を確立する技術作業も残る。現在法で不可能という一般的結論は出さない。

**Mandatory STOP。** 次の担当はユーザー／研究C：上のrandomness・CA資格の定義と必要な保証水準を判断する。
Codexの次の候補作業は、その定義を踏まえたuniform compiler/reference評価方法の具体化と新契約の固定。
現designのまま実H4計算を開始せず、追加sampling・compile・RPE・移送・重要レビューへ自動進行しない。
