# 研究C：PR native費用構造・合法変換の人工検証と優先順位判断

2026-10-30 JST。起点`c978d53c94e1ec487184b15abb64f4b750151615`、独立branch `codex/research-c-pr-native-structure-20261030`。

**結論は`NATIVE_STRUCTURE_NO_GO`の提案。** A/B/Cには数学的に合法な変換があるが、今回の範囲では、次の実H4 native32実験を優先する候補を選ばない。Aは人工complete kernelの両fixtureでP0後RZが増加した。CにはP0後も有効な局所削減があるものの、同じcomplete fixtureでは費用不変で、作用範囲はtail中央部に限られる。Bは同じ相対変換を外側2境界へ適用する案で、実分子で有力とする費用証拠がない。これは**実H4での改善不可能性やnative最適化全体の否定ではなく、追加実験の優先順位に関する限定提案**である。

新11＋既存回帰57＝**68 distinct人工テストPASS**。最後のC complete-kernel比較追加後も、新11件を再確認した。実分子の数値decode／保存QPY decode・再集計／build・compile／sampling／DF fit／PF-RTE／shots／新しい総資源会計はすべて**0**。旧SOURCE・科学契約・raw・grant・ledger・marker・STOPは保全した。技術PoCは科学runnerへ接続していない。

## 1. 対象と引用の範囲

主対象は保存直鎖H4・1.00Å・STO-3G、通常number PR-aware CDF8 γ=.1、Nα=Nβ=2/sector36、元frame順/L_D=2、partial ordered symmetric S2、δ=h=.1/s1/r128/K0/m0..12。元g補正κ・核scalar一回・tail identity抽出、own-DF ground oracle/状態準備費用除外を維持する。比較先は**同じ保存J/U**の強決定論S2 s1である。

次の値は既報の引用で、新しいNPZ/QPY復元、特徴量集計、価格・shots再会計ではない。

| 既存証拠 | Kernel RZ | 条件付き総RZ | 今回の意味 |
|---|---:|---:|---|
| [M4 PR-aware CDF8 s1/r128/K0](research_c_pr_aware_df_M4_native128_execution_20261028.md) | mean 14,076.4375、SE 255.139624 | 234,117,788,135 | 主対象。理想仮定shots16,056、W=16,631,888 |
| [M2 同J/U強決定論S2 s1](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md) | 11,026 | 101,825,929,455 | 強対照。PRとの条件付き総RZ比2.299196はM4既報から引用 |
| [通常CDF7 native32](research_c_cdf7_native32_execution_20261030.md) | mean 17,754.9375 | 638,880,821,593.5 | 同じH4/LD2/PR政策だがJ/U/lawが異なるcohort。Rank削減だけの因果効果ではない |

[Surrogate修正後148-call](research_c_df_basis_surrogate_repair_execution_20261029.md)は、M4内の価格変動とbasis switchesの強い記述的相関を報告する。ただし複合Cモデルは不適格、head単独・isolated basis・basis-pair価格はnull。相関からGaussianの費用寄与率や因果効果を確定しない。[Shared-basis設計](research_c_shared_basis_df_design_20261029.md)の通常低rank CDFとの同値性、既存head相対融合・tail exact-key共有も引き継ぐ。

M4の科学SOURCEは`dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad`、input snapshot SHA256は`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、保存Proposed J/Uは`f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`、元P-H4来歴は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。この旧fit/source/inputと新PoCのidentityを混ぜない。

正式共通指示2026-10-25版の§8.1/8.2/8.4/10/10.1/10.3を実ファイルで確認した。承認証明、承認原文保存、新grant、独立fitter/compilerは追加していない。既存の詳細レビューを資料として参照したが、新しいGPT詳細科学レビューは実施していない。

## 2. 再利用対応と既存kernelの構成：Q1/Q2

| SOURCE | 確認した処理・再利用 | 今回追加しないもの |
|---|---|---|
| [Stage B](../../src/trottertracks/c_pr_aware_df_stage_b.py) `prepare_arm/build_kernel` | 元g補正κ、一般J、spin対応one-body、full-frame tail Provider、StrongBuilder | DF/J/U/atom/lawの変更 |
| [native bridge](../../src/trottertracks/c1_stage3_native_bridge.py) `partial_kernel` | unsigned格納角を検証し、符号を既存propertyで適用。前head→tail→後head→scalar | 旧[connection](../../src/trottertracks/c1_stage3_connection.py)の負係数angle検証へ戻すこと |
| [Stage 2 native](../../src/trottertracks/c1_stage2_native.py) | `runs/combined_diagonal/actual_adjoint`、完全な回路inverse | 別のadjoint実装 |
| [typed circuits](../../src/trottertracks/pr_dictionary_validation/circuits.py) | exact-key run、`append_central`、event phase/product順序 | 新samplerやevent並べ替え |
| [StrongBuilder](../../src/trottertracks/c1_h4_strong_baseline.py) | `transition/sequence/vacuum_phase`、解析的位相、相対Gaussian cache | 別のGaussian compiler／位相fitting |
| [PrecisionBuilder](../../src/trottertracks/c1_h6_native_precision.py) | exact-zero Givens、現在のsingle-particle precision政策 | 小角度の閾値刈込み |
| [compiler phase](../../src/trottertracks/c1_h6_compiler_precision.py) | P0＋MPFR160 declared global-phase蓄積 | 出力operatorを使った位相補正 |
| [Stage B2](../../src/trottertracks/c_pr_aware_df_stage_b2.py) | raw CAR、full controlled、compiled/raw、actual inverse、complex X/Y、metrics | 新しい参照経路 |
| [M4 native](../../src/trottertracks/c_pr_aware_df_m4_native.py)・[structural native](../../src/trottertracks/c_df_structural_pr_native.py)・[accuracy](../../src/trottertracks/c1_stage3_accuracy.py) | Complete kernel価格、Wμ+S、共通SE、誤差資格区分の既存設計を確認 | 保存値による新会計／未知誤差0補完 |

時間順に、前headはone-body/F0/F1の各h/2、後headはF1/F0/one-bodyの各h/2。**各half内部の2境界、合計4方向のrelative fusionは既実装**である。One-bodyもspin対応basisを使い、controlled diagonalにはI/Z/ZZの位相を含める。各halfの最初・最後のGaussianは残る。

Tailは`basis_key = sha256(repr(rotation))`の厳密一致でrun化する。Frame IDだけでは決めない。同じrun内では最初のinverseと最後のforwardを共有する。現在のM4 Providerはsingletonにも同じ精密full-frame Gaussianを返す。辞書用Providerにあるsingleton support completionを、M4で実測済みの最適化と混同しない。

異なるtail runの境界には、現在も前basisのforward→次basisのinverseの2個がある。Head/tailをまたぐ一般relative融合はない。κ/scalar/tail抽出identityは、中央演算や末尾ancilla phaseとして既定通り保持する。

**Q1の答え**：SOURCE上の主要な反復費用候補は、tailのrunごとのbasis出入りと中央Z/ZZ、およびcoherent kernel反復ごとに必要な前後headである。どれがcompiled価格の何%かは未測定。M4既報のexpected switches58.389326/runs59.389326、raw central RZ256という記述は引用できるが、Gaussian単独価格・head単独価格・新方式価格は`UNKNOWN`。Raw中央数とcomplete compiled総RZを引き算して費用分解しない。

## 3. 合法性と適用機会：Q3/Q4/Q5

### A：Tailの異basis境界

既存GaussianをB(V)、時系列で前runの終端B(V_i)→次runの入口B(V_j)†と書く。行列積は **B(V_j)† B(V_i)**。既存`StrongBuilder.transition`は単一粒子行列`V_j.conj().T @ V_i`を現在のGaussianで合成する。

Gaussianのvacuum phaseをφ、相対回路をRとすれば、補正は`fsum(φ_i, -φ_j, -φ_R)`である。これは既存RZ/XXPlusYY構造から解析的に得て、出力状態やoperatorを見てfitしない。回路全体のcontrol=0枝が閉じてidentityになること、control=1枝の相対位相、tail両端basis、scalarを保持する。Spin-lift・列符号・反射・signed zeroが違うbasisを無料に同一視しない。

したがって理想演算では合法。Binary64で同じprecision政策を適用し、有限人工回路でfull controlled検証をすることは別の数値確認である。**1個のrelative合成が2個のfull合成より安い保証はない。** 非常に近いbasisでも、正確な相殺を証明せず小角度を捨てない。

平坦化した非identity中央演算N個がR個のrunになると、旧tailには2R個のfull Gaussianがある。同一run共有によるN−R箇所は既済。Aの適用箇所Dは、identity barrierを跨がない隣接run境界数で`D≤R−1≤N−1`。現在のK0尾部は明示的identity atomを持たずN=rなのでD=R−1。AはD箇所の「2個→1個」置換に留まり、tailの両端2個は残す。**このSOURCE式からcompiled RZ削減率は決まらない。**

### B：Head/tailの2境界

同じ`B(next)† B(previous)`と解析的vacuum phase補正を使えば代数的に合法。前head終端→tail入口、tail終端→後head入口の最大2箇所であり、head内部4方向の実装済み融合とは別。境界を単純に削除せず、両側の基底をつなぎ、全kernelのcontrol=0/1、scalar、actual inverseまで保持する必要がある。Basisの近さ、DFのgauge同値だけではgate削除できない。

人工試験では境界の完全行列積を既存transitionと比較した。**Bを組み込んだcomplete kernel emitterやBのP0価格実験は作っていない。** Aの価格結果をBの実測値として転用しない。

### C：同一basis内の中央演算

同じ厳密basis runのcontrolled Z/ZZは対角で可換。同一supportの回転を`math.fsum`で結合し、厳密な0だけを省くことができる。Unsigned Z/ZZ productは同一supportの偶数個を消去できる。ただし、負productに伴うπ位相、RTE eventのphase、係数符号を反映する`event.unsigned_rotation_angle`は元規約のまま保持する。Identity phaseも保存する。

別basisを跨いだeventの並べ替えや近似的な相殺は禁止。PoCはrun内の同一supportだけを結合し、support間のXOR簡約、角度modulo化、tolerance pruningは行わない。N個の中央命令と、run別の異なるsupport数・残るproduct parityとの差が機会になるが、今回は保存eventを再集計していないので実H4の機会数は`UNKNOWN`。

## 4. A/B/C比較・P0重複：Q6/Q7

| 項目 | A：異tail basis | B：head/tail | C：中央局所融合 |
|---|---|---|---|
| 合法性 | 位相補正付き行列積で保持 | 同じ積。終端・head順保持が必要 | 同一exact-basisの可換対角だけ |
| 既存処理との差 | Head用transitionをtailへ拡張。既存同basis共有はそのまま | 外側2境界。Head内部は既済 | Tailは中央命令を逐次emit。決定論`combined_diagonal`のfsumは既済 |
| 最小変更 | Tail専用adapter | Kernel組立境界のadapterが必要 | Tail runごとの集約。新Gaussian不要 |
| 価格の対象 | Gaussian由来RZ/CX/depth/size | 同左、最大2箇所 | 中央RZ/CX/parity ladder/depth |
| 機会 | D≤R−1、K0なら最大r−1 | 最大2/kernel | 同run内の繰返しsupport。実頻度不明 |
| 追加価格 | Relative合成自体が高くなり得る | 同左。実測UNKNOWN | 古典的fsum/辞書集約。Compilerとの相互作用不明 |
| P0との重複 | 自明なinverse相殺はP0既済。一般のGaussian積合成はP0 opt1にない | 同左 | 隣接inverse/単一qubit簡約は既済。離れた同support集約は全て実施されるわけではない |
| Law・物理条件 | 変えない | 変えない | 可換run以外の順序・lawを変えない |
| 人工証拠 | Full controlled/P0/adjoint/X-Y PASS、RZ増加の反例 | 境界積・解析的位相PASS、complete価格未評価 | 局所削減例あり、固定complete fixtureでは費用不変 |
| 実H4効果 | UNKNOWN | UNKNOWN | UNKNOWN |

Qiskit **1.3.0**の`builtin_plugins.OptimizationPassManager`のlevel1は、`Optimize1qGatesDecomposition`と`InverseCancellation`等を用いる。Level2/3の`CommutativeCancellation`をlevel1の能力と取り違えない。一方、level1でもinit/translationと反復最適化が働くので、source上未集約の命令数をそのまま追加削減量としない。[環境記録](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/environment.json)に対象versionとplugin SOURCE hashを残す。

## 5. 人工PoC・回帰と価格観測

追加SOURCEは[人工専用PoC](../../src/trottertracks/c_pr_native_structure.py)と[人工test](../../tests/management/test_research_c_pr_native_structure.py)。旧科学SOURCEは無変更。既存typed emitterのflatten部分は公開helperでないため、符号/role/identity検証を伴う狭いflattenだけをtask-localに実装した。Gaussian・中央gate・raw CAR・compiler・inverse・X/Y・metricsは既存処理を呼ぶ。PoCに分子loader、sampler、科学CLI/run関数はない。

新試験は人工2/3空間軌道・5frame・L_D=2、一般対称J、h非対角、g=0、scalar .31を使う。固定seed904+nで生成し、同basis run、反射frame、near-but-distinct frameを含める。価格を見てseedや基底を選び直していない。2軌道fixtureのRZ増加後、非可換SO(3)を含む既存3軌道fixtureの価格も確認し、両結果を開示した。探索や良い例だけの採用はしていない。

| 検証 | 範囲と結果 |
|---|---|
| 完全controlled operator | 人工K0（全32次元）／K2（全128次元）。Raw CAR＋既存event_unitary/head指数を独立参照としてPASS |
| 位相・順序 | 負係数、product/rotation、identity/負identity product、spin-lift、scalar一回、control0 identity、control1 targetを確認。Ancilla phase改変はFAIL、event逆順は別operator |
| Gaussian | det±、π、signed zero、complex branch、near basisを含む境界積。解析的補正、phase fittingなし |
| P0/adjoint/X-Y | 固定P0 raw/compiled、両actual inverse、復帰、complex X/Yを既存B2 gatesでPASS |
| 不正入力 | Exact-key不一致、stale sign、別basisの中央集約、空runは明示的例外。Fallbackなし |
| 回帰 | Stage B、StrongBuilder、Stage B2、M4 metadata revisionの人工検証を再利用。旧H4 loaderや科学runnerを起動しない |

許容条件は既存B2と同じ、component abs1e-10、unitarity abs1e-10、complex X/Y2e-10、rtol0、phase fittingなし。人工PASSから実H4や全law enclosureを認定しない。

P0後の**人工complete kernel**（同じevent列、同じhead/scalar、A単独またはC単独）：

| Fixture | Route | RZ | CX | RZ depth | CX depth | Total depth | Size |
|---|---|---:|---:|---:|---:|---:|---:|
| 2軌道/K0/4events | 旧 | 748 | 474 | 411 | 430 | 893 | 1449 |
| 同上 | Aのみ | 780 | 482 | 421 | 430 | 909 | 1505 |
| 同上 | Cのみ | 748 | 474 | 411 | 430 | 893 | 1449 |
| 3軌道/K0/4events | 旧 | 1903 | 1134 | 952 | 998 | 2078 | 3648 |
| 同上 | Aのみ | 2039 | 1170 | 951 | 994 | 2069 | 3895 |
| 同上 | Cのみ | 1903 | 1134 | 952 | 998 | 2078 | 3648 |

Cの別の固定局所fixtureでは、同一runの異supportを挟んだ回転`.17,-.11,.23,+.11`を結合し、P0後RZ8→2、CX8→2、depth13→4となった。一方、重複unsigned product2個は旧P0でも完全に消えており、Cの追加改善は0。**有効な局所機構は存在するが、whole-kernelへの効果を確認したことにはならない。** Aのdepth副指標の一部改善をRZ主指標の改善に置き換えない。

[68件のlog/XML](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/pytest_final.log)、[最終11件再確認](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/pytest_new_final.log)、[2軌道metrics](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/artificial_P0_metrics_n2.json)、[3軌道metrics](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/artificial_P0_metrics_n3.json)とfull-control/adjoint/X-Y診断を保存した。68は11新規＋57回帰のdistinct数で、再確認11を加算しない。Local worktreeの人工検証でありimmutable CIや外部独立再現ではない。

実行前に32 CPU、available RAM約52GiB、load約1、T1000空き約3.7GiBを確認した。数秒の小行列・小回路試験には既存CPU/vectorized経路を使った。68件53.56秒、最終11件8.15秒。共有依存環境はread-only、venv/cache/tmpはtask-local。CPU/RAM/worker/wallの新上限や共有設定変更はない。

## 6. 強決定論との差とNO-GOの意味：Q8

同じ保存shots/W/shell会計なら`C_new=W μ_new+S`であり、kernelだけを等価変換する場合は`ΔC=W(μ_new−μ_old)`。同じ32 eventを使えば差分はpairedであり、その記述的SEを求め、全roundに同じ校正平均を使うためtotal SEはW倍する。旧別DF/cohort間のnon-paired比較と混ぜない。

M4既報の同J/U M2に対するkernel損益分岐は約**6,122.327980 RZ**。旧平均14,076.4375から約**56.5%**の削減が必要という依頼の尺度を維持する。これは新候補の費用を測った結果ではない。

AはO(run境界)の機会があり、理論上は情報価値を持つ。ただし現在precision/P0で単純な2→1がRZ改善を意味しない反例を得た。Bは最大2箇所で、Aの弱点を解消する別機構でもない。Cは厳密な局所改善を作れるが、作用する直接の中央命令の範囲は狭い。既報raw中央RZ256はcomplete compiled RZ14,076の削減上限として使えないものの、56.5%を期待する根拠にもならない。

したがって、**PR内部の小さな局所改善、条件付き総費用改善、強決定論への競争力**を区別した上で、今回は実H4 native32への候補採択を見送る。人工n=2/3の結果をH4の一般negativeへ移送せず、H4の新価格・削減率・mother mean・全law supportはすべてnull/UNKNOWNのまま保つ。新compiler、大きな融合探索、都合のよいbasis/angleの探索を続けない。

## 7. 科学契約を作らない理由・再開時の条件：Q9

依頼§9/§10のNO-GO分岐を適用する。**今回は最小実科学検証候補を選ばず、実行可能な科学契約は作成しない。** [非実行の判断記録](research_c_pr_native_structure_decision_unapproved_20261030.json)は`executable=false`、call上限0、selected_candidate=null、fresh scientific output=nullであり、固定native32を実行できる状態と表示しない。

将来、現在のprecision/P0でcomplete kernelに作用する具体的な価格改善機構が得られ、研究Cが再検討する場合の最小比較単位は、**保存M4の同一32 event列によるpaired native比較**である。旧32のQPY/参照/metricsとJ/Uをhashで固定し、全32を使い、再samplingや安い列の選別はしない。新raw/compiled full512、control位相、actual inverse、complex X/Y、6metrics、追加合成/古典時間を測る。保存W/Sと同J/U M2比較を再利用し、PF/RTE/shots/fitを再計算しない。

その段階でscientific runner接続・SOURCE・入力event全件・call内訳/上限・fresh output・retry/resume0・最初のcorrectness FAILでSTOPを固定し、別の科学実行許可を得る。現時点で存在しないrunnerや価格を既成事実にしてcall契約を作らない。Point μ/paired Δと全law cost support/保証を分離し、finite sample最大値や数値残差を厳密保証へ上げない。主RZのpaired差が非負なら局所案を棄却、負でもM2との損益分岐を別判定する、という採否軸は維持する。

## 8. 次に検討する一つの課題：Q10

**提案は「同じ保存DFのenergyを対象とした、roundごとの決定論／PR割当が成立する条件」の技術設計に一つだけ絞る。** 高mのattenuation/shotsとcoherent反復head負担を含む同一CA taskで、固定PRを全roundへ使う必要があるかを問う。先に各roundのcomplex signalが共通DF energyへ戻る誤差/alias条件を明文化し、強決定論にも同じ予算・割当最適化の機会を与える。今回は実装・再会計・候補探索・採択をしない。

これは新しい一般的schedule最適化の発明ではない。重複と差分を以下の既存資料から確認した。

| 既存検証 | 今回引き継ぐこと／繰り返さないこと |
|---|---|
| M2/M3/M4 | 同J/U決定論、s4/r16→s1/r64、s1/r128、head反復/attenuation/shotsの交換、高m counterfactualは既存。r64やr128を新案として再提案しない |
| [OPT-01/OP-III2](pr_algorithm_improvement_landscape_20261023/PR_IMPROVEMENT_OPPORTUNITIES.md) | round別r/K・β/α配分は既存。表現/split/方式を跨ぐ同一energy bridgeを課題とするのは既存の未採択方向。先にscope差を確定する |
| [Track A EVAL-01/02](pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md#eval-01) | 同precision/強baseline/測定込みnative/準備費用/不確かさを揃える。別のDF rank12・geometry・finite-time taskの結果を今回のenergy CAへ移送しない |
| [Track B Parent allocationの区切り](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md) | 平均を保つ既知Euler/child配分、normalizer/価格の交換、既知cost-aware IS、同fixture追加探索の限界を保持。配分だけを新PR算法にしない |
| [AX-1b一次結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md) | 保存費用モデルとconditional-oracle選択はoperational bias/shot predictorと別。Finite-time signal精度とenergy CAは別。新たなモデルfitや過去レビューの再認証をしない |

Track A/B/AXの参照は関連資料の技術的な重複確認に限定し、全branchの網羅監査や新しい詳細科学レビューではない。未知費用や削減率を予測せず、round別配分で競争力が生まれるとも主張しない。必要なbridgeが既存コードで完結するなら、新アルゴリズムではなく既存法の限定適用として扱う。

## 9. 保全・公開・STOP

[技術SOURCE/環境/証拠](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/)を保存した。旧原本は既存保護mapを再利用した2681 pathのhash-only照合と、baseからの変更path検査で確認する。数値NPZ/QPYのdecodeは行わない。新SOURCEはPoC/test、再利用SOURCEは既存manifestのidentityを参照し、旧契約を書き換えない。

ユーザー／研究C：この限定NO-GO提案と、次のround割当の技術設計を進めるか判断する。Codex：具体的な次の依頼を受けて技術設計を行い、新科学計算や重要詳細レビューはその範囲の明示許可後だけ実施する。今回はSOURCE・成果物をcommit/pushし、remote SHA/tree・SOURCE・旧原本を照合して**mandatory STOP**する。
