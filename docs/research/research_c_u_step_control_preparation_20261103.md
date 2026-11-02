# 研究C：Reduced U最適化のstep制御改善・技術準備

2026-11-03 JST。**技術的GO：既存PR目的・数値資格を保つ有限Armijo backtrackingを実装し、194件の人工・回帰検証を通過した。実H4での有効性・収束・資源改善を認定したGOではない。未承認科学契約を準備し、公開照合後mandatory STOP。**

共通起点は[`a4d24682e79857f1e97f105cc03448d0e1786ea2`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a4d24682e79857f1e97f105cc03448d0e1786ea2)。独立branch `codex/research-c-u-step-control-preparation-20261103`、worktree `/tmp/prt-c-u-step-control-preparation-20261103`。Codex-Aのr再調整、旧SOURCE・結果・実行契約は変更しない。

## 1. 対象・今回実施していないこと

固定条件は直鎖H4・隣接1.00 Å・STO-3G、number CDF8、元frame順LD2、signed実対称J・実直交U、元g補正κ・核scalar一回、Nα=Nβ=2/sector36。元目的は

\[
F_\gamma(J,U)=f(J,U)/f_*+\gamma(M_\epsilon(J)/M_*)^2,
\quad f=\tfrac12\|g-g_{DF}\|_F^2,
\quad\Phi(U)=\min_J F_\gamma(J,U).
\]

γ=.1、ε=1e-8 Ha、f*=2.1839632053132894e-9 Ha²、M*=2.2093719370722074 Ha、DF予算CA/8=.00019875 Ha、既存codec/gaugeを維持する。[最新Evidence Map][EM]と[元Objective SOURCE][OBJ]を参照。有限精度のcold J solve＋2 refinementが返す値・勾配を、厳密なmin_Jや真のreduced gradientとは同一視しない。

今回はsource・既存JSON/報告の読解、最小step controllerと薄い接続、人工・回帰検証、受動preflight、未承認契約・公開資料の作成だけを実施した。**実H4のNPZ/QPY数値decode、新fit、outer U更新、勾配再評価、DF物理再計算、PF/RTE・shots・native・総資源評価は0。重要GPT科学レビューも実施していない。** 保存NPZはidentityのbyte hashだけを確認し、科学出力`run01/`を作成していない。

正式共通運用指示`research_project_instructions_unified_2026-10-25.md`の§8/8.1/8.4/10/10.1/10.3を依頼元workspaceで確認した。これは今回の固定commitには収録されていないため、存在しない公開URLを付けない。許可の証明書・保存原文・新grant等は追加していない。

## 2. 旧停止について確認できる範囲

[統合実験][I]は科学SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`を固定して543/543 logical units COMPLETE、5 U更新を数値・物理資格付きで受理した。選定update005のF=.00661750979453、物理U₂=.658935814277で、U一次閾値1e-5は未達。次の方向の登録q={1/8,1/32,1/128}はすべて数値適格だが値尺度Armijoを満たさず、`FINITE_STEP_GRID_NO_ACCEPT`で停止した。Correctness FAILやDF予算FAIL、64更新上限による停止ではない。[親result][IR]／[旧STOP][IS]

旧[60-call guarded実験][G60]もα0=min(1,1.01/‖g‖)からhalvingを使っている。しかし8 trial・一方向・最大1更新の契約であり、登録範囲では更新0だった。続く[49-call局所診断][G49]は元初期点のより小さい3尺度でF低下を観測し、別の[19-call実験][G19]で1回の小step更新を受理した。その後が上記統合実験である。

したがって今回の選定は「Armijo自体の初実装」ではない。既存の目的値に依存しない初期尺度を再利用し、値比較・複数recenter・独立終端・毎受理点の物理評価へ接続する。旧8 trialの再開や、旧543-callの未使用枠を使う実験でもない。

**確定できるのは、実際に試した有限stepが不受理だったこと。** update005の新しい小step、局所曲率、inner bias、真の方向微分を今回測定していないので、H4停止の唯一の原因や、追加縮小が実H4で成功することは断定しない。

## 3. 旧尺度の解析的な弱点

旧[small.propose][OLDSTEP]ではd=-g、α=qF/‖g‖²であり、変位は‖Δη‖=qF/‖g‖になる。正のoffsetを持つ人工目的

\[
\phi(x)=c+\frac L2x^2,\quad c>0,\quad g=Lx
\]

では、

\[
\alpha_{old}=q\frac{c+Lx^2/2}{L^2x^2},
\quad |\Delta x|=q\frac{c+Lx^2/2}{L|x|}.
\]

x→0でφがcに残る一方、旧変位は縮むとは限らず、この例では発散する。厳密な二次関数のArmijo条件は0<αL≤2(1−c1)であり、固定3qがその範囲に入る保証はない。これは人工問題の解析であって、H4の局所モデルを同定したものではない。

新方式は初期stepの生成からF/‖g‖²を取り除く。一方、**既存のrelative value toleranceはFの尺度に依存するまま**なので、controller全体がobjective offsetに完全不変だとは主張しない。値尺度を下回る減少は受理せず、有限精度の未資格として停止する。

## 4. 再利用調査と16項目の対応

旧ファイルへのpatchやmonkeypatchによる本番の差し替えは行わない。テスト中のfault injectionは人工入力に限る。

| 確認項目 | 既存SOURCE・規則 | 今回の接続 |
|---|---|---|
| 1. Current J/Uの評価点 | `c_pr_coefficient_point.CoefficientChart.point/_primal/independent` | 一度decodeした同じJ配列で値・独立J・pullbackを評価する経路を再利用 |
| 2. Inner J Newton-CG | `c_pr_inner_point_solver.solve_j` | 同じNewton-CG、解析Hessian、元xtol/maxiter、raw pre-step jac規約 |
| 3. Cold J start | `PRReducedOracle.evaluate` | すべての新Uで元initial.npzのJ。選定Jや前UのJをwarm startにしない |
| 4. 2 refinement | 同evaluate | solve→refine1→refine2を固定。追加refinementなし |
| 5. 数値資格付きstate | `c_pr_reduced_limited.Oracle` | 最終inner/独立J、nonincrease等の元admissionを維持 |
| 6. Physical U tangent | `c_j_u_stable.state` / `c_j_u_precision.evaluate` | 独立physical tangentと同一点gradient比較を維持 |
| 7. Exponential chart | `c_j_u_precision.geometry` | 元の直交chart。失敗点の投影やgauge変更なし |
| 8. Recentered U | `c_pr_integrated.trajectory`の手順 | 受理後Uを新中心としη=0にする。同じ構成を新driverへ限定移植 |
| 9. Reduced gradient | 最終refined chart gradient、recenter時は独立physical gradient | admission済み近似勾配だけを次方向へ使用 |
| 10. Step生成 | 旧3qと`F/‖g‖²`、旧guardedのα0 | 新しい有限backtracking政策だけを変更 |
| 11. Armijo | c1=1e-4 | 値尺度を加えた不等式とstrict decreaseを維持 |
| 12. Value comparison scale | `c_pr_local_direction.value_profile/comparison_scale` | 閾値・refinement/cross-path観測尺度をそのまま再利用 |
| 13. Independent terminal | `c_pr_small_step.assess` | provisional pass後に同じ配列を独立評価。PASS前に受理しない |
| 14. DF物理予算 | `c_pr_integrated_evaluation.Evaluation.assess` | 元Hamiltonian・SectorCAR・rank_assessment・guard・general-K/CARを再利用 |
| 15. Rank/nullspace/tail injectivity | `c_j_u_reduced.applicability`と既存Oracle | 同じguard。追加nullを削除したりridgeを足したりしない |
| 16. Calls・STOP | `c1_stage3_integrated.Calls` | 新fresh出力と積上げ予算。未資格停止とcorrectness FAILを区別 |

上記は[旧inner SOURCE][INNER]、[stable state][STATE]、[admission][ADMIT]、[統合trajectory][TRAJ]、[独立終端][ASSESS]、[物理評価][PHY]に対応する。既存pipelineはstep政策を内部に固定しているので、そのファイルを編集せず、呼び出せるOracle/assess/物理primitiveを新しい薄いorchestrationから使った。

科学条件として維持するのは目的・scale・Hamiltonian・入力・codec/gauge・数値/物理資格・precision・provenanceである。新契約内で変更するのはstep初期化、有限縮小集合、前回受理αの引継ぎ、およびそれに必要なpoint/call枠。旧契約の政策を遡って変更したわけではない。

## 5. ArmijoとTrust-regionの適合性

| 方式 | 再利用可能な資産 | 今回の判断 |
|---|---|---|
| Armijo backtracking | 旧guardedの初期尺度、qualified Oracle、値尺度、small.assess、recenterと物理判定 | 最小変更で接続可能。採用 |
| Trust-region | 固定UのJ Hessian、固定JのU residual-Jacobian/SVD/TRF | これらはそのままΦ(U)=min_J Fの資格付きreduced曲率モデルではない。gauge/有限inner精度・モデル予測減少・ratio判定の追加検証が必要。今回は実装しない |

固定JのTRFとJ coefficient Hessianの区別は[既存curvature SOURCE][CURV]による。Trust-region一般の有効性を否定する判断ではなく、今回の最小差分としての選別である。新しいreduced Hessian、擬似逆・regularization、汎用optimizer frameworkは追加しなかった。

## 6. 選定したstep政策・受理順序・有限STOP

新[source](../../src/trottertracks/c_u_step_control.py)は、初回α0=min(1,1.01/‖g‖)。2回目以後の中心ではさらに`2×前回受理α`とのminを取る。各方向でα_i=α0·.5^i、i=0..31を順に評価する。後続中心へのα引継ぎは、毎回大きいstepからやり直すinner費用を抑えるための固定規則であり、不受理trialのJ/gradientを再利用するものではない。

小勾配領域では初回α0≤1なので‖Δη‖≤‖g‖となり、旧F/‖g‖による変位拡大を避ける。これはstep生成の性質だけであり、未知inner誤差を考慮した厳密降下保証ではない。

1. 現中心の全数値資格・有限値を確認し、d=-gを作る。
2. 既存Oracleで新trialをcold J＋2 refinementから評価する。`J_UNQUALIFIED`／`REDUCED_GRADIENT_UNQUALIFIED`だけは値・勾配を使わず棄却し、登録範囲で縮小する。未知のSTOPやcorrectness例外は隠さない。
3. τvを元の比較尺度として、`Ftrial+τv≤F0+1e-4 α gᵀd`かつ`Ftrial+τv<F0`を要求する。`PROVISIONAL_ARMIJO_PASS`はまだU受理ではない。
4. `small.assess`で同じJ/U/etaの独立終端を評価する。値・chart gradient・直交性・独立J資格の不一致は受理しない。
5. `Evaluation.assess`による独立DF物理判定を行う。元sector誤差＋guard≤CA/8とgeneral-K/CAR整合が成立してから`accepted`を記録し、独立physical gradientでrecenterする。

物理判定に不合格の候補は、`optimization_accepted_updates`上のprovisional履歴と`physically_qualified_updates`を分けて保持し、受理済みterminalへ置き換えない。Exportは最後の物理適格受理点（更新なしなら物理適格初期点）であり、λを見て新たに再ランキングする規則は追加していない。

| 終了理由 | 意味 |
|---|---|
| BOTH_FIRST_ORDER_MET | 元J/U両一次基準を満たす。厳密stationarityではない |
| MAXIMUM_ACCEPTED_UPDATES | 最大64回の登録更新に到達 |
| REGISTERED_OBJECTIVE_STALL | 元のstall尺度・3 cycle規則に到達 |
| BACKTRACKING_NUMERICAL_QUALIFICATION_EXHAUSTED | 最大32trialで数値適格点が得られない |
| VALUE_RESOLUTION_UNQUALIFIED | 登録探索が不成立で、資格付きtrialに比較尺度以下の変化がある。方向全体の降下不存在は意味しない |
| BACKTRACKING_TRIAL_LIMIT_NO_ACCEPT | 32trialまでに受理条件を通らない |
| STEP_ROUNDING_STALLED / NO_QUALIFIED_DESCENT_DIRECTION | 表現可能な変位または有限の非零降下候補方向が得られない |
| 初期／独立終端J／DF予算の未資格 | 最後の適格点を保持して停止 |
| Correctness・nonfinite・identity・直交性・CAR不一致 | failed STOP。縮小で無視しない |

Raw optimizer success/precision lossは観測として保存する。元J一次・latest-pair・rank/nullspace・nonincrease等が決めるadmissionへsuccess flagを代用しない。真のreduced勾配誤差、未知inner value bias、certified descentはnullのままである。

## 7. 実装差分と技術検証

新規科学コードは次の3ファイルに限る。

- [c_u_step_control.py](../../src/trottertracks/c_u_step_control.py)：step政策、有限search、既存primitiveを使うrecentered trajectory。
- [c_u_step_control_pilot.py](../../src/trottertracks/c_u_step_control_pilot.py)：受動SOURCE/input確認、既存物理評価へのpipe、fresh/Calls/STOP/export。
- [run_research_c_u_step_control.py](../../scripts/run_research_c_u_step_control.py)：既定は受動確認。明示的な実行オプションは将来の別実行許可後にのみ使用する。

旧物理workerはperformance経路も持つため、新workerのdispatchをinitialize/assess/closeに限定し、演算本体は旧Evaluationを呼ぶ。新しいSectorCAR/DF誤差計算は実装していない。親が作成した出力directoryで子workerがpreflightできること、子の異常終了でもfailed STOPを残すことを人工検証した。

### 7.1 検証結果

新[test](../../tests/management/test_research_c_u_step_control.py)・[pilot test](../../tests/management/test_research_c_u_step_control_pilot.py)と関連する既存testを使用した。**194 unique PASS（新46＋既存148）、FAIL/skip 0。** Fit環境では新physical-worker testを1件deselectし、同じtestを適合するevaluation環境で実行したため、未実行の1件ではない。初期debugや再実行のPASSを重複加算していない。

| 検証群 | 最終結果 | 保存log |
|---|---:|---|
| Step・orchestration・旧step回帰 | 107 PASS、物理worker1件を別環境へ分離 | [step log](../../artifacts/research_c_u_step_control/2026-11-03/technical/step_and_orchestration_tests.log) |
| 元目的・signed J・同一点・独立gradient・geometry・inner回帰 | 84 PASS | [inner log](../../artifacts/research_c_u_step_control/2026-11-03/technical/inner_and_geometry_tests.log) |
| 人工sector/general-K/κ・旧新worker pipe | 3 PASS | [physical log](../../artifacts/research_c_u_step_control/2026-11-03/technical/physical_synthetic_tests.log) |

必須A/B：φ=1+x²/2、x=.001では旧3qのαLがいずれも降下域を超え、不受理。新α=1は人工降下を通した。L=16の人工例ではhalvingの5trial目・α=1/16で受理候補を得て、4trialで打ち切る例は不受理として区別した。φ=1+(1e8)x²/2、x=1e-9では比較尺度以下の改善を受理せず、32trialの有限`VALUE_RESOLUTION_UNQUALIFIED`となった。いずれも人工目的であり、H4のstepや曲率の新推定ではない。

必須C：J未資格、latest-pair、rank/nullspaceを含む全admission flag、不適格値比較、nonfinite value/gradient、eta identity、独立終端不一致、DF予算FAIL、general-K/CAR異常、worker異常終了を注入した。受理・次方向・terminalへの混入を防ぎ、数値未資格とcorrectness FAILを区別した。

必須D：元Fγ、一般signed J・対称metric、same-decoded-point、独立J gradient、physical tangent、直交性、recenter、すべての新Uでのcold J、3 solve/2 refinement、元admission、元DF物理判定の人工回帰がPASS。旧統合trajectoryの人工testも不変SOURCEでPASSした。旧科学結果の再実行による再現試験は行わず、保存結果・SOURCEのbyte不変を確認した。

### 7.2 環境・効率・証拠の限界

[環境記録](../../artifacts/research_c_u_step_control/2026-11-03/technical/environment.json)。CPU affinity32、利用可能RAM約54 GB、memory pressureは低い状態を確認した。3つの独立した検証群を並列に実行し、群内では既存interpreter/JAXを再利用。小さい人工行列での多重thread競合を避けて、各processのBLAS/OpenMPを1threadに協調した。これは共有CPU/RAMの上限変更ではない。GPUや多数のworkerを必要とする実分子workloadは今回起動していない。

既存fit/evaluation環境はread-onlyで利用し、package install、共有cache/OS・他job設定の変更はない。既存vectorized Newton-CG/Hessian・拡張精度residual・persistent physical contextを再利用する。U→trial→refinementの逐次依存は維持する。速度比較をしていないためspeedup値は主張しない。人工PASSはH4性能・厳密収束・母集団の数値精度の証拠ではない。

## 8. 未承認の実H4科学契約

[契約案](research_c_u_step_control_contract_unapproved_20261103.json)。これは今回実行した契約ではない。

初期U/Jの来歴は統合実験の`update005`、candidate SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。保存先は`artifacts/research_c_pr_integrated/2026-11-02/run01/selected.npz`。[IR]の同id/physical candidate hash・DF PASSと[IS]のSTOPに結び付け、将来実行時は保存J/Uの物理予算を再評価してから、新しいcold J solveを開始する。新experimentのcold Jは元initial.npzであり、selected Jはcold startに転用しない。

新[SOURCE manifest](../../artifacts/research_c_u_step_control/2026-11-03/technical/SOURCE.json)は新3実装＋2testを固定し、旧統合SOURCE manifestをhash付きで参照する。元207個のSOURCEファイル集合を変更せず再利用し、契約の`saved_inputs`で9入力・来歴ファイルを固定した。Manifestは科学SOURCEとprovenanceの記録であり、実行許可を証明する仕組みではない。[技術検証summary](../../artifacts/research_c_u_step_control/2026-11-03/technical/validation_summary.json)に最終test引数・件数・保全状態を記録した。

### 8.1 科学logical callの積上げ

最大更新64、各方向最大32trial、初期を含む新U点P≤1+64×32=2049。

| Logical unit | 最大数 | 根拠 |
|---|---:|---|
| saved_load / initial_identity / physical_context / phase_A_assessment / objective_setup / optimization_driver / export | 各1＝7 | 初期読込・来歴・元物理context・保存selectedの物理評価・目的・driver・export |
| inner_J_solve | 6147 | 3P。元J開始1＋refinement2 |
| precision_state | 4098 | 2P。refined pairの同一点評価 |
| reduced_admission | 2049 | P |
| independent_terminal_assessment | 64 | provisional pass後、更新ごとに最大1 |
| physical_assessment | 65 | refined初期1＋受理候補最大64 |
| **合計** | **12430** | 旧未使用callとは無関係な独立上限 |

物理評価は別枠phase Aを加えて最大66。Innerは各solveの元maxiter1000/xtol1e-12を保持する。Logical unitはinnerの1 iteration/nfevやCPU時間ではない。出力・traceの範囲も2049点/各3solve/各1000iteration等で有限に固定し、不利なtrialの証拠を削除しない。CPU/RAM/GPU/worker/wall-timeの人工capは設けない。

将来のfresh出力は`artifacts/research_c_u_step_control/2026-11-03/run01`。Retry/resume=0、旧STOPの解除なし。Fγ・exact λ・独立J/physical-U一次条件・DF誤差＋guard・更新回数・登録停止理由・全trial/rawを保存する。最後の物理適格terminalをexportし、未資格trialは診断のまま保持する。新J/UのPF/RTE・shots・native・資源会計はこの契約に含めない。

### 8.2 実行前に必要なこと

**ユーザーが、この未承認契約の固定範囲を明示的に実H4実行許可する必要がある。** 技術的GO、人工PASS、`--execute`オプションの存在はその許可を代替しない。許可原文保存・証明書・新grant発行を要求せず、許可後もSOURCE/input・条件・one-shot/STOPを維持する。

今回実施したCLIは以下の受動確認だけであり、科学call=0を確認した。

```bash
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
  scripts/run_research_c_u_step_control.py \
  --contract docs/research/research_c_u_step_control_contract_unapproved_20261103.json
```

`PASSIVE_PROPOSAL_VALID`、maximum logical units=12430。別processでnumpy/scipy/jax/ffsim/qiskitがimportされず、SOURCE/inputのbyte照合だけで完了することも確認した。Python>=3.11の既存環境を使用する。

## 9. 技術判断・未解決事項・次の担当

**技術的GO**は、元目的・資格・物理判定を維持し、固定3stepで捉えられない人工降下と有限精度未資格を区別する最小controllerを用意できた、という範囲である。実H4でより多くのU更新が進むか、U一次条件を達成するか、λや下流資源が改善するかは未取得。より深い縮小でも不成立なら、この契約の有限停止として報告し、結果を見てsolverや政策を切り替えない。

次は**ユーザー／研究C**が上記契約の実H4実行を許可するか判断し、**Codex**はその明示許可後だけ、固定SOURCEのfresh一実験を行う。下流のPF/RTE・shots・nativeの必要性は別段階で判断する。**統合担当Codex**へはCodex-Aのr準備と本branchのリンク統合を引き渡し、今回は共有索引・研究概要・manifest・研究ノートを編集しない。

公開時にremote SHA/tree、公開本文、SOURCE/inputと旧結果・契約・STOPの保全を確認する。旧trackedファイルの差分は0で、現在の科学出力run01は存在しない。**Mandatory STOP：新規実分子科学計算・重要GPT科学レビューへ進まない。**

[EM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/docs/research/research_c_evidence_map_updated_20261102.md
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/docs/research/research_c_pr_integrated_execution_20261102.md
[IR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/artifacts/research_c_pr_integrated/2026-11-02/run01/result.json
[IS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/artifacts/research_c_pr_integrated/2026-11-02/run01/STOP.json
[G60]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/docs/research/research_c_pr_guarded_step_execution_20261102.md
[G49]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/docs/research/research_c_pr_local_direction_execution_20261102.md
[G19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/docs/research/research_c_pr_small_step_execution_20261102.md
[OBJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_aware_df.py
[OLDSTEP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_small_step.py#L31
[INNER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_inner_point_solver.py
[STATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_j_u_stable.py#L31
[ADMIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_reduced_limited.py
[TRAJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_integrated.py
[ASSESS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_small_step.py#L77
[PHY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_pr_integrated_evaluation.py
[CURV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/src/trottertracks/c_separated_j_u_curvature.py
