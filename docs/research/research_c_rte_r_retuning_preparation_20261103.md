# 研究C：新selected固定・RTE分割数r再調整の技術準備

2026-11-03 JST。共通起点`a4d24682e79857f1e97f105cc03448d0e1786ea2`。**技術準備は完了。未承認契約は最大1,265 logical units、r={16,32,64}を新規評価し、同一selectedのr128を保存対照として再利用する。科学実行は開始していない。**

成果物：[契約案](research_c_rte_r_retuning_contract_unapproved_20261103.json)／[接続SOURCE](../../src/trottertracks/c_rte_r_retuning.py)／[entry](../../scripts/run_research_c_rte_r_retuning.py)／[人工テスト](../../tests/management/test_research_c_rte_r_retuning.py)／[SOURCE manifest](../../artifacts/research_c_rte_r_retuning/2026-11-03/technical/SOURCE.json)／[検証結果](../../artifacts/research_c_rte_r_retuning/2026-11-03/technical/verification.json)。科学実行には本契約・公開SOURCE・fresh outputを対象とする別途の明示許可が必要で、unapproved metadataの変更や承認証明・新grantは要求しない。

## 1. 現r128の評価範囲と今回の問い

[Claim統合資料][ALIGN]、[統合実験][I]、[同一新J/U強D比較][D]、別branchの[寄与分析][B]を出発点にした。最新selectedはH4/1Å/STO-3Gの未収束有限endpoint `update005`。Exact λは旧PR1.44242988558→.568349854686 Haへ低下したが、既存の比較はr128を保持していた。

| 同一新selectedで取得済み | 保存値・資格 |
|---|---|
| DF | sector norm5.23337351802e-9＋engineering guard3.22739270291e-9 Ha、予算CA/8 PASS |
| PR s1/r128/K0 | 13round潜在適格。Ideal shots14048、W10487392、kernel平均RZ19825.625、shell込み総RZ207919150188 |
| 強D全frame S2/s1 | shots13728、kernel RZ11026、shell込み総RZ105532782860。数値合成pilotとideal反実laneは同じ整数費用 |
| 比較の制約 | PR/D主RZ比約1.970、CX depthはPR側が小さい。PR価格はfinite32校正、全law誤差・母平均・厳密CAは未資格 |

問いは**同じ低λのJ/Uを固定し、rを減らすことでshots側の利益をどこまで保ち、complete kernel価格も削減できるか**である。新DF目的・U更新・rank変更の研究ではない。最新の`TECHNICAL_NO_GO_CURRENT_EVIDENCE`は新しいcost-aware fit機構の選別判断であり、今回明示された有限r再調整の技術準備を研究C全体の終了へ置き換えない。

## 2. 固定条件・identity・SOURCE

| 項目 | 今回の契約条件 |
|---|---|
| 物理系 | 直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2/sector36、system8/full256・controlled512 |
| DF | 通常number CDF8、一般signed対称J・実直交U、元g補正κ、scalar一回、spin/frame順0..7固定 |
| PR | 元frame順LD2、partial ordered symmetric S2、s1、δ=h=.1、K0、m0..12 |
| 精度・RPE | CA=.00159 Ha、DF reserve1/8=.00019875 Ha、βtotalπ/6・βstatπ/12、failure=.01・axis=.01/26。既存十分shots式 |
| 状態 | own-DF ground-state oracle、状態準備費用除外 |
| P0 | Qiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase・位相fitなし |
| 融合・会計 | head内部相対Gaussian融合・tail exact basis_key共有。Cross-head/tail・近似U共有・追加最適化は含まない。Fenced serial kernel反復＋ancilla head/X/Y shell |
| 主指標 | shell込み条件付き総RZ。CX count/depth、RZ depth、total depth、sizeは副指標 |

Selected SHA256は`938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`、入力snapshotは`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。元P-H4来歴`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`と区別する。

| 固定資料 | 結果収録ref | 科学SOURCE |
|---|---|---|
| 新selected／PF/RTE／native32 | `aced1647b81e8d28091058cdc7b6e3e1ee54896a` | `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` |
| 同一新J/U強D | `38b8d0df39b8943178ecef611194819cff1c2169` | `5f27539ae0ab88181d9cfcd45badb54daf855c28` |
| 寄与分析 | `2639221a59713d431bc7c157854628bbc006a1bd` | 新科学実行なし、上記等の保存値整理 |
| M3 r64・旧J/U | [M3報告][M3] | `5354edecc6aa995082440dd14be0310bdd9902e1` |
| M4 r128・旧J/U | [M4報告][M4]／[Native128][M4N] | `7dbabc7866cc62a5460bf1579ce84c59493d8e1e`／`dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad` |

旧M3/M4のJ/U SHAは`f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`であり新selectedと異なる。旧r64ではkernel平均9105.59375 RZ、ideal shots20734、r128では14076.4375 RZ・16056 shotsという交換があったが、新selectedの価格や適格性として再利用しない。旧s4/r16はh=.025であり、今回s1/r16・h=.1の直接対照ではない。

## 3. r集合の選定理由

既存K0ではτ=λh/r、数学的なattenuationは`A_m=(1+τ²)^(-r·2^m/2)`。小τ近似では`−log A_m≈λ²h²2^m/(2r)`であり、同じAの尺度を保つrはλ²に比例する。旧PR比λ≈.394という保存値から、128より十分小さいrも調べる理由がある。これは候補選定の解析的動機で、新rの誤差・shots・価格を先取りした評価ではない。

| r | 役割・事前の理由 |
|---:|---|
| 16 | λ²の縮小尺度を32とともに挟む下側のpower-of-two。旧s4/r16の性能を転用する理由ではない |
| 32 | 64から一段下げた中間条件。少ないevent数とRTE/減衰の交換を識別する |
| 64 | 過去のs1 macrostep条件。ただし新J/Uで誤差・shots・価格を取り直す |
| 128 | 同一新selectedの保存基準。既存13round・shots・native32・費用を再利用し、新規評価しない |

新条件は16→32→64の順、128はStage Aで保存対照として添付する。登録外r追加、r8/r256、r連続最適化、結果後のcohort/指標変更はしない。r低下はevent数を減らすが、有限RTE誤差やshotsを悪化させ得るため、λやAだけで適格性・費用を判定しない。

## 4. 既存pipelineの適合性と最小差分

| 処理 | r依存性・再利用 | 接続先 |
|---|---|---|
| 元入力・J/U・DF予算 | 不変。保存candidate hash、独立物理評価J/U、rank/κ/scalar、DF PASSを照合。新DF norm/fitなし | `c_pr_aware_df_stage_b_runner.load`、一般K、`SectorCAR` |
| PF kernel・13 PF operator誤差 | 同じJ/U・LD・S2・h/s/frame順でr非依存。保存sector PFとC/resultのPF誤差を再利用 | M4 saved-PF経路。新PF kernel／exact-time exp／PF-error normなし |
| Ground/alias | r非依存。保存最低固有対の来歴＋state norm/residual/energy/alias照合。新eigensolveなし | 既存Stage B/同J/U対照の規約 |
| Tail | λ・signed atom catalog・identity/probabilityは同じJ/U。rでconfig/τ/event角度・normalizationが変わる | `configure_rte`、既存distribution gate |
| RTE・raw signal | rごとにcorrected kernel、13回の反復二乗／RTE operator2誤差／A／own-ground complex signalを取得 | `c_pr_aware_df_m4.corrected_kernel/rte_round` |
| RPE・shots/W | rごとに候補固有PF/RTE/Aを用いる。既存3lane・同じ十分条件 | Stage C `qualification/assess_round/energy`、M3 serial/SE |
| Native | rごとに32独立trajectory、complete controlled kernel。Full CAR/atom cache・Gaussian9・head相対4・shell検証は同一armで一度共有 | `c_pr_integrated_performance.native`、StrongBuilder、既存P0/QPY/adjoint/X/Y |
| r128 native | 同candidate/input/time/target/P0/cohort/STOPの固定来歴を確認。保存aggregate/calibration/lanesを引用添付し、sampling/build/compile/QPY decodeなし | 統合result・C/result・C/STOP・元契約 |
| 強D | 同じnew selected、同CA/time/failure/P0/serial scope。保存numericalとideal費用を再利用 | 同一新J/U D/result、SOURCE/STOP/契約 |
| Calls/STOP | 新しいfresh run専用。旧run・残枠を使わない | 既存thread-safe `Calls`・journal・STOP、task-local runner |

新規は`c_rte_r_retuning.py`とentry/test。既存3ファイルの変更は**opt-in引数**に限定した。

- M4：`expected_steps=128`を既定のまま追加し、指定rのcorrected RTEをsaved PFへ接続。`rte_round(..., state=None)`に人工検証済みのraw complex signal出力を追加した。
- 保存RC2/PR Stage Bの`distribution_gate`：`expected_steps=128`を既定のまま追加し、同じK0 gateをrごとに再利用した。RC2 optimizerや目的は変更しない。
- 統合native helper：従来SAMPLINGを既定のまま、明示的なr-specific samplingと同一arm用prepared contextを受け取る。旧r128の動作条件・定数は不変。異なるarmのcache流用はFAILする。

新しいRTE/PFエンジン・shots式・compilerは作っていない。全law探索・支持集合列挙・承認証明基盤も追加しない。旧科学SOURCEは固定Git commit/manifestに保全し、旧契約/結果のhashを新SOURCEへ置換しない。今回の3ファイルの新版hashだけを新SOURCEへ固定する。

Normalizationの実数式とbinary64評価は区別した。既存libraryは`exp(−r·2^m·log(B0))`を使用する。小τで`log1p(τ²)/2`へ評価順を変更すると人工例でも丸め差が増幅したため、実行経路は変更せず、per-step B0と既存libraryの累積Aを元の数値allowanceで検査する。近似式を丸めenclosureとせず、閾値を緩めて救済していない。

## 5. Stage A〜Dの進行・STOP

**A：入力・表現・保存資格。** SOURCE/input/package/fresh outputをpassive確認後、科学許可時だけNPZを読む。Selectedと物理評価exportのJ/U、2組の保存sector G/H、basis対応、general-K/CAR9因子を照合。DF誤差は保存PASSを引用し、再計算しない。Ground norm/residualと保存aliasを確認し、同一候補r128を対照へ添付する。いずれかcorrectness不一致なら後続へ進まない。

**B：r依存RTE/RPE。** r16/32/64ごとにlaw/config・normalizationを確認し、同じ保存PFを用いて13roundを評価する。PF、補正RTE、A、raw complex signal、raw-signal error/radius/phase room、十分shots、Wを区別して保存する。Evidenceの未知合成/law誤差はnull。Ideal exact native/ideal finite-RTE IIDと、各誤差≤.001Aのassumed laneは反実仮定である。

**C：条件付きnative校正・会計。** 全13roundが`POTENTIALLY_FEASIBLE`かつidealの登録energy/shot十分条件がPASSしたrだけ進む。各rは固定32件・新PCG64 stream・非paired/非adaptive、同じr-event完全kernelをP0へcompileする。Full512 raw/compiled、actual adjoint、complex X/Y、位相・unitarityを既存gateで検証し、6費用指標を保存する。最初の適格rで共通full CAR/reference/Gaussian/shellを準備し、以後のrで再計算しない。小さいsector head exponentialsは既存corrected-kernel branch内でrごとに生成する。支配的なfull512/native workloadへ並列化・共有を適用する。

`C_total,k=W μ_kernel,k+S_shell,k`を使用する。Wは`Σ_m,b N_m,b 2^m`、rはkernel内部event数、32は古典校正数で、どちらもWへ追加乗算しない。各coherent occurrenceは独立trajectoryの法則を維持する。同じ32平均を全roundへ使うSEはW×SE_kernelで伝播し、√13で縮めない。Min/maxをsupportへ昇格しない。

**D：保存対照との比較。** 各適格rについてr128、同一新J/U D numerical、D idealとの6指標比を保存する。主指標はshell込み総RZ、ideal/idealの共通反実条件を明示する。Evidence未資格やINFEASIBLE/UNRESOLVEDから価格優位を作らず、未知費用を0にしない。r最適化を含む新selectedの利益をDF目的だけの因果効果と呼ばない。旧J/U側のr再調整が必要な公平比較は別の後続実験である。

Registered十分条件の不成立・UNRESOLVEDは、そのrのnative/費用をskipして全13roundを保全し、他の登録rを評価する。SOURCE/input/表現/位相/reference/compile/nonfinite等の**correctness FAILは全依存callを停止**し、開始済みtaskを回収する。最初のFAILで条件変更・別SOURCE・fallback・retry/resumeは行わない。A=0やshotsの数値range不足は既存の不適格/未資格規約を使い、巨大値をclipしない。

## 6. 科学callの積み上げと実行規模

| Stage | 科学logical unitsの最大数 | 内容 |
|---|---:|---|
| A | 20 | 保存metadata/入力/JU/reference、binding、CAR/一般K9、ground、target、r128添付 |
| B | 3×80=240 | 各rのlaw/normalization/corrected、13reference＋13分類、summary、3lane×13accuracy、energy/shots、native進行判定 |
| C共通 | 35 | Full CAR1・raw9・sector1・reference context1・law1・Gaussian9・relative4・shell load6・shell assessment3、一度だけ |
| C候補別 | 3×322=966 | 9種×32=288 sample/build/reference/raw/compile/compiled/adjoint/X-Y/metrics、calibration1、cost calibration1、serial26、aggregate3、SE3 |
| D・最終 | 3＋1=4 | rごと比較、最終summary |
| **合計** | **1265** | 全3新rがnative適格の場合 |

すべてnativeへ進まなければ最大264 units。1/2/3条件でnativeまで進む最大値は621/943/1265。共有35は最初の適格rでのみ発生する。新round39、round accuracy117、最大trajectory96・kernel compile96・QPY384。r128・強Dの旧科学callを今回の残枠として転用しない。内部matrix操作・Newton-CG iteration等との単位混同もない（本scopeのoptimizerは0）。

SOURCE manifest67 entries、保存入力27 entries、評価依存10 package pinsを固定した。Manifest SHA256は`65c8aadbbbc2dd0179ef2465ce02c9fa546a0e942f3caa4aaf3920dd44eaeef7`、契約SHA256は`657aa5439bf66d6a9b19e7e726bd558ad82e4c19827f5386a84221c006409fd6`。Fresh outputは`artifacts/research_c_rte_r_retuning/2026-11-03/run01`。同directoryの存在は再開不可を意味する。Source/input/contractは新runへ保存し、journal・r別B partial・全round・QPY・metrics・result・resources・STOPを残す。

将来の実行entry（**現時点では未実行**）：

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_rte_r_retuning.py \
 --contract docs/research/research_c_rte_r_retuning_contract_unapproved_20261103.json \
 --execute
```

`--execute`なしはbytes/JSON/package版のpassive preflightのみ。旧one-fit/統合runnerを再起動しない。重い部分は既存resource_planで実空きCPU/RAM/quotaを確認し、13roundと32nativeを並列化、BLASを協調する。Qiskitのbuild/compile/adjoint作成排他は維持する。CPU/RAM/worker/wallの人工上限は追加せず、他ユーザーjob・共有venv/cache/設定は変更しない。

## 7. 人工・回帰テストとpassive preflight

**91 distinct tests PASS（新18＋既存73）。** 評価venvで87件、JAXを必要とする既存fixture4件は既存fit venvでPASS。最初の評価venvによる全件選択ではこの4件が`ModuleNotFoundError: jax`となったが、適合環境での確認により解消し、共有環境へのinstall/変更はしていない。[検証結果](../../artifacts/research_c_rte_r_retuning/2026-11-03/technical/verification.json)。

| 人工検証 | 確認した内容 |
|---|---|
| 実RTE primitive・人工2軌道 | r16/32/64/128、K0・λ/h/r尺度、13roundのRTE/A/raw signal、PFのr非依存、identity位相を既存referenceと比較 |
| Sampling | r件のK0 signed events、coherent occurrence別seed、r別namespace。理想IID実装距離の保証ではない |
| Stage A・人工4軌道 | 保存J/U/sector/PF/basis・一般K/CAR9・ground残差を実処理で接続。新PF engine/eigensolveを不要に呼ばない |
| 人工RTE/RPE→mock価格 | 成功・十分条件不成立・UNRESOLVED・compile correctness FAIL。Shots/W/shell/共有SE・6指標比較、未知evidenceのnull保持 |
| Native mock32 | 明示rのevent数、128 QPY/条件、actual adjoint/XYのdispatch。次rは同arm cacheを再利用しCAR/Gaussianの重複0 |
| Source/input/fresh/STOP | Candidate/DF/ground/source/cohort不一致、登録外r、byte drift、既存outputでのretry/resume拒否。FAIL時partial B・failure/STOP保全 |
| 回帰 | 元M3/M4・統合r128 native/人工pipeline・RC2 Stage B。既定128の数値経路・旧最適化fixtureを維持 |

実candidateのNPZ/QPY decode、新RTE/PF/RPE/shots/trajectory/build/compile、保存値の科学的再解析は0。人工回路・行列・metadataだけを使ったlocal検証であり、CI/外部独立再現・実H4性能PASSではない。実際のvenv Pythonでentryのpassive preflightを実行し、SOURCE/input/package照合PASS・scientific calls0・fresh output未作成を確認した。

## 8. 得られる結果・残る未資格事項・次の担当

明示許可後は、同一selectedでrごとのRTE/A/raw-signal/phase room・13roundの十分shots/W、適格rのnative32価格とshell込み資源、保存r128/強Dとの差を一群として取得できる。主RZが改善しない結果や、rを減らすと条件を満たさなくなる結果も保存する。全registered rが不適格でも別候補を追加しない。

未資格は、全law synthesis、implemented-vs-ideal law距離、価格母平均/support・選択後統計、厳密CA、状態準備/hardware費用、selectedのJ/U同時一次収束、DF目的だけの因果効果である。新rの価格をr128から比例推定せず、旧J/Uの費用も移さない。Finite32の点結果を保証付き順位へ昇格しない。

**ユーザー／研究C**が、公開された契約・SOURCE・r集合・1,265-call上限を対象とする科学実行の可否を判断する。許可された場合の**Codex-A**の次作業は、独立worktree/fresh run01の再preflightからStage A〜Dを一度実行し、結果公開・remote照合後STOPすること。Codex-BのU更新方式・目的・契約・成果物と共有索引は編集しない。共通運用指示2026-10-25版§8/8.1/8.4/10/10.1/10.3を適用した。

独立branch `codex/research-c-rte-r-retuning-preparation-20261103`で技術SOURCE・人工検証・未承認契約を公開し、remote SHA/tree・SOURCE/input・旧原本の保全を照合する。**Mandatory STOP。今回の依頼では実H4科学実行を開始しない。**

[ALIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4d24682e79857f1e97f105cc03448d0e1786ea2/docs/research/research_c_claim_contribution_alignment_20261102.md
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2639221a59713d431bc7c157854628bbc006a1bd/docs/research/research_c_pr_resource_contribution_analysis_20261102.md
[M3]: research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md
[M4]: research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md
[M4N]: research_c_pr_aware_df_M4_native128_execution_20261028.md
