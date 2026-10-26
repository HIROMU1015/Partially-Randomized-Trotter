# 研究C：PR-aware DF Stage B2・native費用校正の技術準備

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**B2専用runner、full controlled検証、固定標本の費用校正を接続し、人工16＋既存回帰56＝72 testsがPASSした。実H4の新規科学callは0。** 科学実行契約は未承認で、今回の作業からsampling・実回路build/compile・費用取得へ進んでいない。

## 1. SOURCE・保存入力・共通条件

| 項目 | 固定値 |
|---|---|
| 起点／B1結果公開 | `1331ac5df02272cc6edfd46d7de8d69f70ce882b` |
| 新しいB2数値実装commit | `4d7d9dbd3b521e288afd7e47aaecaf16facd6d3c` |
| Scope | `C_PR_AWARE_DF_STAGE_B2_NATIVE_COST_V1` |
| [未承認契約](research_c_pr_aware_df_stage_b2_contract_draft_20261027.json) SHA256 | `2ebd277ee2266314397ac7e870525cb007b64a5229e3b23953615aff14fb81d6` |
| [SOURCE manifest](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/technical/SOURCE.json) SHA256 | `cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf` |
| 元P-H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| Stage A final Baseline J/U SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| Stage A final Proposed J/U SHA256 | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |

Stage A公開`9ae5942d02572498b6ee1b5f309ca504ed58c6ef`、科学SOURCE `e1085c95f5c01b68566e17d543ace5832cb94061`、数値実装`5a185964090f2cb8d19d445d41cf520c4d8ed2c7`。B1科学SOURCE `7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9`、数値実装`366cb090a2c05c6d7ffeafdba2443bfc64b8a035`。これらのidentityをB2とは別に保持する。

対象は保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8のBaseline／Proposed。Nα=Nβ=2、sector36、8 system qubits＋ancilla-last1、controlled full512。元frame順、元g補正κ、核scalar一度、LD2、partial symmetric S2、δ=.1、s4、h=δ/s=1/40、r16、K0を共通にする。CA=.00159 Ha、DF reserve=.00019875 Ha、各DF自身のground-state oracle、状態準備費用除外も維持する。

B1の保存13round・DF reserve・ground・kernel/referenceをhashで固定して参照する。PF/RTE/attenuation、DF norm、ground eigenpairを再計算しない。B2は1 h-stepのcomplete controlled PR kernelの校正であり、m0..12に同じrandom sampleを反復する実験ではない。

## 2. 追加実装と再利用

新規ファイルは`src/trottertracks/c_pr_aware_df_stage_b2.py`、`scripts/run_research_c_pr_aware_df_stage_b2.py`、同名management testの3点。B1の141-call runnerは起動しない。旧数値SOURCEは一切変更していない。

| 既存資産 | B2で再利用する処理 |
|---|---|
| `c_pr_aware_df_stage_b.prepare_arm` | 2 CDFを別source/input identityでgeneral-K・native modelへ接続。κは元g補正、frame順を保持 |
| 同`configure`／`independent_occurrence`、`trotterlib.rte` | armごとのlaw再構成、既存PCG64 sampler、16個のsigned K0 event。新しいsampling lawは実装しない |
| 同`build_kernel`、`c1_stage3_native_bridge` | chronological head forward half→signed events→reverse half、scalar/identity phase、既存full-frame provider |
| `StrongBuilder`／`PrecisionBuilder` | exact interleaved spin-lift κ基底、保存full U、exact-zero/LAPACK/atan2、head内relative fusion、analytic vacuum phase |
| `c1_stage1.car_bilinears/raw_base/raw_frame`／`b.atom_operators` | 回路のGaussianとは独立なfull256 CAR base/frameとsigned rotated Pauli |
| `c1_stage3_reference._ordered_exponentials`／`event_unitary` | headのCAR指数をarmごとに一度cacheし、各trajectoryの独立referenceを構築 |
| `component_check`／`assess_native_probe`の既存検証方式 | full controlled/reference・raw/compiled・adjoint・unitarity。新runnerはdecode済み行列を保持し、費用をgate後に分離計上 |
| `compile_checked`／MPFR160 `compiled_wrapper`／`CompilerProfile` | Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、qubits_initially_zero=false、phase fittingなし |
| `readout_segments`／`shell_unitary`／`signal_gate` | ancilla-only head/X/Y、測定を除いたOperator化、complex signal・norm検証 |
| `circuit_metrics`／`cost_point` | whole-kernel価格と標本平均・SE。新adapterは標本分散・観測範囲を追加 |
| `Calls`／`resource_plan`／`CompileSerialCalls` | 科学call計上、correctness STOP、空き資源による並列度、P0 compile排他 |

`trajectory_reference`の最小cache adapterは既存B1 helperと同じCAR/event積を使う。人工テストで両経路の一致を確認した。再利用したreference helperは毎回head expmを作るため、64 sampleで同じ指数を繰り返さないよう既存exponential cacheをarmごとに使う。新CAR代数、Gaussian合成器、compiler、fitter、shot推定器は作っていない。

Head各半分の内部relative Gaussian fusionのみ適用し、tailはprecision full-frame provider、隣接exact同一basisの共有だけを使う。Singleton completion、head-tailを跨ぐ一般相対融合は無効。未確認の融合効果を計上しない。

## 3. Full controlled correctness

将来の実H4実行では各armのraw CAR base＋8frameをfull256で構成し、B1保存sector36へのmappingを9因子まとめて照合する。tailの非identity atom＋identity位相もfull CARと照合する。このsector gateはfull controlled検証の代用ではない。

各sampleで独立reference W（256×256）を作り、ancilla-lastの`diag(I256,W)`とraw・compiled回路の**全512入力列／全512出力成分**を比較する。control0 identity、control1 evolution、scalar／相対位相を同時に検証し、任意global phase除去やphase fittingはしない。

各sampleの検証は次を含む。

- Raw対reference、compiled対reference、compiled対raw、raw／compiled unitarity。
- Rawとcompiledそれぞれのactual circuit inverseを構築・decodeし、reference adjointと復帰を確認。
- 保存own-DF groundをfull256へ埋め込み、verified head/kernel/readoutの行列作用からX+iYを評価。raw／compiledの双方を`〈ψ_DF|W_trajectory|ψ_DF〉`へ照合する。
- 共通3 shellはancilla-only構造と既知H／H·S†行列、P0前後の位相を確認する。

単一unitary trajectoryのX/Yにattenuationを掛けない。これはB1のideal-law平均信号でも、量子測定結果でもない。Gaussian gateを個別にsectorへ射影しない。

固定allowanceはmatrix max-entryとunitarity各1e−10、complex signal2e−10、oracle normalization1e−12、rtol0。既存native数値条件を継続する。残差はmax-entry／Frobeniusとして保存するが、厳密丸めenclosure、全law operator上界へ昇格させない。正しさPASSは実行した64 trajectoryに限る。

## 4. 32 trajectory／armの校正と費用資格

初回案はBaseline／Proposed各32、合計64独立な古典的校正trajectory。1 sampleはh=1/40、r16／K0のcomplete controlled PR kernelであり、1024 event microstepsを扱う。標本数は探索的pilotの固定cohortで、精度・母平均順位保証ではない。追加sampleや都合のよい停止・cohort変更をしない。

既存seed関数へ`(master=2026102702,cohort=B2_fixed32_per_arm_v1,B2_SOURCE_manifest_SHA256)`、arm、final J/U hash、round sentinel−1、axis label `B2_CALIBRATION_ONLY`、sample0..31、occurrence0を渡す。Armごとに正しい異なるlawからsampleし、同じseed番号から同じtrajectoryが生じるとは仮定しない。Paired variance reductionは採用しない。PRNG substreamの分離を実装上のIIDモデルとして扱い、厳密独立乱数・ideal lawとの一致証明ではない。

各sampleのcomplete kernelを一体build／P0 compileして、RZ count、CX count、RZ depth、CX depth、total depth、circuit sizeを取得する。Head/tail/eventの個別価格の和をwhole-kernel価格と呼ばない。共通head/X/Y shellのmetricsは別記録し、kernel平均へ含めない。将来のwrapper政策はsegmentごとに独立compileし、境界を越えた最適化を仮定しない。今回、反復回数やshotsを掛けた総費用は計算しない。

既存`cost_point`のsample mean・SEに、`Σ(c−mean)²/(n−1)`の標本分散と観測min/maxを追加する。比較は各metricの標本平均差・比を記述するだけである。未知のideal-law期待値、finite-bit law差、全support bounds、保証CI、保証付き資源優位はnull。Rare eventは32 sampleで未観測となり得るため、SEを保証付き精度と扱わず、観測最大値もsupport上界にしない。

## 5. 科学call・実行規模・順序

| logical unitsの内訳 | 最大数 |
|---|---:|
| input/evidence/CAR context、fit load2、native model2 | 7 |
| law prepare2・gate2、ground load2、reference context2、sector gate2、raw factor18 | 28 |
| Gaussian full basis18、head relative Gaussian8 | 26 |
| 共通shell build3・compile3・assessment3・metrics3 | 12 |
| 64 sample×（sampling、raw build、reference、raw gate、compile、compiled gate、adjoint pair、X/Y pair、metrics） | 576 |
| 2 arm calibration＋比較1 | 3 |
| **最大logical units** | **652** |

Runnerの`LIMITS`と一致し、64標本mock E2Eで652/652を確認した。`adjoint_pair_gate`はraw＋compiledの2 inverse circuit build/decodeと全operator比較を一括計上する。`XY_gate`もraw＋compiled各X/Yの一括unit。STOP保存は科学unitを追加しない。Gaussian cacheがexact重複を再利用するとrelative Gaussian実数は上限より小さくなり得る。

主回路は64 complete kernel build＋3 shell build、P0 compileは64＋3＝67。これとは別にGaussian basis要求18、head relative最大8、actual-adjoint補助回路128を明示する。QPYはkernel raw/compiled/両adjoint256＋shell raw/compiled6＝262件、system reference NPZ64件を保存する。

SOURCE47件・保存input22件をpassive preflight→Stage A/B1来歴→共通shell→両arm準備・full/sector／law／ground gate→独立trajectory検証→各metrics→全標本の記述統計・比較の順。実source／input／package不一致は科学call開始前にSTOP。位相・Gaussian向き・順序・reference・compile・adjoint・X/Yのcorrectness FAILでは依存callを停止する。開始済み作業を安全に回収し、sample別JSON/QPY/reference・call journal・failure/STOPを保全する。FAIL後は完成した他sampleから母平均比較を出さない。

Fresh outputは`artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run01`、retry/resume0。新fit／optimizer／rank探索、PF/RTE/DF norm再計算、shots／CA／総資源は0。契約名のdraft／UNAPPROVED metadataは実行許可を証明する仕組みではなく、別途ユーザーのチャット許可が必要な提案状態を示す。承認証明書・原文保存・新grantは不要。

## 6. 性能・非干渉

独立reference／Operator／adjoint／X/Yはtrajectoryごとに既存resource_planで並列化する。full256 CAR・signed atoms・head exponentialsをarm内で共有し、sampleごとの重複CAR／expmを避ける。全Operatorはfull512までで、診断用SVDを行わない。

Qiskit1.3 P0は既存process-local mutexでcompileのみ排他、StrongBuilderのmutable cacheはarmごとのbuild lockで保護する。両lockとも取得後に既存CallsのSTOPを確認し、待機taskがFAIL後に新しいbuild/compileを開始しない。Builder以外の独立taskを直列化しない。BLAS/threadはworker数に協調し、CPU/RAM/worker/wallの人工上限を追加しない。速度向上率・実H4実行時間は未測定。

Passive確認時のaffinityは32CPU、MemAvailable約52.7GiB、pressure avg10/60/300は0。実科学実行時は実際の空き資源・quotaを再確認する。既存evaluation envをread-only使用し、task-local cache/tempだけを作成した。共有venv／cache／他job／サーバー設定の変更は0。

## 7. 人工・回帰結果と保全

| テスト | 結果 |
|---|---:|
| B2専用：passive import／NPY header・hash・来歴、seed分離／K0-r16、full32/full512、CAR cache対応、raw/P0/reference・actual adjoint・X/Y、位相・Gaussian向き・frame順・compile故障、統計/null、人工E2E／STOP／fresh、64標本mock call計上 | **16 PASS** |
| 既存Stage B・strong baseline・feasibility回帰 | **56 PASS** |
| 合計 | **72 PASS** |

[新試験log](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/technical/b2_tests.log)／[既存回帰log](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/technical/regression_tests.log)／[構造化試験結果](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/technical/tests.json)。人工行列・metadata・回路のみを使ったローカル試験で、外部再現や実H4の科学PASSではない。

実kernelの人工E2Eでは2 sample/arm、4 kernelを全pipelineで検証した。別の64標本mockではidentity人工kernelとmock compilerを使い、実行上限652、sample数、QPY262、全集計を確認した。full512の人工試験は4 spatial orbitals／3frameの非自明回転であり、保存H4のrank8 J/Uを評価したものではない。既存Stage B人工試験のH4-shaped一般J・spin／tiny coefficient・負event検証も再利用した。

再現コマンド（既存evaluation Python、process-local cache、bytecode/cacheprovider無効）：

```bash
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_pr_aware_df_stage_b2.py
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_pr_aware_df_stage_b.py tests/management/test_research_c_c1_h4_strong_baseline.py tests/management/test_research_c_c1_stage3_feasibility.py
```

[passive preflight](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/technical/passive_preflight.json)はSOURCE47、saved inputs22、元input、J/U/x/jac header、来歴・環境版をPASS。`numpy.load`を拒否する検査下でreal-array decode0を確認した。Stage A/B1の175保存ファイルと旧数値SOURCEが起点commitから不変、新run01は存在しない。[技術STOP](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/technical/STOP.json)を保存した。

## 8. Q1〜Q5と次の判断

**Q1：公平な比較接続は準備済み。** 同一Gaussian／fusion／P0で新しい2 CDFを別identity・別sampling lawとして扱う。実単価はまだない。

**Q2：Full controlled検証経路は完成。** 独立CARの全512 operatorとraw/compiled/actual adjoint/X/Yを比較でき、人工full512でPASSした。実H4／全lawの科学的資格は別。

**Q3：得られるのはfinite-bit校正の点推定。** 32標本のmean／variance／SE／観測範囲とその比較まで。Ideal母平均・保証CI・support・law誤差・全law合成上界は未資格でnull。

**Q4：初回案は32 trajectory/arm、64 kernel＋3shell、67 P0 compile、最大652 calls。** 128補助adjointと26までのbasis/relative要求を別明記し、標本数とbuild数とlogical call数を混同しない。

**Q5：Stage Cの検討価値を判断する材料になる。** B1のattenuation/RTE改善に対し、Proposedのhead／Gaussian／whole-kernel価格が増えるか減るかを観測できる。単価増だけで失敗、単価減だけで総資源優位とはしない。Stage Cには全law合成誤差、implemented-law差、費用support・不確かさ、13roundのshots／共通CAの別資格が残る。

残る実装blockerは確認されていないが、実H4 native correctness／価格は未検証である。研究C／ユーザーが固定652-call契約の実行可否を判断し、別途明示許可後に限りCodexがfresh run01で一度実行する。今回、重要科学レビュー・Stage C・追加fitへは進まない。Commit・push・remote SOURCE／成果物照合後、**mandatory STOP**する。
