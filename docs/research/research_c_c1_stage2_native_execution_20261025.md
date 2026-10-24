# 研究C・C1 Stage2：native回路の限定科学実行（2026-10-25）

**Stage2の限定検証は正常完了。主検査60件と内包するunitarity52件がすべてPASSし、FAIL／NOT_RUNは0。** 科学call243／最大243、新fit・optimizer・retry・resumeは各0、実transpile12、68 QPYを保存した。`STAGE2_LIMITED_CIRCUIT_VALIDATION_PASS` とmandatory STOPを記録し、Stage3には進んでいない。

この条件でのcompiled full費用はCDFがXDFより小さい。ただし保存CDF rank8とXDF rank10の物理近似精度は異なり、**同精度の総資源優位という結論ではない**。literal／sharedの全費用指標は各arm内で一致し、このone-step案で共有・融合による追加削減は観測されなかった。

## 実行条件・SOURCE・再利用

ユーザーが前回提示した推奨方針の採択・実行に同意し、技術的な中断時の修正・再実行も許可したため、固定Stage2限定範囲を実行した。許可はチャットの指示として扱い、承認証明ファイル・authority・grant発行・認証は追加していない。今回中断はなく、契約の科学retry／resume0をそのまま維持した。準備時点の契約原本にある未採択／未承認表示は歴史的snapshotとして保持し、今回の実行条件を[preflight](../../artifacts/research_c_c1_stage2_science/2026-10-25/preflight.json)と本結果に示す。

| 項目 | 今回の固定条件 |
|---|---|
| Repository / branch | HIROMU1015/Partially-Randomized-Trotter / `codex/research-c-c1-stage2-science-20261025` |
| 公開実行SOURCE | `a977a4390668bc107360c8ff05f43d851dc81dc9`、科学SOURCE変更0 |
| 実装commit | `be4f299470fd896c4bfd50b11bb87bf17cd760b2` |
| SOURCE manifest | [原本](../../artifacts/research_c_c1_stage2_technical/2026-10-25/SOURCE.json)、SHA256 `63cff2a63bb49581e44c437f412af63c3b70c328439d1e12cd35c2551aeca8e0` |
| 契約 | [原本](research_c_c1_stage2_native_contract_unapproved_20261025.json)、scope `C1_STAGE2_SAVED_R8_NATIVE_LIMITED_V1` |
| 契約SHA256 | `d770a8fc6e2c3281fee1cad8d73f64abd939a0030ed4963776356fd84befcbbc` |
| 元P-H4 input | SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、保存snapshot commit `86b8b002704aced8ca239414080f53793de81658` |
| 元fit SOURCE / export | `da2edb465d5ae6432439fbd1df8e57512ee59c11` / SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| 分子 / basis | 直鎖H4、隣接1.00 Å、STO-3G、保存実canonical RHF MO、4空間／8 spin軌道 |
| sector / full | 物理sector Nα=Nβ=2（36次元）。今回のunitary検証はsystem full256／ancilla-last controlled full512 |
| CDF / XDF | 保存CDF実rank8、通常number表現・元g補正κ・元outer order・一般Kを元の1 frameとして保持。保存XDF実rank10、signed λ／元G順序 |
| T / PF | T=1/10 atomic time units、one-step ordered symmetric S2。適用順序κ半分→F0半分→…→F(R−2)半分→F(R−1)全時間→逆順半分→κ半分。model scalar cは全体に1回 |
| 局所時間 | τ=1/20、1/10をone-body+c／全frameの20 blockへ適用、40 case |
| signal状態 | computational occupied(0,1,2,3)、Nα=Nβ=2。ground-state oracleなし。X/YをStatevectorから計算しsamplingなし |
| compiler | Qiskit1.3.0、opt1、seed17、native rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、free topology・layout identity |
| 数値判定 | operator max-entry／unitarity絶対1e-10、paired complex X/Y絶対2e-10、rtol0、phase fit・Hermitian射影なし |
| L_D / delta窓 | PR分割／PF誤差窓は非該当。T1/10一点のordered targetのみ |
| fresh output | `artifacts/research_c_c1_stage2_science/2026-10-25/run01` |

[既存runner](../../scripts/run_research_c_c1_stage2_native.py)と `c1_stage2_native/validation/execution.py`、既存Givens、controlled diagonal emitter、actual inverse、Hadamard X/Y、位相保持compiler、metric抽出を無変更で使った。CDFはStage1で保存した独立raw CAR base＋8 frameの9行列を再利用した。XDFに未保存だったfull one-body＋10 frameだけを元G／λ／oneから既存CARへ接続した。新DF fit、積分・SCF、物理Hやsector DF誤差の再構築はない。

Gaussianの適用はB.inverse→controlled diagonal→B。frame I項とmodel cを分離した既存phase所有を保ち、control0 identityとcontrol1相対位相を含めて比較した。actual adjointはbuild済み回路そのもののinverseである。ordered referenceは局所expmのcacheを時系列に左から掛け、物理H全体のexpmやPF誤差は評価していない。

preflightはSOURCE34ファイル、旧来歴25ファイル、input/export hash、Python3.11.1と主要5 libraryの版をPASSした。既存専用venvをread-only再利用し、cache／tempは本task内へ分離した。実行直前の[人工27件](../../artifacts/research_c_c1_stage2_science/2026-10-25/artificial_preflight.txt)もPASS。

## 全gateと数値誤差

[60主検査の原本](../../artifacts/research_c_c1_stage2_science/2026-10-25/run01/result.json)、[機械可読要約](../../artifacts/research_c_c1_stage2_science/2026-10-25/execution_summary.json)。主検査は52 controlled operator＋8 paired X/Y。各controlled checkがunitarityも内包するため、elementaryな閾値比較は104 matrix gates＋8 signal gates＝112件。表中の誤差は保存済み診断値の最大であり、追加Operator／norm評価はしていない。

| 検証群 | PASS / FAIL / NOT_RUN | 最大誤差 | 固定許容差 |
|---|---:|---:|---:|
| 局所one-body+c／frame controlled | 40 / 0 / 0 | 3.3306690738755414e-15 | 1.0e-10 |
| raw ordered full controlled | 4 / 0 / 0 | 6.3576628511461513e-15 | 1.0e-10 |
| actual adjoint | 4 / 0 / 0 | 5.4514785811733496e-15 | 1.0e-10 |
| compiled full controlled | 4 / 0 / 0 | 5.9172153482173579e-11 | 1.0e-10 |
| raw paired X/Y | 4 / 0 / 0 | 2.445012586343469e-15 | 2.0e-10 |
| compiled paired X/Y | 4 / 0 / 0 | 1.3012864077677942e-12 | 2.0e-10 |
| 内包unitarity | 52 / 0 / 0 | 4.7173376316322901e-13 | 1e-10 |

operatorは無次元unitaryの最大成分誤差で判定しており、Stage1の物理Hamiltonian sector operator norm（Hartree）とは別である。Frobenius／spectral診断も原本へ保存したが、合否規則の変更はない。compiled fullの最大成分誤差 `5.917215348217358e-11` は採択された `1e-10` 以下でPASS。結果を見てallowanceを変更していない。これは限定条件でのbinary64数値検証で、厳密certificateではない。

局所40件をすべて通過してからordered fullへ、raw full／actual adjoint／X/Yがすべて通過してからcompileへ進んだ。compiled full4件はcontrol0／control1を含むfull512で検査し、compiled測定wrapper8件は実X/Y signalで検査した。compiled測定wrapper全体のOperator照合は今回の契約外で、実施済みとはしない。

## native費用の比較

[12 case×8 metric原本](../../artifacts/research_c_c1_stage2_science/2026-10-25/native_metrics.csv)。主比較は同じshared規則を使うcompiled fullで、literalは対照。各armでliteralとsharedの8指標は全件一致し、X/Y wrapper同士の8指標も一致した。以下のwrapper行はX、Yそれぞれ独立に得た同じ値をまとめて記載する。

| arm / circuit | RZ | CX | RZ depth | CX depth | total depth | size | measure | quantum-only depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CDF full | 12118 | 7352 | 5845 | 6412 | 12724 | 23217 | 0 | 12724 |
| CDF X/Y wrapper（各） | 12121 | 7352 | 5845 | 6412 | 12724 | 23227 | 1 | 12724 |
| XDF full | 16659 | 9520 | 7431 | 8056 | 16215 | 31822 | 0 | 16215 |
| XDF X/Y wrapper（各） | 16662 | 9520 | 7431 | 8056 | 16215 | 31832 | 1 | 16215 |

compiled fullではCDFのRZはXDF比27.2585%、CXは22.7731%、total depthは21.5294%小さかった。これはこのH4／時刻／S2／emitter／P0 profileにおける構造的な回路費用差である。rank数だけへの因果帰属、最適な合成器一般、別時刻・分子への一般化、同精度の総資源優位を主張しない。

既存Stage1での物理sector誤差はCDF約7.6660e-5 Ha、XDF約3.3752e-15 Haであり、精度が揃っていない（[既存結果](research_c_c1_stage1_eval_only_execution_20261025.md)の引用、今回再測定なし）。今回のsignal referenceもそれぞれの表現によるordered targetで、共通物理targetへのbias比較ではない。PF誤差・law・RTE・shot・全round総資源は計算していない。

## 科学call・証拠・計算資源

[call集計](../../artifacts/research_c_c1_stage2_science/2026-10-25/call_summary.json)、[一回の台帳](../../artifacts/research_c_c1_stage2_science/2026-10-25/run01/science_calls.jsonl)。論理計上単位は契約と一致し、合計243／243。composite checkの内部を別試行として増やしたり、保存行列の再利用を新たなgenerator構築として数えたりしていない。

| 計上項目 | 実数 |
|---|---:|
| `input_load` | 1 |
| `export_load` | 1 |
| `model_prepare` | 2 |
| `saved_operator_load` | 9 |
| `car_context` | 1 |
| `xdf_generator` | 11 |
| `local_reference` | 40 |
| `local_build` | 40 |
| `gaussian_build` | 20 |
| `local_controlled_check` | 40 |
| `ordered_reference` | 2 |
| `ordered_build` | 4 |
| `ordered_controlled_check` | 4 |
| `adjoint_build` | 4 |
| `adjoint_check` | 4 |
| `wrapper_build` | 8 |
| `raw_signal` | 8 |
| `raw_signal_pair_check` | 4 |
| `compile` | 12 |
| `compiled_controlled_check` | 4 |
| `metrics` | 12 |
| `compiled_signal` | 8 |
| `compiled_signal_pair_check` | 4 |
| **合計／上限** | **243／243** |

新fit／optimizer／retry／resumeは各0。実行はrun01の1回だけで、技術的中断も科学的FAILもなかった。保存CDFの9行列を読み、XDF full generator11、既存eigh12（model_prepare2に包含）、局所expm40、Gaussian20、Operator52、transpile12を契約どおり実行した。結果保存後の追加科学callは0。

[68 QPYとbyte inventory](../../artifacts/research_c_c1_stage2_science/2026-10-25/circuit_inventory.json)は局所40、raw full4、actual adjoint4、raw measured wrapper8、compiled12。QPYを再decode・再compileして費用を数え直さず、今回の同じ実行で保存済みのmetricを転記した。[result](../../artifacts/research_c_c1_stage2_science/2026-10-25/run01/result.json)、runtime、contract、SOURCE、claim、STOPも保存している。新しい承認証明制度や許可text保存はない。

起動時affinity32 CPU、MemAvailable約52.4 GiB、memory pressure0、GPU driver利用不可。既存BLASのavailable32 threadsを保持し、科学的gate依存順を守る外側dispatchは逐次、cache／batched CAR／局所expm再利用／native compilerを使用した。恣意的なCPU／RAM／worker／wall capはない。[GNU time](../../artifacts/research_c_c1_stage2_science/2026-10-25/resource_time_run01.txt)はwall250.51秒、user639.63秒／system171.14秒、最大RSS691,024 KiB、swap0。[process tree監視](../../artifacts/research_c_c1_stage2_science/2026-10-25/resource_usage_run01.json)はwall約251.32秒、1秒間隔251 samples、aggregate RSS最大690,768 KiB、thread数最大64。両者の測定範囲を同一視せず、thread数を同時使用CPU数とはしない。性能speedupの比較はしていない。

## 保全・公開と次の判断

旧SOURCE／Stage1のfit・結果／grant／ledger／marker／STOP、準備時点の契約、元workspaceと共有環境を保全した。今回のSOURCE・条件・allowanceは無変更で、現行索引・研究概要・dated note・validation manifestに今回の完了を追記する。結果・QPY・数値診断をcommit／pushし、remote SHAと成果物を照合する。これは公開保存したローカル科学実行で、独立した外部再現やCIによる資格ではない。

[STOP原本](../../artifacts/research_c_c1_stage2_science/2026-10-25/run01/STOP.json)はmandatory_stop=true、new_fit／retry／resume0、Stage3_authorized=false。

- **ユーザー／研究C（次の判断）**：今回のStage2接続PASSと構造的費用差を踏まえ、CDF物理誤差をどう全誤差予算へ組み込むか、同精度baselineをどう設定するか、Stage3接続準備へ進めるかを判断する。
- **Codex（別途依頼後の作業案）**：採択された誤差予算・比較条件でStage3の接続設計／実行条件を準備する。新fit・PF誤差評価・PR/RTE・sampling・shot／総資源評価は今回から自動実行しない。

**Stage2限定実行は完了。公開照合後もmandatory STOPを維持し、Stage3へ進まない。**
