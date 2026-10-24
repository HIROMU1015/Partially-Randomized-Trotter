# 研究C・C1 Stage 2：native回路接続・科学実行準備（2026-10-25）

Stage 2の最小接続SOURCE、人工検証、専用環境と未承認科学契約を準備した。**人工80 tests＋29 subtests PASS、新規Stage2 testsは27件。今回の実H4数値decode／operator・signal評価／実回路build・compile／新fit・optimizer／科学callはすべて0。mandatory STOP。**

ユーザーは保存C-DF rank8をStage2の限定対象として採用した。Stage1で測定した物理sector誤差はCDF `7.666027761392501e-5` Ha、保存XDF rank10 `3.3751558354542566e-15` Haである（[既存結果](research_c_c1_stage1_eval_only_execution_20261025.md)の引用、今回再解析なし）。精度が異なるため、今後のnative回路費用の比較を同精度の総資源優位とは扱わない。

作業中に提示された2026-10-25版の運用指示を適用し、新Stage2のapproval file／照合処理とtemplateは除去した。チャットでの明示指示を科学実行の許可として扱う。旧Stage1の正式契約・approval・grant・markerは変更せず、SOURCE／入力／call上限／fresh output／retry／STOPも保全する。

## 固定SOURCEと来歴

| 項目 | 今回の固定／準備 |
|---|---|
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| 起点 | `b5fa60f9b682a93fda417452b5d85ceb1527e550` |
| 独立branch | `codex/research-c-c1-stage2-preparation-20261025` |
| worktree | `/tmp/prt-c1-stage2-preparation-20261025` |
| 実装commit | `d8d54b0d4a78a25d8a2b88ac2a6d0c6833c394db` |
| SOURCE manifest | [SOURCE.json](../../artifacts/research_c_c1_stage2_technical/2026-10-25/SOURCE.json)、136ファイル |
| SOURCE SHA256 | `0b659088a817a732969a930341bf32cda744034e8108d99725e7677d32b312e6` |
| 契約 | [未承認契約](research_c_c1_stage2_native_contract_unapproved_20261025.json)、scope `C1_STAGE2_SAVED_R8_NATIVE_LIMITED_V1` |
| 契約SHA256 | `b25a8b3f1611f0b90c1d9706ceea2e6b03e3d6cb86ba45cda15faaf5cae4ac98` |
| 推奨条件のSHA256 | `0c484c98ffdf4e287b52708cc50b45c14022e4010d3beb2ad684f0970a79e3be`。**未採択** |
| 元fit SOURCE | `da2edb465d5ae6432439fbd1df8e57512ee59c11` |
| 保存export SHA256 | `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| 元P-H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| Stage1評価SOURCE / 結果 | `bec41ac7dc7e034bd664817acfdcfed858ddadd0` / 起点b5fa60f9…の37 gate PASS |
| 物理系 | 保存直鎖H4・隣接1.00 Å・STO-3G、4空間／8 spin軌道、Nα=Nβ=2（sector36） |
| Stage2検証範囲案 | system full256、ancilla-last controlled full512。sector DF誤差の再計算なし |
| 表現 | CDF保存実rank8、number／元g補正κ／元frame order、XDF保存実rank10 |
| L_D / delta窓 | 分割PR／PF誤差窓は非該当。下記Tは回路接続用ordered targetの1点案 |

今回の来歴確認はbytes／JSON宣言だけで、保存J/U／元input／Stage1の行列を数値decodeしない。[数値importを禁止したpreflight](../../artifacts/research_c_c1_stage2_technical/2026-10-25/preflight.json)はSOURCE136、旧来歴25ファイル、元input hash、環境52 pinsをPASSした。[既定CLI](../../artifacts/research_c_c1_stage2_technical/2026-10-25/cli_preflight.txt)も科学実行せずPASS。SOURCEの既存部分、旧contract・grant・結果・ledger・marker・STOPは変更していない。

## 実装した接続と再利用

新SOURCEは [c1_stage2_native.py](../../src/trottertracks/c1_stage2_native.py)、[c1_stage2_validation.py](../../src/trottertracks/c1_stage2_validation.py)、[c1_stage2_execution.py](../../src/trottertracks/c1_stage2_execution.py)、[専用runner](../../scripts/run_research_c_c1_stage2_native.py)。新fitter、Gaussian合成器、数値計算libraryは実装していない。既存SOURCEを変更せず、以下を接続した。

| 再利用する既存資産 | Stage2での接続 |
|---|---|
| `c1_general_k_adapter` | 保存J/U／補正κから既存GeneralKModel。一般KのI/Z/ZZ・中央phase係数を再利用 |
| `df_trotter.ops` / `decompose.diag_hermitian` | one-bodyとXDF Gの既存eigh、既存one-body／square対角primitive。Hermitian射影や再分解なし |
| `circuits.native_gaussian` | 既存Nature0.7.2 Givens helper、dense Gaussian fallback禁止 |
| `append_diagonal_primitives` | 既存controlled RZ/RZZおよびancilla PhaseGate。relative scalarを保持 |
| `circuits.actual_adjoint` / `hadamard_wrapper` | 実際にbuildした回路のinverseと、ancilla-last測定付きX/Y wrapper。Stage2用の時刻metadataを薄く接続 |
| `gates.controlled_gate` / `wrapper_signal` / `signal_gate` | full controlled matrix、control0 identity／control1相対phase、unitarity、StatevectorによるX/Y検証 |
| `gates.compiled_wrapper` / `fixture.CompilerProfile` | 既存の位相保持definition展開と固定P0 profileへの接続。共有equivalence libraryの変更なし |
| `f2_execution.circuit_metrics` | 既存RZ/CX数・各depth・総depth・size・measure・quantum-only depth |
| Stage1独立CARと保存raw行列 | CDFは保存base＋8 frameの9行列を再利用。XDFはStage1にfull frameがないため、元G／λ／oneを既存CARへ縮約する11 generatorだけを将来構築 |
| `c1_stage1_execution.load_export` / journal | 元fit SOURCE／metadataを保持して再利用。新fit worker／optimizer経路なし |

B=Γ(V)とすると、回路の**適用順序**は `B.inverse → controlled diagonal → B`。control1で `B exp(-iτD) B†`、control0でidentityを目指す。Gaussian自身のbasis-only global phaseは往復で相殺し、中央I係数はancilla phaseとして保持する。modelの核scalar cはfull evolution全体に `p(-cT)` を1回だけ付け、局所one-body検査ではκ+cを対象にする。frameのI項をcへ移し替えない。

CDFは元の1 frameを分割しない。XDFは保存signed λとG順序を保ち、既存square emitterをそのまま使う。one-bodyはκを再補正せずspin-lift済みデータを既存eighへ渡す。どちらも既存のdescending_abs eigenbasis規則を使用するが、outer frameの係数順ソート・並べ替えはしない。

one-step案にはstep間境界がないため、その融合の利益を測る設計ではない。隣接同basisの機会が実データにあるとは仮定せず、人工testで共有・融合規則を確認した。

literalとsharedの双方へ同じ中央primitiveを使う。sharedでは**隣接し、rotationの全要素がexact一致**する場合だけbasis往復を共有し、同basis内のphase/RZ/RZZ角を `math.fsum` で結合する。異basisを跨ぐ融合、近似一致による同一視、角度のmod2π化、科学cutoffによるpruningはしない。exact-zeroだけを省く。basis cacheはsource／input／rotationへbindingし、回路上のgate費用はcacheによって消えない。

## 人工・mock検証と環境

[人工tests SOURCE](../../tests/management/test_research_c_c1_stage2_native.py)、[最終regression出力](../../artifacts/research_c_c1_stage2_technical/2026-10-25/regression.txt)、[JUnit](../../artifacts/research_c_c1_stage2_technical/2026-10-25/regression.xml)。新規27件に既存一般K／Stage1 CAR／frame修正／評価専用testsを加え、**80 tests＋29 subtests PASS**。新SOURCEを追加した後の一式を検査した。

- 人工一般Kの対角occupation位相、係数の両順・signed square、τの正負、model scalarとframe scalarの所有を確認。
- 既存diagonal emitterにmock circuitを渡し、Gaussian inverse／forward、ancilla位置、RZ/RZZ・phaseのdispatchを確認。
- 人工rank8／rank10とcomplex rotationでspin順序、outer order、12 eigh要求をmock確認。実eigh／Gaussian synthesisは呼ばない。
- exact basis共有、異basis fence、tiny angle保持、source別cache、actual inverse、測定付きX/YのSdgとmetadataを確認。
- operator／signal／compiler／metricの既存APIへmockで接続。全40局所gate後にfull、全raw gate後にcompileへ進むこと、最初の失敗で依存callが出ないことを確認。
- fresh output／namespace／SOURCE driftの科学的管理と、人工load失敗／KeyboardInterruptのclaim・prefix台帳・STOP保全を確認。許可はチャットで受け、approval fileや証明照合は実装しない。

最初の人工検査には、`-0.020000000000000004` を `-0.02` とexact比較したtest側の失敗が1件あった。適切な近似比較へ修正し、その原本も `initial_artificial.*` に保持した。物理条件や科学allowanceの変更ではない。

実QuantumCircuit／Operator／Statevector／transpileはmockまたは禁止constructorで遮断した。人工test PASSはNatureの分子Gaussian conventionや実H4回路の科学的PASSを意味せず、今回その資格を認定しない。

専用 `.venv`（CPython3.11.1）を新規作成した。[52 package hash lock](../../artifacts/research_c_c1_stage2_technical/2026-10-25/environment/requirements.lock)は既存Stage1 eval41 pinsを保ち、不足するQiskit関連11 pinsを既存F1 lockから補った。NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Nature0.7.2／Algorithms0.3.1／Aer0.15.1。Stage1の共有・既存venvは書き換えず、保存wheelをread-only再利用し、不足wheelはhash必須でtask内へ取得した。[runtimeと禁止constructor付きimport smoke](../../artifacts/research_c_c1_stage2_technical/2026-10-25/environment/runtime.json)、[pip check](../../artifacts/research_c_c1_stage2_technical/2026-10-25/environment/pip_check.txt)はPASS。venv Pythonのsymlinkを解決してbase Pythonに置換する処理は使わない。

## 未採択の科学条件：合理的な初回推奨案

**以下は推奨案であり、今回採択・実行していない。** CDF rank8を対象にする判断だけはユーザーが既に採択済み。

| 条件 | 推奨案と目的 |
|---|---|
| 時間 | T=1/10 atomic unitsの1点。局所τ=1/20、1/10を全20 blockで検査し、S2に現れる半時間・全時間を覆う |
| PF | one-step ordered symmetric S2。chronological `H0/2,F0/2,…,F(R-2)/2,F(R-1),F(R-2)/2,…,F0/2,H0/2`、H0はκ。scalar cは別途1回 |
| 参照対象 | 上記ordered productそのもの。`exp(-iT Hphys)`やPF誤差は計算せず、異なる近似表現の物理差も再評価しない |
| route | 同じpolicyのliteral／sharedを両armに適用。primary比較はshared、literalは合法融合の対照 |
| controlled / adjoint | system8＋ancilla-lastのfull512。control0 identityとcontrol1の位相をphase fitなしで比較。actual adjointはbuilt circuitのinverse |
| X/Y | computational occupied(0,1,2,3)、Nα=Nβ=2。ground state準備はせず、独立ordered productの対角要素を複素signalとする |
| compiler | Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、free topology／layout identity。既存P0と同じ案 |
| 数値許容差 | operator max-entry絶対1e-10、unitarity1e-10、paired complex X/Y絶対2e-10、rtol0。既存C4/F2 gate値を初回案に再利用し、**採択は未了** |
| reference精度 | complex128／SciPy1.14.1 expm。追加高精度reference・厳密誤差certificateは含めない |
| 上限 | 下記243 logical units、fit／optimizer0、retry／resume0、最初のgate FAILでSTOP |

Stage1の `binary64_operation_scale_guard8_v1` はStage1記録のまま保持する。Stage2のunitary／signal比較へ無断で厳密証明として転用せず、新しい比較の固定絶対allowance案として上記を別採択してもらう。値やprofileを変える場合は、実行前に契約・SOURCEとの整合を更新して再固定する。実行後のthreshold調整は禁止する。

## 科学call案と実行順序

実装の `call_limits()` と[契約](research_c_c1_stage2_native_contract_unapproved_20261025.json)は一致する。今回は実call0。将来の上限は以下の**243 logical units**であり、Stage1の141を再利用する枠ではない。

| 計上項目 | 上限 |
|---|---:|
| `input_load` | 1 |
| `export_load` | 1 |
| `saved_operator_load` | 9 |
| `car_context` | 1 |
| `xdf_generator` | 11 |
| `model_prepare` | 2 |
| `gaussian_build` | 20 |
| `local_reference` | 40 |
| `local_build` | 40 |
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
| `compiled_signal` | 8 |
| `compiled_signal_pair_check` | 4 |
| `metrics` | 12 |
| **合計** | **243** |

CDF9 block＋XDF11 block×2局所時間＝40局所case。Gaussianは既存Givens生成をbasis cacheごとに追加計上し最大20、同basisなら実績は下がり得る。model_prepare2にはCDF one-body1＋XDF one-body/G11、計12回の既存eighが含まれる。XDF generator11だけが未保存full referenceの新CAR構築である。

local／ordered／adjoint／compiledのcontrolled checkは計52。各checkは既存gateのOperator1回、phase-sensitive equalityとunitarityのmatrix_gate2回（各max-entry／Frobenius／spectral診断）を含み、合計52 Operator変換・104 spectral診断を明示する。local reference40は256次元expm。ordered reference2はそのcacheを左から時系列に掛けるだけで、full Hのexpmはしない。raw8＋compiled8のStatevectorでX/Yを測り、4＋4組の複素signal gateを判定する。compiled measured wrapper全体のOperator一致までは今回の案に含めず、compiled unmeasured controlled fullの完全operator一致と、実measured X/Y signalを検証する。

12 compileはfull4（2arm×2route）＋measured wrapper8。old compile cacheや別armの価格を流用せず、余分なcold/warm反復はしない。RZ count／CX count／RZ depth／CX depth／total depth／size／measurement count／quantum-only depthの8指標をそれぞれ保存する。full回路とstate prep・measurementを含むwrapper費用を別に表示し、law・shot加重や総資源へ外挿しない。

順序は、チャットでの科学条件採択・実行許可 → byte／環境／fresh output確認 → 保存データの許可whitelist読込 → 全40局所controlled gate → ordered full4＋actual adjoint4 → raw X/Y → full compile4と完全operator gate → wrapper compile8とX/Y → 費用・証拠保存・mandatory STOP。局所gateを通る前にordered full回路をbuildしない。最初のFAILで依存する後続callを止め、台帳・既存QPY・partial STOPを保持する。新fit、retry/resume、許容差調整はしない。

成功時はraw local40、full4、adjoint4、wrapper8、compiled12の計68 QPYを保存する。費用指標とgate原本はresult、prefix callはjournal、技術／科学的失敗はfailure／STOPに保存する。将来の実行は、ユーザーがチャットで条件を採択し実行を許可した後に `--execute --output <fresh絶対path>` で行う。approval記録・ID・authority・認証・証明packetは必要とせず、生成もしない。実行時に科学契約・SOURCE・環境・call・結果・one-shot markerを保存することは、再現性と科学的scopeの管理であり、許可の証明ではない。

## 資源・保全と未解決事項

[技術資源記録](../../artifacts/research_c_c1_stage2_technical/2026-10-25/resources.json)にavailable affinity32 CPUとRAM／pressureを保存した。重い将来の計算では、既存batched CAR／BLAS／native compilerを利用し、CDF9行列再構築、Gaussianの重複生成、ordered targetのexpm再計算を避ける。gate順序と最初のFAILで即STOPを保つため、外側dispatchは逐次とし、dense kernel／compiler内部には利用可能CPUを使う。投入前に空き資源・aggregate RAM・thread競合を確認し、実行監視でGNU time／process tree等の使用量を保存する。恣意的なCPU／RAM／worker／wall capは設けない。共有環境・他ユーザーのjob/fileへの変更や性能speedupの断言はしない。

[保全記録](../../artifacts/research_c_c1_stage2_technical/2026-10-25/preservation.json)と[evidence inventory](../../artifacts/research_c_c1_stage2_technical/2026-10-25/evidence_inventory.json)を保存する。固定SOURCE／旧科学結果・grant・ledger・marker・STOPは保全し、現行概要・索引・dated note・validation manifestには今回の技術準備と限定対象採択を追記する。commit／push後にremote SHA、SOURCE、契約、成果物bytesを照合する。公開されたローカル人工証拠であり、CI／外部独立再現や分子回路資格ではない。

技術的blockerは人工／mock範囲では解消済み。**実H4のGaussian向き・native回路・compiled correctnessは未検証**であり、予定された科学gateで確認する事項である。C-DF物理誤差の新たな合否閾値、PF誤差、RTE、shot、同精度総資源の判断は今回の対象外。

- **ユーザー／研究C（次の判断）**：T／local時間、S2・順序、occupied、compiler profile、数値allowance、243-unit範囲の推奨案を採択するか、変更を指定する。その後、公開SOURCE・契約・保存export／input・fresh outputに対してStage2限定科学実行を別途明示承認する。
- **Codex（上記の別承認後）**：preflightを行い、契約順にStage2限定検証を1回実行して証拠を保存・公開しSTOPする。現時点では実行しない。

**今回の技術準備は完了。mandatory STOP。Stage2実回路生成・compile・科学実行へ自動進行せず、Stage3にも進まない。**
