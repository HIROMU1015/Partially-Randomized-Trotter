# C1 Stage 1：ffsim／SciPy互換性修正と人工検証（2026-10-25）

**batch logm非互換とvenv起動問題を技術的に修正した。人工59 test＋29 subtestsは全PASS。科学実行は未承認のままmandatory STOPする。** 実H4 fit・Hamiltonian／norm／residual／energy・保存値の数値再解析・実回路／compileは今回0件。Stage 1の科学的資格、C-DF/X-DFの物理誤差、実Rは依然未取得である。

## 固定点と原因

起点は `2ecf281567cf619adf1268b78d64947caa363a25`。[前回報告](research_c_c1_stage1_execution_20261025.md)、run02のtraceback、固定ffsim v0.0.83 commit `558d3517385778c733f1aa226bb61b8f30f4b0f3` とSciPy1.14.1のSOURCEを照合した。ffsimの `linalg/util.py:207` は初期回転の `(R,n,n)` を `scipy.linalg.logm` へ渡すが、同SciPyの `_matfuncs.py:53` は2次元以外を拒否する。`df_tensors_to_params` のx0生成中に例外となり、`scipy.optimize.minimize` より前で止まる。人工rotationでも旧helperの同じ例外を再現した。分子入力を演算して原因を推測していない。

新実装SOURCE commitは **`303f73275f133bfed0a0cd93a06f4a46f8ffd7be`**。branchは `codex/research-c-c1-stage1-compatibility-20261025`、独立worktreeは `/tmp/prt-c1-stage1-compatibility-20261025`。[SOURCE manifest](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/SOURCE.json)は実装・tests・依存lock等25ファイルを同commitへbindingする。

- SOURCE manifest SHA256：`e7bf161d2ab45a78d772c960ab5c2c41d3ed145bfd0a1193ee280536c1547240`
- [新しい未承認科学契約](research_c_c1_stage1_compatibility_contract_unapproved_20261025.json) SHA256：`e89f8bb323efaf5cfd72a2f8c35b5462fc0ff456b194e4dcd2f0bbc08c0310e8`
- input SHA256：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`

## 選択した最小修正と再利用

| 案 | 科学的意味・数値・依存への影響 | 判断 |
|---|---|---|
| SciPy1.16.0へ固定更新 | 公式logmはbatch対応。Python3.11.1／NumPy1.26.4は公表minimumを満たす。ただしSciPy wheel全体が変わり、linalg／optimizeを含めた再資格が必要。結果が旧版と一致するとは確認していない | 今回不採用、インストール・科学比較なし |
| SciPy1.14.1を保持し、2次元logmを順に呼ぶbridge | 同じ既存logm kernel・既定引数・principal branchを用い、ffsim既存の実部取出し／packingへ返す。全package版・wheel hash、objective・solverが同じ | 採用、人工検証PASS |

比較の一次資料は [SciPy1.14.1 logm](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.logm.html)、[SciPy1.16.0 logm](https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.linalg.logm.html)、[1.16.0 release notes](https://docs.scipy.org/doc/scipy/release/1.16.0-notes.html)。最小影響という採否は、このAPI差と依存変更範囲からの技術判断であり、SciPy版間の精度優劣を測定した結論ではない。[機械可読比較と固定SOURCE hash](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/compatibility_decision.json)。

新設の [c1_ffsim_compat.py](../../src/trottertracks/c1_ffsim_compat.py) はbatch順に `scipy.linalg.logm(mat)` を呼び、結果を `np.stack` する。float／complexを勝手にcastせず、正規直交化・対称化・branch変更・追加toleranceを行わない。元の `ffsim.linalg.util.antihermitians_to_parameters(..., real=real)` を再利用するため、`real=True` で虚部を捨てる元仕様も保持する。

専用1-fit worker内だけでffsimの `unitaries_to_parameters` symbolを一時的に接続し、成功・例外とも `finally` で復元する。SciPyのglobal関数・installed source・既存venvは書き換えない。他のffsim呼出しと同じprocessで並行利用する汎用service用ではない。元のffsim double_factorized、modified Cholesky初期化、cached JAX目的関数／勾配、SciPy optimizerをそのまま1回呼ぶ。新fitterは作っていない。

既存 `c1_stage1.fit_once` のdefault entryをbridgeへ接続し、export/runtimeにcompatibility版を記録した。既存κ、一般K adapter、独立CAR／OF、保存XDF、allowance、評価pipelineの数値処理は変更していない。[既存SOURCEとの差分記録](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/implementation.patch)。新moduleと新testは同実装commitで追加した。

[runner](../../scripts/run_research_c_c1_stage1.py) はvenv Pythonを `.resolve()` せず `.absolute()` で渡す。base Pythonに解決してNumPyを見失う旧起動を防ぐ。承認scopeを `C1_STAGE1_P_H4_ONE_FIT_COMPAT_V2` へ更新し、旧grant・旧exportを新実行へ流用できない形にした。

## 人工テストと環境

[テスト原本・コマンド](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/tests/)、[technical summary](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/technical_summary.json)。localでの人工検証をcommitに固定した記録であり、immutable CI／外部再現／分子fit成功を意味しない。

| 検証 | 結果 |
|---|---|
| 新bridge＋既存fit環境 | 19 PASS。batch形状／不正形状、逆順・非連続入力、float32/64・complex64/128、既存2D logmとのdtype／値一致、real=True/False packing、例外後復元 |
| **実ffsim初期値生成**（上記19件内） | 人工PSD tensorのnorb2／4で実Cholesky→batch logm→実packing→x0を通過。J/Uから独立に並べたx0と一致。L-BFGS-B／jac=True／options={}等を確認。optimizer入口をmockで2回遮断、目的関数実評価・optimizer実行・fit完了はいずれも0 |
| venv起動（同上） | CLIから渡すsymlinkをmock境界で捕捉。その同じpathで実subprocessを起動し、専用venv prefixとNumPy1.26.4 importを確認 |
| 既存Stage1／execution | 27 PASS。人工係数・κ・scalar・spin順序、独立reference、allowance、mock export／STOP／承認境界を再確認。旧scope拒否を検証 |
| 既存一般K adapter | 13 PASS＋29 subtests PASS |
| 合計 | **59 test＋29 subtests PASS、fail/error/skip0** |

旧testの「science directoryが存在しない」は、保存run01/run02が追加された起点には適用できないため、「旧approval scopeが新実行を承認できない」検査へ置換した。保存結果の数値再解析でPASSを作っていない。

[新専用環境](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/environment/)をCPython3.11.1で2つ新規作成し、旧wheelhouseを読取り専用の取得元として `--no-index --require-hashes` でインストールした。fit22／eval41 wheelのhashをlockへ照合、pip check両方PASS。新lockは旧lockとbyte同一で、ffsim Python102ファイルは固定commitに全一致。SciPy `_matfuncs.py` hashも旧原因監査と同じ。旧venv・共有環境を変更していない。

利用可能32 CPU、MemAvailable約52.3 GiB、memory pressure0、GPU driver利用不可を確認した。[資源probe](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/resource_probe.json)。小さい人工行列を既存BLAS/JAX/sector経路で評価し、独立fit/eval suiteは並行実施した。恣意的なCPU/RAM/worker/wall capを追加せず、speedupの主張はしない。

## 新契約、保全、残る承認

科学定義はP-H4（直鎖H4・1 Å・STO-3G、保存canonical MO）、Nα=Nβ=2/sector36、必要base／各frameはfull256、通常C-DF number／元g由来κ保持、max_vecs8／tol1e-8／L-BFGS-B、保存rank10 XDF。一般Kを元の1 frameとして保持する。L_D／delta窓はStage 1非該当。物理CDF／XDF誤差を別測定し、adapter同値性と別判定、物理誤差の絶対閾値なし、Stage 2適格性は別判断のまま。

数値allowance `binary64_operation_scale_guard8_v1` は限定pilotへの採択を保持し、式・演算長・guard8・norm係数32・precision・thresholdに変更なし。厳密certificateへ格上げしない。[契約の科学部分同一性](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/contract_comparison.json)に、入力・科学scope・solver・call・reference・STOPを照合した。R回の2D logmは1 fit内部の初期化処理で、計上規則は成功経路 `83+8R`、最大147、fit1、science retry/resume0のまま。

[読取り専用preflight](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/preflight.json)はSOURCE25ファイルと実装commit、入力bytes、契約／manifest hashのPASS。NPZの数値decodeをせず、grant/outputを生成していない。旧artifact1316ファイルを文書更新前にGit blob照合し、既存科学証拠1315件はそのまま保全した。現行validation manifestだけへ新しい技術記録を追記し、過去のentryは保持した。元workspaceのbyte／diff hashも照合した。[保全記録](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/preservation.json)。旧SOURCE・契約・run01/run02・grant・ledger・marker・STOPと旧科学結果は変更していない。

確認した技術blockerは解消した。ただし実分子fitの収束、他の実行時問題、物理誤差・adapter gateの成否は人工初期化検証から保証できず、未検証として残る。

- **ユーザー／統括（次の判断）**：公開commit、新SOURCE／契約hash、固定input、fresh outputに紐付けた新しい1 fitの科学実行を明示承認するか決める。旧fit枠は消費済みで、再利用・resumeしない。
- **Codex（その承認後のみ）**：preflight・資源状態を再確認し、別approval／fresh outputでStage 1を契約どおり1回実行する。科学FAIL／契約不一致は証拠保全してSTOP。
- **ユーザー／統括（Stage 1完了後の別判断）**：別測定されたDF誤差を評価し、Stage 2のscope・精度条件を採択する。Stage 2/3は未承認。

今回の完了操作はcommit／push、remote SHA／SOURCE／成果物照合まで。科学dispatchを有効化せず、mandatory STOPする。
