# C1 Stage 1限定科学実行：初期化中の技術的STOP（2026-10-25）

**Stage 1は未完了。固定ffsimとSciPyのAPI非互換により、承認済みのfit 1回が初期化中に停止した。** optimizer・Hamiltonian比較へ到達しておらず、C-DF/X-DFの物理sector誤差と一般K同値性は未測定である。科学的な否定結果やadapter FAILとは扱わない。call台帳はinput load1＋fit attempt1の**2計上単位**、fit再試行・resumeは0。追加fitを行わずmandatory STOPした。

[実行summary](../../artifacts/research_c_c1_stage1_science/2026-10-25/execution_summary.json)、[部分原本run02](../../artifacts/research_c_c1_stage1_science/2026-10-25/run02/)、[STOP](../../artifacts/research_c_c1_stage1_science/2026-10-25/run02/STOP.json)、[固定SOURCE](../../artifacts/research_c_c1_stage1_science/2026-10-25/run02/SOURCE.json)、[契約原本](../../artifacts/research_c_c1_stage1_science/2026-10-25/run02/contract.json)、[明示承認記録](../../artifacts/research_c_c1_stage1_science/2026-10-25/approval_run02.json)を参照する。保存state/energy/過去の科学的PASSを代わりのnorm値に使っていない。

## 1. 承認と固定条件

| 項目 | 今回の条件 |
|---|---|
| Repository / 固定公開SOURCE | `HIROMU1015/Partially-Randomized-Trotter` / `de76aee72f35321c00155b0b17d22ef6eb2bcb9a` |
| 実装基準 | `394fcf3a445caf0a5f206b7bbcf4a8c88f971504`、既存ffsim/adapter/CAR/sector処理を再利用、変更なし |
| 専用branch / worktree | `codex/research-c-c1-stage1-science-20261025` / `/tmp/prt-c1-stage1-science-20261025` |
| 固定契約 | `research_c_c1_stage1_science_contract_unapproved_20261025.json`。SHA256 `ab20dcd476610b2c65956e51d4edc3a0152d48fc546aa0605744c8c724d8cb2c` |
| SOURCE manifest SHA256 | `dadd5f25c813144f1683f0b2fc3d5454b336638da722116735ae955172e43a7c` |
| 元input | P-H4、commit `86b8b002704aced8ca239414080f53793de81658` の `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz` |
| input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 物理系 | 直鎖H4、隣接1.00 Å、STO-3G、charge0/spin0、保存canonical RHF MO、4空間/8 spin軌道 |
| 比較範囲 | Nα=Nβ=2、sector36。adapter baseと各frameはfull256でも確認する計画 |
| C-DF | ffsim v0.0.83、通常C-DF number表現、元gから補正したκ保持、元frameを一般K 1 frameとして保持 |
| fit | `max_vecs=8, tol=1e-8, optimize=True, L-BFGS-B, options={}`、1回のみ |
| X-DF | 同一元input由来の保存実rank10、再分解なし。今回は評価未到達 |
| 数値allowance | `binary64_operation_scale_guard8_v1` をユーザーが今回の限定pilotだけに採択。厳密誤差証明ではない。変更していない |
| 誤差政策 | CDF/XDF物理sector normを別報告、絶対DF thresholdなし、Stage 2適格性は別判断 |
| split/時間 | `L_D`、delta窓、PF/RTE条件はStage 1非該当 |

ユーザーの新しい明示承認をfresh outputへ結び付け、専用approvalを保存した。凍結済み契約やSOURCE内の当時の「未承認」表示は履歴として保持し、その原本を書き換えて承認に見せていない。今回の承認は別記録により固定hash・input・output・one-fit・allowance版・STOPへ結び付けた。

## 2. preflightと環境

[SOURCE/input/contract preflight](../../artifacts/research_c_c1_stage1_science/2026-10-25/preflight_run01.json)、[fit環境](../../artifacts/research_c_c1_stage1_science/2026-10-25/fit_preflight_run01.json)、[評価環境](../../artifacts/research_c_c1_stage1_science/2026-10-25/eval_preflight_run01.json)はPASS。固定SOURCE22ファイルとinput byte hash、契約hash、SOURCE manifest hashを確認した。fit環境のffsim Python102ファイルも固定commitに一致した。

既存のtask専用venvを再利用し、共有venv/cache/packageは変更していない。fit側はCPython3.11.1、ffsim0.0.83、SciPy1.14.1、NumPy1.26.4、JAX/jaxlib0.4.38、JAX x64=true、CPU backend。評価側は固定OpenFermion1.6.1/NumPy1.26.4/SciPy1.14.1と既存Python sector kernel。package identityとimportのPASSは、下記のbatch API互換性を保証しなかった。

前回の人工45 test＋29 subtestsはデータ・reference・mock fit・環境import/JAXの確認であり、**実ffsim内部のbatch初期化を通していなかった**。その確認不足により今回の不具合を科学fit前に検出できなかった。以前の人工PASSはその限定範囲のまま保全し、実fitの資格と区別する。

## 3. 技術的な起動修正の履歴と科学callの区別

| 出力・段階 | 起きたこと | 科学call / 対応 |
|---|---|---|
| 監視wrapperの起動前失敗 | task rootの親directoryを1段誤り、Pythonがrunnerファイルを開けなかった | 0、output生成なし。[失敗原本](../../artifacts/research_c_c1_stage1_science/2026-10-25/launcher_precheck_failure/)を保全し、監視wrapperのrootだけ修正 |
| run01 | 固定CLIが `fit_python.resolve()` でvenv Pythonのsymlinkをbase Pythonへ解決し、workerのNumPy importで停止 | 0、fit.claimも台帳も未生成。[STOP](../../artifacts/research_c_c1_stage1_science/2026-10-25/run01/STOP.json)を保全。run01を再開しない |
| run02 | regular executable entryがqualified venv Pythonをexecする形でfresh outputへ起動。固定SOURCE・package・契約・科学条件は同じ | input load1、fit attempt1。ffsim内部初期化でSTOP、追加fitなし |

ユーザーが許可した意図しない技術中断の修正として、科学call消費前の起動不具合を修正した。[entryと人工subprocess確認](../../artifacts/research_c_c1_stage1_science/2026-10-25/technical_stop_run01.json)、[entry本文](../../artifacts/research_c_c1_stage1_science/2026-10-25/fit_python_entry.sh)を保存した。旧approval/output/STOPは再利用・上書きせず、新しいapprovalをrun02へ結び付けた。科学fitのretry/resume0と総fit1回は維持した。

## 4. fitが停止した原因

`ffsim.linalg.double_factorized` へ1回入った後、通常C-DF初期化のmodified Choleskyを経て、`df_tensors_to_params` → `unitaries_to_parameters` の初期値x0作成で停止した。

固定ffsimの `linalg/util.py:207` は `(n_mats,dim,dim)` のbatch行列を `scipy.linalg.logm(mats)` へ渡す。一方、固定SciPy1.14.1の `_matfuncs.py:53` は `len(A.shape)!=2` を拒否する。この組合せでは `ValueError: expected square array_like input` となり、後続の `scipy.optimize.minimize` に到達しない。[実traceback](../../artifacts/research_c_c1_stage1_science/2026-10-25/run02/fit_process.log)、[静的SOURCE照合](../../artifacts/research_c_c1_stage1_science/2026-10-25/static_failure_audit.json)。

この診断はtracebackと固定インストール済みSOURCEを読むだけで行った。実H4配列の追加演算、fit、tensor residual、operator構築は行っていない。API非互換の修正にSciPy変更または明示的なbatch接続が必要だが、今回の固定環境・SOURCEを書き換えたり、もう一度fitしたりしていない。

ffsim呼出しが返らなかったため、J/U/OptimizeResultはglueへ返らず、raw callbackも実行されなかった。内部の初期化配列はexportされていない。**実Rは未記録**であり、上限8から実rank8と推定して記載しない。最終tensor residual・収束情報もない。

## 5. 全gateと要求された数値の状態

| 項目 | 判定・値 |
|---|---|
| SOURCE/input/契約hash | PASS |
| fit/eval環境identity・import・ffsim SOURCE | PASS |
| 明示承認・fresh output・固定allowance版 | PASS |
| fit実APIの初期化 | **技術的FAIL / STOP**、batch logm非互換 |
| optimizer success・有限export資格 | 未到達。収束FAILと判定できるOptimizeResultはない |
| κ/J/U構造gate・保存XDF κ照合 | 未実施 |
| 一般K adapter：full256 base/各frame・sector36合計 | 未実施、同値性のPASS/FAILは未判定 |
| C-DF物理sector誤差（Hartree） | **未測定** |
| X-DF物理sector誤差（Hartree） | **未測定** |
| 最終tensor residual・実R | 未取得 |
| Stage 1表現GO | 出せない、Stage 1未完了 |

数値がない項目はmachine-readable summaryでnull/NOT_RUNとした。0や過去の保存値を代入していない。この停止はHamiltonian近似精度・一般K代数・C-DF資源性能のFAILを意味しない。

## 6. 科学callと資源使用

[fit call ledger](../../artifacts/research_c_c1_stage1_science/2026-10-25/run02/fit_calls.jsonl)は `input_load=1, fit=1`。全outputを通じて同じ2件だけで、fitは**呼出し1回、正常return0回**。optimizer実行0、qualification/residual0、評価input/export load0、operator/norm/energy0、real circuit/compile0、Stage 2/3は0。停止後の追加科学call0。科学retry0、resume0。

契約の `83+8R` は正常終了時の算出規則で、今回はそのprefixの2計上単位のみを消費した。最大147以内。初期化のCholeskyは1 fit内部の処理で、新XDF分解ではない。実Rやsolver nfev等は不明で、成功経路の総数を実績として報告しない。

利用可能資源は起動前32 CPU、MemAvailable約52 GiB、memory pressure 0、GPU driverは利用不可。既存JAX/BLAS経路と将来frame並列構成をそのまま使い、CPU/RAM/worker/wall-timeへ新たな恣意的capを設けていない。frame比較には到達していない。

run02の[GNU time](../../artifacts/research_c_c1_stage1_science/2026-10-25/resource_time_run02.txt)はwall **2.44秒**、user7.67秒/system3.09秒、最大RSS **235,680 KiB（約230 MiB）**、swap0。1秒間隔の[process tree監視](../../artifacts/research_c_c1_stage1_science/2026-10-25/resource_usage_run02.json)はwrapperを含む約3.00秒、sampled aggregate RSS最大232,196 KiB、thread合計最大65。両者の対象とsample時刻が異なるので同じpeakと扱わない。thread数は実同時CPU利用core数ではない。終了は資源不足やtimeoutではない。

## 7. 保全、公開、次の担当

run01/run02のapproval、SOURCE、契約、runtime、traceback、fit.claim、ledger、STOPと起動失敗原本を保存する。新しいevidence inventoryはSHA256で追跡する。環境cache/bytecodeは数値証拠ではないため公開対象から除き、task内に維持する。元inputを再生成・変更せず、固定SOURCE22ファイル、旧grant/ledger/marker/STOP、C4終了・Gaussian singleton保留、元workspaceを保全した。

Stage 2へ進む条件はまだ満たしていない。担当と次の作業は以下である。

- **Codex（次の技術作業案）**：venv interpreterの取り扱いとffsim/SciPy batch API互換性を修正し、実fitterの初期化経路を人工入力で確認する。新しい環境/SOURCE/契約へ固定し、今回の原本には上書きしない。
- **ユーザー・統括**：修正後の固定SOURCE・契約・新outputに対する新たな1 fitの科学実行を承認するか判断する。現在のfit枠は消費済み。
- **Stage 1完了後にユーザー・統括**：測定されたDF誤差を確認し、Stage 2の比較精度・回路scopeを別途採択する。現在はStage 2/3未承認のまま。

今回の許可範囲では追加fitを行わず、部分結果を公開・remote照合後にmandatory STOPする。
