# Track A H8：主要3構成・6軸の直接compile実行準備（2026-10-17）

準備A/Bを完了。**実H8のbuild/control/compile/fit/費用再会計は全て0回。未承認・未seal・mandatory STOP。**

SOURCE：`352581103b7dbd4fa58e28af7a6ca65570d36e8e`。branch：`track-a-h8-key-candidate-compile-readiness-20261017`。
[監査bundle](../../artifacts/resource_applicability/track_a_h8_key_compile_readiness/2026-10-17/README.md)、[固定入力契約](../../references/h8_key_compile_inputs_v1.json)、[source closure](../../references/h8_key_compile_closure_v1.json)。

## 科学的な選定と固定来歴

利用者指定の `track_a_h8_strategic_scientific_review_2026-10-17.md`（34673 bytes、SHA-256 `73a18b20271f0c0c39b7dc3decba651168c7d9132162609b8162e6feadc67bca`）を読んだ。採択済み3構成の実装準備だけであり、追加GPT/Claude科学レビューは行っていない。

| 優先順の候補（各cosine/sine） | 固定モデルRZ予測／軸（exact） | 保存metadataからのnative命令割当bound | prefix fit14–21からの外挿 |
|---|---:|---:|---|
| `B0I_p15_q4_4th` | 4670300/7 | 243121 | なし |
| `B0I_p11_q1_2nd` | 298323/7 | 14981 | あり（p11） |
| `B0_p15_q4_2nd` | 1564196/7 | 81041 | なし |

予測はB1の固定係数をそのまま評価した静的な契約値であり、実測でも新fitでもない。S2はa+bq+dpq、S4はa+3bq+3dpq。a=1592,b=1314,d=25271/7。命令boundはprovider `block_instruction_bound` の既存式を保存操作数へ適用した割当検査であり、RZ費用ではない。

- [B1正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/artifacts/resource_applicability/track_a_h8_phase_b1_run01/2026-10-17/frozen_models_v1.json)。model SHA `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。B0のholdoutとB0+Iのtraining残差を混同しない。
- [B2完了](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md)。旧4軸PASSは今回6軸・全格子の実測精度保証ではない。元B0 run02 STOPは保持。
- [通常Phase C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md)と[非相殺cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md)の結果を見た後の局所確認であり、事前に独立登録された新holdoutではない。費用・ランキング・PR統計を再計算していない。

## 入力とnative規約

Linear H8/隣接1Å/STO-3G、16 spin modes、Nα=Nβ=4、sector4900、保存tol-only DF28（tol1e-8/cutoff0）、signed generation order、T=.8、symmetric-directional controlを維持。S4は負時間の中段を含む非merge 3 S2 Yoshida。

保存入力11件をstreaming bytes/SHAで照合した。NPZは1,299,688 bytes、SHA `d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`。NPZの科学array読込0。保存full preparation export（5,520,922 bytes）からp11/p15のordered block viewをmetadataとして結合し、Gateや回路は構築していない。3構成のcorrected/independent-forward/unrotated full-H reference、各B0+Iの**自身のprefixに対応する**保存full-Fock tail identity scalarをそのままコピーした。p21 scalarやH6係数の代用、新scalar計算、新reference生成はしない。B0は元scalar一回、B0+Iは元scalar＋保存Cを一回、referenceを回転しない。

compiler SHA `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile SHA `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、provider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。rz/sx/x/cx、optimization_level1、seed17、Qiskit1.3.0、backend/coupling_mapなしを固定。既存private venvを変更せず、provider/runtimeの265/34,273 filesのcanonical mapを引き継いだ。必要なnative API5 filesのbyte SHAを今回確認。full runtimeの再hashは無意味に反復せず、別途承認後のfresh gateで行う。旧server compilerとの完全同一性を新たに認定しない。

## 実装と高速化

[plan/契約](../../scripts/h8_key_compile_contract.py)、[native adapter](../../scripts/h8_key_compile_adapter.py)、[並列control/QPY](../../scripts/h8_key_compile_controls.py)、[owned worker](../../scripts/h8_key_compile_parallel.py)、[plan-only default runner/launcher](../../scripts/run_h8_key_compile.py)、[metadata準備CLI](../../scripts/prepare_h8_key_compile.py)、[人工tests](../../tests/test_h8_key_compile.py)。旧科学providerとB2のrestore/prefix観測/compile測定関数を再利用し、既存sourceを変更していない。

snapshot読込1回・full preparation decode1回・prefix view2件。候補ごとにnative evolutionを一度だけ構築する。left/right controlは2 worker、cosine/sineの検証・fingerprint・QPY保存も2 worker。大回路を親から解放し、compile workerが検証済みQPYを1軸ずつ読み込む。compileは各軸の判定後に次へ進む直列1。後続軸のcompileを投機起動しないため、不一致後の余分なcompileが生じない。

各軸で `abs(prediction-measured)/measured < 1/50` を要求し、2%以上は即STOP。分母は実測正整数RZ、判定はFraction。差が0でなくても2%未満は既存の登録軸基準に従い合格とし、exact差を保存する。raw測定・native exit・回収・SOURCE/profile/preparation/fingerprintを保存後に判定する。不合格をacceptedと記録しない。モデルの自動refit、追加compile、retry、次stageは0。

許可され得る実作業は3 native build、6 measured wrapper、12 native control probe、6 compile/6 model checkまで。control probeは保存端点とのnative回路correctness検査であり、別signal探索やtrajectory生成はしない。control誤差はfiniteかつ1e-8未満、失敗はcompile前STOP。

## 資源・終了・観測

提案はdriver1 core＋準備worker2 coreの3物理core。compileはそのworkerの1 core、薄いlauncher兼observerはdriver coreを共有する。具体CPU番号は未指定、`allowed_cpus=[]`。別承認後、既存の本人pinned科学jobとのCPU重複を避けてfresh選定する。共有OS/venv/他jobを変更しない。

新規CPU/RAM/AS/RSS/CPUtime/wall上限は設けず、OS/cgroup制御をそのまま継承する。64GiB aggregate peak＋16GiB headroom＝80GiBは**入場時の見積りであり、実消費・予約・RSS上限ではない**。参考となる旧B2 peak26.252GiBから余裕を取った未実測の提案で、今回の最大使用量の保証ではない。準備時の実効空きは約468.6GiB、disk空き約486.1GiB、inode空き219279085。これは起動gateではなく、起動時に再確認する。

容量はQPY6件、control/native receipt、非返却予約、journal、一時file、stdout/stderrまで含め初期2GiB/1024 inodeを見積もる。監視は1本のappend JSONLなので観測ごとにinodeを増やさない。今回のsampleで約28.6MiB/dayの監視byte増加。wall上限がないため総容量の有限上限とは扱わず、経過日数・stdout/stderr成長を加える。quotaは承認後に有効値/マウント方針を再確認し、不明なら起動しない。

launcherを数値driverから分離し、重いserialization中も通常5秒ごとに資源を観測する。host/上位cgroup PSI・他chatの通常更新・一時観測I/O異常だけではworkerを強制終了しない。観測を再確立し、最終PASSには成功した最終観測とnative終了・整合確認を要求する。実OOM/native異常終了や自分の入力・SOURCE改変はPASSにしない。

own PID/start/PPID/UID・pidfd・PDEATHSIG・wait4で所有を確認し、TERM→必要時KILL→reap。最初のSTOP理由を保持する。launcher中断時はdriverに内部worker cleanupの時間を与える。各workerの終了後にnative exit0・result file/hashを一括正規化し、/proc観測中のreap競合でもhash欠測や非zero終了をPASSにしない。FDを閉じ、二重signal/reapを避ける。

## 検証と未検証

人工22 test methodsがPASS（最終1.110秒）。6つ全ての不一致位置、2%境界、無効grant拒否、SOURCE/profile/input欠測・改変、無関係資料更新、scalar二重加算/別prefix混入、S2/S4 API double、QPY改変、owned小processの成功/異常/回収、実launcher→人工driver、ESRCH/非ESRCH/所有不一致を検査した。テスト内のnative/QPYはdoubleであり、**実transpile0**。人工テストで見つけたreap/hash競合とPython -P準備CLI import不具合を今回の新source内で修正した。失敗した人工logもprivate証跡として残す。

担当自身の限定技術確認であり、別担当による独立reviewとは呼ばない。subagent0。実H8 end-to-end、実測RZ、速度、最大RAM、完了時間、全184候補の実測最良、厳密bias保証、PR母平均優位は未検証。既存SOURCE/結果/grant/ledger/marker・M1-S・他worktreeを変更していない。既存runのcarry/STOP/消費済みgrantも変更しない。

## 残る承認と次の担当

1. **ユーザー**：SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e` と本planに対し、この3構成6軸のone-shot直接compile（保存端点へのnative correctness probeを含む）だけを明示承認する。2%基準、2並列準備/直列compile、CPUのfresh選定、上記資源方針を対象とする。完了・不一致・技術STOP後は停止し、自動refit/追加compile/retryはしない。
2. **Codex**：その承認後に独立実行snapshotをSOURCEへ固定し、SOURCE22 files/入力11件/provider/compiler/runtime・CPU/memory/FS/inode/quota・未使用output/control・one-shot lockをfresh確認。新grantに結合・sealし、一回だけ実compileして結果を保全する。現在の資料branchはREVIEW commitへ進むため、そのHEADをSOURCEとして使わない。
3. **ユーザー**：結果受領後、実測単価による通常・非相殺費用の限定再会計を別途判断・承認する。今回の承認案に再会計は含めない。

現在は `approved=false`、`runtime_authorization=false`、`execution_sealed=false`、`absolute_launch_command=null`。無効grantで実launcherを呼んでも科学import/lock/affinity/child前に拒否することを確認した。output/control/lockは未使用。有効な起動commandは未発行。

## 保存と公開状態

既存のclean資料clone（基準commit `23d46e489af054735d9d328aad0342ecdbddf305`、originの同branch SHA一致確認済み）から `--no-hardlinks` の独立cloneをhome配下に作成した。旧worktree登録・root・run・科学sourceは変更していない。作業場所は `/home/AbeHiromu/projects/h8-key-candidate-compile-preparation/20261017-01/repo`、private draftは兄弟directory `draft02`。過去の無効draftも保全。

source/tests/入力metadataはSOURCEに固定し、本報告・plan・無効grant・test結果・profile/input bindingを別REVIEW commitへ固定する。今回の依頼は準備A/Bのためremote更新は行わず、local commitとして保持する。NPZ・実runtime/QPY/checkpoint・private config・認証情報・内部SSH情報をstageしない。

公開する場合の手動command（この準備では実行していない）：

```bash
git -C "/home/AbeHiromu/projects/h8-key-candidate-compile-preparation/20261017-01/repo" \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h8-key-candidate-compile-readiness-20261017
```

準備完了後mandatory STOP。追加の実H8計算・sampling・科学review・原稿改訂は開始しない。
