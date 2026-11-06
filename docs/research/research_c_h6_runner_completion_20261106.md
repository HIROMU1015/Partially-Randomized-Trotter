# 研究C：H6統合runner完成と科学実行直前準備（2026-11-06）

基点は `68634ef468eb6c912de7bf569484fc454fb32548`。本作業は接続実装・人工検証・byte-only preflight・SOURCE固定・公開であり、**実H6/H4/H2の新科学計算は0**。技術的実行可能と科学実行許可を分離し、公開照合後はMANDATORY STOPする。科学実行は別途ユーザーの明示許可が必要である。

[採用条件を保持した実行用契約](research_c_h6_contract_unapproved_20261106.json)は `execution_ready=true`、`scientific_execution_authorized=false`。旧契約案は基点commitと元準備worktreeに保全し、本branchで実行接続・採用済みD+C・環境・保存先の差分だけを具体化した。[前段技術調査](research_c_h6_feasibility_preparation_20261106.md)の「runner未接続」「設計未採択」は当時の履歴であり、本報告が接続状態を更新する。

## 1. 接続と既存APIの再利用

| 接続 | 完成した経路と既存API |
|---|---|
| 親runner | `scripts/run_research_c_h6.py` → `c_h6_runtime.Run`。既存 `RpcProcess`、fit transport、`Calls`を再利用。新fresh出力のみ、retry/resume=0。 |
| 保存input/C0 | original input/C0のbyte identityを照合。fit環境の既存real-Schur `roundtrip`でtensor fidelityを確認し、共通開始J/Uは元配列のまま保存。x0はgauge同値chartの来歴で、recentered optimizerの開始J/Uを差し替えない。新SCF/積分/中立fitなし。 |
| H6物理/fit RPC | `c_near_transfer.System(6,14,4,...)`、`PhysicalContext`、`SectorCAR(6)`。既存 `run_cell/recipe_cell/recipe_arm`、PR/ENTRY objective・J/U solver・engineering guard・残差とsolver記録を利用。16診断と64終端を保存。 |
| 参照・ground | 候補固有15因子、sector400のHamiltonian/ground/alias。既存 `prepare_arm`、`reference_kernels/reference_round`、`corrected_kernel/rte_round`、feasibility/RPE判定を接続。共通PF/exact-Hとhead指数は同じ候補内だけでcache。r32の不適格性がr64/128の参照を欠損させない。 |
| native | 既存 `StrongBuilder/PrecisionBuilder`、Gaussian回転・relative fusion、`compiled_wrapper`、P0構造検査・metrics。PR/ENTRY共通Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、既存位相規約を維持。 |
| 同J/U D | 既存全frame ordered S2/s1、`ideal_plans`、`serial_round_cost`。同input・同candidateのDだけ共有し、新しい最適決定論探索は行わない。 |
| shell/X/Y | 実raw/compiled shellのancilla-only 2×2を照合。checked回路の全基底出力から線形性でown-groundの出力を構成し、実shellと既存 `signal_gate`でcomplex X/Yを検査。追加の入力列計算は不要。 |
| 会計・統計 | `calibration/cost_calibration/serial_cost/aggregate_totals/calibration_uncertainty/affine`と既存 `primary/arm_summary`。shots、W、kernel平均、shell、総費用を分離。主familyは明示N128/97.5%、D個別95%。 |

新optimizer、Hamiltonian式、PF/RTE式、shots式、compiler、統計式を実装していない。新moduleは既存engineの接続・候補binding・保存・workerの処理である。

共通既存コードの差分は、Callsの入力列数指定と外部worker計数、fit transportの入口指定とcodec操作、atom/reference cacheの任意引数、ground aliasの不適格性区別、head指数cache引数に限る。既定引数では従来動作を保持する。旧Python3.11 byte-preflightの人工testは歴史的version metadataをmockするfixtureへ修正し、旧科学runtimeのpin判定自体は変更しない。

## 2. 固定した科学条件

H6直鎖・隣接1.00 Å/STO-3G、Nα=Nβ=3、sector400/全Fock4096/controlled8192、Number-CDF rank14、元frame順LD4、保存中立CDF14 C0のみ。PR/ENTRY × 正則化初期寄与0.03/0.1/0.3の6 trajectory、最大64受理更新・512 trial、16は診断のみ、64を下流へ渡す。

共通r32と公平r32/64/128を別選択。partial ordered S2/s1、h=δ=.1、K0、m=0..12、各候補own-DF ground、状態準備除外、元g補正κ・nuclear scalar一回を維持する。物理DF予算.00019875 Ha、CA=.00159 Ha、failure=.01、βtotal=π/6、βstat=π/12。

rank14は既存中立CDF14の物理予算PASSを根拠とするが、LD4最適性、初期値頑健性、同時収束は保証しない。等しい更新数は等しい古典負担・到達収束度を意味しない。H4との差を系サイズ単独の因果効果と呼ばない。

## 3. 26回路の資格と価格標本

最大18 eligible selection policyのsample0、最大4 independent final pairのsample0、最大4 same-J/U Dを検査対象とする。費用を見たsampleの選び直しや未使用枠の転用はしない。

checked kernelごとにancilla×sectorの**全800入力、full8192出力**をstream処理する。raw/reference、compiled/raw/reference、独立に構築した実raw/compiled inverseとreference dagger、実inverse return、leakage/norm/相対位相を検査する。既存ColumnStream/state-action、double/zero_threshold0/truncation無効、成分・norm許容1e-10、phase fit禁止、complex X/Y許容2e-10を維持。

これは800入力部分空間上の数値検査であり、全8192入力の演算子一致や厳密certificateではない。逆回路検査を共役転置の自明な比較へ置換していない。

残る最大1,066価格回路にはP0構造・有限parameter・phase/layout等だけを検査し、**`NUMERICAL_EQUIVALENCE_NOT_CHECKED`**を標本別に保存する。全N標本を費用統計へ含め、PASS標本だけに選別しない。unchecked circuitの等価性を仮定した条件付き価格推定であることをselection、accounting、統計、最終比較まで伝播する。全law実装誤差・actual CA・無条件優位はnullのままである。

selection metadataの`correctness=PASS`は既存selectorのschema上の選択行適格性であり、全標本数値等価性を表さない。`numerical_equivalence`、標本別label、`correctness_scope`、`qualification`で実際の資格を別記する。

## 4. cohort・D・統計

selection32とfinal128のseed namespaceはSOURCE/input/arm/candidate/r/roleに結び付ける。選択結果を`selected.json`へ固定してからfinalを生成する。同じ(arm,candidate,r)が共通比較とretuningの両方に現れた場合だけfinalを共有し、異なるarmではcandidate bytesが同じでも独立cohortを使う。

Dは同input/candidate・固定全frame S2/s1政策だけで重複排除する。精度未適格なpolicyやDはUNRESOLVED/nullで残し、別候補・r・標本数へ延長しない。

主費用は `W * mean_kernel_RZ + shell_RZ`。rや校正NをWへ重複乗算しない。2比較それぞれN128/97.5%の近似Welch差区間、halfwidth/ENTRY≤5%、既存skewness/influence診断を使用。PR/ENTRY比区間は作らず、Dは個別nominal95%の副解析。有限標本IIDモデルの近似であり、coverageやactual CAを保証しない。

## 5. 人工検証

[最終verification](../../artifacts/research_c_h6_runner/2026-11-06/technical/verification.json)・[eval JUnit](../../artifacts/research_c_h6_runner/2026-11-06/technical/evaluation_tests.xml)・[fit JUnit](../../artifacts/research_c_h6_runner/2026-11-06/technical/fit_tests.xml)が正本。作者自身の技術検査であり、別reviewerの独立科学レビューではない。 最終版はeval **192 PASS**（H6 runner人工E2E等17件を含む）＋fit **8 PASS**＝**200 PASS、FAIL/ERROR/SKIP 0**。eval85.41秒、fit4.56秒。

- 親E2Eは人工mockで6trajectory/16・64checkpoint/18policyを通し、DF不適格、r不適格、同pair共有、同candidate異arm独立、D重複排除、26検査/124800列/218350 units、phase/inverse/leakage FAIL→partial/STOPを確認。
- 実際の親PoolBackend・eval子process往復で、人工小次元のnative selection32/final128、shell/X-Y、会計、D、cohort独立性を確認。cache解放後の再構築も通す。
- 六軌道・rank14の**人工**J/U/gを使い、実PhysicalContext/RPCでsector400・15因子export、別fit interpreterのreal-Schur codec/504変数を確認。実分子は使わない。
- 人工六軌道のPR/ENTRY初期尺度・gradient/有限差分と、既存H4 selector・fit schema・会計・state-action・phase/compilerの回帰を確認。
- r32のattenuationゼロを模擬しても共通PF cacheが欠損せずr64を独立評価できること、service close失敗時に成功resultを公開しないことを確認。
- [人工13q再検証](../../artifacts/research_c_h6_runner/2026-11-06/technical/artificial_13q.json)：全800×6作用/8192出力、約39.370秒、peak RSS約209.5 MiB、最大成分残差1.60e-15、PASS。対角の人工49/compiled121操作に限り、実H6 kernelのETAや速度倍率ではない。

開発中にRPC応答のNumPy型、更新される計数summary、人工testの環境分離などを修正した。最初の広いtest投入でfit依存ffsimをeval環境へ誤投入した失敗と、旧Python3.11 test fixtureの不整合を確認し、環境に応じた最終testで再確認した。既存科学pinや許容差を緩めていない。

## 6. 規模・高速化・資源・保存

| 登録作業 | 最大 |
|---|---:|
| trajectory / checkpoint / policy | 6 / 12 / 18 |
| selection / final / D native | 576 / 512 / 4 |
| numerically checked kernels | 26 |
| state-action入力列 | 26×6×800 = 124,800 |
| input/optimization/reference/native-build/state-action/statistics units | 32 / 19,278 / 4,096 / 69,888 / 124,800 / 256 |
| 全logical units | **218,350** |

unitはmacro処理/入力列作用で、CPU時間ではない。optimizer内部nfev/njevやevent/gate数を別記する。

候補processを並列化し、同candidateのatomとtail検査・head指数・PF/exact-Hをr間共有する。r別law/normalizationは毎回検査し、tailのhash・λ・全component identityが一致するときだけtail数値検査を再利用する。unchecked標本のtrajectory referenceやinverseを作らない。atom/event cacheは選択の各候補終了時とfinal pair終了時に解放し、必要な候補のchecked回路で再構築する。

[現在の資源観測](../../artifacts/research_c_h6_runner/2026-11-06/technical/resources.json)：affinity64、観測available RAM約466 GiB、disk約348 GiB。読めた階層のCPU/RAM quotaはmax、scheduler/quotaコマンド未検出で、disk quota無しとは断定しない。短いaffinity内CPU idle観測では約17 CPU相当が空き、モデルは候補6process×2threadsとなった。これは将来の固定上限や予約ではない。

実行時にaffinity/quota/空きCPU/RAMを再測定し、候補数と約5.72 GiB/active-workerのcacheモデルを使ってprocess数とBLAS/OpenMP/Aer threadsを協調させる。共有設定・他processは変更しない。GPUは既存Aer CPU buildで利用しない。一時的PSI上昇だけで科学workerを再起動しない。

保存先はfresh output。最大QPY **2,242**（全1,092のraw/compiled各1、checked26のinverse各2、shell6）、checked reference26、sample events/metrics/資格、最適化raw/partial/STOPを保存する。full8192 dense operatorの標本別保存は行わない。物理評価最大391のJ/U/15因子/H exportは約14.9 GiBの非圧縮配列モデルで、optimizer raw等は別費用。**60 GiBは計画envelopeで、上限でも完走保証でもない**。現在の空きは計画値を上回るが、実kernel QPY単価は未測定。容量不足・I/O失敗時はpartial/STOPを残し、旧rawを削除しない。

最初の登録selection32 batchについてsample0検査時間と残る31価格回路を分けて測り、最大残件のnative時間を更新する。候補/rを一般化する仮定付きのserial-equivalent/parallelモデルであり、fit/referenceを含む全体wall timeや信頼区間ではない。未登録pilotを追加しない。実H6の確定ETAはまだない。

## 7. SOURCE・入力・環境・実行入口

[実行用SOURCE](../../artifacts/research_c_h6_runner/2026-11-06/execution_preflight/SOURCE.json)は接続済みコード・literal legacy動的依存を含む148ファイルと契約configurationを固定する。契約自身のSOURCE hash欄だけをhash cycleから除外し、残る全条件は一致を検査する。

- SOURCE SHA256: `0be3f9e073172018fc51fe30433dea9280c6d415a1fcfdd1d94ccabd61ac46ef`
- 契約 SHA256: `351c2321092ea46116b96d6a08a9029aad8c00cfc6ae973d4e0128ad35acbf5a`
- input: `artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/input.npz`
  SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447`
- C0: `artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01/fit/final.npz`
  SHA256 `cd4de687f99d32bd6f644ac33100fb52d4f7543e710e6c273723a0908b87dc45`
- Fresh output: `artifacts/research_c_h6_independent_validation/2026-11-06-runner/run01`。今回未作成。
- Python3.12.3。Fit: NumPy1.26.4/SciPy1.14.1/ffsim0.0.83/JAX・jaxlib0.4.38/Qiskit2.4.2/PySCF2.14.0/threadpoolctl3.6.0。Eval: NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0/Aer0.15.1/Nature0.7.2/PySCF2.7.0/SymEngine0.13.0/threadpoolctl3.6.0。実行前に両interpreterのmetadataを照合する。既存専用venvはread-onlyで使う。

[Passive preflight](../../artifacts/research_c_h6_runner/2026-11-06/technical/passive_preflight.json)はSOURCE/input bytes・契約・package/Python versions・fresh不存在を検査するだけで、NPZをdecodeせず量子/化学engineを呼ばない。

```bash
task_env_root=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106
python3 -B scripts/run_research_c_h6.py --dry-run \
  --fit-python "$task_env_root/.venv-fit/bin/python" \
  --eval-python "$task_env_root/.venv-eval/bin/python"
```

将来の別途明示許可後の入口は同じscriptの`--execute`（両専用Python引数必須）。今回は起動していない。`prepare_research_c_h6_validation.py`もpassiveのままであり、科学実行optionを追加していない。

## 8. 残る問題とMANDATORY STOP

接続が未完のAPIは残していない。人工E2E・回帰・passiveの合格を実H6 fit/ground/PF/RTE/native/統計の成功・性能保証へ昇格しない。実分子の数値的適格性、QPY単価・wall time、共有資源とdisk quotaの実行時状態は未測定/再確認事項であり、結果依存の条件変更は認めない。

[保全照合](../../artifacts/research_c_h6_runner/2026-11-06/technical/preservation.json)で旧H4 SOURCE386、旧H6準備SOURCE32、旧契約原本、H6 input/C0 bytesを確認した。旧worktree/SOURCE/raw/STOP、共有package・設定・他ユーザーのファイル/processは変更しない。

次担当は**ユーザー／GPT**。新SOURCE・固定契約・人工検証の資格を確認し、このH6・C0・rank14/LD4・6trajectory・r32/64/128・selection32/final128・subset26・218350 units・fresh出力・retry/resume0の科学実行を許可するか判断する。承認証明書・grant・承認原文保存は不要。Codexは公開SHA/tree/固定SOURCE照合後に**MANDATORY STOP**し、別途明示許可があるまで実H6科学計算と重要レビューを開始しない。
