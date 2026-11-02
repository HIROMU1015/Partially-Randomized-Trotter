# Track A H6精度一致・測定込み資源比較 v1：起動時STOP・証拠索引

2026-10-11。利用者が公開commit `12183db780bef14c25c93ba6523c44c52edd7828` の固定契約を一回実行認可し、新grantを `cf52918e313800e81be25c563035a94dad9579ee` で公開・remote照合後、固定runnerを一回起動した。**signal workerの資源設定でSTOPし、科学的比較の数値は得られていない。** 認可は消費済み。source修正・retry/resume・H8・追加探索には進まない。

## 原記録と実行来歴

- [結果前契約](track_a_h6_matched_accuracy_execution_contract_v1.md)、[sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/sealed_manifest_v1.json)：認可対象 `12183db780bef14c25c93ba6523c44c52edd7828`。
- [新grant](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/authorization_v1.json)、[実行前seal](track_a_h6_matched_accuracy_execution_seal_v1.md)、[remote起動前照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/remote_prelaunch_verification_v1.json)：grant公開 `cf52918e313800e81be25c563035a94dad9579ee`、258 repository pathのremote bytes一致。実行前文書の「未起動」は当時の履歴として保持する。
- science source `057ba97ba4db66775cf713c65cb6955e9dcb0d85`。[source closure](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/sealed_manifest_v1.json)の211 runtime pathに変更なし。[runner](../../scripts/resource_applicability/run_track_a_h6_matched_v1.py)、[supervisor/worker](../../src/trottertracks/resource_applicability/h6_matched_execution_v1.py)、[static auditor](../../scripts/resource_applicability/audit_track_a_h6_matched_v1.py)。
- 保存入力source `f37005f01b2be38c5993d6e82df91abe9c643d21`、入力結果 `554fc52add39c2c1b45b765a3135df76fda6f15a`、[入力監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json) `f52787a22542b31bd39fd004a8d3d71325bc56b0`。親33件とsnapshotは元commit/bytesのまま。
- 一回出力の[terminal](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/execution_terminal.json)、[raw inventory](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/execution_inventory.json)、[launch binding](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/launch_binding.json)、[execution claim](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/execution_claim.json)。
- [signal worker原ログ](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/signal/worker.log)、[task資源binding](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/signal/task_binding.json)、[dispatch記録](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1/progress_0000.json)。
- 公開結果の完全なpath/hash対応は[外部証拠inventory](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/execution_evidence_inventory_v1.json)。この文書・原出力・外部監査の保存commitはGit履歴および後続remote公開照合記録で特定する。実行時source commitとの混同を避ける。

## STOPの技術的原因と監査の限界

supervisor terminalは `H6_MATCHED_RESOURCE_STOP`、reasonは `RuntimeError:WORKER_EXIT:1`、runner exit codeは1。worker tracebackは `resource.setrlimit(RLIMIT_AS, (address_bytes, address_bytes))` で `ValueError: not allowed to raise maximum limit`。変更していないsourceではcoordinatorがsoft/hard両方を2 GiBへ設定した後にworkerをspawnする。workerはそのhard limitを継承し、固定bindingの8 GiBへ設定しようとして失敗する。これは原tracebackとsource制御フローの対応による技術的原因の確認である。子の`/proc/limits`実測receiptは保存されていない。

例外は`execute_worker()`の`verify_worker()`呼出し中に発生した。数値portのimport、保存配列の数値decode、reference/setup/correctness、sampling、回路build/compileへ進んでいない。`signal/worker_claim.json`と`signal/worker_terminal.json`も未作成である。親supervisorのterminalは保存されている。再現試験や新しい科学計算でこの停止を救済していない。

[静的監査](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/static_identity_audit_v1.json)はexit0、返却値 `STATIC_IDENTITY_AND_COVERAGE_PASS`。STOP経路ではsource/input/grantとraw inventory closureの照合を意味し、全92 signal・費用標本のcoverageや数値精度への合格を意味しない。[外部事後記録](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/execution_observation_v1.json)でscopeを明記した。原inventoryの`source_input_after_verified=false`は書き換えず、外部監査でsource211・入力33・環境の事後一致を別に確認した。

原inventoryの`live_aggregate_peak_rss_bytes=0`は、失敗batchがreturnしなかったため親に残った初期値であり、実測RSS peakではない。`aggregate_calls.json`は未作成。科学処理0という記述は失敗したpre-import経路と欠測からの監査判断で、保存されたaggregate counterとは区別する。tool観測の起動からreturnまで約0.50秒は計算所要時間の見積もりではない。phase/total wall capはnullのまま。

## 比較範囲・欠測・保全

対象はlinear H6、1.00 Å、STO-3G、12 modes、α3/β3、sector400、tol-only `1e-8`のsigned generation-order rank19、cutoff0、採用済みweighted Hermitian projection・保存state、T=0.8。generation-prefix5/10/15のB0/B2、full rank19のB1 S2/S4、q1/2/4/8、B2 r1/2/4・R=qr・K2/4、delta0.8/0.4/0.2/0.1、ε_sig=0.05/0.01/0.005/0.001を固定した。

[欠測一覧](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/missing_coverage_v1.json)に92候補を全て`NOT_EVALUATED_PRE_IMPORT_RESOURCE_STOP`として記録した。bias、empirical u、normalization、N、wrapper費用、Gは未取得。B3 normalization-onlyの168組も未取得。適格性を評価していないためexplorationの実際の登録数、confirmation選抜集合・数は未定であり、上限596/256 group・1708 wrapperを「残タスク数」とは扱わない。資源的優位、予測精度、化学的精度、formal winnerについて新しい結論はない。

signal CPU [0,2,5,6]・Numba/OMP4/BLAS1、費用は2 disposable worker・各thread1の計画を変更しなかった。coordinator AS2 GiB、worker AS8 GiB、aggregate RSS18 GiB、output2 GiB・log1 MiB・call guardを維持。費用workerはdispatchされていない。[保全記録](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/preservation_v1.json)は既存4474 pathのbytesを今回の明示的な文書追記だけを除いて照合し、旧dirty・未追跡差分、凍結source、保存入力、旧pilot142 raw fileの保全を確認した。旧pilotの7 cell・32/36 wrapper・STOP・欠測はそのまま保持する。

**mandatory STOP。** `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、`next_stage_authorized=false`、H8・追加探索未認可を維持する。今回のgrantで再実行しない。以後の資源limit継承修正、process境界のsynthetic検査、別source/seal/出力identityの準備、再実行認可は別作業とする。GPTへは数値比較結果が未取得であることと、この技術的停止・監査scopeを引き渡す。科学的GO/STOPをCodexで代行しない。

公開確認：原STOP出力・外部監査の保存commitは `274b984971f9afc3425f3b3d1ce36e1c478150ea`。[GitHub再取得照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/remote_result_verification_v1.json)で419 path（source211、入力33、旧pilot142、新公開32 pathと元manifest）をSHA-256照合し、重要な相対リンクの存在を確認した。実行時sourceは上記057ba97のままである。
