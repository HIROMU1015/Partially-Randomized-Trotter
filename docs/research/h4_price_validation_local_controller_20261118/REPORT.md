# H₄ 簡易価格検証のローカルcontroller統合

2026-11-18 JST。担当はCodex（本作業の著者）。独立科学レビューではない。

**コード統合、専用領域への配置、保存入力・人工検収、結果公開を実施した。別セッションでcontroller移行が進行中との利用者情報に従い、共有registryのcurrent切替と稼働clientの上書きは行っていない。** 最終追加版は `price-validation-c4e90c9e5`。採用手順と残件は [HANDOFF](HANDOFF.md)。新しい通常探索・実compile・実Q3・信号・fitは各0件。公開後mandatory STOP。

## SOURCEと既存成果の再利用

- 統合実装SOURCE：[`c4e90c9e5ffc7230583567500a99c4ec5c255f7b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/c4e90c9e5ffc7230583567500a99c4ec5c255f7b)。その後の `7249f008c` は旧人工テストの未定義出力先だけを修正した。
- 簡易検証 `92b3e617…`、配置修正 `a18d5a04…`、配置報告 `5910058f…`、C表示 `0d46a1e0…` を調査した。モデルID、PriceRules、代表点選定、worker、optional queue、publication consumerを再利用し、再実装していない。
- その後のローカル移行SOURCE `efeb628fa` / `d887abdca`、人工Gateway修正 `3e50475ba`、Windows通常client `ee07a2841` / `e197348db` も本branchへ統合した。他セッションの未commit変更は取り込まず、そのworktreeを変更していない。
- 新しい通常入口も `run_rpe_local0_search.py` → 固定release → `run_rpe_schedule_search.py` / `make_executor`。価格検証は同じ `DistributedExecutor.submit_preparation`、永続queue、claim、結果回収を使う。
- 既存の通常科学計算SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。価格workerは `a18d5a04df5ec8c789f274730bff87a2c0a7cc89`、既存候補組立ては `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2` を別に固定する。controllerの版を科学workerの許可へ転用しない。

最初に主checkoutと作業場所のAGENTS、PROJECT_MAPを読んだ。指定された11月14日版共通指示書はサーバー2の `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-11-14.md` を取得して全文確認した。Windows/WSLの配置場所のAGENTSも確認した。

## 差分

1. `prepare_romutail_controller_deployment.py` が全release/run共通の `price-validation/models.sqlite3` を設定する。旧共通Git領域のstoreがcontroller移転時に脱落する問題を解消した。
2. `transfer_rpe_controller_state.py` は明示されたstoreと既定storeをread-only SQLite backupで保全する。`reconcile_rpe_price_validation_store.py` は既存ModelStoreへ同じモデル・要求・所有者・部分成果を取り込み、同一snapshotの反復でローカルの進んだ状態を巻き戻さない。凍結要求や所有者の競合はHOLDし、別jobを作らない。元storeは変更しない。
3. 通常controllerが自身のrunに属するSSH relayを起動・回収する。既存relayのlockと同じendpoint/configを照合し、controller切断後の生存pollerを重複起動しない。停止はrelayの受付だけを止め、既受領workerの科学childをkillしない。queueとattemptは維持する。
4. 通常Gatewayに `scope.preparation_budgets` を追加した。signal/compile/auditの段階task数を明示し、Q3予算からcompile許可を推定しない。未指定時のcompile 0を保つ。今回の配置templateも全段階0のまま。
5. HEAD完全一致を、固定SOURCEに対する実装・設定の差分照合へ変更した。src/scripts/tools/configs/schemasと依存宣言のcommit済み・staged・working差分、新規未追跡実装を検出する。docsの結果metadata commitは許容する。runの固定入力・scope・評価契約の照合は残した。
6. PR-DF LD7の保存holdoutについて、`model_commit`、凍結係数、生成SOURCE、AXIS_PRICESのm/q/R・軸・物理ID・保存予測値を照合する既存形式の読取りを追加した。予測の再生成・fitは行わない。異なる係数は拒否する。
7. controllerが採用済みの共有13モデルを既存 `PRICE_MODEL_VALIDATIONS.json` へ公開できるようにした。各runの適用q/Rはapplicationsに分けて保持する。

モデルIDには方式・分割・公式・実step・辞書/law/event recipe・回路実装/compiler条件・凍結係数/SOURCEを対応させる。shot配分、κのラベル、run、実行hostではモデルを分けない。較正外の実schedule代表1点という既存選定と、PR独立2経路×X/Y＝4wrapperを保つ。X/Yを4独立経路として数えない。通常Q3・通常準備・先読みを優先し、optional検証待ち・失敗で適用可能候補の探索を止めない。

## 引継ぎと配置

[STORE_ADOPTION.json](STORE_ADOPTION.json)、[保存metadata照合](SAVED_METADATA_RECONCILIATION.json)、[DEPLOYMENT.json](DEPLOYMENT.json)。

サーバー2の共有storeから13モデルを保全・照合した。取得時は独立実測9、待機3、未完了1、価格検証job ID/owner/external jobは各0件。待機3件の選定済み代表点も保持した。保存DF7のSOURCE対応を照合後、ローカルは独立実測10・待機3となった。旧storeの未完了表示は保全し、新compileを起こして埋めていない。旧Q3のUNKNOWN・STOP・原結果は既存移行snapshotと通常preflightに残し、今回再投入していない。

|環境|実ホスト・役割|今回の配置・確認|
|---|---|---|
|ローカル|`abe-Precision-3260`、SSH上のRomuTail。controllerとlocal worker|固定追加release、共通store、local profile/CASを配置。価格要求2件のSOURCE/入力/runtime/compiler metadata PASS|
|サーバー2|`GPUA100-gleap02`、`qg2`。遠隔worker|固定a18 workerを再利用し、今回専用PROFILE/CASを配置。価格要求2件metadata PASS|
|サーバー1|`qg1`、既存設定のserver1 worker|利用者の停止情報に従いdisabled、required hostから除外。接続・起動していない。実ホスト名の今回のlive確認は未実施|
|ローカル0 Windows|`ABE`、ユーザーromuk。操作・閲覧client|既存Preview/Syncを使用。別環境のローカルと同一視しない。今回client設定を上書きしていない|
|ローカル0 WSL|Windows ABE上のUbuntu、Linux hostname `abe`。任意worker|専用a18 transport、固定13ab/326fe legacy view、PROFILE/CASを実配置。価格要求2件metadata PASS。既存SETTINGSのhashを保持し `accepting_new_jobs=false`|

[ローカルmetadata結果](LOCAL_READINESS.json)、[サーバー2ログ](SERVER2_DRY.log)、[WSL結果・受付hash](WSL_READINESS.json)。metadata PASSは実compile成功や新しい科学REQUESTのadmissionではない。実compile・実Q3は各0件である。元12工程の保存検査を今回の価格2件の実測として数えていない。

ローカルの保存bundle欠落、WSLのGit object/legacy view不足を解消して最終検査した。途中のHOLDを実compile失敗と扱わない。ソース・bundleは専用領域へ配置し、固定旧checkoutや他者のjobを変更していない。

## 検収

[ACCEPTANCE.json](ACCEPTANCE.json)、[TESTS.log](TESTS.log)。83件中82 PASS・1 SKIP・0 FAIL。SKIPは旧サーバー2の絶対pathに依存するbootstrap専用検査。すべて著者の人工・保存metadata検査である。

|要件|確認|
|---|---|
|保存LD2をcompile 0で再利用|通常method CLIの保存入力再生、通常make_executor、共有storeから取得。新Q3/compile各0、価格job未作成|
|未検証モデルの代表1点を空き枠へ|人工13round、4同時要求・複数runで一意の代表点。通常Q3の後に同じqueueでclaim|
|引継ぎ・切断・再開の重複防止|凍結dispatch intent、同時claim、応答喪失、同じattemptの回収、旧owner/UNKNOWN/部分成果のstore引継ぎ、relay lockを人工検査|
|待機中の通常探索継続|通常Q3・通常準備がoptionalより先に進み、検証待ち/失敗/終了時detachでも保存queueを保つ|
|metadata公開後の同一run再開|人工Git repositoryのmetadata commit後に同じrunから通常child入口を起動（childは人工代替）。科学コード変更・未追跡実装は拒否。queue再開で既存attemptを維持|
|C・同じマップ・ローカル0|通常SEARCH_PUBLICATION発見→C通常sync→公開map→ローカル0通常Syncの実ファイルを照合|

通常method CLIの再生は `normal_saved_entry/`。元の資格・十分N・科学的順位の再認定ではない。人工検査の数値を実H4結果に混ぜず、人工callをARTIFICIALとしてログへ保存した。実H4の通常Start/Resumeは今回起動していない。

保存表示の条件：Linear H4、1 Å、STO-3G、tol-only DF 1e−8・観測rank 10、one-bodyはhead、two-body prefix L_D=2、PR-Pauli/S2/K=2、保存binary64 h/実PF幅0.1、m5/q32/R33。Qiskit 1.3.0、opt1、seed_transpiler17、既存full controlled Hadamard X/Y wrapper・状態準備費除外の保存範囲。独立2経路の各軸68,228/68,300 RZ、X/Y各平均68,264 RZ、保存予測6169957875465/90343469、符号付き相対差2751307649/6167206567816 ≈ **+0.044612%**。

元receipt：[`0e1ce866…/COMPILE_RECEIPTS.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e1ce866e8f8d4e042801260fb8d543d0b1bee53/docs/research/d_pauli_ld2_b1_holdout_results_20261115/run01/M5_ROUND/COMPILE_RECEIPTS.json)。m6の1経路を2経路に格上げしていない。1点の観測差を全roundへ適用せず、期待価格/CI/正式G/winnerの資格も変更しない。

## 公開とローカル0

- ローカル公開metadata：[`187c1b088…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/187c1b088638d9d4cf0a298b56d64e8a91c59105/docs/research/romutail_controller_runs/normal_saved_entry)。PRICE_MODEL_VALIDATIONSとSEARCH_PUBLICATIONの既存経路。
- Cの通常同期済みマップ：[`ca5905e79…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ca5905e79d7654bdfe42a679b7b5e74aef80b1be/docs/research/h4_search_map_20261117)。同時更新で最初のpushがnon-fast-forwardになったが、相手の3 commitをmergeし、保存INDEXからHTMLを再生成して通常pushを完了した。force pushなし。
- [LOCAL0_SYNC.json](LOCAL0_SYNC.json) と [LOCAL0_MAP_CHECK.json](LOCAL0_MAP_CHECK.json) は、通常Syncが同じ公開SHAとGit blobを取得し、実閲覧フォルダで13モデル・LD2 m5/q32/R33・元receiptを確認した記録。
- Windowsの通常Previewは追加14a版でPASS：[記録](LOCAL0_PREVIEW_14a6102d4.json)。最終c4版はローカル通常CLIでPreview PASS：[記録](LOCAL_CONTROLLER_PREVIEW_FINAL.json)。

**残件：最終c4版のWindows通常Preview再検査とcurrent採用。** 並行移行中のWindows client更新後、live clientの再Previewは `Start-Process.ExitCode` がnullとなるtransportエラーで未完了。[観測ログ](LOCAL0_LATEST_CLIENT_PENDING.log)。同じ箇所が別セッションで修正中のため、稼働clientを重ねて修正していない。これはcontrollerの科学SOURCE不一致やcompile失敗ではない。過去のPreview PASSと、更新中clientの未確認を区別する。

ロードマップ本体の編集は行わない。技術的な配置・再開・診断公開の接続であり、新しい科学的優位性、最終総cost評価、価格資格を得たものではない。
