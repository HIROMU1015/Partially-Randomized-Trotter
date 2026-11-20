# H4 RPE分散探索の運用ルール

RomuTailの中央担当、Romulaptopの操作担当、各環境のworker担当が参照する。個別runの固定契約と利用者の最新指示を適用する。具体的な採用版、受付状態、適格な工程は稼働時に中央の設定と実機の応答で確認する。

## 目標と受入条件

利用者が通常入口で探索範囲を指定すると、RomuTailが保存成果を照合し、入力・descriptor準備、信号生成、価格用compile、監査、Q3を同じrunのキューに載せる。PBS、server2、RomuTail、受付中のRomulaptopは、各要求の固定条件と資源に適合する仕事を引き取る。PBSを含む複数の実行先で各工程を処理できれば、中央はホストごとの空きに応じて準備と評価を並行させる。回収した結果は元の候補へ戻り、探索、公開、マップ同期へつながる。

検収はホスト×方式×工程で記録する。保存入力照合、人工の配送・回収、新規の実処理と中央採用は、それぞれ実施した段階を記録する。PBSの同時実行は、適格な仕事が十分にある区間で、確保CPU時間、使用CPU時間、工程別の完了件数、待ち時間、空き理由を測って判断する。利用上限内で、必要な仕事量に応じて割当を増減・返却できることも確認する。

作業順と残件の経緯は[分散探索HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/codex/rpe-reviewed-fleet-20261118/docs/research/rpe_work_conserving_search_20261118/HANDOFF.md)、旧入力・共有mailbox・PBSの容量と依存関係を修正した経緯と検収範囲は[RomuTail運用版のHANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/codex/rpe-reviewed-fleet-20261118/docs/research/rpe_reviewed_fleet_20261118/HANDOFF.md)を参照する。

## 通常入口と役割

- 利用者は方式、探索範囲、要求幅・精度、shot配分、予算、および開始・停止・再開をCodexへ指示する。RomulaptopのCodexはその指示をRomuTailへ渡す。必要な準備タスクと割当はRomuTailが判断する。正本はRomuTailの `.rpe-central/state` にある。
- CodexはWindowsの `tools/local0-search/Invoke-RpeSearch.ps1` または共通CLIの `scripts/run_rpe_search.py` を通常入口として使う。利用者はCodexへ指示し、Start・ResumeはCodexが実行する。新規runは渡されたscopeで `Start`、既存runは指定run IDで `Status`・`Stop`・`Resume` を使う。すでに送信された指示や起動応答の時間切れは、先に指定runの状態と所有を照会する。実行許可と対象が明確な通常操作では既存入口の照合をそのまま使う。
- RomuTailが保存成果の照合、候補選択、工程別予約、結果採用、checkpointを担当する。ハミルトニアン入力やdescriptorの準備、信号生成、価格用compile・監査、Q3を、工程ごとに適格なworkerへ共通キューから配る。独立した信号と価格準備は並行し、証拠の揃った候補から同じ探索に戻す。次の探索判断に必要なQ3を優先し、余裕枠を独立した準備や先読みに使う。
- 科学実行は許可された範囲と予算内で行う。信号・compile・auditとQ3の予算を個別に管理し、準備予算の省略と明示的な0を区別する。compile予算のstage taskと実compileやwrapperの件数は別々に記録する。利用者が指定した `UNLIMITED` はQ3累計件数の設定として保持する。科学条件上未対応の候補は理由を記録して保留する。通常探索はほかの実行可能な候補で継続し、任意の価格モデル検証は低優先で進める。

## 継続run・配置を作るとき

元runの `APPLIED_SCOPE.json`、`ROUND.requested.json`、実際の `ROUND.json`、`CONTROLLER_CONFIG.json` と、採用済みreleaseの工程別manifestを照合する。変更した項目と根拠を記録し、以下を組で確認する。

| 対象 | 照合する内容 |
| --- | --- |
| 探索と状態 | 方式、範囲と両端、厳密数・binary64、候補生成、shot、MSE・費用定義、工程別予算と消費、候補進捗とcheckpoint。 |
| 保存実入力 | 入力本体とreceipt、descriptor、固定Git locatorとhash、controller側の保存先、workerの作業先。PR-Pauliは `candidate_completion.pauli_saved_inputs` と `pauli_signal_source` を確認する。 |
| 準備の配布 | `executor.distributed.preparation.bundle_validation_sources`、`controller_paths`、ホスト別 `preparation_profiles`、工程・provider別の適格性。新規workspaceでbundleの基点、Gitオブジェクト、保存結果の再利用時に読む固定コードまで解決する。 |
| 実行と資源 | worker・adapter・provider・runtime・固定科学SOURCEの組合せ、受付と資源、DPキャッシュの互換性、PBSの工程別walltime規則と専用回収Python。 |
| 成果の継承 | §10.5で入力・信号・価格・Q3を個別に照合し、完了receipt付きの一致結果を再利用する。元run、予算台帳、実行所有、未回収結果、原結果、STOPを保全する。 |

通常のResumeは記録済みのreleaseと入力を使う。運用設定の修正に別の継続runが必要なら、元runの新規受付を止めて、実行中の `job_id`・`attempt_id` と結果を照合してから切り替える。切断・応答欠落・`UNKNOWN` は元ホストのプロセスと保存receiptを先に調べる。再送や再開は既存の所有・採用を使い、未確定の同一評価と科学上のSTOPは保留する。

継続版の初回Resume前に、元runの準備台帳・保存結果が参照する固定Git locatorを継続版のcontroller checkout内で一括照合する。元checkoutにしかないオブジェクトは、固定SHAを保持したまま取得するか、保全した元オブジェクトストアを参照する。継承した台帳が参照する保存成果も維持する。

新規run用currentの切替は中央担当が行う。保存入力と変更経路の技術検収を済ませたreleaseを登録し、明示版のPreview、切替直前のcurrent・所有、切替後の版指定なしPreviewを確認する。稼働中runはそのrunの固定版を使い続ける。

## 固定runの配送版を通常再開で引き継ぐ場合

科学controllerを固定した既存runでは、中央SERVICEの `relay_transport_command` が
通常Start/Resume成功後、公開同期より先に指定runの配送版を選ぶ。
`ensure_rpe_relay_transport.py` は既存のREGISTRY.lock、run・invocation・CONFIG所有を
照合し、配送relayのみを切り替える。同じinvocation内の区間切替でrelayが再生成されても
run専用の補助プロセスが追従し、Stop・driver終了・invocation変更で終了する。
区間間の空きを科学実行需要とみなしてrelayを起動しない。

通常応答の `relay_transport.status=RELAY_TRANSPORT_CONFIRMED`、保存した
`distributed/relay/TRANSPORT.json`、`WATCH.json` と新しいheartbeatを確認する。
`RELAY_TRANSPORT_PENDING` は科学runの起動状態と別の配送保留である。
固定SOURCE・scope・raw・UNKNOWNは書換えず、通常の所有・STOP照合を維持する。

## 割当前に確認する入力と実行環境

- 工程・providerごとに、固定SOURCEの依存関係をREQUESTに記載し、実行先でその工程に使う固定Pythonを選ぶ。割当前の適格性照合では、要求したパッケージの版と、必要なimportがそのPythonで成功することを確認する。子プロセスで固定providerを読む経路を変えた場合は、その子プロセスでも確認する。不足時はホスト・工程・依存名を記録し、該当する割当を保留する。対応する別ホストや独立した候補は進める。
- PBSのPR-Pauli信号では、固定SOURCEの `requirements.txt` に従って `openfermion`、`pyscf`、`openfermionpyscf` の版を照合し、専用の信号Pythonで `openfermionpyscf` のimportを確認する。依存関係は対象releaseの専用環境へ配置する。通常回収とPBS終了後の回収は、双方でprofileに指定した専用回収Pythonを使う。回収経路を変更したら、巨大な整数を含む保存OUTCOMEが厳密値のまま回収できることを確認する。
- 変更したQ3 adapterの固定SHAは、release manifest、ホストの基本・route別profile、ROUNDに埋め込まれたローカル・PBSのprofile、PBS所要時間ルールのすべてで照合する。保存要求のdry-checkは選択されたホストの実パスと固定Pythonで行う。短いMを使う場合は元のsignal/priceから対応範囲を照合し、固定科学Q3の検証域外は候補単位で保留する。
- 旧パスの入力receiptが使えない場合は、後続の照合済み保管先も調べる。固定SOURCE、入力ハッシュ、構成、検査完了の記録が一致した成果を再利用する。bundleを変更したら、既存Gitキャッシュに依存しない新規workspaceで固定入力を復元できることを確認する。
- 遠隔relayは、空きのあるworkerごとに準備要求を1回の照会で1件選び、入力bundleを届けて適格性を照合する。照合結果を中央へ報告し、残りの準備bundleを待たずに通常の割当照会へ進む。繰り返し保留された要求はホスト別の照合回数に応じて交替させる。Q3のまとめ照合は従来の経路を使う。照合成功は適格性の確認として記録し、実行開始と中央回収はworkerの記録と台帳で確認する。

## 切断・失敗後の回収と再試行

- 共有mailboxから送る状態通知は、現行runが所有するホスト・`job_id`・`attempt_id` に絞る。旧runの通知やACKは所有元に残す。現行runの完了結果を先に回収し、新規割当の照会を続ける。PBSの投入応答が欠けた場合も、スケジューラーと共有spoolで元のattemptを照合してから次の処理を決める。
- 失敗したattemptのREQUEST・OUTCOME・終了記録を保全する。再試行が許され、元attemptの終了と同一評価の所有を照合できたときは、同じ固定SOURCE・入力・数値条件で不足分だけを再実行する。回収した信号は元と同じ候補IDへ戻し、後続のQ3まで確認する。終了不明の評価は個別に保留し、独立した候補は継続する。
- 稼働中の固定runと並行して候補を限定した復旧を行う場合は、対象・元run・照合した原本と復旧結果の所在を記録する。復旧結果を親runで使う際は通常の保存評価照合を通す。

## 実際に動くことの確認

1. 通常CLIまたはWindows入口でPreviewを確認する。入力配布を変えた工程は、対象ホストの新規workspaceで保存REQUESTから固定入力とruntimeを照合する。準備とQ3は工程ごとに確認する。
2. 変更した経路の人工runを通常入口から通し、workerの開始・終了、結果回収、中央採用、同一run再開の追加起動と二重採用が各0であることを照合する。人工結果は検証用の保存先で扱う。許可済みの実探索では、該当工程が実際のworkerで開始し、保存結果を中央で回収できたところまで確認する。
3. 報告は「受付・割当」「workerで実行開始」「結果回収と成否」「科学結果の採用」を分ける。`ASSIGNED` は割当済み、`STARTING` は起動処理中、`ACCEPTED` は中央が転送成果を照合して保存した状態。準備の成否は `OUTCOME.json` と終了receipt、実Q3開始は `Q3_BODY_START` などのworker記録で判断する。ホスト・工程ごとに確認済みの段階と具体的な保留理由を残す。
4. 公開は既存のC同期経路を使う。Romulaptopの `H4-search-map/index.html` まで、開始・再開時、実行中の10分間隔、終了時の同期状況を確認する。公開待ち・同期失敗は探索の稼働状態と分けて報告する。

## 価格用compileから監査への引継ぎ

- PR-Pauliの監査要求は、各グループの `PRICE_STATE.json` が `COMPILE_READY_FOR_AUDIT` になった成果から作る。compileの原結果を保存したGit commitの `BINDING.json`、`COMPILE_RECEIPTS.json`、`RESULT.json` と、入力を固定したcommitの `CONFIGURATION.json` を、commit・パス・SHA-256付きの監査入力として渡す。原結果が別のGitブランチにある場合も、そのcommitで該当ファイルを読めるように既存bundleへ含める。監査が参照する回路ファイルや実行記録は、receiptにあるハッシュを使って受け渡す。
- 配送経路を変更したら、対象workerの新規作業領域で、compile結果を作ったGitブランチに直接アクセスせずに、受け取ったbundleから4ファイルを照合する。保存入力で監査の `--build` を通し、各経路・軸の参照ファイルが解決できることを確認する。実監査の完了・中央採用は、それぞれの結果と台帳で確認する。
- 入力が届かず監査が止まった場合は、元のcompileの `job_id`・`attempt_id`、完了receipt、原結果を先に照合する。成果がそろっていれば同じcompile結果から配送を復旧する。失敗した監査の記録と、科学実行のSTOP・再試行条件を保全する。

## 空き枠・容量・障害の切り分け

- PBSの100 CPUとserver2の30 CPUはCPUの使用上限。PBSは採用版の工程・ルート別見積りを使い、適格な仕事の数と必要CPU・RAM・残りwalltimeから割当要求を決める。RomuTail・Romulaptopには追加CPU上限を置かず、既存autoと実機条件を使う。Romulaptopは起動中で受付ONのとき参加する。ホストのPauseは新規受付に反映する。server1は復帰後、固定版の配置と対応工程の検査を経て参加する。
- 割当が少ないときは、`QUEUE_WAIT_STATE` と工程・ホスト別の台帳で、候補不足、準備待ち、適格なホスト不足、配送・割当待ちをたどる。旧attemptが枠を占めている場合は、元ホストの状態と台帳上の所有を照合する。同時実行数を増やすときは、ホストでの実測、1ジョブあたりのDP並列数、合計CPU・RAM、既存予約を確認する。
- 空き枠と候補の補充は、worker完了・枠追加・候補の準備完了に合わせて進める。独立した準備は残りの工程別予算と適格な枠に合わせて進める。PBSのCPU要求には予算内で割当可能な仕事を使い、同じ枠に後続タスクが入ったか、入力照合から実行開始までの時間と実使用CPU時間を記録する。
- PBSのタスク別予約では、適格なREQUESTが選ぶ固定profileからCPUを、工程別の所要資源ルールからメモリを求める。準備工程のメモリが未計測なら同時実行から保護するため枠の使用可能メモリ全体を予約する。割当前に台帳の所有中attemptを含めて合計を照合する。新規枠は適格な需要から計算し、実行中の割当は元へ回収してから拡張する。
- workerの作業領域・未回収結果・キャッシュは、各環境50 GiBの受付制限に従う。`STORAGE_LIMIT_REACHED` が出たら、対象workerの使用量と受付状態を確認する。転送ack済みのdry-check一時領域、正常完了した準備の展開コピー、回収済みbundleの作業コピーを優先して整理する。それでも上限に達していれば、基本profileに加えて経路別profileの保存先も確認し、再生成できるDPチェックポイントを既存の整理処理で削減する。
- workerの使用量が50 GiB未満でも、保存先のファイルシステムの空きが最低値を下回ると `FILESYSTEM_SPACE_LOW` で新規受付を保留する。現行実装の最低値はworker上限の10%（50 GiBなら5 GiB）。`QUEUE_WAIT_STATE` またはworkerの容量報告で `filesystem_free_bytes`、`filesystem_min_free_bytes`、`filesystem_low_space_roots` を確認し、空きが不足した保存先で既存の安全な回収処理を使う。起動済み処理の回収と他の適格なホストの受付は継続する。
- RomuTailでさらに整理が必要な場合は、登録上そのrunだけが稼働し、ローカルworkerに割当済み・実行中・回収待ちの作業がないことを確認する。同じrunの中央原本とSHA-256が一致する作業用キャッシュのコピーだけを整理する。待機中、準備中、未確定の要求が参照するデータ、原結果、固定入力、失敗記録は保全する。
- PBSの信号は、要求登録から割当までの時間をbundle転送、固定入力照合、受付・予算待ちに分けて記録する。大きな既存bundleの先行配置は、対象runの要求、PBSの残容量、転送前後のSHA-256を確認する。保存入力の木（tree）を照合するときは、既存の `git show` 表示に対するハッシュを継承する。
- 容量整理では回収済み作業領域の同一Git packを内容照合して共有できる。過去のメタデータキャッシュを圧縮する場合は、取込みと同じロックで作業し、Gitオブジェクト一覧と新packの整合を検証する。実行・回収中や終了未確定のattemptは整理対象にしない。
- WSLやPBSで追加のキャッシュ整理が必要なら、受領済みの中央保存データとのハッシュとサイズの一致、転送ACK、実行中・未回収・未確定attemptから参照されていないことを確認する。共有のGitメタデータを整理する場合は、元bundleで再生成できることを確認し、取込みと同じロックを使う。
- RomuTailのユーザーサービス `rpe-storage-recovery.service` が `scripts/reclaim_rpe_worker_storage.py --state-dir STATE_DIR` で稼働中のrunを追跡する。約1分ごとに既存のSSH経路で点検し、点検結果は `journalctl --user -u rpe-storage-recovery.service` に残る。50 GiB上限のworkerは40 GiBに達した時点で、回収済み作業コピー・再計算可能なキャッシュの整理を始める。40 GiBは保守開始の基準で、受付判定には既存の50 GiB上限とファイルシステム空き容量の条件を使う。空きが戻れば既存workerの受付が自動復帰する。ロックとGitオブジェクトの一致を確認できる場合は、旧メタデータpackも容量外の専用一時領域で圧縮する。対になるpack・idxを持ちGit検証を通る `.rev` は整理対象に含め、オブジェクト集合を保つ。個別runを調査するときは、常設サービスと同時に同じrunを操作しない。
- 先行整理は35 GiBを目安にし、整理できる安全なコピーが少ない場合は実際に空けられた量を記録する。worker使用量が上限未満で、対象の保存先すべての空きが最低値以上になれば、次のworker照会で容量による受付保留が自動で解除される。利用者のPause・受付OFFとWSL停止は各環境の設定に従う。実行可能な仕事が待機中なら、`QUEUE_WAIT_STATE` と台帳で新規割当まで確認する。

2026-11-20に、`rpe-storage-recovery.service` を `.rpe-central/deployment/worker-storage-recovery-6bd2d0f254/` へ切り替えた。40 GiB未満では先行整理せず、40 GiBに達した時点で受付を維持したまま35 GiBを目標に整理することを人工検査した。通常のworker照会には既存の受付判定と緊急時の整理条件を使う。保守サービスの報告には `reclaim_start_bytes`、`reclaim_target_bytes`、`reclaim_checked`、`reclaim_pending` を残す。

2026-11-20のRomuTailでは、準備メタデータ用Gitキャッシュ3組を別領域で再圧縮し、全Gitオブジェクトの集合と参照の解決結果が一致することを前後で照合した。ローカルworkerが容量保留中で実行中のattemptがないことを台帳・実機から確認し、容量測定ロックと各キャッシュの取込みロックを確保して切り替えた。元の一時pack、科学結果、未確定attemptは保全した。worker容量は50.011 GiBから39.313 GiBとなり、同じworkerの受付ONと空き2枠を中央で確認した。切替記録は `.rpe-central/storage-maintenance/local-pack-20261120/APPLIED.json` に保存した。

## RomuTail中央の常設保守

- `rpe-central-maintenance.timer` は、登録済みrunの中央台帳と固定版から、依存入力の照合、停止後の回収、待ち理由の集約を行う。優先対象は現在のrunと、稼働中に追跡を開始して所有中attemptが残るrun。常設保守から通常Start・Resume、新規予約、割当、科学計算、再試行は行わない。
- 停止後は元のhost、job、attempt、REQUESTのSHAを照合する。PBSでは既存gatewayの `snapshot`、`harvest` を使う。SSH・WSLでは保存状態と特定attemptの `result` を使う。割当済み処理を起動できる一般のmailbox `status` は使わない。元の完了receiptと保存結果が照合を通った分だけ同じJobStoreへ受領し、科学的な採用は通常探索に委ねる。終了証跡が揃わない元attemptの所有・STOP・予算は保持する。
- 待機中の監査等が参照する中央CASの入力が欠けたときは、元の成功・受理済み準備jobの読み取り用成果から、元manifestのSHAと一致するものだけ戻す。通常relayが不足分を配送する。原結果や固定入力の欠落が残れば該当依存関係を保留する。
- Statusの `central_maintenance` に最新観測時刻、キューの待ち理由、hostとrelay・PBSの状態、自動回収・復旧の結果を返す。正本のjob状態はJobStoreを使う。保守記録は `.rpe-central/state/maintenance/` に保存する。
- 終了runの配送bundleは `ARCHIVE_CANDIDATES.json` に候補を記録する。稼働controllerがない間に、終了・採用・参照・所有を照合し、空の作業領域で元SHAの復元、入力照合、保存成果の再読を通す。新しいアーカイブ分を含めて削減できる場合に限り、復元記録を先に固定して配送キャッシュ本体を整理する。現在の対象run、未確定run、受領・採用前の成果、別runから参照中のbundleは対象外とする。

RomuTailには `.rpe-central/deployment/central-maintenance-2c68c0544b/` の固定ツールとユーザータイマーを配置した。タイマーは約1分ごとに起動し、重い容量整理の判定は約5分ごとに行う。定期処理で既存runの起動・停止・再開は行わない。通常Statusに保守記録が出ることと、ホスト側で稼働中のcontrollerがある間は容量整理を延期することを確認した。保存済み旧runの2件で配送bundleのSHA、空の作業領域への入力、保存成果の再読を照合した。人工ジョブでは同じattemptからの受領を一度だけ行い、欠けた監査入力を元の成果から戻した。今回の新規科学実行は0件。

## 探索データの保存期間と整理

| 区分 | 対象と扱い |
| --- | --- |
| 保存する | 科学SOURCEと固定入力の参照先、探索範囲・評価条件、候補と予算の台帳、checkpoint・STOP、元REQUEST、原結果・終了receipt・採用記録、信号・価格・監査の再利用に必要な成果を保全する。終了済みrunの成果も参照元とSHA-256を辿れるようにする。失敗・UNKNOWNの唯一の証跡、再開に必要な部分成果も保全する。 |
| 短期間だけ保存する | workerの展開入力・作業ディレクトリ・転送用コピーは中央回収の確認まで保持する。失敗後の診断用ログや作業ディレクトリは、終了と原因の確定まで保留し、部分成果や再試行の根拠は保存対象へ移す。診断が終わった旧runの作業コピーをT7へ移す場合は、元パス・SHA-256・復元手順を残す。DP prefixや配送のキャッシュは再利用のための作業領域として上限内で運用する。 |
| 別コピーを保存し続ける必要がない | 中央への回収が確認済みの展開コピー、回収済みで中央の参照先と内容一致を確認したworkerキャッシュ、完了済みの試験用展開入力が該当する。科学処理を開始していないと明示した準備失敗は、受領済みの失敗receiptとログを残した上で、出力を持たない展開入力・作業コピーを整理できる。 |

状態は時刻だけで判定せず、元run・job・attemptの所有、終了receipt、中央受領、参照先を照合する。
実行中・回収待ち・UNKNOWNの作業領域や、計算本体開始後の失敗の部分成果は自動整理しない。
中央の `distributed/artifacts` は後続候補から参照されることがある。別パスに同じinodeが見える場合は共有分を二重計上せず、中央のハッシュ付き原本と配送コピーの役割を確認する。

workerの使用量が40 GiBに達したときは、中央側と内容を照合できる配送キャッシュも、既存の所有・保護・ロック条件が揃ったものから35 GiBを目安に整理する。
Gitの正式packと参照先は保存対象とし、停止済みの一時packは正式packと区別する。
既存データを移すときは、実際に空く容量と現行runからの参照を確認して対象を絞る。診断用ログの保管や重複の削減は、構造化された台帳と原結果の保存を前提に行う。

### 再開・監査・再現に必要な長期保存

- Q3は採用結果だけでなく、後続候補の重複照合に使う確定済み評価を保存する。構成・配分・評価契約、元REQUEST、厳密なMSE上下界・判定、終了receipt、照合に必要な証跡を同じjobに結び付ける。作業用DP cacheと同じ評価の再送は追加の保存結果にしない。
- 価格較正・独立監査は、元のコホート区分、pathと軸ごとのreceipt、seed、event digest、compiler・回路fingerprint、計測値、採択モデルと判定を保存する。同一pathのX/Yは個別receiptを持ち、同じイベント原本はSHA-256が一致する1部を参照する。現行の保存軸の再利用はreceiptを読む。元イベントが科学契約上必要な場合や凍結実装での厳密再現を確認できていない場合は、イベント原本1部も保持する。
- 準備の中央受領時は、ハッシュ付き原本とjobの読み取り用参照を同じ実体で保持する。計算用の展開入力や、後続処理が書き換える候補作業ファイルには独立したコピーを使う。既存の別コピーを整理するときはSHA-256と利用箇所を照合する。
- 配送bundleは、固定SOURCE・入力commitとパス・SHA-256、生成した入力の原本、必要なGitオブジェクトと復元手順に分ける。保存済みbundleを短期扱いにする前に、空のキャッシュと固定SOURCEから元のREQUESTが要求するbundleのSHA-256、入力HEAD、各入力のハッシュを照合する。受領済み結果のGit差分の取込みと、後続の保存結果再利用も通す。固定SHAのbundleを復元できない既存REQUESTでは、bundle本体を保持する。

#### 配送bundleの復元と整理

保管する単位は原REQUEST、固定SOURCE、入力のcommit・パス・SHA-256、生成入力の原本、および復元元のGit refとオブジェクトとする。復元元は短期workspaceから独立して保存し、参照先も記録する。確認済みの配送bundle本体は短期キャッシュとし、同じバイト列を長期アーカイブへ重複保存しない。計算で新たに生成したGit差分bundleは結果の原本として扱う。

復元は固定REQUESTを使い、削除や科学計算を伴わない確認から始める。

```bash
python3 scripts/restore_rpe_preparation_bundle.py --request "$request_file" --sha256 "$bundle_sha"
```

別の保管済みGit checkoutを参照するときは `--repository "$source_repository"`、元のSHAの位置へ戻すときは `--restore-into "$artifact_cache"` を加える。checkoutからの再生成は固定SOURCEだけを前提に生成された `inputs-*` を対象とする。SHA-256が一致した場合だけ書き込む。失敗した場合は元bundleか固定オブジェクトを回収する。Gitの版や設定を変えた場合もSHA-256を再確認する。

一時checkoutから独立させる場合は、`archive_rpe_preparation_inputs.py` で元のGit pack内の圧縮済みオブジェクトを一度ずつ保管する。各bundleのヘッダー・部品順序・末尾チェックサムも保持するため、復元時にcheckout、alternates、Gitの再圧縮を使わず、元のバイト列に戻せる。アーカイブ作成時は全件のSHA-256を照合し、元キャッシュを残す。アーカイブの増分も含めて容量が減る対象を選ぶ。

```bash
python3 scripts/archive_rpe_preparation_inputs.py \
  --request "$first_request" --request "$second_request" --output "$archive"
```

配送キャッシュを整理した旧runは、RomuTailの通常Resumeで、起動・終了記録とSTOPの既存判定を通った後に復元する。対象runの回収記録 `*.reclaimed.json` と `ARCHIVE.json` を照合し、正式に除いたbundleのうち不足する同一SHAを復元する。すでに同一SHAが存在すれば再利用する。復元後に通常の§10.5照合を行い、開始時の固定版に渡す。復元件数は `DUPLICATE_PREFLIGHT.json` の `preparation_archive_recovery` に記録する。復元が失敗した場合はResume応答に `preparation_archives` の理由を返す。

作業領域やアーカイブを単独で点検する際は、次の復元コマンドを使える。

```bash
python3 scripts/restore_rpe_preparation_bundle.py --archived-run "$run_directory" --archive "$archive"
```

中断でResumeが保留された場合、RomuTailの通常入口は `resume_reconciliation` に対象run、最後のinvocationと対応終了証跡の有無、未回収・実行所有中・未確定のjob、元hostとattempt IDを返す。`RUNNING` は台帳上の記録なので、実機での継続は元workerで確認する。部分書込みだけの準備成果や診断用Q3ファイルは、完了receiptや有効なDPチェックポイントと区別する。

RomuTailの通常入口は `.rpe-central/CLIENT_GATEWAY.py` を使い、復元処理は `.rpe-central/deployment/resume-archives-a1dbe26e87/` に固定した。同ディレクトリに切替前のbootstrapと適用差分を保持する。登録release・実行時の科学SOURCE・固定profileは各runの記録から選ぶ。旧固定版を使う人工Resumeで、終了証跡が欠ける場合の保留と、終了証跡がある場合のSHA一致復元を確認した。

既存runの整理では、起動と終了の記録が同じinvocationであること、未確定attemptや配布・回収待ちの要求が対象を参照していないこと、元の固定版でResumeする前に復元できることを確認する。復元元が候補の作業checkoutや一時的なGit alternatesにしかない場合は、復元元を保全するまで実体を残す。未検査の `workspace-*` やbase bundleも保持する。

2026-11-20に、保存済みrun `search-e6361f01b6a480718127` の次の2件を照合した。元のbundle本体を再生成元に使わず、空のworkerキャッシュと固定SOURCEから新規workspaceを作った。中央では受理済みの台帳・receiptを参照し、保存結果を別領域へ取り込み、出力をハッシュ照合した。

| 工程とjob | 配送bundle | 固定Git入力／作業入力 | 保存結果の再利用 |
| --- | ---: | ---: | --- |
| PF_COMPILE `prep-80abb4e4c35046236650ffc7` | 46,527,487 bytes、元SHA一致 | 24／9 | 23成果のhash一致。返却Git差分から元の結果commitを取得 |
| DF_SIGNALS `prep-92c07e13c05743421b2c679c` | 195,064,029 bytes、元SHA一致 | 127／27 | 33成果のhash一致。保存信号を別workspaceへ復元 |

2件で合計241,591,516 bytes（約230 MiB）が短期キャッシュ化の候補となった。検査時の元runは起動と終了のinvocationが一致せず、台帳にも未割当・未確定が残るため、元キャッシュを保全した。2件の検査で確認したのは準備成果を同じSHAで再利用するところまでで、新規科学計算は0件。

2026-11-20の独立アーカイブ検収では、24件すべてを元SHA-256へ復元した。終了記録と台帳が一致する4 runで、workerの空のキャッシュへの取込み、中央の元receiptと保存結果の再読まで通った21件のキャッシュ本体を整理した。元REQUEST、成果、STOP、未確定の試行は保持した。

| RomuTailの保管先（`.rpe-central/archives/preparation-inputs/`配下） | 整理した旧run | 配送キャッシュから除いた件数 | 保管分を差し引いた削減 |
| --- | --- | ---: | ---: |
| `pilot-20261120-pack-entries` | `search-51ec4b40deb95d9b8e5b` | 5 | 176,418,816 bytes |
| `closed-runs-20261120` | `search-9c1865e787c6aab3d251`、`search-2bf68d58d3d7c32243c8`、`search-eea38701fa5a6a593e9f` | 16 | 1,155,612,672 bytes |

実割当容量で合計約1.24 GiBを削減した。整理後に元キャッシュがない状態から21件すべてを、Gitが使えない状態の別キャッシュへ復元し、SHA-256を確認した。現行runの2件と、旧runの保存結果に採用記録がなかった1件は独立アーカイブでSHAを照合したうえで、元キャッシュを残した。新規科学計算は0件。
