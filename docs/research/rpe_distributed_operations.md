# H4 RPE分散探索の運用ルール

RomuTailの中央担当、Romulaptopの操作担当、各環境のworker担当が参照する。個別runの固定契約と利用者の最新指示を適用する。具体的な採用版、受付状態、適格な工程は稼働時に中央の設定と実機の応答で確認する。

## 目標と受入条件

利用者が通常入口で探索範囲を指定すると、RomuTailが保存成果を照合し、入力・descriptor準備、信号生成、価格用compile、監査、Q3を同じrunのキューに載せる。PBS、server2、RomuTail、受付中のRomulaptopは、各要求の固定条件と資源に適合する仕事を引き取る。PBSを含む複数の実行先で各工程を処理できれば、中央はホストごとの空きに応じて準備と評価を並行させる。回収した結果は元の候補へ戻り、探索、公開、マップ同期へつながる。

検収はホスト×方式×工程で記録する。保存入力照合、人工の配送・回収、新規の実処理と中央採用は、それぞれ実施した段階を記録する。PBSの同時実行は、適格な仕事が十分にある区間で、確保CPU時間、使用CPU時間、工程別の完了件数、待ち時間、空き理由を測って判断する。利用上限内で、必要な仕事量に応じて割当を増減・返却できることも確認する。

作業順と残件の経緯は[分散探索HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/codex/rpe-reviewed-fleet-20261118/docs/research/rpe_work_conserving_search_20261118/HANDOFF.md)、旧入力・共有mailbox・PBSの容量と依存関係を修正した経緯と検収範囲は[RomuTail運用版のHANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/codex/rpe-reviewed-fleet-20261118/docs/research/rpe_reviewed_fleet_20261118/HANDOFF.md)を参照する。

## 通常入口と役割

- Romulaptopは方式、探索範囲、要求幅・精度、shot配分、予算、および開始・停止・再開の指示を渡す。必要な準備タスクと割当はRomuTailが判断する。正本はRomuTailの `.rpe-central/state` にある。
- Windowsの `tools/local0-search/Invoke-RpeSearch.ps1` と共通CLIの `scripts/run_rpe_search.py` が通常入口。新規runは渡されたscopeで `Start`、既存runは指定run IDで `Status`・`Stop`・`Resume` を使う。すでに送信された指示や起動応答の時間切れは、先に指定runの状態と所有を照会する。実行許可と対象が明確な通常操作では既存入口の照合をそのまま使う。
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

## 割当前に確認する入力と実行環境

- 工程・providerごとに、固定SOURCEの依存関係をREQUESTに記載し、実行先でその工程に使う固定Pythonを選ぶ。割当前の適格性照合では、要求したパッケージの版と、必要なimportがそのPythonで成功することを確認する。子プロセスで固定providerを読む経路を変えた場合は、その子プロセスでも確認する。不足時はホスト・工程・依存名を記録し、該当する割当を保留する。対応する別ホストや独立した候補は進める。
- PBSのPR-Pauli信号では、固定SOURCEの `requirements.txt` に従って `openfermion`、`pyscf`、`openfermionpyscf` の版を照合し、専用の信号Pythonで `openfermionpyscf` のimportを確認する。依存関係は対象releaseの専用環境へ配置する。通常回収とPBS終了後の回収は、双方でprofileに指定した専用回収Pythonを使う。回収経路を変更したら、巨大な整数を含む保存OUTCOMEが厳密値のまま回収できることを確認する。
- 変更したQ3 adapterの固定SHAは、release manifest、ホストの基本・route別profile、ROUNDに埋め込まれたローカル・PBSのprofile、PBS所要時間ルールのすべてで照合する。保存要求のdry-checkは選択されたホストの実パスと固定Pythonで行う。短いMを使う場合は元のsignal/priceから対応範囲を照合し、固定科学Q3の検証域外は候補単位で保留する。
- 旧パスの入力receiptが使えない場合は、後続の照合済み保管先も調べる。固定SOURCE、入力ハッシュ、構成、検査完了の記録が一致した成果を再利用する。bundleを変更したら、既存Gitキャッシュに依存しない新規workspaceで固定入力を復元できることを確認する。

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
- RomuTailでさらに整理が必要な場合は、登録上そのrunだけが稼働し、ローカルworkerに割当済み・実行中・回収待ちの作業がないことを確認する。同じrunの中央原本とSHA-256が一致する作業用キャッシュのコピーだけを整理する。待機中、準備中、未確定の要求が参照するデータ、原結果、固定入力、失敗記録は保全する。
- PBSの信号は、要求登録から割当までの時間をbundle転送、固定入力照合、受付・予算待ちに分けて記録する。大きな既存bundleの先行配置は、対象runの要求、PBSの残容量、転送前後のSHA-256を確認する。保存入力の木（tree）を照合するときは、既存の `git show` 表示に対するハッシュを継承する。
- 容量整理では回収済み作業領域の同一Git packを内容照合して共有できる。過去のメタデータキャッシュを圧縮する場合は、取込みと同じロックで作業し、Gitオブジェクト一覧と新packの整合を検証する。実行・回収中や終了未確定のattemptは整理対象にしない。
- WSLやPBSで追加のキャッシュ整理が必要なら、受領済みの中央保存データとのハッシュとサイズの一致、転送ACK、実行中・未回収・未確定attemptから参照されていないことを確認する。共有のGitメタデータを整理する場合は、元bundleで再生成できることを確認し、取込みと同じロックを使う。
- RomuTailのユーザーサービス `rpe-storage-recovery.service` が `scripts/reclaim_rpe_worker_storage.py --state-dir STATE_DIR` で稼働中のrunを追跡する。約1分ごとに既存のSSH経路で点検し、点検結果は `journalctl --user -u rpe-storage-recovery.service` に残る。50 GiBの95%から、回収済み作業コピー・再計算可能なキャッシュを整理する。空きが戻れば既存workerの受付が自動復帰する。ロックとGitオブジェクトの一致を確認できる場合は、旧メタデータpackも容量外の専用一時領域で圧縮する。対になるpack・idxを持ちGit検証を通る `.rev` は整理対象に含め、オブジェクト集合を保つ。個別runを調査するときは、常設サービスと同時に同じrunを操作しない。
- 整理後は45 GiBを目安に再計測する。50 GiB未満になれば、次のworkerの照会で容量による受付保留が自動で解除される。利用者のPause・受付OFFとWSL停止は各環境の設定に従う。実行可能な仕事が待機中なら、`QUEUE_WAIT_STATE` と台帳で新規割当まで確認する。上限を下回れない環境は新規受付を保留し、回収と他の適格な環境での探索を続ける。
