# PR最適化の空き枠補充と評価順

配置・継続・障害復旧の共通手順は[H4 RPE分散探索の運用ルール](../rpe_distributed_operations.md)を参照する。

中央固定SOURCEは `5c8ced4b2d407a3828d189e1e0f0a24a4807f94a`。採用currentは `romutail-work-conserving-5c8ced4b2`。中央状態はRomuTailの `.rpe-central/state`。旧current `romutail-auto-preparation-5b48aabbc` の運用設定に、検収済み後着READYと固定 `1e5d63c326282382a311304e21d995ac09954003` の補充・順序選定と保存Q3索引を統合した。切替時に既存run 30件の固定設定を照合し、起動中のcontroller・relayがないことを確認した。R08の切替スクリプトと固定入力は参照した。

## 通常操作

既存scopeでPR-Pauli／PR-DFの要求幅を指定する。WindowsとRomuTailのPreviewが返す `applied.work_conserving_search=true` で適用を確認できる。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\SCOPE.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId RUN_ID
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId RUN_ID
```

共通入口は `scripts/run_rpe_search.py --config "$client" --action preview --request-file SCOPE.json`。Startは対象・範囲の承認と§10.5照合を済ませた要求で行う。必要に応じてWindowsは `-Release`、共通入口は `--release` に新release IDを明示できる。同じrunのStatusとResumeは保存releaseに従う。

## 共有キューと計時

V2が選ぶ要求Q3を先に予約し、許容予算内で独立READY候補を補充する。worker完了とhostの新しい受付通知で選定待ちを更新し、空き枠候補があれば既存のrequest別適格性判定・PBS walltime判定を通して割り当てる。各幅を交互に準備し、幅内のLD・物理κ schedule規則と方式別準備上限を引き継ぐ。Q3とsignal／compile／auditの予算・STOP・job／attemptとreceiptの照合は既存台帳を使う。採択の手続と別に、既存の確認済み最安の検査が受入可能な先読み成果も取り込む。

方式内最安を更新し得るREADY候補を `削減可能RZ × 暫定適合率 ÷ 推定Q3時間` で比較する。分散時の時間は同一構成・route・固定host profileで受理したQ3本体の完了観測を使い、転送や待ち時間を含めない。比較対象に観測不足がある場合と再開直後は既存順に戻る。未評価候補への定期的な探索枠はV2に従う。

`PROGRESS.jsonl` の `Q3_STANDBY_REFILL`、`CANDIDATE_READY`、`LAUNCH`、`SEARCH_ADOPT.order_reason`・`order_baseline`、`DRIVER_PROGRESS.jsonl` の `PREPARATION_WAVE_START` を使用する。空き枠の理由は `QUEUE_WAIT_STATE` で候補不足、準備待ち、適格host不足、割当待ちを区別する。既存の `scripts/summarize_rpe_q3_timing.py` で集計できる。

## 人工検収

WindowsとRomuTailは、明示版とcurrentのPreviewで同じrun IDを解決した。回帰115件・サブテスト4件がPASS。

- RomuTailとserver2の実機2台を既存SSH mailboxで接続した。RomuTailがPR-Pauliを実行中に、server2が信号と価格の準備を終えたPR-DFを同じIDで自動shot評価へ入れた。中央受領・採用を確認した。Q3はRomuTail 2件、server2 1件。信号とcompileは各1件。全5件の割当・受領・採用は各1回。準備完了から割当まで0.562秒、結果受領から探索採用まで0.031秒。完了runを再開した際の追加起動は0件。
- 人工RPCで、優先Q3が既存hostで実行所有中に、後から参加した別hostが空き枠で3件を連続処理した。Q3の上限4件と同時2枠を使用した。1 hostの人工workerが実際に返した2つのQ3本体時間で、順序が従来の候補から速い候補へ切り替わった。
- 独立worker spool 2つで通信を切り、元workerの保存結果を同一attemptで回収した。別試験でcontroller停止・再開を挟み、元attemptの実行は1回、完了済みの追加起動は0件。準備とQ3の予算超過は0件。
- RomuTail、server2、Romulaptop WSL、PBS gatewayの固定配置を検証した。server1は停止扱い。今回の人工投入はRomuTailとserver2で行った。通常SERVICEのPBS実Q3見積りは従来の未設定。PBSで実Q3を選択するには対象runの時間見積りとrequest別適格性が必要。

実H4信号・実compile・実Q3・価格fit・本番探索の新規起動は各0件。元runの科学結果、未確定attemptとSTOPを保全した。[検収データ](ACCEPTANCE.json)

## PR-Pauliの保存入力と信号準備

`romutail-pauli-inputs-8d03cd254-r2` は、旧 `romutail-pbs-sizing-48ebf75d5` を基点に、PR-Pauliの候補構成と信号計画が固定の保存入力を各hostで解決できるようにしたrelease。controllerと信号wrapperは `8d03cd254db4ae5f7b4d786f54e4fc66f0fcb4a8`。Pauli providerは `8478b8200286815ad7ba2652e535d4756ca3c330`、Q3の計算SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774` を使用する。

通常の2つのROUNDで `candidate_completion.pauli_saved_inputs` に、固定 `SOURCE_INDEX.json` の3つの元パスとRomuTailの保存場所を設定する。信号workerには既存の内容ハッシュによる配布経路を使い、receipt、snapshot、H4 contextの計26,861バイトを割当の作業ディレクトリに置く。controllerとworkerの両方で固定indexのSHA-256を照合する。PBSは同じ信号処理に既存の120秒見積りを引き継ぎ、専用Git保管先で不足した履歴オブジェクトを補完した。CPU・メモリと同時実行の設定は従来どおり。

保存候補32件で構成生成を確認した。RomuTail、server2、Romulaptop WSL、PBS gatewayで同一の保存REQUESTを検証用作業ディレクトリに展開し、13ラウンドの信号計画生成と3ファイルのSHA照合を確認した。PBS計算ノードへの投入は0件。関連回帰36件がPASS。別の準備回帰で失敗した5ケースは旧releaseでも同じ箇所で再現した。[保存入力の検収](PAULI_SAVED_INPUTS_ACCEPTANCE.json)

新規runの通常Previewは、Windowsと共通CLIの両方から同じreleaseを選ぶ。旧 `search-8b10307fd1b9fc375b6b` は、そのrunが記録したreleaseで照会する。停止理由は `V2_RESULT_REQUIRES_REVIEW` で、終了receiptがないWSLのattemptが4件残る。継続時はCodexが旧台帳・元host・receiptを照合し、既存の継続手順を使う。server1は停止扱いで、復帰時は同じmanifestで不足分を配置し、保存入力の照合後に参加する。

## 履歴再利用と並列準備の分散運用（2026-11-19）

新規run用のcurrentは `romutail-prep-throughput-8cc86e60a`。controllerとPauli信号wrapperは `8cc86e60a337272b2b013bd100a6a9c6c8da93ae`。`e25179a74b`、`748c4e0b49`、`587deb5c6a` の差分を `romutail-pauli-inputs-8d03cd254-r2` の運用へ取り込んだ。Q3計算SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`、Pauliの固定provider、DF信号worker、PFの科学workerと回収用runtimeを使用する。

- 4方式の過去Q3照合でSQLite索引を使用。PFとPRの信号準備で検証済みの索引を再利用する。PR-DFの価格は軸ごとの適用条件で選択し、価格が保留のPR候補は未着手の信号も保留する。
- 通常と絶対κの両ROUNDで `candidate_completion.parallel=2` と `pf_completion.max_candidates=2` を設定した。PRは方式内で異なる `L_D` を並行準備し、同じ `L_D` 内は順番を維持する。PFは独立した候補だけを候補別の所有領域で並行準備する。割当は既存の共通キューを使い、科学予算とCPU・メモリ等の資源上限は従来の設定に従う。PBSは新Pauli信号wrapperに既存の120秒見積りを引き継いだ。
- 固定Pauli入力3件のSHA照合を信号索引より先に行う。準備結果のGit取込みは検証済みの参照へ直接fetchする。並行回収で `FETCH_HEAD` を共有せず、元入力からの差分取込みと受領済みattemptの回収を保持する。

人工・回帰155件で候補並行、保存索引再利用、停止後の回収、重複防止を確認した。RomuTail、server2、Romulaptop WSL、PBS gatewayで同じ保存入力を新規workspaceへ配布し、固定入力3件、生成した索引3件、13ラウンドの信号計画を確認した。実信号・compile・Q3・PBS投入は各0件。server1は停止扱い。戻った際はこのreleaseのmanifestで配置し、検証後に参加できる。

Windowsと共通CLIで明示版・currentのPreviewが同じ版を解決した。旧run `search-8b10307fd1b9fc375b6b` のStatusは元の固定版を解決し、記録済みのSTOPを確認した。[検収記録](PREPARATION_THROUGHPUT_ACCEPTANCE.json)

## 適格性照会とSSH空き枠通知（2026-11-19）

新規runのcurrentは `romutail-dispatch-b3a90060f`。controllerとSSH mailbox workerは `b3a90060fd8ff30fddbfb89e965b82481db4c8c6`、計算SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。切替前のcurrentは `romutail-prep-throughput-8cc86e60a`。

- Q3の遠隔dry-checkを中央の割当・回収用ロックの外で実施する。同一REQUESTの照会を共有し、新たに参加したhostの適格性をバックグラウンドで更新する。不成立の照会は、同じREQUESTが再び選ばれた際に再確認できる。予約の直前にSTOP、予算、既存jobの所有を再確認し、割当時にはworkerの現行fingerprintを照合する。
- SSH中継は空き枠と未転送attemptの識別情報を先に通知し、中央が所有するattemptの結果本体を個別に取得する。回収のack後に空き枠を更新する。SSH中継の空き枠通知は15秒、ローカルとPBSは3秒で失効する。
- 人工・回帰124件がPASS。遅い照会と並行した結果採用、照会中にSTOPされた候補の新規予約防止、host参加後の適格性更新、ack欠落後の同一attempt回収を確認した。RomuTail、server2、Romulaptop WSLで人工mailboxを確認した。PBSはgatewayへの配置と回帰テストで確認した。実信号、実compile、実Q3、PBS投入は各0件。

Windowsの `Invoke-RpeSearch.ps1 -Action Preview` と共通CLIの `scripts/run_rpe_search.py --action preview` で、release明示・省略の両方が新releaseを解決した。旧run `search-8b10307fd1b9fc375b6b` は元の固定releaseで `STOPPED`。server1は復帰後に新releaseのmanifestで配置し、人工mailboxと保存入力の照合後に参加する。

## PR準備の継続補充とPBSのPR-DF対応（2026-11-19）

新規runのcurrentは `romutail-pipeline-926bdd226`。controllerとPBSのDF信号用準備adapterは `926bdd2265a4849fab0dc90e6f14c776b3889afe`。前currentは `romutail-dispatch-b3a90060f`。計算SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。

- 通常の分散探索は、方式ごとの準備並列数の範囲で、候補が終わるたびに同じ探索サイクルの次の独立候補を補充する。Q3が実行枠を使用中でも準備を共通キューへ入れられる。実行時には探索判断が待つQ3の優先順位と資源会計を適用する。同じ `L_D` の準備は順番を保ち、要求幅の交互選択と準備工程別予算を引き継ぐ。
- Q3のhost別dry-checkを完了順に受け取る。最初のPASSを保存し、同じREQUESTで中央予約と割当に進める。後から確認が済んだhostも、同じREQUESTの適格先に加えられる。割当時には現在のfingerprint、受付状態、空き枠を確認する。
- [PBS walltime設定](../../../configs/rpe_method_optimizer/pbs_walltime_h4.json) に、PR-DFの `prdf_d2_split`（view `23160cc876807c0300be3de811f767b5810640a7`、adapter `349558b748a82f384a4589434a63ef9b5572032c`）を3600秒、`DF_SIGNALS`（runner `9ed5fcc98d38c328c447f4270eabc73090fcaa02`、provider `fefd548ecd56f60d72031b3d385f6f5e6ee95d06`）を120秒で登録した。Q3は同じルートとadapterの保存計時53件（1.29〜152.39秒）、信号はrunnerが同一のserver2保存receipt 27件（4.94〜6.44秒）を根拠に余裕を持たせた見積り。PBSの信号worker数をタスクあたりの予約CPU数2に合わせ、他hostの信号worker設定を引き継いだ。PBS側に空きが必要で他hostに空きがない場合、他hostが先に適格性を確認した待機中の準備でもPBS側の確認を開始できる。PR-DFの価格較正・監査には新しいPBS見積りを設定していない。

固定SOURCEで人工・回帰165件がPASS。Q3実行中の準備キュー利用と補充、遅い照会先を待たない割当、PBS側の照会・資源会計を確認した。RomuTail、server2、Romulaptop WSL、PBS gatewayに配置した。PBS gatewayでは保存Q3 REQUEST `391c6ca338b629730109` と `6ef331c3b66ff27924da` のdry-check、およびDF信号の固定SOURCE・運用profile・専用runtime照合がPASS。新規の実信号・compile・Q3・PBS投入は各0件。新releaseのPBS計算ノードでの実行時間は未測定。

Windowsと共通CLIは、明示版・currentのPreviewで同じ版を選ぶ。通常のStartは新current、既存runのStatus・Resumeはそのrunの固定版を使う。旧run `search-8b10307fd1b9fc375b6b` は `STOPPED`。server1は復帰後に新releaseのmanifestで不足分を配置し、固定入力を照合してから参加する。

## Q3中間状態の選択保存と空き枠の先読み（2026-11-19）

新規runのcurrentは `romutail-selective-q3-ed4d7855b`。controller、PR Q3 adapter、SSH mailbox worker、PBS workerを `ed4d7855bc3c71ef057be6a23092e6addf33fb6d` に固定した。上流最適化器は `6479a79e7e25830486e1259fad0c93345f0960c1`。直前の `romutail-pipeline-926bdd226` の準備補充と最初の適格枠への割当を継承した。科学SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。

通常・絶対κの両ROUNDで、保存先ごとの上限を40 GiBにし、m5を通常保存、m8/m11を600秒内に同じ厳密キーが再照会された場合に保存する。各hostの既存cacheを参照する。空き枠の先読みは、評価条件が揃った配分から処理する。本命の選択規則は従来どおり。DP残差のバッチと親プロセスの合算時間は `DP_PREFIX_EVENTS.jsonl` の `RESIDUAL_TIMING` に記録する。

対応ルートは既存host profileの `pauli_b_common_entry`、`prdf_d2_common_entry`、`prdf_d2_split`。PR-DF splitは個別ファイルの証跡を維持し、SQLite拡張を持たないWSLの固定Pythonで動作する。PF系は既存adapterとprofileを使う。PBSの適用ルート・CPU・メモリ・walltimeは従来の設定を引き継ぎ、対応PRルートのadapter登録を更新した。

固定SOURCEで人工・回帰186件がPASS。人工DPで厳密一致、m8からの再開、容量、残差計時、既存raw証跡の回収を確認した。RomuTail、server2、Romulaptop WSL、PBS gatewayに配置し、保存入力を用いたhostとルートの組合せ12件で、元入力と新adapterのDPガードを照合した。PBSの別の旧Pauli入力 `3b58bd604a40222989a6` は、前版でも再現する辞書identityの不一致で保留した。旧adapterの固定版専用テスト2件は前版でも同様に失敗した。

WindowsとCLIの明示版・currentの通常Previewで新設定を確認した。旧run `search-8b10307fd1b9fc375b6b` は元の固定版で `STOPPED`。新規の実信号・compile・Q3・PBS投入は各0件。新releaseでの実H4短縮率は未計測。server1の復帰後は新releaseのmanifestで配置し、保存入力を照合してから参加する。

## Worker保存領域の50 GiB受付制限

新規runのcurrentは `romutail-storage-limit-d39c21ab9`。controllerとworkerのSOURCEは `d39c21ab9eb6bf7527cb2f8604fe5265db7a5ca3`。前currentは `romutail-selective-q3-ed4d7855b`。

RomuTail・Romulaptop WSL・server2・PBS・server1に各50 GiB（53,687,091,200バイト）を設定した。環境別profileの `storage_limit` が、最適化workerの作業ファイル・未回収結果と入力・DP等のキャッシュの合計を測る。PBSは全run共通の保存先とキャッシュの合計を使う。RomuTailではworker側の領域を対象とする。各DPキャッシュの40 GiB設定も適用される。

測定値が50 GiB以上、または使用量が取得できない環境は、新規受付を停止する。PBSでは新規allocationを要求する前にも確認する。実行中jobと結果回収は続き、使用量が減ると再計測で受付を再開する。この受付制限では実行中の書き込みで一時的に50 GiBを超え得る。通常Previewの `applied.worker_storage_limits_bytes` と稼働中のキュー計時 `QUEUE_WAIT_STATE` に、設定値と受付の状態が出る。

人工・回帰128件が通過した。RomuTail（31.8 GiB）、Romulaptop WSL（20.4 GiB）、server2（9.6 GiB）、PBS（1.6 GiB）で50 GiB設定と現在使用量を確認した。PBS gatewayの新規投入前の照会も通過した。server1の設定は新releaseに固定し、復帰後にworkerを配置して容量照会を確認する。旧runは元の固定版でSTOPを維持。新規の実科学計算とPBS投入は各0件。

### 回収済み作業コピーの自動整理

50 GiB到達時は、転送ack済みのdry-check一時領域、正常完了した準備の展開コピー、回収済みbundleの作業コピーの順で整理する。45 GiBを目安に再計測し、50 GiB未満で受付を再開する。保持対象は中央成果、attemptのreceiptと所有、実行中・未回収・未確定・失敗した準備の作業成果、固定入力とキャッシュ。

固定版で稼働中のrunには `scripts/reclaim_rpe_worker_storage.py RUN_DIR --watch` をRomuTailで実行する。既存のmailbox接続とPBS gatewayの専用Pythonを利用し、停止中のWSLを起動せず、run終了まで1分ごとに点検する。ログはrun内の `distributed/STORAGE_RECLAIM.log` に残す。容量上限の照会、転送ack後の整理、転送前・実行中・未確定の作業コピーの保持を人工回帰で確認した。

### 準備の固定Git入力と割当照会

PR-Pauliの信号では保存平均のカーネル照合に使う固定Gitファイルを、価格準備では既存adapterが照合する較正・監査SOURCEのファイルを、準備REQUESTの入力に含める。workerはbundleのhash・前提・宣言refを照合して専用キャッシュに直接取り込み、REQUESTの固定Git入力を再照合する。

準備の適格性照会はホストごとに4件ずつ順送りする。成果物のハッシュ照合・転送は割当のロック外で処理し、jobとattemptの予約・採用には既存のロックを使う。これらは稼働中の固定コントローラーへ適用するには版の切替が必要。

2026-11-19の継続では、SOURCE `39efb5ad191b180fba84174bef54a1d025c5d9dd` をRomuTail、Romulaptop WSL、server2、PBSに配置した。旧run `search-d7b7c5848d3b46f445b3` にWSLの終了済み3件を元のattemptで回収し、各1回の受付とackを確認してから通常Stopで終了した。継続run `search-2029d134639349cb5961` は `romutail-prep-recovery-39efb5ad1` を明示指定し、旧runの準備台帳15件と元の探索範囲・予算を継承した。

通常Startのクライアント応答は時間切れだった。同じIDを通常Statusで照会し、起動・稼働と§10.5 PASSを確認した。RomuTailの `prep-06b0565d847ea43c7188b7fc` と `prep-1b04d50e54d6437d1e05d80b` は通常キューでPR-Pauli信号を完了し、各1回回収・採用された。新規信号は計12ラウンド、保存信号の再利用は計14ラウンド。RomuTailの `.rpe-central/deployment/romutail-prep-recovery-39efb5ad1/LIVE_ACCEPTANCE.json` にreceipt照合を保存した。PR-Pauli compileの受付条件は旧runから継承した。

### 準備キューの継続と空き枠の復旧

2026-11-19に、照合済みキューを先に用意する継続runが、明示した `auto` を同時1枠として保存してしまうことを確認した。PBSの準備1件が未確定になるとほかのホストに割当が進まなくなった。修正後は保存ROUNDの `auto` を継承し、元runの前提と同じ方式・範囲・予算で動作する。終了済みの親PBS割当から引き継いだ `UNKNOWN` の準備attemptは同じrequestの保留に使い、引継ぎ先のCPU枠を占有しない。

準備workerは、実行版の作業ディレクトリ作成処理を使い、providerごとの固定SOURCE・実行コマンド・入力・runtime照合をそのまま適用する。疎なGit indexで、対象外の保存成果のblobを読み込まずに作業領域を作成できる。PBSは適合する待ち仕事が増えた場合、所有attemptがない小さな割当を終了し、既存のCPU・RAM条件の範囲で確保し直す。

WSLでは容量上限を超えていたため、再生成できるDP checkpointを既存のLRUで整理し、受付4枠を回復した。新版は50 GiBに達したとき、回収済み作業コピーの整理に続き、必要ならprofile指定のDP checkpointも整理する。原結果・所有・receipt・未確定成果は保持する。

固定版 `1232aa3eb3de4afc28ff63e412169b74f10e3dfa` のreleaseは `romutail-prep-fleet-1232aa3eb`。RomuTail、server2、WSL、PBSに配置済み。server1は復帰後に同じmanifestで配置する。人工・回帰125件とサブテスト3件が通過した。元runは `search-2029d134639349cb5961`、引継ぎ先は `search-f93ece39d4907cd3d3dd`。元runの準備受付済み25件、Q3の採用済み9件、元PBS準備の未確定1件を照合し、通常Startの§10.5が通過した。元の保存価格はGit入力の出所とSHA-256を保全した。引継ぎ後の同時実行の保存設定は `auto`。中央記録は各runの `RUN_LINEAGE.json`、`DUPLICATE_PREFLIGHT.json` とreleaseの `TRANSPORT_ACCEPTANCE.json` を参照する。

PBS実機の人工検収は先行版 `b62f0353d6b49445da0eaedfb6f10871cf687d17` で実施した。通常Startの人工run `search-3c1090aeeb0e3c8d80e9` で、PBSが人工compile 1件、人工Q3 4件を処理し、RomuTailが人工信号2件と監査1件を処理した。完了後の通常Resumeは `REUSED_COMPLETED`。PBS割当 `267297.CPUcluster-PRISM` は2 CPUで終了した。実H4の準備との照合は別に記録する。

引継ぎ先の実探索で、PBSが2 CPUの割当 `267298.CPUcluster-PRISM` を終了し、8 CPUの割当 `267299.CPUcluster-PRISM` へ切り替えた。RomuTailの中央台帳でPBSおよびRomulaptop WSLへのQ3実行開始を確認した。server2では、旧未確定attempt `a68a78ab458e0a0d0987` の0バイト終了ファイルで受付照会が停止した。空ファイルを同じattempt内の `PROCESS_RETURN.interrupted-empty` に保全し、`INTERRUPTED_RECEIPT_LOCATION.json` に出所とSHA-256を記録した。コードにも中断書き込みの例外処理を追加した。元の状態は `UNKNOWN` のまま再実行を禁止する。

PBSの信号準備 `prep-7eedb01dbec53998d13aca2b` は、初回の作業領域でGit objectが不足し、信号本体に入る前に終了した。専用の準備キャッシュに必要な1,252 object（pack 8,602,065 bytes）を追加し、同じ保存入力で出力先の展開まで通った。元attemptは既存の終了コードとログを伴う準備失敗として通常の結果入口で一度回収した。実行結果への再分類や再実行は行っていない。記録は `PBS_GIT_OUTPUT_CONES_REPAIR.json` と `PBS_WORKSPACE_FAILURE_RECOVERY.json`。PBSとWSLでは実Q3の完了を確認した。WSLの終了記録が欠けた評価は保留し、同じ評価の再投入を避ける。

終了時に中央が回収した新規Q3はRomuTail 6件、server2 8件、WSL 4件、PBS 1件。RomuTailの信号は9件を回収し、探索へ取り込んだ。server2の信号完了1件は元runで確認した。WSLとPBSの信号本体完了は引継ぎ先では未確認。二重割当と二重結果受付は各0件。PBSの2割当はいずれも終了した。WSLの容量は中央の保存物とSHA-256が一致した旧配布cache 33点の整理により 50,417,528,832 bytesとなり、既存の受付を戻した。

通常Statusは `RECONCILIATION_REQUIRED`。WSLのPR-Pauli `1372275cb36e31a0940b` の終了記録がなく、探索は `V2_RESULT_REQUIRES_REVIEW` で終了した。同じ期間にB0IのWSL評価2件にも終了記録がなかった。3件は保留している。上記実運用の修正をコードへ反映した次回用SOURCE `35f8ddd8ad92efa347ed86afa979237bf72b3445` は4環境に固定配置済み。

PR-Pauliの価格較正は、元の固定設定のホスト別参照と、科学実行器が確認する公開済みGit束縛を準備するまで保留する。通常の信号生成および評価可能なQ3は独立して割り当てる。

### Q3終了記録の欠落とPR-Pauli信号の作業領域

workerが `LAUNCH_STATE_UNCERTAIN_NO_ADAPTER_EXIT_RECEIPT` と報告し、中央台帳で `job_id`、`attempt_id`、実行ホスト、`automatic_retry=false` を照合できるQ3は、該当構成だけを保留する。attemptの所有と予算消費を保持し、独立した候補の評価を続ける。通常の探索規則を進めて保留が残った場合、通常Statusは `RECONCILIATION_REQUIRED`、停止理由は `SCOPE_HELD_Q3` を返す。終了記録の欠落として特定できない異常には、従来の停止処理を適用する。

PR-Pauliの信号に必要な疎なcheckoutのオブジェクトを、通常の準備REQUESTと既存の共有artifact cache経由で配布する。作業領域を展開する前にGit bundleを照合して取り込み、固定SOURCEと入力の照合を各workerで行う。作業領域を構築するための補助bundleは、回収する科学成果のGit履歴から除外する。

人工・回帰101件とサブテスト3件が通過した。PBS gatewayでは `prep-7eedb01dbec53998d13aca2b` の保存REQUESTを新規作業領域で検査し、固定Git入力206件、作業入力3件、出力先の展開と人工Gitコミットを確認した。この検査の追加bundleは2点、計38,314,097 bytesで、同一の内容は既存cacheで共有する。RomulaptopはSSH照会時に対象WSLが停止していたため、同じ手順での現地再検査を残す。新規の実信号・compile・Q3・PBS投入は各0件。元run `search-f93ece39d4907cd3d3dd` のSTOPと3件の未確定attemptを保全した。

### 継続runでの再開（2026-11-19）

引き継いだPBS準備の未確定attemptは、今回の親runに加え、最初にその割当を所有したrunの終了記録と照合する。`job_id`、attempt、ホスト、入力、`automatic_retry=false` が一致した場合、元の保留を継承する。引継ぎ先の予約枠からは除外する。途中の台帳が異なる場合は照合を止める。人工・回帰34件で検証した。

継続runは `search-6478c133043b892af38c`。release `romutail-prep-fleet-d447eb538` と科学SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774` を固定した。元runの探索範囲と予算を引き継いだ。§10.5の保存Q3照合はPR-Pauli 234件、B0I 159件、PR-DF 64件が再利用可能。準備台帳36件を引き継ぎ、そのうちPBSの1件を元のattemptで保留した。元runで新たに未確定となったQ3 3件も照合し、再投入を抑止した。

通常Resumeを1回送信した。クライアント応答は時間切れとなったため、通常Statusで `RUNNING` を確認した。RomuTail・server2・PBSで実Q3が始まり、RomuTailで新規PR-Pauli信号準備の完了と採用を確認した。確認時点で元runで受付済みまたは未確定だったQ3の重複開始は0件。中央の記録は `.rpe-central/deployment/romutail-prep-fleet-d447eb538/LIVE_CONTINUATION.json` にある。以降の状態確認と停止には通常入口で `search-6478c133043b892af38c` を指定する。

### 未確定の実行枠、PBSメモリ、Pauli較正入力の公開

終了記録のないQ3は同じattemptのまま保留する。workerが同一Linux起動中の監督プロセスと実行セッションを照合し、プロセス保護ロックを取得でき、セッション内の稼働プロセスが消えたことを確認した場合、中央の資源会計からその枠を外す。終了receiptは計時ログより先に完全なファイルとして公開する。遅れて到着した元の結果は既存の照合・回収経路を使う。

PBSは、見積りを設定したPauli/B0I Q3を1件8 GiB、固定版Pauli信号を1件4 GiBとして、1 GiBの余裕を加えたメモリを要求する。根拠は `search-6478c133043b892af38c` のworker計測で、Pauli/B0I Q3の観測最大1,435 MiB、Pauli信号の観測最大509 MiB。小さいメモリで起動した割当は、対応する見積りがあるタスクに限定する。未知の工程には既存の予約を使う。設定値は実行全域のピークを保証するものではない。

Pauli compileでは、固定providerと一致する公開済み元コミットを確認し、jobとattempt固有のブランチに較正入力とBINDINGを公開する経路を追加した。既存の科学実行器が行う公開コミットとBINDINGの照合を使う。公開履歴は候補内の許可ファイルに限定し、rawと監査結果は既存の回収キューに渡す。複数の較正グループを含む要求では、各グループを同じ公開基点から専用ブランチに分ける。公開refは各グループの入力とBINDINGまでとし、rawを含む履歴はローカルで結合して中央キューで回収する。

継続run `search-5c2b9e9b63ece99a4608` は `romutail-prep-fleet-054489a9a` を固定し、通常の Resume で起動した。controller SOURCE は `054489a9a907681ae152d83511201df2c5573111`。RomuTail、Romulaptop WSL、server2、PBS で SHA を確認した。Q3・信号の科学 SOURCE、価格の固定 provider、元の探索範囲と予算、50 GiB のホスト別保存上限を継承した。

価格較正の固定ビューを候補メタデータをコミットする optimizer から分け、中央と RomuTail の worker に同じ工程 profile を置いた。manifest の compile と audit に較正ラッパーの SOURCE を登録した。較正作業領域の固定 SHA と GitHub の公開基点 `codex/rpe-pauli-binding-base-054489a9a9` の一致を確認し、既存の公開照合と同じ経路で権限検査の dry-run が通った。最新配置の保存要求は27件の入力照合を通った。

前の継続run `search-5cdeb457cae5b7ba22c7` までのSTOPと原結果を保持した。較正の旧attemptは、配送前の拒否1件と、配送後の公開基点照合エラー5件を実行台帳と元ログで確認した。科学処理の開始前であることを照合し、旧attemptと予算の消費履歴を残した。旧controllerは新規受付停止後、全結果の回収とPBS割当の終了を確認してから、未割当の準備を待ち続けた専用controllerを終了した。通常の継続前照合で再利用可能な保存Q3は559件（B0I 242、PR-Pauli 253、PR-DF 64）。準備台帳63件と既存の保留を継承した。未予約の準備待機を解放する修正後は、通常Stopで旧runが終了した。

WSL の旧4 attempt は実行セッションの終了を確認し、結果は未確定として保全した。先行runでは、解放した枠に別のQ3を割り当て、WSL と PBS から実結果を回収した。PBS の予約は 2 CPU・9 GiB、4 CPU・17 GiB、6 CPU・25 GiB、8 CPU・33 GiB を確認した。割当 `267348.CPUcluster-PRISM` は4 CPU・17 GiBで実Q3 4件を回収して終了した。

直前の固定版（`ffc1df03964397524a0b53e4017969a142adf221`）では、過去に失敗した信号要求で WSL の固定Git入力168件、PBS の固定Git入力206件と、各3件の作業入力を照合した。両環境で一時作業領域の展開と人工コミットに成功した。この検査の実信号生成は0件。WSL は再生成可能なDPチェックポイント57件の整理で約2.1 GiBを解放した。RomuTail は元bundle 131件のSHA-256を照合し、旧展開キャッシュ約7.7 GiBを回収した。

人工・回帰123件とサブテスト3件、配送前の照合追加で28件、較正前照合で26件とサブテスト12件が通過した。

複数グループの公開分離、停止時の未予約準備の待機解除、継続時の所有照合を含む後続SOURCEは `547121fbb02864098bdc834b3a9dd651d3df99dc`。人工・回帰48件とサブテスト12件が通過した。人工Git検査で、公開された2つのrefに各BINDINGだけが入り、両グループのraw commitは中央回収用の履歴に残ることを確認した。

旧Pauli較正の `--input-preflight` は、server2の原本にSHA-256で一致する保存config、封印済みprovider 278ファイル、H4の保存snapshotとreceiptを読み取り専用で対応付け、固定checkout `13ab1acfaebe5f33958e1db67862f5c45725ab91` に同一blobの参照JSONを展開して通過した。固定checkoutと科学コードのSHAは同じ。対応付けはRomuTailのcompile worker内に限定し、通常のホストファイルを変更しない。最後の関連回帰は49件とサブテスト12件。

実検収では通常キューの較正 `prep-e77324f01c49371095250002`（attempt `1d2eef60123d4a7ebdc9b26ac7f108c9`）をRomuTailが実行した。3グループの公開refを照合した。各グループの2経路×X/Yの4回、計12回の実compileが終了し、中央が `PREPARATION_COMPLETE` を1回採用した。前版の保留attemptは元の台帳に保持した。

この継続では4環境から実Q3を回収した。WSLとPBSの実信号については、新版から今回の確認時点までに起動・完了したジョブがなく、保存入力・一時作業領域の検査まで。PBSの最大確認予約は8 CPU・33 GiB。通常Resumeはclient応答がタイムアウトしたため、同じrunを通常Statusと中央の採用記録で確認した。全体の既定currentは維持し、この継続runに明示releaseを選んだ。

### 較正から監査への引継ぎと経路別キャッシュの整理

Q3の入力は準備済みの信号、候補の条件、ショット配分。Q3はその配分でのMSEの上下界と基準適合を評価し、探索器が結果を使って次の候補と配分を決める。信号生成、価格用compile、回路とreceiptの監査はそれぞれ別の準備タスクとして扱う。前の継続runではPR-Pauli信号がRomuTailで34件、server2で5件成功した。WSLとPBSにも同工程の割当はあり、成功の回収は未確認。

監査タスクはcompileの `PRICE_STATE.json` から、選択済みグループのraw commit、BINDING、COMPILE_RECEIPTS、RESULT、固定CONFIGURATIONを入力として宣言する。これらのGitオブジェクトとhashを既存bundleで配送する。保存済みの失敗例を独立した作業領域で再現し、配送後に同じ固定監査スクリプトの `--build` で2経路と12ファイルの所在を確認した。監査の科学処理とcompileの再実行はこの検査で行っていない。

容量整理は、基本profileと経路別profile両方のDPチェックポイントを対象にする。既存の50 GiB上限、回収済み作業領域の照合、実行中や未確定の保全に従う。実機ではWSLの再生成可能なチェックポイント約1.7 GiB、RomuTailの中央とSHA-256が一致する作業キャッシュ約4.1 GiBを解放し、受付を復旧した。関連の人工・回帰試験35件が通過した。

修正版は `4d20f4b7a009c42d9a4918f8db17dc87503d060e`、継続runは `search-04c58930814ee4ae1d25`。4環境の配置を照合し、旧runの通常終了を待って、保存Q3 572件と準備台帳70件を引き継いだ。通常Resumeで再開し、容量点検をrun終了まで起動した。以前の監査失敗と科学実行のSTOPは旧台帳に保持した。

RomuTailの容量点検には、同じrunの中央原本とSHA-256が一致する作業用artifact cacheの整理も加えた。容量上限に達し、登録上このrunのみが稼働している条件で、ローカルの所有・予約・キューを照合して整理する。中央にないコピーや未確定attemptの参照は保全する。点検用の固定変更は `44edf793614129a03615e6ab06f596309af3ae7d`。追加の保全試験を含む関連試験36件が通過した。

### PBS割当の再利用とworker容量の先行回収（2026-11-19）

新規runの `current` は `romutail-fleet-efficiency-ffd406f4e`。controllerとworker輸送SOURCEは `ffd406f4ed966f8ec712cba65a4d378cec43cac8`。RomuTail、server2、Romulaptop WSL、PBSゲートウェイで配置後の固定版を照合した。科学Q3のSOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。既存run `search-04c58930814ee4ae1d25` は通常Statusで固定release `romutail-prep-fleet-4d20f4b7a` を解決した。旧runのSTOP、原結果、台帳、受付規則を引き継いだ。

- 準備の初回ホスト照合待ちを `PREPARATION_PROBE_PENDING` に記録する。保留時はホスト別に工程profile、PBS見積り、受付状態、runtime・入力照合の理由を残す。PBSだけが照合可能な準備では、その候補の作業だけを時間を区切って待たせる。予約と予算消費は既存の要求別fingerprint通過後に行う。旧runの一律 `NO_MATCHING_PREPARATION_STAGE_HOST` 13回はいずれも後から候補に適格ホストが見つかった。
- PBSでは、直近に完了した必須準備が次の工程・Q3に戻る間、既存の割当を短時間保持できる。判断に使う直近数件の投入から受付可能までの実測は、15〜30秒の範囲に制限する。適格な待機仕事、CPU・RAM・残りwalltimeに沿う既存の割当サイズ選択とdrainを使う。通常配置では単一割当の最大96 CPUと既存のPBS側CPU会計を使用する。
- PR-Pauliの監査 `PAULI_AUDIT` は固定 `4d20f4b7a009c42d9a4918f8db17dc87503d060e` の保存入力222件をPBS上で照合した。保存済みの実行時間とピークRSSに余裕を取り、同じSOURCEに限り180秒・4 GiBの見積りを登録した。PR-Pauliのcompileは固定providerでGit公開とローカルの分離実行条件を要求するため、現在の適格先はRomuTail。
- ホストの容量制限は50 GiB。成果の転送ack後に再生成可能な作業コピーを直ちに整理できる。47.5 GiB以上では45 GiBを目安に点検し、ack済みコピーを優先する。原結果、終了receipt、実行中・未回収・未確定attemptは保持する。容量が戻れば次のworker照会で受付が復旧する。手動PauseやWSL停止は既存設定に従う。

回帰・人工試験77件が通過した。Romulaptop WSLの一時領域で、未転送コピーが保持され、転送済みコピーの容量点検後に528384バイトを回収し、原結果と受付可能状態を確認した。通常Startによる隔離人工run `search-e1ff566f81b6a76eb034` は、実PBS `267426.CPUcluster-PRISM` の2 CPUで人工compile 1件・Q3 4件を実行し、RomuTailの人工信号2件・監査1件と合わせて8件を各1回採用した。通常Resumeは `REUSED_COMPLETED`、追加起動・採用・PBS投入は各0件。PBSは終了コード0で枠を返却した。先行の人工runはクラスタの管理対象パス外を指定したため、PBS投入0件で通常Stopし、管理対象内の専用名前空間で再検証した。明示・無指定の通常CLI Previewと旧runの固定版解決が通過した。

検収記録は `.rpe-central/deployment/romutail-fleet-efficiency-ffd406f4e` と `.rpe-central/acceptance/romutail-fleet-efficiency-ffd406f4e-scheduler` に保存した。今回の新規実信号・実compile・監査による新規科学計算・実Q3は各0件。次に実際の探索を行う際は、通常Startで利用者指定の範囲を渡し、工程別の準備完了からQ3開始までの待ち時間、容量による受付保留時間、PBSの確保CPU時間と完了件数を採取する。server1は復帰後に同じmanifestの配置と保存入力照合を行ってから受付を判断する。

### 通常Startからの実探索検収とPBS枠の引継ぎ（2026-11-19）

実検収run `search-d1fd259351dae2f86070` は `romutail-fleet-efficiency-3cea00961` を固定した。前回の探索範囲、科学Q3 SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`、準備providerと既存の資源設定を継承した。§10.5で再利用可能な保存評価597件（B0I 255、PR-Pauli 278、PR-DF 64）と準備台帳100件を照合した。旧準備のメタデータ失敗は、科学処理が始まっていないことを確認できる場合にだけ保存入力で再照合する。複数の保存Q3が同じ評価を指すときは、現行routeに一致する終了済み証拠を優先する。旧PBSの未確定attemptと元runのSTOPは保持した。

実検収ではPR-Pauliの新規Q3を35件回収した。中央の探索採用は16件で、先読みとして回収した結果も元の評価IDで保持した。今回新しく割り当てた評価の二重割当・二重受付・二重採用は各0件。検収後に通常Stopで終了し、終了コード0とrelay停止を確認した。成果は通常公開の `SEARCH_PUBLICATION` に入り、公開commitは `9959b7d7b98e7d64e09f40da50bdf1ca88102b56`。

| 実行先 | 実Q3回収 | 探索採用 | 信号準備の回収 |
| --- | ---: | ---: | ---: |
| RomuTail | 4 | 3 | 1 |
| server2 | 15 | 8 | 1 |
| Romulaptop WSL | 15 | 4 | 1 |
| PBS | 1 | 1 | 1 |

信号準備4件は保存信号を再利用した。特にWSLとPBSで確認した範囲は、実入力の照合、workerでの保存信号準備、中央回収・採用まで。RomuTailで価格の較正1件（独立2経路×X/Yの実compile 4回）を完了し、server2の監査1件は別ホストで実行・回収され、`SCOPED_STEP_MODEL_READY` を返した。検収終了時点の監査ジョブには探索採用時刻がない。別のLD4較正3件は固定契約の `STOP_LD4_CAL` でcompile 0回。確認した失敗記録では保存 `s_R` とnative DF tailのidentityが一致せず、元の停止を維持した。

PBSの実Q3 `20654eb51a02ce12f289` は `267436.CPUcluster-PRISM` の12 CPU・49 GiB割当で実行され、終了receiptから中央が1回採用した。PBSの終了コードは0。実検収では、大きい割当へ交換した直後に前の割当で測った同時数1を再使用していた。新版では並列数の実測と試行をPBS allocationディレクトリに保存し、同じallocationの再起動で引き継ぐ。別allocationはそのCPU・RAMに応じた初期枠を使う。共有spoolは同じjobとattemptに使い続ける。中央の空き枠記録も、終了済み割当が保持する未確定attemptを実行中の枠から除いて数える。

`pauli_b_common_entry` のREQUESTには選択された構成の辞書とP0残差receiptの固定Git参照を加えた。PBSの疎なcheckoutで欠けていたreceiptを同一Git blobで配布し、保存REQUEST `e628f38ab2f1695aef1d` の照合と実検収runでの後続Q3を確認した。較正へ引き継ぐ保存 `PREPARATION.json` では、物理入力が一致する場合に限り、信号の来歴表示の差を許容する。較正用原本と科学条件は固定している。

新規run向けcurrentは `romutail-fleet-efficiency-b65a30d81-r2`、controller・PBS worker SOURCEは `b65a30d8155285b5612d8fb9a8808adadaec7d95`。RomuTail・server2・WSL・PBSに固定配置した。`-r2` ではPBS時間見積りの値を維持し、見積りpolicyが参照する輸送SOURCEを新workerに合わせた。通常Startの隔離人工run `search-0f04c0d6991998a0325c` は、同じ固定コードのPBS `267438.CPUcluster-PRISM` で人工Q3 4件と人工compile 1件、RomuTailで人工信号2件と監査1件を処理し、計8結果を各1回採用した。通常Resumeは `REUSED_COMPLETED`、追加割当・採用・PBS投入は各0件。PBSの終了コードは0。回帰92件とサブテスト3件が通過した。同一spoolを使う新旧allocationが1枠と6枠を別々に報告することは人工回帰で確認した。新版の実PBSでの96 CPU相当の飽和や実測効率は未計測。

50 GiB上限に達したWSLでは中央保存分と同じhashの作業キャッシュを整理した。RomuTailでも回収済みbundleの重複コピーと、別の作業領域から参照されていない再生成用Gitキャッシュを、元bundleのhashを照合して整理した。結果、RomuTailは49,420,140,544 bytesとなり、上限を維持したまま受付可能に戻った。原結果、未回収・未確定の証拠を保全した。実機記録は `.rpe-central/deployment/romutail-fleet-efficiency-b65a30d81/LIVE_ACCEPTANCE.json` と同階層の容量回復記録を参照する。

### 分散運用の完成条件と残件

通常入口で探索範囲を一度指定し、RomuTailが保存成果との照合、候補選択、工程別の予約、割当、回収・採用、次候補選択、公開を進める。入力準備、信号、価格用compile、監査、Q3を、固定入力とruntimeが一致する空き枠に配る。PBSとserver2に同じ工程を配分できる条件を整え、PBSは適格な仕事に見合うCPUとRAMを確保する。RomuTailと受付中のRomulaptopも共通キューを使い、任意の実行先の離脱時は他の仕事を続ける。

| 順序 | 残件 | 完了と判断する証拠 |
| --- | --- | --- |
| 1 | PBSのB0I価格用compile・独立検証compile。現行の割当先はserver2のみ。PBSで保存要求2件のGit bundle検証が不足オブジェクトでHOLD。 | 新規作業領域で両要求の固定SOURCE・全入力・runtimeを照合し、通常配送・回収を人工で検査する。実運用の受付追加は固定releaseに記録する。 |
| 2 | PBSの新規信号、実compile、実監査から探索への引継ぎ。Q3の実回収と保存信号を使う準備、監査の保存入力照合は確認済み。PR-Pauliの価格用compileはRomuTail限定の固定profile。 | 実探索に必要になった未実施タスクで、方式・工程別にworkerの数値処理開始、正常終了、中央回収・採用、同一候補の次工程を確認する。PR-Pauli compileのPBS対応は既存の公開Git束縛と隔離runtimeを再現してから判断する。 |
| 3 | PBSで複数工程を並行させたときの使用効率。12 CPU時の旧並列数持ち越しは修正済み。新版の実PBS実行は2 CPUの人工検収まで。 | 適格な仕事が十分にある区間で、確保CPU時間、実使用CPU時間、工程別完了件数、割当待ち、転送待ち、idle理由を測り、枠が埋まらない原因を特定する。100 CPUの使用上限を維持し、空いた割当は既存のdrainで返却する。 |
| 4 | 全体の通し。直近の実検収はPR-PauliのQ3と各環境の一部工程を確認した。 | 一つの通常runで準備からQ3・公開まで追跡し、停止・再開やホスト切断後も同じ結果を二重採用せず、他ホストの探索が継続することを確認する。探索マップで結果を確認し、未対応セルを理由付きで残す。 |

実環境の確認表はホスト×方式×工程で更新する。保存入力照合、人工の配送・回収、実処理と中央採用をそれぞれ記録する。既存の実結果・STOPと未確定attemptを保持し、検収のために同じ科学評価を再実行しない。工程が未確認の場合は候補、入力、runtime、資源、回収のどこで止まったかを記録する。

PBS効率の比較起点は実検収の `267436.CPUcluster-PRISM`。PBSの終了統計は12 CPU、割当時間391秒、使用CPU時間338秒で、割当全体の利用は約7.2%。この割当では旧並列数の持ち越しがあり、輸送・準備の待ち時間も含む。次回は同じ統計に加え、適格な待機仕事がある区間と仕事を待っている区間を分けて記録する。

B0Iの保存REQUESTは3件を比較した。RomuTailで失敗を記録した通常compile `prep-01d7ea983c7bb441b3d27e41`、server2で完了した独立検証compile `prep-2b918638d8153a9181da8b19` と通常compile `prep-f6e4f2f9b315d62a52dbed53`。PBSでは、旧bundleの前提commitを揃えた後にも展開に外部delta基底が必要だった。さらに保存REQUESTが参照する疎なcheckout外のデータを確認した。これらを約2.1 MiBの補助packとして作り、元bundleとREQUESTのhashを保持したまま、PBSの隔離領域で3件のSOURCE・Git入力21、31、24件・runtime（Qiskit 1.3.0）を照合した。記録と補助packは `.rpe-central/deployment/romutail-fleet-efficiency-b65a30d81-r2/PBS_PF_SAVED_INPUT_PARITY.json` と同名の `.pack` に保存した。関連回帰試験65件とサブテスト3件が通過した。

通常compileと独立検証compileの新規REQUESTにも、既存の疎なworkspace用bundleを追加する修正を `da84fc28aa` に固定した。固定SOURCE、宣言済みの出力先とその上位、workflow、ライセンス、参照ファイルを読むためのGitオブジェクトを含む。保存済みREQUESTの変更を伴わない。次はこの配送を通常入口からPBSへの人工実行・回収へ接続する。実機の通常キューで受付が確認できた工程から、配置・新規releaseの採用・採択済み実探索の検収へ進む。

保存REQUESTの3件では、上記と同じ範囲に出力先の上位ディレクトリを含め、PBSの隔離workspaceを展開できた。既存workerの固定SOURCE照合と出力先追加まで3件が通過した。旧bundleの補助packは3件で共用し、25,084,145 bytes。結果とpackは `PBS_PF_SAVED_WORKSPACE_PARITY.json` と同名の `.pack` に保存した。通常キューへの切替は未実施。今回の新規科学実行は0件。

### B0I準備の通常キューとPBS回収の検収

通常Startの人工run `search-7094094e98e248075f1c` で、PBSが未受領の準備bundleを先に照合する問題を再現した。PBS `267441.CPUcluster-PRISM` は4 CPUを返却したが、中央が予約した通常compile・独立検証compile各1件はworkerに永続化されず、旧runの台帳では `UNKNOWN`。このrunとattemptは保存した。

修正版 `bf7fac6d82dd675b7d36a63ebcc420c8b75c1ba7` では、受信したREQUESTのhashを確認し、準備bundleを取得してから固定release・SOURCE・入力を照合し、同じattemptをworkerに記録する。PBSの自動返却は中央に所有中のattemptがある間は待つ。明示Stopとwalltimeの受付制限は従来の処理を使う。固定release `romutail-pbs-pf-preparation-bf7fac6d8` をRomuTail、server2、WSL、PBSに配置した。通常運用のcurrentは `romutail-fleet-efficiency-b65a30d81-r2`。

通常Preview・Startで人工run `search-b59a41c1f4a57987d510` を起動した。既存のB0I要求生成器が `PF_COMPILE` と `PF_HOLDOUT_COMPILE` を生成し、PBS workerの初期checkoutにない固定blob 2個を両要求に共通のbundleで供給した。PBSの計算ノードで両処理が正常終了し、中央が各receiptと成果のhashを照合して一度だけ採用した。RomuTailの監査2件が対応するcompileの成果とhashを読み、人工Q3 4件も通常経路で終了した。信号準備2件を含む計10件を各1回採用した。通常Resumeは `REUSED_COMPLETED`。追加割当、二重採用、追加PBS投入は各0件。

PBS割当 `267442.CPUcluster-PRISM`（4 CPU、9 GiB）と `267444.CPUcluster-PRISM`（2 CPU、12 GiB）はともに終了コード0で返却された。検収記録は `.rpe-central/acceptance/romutail-pbs-pf-preparation-bf7fac6d8-artificial-01/ACCEPTANCE.json` と同階層の `PBS_EXIT.json`。関連回帰試験は84件とサブテスト3件が通過。人工出力は隔離領域に保存し、新規科学実行は0件。次の工程は、実探索で必要になる新規信号・価格用compile・監査を固定条件で実機確認し、完了成果を同じ候補の次工程へ戻すこと。PBSのCPU利用効率は別途、適格な実タスクが十分にある区間で測る。

### PBSでのB0I実入力と通常起動の確認（2026-11-19）

通常Startの対象をB0I `p=8`・`4th`・要求幅 `8h` の登録済み1構成に固定した。固定準備SOURCEは `36ad751d4ebf0ec160532615da6425b268a0c18a`。既存のQ3科学SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。元契約の入力、運用中の版、過去のSTOPを保全した。§10.5と固定providerの重複照合を通過した。準備計画は保存信号3ラウンドを再利用し、新規信号10ラウンド、通常compile 6ラウンド、独立検証compile 1ラウンドを要求した。検収設定は1探索サイクル、1探索step、工程別予算はsignal 1、compile 2、audit 4とした。

| 通常Start | PBSの割当 | 確認結果 |
| --- | --- | --- |
| `search-e7e248d504cdaf8f31cd` | `267453.CPUcluster-PRISM` | 8 CPUを確保。準備入力照合の待機期限に達し、キュー予約0件。31秒で終了コード0。 |
| `search-b909673782427d4eeb02` | `267454.CPUcluster-PRISM` | 入力照合後、信号1件を割当。起動設定が共通supervisorを呼べず、科学処理開始前に終了。compile 1件は未割当のまま取消。通常Stop後、8 CPUの割当は119秒で終了コード0。 |

信号 `prep-05d084ffe4d5e45e787d4737`、attempt `d437dcdd6630446e90c24786d01264a7` の `UNKNOWN` と `automatic_retry=false` を保存した。元のPBS spoolでREQUEST・attemptを照合し、supervisorのエラーと `SCIENTIFIC_STARTING.json`・adapter起動・終了receiptの不在を確認した。新規実信号・実compile・監査・実Q3の実行と採用は各0件。CPU計測には準備照合と制御のみが含まれる。

`0cb7952c6f` で元providerの絶対パスを隔離runtime内で解決し、保存入力をhash照合して読み取り専用で供給した。`a33370d47d` で通常入口の単独方式指定に予算親、未確定評価、既定の幅を関連付け、重複照合の履歴を保全した。`65769c5e0` でPBS起動後の入力転送・照合を待つ時間を確保し、未予約の要求は再開時に照会をやり直せるようにした。`d1804fe9f8` で準備用起動設定を共通supervisorに対応させた。

`search-b909673782427d4eeb02` の通常Start応答は120秒でタイムアウトした。中央の `LAUNCH.json`、台帳、終了記録で同じrunを追跡し、Startの再送は行わなかった。

修正版の配置 `romutail-pbs-pf-live-d1804fe9f-r2` はRomuTailとPBSの隔離領域で固定SHAを確認した。PBSの保存入力dry-checkでは元のsignal・compile REQUESTを変更せず、各19件の固定Git参照、既存Python 3.11.0rc1、compile用Qiskit 1.3.0の照合が2件とも通過した。PBS実機では、修正版が既存supervisorに到達し、完了済みの人工attemptを再起動しないことも確認した。回帰73件が通過。記録は `.rpe-central/deployment/romutail-pbs-pf-live-d1804fe9f-r2/` の `PBS_SAVED_RUNTIME_CHECK.log` と `PBS_SUPERVISOR_TERMINAL_CHECK.log`。通常運用のcurrentは `romutail-fleet-efficiency-b65a30d81-r2`。次は中央担当が元attemptとSTOPを照合して継続を決め、計算ノードで信号・compileの実処理、監査、同じ候補の評価への引継ぎを確認する。複数の実タスクを使ったPBSの効率計測はその後に行う。

### B0I単一構成の実処理・Q3・公開の通し検収（2026-11-19）

承認済みのB0I `p=8`・`4th`・`8h`（候補 `b3430ea5eb978ab4fbaf22e83c3c112fc0609cb8d00f06419e56f9d2b9164d6d`）を通常CLIから引き継いだ。旧runのSTOP、未確定attempt、失敗receiptと工程別消費は元台帳に残し、後継runに `RUN_LINEAGE.json` を保存した。固定準備SOURCE、科学Q3 SOURCE、範囲、精度条件、自動shot提案は継承した。後継の検収設定は各1サイクル・1探索stepとした。

| 工程 | 実行先と確認した結果 |
| --- | --- |
| 信号 | `search-7a4dffac596af1366c09` のPBS `267457` で新規10ラウンド、保存信号再利用3ラウンド。中央採用した `MEAN_HANDOFF.json` は `a2bd134570f0fcca950438d34f93756844e0c8e5`。以降は13ラウンドを再利用し、新規信号の予算を0にした。 |
| 通常compile | 同じrunのPBSでは、6ラウンドともB0I tail定数項の厳密一致検査に失敗し、回路compile 0件。`search-3e6b70bba6c78593eec3` のserver2が同じ検査を通過し、6ラウンド・12回路を完了。中央採用した原結果は `ded10db4e9ada379e479942a52dd45ae2a074c8e`。 |
| 監査・独立検証 | 最終run `search-51ec4b40deb95d9b8e5b` は通常compileの12回路を再利用。PBSが通常監査とモデル固定を行い、server2が独立検証の1ラウンド・2回路をcompileし、PBSが独立検証監査と26軸の価格集計を完了した。独立検証後の再fitなし。価格は既存契約の条件付き比較モデル。 |
| Q3・通常公開 | 同じ候補に `5a9e29b8eb7e795f61fb` をRomuTailで1回実行・回収・採用した。結果は `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`。今回の配分ではMSE適合点なし。Q3本体の壁時計91.64秒。通常公開 `74414217b1683466f49d735dd0d657eae089d193`。 |

修正 `590a5fb40e` は固定providerの版指定済み化学依存を割当前に照合する。`4640caa0e1` は保存演算子とnative guardの固定Git参照を、`4b3e5b7c2b` はC監査が読む固定B/D APIを既存配送bundleへ含める。PBSの依存追加は検収領域の `openfermionpyscf==0.5` のみ。共有環境のPythonと既存科学SOURCEは固定。関連回帰52件とサブテスト3件が通った。

最終runは `romutail-pbs-pf-live-4b3e5b7c2-r3` を明示して開始。信号・通常compileの再実行は0件。最終runの6タスクについて台帳の割当・受領・採用は各1回。通常終了後のrelay停止とPBS枠の返却を確認した。他候補の旧Q3 holdは維持し、この候補の評価は探索step上限1で終了した。通常運用currentは `romutail-fleet-efficiency-b65a30d81-r2`。

成功したPBS割当の使用CPU時間／確保CPU時間は、信号 `267457` が39／1,312 CPU秒、通常監査・固定 `267459` が34／1,208 CPU秒、独立検証監査・集計 `267461` が33／1,232 CPU秒。1構成の配送待ちを含む測定なので、複数の適格な実タスクでの高負荷効率は未検収。PBSでのcompileの厳密再現も残る。埋め合わせのcompileや追加Q3は実行していない。既存のC同期で探索マップを `75c736406df4fe2cc956c2932b1a4bb4f8adbf96` に公開した。

Romulaptopでは通常の観測SSHが120秒でタイムアウトした。Cの通常observerが最終結果を照合した公開データをRomuTailから転送し、Windowsの既存同期エンジンでGit blob、SHA-256、配置先を検証して5ファイルを反映した。`H4-search-map/index.html` はC公開版と同じSHA-256 `b6ca0b4578844f8c244a473419cde5bc53819e8e42e03c4e8bf715d0eb3882c1`。`INDEX.json` のrun IDと、受領状態 `FINAL_RESULT_PUBLISHED` を確認した。Windowsの既存設定は維持し、通常の観測SSHのタイムアウトは継続課題として記録した。同期の実機記録は最終release配置の `WINDOWS_FINAL_SYNC_VERIFIED_COPY.log`。

## 4方式の共通入口とPBS準備配分（2026-11-19）

新規run用currentを `romutail-fleet-efficiency-b65a30d81-r2` から `romutail-mixed-methods-0c67daacd` へ切り替えた。controllerとPBS transportは `0c67daacd628ed1fe23a53b445179596fa1d2d4d`、科学Q3 SOURCEは引継ぎの `28d976b99cdf8329adcc1b79327e329e947ef774`。Windowsの `Invoke-RpeSearch.ps1` と共通 `scripts/run_rpe_search.py` は、版を明示しても省略しても4方式を含む同じPreview `search-897fdfe64b6b045839a3` を返した。検収範囲では科学予算を全工程ゼロとした。既存runのStatus・Resumeはrun IDを指定する。B0I実検収runのStatusが固定release `romutail-pbs-pf-live-4b3e5b7c2-r3` を解決することを切替後に照合した。

| 方式 | この配置で確認した経路 | 残る実機検収 |
|---|---|---|
| Full PF | 固定workerとadapterが一致する `pf_b_pf6` Q3をPBSのwalltime規則に追加。旧PBS Full PF M5のQ3本体49.64秒・回収まで66.94秒を参照し、運用見積りを14400秒とした。PF信号・監査の固定runtimeと入力配布はB0Iとの共通経路。 | Full PF固有の入力で、新currentから準備・実Q3・公開までを確認する。 |
| PR-Pauli | PR-DFと同じ人工runで、RomuTail信号、PBS compile、RomuTail監査を回収しQ3まで採用した。処理は人工入力。既存の実運用では固定の価格compileをRomuTailへ配分する。 | 固定公開Git束縛と隔離runtimeを満たすPBSの実compile。 |
| PR-DF | PR-Pauliと同じ人工runで準備結果を回収しQ3を採用した。PBSの固定 `DF_SIGNALS` と `prdf_d2_split` の適格性規則を引き継いだ。 | PBSでの実価格較正・独立検証・監査は、該当SOURCEでの適格性と実績が必要。 |
| B0I | 前項の実検収を引き継いだ。PBSで保存PF信号入力19件、監査入力35件を新配置から照合した。 | 固定tail scalarの一致を必要とするPF compileはserver2へ配分する。該当する旧PBS実行の数値不一致は継続課題。 |

以前はPBS起動後の事前照合を待つ準備を合計1件として資源要求を作っていた。現在は他hostに適格な空き枠がない独立準備を個別に数え、既存のCPU・メモリ・同時枠制約で確保する。事前照合の成功後に各タスクを割り当てる。人工回帰では3方式の独立準備を6 CPU・3枠とし、未対応の工程を除外した。配置先の回帰73件、追加対象を含む回帰92件を通過した。

実PBSの人工検収では、準備とQ3の14件を共通queueが採用した。RomuTailは13件、PBSは共用の人工compile 1件。PBSの1 CPU検収割当 `267463`、`267464` はともに終了コード0。初回のmanifest不一致で研究計算を始めなかった人工割当 `267462` は終了済み。人工結果は専用ディレクトリに保存した。今回の新規実信号・実compile・実Q3はいずれも0件。96 CPUの実探索での稼働率は未測定。

RomuTail・server2・PBSの固定配置を確認した。検収時に休止中だったRomulaptopのWSLは、起動後に同じreleaseのmanifestで不足componentを取得し、worker profileと保存入力を照合する。server1も復帰後に同じ手順を適用する。Windows同期の `sync.py` には、元のウィンドウ非表示設定を維持し、既存SSHの応答をファイルで受け取る修正を配置した。通常同期経路で公開済みrun `search-51ec4b40deb95d9b8e5b` の `FINAL_RESULT_PUBLISHED` とC公開commit `75c736406df4fe2cc956c2932b1a4bb4f8adbf96` を照合した。

RomuTailの配置・検収証拠は `.rpe-central/deployment/romutail-mixed-methods-0c67daacd/TECHNICAL_ACCEPTANCE.json` と同ディレクトリのPreviewにある。Full PFの旧実Q3の証拠は[旧固定版の検収記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/840525ed6a2dc3e018dd568898b9bc9e06349578/docs/research/rpe_pbs_h4_q3_portability_20261118/HANDOFF.md)。

### 停止済み本探索の再開と容量監視

通常Resumeで `search-d1fd259351dae2f86070` を再開した。固定版 `romutail-fleet-efficiency-3cea00961` の探索範囲と工程別予算を継承し、runの `HOST_ADMISSION.json` でWSLの新規受付を閉じた。§10.5照合はPASS。RomuTail・server2・PBSで実PR-Pauli Q3を中央回収した。4 CPUのPBS割当 `267466` の観測値は、壁時計197秒までの110 CPU秒で、全割当CPU時間の約14%。同じ採択条件の実Q3本体114.10秒に対し、割当から中央受領までは約180秒だった。準備済み要求の供給と割当から受領までの差が残る。

容量監視は `LAUNCH.json` と `EXIT.json` の invocation IDを照合して、以前の終了receiptが残る再開runでも継続する。受付を閉じたrelay先は監視の対象外。事前に同じ評価条件の終了済みrunに中央原本があるローカルキャッシュも候補にし、コピーと原本の両方のSHA-256を照合する。待機・未確定入力、別の稼働中run、原本とreceiptは保全する。この手順でRomuTailの重複コピー65個（2,550,206,464 bytes）を除去し、50 GiB未満での受付再開を確認した。調査・回収のログは対象run内の `distributed/STORAGE_RECLAIM.log`。

### PBS向けの先行照合と割当ごとの枠数

分散autoの待機キューは、受付中で直近の状態が分かるworkerの枠数を使い、稼働中の処理に加えて次の一巡分を事前照合する。通常の予算・要求別適格性・割当・回収を使う。2件稼働中に次の2件が揃い、終了直後の次回pollで別の要求を2件取得できる回帰を追加した。継続・通常入口・PBSの回帰66件とPBS workerの回帰11件が通った。

稼働中の元run `search-d1fd259351dae2f86070` から、運用修正版の `search-0d8c2bc34c58f349a12c`、さらにPBS workerの更新を含む `search-3a962b43686c76679768` へ順に引き継いだ。通常Previewで探索範囲と予算が同じであることを照合し、元の実行を回収し終えてから通常Resumeした。最終の明示releaseは `romutail-pbs-ready-wave-54b7f7416-allocation`。科学Q3 SOURCE・adapter・固定入力は元の探索から継承した。旧B0IのPBS trialは、終了した割当と元試行のファイル・起動前エラーを照合し、重複照合で安全に扱った。

元runの再開後に終了したPBS割当4件は、PBSのcput合計1137秒／確保4242 CPU秒で26.8%。最初の継続版は待機の供給が増え、12 CPUを確保した。旧PBS workerは小さい割当時の1枠を持ち越していたため、同版全体は443／3448 CPU秒で12.8%。最終版には、既存の割当単位でslot trialとfeedbackを保存するPBS workerを適用した。Romulaptop WSLは中央の受付を戻し、固定releaseを配置した。

2026-11-19 23:05 JST時点で最終版の最初の終了済みPBS割当2件（`267473`、`267474`、各終了コード0）は241／582 CPU秒で41.4%。`267474` では確保4 CPUで実PR-Pauli Q3が2件並行し、両方を中央で照合・回収した。同時に待機キューに4件残っていることを確認した。その後は待機3件を根拠に6 CPUの割当 `267475` を起動した。測定区間の候補・割当構成は同一ではないため、この比率は探索中の観測値として扱う。

23:06 JSTまでに終了した3割当全体では274／936 CPU秒で29.3%。6 CPU割当の利用率は9.3%で、個別割当の利用率にはばらつきがある。WSLの50 GiB到達後、中央原本とバイト数・SHA-256が一致した作業用キャッシュ6件（1,167,388,672 bytes）を回収した。回収記録は継続runの `distributed/WSL_VERIFIED_CACHE_RECLAIM.json`。
