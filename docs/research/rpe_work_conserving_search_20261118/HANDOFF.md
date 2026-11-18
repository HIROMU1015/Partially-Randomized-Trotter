# PR最適化の空き枠補充と評価順

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
