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
