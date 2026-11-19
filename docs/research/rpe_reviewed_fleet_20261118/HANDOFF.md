# PRレビュー・容量対策を統合したRomuTail版

初回の採用版は `romutail-reviewed-pr-3b38e4af6`。統合実装は `3b38e4af67f6b673303cff9f963ac1381627c4d0`、取り込んだPR実装は `e433938dd39ac4c2cb6b7ab8865e41c0b996d232`。直前の `romutail-startup-e97575eb0` から、起動時の一括mapperと既存の配送、準備bundle・Gitオブジェクト共有、中央受領ファイルの共有を継承した。[検収記録](ACCEPTANCE.json) · [探索設定とアルゴリズム](../../PERFORMANCE_METHODS.md)

## PR-DF保存信号照合の追加採用

PR-DF保存信号照合を採用したreleaseは `romutail-reviewed-pr-9187e39b5`。optimizer SOURCEは `9187e39b5af87bfca4132875a59ce40a756e4cb6`。保存信号照合の上流は `f79708a7126ecbd827747bd261289d93496d9fa4`。同一物理キーで数値・検査・参照入力が一致した保存信号を使う。差があればラウンド・出所・相違項目を残し、該当候補の信号・価格準備を投入前に保留する。

RomuTail、server2、local0-wsl、PBSゲートウェイへoptimizerを配置した。worker・adapter・runtime/profileの固定版と工程別の受付条件は前の検収を継承する。Windows/CLIは明示指定とrelease無指定の両方でPreview通過。回帰32件中30件が通過し、ルート登録の残り2件は旧採用版でも同じ失敗を確認した。server2の旧R08はユーザー停止で終了しており、旧runの固定版とSTOPを保全した。新規信号・compile・fit・Q3は各0件。[追加検収](PR_DF_RECONCILIATION.json)

## PBSの共通所要時間ポリシー

PBSの共通ポリシーを最初に採用したreleaseは `romutail-pbs-walltime-976c26c17`。controller SOURCEは `976c26c17d04540b79fcdc7c69c1a34b85ccd542`。RomuTail、server2、local0-wsl、PBSゲートウェイへの配置を確認した。共通ポリシーを更新する際は、対応する固定SOURCEとプロファイルを照合し、新releaseとして通常のregistryへ登録する。 [ポリシー設定](../../../configs/rpe_method_optimizer/pbs_walltime_h4.json)

- 通常のPreviewとStartはreleaseからポリシーを解決する。StartでポリシーID・ハッシュ・適用した秒数をrunのPBS設定に固定し、StatusとResumeでも表示する。
- Q3の4時間見積りは、固定プロファイルに合うPR-Pauliの `pauli_b_common_entry` とB0Iの `pf_b_pf6` に適用する。準備は工程・provider・SOURCEが一致する9ルールを使う。PBSは見積りにdrainの猶予を加えた時間を確保できる枠を使う。PR-DFなど見積りが未登録のQ3はPBSで `PBS_WALLTIME_ESTIMATE_UNAVAILABLE` となり、他の適格workerは利用できる。
- 個別runで見積りを変更する場合のみ、開始要求の `scope.pbs_walltime_overrides` にルールIDと秒数を指定する。例は `{"pauli-common": 18000}`。`0` はそのルールのPBS割当を無効にする。既存runは記録済み設定でStatusとResumeを行う。
- 回帰72件がPASS。WindowsとCLIの通常Previewは明示指定とrelease無指定の両方で新releaseを解決した。B0IのCLI Previewと、個別上書き `0` のCLI Previewも確認した。PBSゲートウェイの保存入力照合はPR-PauliとB0IがPASS、PR-DFは見積り未登録でHOLD。実信号・実compile・実Q3・PBS投入は各0件。
- `search-6be05579540d7ed291b7` は切替前後で同じcontrollerと固定release `romutail-reviewed-pr-9187e39b5-h4-continuation` を使用している。このrunのPBS所要時間設定は未設定のまま継続する。ローカルの検収記録は `.rpe-central/deployment/romutail-pbs-walltime-976c26c17` に保存した。

## PR-DF信号workerの修正

PR-DF信号workerの修正を採用したreleaseは `romutail-df-signal-9ed5fcc98`。controllerとPR-DF信号runnerは `9ed5fcc98d38c328c447f4270eabc73090fcaa02`。上流 `4d4fc04a10fb07a5561bebc0ecf7333c69b808f8` の固定provider読込と並行登録時の旧失敗照合を取り込んだ。候補の一覧をlock内で取得し、固定Git入力をlock外で読む。同一物理候補の失敗記録と自動再試行禁止を引き継ぐ。上流の後続 `438af4c3f` はR08専用の固定入力と引継ぎ記録。 [上流修正](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4d4fc04a10fb07a5561bebc0ecf7333c69b808f8)

- 通常入口で `applied.df_signal_source` を確認できる。分散REQUESTの `DF_SIGNALS` だけをmanifestの `df_signal_runner` へ固定し、workerは従来の計算provider `fefd548ecd56f60d72031b3d385f6f5e6ee95d06` を自身と子プロセスの両方で読み込む。NumPy 1.26.4、SciPy 1.14.1、科学Q3 SOURCE、価格provider、共同キューと受付設定を引き継いだ。PBSのPR-Pauli・B0IのQ3見積りは各4時間。PR-DF信号とQ3のPBS見積りは未登録。
- RomuTail・server2・Romulaptop WSL・PBSゲートウェイのrelease/profileと固定providerの子プロセス読み込みを確認した。PBSの新release専用ストアには不足していた固定参照を補った。server1が復帰したら同じmanifestで配置し、固定参照が不足する場合は `DF_SIGNAL_FIXED_REFERENCES.pack` を当該releaseの `objects.git` に `git index-pack --stdin` で取り込んでから入力照合を行う。
- server2のR08保存REQUESTを8候補・39ラウンド照合した。人工・回帰は84テスト中82件PASS。残る2テストの計4エラーは既存モックの準備REQUESTに必須フィールドが不足するもので、旧SOURCEでも再現した。WindowsとCLIの明示・無指定Previewで、信号runnerとPBS設定を確認した。検収記録は `.rpe-central/deployment/romutail-df-signal-9ed5fcc98` と同名のfleetディレクトリに保存した。
- 既存の中央runは固定releaseで継続し、今回の切替でR08の起動・切替は行っていない。価格未確定の6候補は上流の残件。新規の実信号・実compile・fit・実Q3・PBS投入は各0件で、実信号からQ3までの通しは将来の承認済み探索で確認する。

## 共有mailboxの旧attemptを除外する中継

新規runの `current` は `romutail-owned-relay-0fc8268cf`。固定controller SOURCEは `0fc8268cfc7a2c6322986ad5a6895c185604d58e`。RomuTailの中継は接続中のrunのJobStoreを読み取り専用で照合し、実行ホスト・job ID・attempt IDがそろう通知だけを送る。遠隔の旧controllerへ接続する場合は、所有不一致を明示する409だけを処理して次へ進む。外したattemptの結果とackはその所有runに残る。現行run所有の回収は新規割当より先に行う。

- `search-6be05579540d7ed291b7` は2026-11-19 00:18 JSTに中継のみを交換した。controllerと固定設定は継続。旧B0I `bf164759c491f8d2c68b` は元runの台帳で `UNKNOWN`。server2の既存受付停止も保持した。WSLはQ3を4件受領し、少なくとも1件を中央が回収・受理した後に次の割当が届いた。
- 人工・回帰12件がPASS。別runの未確定attemptと同一job IDの旧attemptが混在しても、現行runで割当・回収が各1回であることを確認した。WindowsとCLIの通常Previewで新releaseを確認した。検収記録は `.rpe-central/deployment/romutail-owned-relay-0fc8268cf`。
- 旧固定版のcontrollerが中継を新規起動したときは、中継のSOURCEを確認する必要がある。旧版に戻っていれば、そのrunの `distributed/relay/CONFIG.json` を使い、中継だけを `--stop` でdrainし、同じ設定のまま修正版の `run_rpe_q3_relay.py --config CONFIG.json --start` で起動する。controllerの稼働、relay lockの解放、利用者STOPの有無を切替時に確認する。

## 稼働中のH4探索へPBSを追加

2026-11-19 JSTに `search-6be05579540d7ed291b7` を通常Stopでdrainし、同じ探索条件とチェックポイントから継続run `search-f0ba946f506fd1e8e386` を通常Startで起動した。固定releaseは `romutail-pbs-h4-442bef4b0`、controller SOURCEは `442bef4b04903023104e08c93478d7c5162cb105`。範囲、shot規則、累計Q3上限なし、準備予算、科学Q3 SOURCE、既存host受付を引き継いだ。

- 元runのQ3台帳は受理98件、配分前の取消26件で、所有中のattemptは0件。元runのSTOPを保持した。継続時に同じ配分の再試行を防ぐため、旧未確定2件と、実Q3開始前に終了して受理されたUNKNOWN 1件を個別に保留した。§10.5でB0I 53、PR-Pauli 148、PR-DF 64件を照合した。
- PR-PauliとB0Iの既存の4時間見積りを適用し、PBSの `267266.CPUcluster-PRISM` を `CPUcluster011` の96 CPUで確保した。最初のPR-Pauli Q3 `e2b92bf64e9c918f6d59` はworkerの `SCIENTIFIC_ROUTE_START` まで確認した。実行可能なPBS割当は、固定SOURCE・入力照合と空き枠・残りwalltimeを満たすタスクに限る。
- 回帰39件がPASS。新runは通常 `Status`・`Stop`・`Resume` でrun IDを明示する。運用記録は `.rpe-central/deployment/romutail-pbs-h4-442bef4b0`。

## PBSのCPU要求を実行待ち需要に合わせる

`h4-pbs-walltime-v2` は、固定workerが実行でき、他ホストの空き枠で処理しきれないREADYタスクからPBSへのCPU要求を決める。現在の固定プロファイルでは1件あたり2 CPU。1件なら2 CPU、3件なら6 CPUを要求する。新規枠の単一ノード上限96 CPUとプロジェクトの合計上限100 CPUを適用する。メタデータ照合待ちの準備工程だけの場合は、1件分の枠で照合を進める。

すぐに実行できる見積り登録済みQ3がそろっていれば、初期同時実行数もその需要に合わせる。PBSで保存したQ3の所有プロセスのピークRSSは最大約1.8 GiB。初期枠の算出には1件あたり8 GiBの余裕を見込み、以後は既存のCPU・メモリ・計時による自動調整を使う。準備工程のための350 GiBのメモリ要求と固定の所要時間・実行プロファイルを継承する。 [共通設定](../../../configs/rpe_method_optimizer/pbs_walltime_h4.json)

新規run用の `current` は `romutail-pbs-sizing-48ebf75d5`。固定SOURCEは `48ebf75d5b20459b66fb3e39ace96c8fc54ab345`。通常Windows・CLIの明示指定および指定なしPreviewと、回帰55件がPASS。

稼働中のH4探索は `search-f0ba946f506fd1e8e386` から `search-8b10307fd1b9fc375b6b` へ引き継いだ。旧PBSの3枠は終了し、旧PBSで受理したattemptは10件。§10.5でB0I 81、PR-Pauli 181、PR-DF 64件を照合した。最初のPBS枠 `267269.CPUcluster-PRISM` は `CPUcluster011` 上の4 CPUで、PR-PauliとB0IのQ3を各1件並行起動した。実測ピークが1件あたり2.06 CPU相当に達したため、既存の自動資源制御は稼働中の2件を継続させながら追加受付の上限を1件に下げた。運用記録は `.rpe-central/deployment/romutail-pbs-sizing-48ebf75d5-h4-continuation`。

## 2026-11-19 PBSの候補補充とWSLの容量復旧

H4の継続runは `search-bec31e449b46e6576743`、明示releaseは
`romutail-pbs-candidate-feed-aacbd846f`。元run `search-3a962b43686c76679768`
は受付停止後に結果を回収し、終了コード0を確認した。元runの準備台帳109件とQ3
55件を引き継いだ。§10.5ではB0I 255件、PR-Pauli 278件、PR-DF 64件の保存評価が
再利用可能だった。既定のcurrentは `romutail-mixed-methods-0c67daacd`。

予算に達した準備要求がPBSのCPU需要に含まれ、確保後も実行に移れない状態があった。
準備工程の予算残数の範囲で、未予約の入力照合だけを需要に数える。空き理由では、
準備待ち、予算待ち、適格な候補の不足を分ける。今回の探索では保存価格が揃い、
固定信号入力を照合できたPR-Pauli 3候補に信号枠を追加した。
信号は67→70件、compileとauditは35件・14件。旧消費とQ3の累計上限なしを継承した。
3件はRomuTailとserver2で完了し、候補が探索に戻った。

PBSでは準備から戻った候補のQ3を1件採用した後、`267485.CPUcluster-PRISM`
の14 CPU枠に別の評価を6件渡した。6件とも計算ノードでQ3本体を終了し、中央が採用した。
23:41 JSTのPBSスケジューラー集計では、CPU時間を確保CPU数×稼働時間で割った利用率は
直前runの10.2%（1613/15804 CPU秒）から17.5%（895/5112 CPU秒）へ上がった。
14 CPU枠は568/3262 CPU秒で終了コード0。全96 CPUを埋めるだけの継続的な候補数は、
この観測では確認できていない。

WSLの保留4件は元attemptで実行・回収し、中央で割当と採用が各1回だった。
実機では中央とバイト単位で一致したキャッシュ12個を整理し、約2.17 GiBを回収した。
稼働中の専用容量監視は、50 GiBに達して通常回収で足りない場合に、中央CASとの
SHA-256照合を通ったWSLコピーを追加で回収する。実行中、未回収、未確定attemptの
入力を保護する。使用量が下がれば次の容量確認から受付が戻る。WSLの受付スイッチは
従来の状態を使う。この監視を別の継続runでも使う場合は、そのrunの実行に合わせて
`scripts/reclaim_rpe_worker_storage.py RUN_DIRECTORY --watch` を起動する。

固定版はRomuTail、server2、WSL、PBSに配置し、人工・回帰72件とsubtest 2件が通過した。
server1は停止中。旧run、配分・採用の証拠と使用量ログはRomuTailの
`.rpe-central/state/runs` に保存している。

## 通常入口

- Romulaptopでは既存の `Invoke-RpeSearch.ps1 -Action Preview -RequestFile SCOPE.json` を使う。RomuTailのCLIは `scripts/run_rpe_search.py --config .rpe-central/CLIENT.json --action preview --request-file SCOPE.json`。
- 承認済みの範囲・準備予算・Q3予算を指定して `Start`。既存runには同じ入口で `-Action Status|Stop|Resume -RunId RUN_ID`、CLIでは `--action status|stop|resume --run-id RUN_ID`。既存runはrunに保存されたreleaseを解決する。
- 正本はRomuTailの `.rpe-central/state`。登録は `.rpe-central/RELEASES.json`、版ごとの設定は `.rpe-central/deployment/RELEASE_ID`、manifest・配置・profileは `.rpe-central/fleet/RELEASE_ID`。新releaseは既存のregistryに登録し、明示Preview後に `--activate --expected-current OLD_RELEASE` で採用する。
- 停止中のserver1は、復帰時に同じmanifestと専用host layoutを使い `manage_rpe_release_deployment.py --install` で配置する。固定入力・runtimeを照合し、人工ジョブの配送と回収を確認してから既存の受付設定で参加させる。

## 配分と保存

新規runのPR-Pauli／PR-DFでは `self_price_floor`、`review_search_order`、`common_pauli_price_floor`、`allocation_exchange.enabled` を有効化した。同一モデルで根拠を確認できたPauli範囲でκ方向の除外を使う。PR-DFの高価な候補は準備順を調整する。RomuTailが候補選択・既存予算・採用を一元化し、準備とQ3は従来の同じキューを使う。

`pauli_b_common_entry` と `prdf_d2_common_entry` ではREQUESTに圧縮証跡と厳密残差並列を記録し、適合する固定adapterのあるhostを選ぶ。cacheは各hostのprofileが指定する `-reviewed-pr` のパスで、各ディレクトリの設定上限は40 GiB。科学Q3 SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。旧形式のPF・PR-DF split用adapterは `349558b748a82f384a4589434a63ef9b5572032c`。

PR-DFの価格較正・holdoutと対応する監査を新runnerに固定し、X/Yをhost profileのCPU予約内で並列化する。元の共有イベントdigest、個別receipt、較正・holdout区分、stage別予算を保持する。実行する価格providerは `ea9e66092465d39dfcd4aeb6fea8e9f09808cabb`。PR信号の準備SOURCEは `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`。

圧縮済みの原証跡は、runに記録したadapterで次のように復元できる。

```sh
"$FIXED_PYTHON" -B "$FIXED_ADAPTER/scripts/run_rpe_optimizer_q3_job.py" \
  --mode export-evidence --request "$JOB/REQUEST.json" --output "$JOB" \
  --evidence-directory "$NEW_DIRECTORY"
```

## 検収の範囲

RomuTailとserver2の人工runで、先行Q3の実行中に別候補の信号・価格準備が揃い、同じ候補IDがserver2の空き枠へ戻った。割当・採用は各1回、同一run再開後の追加起動は0件。別の人工検査で、圧縮証跡を中央の回収経路に流し、exact整数を保持したまま原サイドカーをバイト単位で復元した。合成ケースではworker側のattempt保存量が約50%、中央のjobと転送を合わせた重複除外後の保存量も約50%減った。server2の既報約54%はR08の完了42件の結果ファイルに対する値であり、今回の中央全体使用量の測定値ではない。

新規実H4信号、実compile、fit、実Q3は各0件。価格準備は保存入力を使ったREQUESTの構築と人工の軸・receipt検査を実施した。外部runnerの `prepare` 呼び出しは自動審査で拒否され、コード内の非実行検査に切り替えた。既存中央run `search-be59f46fceee86257816` は `romutail-b0i-hold-b89818df0-r08` を保持し、server2のR08は元のtmux、固定SOURCE・出力先で継続した。新規runの実計算は承認済み範囲の通常Startから行う。[R08の記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/882e66644ee08034a63aa92924fee2501b4b34a1/docs/research/pr_r08_storage_review_resume_20261118/HANDOFF.md)

保存REQUESTのdry-checkはRomuTail・server2・local0-wslで両PR方式がPASS。WSLは新release専用のPython 3.11.0rc1 launcherにSQLite拡張を追加し、PR-DFの不足Gitオブジェクト2件を補った。PBSゲートウェイではPR-Pauliの保存入力がPASS。PR-DFの保存fixtureでは固定providerの旧 `INPUTS.json` が不足し、該当routeはHOLD。PBSの実Q3には既存の残りwalltime照合と工程別見積りが必要。server1は停止状態。

人工回帰126件と、固定DP実装を使ったprefix cache・並列残差の人工回帰15件がPASS。
