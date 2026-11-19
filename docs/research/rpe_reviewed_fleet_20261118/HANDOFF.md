# PRレビュー・容量対策を統合したRomuTail版

配置・継続・障害復旧の作業前に、[H4 RPE分散探索の共通運用ルール](../rpe_distributed_operations.md)を参照する。PBSの割当前依存関係照合、旧attemptの扱い、容量からの受付復旧、各工程の検収をまとめている。

新規runの倍率・M探索の指定方法、採用releaseと検収は[共通ショット倍率・M探索HANDOFF](../r08_scaled_shots_horizon_20261119/HANDOFF.md)を参照する。

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

2026-11-20 JSTの新規run用currentは `romutail-pbs-task-resources-f67a86291`。
固定実装は `f67a86291a8a0247f99bd952d7ea0b41ab1a14b9`。従来の1枠96 CPU、
PBS合計100 CPUの制約内で、Q3・信号・compile・auditを個々の固定profile・所要資源で
見積もり、同一枠の所有中attemptも含めCPUとRAMの合計を割当前に照合する。
メモリ未計測の準備は他処理を同じ枠に入れず、Q3には従来の既定8 GiBを維持する。
タスク受理・完了後の枠補充には既存の候補準備・先読み・JobStoreを使う。
新しい大枠への更新は、旧割当の実行中・回収待ちのattemptを保全して行う。

実PBSの同じ16 CPU上で人工Q3を各16件実行し、中央採用は旧方式・新方式とも16件、
重複採用0件。従来 `267500` は同時2件・61 CPU秒／203秒（利用率1.88%）、
新方式 `267501` は同時8件・63 CPU秒／165秒（2.39%）。両PBS枠は終了コード0。
利用率は確保CPU数×枠の全稼働秒で割った値で、起動・配送・回収の時間を含む。
実H4の追加検収は0件。96 CPUの混在要求は人工回帰で照合した。実探索では
候補供給とメモリ見積りを確認して増枠する。2 CPU・8 GiBのQ3だけでは350 GiBの
ノードで43件・86 CPUが上限。検収記録は
`.rpe-central/deployment/romutail-pbs-task-resources-f67a86291/TECHNICAL_ACCEPTANCE.json`。
従来の探索 `search-c18e36af4fdf8186b360` は固定版・起動記録・予算・未確定attemptを保持した。

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

## 2026-11-20 旧入力receiptの再照合とPBSの起動短縮

旧パスで `NEEDS_INPUT_SOURCE` となったPR-Pauli 14候補について、後続の
`saved-input-recheck` に固定入力で構成を生成したreceiptが存在した。固定SOURCEの
入力ハッシュ、メタデータ段階の終了、候補の設計・保存構成を照合し、コピーされた
検査済みディレクトリへ戻るよう修正した。価格も揃った3候補を準備順で優先する。
元の価格較正失敗、未回収評価、そのほかの科学条件は元の記録で管理する。

PBSは空き枠の範囲で独立したQ3の入力照合を並行し、台帳とworkerに保存した割当を
次の照会を待たずに起動する。終了時刻・停止・固定版の検査とattempt IDを引き継ぐ。
`search-3566160c558272510a4c` のPBSでは、同じ枠への後続Q3と信号準備の補充を
実機で確認した。終了済みPBS枠の合計CPU利用率は、旧runの19.5%（1601/8200 CPU秒）
から35.6%（1160/3260 CPU秒）へ上がった。workerへの割当保存からQ3本体開始までの
中央値は旧13件の16.2秒から新6件の10.3秒へ短縮した。比率にはPBS枠の立上げと
回収待ちを含み、同一タスクセットの性能比較ではない。

通常の継続先は `search-c18e36af4fdf8186b360`、releaseは
`romutail-pbs-receipt-priority-134711613-r2`。先行runの準備・評価と旧STOPを保全し、
§10.5の照合後に起動した。元の準備会計を保持し、信号枠は70→73→76件、compileと
auditは35件・14件。途中で4件目も価格が揃い、直近の追加3枠は価格済み候補に使った。
RomuTail、server2、PBSに最終releaseを配置した。WSLは最終配置時に停止中で、
引継ぎ時に終了receiptを確認できなかった旧2評価を個別保留にした。
Resumeのクライアント応答が時間切れでも、通常Statusで既存controllerを確認してから
操作する。ソースは `134711613ba4de46fff7c2b1e03db13aee8f7a19`。

優先復帰版では、LD5の2候補（κ=13・幅h、κ=64・幅8h）がQ3へ復帰し、
初回評価を中央で採用した。κ=63・幅4hはPBSの固定信号runtimeで
`openfermionpyscf` がなく信号計算0件で終了し、元attemptを保全した。
κ=2・幅4hは信号枠待ち。PBSの10 CPU枠では後続Q3の5件が本体を開始した。
最終固定版の回帰104件とsubtest 5件がPASS。別途、固定release束縛の旧mock
2件は前版でも失敗し、入力を照合して保存する実経路は既存・新規回帰で確認した。

## 2026-11-20 PBSの固定Pauli信号runtime

Pauli信号の通常REQUESTは固定SOURCEの `requirements.txt` から
`openfermionpyscf 0.5`、`openfermion 1.6.1`、`pyscf 2.7.0` を宣言する。
割当前に版と `openfermionpyscf` のimportを検査する。PBSのPauli信号profileに
`runtime/venv311rc1-pauli-signal-chem05/bin/python` を追加し、既存のPython
3.11.0rc1と数値ライブラリは読み取りで参照する。固定配置は
`romutail-pbs-pauli-chemistry-628e7f38c`。

`search-c18e36af4fdf8186b360` を停止せず、LD5・κ63・幅4hの既存障害だけを
同じSOURCE・入力・丸め条件で、通常の分散キューを使う候補限定の復旧として再実行した。
旧job `prep-72214e2a90fdb8296fa7143e` のREQUEST、OUTCOME、PROCESS_RETURNを
ハッシュで保全した。新job `prep-4ff76743bda54b34e85e3b91` はPBSで
不足ラウンド7～12を計算し、保存ラウンド0～6を再利用して中央が採用した。
同じ候補の自動shot提案を評価したQ3 `cb476bbc3385a5899c7f` はPBS
`267496.CPUcluster-PRISM` で終了コード0となり、中央キューが1回採用した。
判定は `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`。信号とQ3の採用記録、
原本ハッシュは新配置の `search-evidence/recovery-eb8aa01e0a2e63a69147` に保存した。
稼働中の親runは従来の固定版で継続した。復旧結果はこの候補限定runに属する。
親runの最安や公開表に反映する際は、通常の保存評価照合を行う。

10 CPUはPBSの固定上限ではなく、そのときの適格Q3 5件に各2 CPUを確保した結果。
同枠の実績は2051/4690 CPU秒（43.7%）。次の枠は適格な待機仕事とメモリに合わせて要求する。
設定済み上限は1枠96 CPU・PBS合計100 CPU。1件2 CPU・8 GiBの適格Q3が6件あれば、
従来と同じ見積りで12 CPU・49 GiBを要求できる。検収で確保中の枠は変更していない。
PBSとRomuTailの50 GiB受付上限は従来どおりで、元の保全先と照合できた再取得可能な
キャッシュを整理して受付を戻した。

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

## 2026-11-20 実探索の継続とPBSタスク別予約

`search-c18e36af4fdf8186b360` は通常Stopで受付を止め、PBSの終了と19件の
回収ACKを照合してから、`search-300182abb6401030e0a6` を通常Resumeで起動した。
継続版は `romutail-pbs-task-resources-f67a86291-continuation-r3`。
固定実装は `f67a86291a8a0247f99bd952d7ea0b41ab1a14b9`。継続では元の探索範囲、
shot規則、Q3のUNLIMITED、方式比較規則、サイクル条件を継承した。
新規run用currentは `romutail-pbs-task-resources-f67a86291`。

保存成果の§10.5照合ではB0I 255件、PR-Pauli 278件、PR-DF 64件が
再利用可能。値は事前照合の件数。準備台帳は118件を引き継ぎ、
未確定 `prep-e298f74fb699ea3ebde0b633` の元attempt、旧STOP、元の原結果を保全した。
未予約の要求だけを数え、準備枠を信号76→86、compile 35→38、audit 14据え置きとした。
追加した枠と実行件数は別に会計する。

最初の移行候補r2は継承した保存台帳のGit locator 27件を解決できず、科学ジョブを
起動する前に終了。r3では元checkoutのオブジェクトストアを読み取り参照し、同じ
locatorを解決した。通常Resumeのクライアント応答はタイムアウトしたが、中央の
起動記録・キュー・実行を確認したため再送しなかった。詳細はRomuTailの
`.rpe-central/deployment/romutail-pbs-task-resources-f67a86291-continuation-r3` に保存。

PBSは適格な需要に応じて2 → 4 → 8 → 10 → 14 CPUまで拡張した。新しい採用結果を
含む最初の8 CPU枠 `267504.CPUcluster-PRISM` では実Q3 4件を並行実行し、
中央が4件を採用した。PBSの会計では起動・回収を含む割当全体で
678 / 1,456 CPU秒（46.6%）。元runの直近4 CPU枠 `267495.CPUcluster-PRISM` は
2,214 / 14,272 CPU秒（15.5%）。8 CPU枠の稼働中34秒間の差分は
168 / 272 CPU秒（61.8%）。計測区間・タスクの内訳が異なる。
14 CPU・53 GiBの枠は `267506.CPUcluster-PRISM`。96 CPUは設定上限であり、
今回の実探索で実際に確保した最大枠は14 CPU。

追加信号 `prep-229d419c0903b05dfac470aa` をPBSで実行し、中央が
`PREPARATION_COMPLETE` として採用した。PR-Pauli LD5・κ2・幅4hが
`READY_FOR_SEARCH` で同じrunへ戻った。後続の実Q3 `a03100bca051988fd19d` は
PBSで実行を終え、中央が採用した。同じ構成の別配分4件はWSLで実行を確認した。
現行固定経路でRomuTailだけが適格な一部 `PAULI_COMPILE` は50 GiB受付条件で保留中。
追加信号の一部ではPBSの開始前照合待ちが期限切れになり、descriptor待ちの候補も残る。
実行可能な候補は同じ継続runで処理を続けている。

## 2026-11-20 事前照合と50 GiB受付の実探索復旧

稼働中の `search-300182abb6401030e0a6` の固定版と予算台帳を保持した。PBSでは
約195 MBのbundleをスプール経由で転送する待ちと、複数の固定入力照合が重なった。
先に回収した波では、中央の時間切れの後にPBSから照合成功が返っていた。
回収済み入力の先行配置と期限付きの確認は科学実行を発生させない。
終了済みのPBSメタデータキャッシュの全15,287オブジェクトを同じ内容のpackへ
まとめ、6.89 GiBを回収した。容量上限50 GiBのまま約43.2 GiBで受付を再開し、
既存要求4件がworkerの照合を通過した。中央ではそのうち新規2件をPBSに割り当てた。

RomuTailでは転送確認済みattemptの同一packを共有して2.18 GiB、旧メタデータ
キャッシュ2か所の全オブジェクト一覧を保った圧縮で3.14 GiBを回収した。
復旧後の価格用compileは通常経路から3件を実行し、2件は準備完了、1件は
`HELD_CALIBRATION_GATE_OR_EXECUTION` で保全した（その1件の実compileは0件）。
元の未確定attemptを再投入していない。運用中に新たに発生したWSLの終了未確定
Q3 4件も個別に保持し、独立したQ3が継続している。

PBS `267506.CPUcluster-PRISM` の同じ14 CPU枠で、受付待ちを含む区間は
54 / 4,466 CPU秒（1.2%）。信号の再開区間は93 / 1,694 CPU秒（5.5%）。
対象区間はPBSが報告した使用CPU時間・経過時間の差分で計算した。
14 CPU超の増枠は未確認。追加信号の既存枠は86件に到達し、価格証拠待ちも残る。
採用済みの科学条件と予算のまま、実行できる候補が処理されている。

次回配置向けのソースには、Gitオブジェクト照合を1プロセスにまとめる処理、
PBS上の空きCPU内での照合並列化、応答待ちの調整、容量上限に含まれるPBS
メタデータキャッシュの共有、容量逼迫時の転送済みpack共有を追加した。
固定信号の保存入力246件は、新旧照合のハッシュが一致した。PBSゲートウェイでの
固定Git入力照合部分は3回の中央値で2.50秒から0.50秒。稼働中のrunのコードは
固定のままで、実運用の復旧はキャッシュ圧縮と検証済み入力の先行配置で行った。

## 2026-11-20 容量回収の常設化

RomuTailに `rpe-storage-recovery.service` をユーザーサービスとして配置した。
稼働中runを中央台帳から検出するので、runの切替や端末の終了後も監視が続く。
固定監視コードは `32c0d36f17c6e4bef9a4f26bbf013f50cbfd363a`。
稼働中の探索・workerの版および50 GiB上限を保持した。受付を停止する容量に
達した場合は、再生成可能な保存物を調べ、安全に回収できた量と受付状態を記録する。
受付設定がOFFのホストは監視経由で起動しない。

実際にserver2が50 GiBを再度超えた後、周期点検が中央正本と照合できた
キャッシュ27件、約5.28 GiBを回収した。使用量が約44.9 GiBに下がり、
中央がserver2の3つの空き枠を再認識した。WSLでも作業コピー等を回収した。
対象の原結果、台帳、未確定attemptは保持した。技術回帰74件、サブテスト14件通過。
ログは `journalctl --user -u rpe-storage-recovery.service`。

## 2026-11-20 PBS限定の4方式検収

通常入口のscopeに `"execution_hosts": ["pbs-0"]` を追加した。Previewに選択先を表示し、
Prepareで固定設定に保存する。対象runからローカルworkerとrelayの起動設定を除き、
コントローラーも範囲外のホストへの新規割当を拒否する。provider・SOURCE・runtimeの
適格性と工程別予算は従来の規則に従う。省略した通常runの配分は従来どおり。
RomuTailの中央台帳・候補選択・回収・採用をそのまま使う。

### 人工ジョブの実機検収

RomuTailの `.rpe-central/acceptance/pbs-only-four-03` に隔離。
通常Preview、PlanOnly、Start、Resumeを `scripts/run_rpe_search.py` から検証した。
人工run `search-da86d3996b14b14f03bf` は固定controller `62eff1c7f2` で実行し、
PBS `267523.CPUcluster-PRISM` の4 CPU枠を終了コード0で返却した。

| 方式 | 信号 | 価格用compile | 監査 | Q3 |
| --- | ---: | ---: | ---: | ---: |
| PR-Pauli | 1 | 1 | 1 | 2 |
| PR-DF | 1 | 1 | 1 | 2 |
| Full PF | 1 | 1 | 1 | 2 |
| B0I | 1 | 1 | 1 | 2 |

全20件をPBSの計算ノードで実行し、RomuTailが全件採用した。他ホストの実行は0件。
Full PFのQ3と別候補のPR-Pauli監査が並行した。台帳の中央値はキュー登録から
割当まで11.93秒、割当から実行開始まで5.53秒、完了通知から採用まで0.009秒。
完了済みrunの通常Resumeは `REUSED_COMPLETED`。追加起動、二重採用、PBS再投入は0件。

通信検収は `.rpe-central/acceptance/pbs-only-fault-01` の
`search-b9c0e09794cc0365d164`。通常Startから投入したPBS `267530.CPUcluster-PRISM` で
投入応答欠落を再現し、同じ割当を照合した。完了後・転送前の切断を解除すると、
`prep-e78995612fef442d20d07810` が元の `27f347c828fc4d00a50472e47b512d92`
で採用された。通常Resumeで追加起動と二重採用は0件。

### 保存実入力と検収用の固定設定

- `.rpe-central/acceptance/pbs-only-four-02/PBS_SAVED_PREFLIGHT.jsonl` に、
  PR-Pauliの信号・compile・監査、PF共通経路の信号・compile・監査、
  PR-DFのdescriptor・信号・較正の保存REQUEST、計9件のdry-check PASSを保存した。
- PR-Pauliの固定compileは既存の隔離runtimeに保存入力を対応付け、公開固定refの照合に
  専用PythonラッパーからHTTPSを使う。固定refのSHAをPBSログインノードで確認した。
  PF compileは既存のPF専用runtimeを参照し、`openfermionpyscf` を含む依存関係を照合した。
  PR-DFの不足Gitオブジェクトは、固定bundleの前提から特定して隔離キャッシュに補った。
- PR-DFの現行信号SOURCE `9ed5fcc98` と現行価格SOURCE `3b38e4af6` の較正・holdout・
  モデル固定・比較で、固定ファイルとPython環境を照合した。保存REQUESTで較正まで
  通したのは旧 `d887abdc`。現行版の監査までの保存REQUEST通しは未検収。
- PR-Pauli、PR-DF split、Full PF、B0Iの保存Q3入力は、PBSの固定profileで照合した
  （`PBS_SAVED_Q3.jsonl`）。Full PF/M5の実Q3の既報は `925985b4e` での1件。
- 固定profileとwalltimeを入れた実入力用の隔離release
  `romutail-pbs-only-62eff1c7f-saved-r4` は、科学予算をすべて0にした通常Preview、
  PlanOnlyまで確認した。RomuTailの同名acceptanceディレクトリにCLIENT、SERVICE、
  SCOPE、ROUND、PBS_WALLTIME_POLICYを保存した。現行PR-DF監査・価格用実compileと、
  4方式すべての実H4通しは、次回の個別検証で確認する。
  通常運用のcurrent `romutail-pbs-task-resources-f67a86291` と探索
  `search-300182abb6401030e0a6` は切り替えていない。新規実H4信号・実compile・
  実Q3・価格fitはいずれも0件。

## 2026-11-20 PR-DFの価格工程と共通Q3ルートの追加照合

実入力の確認先はRomuTailの `.rpe-central/acceptance/pbs-prdf-current-check`。
PR-DF LD8・幅h・shot autoを既存H4のbinary64表現で指定した。
新規候補が選ぶ `prdf_d2_common_entry` 用に、既設の固定view `23160cc87` と
固定adapter `f67a86291` を指定した隔離PBSルールを追加し、通常Previewで
`pbs-0` に合致することを確認した。4時間・8 GiBは検収用の予約見積り。
splitの保存実績や、共通入口の実行時間を置き換える観測値ではない。
価格工程は較正・holdout・モデル固定・独立比較を現行固定SOURCE
`3b38e4af67f6b673303cff9f963ac1381627c4d0` に留める。

実入力の通常Startは予算・継続照合で停止。継続・予算親を除外して進める変更は
自動審査で拒否された。元の情報を残した読み取り専用の§10.5照合もHOLD。
現探索 `search-300182abb6401030e0a6` は同じPR-DFを探索範囲に含み、
過去の未確定attemptも残っている。理由は `READ_ONLY_DUPLICATE_PREFLIGHT.json`。
隔離実入力runにLAUNCHとキューDBは存在せず、新規実H4信号・価格較正・監査・Q3は0件。
現探索の起動・台帳・固定版は維持した。

独立した人工検収は、通常Startと既存のPBSキューで縮約した依存順を再現する。
人工のモデル固定が生成するreceiptを独立比較へ渡し、holdoutのcompile成果と
そろうまで比較を割り当てない。実価格コードの検収と混同しない。

修正後の人工run `search-3a7e66251806b683cafb` は `pbs-df-chain-02` に保存。
固定 `73af8d69eb` で通常Startし、PBS `267532.CPUcluster-PRISM` の4 CPU枠で、
信号、較正、モデル固定、holdout、独立比較を各1件、Q3を2件完了した。
7件すべてをRomuTailが一度採用し、通常Resumeは `REUSED_COMPLETED`。
PBS枠は終了コード0で返却済み。関連する回帰75件を通過した。

§10.5の追加照合で、設定IDを持たない候補と無関係な準備attemptが
同じ空のIDとして一致していたことを確認した。実値があるIDだけを比較して修正した。
旧PBS計算枠 `267295` と `267454` は終了済み。前者はPauli信号で終了コード1、
OUTCOMEなし。後者はPF信号で科学本体の終了receiptなし。元証跡とUNKNOWNは保全した。
修正後の読み取り専用照合では、実入力STARTの所有保留は現探索だけになった。
現探索には同じPR-DF LD8・幅h・κ1の保留候補があり、controllerは稼働中。
既存の稼働中run保護と予算継承チェックはそのまま適用した。
証拠は `pbs-prdf-current-check/READ_ONLY_DUPLICATE_PREFLIGHT_AFTER_ID_FIX.json`。
通常入口の関連回帰22件を通過。実入力の計算と採用は未実施。

### 停止済みrunからのPR-DF LD8の引継ぎ

元run `search-300182abb6401030e0a6` は通常Stopを記録した。中央driverの
タイムアウト記録、中継の終了、未所有のdriver lock、PBS枠
`267529.CPUcluster-PRISM` の終了コード0、台帳のattemptを照合した。
欠落した終了記録を追加せず、PR-Pauliの未確定5件と元台帳を保全した。

固定controller `8b941595e6` の通常Startで
`search-0285a4cc1bae8ce03c22` が§10.5を通過した。旧配分規則、M=12、
PR-DF LD8・幅h・κ=1、PBS指定を確認した。候補の旧較正
`prep-f8326fce37e9b1bdc80c9b72` はコンパイル前の入力照合で終了していた。
保存スナップショットで確認したλは、要求の `0x1.5d3f2ff8b67fep-14` と
固定providerの `0x1.5d3f2ff8b67ffp-14` でbinary64の1 ULP異なる。
既存の厳密照合と保留を保持した。新規PBS投入、実較正、監査、Q3、採用は各0件。
RomuTailの記録は `.rpe-central/acceptance/pbs-prdf-ld8-continuation-8b941595-r3/`。

### LD8の再計算descriptorとPBSの固定入力

保存スナップショットを両経路で読み、既存のdescriptor導出と固定価格providerの
λが `0x1.5d3f2ff8b67ffp-14`、tailのスカラーが `0x1.9a41efba5338cp-17`
に厳密一致することをRomuTailで確認した。旧descriptor、STOP、台帳を残し、
新しいlocatorを候補の準備記録に固定した。13ラウンドの物理時間、PF回数、
外側反復数、実際の幅も旧候補と照合した。
[descriptor](PR_DF_LD8_DESCRIPTOR.json) · [固定構成](PR_DF_LD8_FAMILY.json)

PBS用の臨時配備では、入力bundleが移動するcontroller HEADを前提にしない設定
`bundle_fixed_source_only` を使用した。固定worker SOURCEしか持たない新規Git環境で
通常 `git bundle verify` と固定入力の取り込みを確認した。旧attemptを起動せず、
PBSの価格用と信号用の専用キャッシュそれぞれで不足オブジェクトを補った。
