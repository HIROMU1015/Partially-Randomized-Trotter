# RomuTail中央管理の移行記録

2026-11-18。担当はCodex rootとsub-agent `direct_transport`、`local0_client`、`prep_inputs`。
各担当の実装検査・保存入力inspectionであり、独立科学レビューや新しい科学性能評価ではない。

## 参照と再利用

指定されたdea1cb530、407fd460a/349558b、91fe799dd、9ad026077の実装・配置記録を先に確認した。
後続の価格検証と候補継続の統合版 `a18d5a04df5ec8c789f274730bff87a2c0a7cc89` を基にした。
既存の通常schedule/round入口、DistributedExecutor、JobStore、f956分散準備、worker、
SSH mailbox、auto枠調整、timing、checkpoint/resume、C公開・Windows同期を再利用した。
全く別のqueueや科学optimizerを作り直していない。

最終制御コードの固定SHAは6da8554e0040f98578667973e8802c680f87982e。
3ホスト通しは89b3da81a0905b41238dea06b0f4f993d28c86d6で実施し、6daは利用者Stopの受付拒否だけを修正した。
通常通しの未変更経路はその検収を再利用し、6daでは人工回帰55件と実行中Stop/Resumeを検査した。
準備wrapperと遠隔workerはd887abdcadf47a969aa2f69c1dd3bdc88c20f1c2で固定。
科学Q3は28d976b99cdf8329adcc1b79327e329e947ef774、timed adapterは349558b748a82f384a4589434a63ef9b5572032c。
準備のnative provider SOURCE・入力manifestは元の契約を保ち、制御ラッパーとhost path配置を別に記録する。
Windows frontendの後続通信修正と配置overlayも、固定科学コードへ混在させず版を記録する。

## 中央状態と保全

server2のcurrentサービスはa18dの価格検証統合版。選択中のstateは空で、稼働中のRPE controllerはなかった。
他担当の異なるrun・開発検査は変更しなかった。引継ぎ前と切替時にcontroller processを照合した。
現在の中央正本は [HANDOFF](HANDOFF.md) のRomuTail配置。server2の古いcurrent入口には移行済み表示を置き、
元pointerをbackupした。移行marker以外の旧state・科学SOURCE・過去のSTOPは書き換えていない。

現行serviceと9本の保存run根から、索引・探索規則・予算・消費記録・checkpoint・停止理由・REQUEST/OUTCOME・
再利用情報をhash検証して取得した。REQUESTファイル5,388件、OUTCOMEファイル5,358件（保存履歴内のファイル数）。
起動済みで結果未確定の21件と未起動9件は元の状態を保存し、今回再投入しなかった。
currentサービスに引き継ぐ実行中runは0件。過去のSTOP済み履歴を新しいactive runとして復活させていない。

metadata snapshotは37,867ファイル、502,558,932 bytes。
archive SHA256 `00cfe1248b1329dce214e617865a6043056823906ea682d324023fa9ae8a0149`。
snapshot manifest SHA256 `a8b039715336646bb6a2220ecadf79642172f62e80712c5fd600768aba59dcd2`。
[元path・未確定job・未取得rawの一覧](transfer/STATE_TRANSFER.json)。
大きい原成果のうち84,231,298,005 bytesはserver2の元保存場所で保全し、再利用で必要となる実体は出所とhashを照合して取得する。
今回の保存metadata転送を、すべてのrawの複製や未確定結果の採用完了とは扱わない。

## 保存実入力の工程×host

| 工程 | RomuTail | server2 | local0-wsl | server1 |
|---|---|---|---|---|
| Full PF signal / compile / audit | 保存inspection各PASS（既存再利用） | 各PASS（既存再利用） | 各PASS（既存再利用）、受付OFF | 停止中・未確認 |
| B0I signal / compile / audit | 保存inspection各PASS（既存再利用） | 各PASS（既存再利用） | 各PASS（既存再利用）、受付OFF | 停止中・未確認 |
| PR-Pauli signal / compile / audit | 保存inspection各PASS（既存再利用） | 各PASS（既存再利用） | 各PASS（既存再利用）、受付OFF | 停止中・未確認 |
| PR-DF signal / compile / audit | 不足補充後3 PASS | 3 PASS（既存再利用） | 不足補充後3 PASS、受付OFF | 停止中・未確認 |
| Q3 | 固定routeの既存検収を継承、新規実Q3なし | 同左 | 同左、受付OFF | 復帰時に検査 |

保存準備inspectionは稼働3hostで各12 PASS。再実施したのはRomuTail/WSLのPR-DF計6セルだけ。
descriptor固定commit62ee29f、元config SHA52e221c7、provider278ファイルを取得・照合した。
private本文はGitに含めない。[配置・READINESS・入力出所](preparation/HANDOFF.md)。
保存入力inspectionは新しい科学REQUESTの実行資格ではない。新runは通常のSOURCE/入力/compiler/fingerprint照合を行う。

## 受入記録

個別の検査結果・実機検収は `checks/`、`direct_transport/`、`preparation/` に保存する。
人工runの出力と本番公開結果は保存先を分ける。server1は停止中のため対象から外し、復帰手順を残す。
すべての検査で新規実H4信号・実compile・実Q3・本番探索の起動は0件。

残る工程は、新規科学実行許可後の完全REQUEST別admissionと実探索、およびserver1復帰後の実機検査。
今回の技術検収を科学的GO判定へ読み替えない。

通常factoryが付けるprice-validation終了処理がmethod viewへコピーされ、準備継続中に共有queueを閉じる不具合を人工通しで発見した。
method viewはmethod単位の終了処理を使うよう修正し、通常factoryで複数回の人工最適化後もqueueと準備受付が開いていることを2件の回帰検査で確認した。
最初の人工runは `SETUP_FAILED_EXECUTOR_CLOSED` として原ログ・queue・受領成果を保全した。成功した検収へ読み替えていない。
修正後は新しい固定controllerと独立spoolの人工runを使う。科学kernel・閾値・入力・旧STOPは変更していない。

## 実機による人工通しとWindows運用

[3ホスト検収](direct_transport/REAL_THREE_ACCEPTANCE.json)では、保存先
`.rpe-central/acceptance/real-three-v2-d887abdca` の13件（signal4・compile1・audit1・Q3 7）を
RomuTail7、server2 4、local0-wsl 2件で採用した。WSLはauditとQ3を実際に処理した。
[主な故障注入結果](direct_transport/REAL_THREE_FAULT_CHECKS.json)、
[host/clockとSOURCE](direct_transport/REAL_THREE_PROVENANCE.json)。
切断は3つの実ホストを使ったrelay経路の人工遮断であり、Windows本体や物理networkの停止ではない。
切断中の元ホスト完了・未転送保存、他hostの進行、同attempt・同hashの回収、Pause中の回収を確認した。
compile/auditは異なるhost、signalとcompileの重複時間は同じcontroller時計で確認した。
Q3と準備の予約が各hostの同じ1枠で重ならないことも同じ時計の台帳で照合した。

完了後の通常Windows Resumeは `REUSED_COMPLETED`。全LAUNCH hash、ASSIGN/ACCEPT/ADOPT数、
attempt・成果hashに変化がなく、追加起動0。
[単一host参照](direct_transport/REAL_THREE_SINGLE_HOST_REFERENCE.json)では同じ保存人工Q3 REQUEST 7件を
固定d887人工leafで逐次実行し、MSE上下界・判定が全件exact一致した。実分子信号や新しいDP科学評価は使っていない。

Windows標準SSHではこの起動環境で戻り待ちが継続した。実機で反復確認した既設Git for WindowsのSSHを明示し、
通常のalias・認証を保った。[Windows反復検収](checks/WINDOWS_GIT_SSH_ACCEPTANCE.json)、
[最終版r4の範囲と版](checks/WINDOWS_R4_METADATA.json)。
PR-Pauli LD2..3、絶対κ[1/10,1]の両端込み、binary64の要求幅.1/.2、shots autoがoptimizer入力まで一致した。
これは0 Q3予算のmetadata検査で、本番Startは行っていない。

Cの通常公開branchをoriginから照会して、指定のOneDrive探索マップへ実結果11 runを取得した。
5ファイルのhashを照合し、古いRomuTail cloneのHEADだけを見る経路ではないことを確認した。
[同期実機記録](checks/WINDOWS_C_MAP_SYNC_GIT_SSH.json)。
人工runのStart再送と同期hookは隔離したWindows map/stateへ接続し、通常Cへの人工公開は0。
実行中の1 monitor、同じrunの重複抑止と600秒設定、終了時の最終同期・「最終結果は公開待ち」・monitor停止を確認した。
実際に10分間経過させた検収とは区別する。

最終の [Stop/Resume検収](direct_transport/ACTIVE_STOP_FINAL.json) は
`.rpe-central/acceptance/active-stop-v3-d887abdca` の通常Windows入口を使用した。
利用者Stop時に実行中だった2件目は元workerで完了し、先に完了していた1件目とともに保存された。
停止はEXIT0・`USER_REQUESTED_STOP`、Resume後もEXIT0。元の2jobのattempt、原成果hash、
LAUNCH hash、ASSIGN/ACCEPT/ADOPT数は変わらず、追加起動0件。
利用者Stopで記録される既存KeyboardInterruptログは保存し、科学UNKNOWNやSTOPを消していない。
途中の検収で見つけた受付拒否のKeyErrorは6daで修正し、最終runには発生していない。
[最終人工回帰55件](checks/FINAL_CONTROLLER_REGRESSION_6da8554e0.log) は固定checkoutで全件PASS。

[実行中同期monitor](checks/WINDOWS_RUNNING_SYNC_MONITOR_V3.json) と
[正常停止後の自動最終同期](checks/WINDOWS_SYNC_MONITOR_FINAL_V3.json) は成功したQ3 jobの実行中から確認した。
通常60秒の状態pollでmonitorがSTOPPEDとなり、600秒の定期同期は無効化され、同じTask1個がReadyへ戻った。
Resumeは同じrunの新しいcontroller invocationを使う。controller正本・公開元・Windows同期はRomuTail向け、
人工保存先と実際のOneDrive mapは別であり、人工結果が通常Cへ公開された件数は0。

最初のsetup run、2本のmetadata不整合による0job fixture、修正前Stopの診断記録も各元directoryで保全した。
それらを最終PASSには含めていない。所有する診断プロセス・検収relayだけを終了し、
通常relay、利用者SSH、他担当run、保存cacheを維持する。

[通常Resumeの自動同期hook](checks/WINDOWS_V3_RESUME_AUTO_SYNC.json) も11項目PASS。
同じrun/watch/taskに新しいinvocation/generationを結び、即時同期、RUNNING観測、
最後のRUN_FINAL・公開待ち表示・task Readyへの移行を確認した。実利用mapの5ファイルは変わっていない。
[最終清掃](direct_transport/CLEANUP_FINAL.json) で全人工fixtureのrelay/controller/workerを終了し、
通常relay PID516949と実WSL受付OFFを維持した。
