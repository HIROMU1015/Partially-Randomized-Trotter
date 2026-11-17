# 最新通常運用版へのPBS統合（2026-11-18 JST）

技術統合・配置・採用を完了した記録。固定SOURCEは `95a74e4486131a97bd5235221bf2d8ab9f99f591`、
新releaseは `romutail-pbs-current-95a74e448`。着手時currentは
`romutail-price-fleet-e917165ba`、採用後currentは `romutail-pbs-current-95a74e448`。
[採用前後・所有状態](checks/ADOPTION.json) を確認して切り替えた。
中央正本はRomuTailの `/home/abe/Project/Partially Randomized Trotter/.rpe-central/state` のまま。
今回の新規実H4信号・実compile・実Q3・価格fit・本番探索はすべて0件。

## 採用元と差分

remote tip、実registry、専用worktree、実行processを確認した。
採用済み中央は公開 `5760fffc850e1da06112cc9698a37e0726bab057`／SOURCE
`e917165ba1f36b7627614cd25e88b5108d9b20b9`。
PBS合流版は公開 `4daddd65ee686d70b3ec9b93b887e53fa42e4325`／SOURCE
`925985b4ea76a15f8a07c507179d4fa6c492eabd` までで、未activateだった。
その後のPBS科学記録は `840525ed6a2dc3e018dd568898b9bc9e06349578`。
各branchにこれらを越える公開後続はなかった。進行中の科学controller/relayはなく、既存operation gatewayを保全して着手した。

既存 `codex/rpe-romutail-controller-migration-20261118` を継続し、PBS branchをmergeした
`acb13d4d8` を経てSOURCE95を固定した。別の探索正本・queue・controllerを追加していない。
e917の共有価格store、release manifest、環境別配置、共通入口、工程別予算、run所有relay、
method終了処理、Stop/Resume、Windows/WSL輸送、探索マップ同期を保持した。
PBSのPbsManager、JobStore、共有spool、token/job/attemptの復帰、qsub応答欠落照合、
残りwalltimeの受付停止・drainを取り込んだ。

追加は通常Gatewayの既存人工hookからPBS driverを選ぶ接続、host別のPBS transport/runtime役割、
PBS worker参加・assignmentの既存manifest照合、保存exact値の技術的回収検査である。
科学REQUEST・job ID・hashの外側に要求releaseを置き、不一致を元workerのacquire前に拒否する。
旧profileには新しい必須fieldを要求せず、既存runは元の固定版で再開できる。

科学Q3 SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`、timed adapter
`349558b748a82f384a4589434a63ef9b5572032c`、通常準備wrapper/既存3host worker
`d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`、価格worker
`a18d5a04df5ec8c789f274730bff87a2c0a7cc89` と各providerの契約を維持した。
PBSのcontroller/gateway/transport workerは95であり、科学SOURCEと区別する。
公開component manifestとprivate host layoutを既存fleet形式で管理し、全工程を同じSHAへ変更していない。

## 回収Pythonと保存exact値

新PBS設定の `gateway_python` は明示的に
`/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python`。
既存専用Pythonを再利用し、system Python・共有venv・科学SOURCEは変更していない。
PbsManagerが保持する一つのGateway設定を通常exchange/collectと終了後harvest・再接続が使用する。

旧925の完了済みFull PF attemptから、保存OUTCOME・collection receipt等のbyteだけを
新しい検査用spoolへコピーし、既存gatewayの通常 `/v1/complete` 配送とpost-exit harvestへ通す。
7168桁のexact整数はbyte・整数値・decimal hashで照合し、float化・丸め・切詰めを行わない。
元attemptへの書込み、再計算、再採用、qsubは行っていない。
[回収結果](harvest/EXACT_TRANSFER.json) と [原本保全](harvest/ORIGINAL_PRESERVATION.json) がPASS。
OUTCOME全35,529 bytesと長整数4項目（7168/7174/7741/7747桁）を一致確認した。
初期checkerは7168を最大桁数と仮定して停止したため、同じコピーを使う独立したmetadata検査callerでその仮定だけを訂正した。
[検査callerと出所](harvest/README.md) を分けて記録し、固定95のcontroller/gateway/workerは変更していない。

## 通常入口の検収

[固定版の技術テスト](checks/TECHNICAL_TESTS.log) は173件実施、172 PASS・1既存環境依存skip。
価格store/予算/manifest/共通操作、PBS割当・復帰・walltime/drain、method終了・Stop/Resume、
巨大整数転送、誤release拒否を確認した。既存の科学worker/profile・保存入力の検収は再利用する。
この結果を科学的GOや独立科学reviewとして扱わない。

通常の `Invoke-RpeSearch.ps1` と `scripts/run_rpe_search.py` を使う。
新private clientから、同じGateway/通常schedule入口へ人工scopeを送り、
Preview→PlanOnly→Start→PBS計算node→中央照合・採用→完了Resume/同じStart再送を通した。
[通常入口の実機証拠](entry/FINAL_ACCEPTANCE.json) と [中央の採用・再開照合](checks/NORMAL_ACCEPTANCE.json) はPASS。
人工8task（signal2、共有compile1、別host audit1、Q3 4）を一度ずつ採用し、
PBSでcompile1/Q3 4、RomuTailでsignal2/audit1を処理した。
PBS `267198.CPUcluster-PRISM` はworkq、1node/2CPU/4GB/4分、node `CPUcluster031`。
[F/exit0・worker EXITED](checks/PBS_FINAL_SNAPSHOT.json)、scheduler run_count1、残存自分PBS job0。
完了Windows Resumeと共通CLI Start再送の追加LAUNCH/ASSIGN/ACCEPT/ADOPT/PBS再投入はすべて0件だった。
移植検収専用 `check_rpe_pbs_saved_q3.py` の再実行で代替しない。

人工専用state・registryは `.rpe-central/acceptance/pbs-current-entry-20261118/gateway-v2`、
fixtureは同parentの `fixture-v2`、価格storeも同parent内で隔離する。
PBSのprefix/artifact cache、spoolも検収専用とし、通常C/publication・比較表・実マップへの公開と同期を無効にする。
初期の未起動fixture/gatewayは配置検討記録として残したが、job/科学callは0件である。
通常SERVICEの科学false・Q3/準備予算0は変更せず、人工fixtureだけに人工予算を設定する。

qsub応答欠落・共有spoolからの元attempt回収・controller再開の詳細障害検査は
[925固定版の人工証拠](../rpe_pbs_central_integration_20261118/cluster/COMPARISON_AND_RESUME.json) を再利用する。
今回の変更部分は新SOURCEの通常入口・実PBS人工処理・完了再開と保存exact回収で照合し、旧証拠を95へ付け替えない。

## 実H4証拠の帰属と残る検証

[既存の実H4移植記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/840525ed6a2dc3e018dd568898b9bc9e06349578/docs/research/rpe_pbs_h4_q3_portability_20261118/REPORT.md)
は**925固定版のFull PF/M5・保存REQUEST一件だけ**である。
linear H4、1 Å、STO-3G、tol-only1e-8/観測DF rank10、L_D10/FULL_H/random tailなし、
保存h約0.1・要求幅16h約1.6・m0..12という元条件を保持する。
元job `98693b829295fc884330`、attempt `18c2c7bc2d7b4a5581ef2543b1359e1b`、
PBS `267197.CPUcluster-PRISM` の検収で、exact MSE上下界・判定・gateが元結果と一致した既存記録を参照した。
この一件を再計算せず、95の実H4検収と呼び替えない。元の閾値不通過判定も維持する。

**新統合版の通常Startからの実H4実行は、次の採択済み探索で確認する事項として残る。**
本番PBS資源・適用可能なtask時間見積りは、次回の採択scope/既存契約に従って設定する。
通常PBS設定の `task_seconds={}` を一件のFull PF実績から全方式共通480秒へ変更していない。
既存の時間見積り未設定時の科学task受付保留を維持し、人工のPASSを全方式の科学実行資格にしない。

## 配置・所有・保存状態

新しい固定checkout・SERVICE・manifestを別directoryへ配置し、旧登録SERVICEと完了runは変更しない。
RomuTail/server2/PBSへ必要なcodeを配置し、WSLは必要なoptimizer配置のみ更新して受付OFFを維持する。
server1には接続せず、同版packageと復帰後の保存入力・人工参加手順を残す。
各hostのsource/runtime・再利用・不足は [fleet/配置記録](fleet/DEPLOYMENT_RECEIPTS.json)、
通常入口は [entry/HANDOFF](entry/HANDOFF.md) に記録した。
[通常Windowsの明示/default Preview](entry/FINAL_ACCEPTANCE.json) は同じ新release・SOURCE・runを解決し、
[最適化器が読む設定](checks/NORMAL_RESOLUTION.json) で中央RomuTail・PBS gateway/transport・工程別予算0を照合した。
[実旧r4 Status](entry/windows_OLD_R4_PIN_STATUS.json) は元6da版を保持し、
[旧e917人工完了runのStatus/Resume](checks/OLD_PINNED_RESUME.json) も新しいglobal currentの下で元固定版を使い、追加起動0件だった。
operation gatewayはe917と95の該当3ファイルがbyte一致するため、本人所有PID597348を保持した。
毎要求fresh子processで選択した95を読む [互換性の記録](entry/OPERATION_GATEWAY_COMPATIBILITY.json) がある。

共有store `.rpe-central/price-validation/models.sqlite3` は13モデルの進行状態を保持する。
共有storeの13 record byte hashと旧e917/925 SERVICE hashは前後不変。
既存run/原結果/未確定job/STOPを保全し、切替直前にcurrent・controller/relayの所有を再照合した。
稼働科学controller/relayは0、旧常駐relayは停止、人工runは終了しており、他runやprocessを上書きしていない。
科学的証拠・価格資格・順位・fit・総コスト評価を変更しておらず、研究ロードマップの科学状態更新は不要。

検収・配置・current採用・公開の後はmandatory STOP。
次回は利用者の対象・範囲の明示指示後、Codexが§10.5の重複照合を行って通常Start/固定run Resumeへ進む。
