# 通常探索のCPU上限

currentは `romutail-pbs-current-95a74e448-cpu100-qg230`。controllerと科学workerは同じ固定SOURCEを使う。

| 環境 | RPEのCPU上限 | 運用 |
|---|---:|---|
| PBS | 100 | RPEのPBS割当要求を合計。現行の単一ノード割当は96 CPU |
| qg2 | 30 | プロジェクト用workerと子プロセスで論理CPU 0–29を共有 |
| Romulaptop / local0-wsl | 追加上限なし | 既存の受付OFF |
| RomuTail | 追加上限なし | 既存の自動資源会計 |
| qg1 | 未設定 | 停止中。接続せず |

これらはCPUの上限。並行タスク数は既存の自動会計と科学契約で決まる。
新規探索にはこのcurrentを使う。既存runのResumeは元の固定releaseに従うため、旧releaseのrunに今回のCPU設定は遡及しない。
この設定の対象はRPE通常探索。共有サーバーや同じユーザーの別作業には適用しない。

通常入口は [元のHANDOFF](../HANDOFF.md) のとおり。Windows、RomuTailとも新規Previewのrelease引数は不要。
新しいSERVICEは中央の
`.rpe-central/deployment/romutail-pbs-current-95a74e448-cpu100-qg230/SERVICE.json`。
公開設定には元manifestに加えたCPU値と配置スクリプトのSHA256、反映後のmanifest識別値を記録した。 [RELEASE_CPU_LIMITS.json](RELEASE_CPU_LIMITS.json)

qg2のrelayは配置版 `qg2-python.sh` を通して通常mailboxを呼ぶ。
`taskset`で指定したアフィニティを切り離したsupervisorと実行子プロセスが引き継ぐ。
profileの `slot_policy.resource_limits.cpu_count=30` を自動枠の資源会計も読む。
元のinterpreter、科学profile、worker spoolは同じ。
qg2実機で起動元と別セッションの子プロセスが論理CPU 0–29を使うことを確認した。

PBSのゲートウェイは既存専用Pythonで起動する。
新規qsubの直前にスケジューラ上のRPE allocationと、まだqstatに現れていない永続SPECのCPU要求を合計する。
CPU合計が100を超える新規投入は待機し、既存tokenや処理済み成果は従来のPBS経路で照合する。
現行PBSノードは96 CPU。同じ割当内のジョブ数やプロセス数を100で切らない。
PBS実機gatewayでsnapshotを通し、101 CPUをqsub前に拒否することを確認した。
人工競合テストで96 CPUと4 CPUの同時予約、超過待機、完了後の枠解放、同一tokenの再照会を確認した。 [check_budget.py](check_budget.py)

RomuTailとWindowsの明示版・指定なしPreviewは、同じrun ID `search-4ffb42344083eb979cda` と同じ新releaseを返した。
新規実信号・実compile・実Q3・価格fit・本番探索・PBS投入は各0件。
server1復帰後は元の復帰用パッケージで固定コードを準備し、新しいreleaseの `.rpe-central/fleet/romutail-pbs-current-95a74e448-cpu100-qg230/packages/server1.json` に保存したホスト配置設定を適用する。その後、既存の保存入力検査と人工参加検収を行う。
