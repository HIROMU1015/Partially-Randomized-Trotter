# PBSノードを先に確保して複数の探索runで使う

2026-11-21 JST。通常入口はRomuTailの `.rpe-central/SERVICE.json` の `schedule_releases` と `schedule_resumes` から、対象の運用gatewayと回収ランチャーを選ぶ。現行releaseは `romutail-scaled-fleet-373b52ae1b`。科学release、計算worker、run台帳、STOPは開始時の固定条件を使う。

## Codexが行う予約操作

RomuTailから次のシェル関数で操作する。下記の `reserve` はノードだけを予約する。通常Start／Resumeに適格なPBS仕事が届いたときにも同じ予約を解決する。

```bash
pbs_node() {
  ssh -o BatchMode=yes prism /home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python -B \
    /home/AbeHiromu/projects/rpe-pbs-20261118/runs/operational-launchers/pilot-b20b5c04bce5.pyz \
    --pool /home/AbeHiromu/projects/rpe-pbs-20261118/runs/shared-node --action "$1"
}
pbs_node status
```

- `pbs_node reserve` は1ノード、96 CPU、350 GiB、24時間をworkqへ要求する。同じ予約が待機中・稼働中なら同じ物理ジョブIDを返す。qsubの応答欠落時は保存tokenとPBS記録を照合し、未確定の予約を重ねない。
- `pbs_node status` の `status=Q` は順番待ち、`R` はノード確保。実計算は各runのattemptと実行receiptで確認する。待機中は他の適格ホストが仕事を進められる。確保後の待機時間は24時間に含まれる。
- 探索の通常Stopで閉じるのはそのrunのworker受付。待機中のrunは即時に切り離し、実行中のrunは元の仕事を排出してから次のrunに渡す。完了成果が転送待ちなら、元runが同じspoolから照合・回収する。古いrunの結果や未確定attemptを次のrunに移さない。
- ノード自体を返す指示を受けた場合に `pbs_node release` を使う。待機中はそのジョブを取り消し、稼働中は処理を排出して返す。予約状態が不明な間は同一tokenで照会する。PBSがworkerを異常終了させた場合は元runのreceiptを照合する。
- 非対応の旧固定gatewayは従来どおりrun単位のqsubを使う。共通予約がある間は、別の旧経路からPBS割当を重ねない。旧runの直接投入済みジョブはその元の所有で回収する。

## 24時間から72時間の共通予約への切替

2026-11-21 17:10 JSTに、旧 `267684.CPUcluster-PRISM` の後継 `267693.CPUcluster-PRISM` がworkqに入った。両方とも1ノード・96 CPU・350 GiB。投入後のPBS照会では旧がR、後継がQで、後継は空きノード待ちだった。

Codexは対象を旧物理ジョブIDで指定する。同じ指示の再送では同じ後継の投入記録を照合する。後継を投入したジョブ自身が切替を実行する。運用launcherのSHA-256は `4e45dd1ecfd3d20aa621058baba7911af64917c32030cbf5199035993b8359c2`。

```bash
ssh -o BatchMode=yes prism /home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python -B \
  /home/AbeHiromu/projects/rpe-pbs-20261118/runs/operational-launchers/pilot-4e45dd1ecfd3.pyz \
  --pool /home/AbeHiromu/projects/rpe-pbs-20261118/runs/shared-node \
  --action stage-72h --replace-job-id 267684.CPUcluster-PRISM
```

後継が待機中は旧予約に配送する。後継が実ノード上でPBSの割当を照合した時点で、旧側の新規受付を閉じ、旧workerが実行中の処理を終えてノードを返す。後継への新規セッションは、旧ジョブの終了を確認してから処理を開始する。PBSで予約が一時的に重なる間も、計算するノードは最大1台。旧attemptの取得・照合先は元runのspoolを使用する。

状態確認には `qstat -x -f 267684.CPUcluster-PRISM 267693.CPUcluster-PRISM` を使用する。後継のノード起動後は `runs/shared-node/allocations/48c932d5b0ef43c69f712aac11c0c274/HANDOVER.json` の `DRAINING_PREDECESSOR` が旧側の処理終了待ち、`ACTIVE` が切替完了。通常入口の既存runは従来の共通予約プロトコルで接続し、workerが実ノードの残り時間を読む。

旧予約の自然終了を待つ先行予約には `--action stage-after-exit --replace-job-id OLD_ID` を使う。現行の24時間または72時間の枠を識別し、同じ共通プールに72時間の後継を1件登録する。PBSの `afterany` 依存で旧予約が終了するまで後継を待機させる。後継ノード自身も旧予約の終了と所有を照合し、旧workerの制御と終了記録を保全したまま共通プールを引き継ぐ。

旧ノードを維持して空き次第切り替える場合は `--action stage-when-available --replace-job-id OLD_ID` を使う。PBSの空き待ちに1件登録し、先に作った旧終了待ちの予約がある場合は未起動・依存一致を照合して置き換える。後継が実ノードを取得すると新規接続先を切り替え、旧workerに割当済みの仕事は元のノードで終了・回収する。通常の中央管理は受付終了済みの旧セッションを検知し、後継用の別作業領域を開く。古い割当の再送を後継から除外し、通常の容量・入力・科学条件のチェックを通った新規仕事を送る。

## 実配置と確認範囲

RomuTailの配置記録は `.rpe-central/deployment/shared-pbs-pilot-20261121/`。PBSの物理予約の記録は `runs/shared-node/`、個々のworkerと受領待ちの成果は従来の `runs/<runの名前空間>/` に残る。通常入口の運用設定では、対応するreleaseの `worker_trial.pbs_dispatch_gateway` と `pbs_storage_launcher` の2組のパス・SHA-256を同時に選ぶ。展開先の照合値は[配置設定](DEPLOYMENT.json)。

| 確認 | 結果 |
| --- | --- |
| 先行予約 | 03:58:32 JST、PRISM workqに `267684.CPUcluster-PRISM` が1件登録された。 |
| 実PBS待機中のrun切替 | 通常入口の設定解決と実運用のPBS管理器を使い、人工run Aを接続・再送・停止、別名前空間の人工run Bを接続・再送・停止。その後の予約照会・再送でも同じ物理IDとqtimeを確認。追加qsub 0件。 |
| ノード稼働後の切替 | 模擬PBSと人工workerプロセスで、Aの終了後にBが同じ物理ID・残りwalltimeを使うことと、両runに同じattempt名がある場合も個別の元成果を回収することを確認。 |
| 回帰 | PBS予約・workerの50テストと3サブテスト通過。 |
| 科学計算 | この配置検査の新規実信号・compile・Q3・本番探索は各0件。 |

実機での今回の検査範囲はPBS待機列とrunの接続・停止まで。次の通常探索で、確保後の実PBSノード上の開始・回収を元runのattemptと照合する。
