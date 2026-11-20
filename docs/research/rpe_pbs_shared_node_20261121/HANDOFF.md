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
