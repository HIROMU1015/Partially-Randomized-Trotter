# server1の分散探索への復帰

2026-11-21 JST。RomuTailから既存の `qg1` 経路で `GPUA100-gleap-01` に接続し、踏み台に登録済みのサーバー鍵と照合した。サーバー1の専用Git領域に不足分を配置し、固定コードは現行 `romutail-scaled-fleet-373b52ae1b` と継続用 `romutail-h4-rerun4-1355ad65fa` のそれぞれのmanifestで確認した。科学Q3の `28d976b99` と固定Python 3.11.0rc1、既存のQiskit・OpenFermion環境を再利用した。PF／PR-DF信号は条件を満たす既存Python 3.12を使い、PR-DF価格用の固定入力は原REQUESTのハッシュで照合した。

- RomuTailの `.rpe-central/deployment/server1-participation-20261121/` に配置・検収記録を保存した。`PARTICIPATION.json` を受付ONにして、`SERVICE.json` の現行Startと既存の2つのResume向け運用設定からサーバー1を選ぶ。通常Start／Resumeは利用者の指示を受けてCodexが実行する。
- 通常のrelayはサーバー1から空き枠を確認し、各REQUESTのSOURCE・入力・runtime照合を通った仕事を配る。空き枠はautoで初期1枠。受付を一時停止する場合は `PARTICIPATION.json` の `accepting_new_jobs` を `false` にする。元のattemptの回収は続く。既存のPBS共通割当と、他ホストの配分を維持した。
- 人工runでは信号4、compile 1、監査1、Q3 8の計14件をサーバー1で実行し、RomuTailの予約・受領・採用はいずれも各1回だった。保存実REQUESTでは準備12件とQ3の4ルートを実機で照合した。Q3の不足Gitオブジェクトは通常の配布処理で補い、保存入力のdry-checkを通過した。
- 通常Previewは現行releaseを解決し、両固定版の実SSH mailboxで受付ON・空き1枠を確認した。作業時点の実runは停止状態で、新規の実H4は0件。次回の認可済み探索で、サーバー1への実タスク割当と中央採用を確認する。
