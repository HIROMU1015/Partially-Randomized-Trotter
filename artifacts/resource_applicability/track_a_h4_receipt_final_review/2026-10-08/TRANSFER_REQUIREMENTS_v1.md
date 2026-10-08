# 凍結byte受領の要件（科学array読込なし）

旧handoffのfile一覧とSHAを独立照合する。旧handoffに個別byte数はないため、転送元でstreaming hashとstat byte数をmanifestへ捕捉する。未知のbyte数を一致済みと報告しない。

| file | 期待SHA-256 | 個別期待bytes |
|---|---|---|
| input-0.70.npz | `fc0cc9676ee576d7b4e150ab555acba798ffb8659b98f0a8165118e4e59e52b9` | 未提供 |
| input-0.80.npz | `25d8d222a2798b9986926cabeaeddb963bf476db92991cfd4f10f6e454ddcacb` | 未提供 |
| input-0.90.npz | `06672860ed8f9298053b4d84693fbab6d13e38c3166c27dbabcc161f33673c3d` | 未提供 |
| input-1.10.npz | `ed83af66303cdcb6847be1b89f77a86193c725e942ef4d0d717f0cd5b7e714dc` | 未提供 |
| input-1.40.npz | `50a9cece3c2d14b830233c571b543e916126efe581593c194fd5649ee638ec22` | 未提供 |
| input-1.60.npz | `504c7647e35fe17359b6545078e30b85fee6795316081b52f28c426c4490e1e8` | 未提供 |
| generation-freeze.json | `75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69` | 未提供 |
| byte-budget.journal | `6b68368565cc336d283bc094f844a37ceb1966838ca5482f1f12347d0c5d669e` | 未提供 |
| runner.log | `7108181a3b295ca92d4a73de7e7420d160898270d45a95657a34e7c6526947b1` | 未提供 |

元control log `runner_stdout_stderr.log` はreceipt内で `stop/runner.log` として扱う。元fileをrenameしない。
必要native controlはrun05のprocess/startup/stop/exit記録。旧runのledger deltasは消費予約の証拠として保持し、新結果/cacheへ再利用しない。
旧hostのPIDを新hostでkillしない。handoff metadataのall_owned_ended=trueだけでnative停止proof合格にしない。

手順：旧serverの既存ログイン済みターミナルでbyte-only packetを作成 → 既存SCP/SFTPでpacketとSHA256SUMSを新hostの専用incomingへ一度copy → source manifest bytes/SHA、9 known SHA、native controlを確認。

接続先と実行可能な1-command producer/SCP手順は、local-onlyの `/home/AbeHiromu/projects/h4-handoff-evidence/20261008/receipt-final-review/USER_TRANSFER_STEPS.md` に固定。認証設定/鍵登録は行わない。
新受領先 `/home/AbeHiromu/projects/h4-handoff-evidence/20261008/receipt-final-review/incoming/`。安全な受領はregular filesのみ、relative path/重複/リンク/escapeを拒否し、home配下の新exclusive directoryへ64KiB単位でcopy/hash。

raw packet・NPZ・runtime/control・credentials・内部SSH情報はGitへ収録しない。
