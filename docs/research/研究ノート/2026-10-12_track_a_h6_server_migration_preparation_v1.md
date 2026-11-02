# 2026-10-12 H6共用サーバー移行準備

利用者はH4を継続したままH6専用環境・worktreeを準備し、旧8 GiBでSTOPしたB1 S4 q8だけをAS32 GiBで一度実行する計画の作成を指示した。
科学計算は別の明示認可まで開始しない。

固定科学source5a903aea・公開証拠054bb7b2から専用worktreeへ移行。旧224 source/33入力/5220 rawをbytes/SHA/Git blobとreuse契約で検査し、449 group/902 wrapper/92 signal/84 B3を受理した。
未完了は447および450–591の143 group。Phase Aは447 `B1_p19_q8_4th` replica0だけ、Phase Bは残り142として切り離した。
科学plan/native lowering/full-wrapper/compiler/seed/shot/精度規則は変更しない。

H4 Python3.12.3を流用せず、home配下にCPython3.11.0rc1と専用venvを再現した。固定package/integral-provider/compiler gateは一致、依存wheel/native/source全fileを新profileへ固定。
NumPy同梱pycのRECORD差は対応sourceを専用rc1でcompileしたbytecodeとの一致を確認した。
旧native binary inventoryがないためbinary bit同一性・実H6 compiled output完全一致は未確認として残した。

新resource実装はworker32 GiB、coordinator soft2/hard32 GiB、RSS40 GiB、job RAM予算64 GiB、CPU20–23、thread1。
job RAMは予約を意味せず、正式なサーバー運用規則と実行認可が残る。
workerとは別processのcoordinator監視、parent-death guard、identity付きown group停止/wait/reap、one-shot、部分出力保全を実装した。
38 control/mock/minimal-process検査と18旧人工wrapper/compiler検査、計56 distinct local PASS。実H6 scientific runner0。

SOURCE `61cadec5b5b2c8f1138828f43f7780d36319ccc3`を公開・remote/tree照合した後、利用者が「プッシュはこちらでするのでしなくてよい」と指示した。
以降のREVIEW_BUNDLEはローカルcommitのみとし、push commandを渡す。旧H4 worktree/venv/processと無関係dirty/untrackedは保全した。

最新の仕様・証拠・未実行command・認可待ちは[移行準備契約](../track_a_h6_server_migration_preparation_v1.md)と[bundle入口](../../../artifacts/resource_applicability/track_a_h6_server_preparation_v1/2026-10-11/README.md)へ集約した。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION` / next_stage_authorized=false。Phase A成功/失敗後はmandatory STOP、Phase B/確認32/H8へ自動進行しない。
