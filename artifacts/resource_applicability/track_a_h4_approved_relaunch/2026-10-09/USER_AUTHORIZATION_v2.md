# H4再実行の利用者認可（2026-10-09 JST）

最新の利用者指示（原文）:
「これについては問題ないので再実行して」

対象の引用: 「再実行予算12.26GiB > 承認10GiB」。
直前に固定した再実行案は累積output charge上限10GiB→13GiBと、新SOURCEで一度のH4 signal/compile map再実行。
この指示を、その13GiB案と一度の再実行の承認として適用する。利用者にも適用額13GiBを説明済み。
これは過去の消費返却/上限撤廃ではなく、累積charge capだけの明示改定。
carry20 actual /4428938712 bytes /5472.345380863175秒を保持する。
以前承認済みのprivate candidate environment/compiler、worker12 CPUs[2,4,5,6,8,9,10,11,12,13,14,15]、
driver16/observer18、own-run affinity・内部thread1、observerAS256MiB/RSS64MiB、admission120.25GiBとactual74804を保持。
旧compiler output完全同一性未検証は記録に保持する。
72h・各driver/workerAS/RSS8GiB・headroom16GiB・monitor5秒・科学条件・compiler optionsは変更しない。
mapを一度実行し、完了またはfail-closed STOP後に終了する。自動retry/入力再生成/旧partial/cache再利用/次stage/GPUなし。
SOURCE・profile・input・carry・fresh CPU/memory/PSI/OOM/filesystem/block/inode/quota・unused output/controlとone-shotを検査。
独立最終review・artifact固定・fresh gatesに合格すれば、追加利用者承認待ちで停止せず一度起動する。
共有環境・既存venv・他jobを変更しない。新しい作業・証跡・logはhome配下。
