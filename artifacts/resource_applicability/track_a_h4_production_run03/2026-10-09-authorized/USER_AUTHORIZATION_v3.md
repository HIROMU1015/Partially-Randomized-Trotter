# H4 run03 累積上限と一度のmap起動認可（2026-10-09 JST）

利用者の最新指示（原文）:
「上限の変更を承認するので本計算に入って」

直前に提示済みの累積output charge13→17GiB（18253611008B）、累積actual74804→74805の2件と、
新SOURCE6e68fd9bcc68e788db6f5d43eaa6a03866e53d3bによる一度のH4 signal/compile map起動を明示承認として適用する。
carry21 actual消費/予約・8692723164B・5766.582514658794秒 upperを保持し、失敗返却/resetなし。
worker12 CPU[2,4,5,6,8,9,10,11,12,13,14,15]、driver16、observer18、own-run限定affinity、内部thread1、
既存private venv不変のenvironment/compiler採用、observer AS256MiB/RSS64MiB、admission120.25GiBは過去の認可を引き継ぐ。
旧compiler output完全同一性と元run02科学compiler/IPC原因は未検証。追加人工compileは利用者指示で省略する。
科学条件/compiler options/72h/driver-worker各AS-RSS8GiB/headroom16GiB/monitor5秒は維持する。
SOURCE/profile/input/carry/plan/auth/reviewを結合し、独立最終review・immutable artifact固定・直前fresh資源/容量/quota、
未使用output/control/one-shot検査が合格したら、追加利用者承認待ちで止めずそのまま一度起動する。
完了またはfail-closed STOP後に停止、自動retry/次stage/入力再生成/旧partial-cache混合/GPUなし。
共有環境/既存venv/他jobを変更せず、新作業/log/tempはhome内だけ。実科学arrayはこの認可済み本計算内だけで読み込む。
