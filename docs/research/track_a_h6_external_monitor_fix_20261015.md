# H6 pilot監視修正：長いC/GIL処理中の独立資源観測

2026-10-15。利用者の「本計算をもう一度行って　計算が完了しなかった場合は修正を行って」に従い、run02停止後に必要な監視修正と新run03を実施。サブエージェント0、同担当の限定差分確認であり、新しい独立reviewとは扱わない。

SOURCE `b73580dd7402df5d8ac0a95789ec351c05209049`、[source closure・test・新grantの入口](../../artifacts/resource_applicability/track_a_h6_external_monitor_fix_v1/2026-10-15/README.md)。旧SOURCE `6b4d3026bf6c30bea3d74e7ccaeb06bda0e9baaf`、[run02停止証跡](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_run02_v1/2026-10-15/README.md)。

run02はPhase Aの3準備identityが一致し、Phase Bのp5q1/p10q2 correctness誤差は各約1.1e-14で合格した。p15q8の検証中、直前観測から5秒超のcallback遅延によりOBSERVATION_STALEでfail-closed。native wait2/recorder0/driver元identity不存在、compile0、launcher wall453.47913644835353秒。snapshot1/overlay3/native3/wrapper6/probe copy4/control probe10は呼出前予約として保持し、失敗した操作の完了数を予約から推測しない。

従来Python signal/tickは長いC処理等でコールバックが戻らず、復帰後にstale STOPする仕様だった。タイムアウトを延ばさず、既存metadata launcherへstdlibの独立観測を追加した。追加process/本番role0、CPU数・numeric worker最大6・内部thread1・launcher予算・科学/compiler/入力を維持。

新module `scripts/h6_external_resource_monitor_v1.py` が1秒ごとにdriverと登録compile childのidentity、親子所有、CPU配置/thread、RSS/HWM/AS/CPU、host/cgroup PSI/OOM、空きmemory、filesystem/inode/quotaを検査する。各観測の開始間隔・所要・sequence・phase・first STOPをprivate controlのbounded journal/atomic heartbeatへ保存。heartbeatはSOURCE/contract/auth/launcher/driver identityへ結合し、5秒欠測・非finite・逆sequence・run/owner不一致・過去の5秒超gap/IO時間も拒否する。

従来のdriver/child内の資源checksとOS AS/CPU上限を残し、ローカルcallbackが遅れた場合は継続した独立観測のproofを必須にする。長いcallback間隔と実際の観測間隔を別fieldで保存し、単にstaleness判定を削除しない。observerの最初の停止理由を保持し、既存launcherのowned pidfd TERM→wait5→KILL→wait5、compile childのparent-death/cleanupを維持。共有環境・他jobは操作しない。

childは親がatomic公開したowner/割当CPU登録と、自分の実観測を受けるまでcompileしない。fork直後のdriver CPU継承は最大5秒だけbootstrapとして扱い、その間はleaseを発行しない。一度worker CPUを確認した後の変更は即STOP。未完了JSONの公開競合を避け、observerはincremental artifactを直接読まずatomicな別登録を読む。observer journalとlauncher stdoutの合計を従来log32MiB以内へ制限する。

限定mockは新14＋parallel24＋runner23＋既存の別担当作成23＝84件PASS。AS512MiB/CPU120秒/wall180秒、実child0/science import0/transpile0。初期82件も保存。追加のGIL実動作fixtureは最大2process/thread1、child AS512MiB/RSS256MiB/CPU60秒/wall60秒/output2MiBを事前記録。32M-round OpenSSL PBKDF2の人工C処理だけをPyDLLでGIL保持し、約12.21秒のcallback gap中も独立観測48件・最大間隔約.259秒・最大IO約.008秒で継続、native exit0/wait/reap/元identity不存在を確認。科学array/build/probe/compile0。最初のfixtureで見つかったaffinity起動競合の失敗も保存し、ready handshakeとworker bootstrapを補正。最終source差分後の実動作再検査も合格した。

SOURCEはruntime4変更＋新test/harness3を固定し、runtime code12＋既存identity metadata1のclosure13件を再結合。現在profileの準備identity参照、science provider230、profile34273files、入力5件は変更なし。旧run01/run02・診断のknown/unknown/carryを維持し返却/reset0。

run03は新しい未使用grant/output/controlへ結合し、短いauthor binding確認とfresh source/profile/input/carry/CPU/memory/PSI/OOM/fs/inode/quota gate後に一度起動。今回の指示は失敗後の必要修正と新grant再実行を認可する根拠とし、runner内部のautomatic_retry=falseと次stage未認可は維持する。実結果・終了監査はrun03資料へ別保存する。
