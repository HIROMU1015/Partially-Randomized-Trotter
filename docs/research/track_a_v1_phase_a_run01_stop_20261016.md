# Track A V1 Phase A run01：科学予約前のlauncher STOP

利用者の「V1 Phase A を開始して」に基づき、SOURCE
`69e9914345ab29876b8b2e19b0411a28bae3b208` のone-shot launcherを一度実行した。
起動前gateは通過したが、最初の監視で `OBSERVER_RSS_CEILING` を検出しSTOPした。
追加signalは未取得。driverはTERM/native exit -15で回収され、科学lock・output・予約は作成されていない。
**自動retry/resumeは行わず、mandatory STOPを維持する。V2・compile・次Phaseへ進まない。**

[終了receipt・保全・診断](../../artifacts/resource_applicability/track_a_v1_phase_a_run01/2026-10-16/README.md)。
これは[並列実行準備](track_a_v1_phase_a_parallel_readiness_20261016.md)に続く実行記録であり、signal検証の成功結果ではない。

## 承認範囲とfresh gate

対象はlinear H6、1 Å、STO-3G、DF rank19、12 modes/sector400、T=.8。
新104候補（B0 52＋B0+I解析overlay52）、PR prefix3/4のouter-S2診断8件。
SCF/DF/state solver・trajectory・native full wrapper・compile・費用計算は対象外。

SOURCE15・provider230・runtime profile34,273 file・入力5件のbytes/SHAが一致。
既存H7処理のCPU56/63を避け、5秒負荷観測後、driver4、observer12、
worker `[13,14,15,36,37,38,44,45,47,49,62,18,2,10,9,23]` をsealした。
最大16worker・各thread1、driver/worker AS32/RSS9 GiB、observer各512 MiB、
aggregate RSS160 GiB・CPU96h・wall12h・output2 GiB・stage保存4 GiBは承認範囲から変更していない。
有効memory・PSI/OOM・FS/inode gateを確認。quotaの独立証明は未取得として区別した。
metadata gateは8.412665秒。既存draftを残して新grant01を発行し、上書き・reset・refundしていない。

## 終了・保全

launcher監視区間0.027043秒、最大観測間隔0.006079秒。
最初のRSS判定でSTOPしたため、観測JSONLは空でaggregate peakの保存値0は実測peakを意味しない。
`all_reaped=true`、cleanup error0、driver/observerの退出を確認した。
科学lock/output/予約なし。新signal/reference/primitive probe/native full wrapper/compile/sampling/SCF/DF/state solverは0。

実行前に記録したH5/H6/H8の既存grant・ledger・carry・one-shot関連644ファイルが全て不変。
SOURCE15と入力5件も終了後に再照合した。H5・H7・V2・既存結果・共有venvに変更やsignalを加えていない。
launcher試行1・新grant発行1を記録し、科学予約0を再実行許可へ読み替えない。

## 停止原因について確認できたこと

本番のobserver RSS実測値は、例外判定より前に永続化されていなかったため**未取得**。
512 MiB上限を検出したことはterminalに残るが、真のメモリ増大か計測上の問題かは未確定。

科学import・H6 array演算なしのpinned Python診断では、
config読取後のru_maxrssは41,484 KiBで、NumPy未import。
同じobserver関数を人工stdlib childに接続した診断はexit0・全回収、
observer peak counter50 MiB/current RSS35,016 KiBで通過し、本番STOPを再現しなかった。
metadata/hash確認だけの診断では、exec前後のru_maxrss49,664 KiBに対しexec後のVmHWM/RSSは10,240 KiBだった。
exec以前のpeakがru_maxrssに残る性質は確認したが、今回の512 MiB判定の原因と断定しない。

本番再実行・上限増量・判定の迂回は0。
## 再実行せず準備した監視修正

[launcher](../../scripts/launch_track_a_v1_phase_a.py)にobserver現在RSS・VmHWM・ru_maxrss・ASの取得を追加し、
停止判定時の値をterminalへ保存する。RSS上限は現在のexecutableのVmHWMで確認し、exec前のlifetime peakは別fieldに保持する。
AS/RSS各512 MiB、全体RSS160 GiB等の上限値は一切増量していない。
真のcurrent-exec peak超過・AS超過は引き続きSTOPし、値の未取得や不整合も成功扱いしない。

旧44＋新3＝47限定人工tests PASS（12.555秒）。追加3件はlifetime/current-exec peakの区別、
実peak/ASの上限超過拒否、RSS STOP時の実測値保存とchild回収を確認した。
変更した並列・監視テスト17件をpinned Pythonでも実行してPASS（8.635秒）。科学入力・本番grantは使用していない。
これは技術修正の人工検証であり、run01原因の証明や実H6完走の認定ではない。

別private領域に修正SOURCE hashを固定した[未承認configのreceipt](../../artifacts/resource_applicability/track_a_v1_phase_a_run01/2026-10-16/observer_fix_preparation_v1.json)を作成した。
旧config/grant/controlは保全し、再実行用の新grant・CPU割当・command・本番起動は0。
旧失敗記録・grantを保持し、新しいone-shotの扱いはこの報告後に判断する。
主担当の技術検査であり独立reviewではない。subagents0、push0。

## 後続run02の記録

利用者の別の再実行承認により、[run02](track_a_v1_phase_a_run02_results_20261016.md)がSOURCE26b9fde6…で完了。新しいgrant/outputを使用し、本run01のgrant/controlは保全した。run02ではlifetime peak647,266,304 bytesとcurrent-exec peak52,953,088 bytesの違いを本番で確認した。run01の未記録値を後から補ったり、その個別原因を断定したりはしない。上記の未承認草案・再実行0の記述はrun01報告時点の履歴である。
