# 人工回帰：既存360＋新77＝437 PASS

[全test ID・SOURCE/log SHA](artificial_test_results.json)。**FAIL/ERROR/SKIP各0、forbidden science imports各suite0**。
新規SOURCE・tests以外を変更せず、8独立suiteを並列processで再実行した。最後の費用CSV変更後は影響する新suite77件を再実行し、既存360と分けて集計した。

| Suite | PASS |
|---|---:|
| Admission | 69 |
| Readiness | 30 |
| Driver | 36 |
| Cost preparation | 48 |
| F01 | 26 |
| Host integration | 66 |
| 既存offline verifier | 85 |
| 新parallel independent verifier | 77 |

新規は8pair×3phase正常、worker完了順変更、ID・counter・state/file長・code・offset・budget・schema不一致、重複予約/完了、STOP後開始、barrier、ACK未確認、CPU/partition二重計上、missing summary/receipt、path衝突、親/worker crash、複数failure、F01 emit内partial/flush/file hashを検査。
公開archiveの取得bytes/SHAと展開rawを確認し、18ケースの実読取りを試験。独立fixtureは元のStream/run_pairをfake modules/clockで使い、832 toy trialを生成する。
SQLite/Admission回帰はprivate人工namespaceのみ。実authority/claimを使わず、実scientific moduleや保存科学値の数値解析へ進まない。

[parallel_bounded_memory.json](parallel_bounded_memory.json)：state file 33,024bytesでPython allocation peak17,892bytes、3,277,056bytesで144,926bytes。
stateは64KiB blockで一回走査・SHA更新し、全state/trialをRAMへ保持しない。約100倍のfileでも最大blockに応じた有限buffer内で処理した。
これはtracemallocで追跡する当該走査のPython allocationであり、process RSS/OS cacheや64GiBの実測上限ではない。
旧pair streamingの再実行結果も[bounded_memory.json](bounded_memory.json)へ保存した。

Linux/Python3.10.12、32 logical CPU、RAM62GiB/available約49GiB。詳細は[environment.json](environment.json)。共通§8.4を適用し、人工CPU/RAM/worker/wall上限や共有設定変更なし。
log gzipはraw stdout/stderrのbytesを保持し、machine JSONは解凍後log SHAを記録する。局所人工検査でありimmutable CI/外部再現/実storage耐電源断の証明ではない。
