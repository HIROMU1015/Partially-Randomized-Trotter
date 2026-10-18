# R1互換性・F01・STOP

Aは科学・保存・計測・逐次STOPの意味を維持し得る候補、Bは契約変更が必要な候補、Cはそのままでは不適格。Aでも現行SOURCE pinへ無断反映せず、別途採用判断・新SOURCE固定が必要。人工PASSは実F3全domainでの統合認定ではない。

| Candidate | 分類 | 根拠・残る条件 |
|---|---|---|
| 冪の漸化式 | A候補 | exact partial/remainder/端点一致、Jと丸め・guard・再queryは不変にする |
| GMP rational adapter | A候補 | Fractionへ戻す型境界を維持。実F3全体は未接続・未検証 |
| fused/native canonical | A候補 | Unicode/surrogate/DEL fallback、元bytesとF01を人工確認 |
| incremental SHA更新 | A候補 | 同じUTF-8列とchain順序。転送・preencode条件を実装時に維持 |
| 8 pair process並列 | B | fixed order/比較競合/最初の失敗時の逐次prefixが変わる |
| pass並列・worker reuse | B | pass順序、setup/HWM、CPU観測の比較条件が変わる |
| buffering | B | emit完了とOS write受理が一致しなくなる |
| function profiling省略 | B | 研究で求めるfunction費用内訳が失われる |
| float置換・余剰sampling無課金・failureをZERO化 | C | 厳密性/予算/STOP違反。実装・採択していない |

## 人工failure injection

全workerのfirst sampleをbarrierでそろえ、最初のpairのX trial 3で人工失敗。共有lock下のtrial開始gateが失敗時刻を記録し、その後の新規開始を拒否。未開始pairはdispatchしない。開始済みtrialは安全に完了・prefix保存し、次のgateで停止する。強制killや全process走査を使わない。

| Workers | 通知後の新規開始 | 通知後の完了trial | 通知→全join median ms | 未開始pair数 |
|---|---|---|---|---|
| 1 | 0–0 | 0–0 | 2.85 | 15–15 |
| 2 | 0–0 | 1–1 | 3.49 | 14–14 |
| 4 | 0–0 | 2–3 | 4.41 | 12–12 |
| 8 | 0–0 | 7–7 | 4.93 | 8–8 |

「開始」はlock下でtrialをadmitした時点。通知から全joinまではcooperative終了とIPC/cleanupを含み、物理的な即時停止ではない。失敗通知後0新規開始を保証しても、開始済みの複数trialは存在する。失敗を最初に観測・通知した時刻が基準であり、未検知の失敗発生を先取りできない。従って逐次R1と完全同義ではなく、成功時のtrace一致だけでAに分類しない。

実科学契約に採用するなら、すべての開始済みcall（失敗・取消・後で破棄するものも含む）を予算へ課金し、atomic reservation、最大in-flight、失敗の順序/採用prefix、既存grant/claimとのbindingを新契約で定義する必要がある。今回のControlは人工start gateのみで、実科学call budget reservationやadmissionを提供しない。

F01は失敗sampleのresult/eventを前trialから流用しない。completed prefixはemit正常return後だけ更新し、streamの完全行prefixとtrial completionを分離する。Python/native unbufferedそれぞれで元F01全26 testsを維持。partial write、file hash、flush、encoder失敗を含む。

人工os._exit(17)では、unbufferedはemit受理2行/496 bytesを再読でき、64KiB bufferedは受理2行を報告していても再読0行/0 bytesだった。電源断耐性の証明ではなく、bufferingによる保証差の具体例である。どちらもretry/resumeなし。同一run directory再使用は拒否。
