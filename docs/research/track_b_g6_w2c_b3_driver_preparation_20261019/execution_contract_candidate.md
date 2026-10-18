# B3実行契約候補（未採択・未認可）

[JSON正本](execution_contract_candidate.json)／[schedule](comparison_schedule.json)／[angle等](angle_cache_output_contract.md)。
`adopted=false`、`contract_grants_execution=false`。今回のscopeはdriver準備とmockまで。
科学SOURCEは固定6件＋既存helper＋新driver＋package initを完全SHA256で結ぶ。
基点`c8cbeb18f76aef2cb1329d8c810036372fa07cc0`と各取得元commitは別項目。新payloadの完全commitはpublication receiptで固定し、自己hash循環を作らない。

## 固定科学call予算

| arm | 保存N/axis | 各layer・passのX/Y合計 |
|---|---:|---:|
| Ordinary | 1,174,526 | 2,349,052 |
| Partial Return+tail | 894,664 | 1,789,328 |
| Closed P3+tail | 880,312 | 1,760,624 |
| Full Return | 1,014,322 | 2,028,644 |

4 arm×A/B×X/Y=16 cell。初回correctnessは各16 trial、計256。
baseline full-Nは15,855,296 sample、instrumented同数。**合計31,710,848 sample**を一束one-shot候補とする。
これはcall予算の整数bookkeepingであり、保存科学値の新規再解析／N再計算ではない。
2passはobserverの影響を同じ全N workloadで記述するための追加再生で、隠れた無料runではない。

constructor24、budget再評価0、保存budget file load1、Full F3 prepare3（constructor込み）、canonical preflight9。
acceptedごとのB angle・event再queryは各1。canonical Bは各arm `4*N_a+32`、Full Bは実accepted数（上限`4*N_F+32`）。
Full packetは各raw-valid trialで1（coin rejectも含む）＋B accepted angle時の原F3再query。
それ以外のfrozen内部callは元path通りに残し、実回数を記録する。新回路/compile/synthesis、B1/B2再取得は0。

## 受入と停止

全16 cellのcorrectness16 trialを先行し、各cellでacceptedを少なくとも1件確認、
event phase/order/πW、B preflight/source/state、要求したangleのradiusチェックを通す。
A/Bのexact event traceを照合する。raw zero/coin rejectがこの有限prefixで出るとは保証せず、
人工fixtureのpath coverageと実prefixのobserved coverageを区別する。不足を埋める追加trialは禁止。
acceptedなし／precision不足／source不一致／技術例外なら全束STOP、profileへ進まない。

full-NではR+C+A=N、各axis保存N厳守、A/Bと両passのtrace一致、correctness prefix一致、
coarse/fineそれぞれの排他合計<=outer total、output chain/file hashと完了を要求する。
時間同値やCPU勝利、実測分散、新confidenceは受入条件にしない。
失敗をzero扱いせず、prefix/resource/failureを保存しretry0/resumeなし。完了／失敗ともmandatory STOP。

## 採択・別途承認の境界

未決はτ/Kφ/J（math-only）、新seed規則、setup/cache/保持/2pass schedule、公開payload完全SHA、
新しいatomic one-shot admission port／claim binding。CPU/RAM/worker/wallの人工上限は設定しない。
外部portはユーザーの別途明示承認を採択契約digestへ結ぶ仕組みで、候補やCLIがauthorityを発行するものではない。
現releaseではport未提供・候補未採択のため、実G7 import/constructor前にfail closedする。

次のユーザー承認は、(1)具体的τ/Kφ/Jと上記候補規則の採択、(2)新claimと固定SOURCE/input/call予算を名指した
初回correctness＋固定全N二pass profilingの一束one-shot実行、を区別する。
承認後Codexが別の正式契約を固定し、外部admissionへ結び、承認されたbundleだけ実行してSTOPする。
新しい科学的誤差allocationが必要なら本候補では実行せず、その判断を先に返す。
旧128診断や旧claimは新全Nの承認／IID保証にならない。

今回実G7 correctness・sampling・CPU profiling・保存値数値再解析0。C1/C2、G7/G9監査には進まない。
