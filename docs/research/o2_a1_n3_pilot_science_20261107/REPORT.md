# A1対B1-OL：752-shot予備検証の実行結果

2026-11-07 JST、SERVER1 GPUA100-gleap-01。
**一回起動したが、最初のtrialより前の資源判定で正規STOP。取得0/752件、A1対B1-OLの有望性は未判定。科学retryはしていない。**

## 固定対象と実行SOURCE

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol0、X/Y。N1 actual finite-grid Q・selected full policy、N2保証ground、認証済みexact Fraction engine、3-control/全5正層を維持した。

準備版36ee9c6244eeedc875b5ddfafffb4c4cf76c8b8bの条件をユーザーが採択・実行承認した。[実行仕様](EXECUTION.md)に対応する科学SOURCEは [a139d7dee53cab5f4161a158d644faee8c033fc4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a139d7dee53cab5f4161a158d644faee8c033fc4)。実入力前に17公開ファイルの本文/blob、tree `25bc854f4efe59a6ec35f49e3c124a019c316a26`、remote SHA、依存/native profileと保存入力identityを照合した。新activationのみを追加し、旧kernel/統計/乱数/測定・価格SOURCEは変更していない。

人工[44件PASS](ARTIFICIAL_TESTS.txt)。新activation10、準備済みdriver/design23、旧campaign11。これは実科学のCV改善を示す結果ではない。

## 起動・停止の証拠

consume-once cronからqualified managerを一回起動。triggerは起動前に除去され、親/managerは`/system.slice/cron.service`で動作した。自動再起動なし。

| JST | 保存された処理 |
|---|---|
| 00:55:09.644 | managerが親PID3281254を開始 |
| 00:55:17.386 | 親Runtime開始 |
| 00:55:21.113 | `no observed idle CPU for noninterfering execution`でSTOP要求を耐久保存 |
| 00:55:21.828 | 親exit1をmanagerが回収。signalなし |
| 00:55:21.913 | outerがmanager exit1を回収 |

既存`available_workers()`は、`/proc/stat`の0.25秒差分のidle+iowait比率とCPU affinity数からworker数をfloorで求め、1未満で例外を返す。今回この条件で停止した。恣意的なworker上限を新設した結果ではない。
**その差分tick自体は旧関数が保存しないため、当該比率・個別CPUの空き・継続的な不足はUNKNOWN**。短い一標本だけでサーバーの恒常的な不足を断定しない。managerの保存cgroup記録にOOM killは見られず、exit1は捕捉された資源例外による終了である。後刻の高いCPU pressure/loadは[別時点の観測](POST_STOP_OBSERVATION.json)として区別する。

STOP保存→既存reconcile/classify→親/manager終了を確認。finalization_errorsなし。workerは一件も作成されていない。cleanup記録は`IDENTITY_GONE_OR_CHANGED`で、追加signalは送られていない。親は消失済み、own crontabは空。複数のOBS `PARENT_START`はmanager/outerがRuntimeを作った記録であり、新しい科学親やretryの起動ではない。

## 科学call・取得・費用

| 項目 | 結果 |
|---|---:|
| 保存bundle decode | 1 |
| stream book | seed20261108で32domain/64子streamを初期化 |
| 53-bit wordの取得 | 0 |
| trial計画・path固定・worker開始 | 0 |
| 測定bit・Born・gate作用・compile | 0 |
| train/probe/evaluation | 0/0/0 |
| 実現compiled RZ | 0 |
| N1/N2/engine認証の再計算 | 0 |
| retry | 0 |

量子shot費用0に対し、古典setup費用は生じている。保存bundle読込3.464秒、親のapplication区間3.735秒、親peak RSS **506,425,344 bytes（約483 MiB）**。managerによる親生成からwait回収までは約12.18秒で、Python起動/静的照合も含む。旧campaignの費用は今回へ混入せず、そのまま保全した。

| 比較 | X/Yの評価 | 分散・RZ傾向 |
|---|---|---|
| A1 | 0件、fit/rank/range未評価 | UNKNOWN |
| B1-OL | 0件、probe/配分未評価 | UNKNOWN |

未完了trialや価格欠測は0件だが、比較自体が未実施なのでCV改善、精度、期待費用改善を主張できない。STREAMS.jsonは生成器とspawn方針の記録で、word/測定が消費された証拠ではない。

## 成果物・保全・次の条件

[科学run原本](../../../artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run01/)と[manager原本](../../../artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_manager_run01/)の全28ファイルを公開する。[RAW_MANIFEST.json](RAW_MANIFEST.json)で全byte hashを固定し、SOURCE、stream、STOP、全call会計、attempt、親/manager/cgroupログと空のログも保存した。[RESULT_SUMMARY.json](RESULT_SUMMARY.json)は取得・未評価事項の機械可読な要約。

元mainのHEAD/dirty/untrackedと3,087原本、旧停止済み実行の1,086ファイル、旧SOURCE/入力/STOP、C1凍結を保全した。旧397件・新B1339件のbit/係数/乱数/cacheを新pilotへ読み込んでいない。

次回は実際の空き資源を確認してから、**新しい一回限定attemptの明示承認**が必要。0.25秒のadmission記録不足を改善する場合は、科学SOURCEを保全した別差分でraw tick/affinity/判定値を保存し、人工テストと公開固定を行う。資源待機を導入するなら、実入力・stream初期化より前の有限待機として次契約へ明記する。今回は待機/retry/条件変更を実施しない。word未消費でも既存SOURCE/STOPを消して再投入してはならない。

公開remote SHA・本文/blob/tree照合後mandatory STOP。追加測定・保証付き本検証へ進まない。
