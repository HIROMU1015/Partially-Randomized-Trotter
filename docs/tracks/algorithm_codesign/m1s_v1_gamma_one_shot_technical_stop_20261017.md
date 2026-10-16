# M1-S V1 γ依存保存信号解析：one-shot技術停止と保全

2026-10-17 JST。`TECHNICAL_STOP / ONE_SHOT_CONSUMED / RETRY_0 / MANDATORY_STOP`。

## 結果

明示承認された一回実行を開始したが、固定SOURCEのworker起動処理で`AttributeError: module 'sys' has no attribute 'set_int_max_str_digits'`となった。Python 3.11.0rc1の固定実行環境にはこのAPIがない。SOURCEのruntime適合確認でこの差を見落としていた。

|項目|保存状態|
|---|---|
|one-shot起動|1回、exclusive marker消費済み|
|retry|0回|
|worker／supervisor／outer|すべてTECHNICAL_STOP、worker exit2、outer exit2|
|方式行|108行すべてNOT_EVALUATED|
|対応比較|48対照すべてNOT_EVALUATED|
|保存H4信号JSONの科学数値decode|未到達|
|π／log係数K／margin／γ境界|未評価|
|shot forecast／絶対shot／native費用／RPE総資源|未評価|
|科学的GO/NO-GO|未判定|

SOURCEの`control.py:278`で停止し、科学engine importは次の279行、保存信号JSON decodeはその後に位置する。これはtracebackと固定SOURCEの静的照合、同じ実行ファイルSHAを持つPythonのAPI有無確認に基づく。workerの再実行や不具合の数値再現は行っていない。最初のsnapshotとraw failure bytes／SHA256を保存した。

## 固定した対象・承認来歴

- SOURCE S：`f2a5611ca9fc7461c64f2deb74a1f624ac8aa7c5`
- 正式契約C：`9aafa8b754aa08e38f92cd06e503a17904908f5e`
- 専用authority-only A：`b0b488364205fdf0f8082bbff2a0a585d3ef18f7`。Cの直接の子でauthorization1fileのみ。実行前に公開した。
- branch：`track-b-m1s-v1-gamma-one-shot-20261017`
- 採択契約SHA256：`0d89ce5dd1e6f56b04cbd57402afc5a20ec6df68b8fc958d90931a6eeb232286`
- authorization path：`artifacts/track_b_m1s_saved_signal_gamma_v1_execution_authorization/2026-10-17/v1/authorization.json`

ユーザーの「固定SOURCEと正式契約Cに基づき一回だけ実行、結果または失敗記録を保全してmandatory STOP」という承認全文をAに保存した。以前の契約採択記録にある実行未承認の記述は当時の履歴として保持し、今回の新しい実施承認・消費状態を別記録で追跡する。旧grantやmarkerを流用しない。

対象はlinear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9、保存stateの元norm²商、`+iTH`。δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1、shared_pm／ordinary／independent_pm、72非零方式行＋36保存zero control＝108行、48対照を維持した。

入力は旧完了commit `025c5a90888b02b73d2815fbb10d6bd1d2823602`の保存raw結果。454422 bytes、SHA256 `ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`。今回はhash照合を通過したが、科学数値の解析には到達していない。

## 実行条件とresource

PER_METHOD_ROW、α=0.05、fresh-IID／Hoeffding／X/Y等配分、保存数値の条件付きcoverage、p=128、γ全共通領域の記述診断だけという採択条件を変更していない。SOURCE9 file／契約／入力identityを実行前に照合した。

数値worker1／GPUなし、thread各1、CPU160秒（hard161秒）、wall320秒、aggregate virtual memory2 GiB、result/log各16 MiB等を適用した。数値SOURCEのπ/log共有とbias別trig cacheをそのまま使い、既存の登録外34 testsのPASS記録を確認した。追加のtoy数値実行、SOURCE修正や科学条件調整は行っていない。

32 logical CPU・32 affinity CPU、約50 GBのMemAvailableをmetadataで確認したが、採択上限に従ってworker数を増やしていない。

保存resource記録ではworker CPU約0.003736秒、worker maxRSS13796 KiB、supervisor計測wall約0.020685秒、outer wall0.10秒／maxRSS23040 KiB。親VM32075776 bytes＋子AS cap2115407872 bytes＝採択aggregate2147483648 bytes。CPU/wall超過ではなくAPI不整合による停止である。これらは起動prefixの計測値であり、未実施のH4解析の速度やresource適合を検証した値ではない。

## 原因と未適用の技術修正候補

固定Pythonのexecutable SHA256は`320707a1d684b043661306d508d1a8655f507c0bb8e827398cb53680054aa7c8`。SOURCEはこの環境に対し無条件に`sys.set_int_max_str_digits(25000)`を呼ぶ。登録外の算術／guard testsはproduction workerのこの起動処理を通しておらず、runtime API差を検出できていなかった。

修正候補は、APIが存在するときだけ整数文字列変換上限を設定し、存在しない環境では既存rational-bit／memory／output制限の下で既定変換を用いること。SOURCE固定前に必要なruntime APIを照合するmetadata preflightも候補となる。**いずれも未適用で、新SOURCE・新契約・新authorizationを作っていない。** 修正の妥当性や新runtimeを採る場合の条件は、別途の技術準備で検証する必要がある。

今回のone-shotは消費済みなので、修正を適用して同じ承認を再利用することはできない。新しい実行には修正版の固定と別途明示実施承認が必要。数学的境界式や研究のGO/NO-GOをこのruntime失敗から判断しない。

## 保存資料・証拠境界

raw outputは`artifacts/track_b_m1s_saved_signal_gamma_v1_result/2026-10-17/v1_one_shot/`。

- `result.json`：例外・traceback・108行／48対照の未評価placeholder・usage
- `supervisor_result.json`：worker exit2、wall上限未到達、失敗分類・メモリ予算・provenance
- `exclusive_launch_marker.json`：consumed=true／retry=false、固定S/C/A

監査は`artifacts/track_b_m1s_v1_gamma_execution_audit/2026-10-17/v1/`。prelaunch identity、stdout/stderr／outer usage、runtime API原因記録、raw bytes固定、saved failure audit、preservation audit、evidence indexとpublication receiptを保存する。

旧v4 STOP/null、Gate0 v5／独立reference、finite pilot、V0／γ候補、正式契約、旧SOURCE・結果・ledger・grant・marker、親／他worktreeを保全する。条件付き証拠をplatform certificateへ格上げせず、Task A/B・整数shot／native費用の限界を保持する。

**run1／retry0、one-shot消費済み。新しい科学数値処理・γ境界・shot予測0。保存・監査・GitHub公開後mandatory STOP。**

## 保存監査・保全監査

保存failure監査98 PASS／0 FAIL、保全監査10 PASS／0 FAIL。schema・S/C/A・例外・全未評価行・resource prefix・raw bytesを確認した。これは失敗記録の完全性検査であり、V1の科学数値correctnessがPASSしたという意味ではない。

既存7,665非index file、親1284 tracked file・index・HEAD、全207既存worktree HEAD、新raw output3 fileのbytesを保全。新execution authorizationはAの1件、新markerは消費済み1件、追加grant fileは0。残りの実施権限は0で、旧grant／markerを再利用しない。公開receiptをpayloadとは別commitで固定し、remote SHAと公開bytes／SHA256を照合する。
