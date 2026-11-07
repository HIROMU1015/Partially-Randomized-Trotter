# Signed Support 448-shot：一campaignの実行条件

2026-11-07 JST / SERVER1 GPUA100-gleap-01。

公開・人工検証済みSOURCE `f70e555fbfe08676f530d989192aade75a57ab1a` と、その[実行前preflight](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f70e555fbfe08676f530d989192aade75a57ab1a/docs/research/scv_signed_support_448_preparation_20261107/SCIENCE_SOURCE_PREFLIGHT.md)を使用する。ユーザーの進行許可を受け、同じ448-shot一campaignを一回だけ投入する。新しいgrant/approval packetは作らない。

固定対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子sector/L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual finite-grid Q/selected full、N2保証ground、認証済みexact Fraction engine、理想gate、X/Y。

- Campaign `SCV-SIGNED-H4-H1-PAIRED-20261107-v1`、purpose `SCV_SIGNED_SUPPORT_H1_FRESH_PAIRED_DIAGNOSTIC`。
- 独立h1 train64/軸、両軸取得後X/Y各一回intercept OLS・binary64 slope→Fraction freeze。Signed width≤6、inactive4層λ0。
- 共通fresh評価quota=(8,128,8,8,8)/軸。B1-OLとSCVは同じ評価recordを使い、追加測定/二重課金しない。
- train128＋eval320＝最大448 physical shots、1 campaign、science retry0。
- Seed20261110、PCG64/SeedSequence、12domain/24children、path64+bit1 word/trial、総word≤29120。
- 全5層Full＋trial ID固定half A/B、各軸3、計6集計。追加bit/fit/価格なし。
- 主費用はfull compiled RZ、状態準備のみ除外、Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed_transpiler17、backend/coupling/layout/routing=None。
- finite cap・独立stream・STOP条件は固定SOURCEのSETTINGS/PROFILE_LOCK/SOURCEと同一。

旧224/752/397/1339件のbit・係数・seed・cacheは使わない。N1準備/CDF、N2 state solve/spectral、engine認証、signed moment計算を再実行しない。Float/GPU/O1/O2 kernelへ置き換えない。

実SOURCE worktreeは `/tmp/scv-signed-support-448-preparation-20261107`、そのbranchとremoteはf70e555のまま維持する。出力は `artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01/`、管理記録は同親の`manager_run01/`。結果公開は別worktree・branchで行い、実行中SOURCEを変更しない。

固定SOURCEの既存cron→outer recorder→manager→private parent/workerを使用し、consume-once、Restart=no、trigger撤去を維持する。Wは実idle/affinity・memory/pressure・有用task数から決める。共有設定・他process/job・並行O1/O2へ干渉しない。

identity/stream/rank/nonfinite/width/call/価格/保存/非干渉不成立で全campaign STOP。START/STOPの短い共有排他を維持し、新callを止めて受理済みin-flightだけを回収する。bit-before-price、全費用・欠測UNKNOWNを保全し、失敗時のrefit/追加shot/seed変更/再開・再投入なし。

結果は探索的paired診断で、母分散低下・厳密精度・期待RZ改善・一般的優越の保証ではない。完了または中断後、保存・公開・remote内容照合を終えてmandatory STOP。追加campaign/保証付きcalibration・production/詳細科学レビューへ進まない。
