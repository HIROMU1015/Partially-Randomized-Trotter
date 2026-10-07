# 2026-10-07：RA-D0 v3 certified infeasibilityの修正

利用者reviewで、v2 `4012cbd167ec10f143fbfddc89feff4dec54bf2b`がB2 minimumのexact-certified
infeasibilityもtechnical STOPにしていたと指摘された。独立branch
`track-b-ra-d0-source-review-v3-20261007`でそのblocking issueを修正した。
[source report](../../tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md)と
[GPT handoff](../../tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md)を新しい入口にする。

minimumをcertified feasible / certified infeasible / technicalへ分け、exact rayを同じLPで再検証する。
一objectiveでも証明済みinfeasibleならpointをfreezeに記録し、budget/queryを空にして次nへ進む。
B0_savedだけでprimary budgetを作らず、新しいB3 queryも加えない。Case Cは従来どおり停止する。
coverage/STRONG/LOCAL/NOの定義と数値・call/resource契約は維持した。

旧80 testsをbyte-exactで残し、20追加、計100 focused tests PASS。
21 columns/x、18 sign controls、旧R1/v1/v2 evidence・旧STOP・Track Aは保持。
registered solver/minimum/budget/B3/witness、新synthesis/science/NPZ/GPUは0。authorization/marker未作成。
`READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION`でGPT最終source reviewへ戻し、mandatory STOPする。
このstatusは実行承認ではない。旧v2記録は変更しない。
