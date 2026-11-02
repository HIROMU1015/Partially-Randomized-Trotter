# 2026-10-06 Track B RA-D0 preparation

利用者のformal design v1 §16に従い、数学監査8a04c148基点の独立branchで準備を実施。
保存21 columns/x、18 sign pairs、126 sequencesを静的照合した。
ideal class nestingは成立する一方、B0_savedとideal B0の差およびdyadic qとB1/B2 exact membershipの
不整合を確認した。原文を変更せず、数値baseline規約をGPT reviewへ返す。

LP compiler、rational primal/dual/Farkas certificate kernel、固定sampler丸め、grid recipeを追加。
35 focused synthetic tests PASS、synthetic solverは人工1問題。登録tableはsolve=0。
737 grid points、上限444,411 LP呼出しのため、query実行scope・計算資源上限も結果前判断が必要。

判定はREVISE_RA_D0_NUMERICAL_BASELINE_AND_EXECUTION_CONTRACT、RUN_READY=false。
既存R1/R1.5/数学監査と全科学分類・旧STOP・Track Aを保持。新合成／science／GPU等0。
[source review](../../tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)、
[GPT handoff](../../tracks/algorithm_codesign/ra_d0_gpt_handoff_20261006.md)。
必要資料をcommit/pushしてmandatory STOPし、修正採用・追加実行の範囲判断をGPTへ戻す。
