# 2026-10-13：保存H4 RTEの基底変換費用内訳

利用者が小規模費用診断の実行を指示した。旧H4／1.00 Å／STO-3G／DF12／L_D3／delta .02／K2の同一saved seed列を復元し、列長3／6の各先頭4本を測定した。

[診断報告](../../diagnostics/rte_basis_cost_attribution_20261013.md)：加法的内訳で基底変換はRZ84.8266%、CX68.9655%。部分費用合計は通常の全体compileよりRZ8.4882%、CX3.3079%大きく、最適化後の厳密占有率とはしない。

unique compile24回。初回はstrict全体位相検査でtechnical停止し、原記録を保持。追加compileなしのsaved演算子照合で全体共通位相のみの差を確認し、制御枝間位相の変更ではないことを保存した。全8列36 eventはorder0で、母平均・T費用・全PR優位を認定しない。

Track Bの新規探索休止、既存source／契約／結果／marker／STOPは不変。既存RTE builderへ変更なし。今回の費用診断を完了しSTOPする。
