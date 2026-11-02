# 完了済み条件付きpilotの保存値（再解析なし）

2026-10-21 JST。全行linear 1 Å/STO-3G/T=.8/P=0、full Hadamard wrapper、各群n8。両軸平均が同値だったという各報告の範囲でRZ/軸を転記する。母平均・全support費用上界ではなく、今回Pへの代入は行わない。

|条件|層|保存RZ平均/軸|保存標本SE|正本|
|---|---|---|---|---|
|legacy DF12 p3/q4/r4/K4|s2_multi|13,004.75|131.58|[H4_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/97d25cc431516d720ce91c7cb4f148e061d8ea34/docs/research/track_a_h4_followup_20261018.md)|
|legacy DF12 p3/q4/r4/K4|s4|13,144.25|200.34|[H4_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/97d25cc431516d720ce91c7cb4f148e061d8ea34/docs/research/track_a_h4_followup_20261018.md)|
|legacy DF12 p3/q8/r4/K4|s2_multi|25,306.125|157.07|[H4_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/97d25cc431516d720ce91c7cb4f148e061d8ea34/docs/research/track_a_h4_followup_20261018.md)|
|legacy DF12 p3/q8/r4/K4|s4|25,731.625|263.65|[H4_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/97d25cc431516d720ce91c7cb4f148e061d8ea34/docs/research/track_a_h4_followup_20261018.md)|
|DF19 p5/q8/r2/K2|s2_multi|97,173.5|444.8895|[H6_DBA](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/648184eb2a6f1c2406df17c1796eb5cc195908a8/docs/research/track_a_h6_dba_20261018.md)|
|DF19 p5/q8/r2/K4|s4|97,839.625|589.9994|[H6_DBA](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/648184eb2a6f1c2406df17c1796eb5cc195908a8/docs/research/track_a_h6_dba_20261018.md)|
|DF28 p7/q4/r4/K4|s2_multi|130956.75|1043.5897|[H8_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/839c403004cf9343730c43adb346024aea41999b/docs/research/track_a_h8_ab_20261018.md)|
|DF28 p7/q4/r4/K4|s4|132589.75|1031.1079|[H8_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/839c403004cf9343730c43adb346024aea41999b/docs/research/track_a_h8_ab_20261018.md)|
|DF28 p7/q8/r4/K4|s2_multi|256947|1330.3673|[H8_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/839c403004cf9343730c43adb346024aea41999b/docs/research/track_a_h8_ab_20261018.md)|
|DF28 p7/q8/r4/K4|s4|258848.125|2405.6575|[H8_RARE_COMPLETED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/839c403004cf9343730c43adb346024aea41999b/docs/research/track_a_h8_ab_20261018.md)|
|DF28 p7/q4/r4/K4|s2（実現全single）|127,062.125|952.457994|[H8_S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b21e90b8368645a5066c06a2a0a2ceb2850d162e/docs/research/track_a_h8_conditional_s2_run02_results_20261018.md)|
|DF28 p7/q8/r4/K4|s2（実現全single）|255,555.875|1,317.070172|[H8_S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b21e90b8368645a5066c06a2a0a2ceb2850d162e/docs/research/track_a_h8_conditional_s2_run02_results_20261018.md)|

H6旧s2 pilotはp5/q8/r2/K2、n8のRZ平均96,285.25、SE324.0004で、実現全N2=1。別のmulti pilot取得済みという後続状態と両立する。出典：[H6_S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/references/track_a_h6_conditional_s2_pilot_result_v1.md)。

後続B診断は各群の存在を確認する一方、ordinary sampleのpayload結合がなく、無条件平均をnullで保持した。元sensitivityがmulti/s4をnull変数にした履歴と、後続pilot取得済みの現在の証拠を区別する。条件付きpilot費用の取得、無条件期待費用への結合、厳密な推論を別段階にする。
