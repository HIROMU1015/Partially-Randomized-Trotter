# 人工End-to-End受入

最終測定SOURCE `b7ad344401b10395c49e4cfa806909282859bab5`。新規102 unit/negative、30 integration、20 semantic mutation、既存107 regressionを別集計した。全test PASS、FAIL/ERROR/SKIP0。既存回帰はSOURCE8d5135eeで実施し、対象旧SOURCEが後続Canonical補完で不変であるため再実行していない。

22 case：正常2（n16/n256）、意図した技術失敗13、証拠不足7。保存障害をPASS結果に変更しない。SOURCE固定後の全caseは新しい人工scopeを使用した。

n256は条件あたりcorrectness256＋baseline4096＋instrumented4096＝8448、S/P1計16896。planned/reserved/started/completedすべて一致しunknown/cancelled0。EVENT14728、ZERO_R1020、ZERO_C1148は人工fixture観測であり科学的頻度ではない。1536 full document＋15360 hash-onlyをsynthetic replayで照合し、全16896をworker PID/global offset/ENTER/COMPLETE/state/receipt/callerへjoinした。

20 mutationは原event実行を増やさず、保存envelope/file/final/delivery/callerのhashを再構成した不正Evidence。既存storage整合検証は20件を受理し、新adapter検証は20件すべて拒否した。codec child/tag/digest、classification/fingerprint、factory/runtime/source/global ID、seed/value/width/discard、angle/requeryを対象とする。

S失敗時P1禁止、P1失敗時S証拠保持、worker crash/ACK不明UNKNOWN、trace/state/receipt/marker/delivery障害、call予算、monitoring新関数を確認した。fake authority raceは32 attempt中winner1、commit ACK unknownはspentのまま二重claim不可。原DBと観測をarchiveへ保存した。

実G7/F3 law、actual variable consumption、数学的angle enclosure、actual科学費用、native、実authorityはNOT_ESTABLISHED。同一作者の技術自己検証であり、独立受入ではない。
