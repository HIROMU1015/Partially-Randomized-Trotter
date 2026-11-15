# 粗幅p7/Yoshida4：幅固有価格と固定A_SHAPE費用

保存H4/1Å/STO-3G/tol-only DF10・B0I p7/標準Y4/32h/5・h_RPE=.1・P0除外。ユーザーの最大6 compile許可に従い、m5/q5とm6/q10をX/Y別に取得し、axis別307+16539qを凍結した後、m9/q80を取得。各1,323,427 RZで凍結予測と一致。直接16軸（保存10＋新6）、モデル10軸の固定A_SHAPE条件付き費用は372,035,410 RZ。3固定基準との差は−72,771,600、−63,606,158、−23,927,268 RZ。

総額の84.831223%はモデル、75.400727%はq80外への未検証外挿。q640の価格誤差・CI・正式G/winnerはUNKNOWN/null。Q3は8855f420aの適合を引用し、追加信号/Q3/他配分0。m9をfitに戻していない。

[REPORT](../d_p7_y4_coarse_price_calibration_20261116/REPORT.md)／[HANDOFF](../d_p7_y4_coarse_price_calibration_20261116/HANDOFF.md)／[26軸と費用](../d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json)。係数凍結d4ae4c408、結果表c7c8c1edd。

初回は未展開referenceでimport停止し、H/state復元/prepare/build/compileは0。元blobを復元し旧run01/STOPを保持した別namespaceで同じ未取得6軸を実行した。C sidecarの旧namespace参照も値・科学gateを変えず正しい公開proofへ結合した。最終11技術testと実6軸作用gate/C guardが成立。科学回路のretry/条件緩和なし。統合担当へ条件付き点の追加を渡し、ロードマップ本体は変更せず公開照合後mandatory STOP。
