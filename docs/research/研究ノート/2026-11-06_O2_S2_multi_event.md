# 2026-11-06 O2-S2 固定pair-table評価の途中停止

ユーザー指示により、設計 `5eb90e8241e69e85a1ca3e79938a367700c24a48` のD2 Closed P5 r2、τ=5/7、x=5/14、degree5、T主指標、旧D2誤差/precisionによる六cell代替forecastを採用して実行した。固定SOURCEは `16d10faabb1f6aa4bda42f3aef81d89bf9ca9832`。

人工correctness38件PASS後、3,969 pairの価格表とP0/A・P0/Bの2cellを取得した。P0のN/axisは773,966、既知二axis T費用は429,626,133.047707／427,870,692.302325。3-qubit synthetic R、p=(1/5,3/10,1/2)、state000。分子geometry/basis/DF rank/L_D/δ窓は非該当。未知共通費のaffine項は残る。

ISの巨大Fractionをprocessから返す際、整数文字列変換4300桁制限によりSTOP。qA/qB四cellの結果は未取得であり、追加利益・失敗・同等性を判定しない。I/O制限の適用範囲をpool全体へ広げる修正案と人工転送回帰1件PASSを準備したが、実行済みSOURCEへ適用せず科学retryも行っていない。

理由：今回指示§14の停止後無断retry禁止を維持する。次はユーザーの同一科学条件での再実行許可後、Codexが新SOURCEを固定して六cellを評価する提案。分子/追加条件/重要レビューは別範囲。

[報告と出力](../o2_s2_multi_event_validation_20261106/REPORT.md)を参照。旧SOURCE/raw/results/STOP、mainと元dirty checkoutは保全する。
