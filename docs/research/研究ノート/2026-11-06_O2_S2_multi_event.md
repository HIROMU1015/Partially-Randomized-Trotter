# 2026-11-06 O2-S2 固定pair-table評価の途中停止

ユーザー指示により、設計 `5eb90e8241e69e85a1ca3e79938a367700c24a48` のD2 Closed P5 r2、τ=5/7、x=5/14、degree5、T主指標、旧D2誤差/precisionによる六cell代替forecastを採用して実行した。固定SOURCEは `16d10faabb1f6aa4bda42f3aef81d89bf9ca9832`。

人工correctness38件PASS後、3,969 pairの価格表とP0/A・P0/Bの2cellを取得した。P0のN/axisは773,966、既知二axis T費用は429,626,133.047707／427,870,692.302325。3-qubit synthetic R、p=(1/5,3/10,1/2)、state000。分子geometry/basis/DF rank/L_D/δ窓は非該当。未知共通費のaffine項は残る。

ISの巨大Fractionをprocessから返す際、整数文字列変換4300桁制限によりSTOP。qA/qB四cellの結果は未取得であり、追加利益・失敗・同等性を判定しない。I/O制限の適用範囲をpool全体へ広げる修正案と人工転送回帰1件PASSを準備したが、実行済みSOURCEへ適用せず科学retryも行っていない。

理由：今回指示§14の停止後無断retry禁止を維持する。次はユーザーの同一科学条件での再実行許可後、Codexが新SOURCEを固定して六cellを評価する提案。分子/追加条件/重要レビューは別範囲。

[報告と出力](../o2_s2_multi_event_validation_20261106/REPORT.md)を参照。旧SOURCE/raw/results/STOP、mainと元dirty checkoutは保全する。

## 同日追記：v2転送修正再実行で六cell完了

ユーザーの明示再実行許可を受け、private processの整数I/O contextをpool全体へ適用しfork継承を固定した。科学条件は変更せず、実行SOURCE 4c9f3fb32deef5c50fd98517ccbcd03311f0b059とv2出力を分離。人工39件PASS後に六cellを完了し、旧D2価格exact一致・全3 lawのsupport/係数/q/weightを確認した。

同じB上の既知二axis T費用はP0=427,870,692.302325、qA=427,931,933.535182、qB=427,926,178.861741。qBはqAより約0.00134476%低いがP0より約0.01296807%高く、元lawに対するT改善を支持しない。十分shot数N/axisはP0=773,966、qA=774,100、qB=774,093。固定入力・価格・精度内の結果であり、一般的なISの失敗とはしない。

理由：期待価格単独ではなく、range/M2を使う同じBernstein十分Nとの積を判断対象とした。共通非負h_T下でも主比較の向きは変わらない。古典online費・分子全RPE・新規性/科学GOは未判定。次はユーザー／GPTが本S2案の拡大可否を判断する。追加計算・重要レビューへは進まず、[v2公開報告](../o2_s2_multi_event_validation_v2_20261106/REPORT.md)の取得確認後STOPする。
