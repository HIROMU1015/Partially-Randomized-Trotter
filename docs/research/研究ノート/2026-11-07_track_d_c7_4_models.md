# 2026-11-07 Track D C7-4：モデル再解析と数値安定性

確認済み。[最終技術報告](../track_d_c7_4_model_stability_20261107/REPORT.md)／[モデル](../track_d_c7_4_model_stability_20261107/MODEL_REPORT.md)／[数値安定性](../track_d_c7_4_model_stability_20261107/NUMERICAL_STABILITY.md)／[次段階案](../track_d_c7_4_model_stability_20261107/NEXT_EXECUTION_DRAFT.md)。

C7-3最終acea439を基点に、新SOURCEdd31e1e2878479eabe2e7cc27c21b26c8c7dc794で保存q2/4/8/16/32の88 pathを再解析。旧C6aラベルの実装誤認による再解析前run01 STOPはSOURCEdeb92995と共に保全し、schemaだけ修正してrun02を初の科学再解析とした。元結果・係数・共分散・SOURCE/STOPは不変。

4 familyの低次元fit/5fold CV/whole-path bootstrap、方式/資源/XY依存、過学習/未取得biasを保存。全88 CX残差0は全path公式ではなく、P0 bridgeは不足。paired depthはbasis-run、Reference絶対depthはlane-maxが候補。保存モデルを同じデータで選んだので確認的holdoutではない。

q32最大残差の保存NPZだけを全要素再照合しPASS。人工長系列6件/短P0 2本・distinct回帰18件PASS。native変換保持版は旧genericと同じ結果で精度改善を示さず、q≥64精度は未資格。新H4 sampling/build/compile/旧科学run retry/N/総費用0。

次はユーザー/GPTが、先に数値経路を補うか停止条件付きq64/cut32/fresh8 pilotを許可するか判断する。q64案は未採択・未実行。公開SHA/tree/bytes照合後mandatory STOP。Codexは別の明示許可後だけ新しいscopeを担当する。

2026-11-08 JST：報告書・次段階案を最終収録。科学SOURCE/許容値/モデルfit結果は変更せず、公開照合へ進む。
