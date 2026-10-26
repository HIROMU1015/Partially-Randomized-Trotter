# H4 XDF9 native／楽観的shotsスクリーニング：限定実行結果

固定SOURCE7328ecc9a0602524810e2055cf99c2b1e07729dbを無変更で一度実行し、83/83 COMPLETE、correctness FAIL/inflight/retry/resume0。[全結果・資格の限界](../../../docs/research/research_c_c1_h4_xdf9_cost_screen_execution_20261026.md)／[raw48・QPY4・SOURCE・journal・STOP](run01/)／[summary](execution_summary.json)。

直鎖H4・1Å・STO-3G、Nα=Nβ2、保存元順XDF9、全frame S2 s1/s4、δ.1・m0..12、own-ground oracle/preparation excluded。共通precision/P0・既存shared/fenced serial policy。kernel RZは両sとも15720。楽観的総RZはs1=144805110560／s4=530107748679で、各保存XDF8/CDF8 totalより大きい。4比較とも固定処方／実装の数値モデル内のstrict改善除外で、普遍的下限・actual CA・資源優位は示さない。

[native metrics](native_metrics.csv)／[52軸shots](optimistic_shots.csv)／[52軸1測定cost](serial_per_shot_costs.csv)／[4比較](screen_comparisons.csv)／[preflight](execution_preflight.json)／[postflight・原本hash](execution_postflight.json)。実際の合成誤差・full operator・strict enclosure・actual CA／費用は未資格、該当値nullを維持。SOURCE56/input24/原packet不変。run01は消費済み、公開照合後mandatory STOP。

## 以下は準備時点の記録

[技術報告・単調性・制約・full512案](../../../docs/research/research_c_c1_h4_xdf9_cost_screen_preparation_20261026.md)／[未承認83-call契約](../../../docs/research/research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)。

人工27＋既存回帰70＝97 tests PASS。[SOURCE／logs／reuse／byte preflight](technical/)。SOURCE56／saved inputs24／原packetと依存9版を照合。実H4 NPZ／QPY数値decode・回路build／compile・shots／費用・科学callは0。run01は未作成。

lower boundは固定PF・Hoeffding十分shots処方・非負追加signal誤差・固定serial実装の範囲。未知の実合成誤差はnull。厳密rounding enclosure・full operator・actual CA／shots／費用・普遍的下限は未資格。科学実行には別の明示許可が必要。公開照合後mandatory STOP。
