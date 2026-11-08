# 保存済みPASS結果の要約完了

2026-11-08 JST。SOURCE `0df0134b5f259bbb2603719227e21831d876af9f` のrun01は、全256入力raw PASS、S1/R1各1 core・4 wrapper・新P0 4・新postcheck4全PASSを完了した。その後、最後のanalytic χ要約で、行列用component_checkへshape(1,)を渡したため `ValueError: Invalid norm order 'fro' for vectors` となった。科学的な閾値超過ではない。旧SOURCE・87 fileの結果/private原本・失敗・STOPは不変で保全する。

利用者の今回指示§2の範囲内の技術的不具合修正・§11/15/17の未完了結果整理だけを行う。§14に従い新しい科学実行・wrapper postcheck・retryは行わない。元runner・実行SOURCEを改変せず、別SOURCEのsummary専用入口で保存PASS結果だけを読み、旧未完了のpaired X/Y・full coherence・analytic χの集約6比較と全7資源差を出力する。修正はχの同じ一つのcomplex値を1x1行列へ包むことだけで、値・入力・phase・精度・checkerは不変。

新sampling/core/wrapper/raw/P0/postcheck/ground/DF fit/model fit/N/13roundは全て0。旧4 wrapperを再compile・再postcheckしない。別path/cut/qへ進まず、旧STOPを解除しない。再利用するmetric記録から指定7資源だけを選び、global_phaseやmeasurement_countを資源差に混ぜない。元の全metric/phase記録は旧postcheckに保持する。

sourceを固定してから専用 `run02_summary_completion` を一回だけ生成し、旧失敗型・全4 PASS・元hashをguardする。新旧call数とSOURCEを区別し、公開照合後mandatory STOP。これは保存結果の未完了整理であって、新しい科学条件を採用した実行や科学的FAILのやり直しではない。一般性・新規性・全RPEの判断へ拡張しない。
