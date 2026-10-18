# 説明側の原因調査

状態：STATIC_CAUSES_IDENTIFIED_TECHNICAL_FIX_PREPARED。実科学数値再解析0、説明完了の新認定0。既存core32/32は原結果のまま保持する。

|対象|静的に確認した問題|分類|今回の処置|
|---|---|---|---|
|PR28 sideのNORMALIZATION_LOG_BINDING|1-stepのbを累積Bとして渡した。さらにshot規則のlog_B_marginを落とした|TYPE 1 binding不具合、TYPE 2 段階を同一視するguard適用の誤り|段階を明示し、元shot入力を再現する説明専用SOURCEを準備|
|PF32 sideのLOG_BOUND_MISMATCH|保存log_shot_boundはlog_B_upperを使用、旧runnerはnominal log_Bを使用|TYPE 1 field／式入力不具合|empirical marginを別fieldで保持し、元の加算・乗算順序を採用|
|欠測4件|保存endpointが元signalのprecision_rowsにない、またはsignal自体が未結合|TYPE 1 入力結合不足／TYPE 3 原signal未回収|3件の説明fieldを保存会計から回収、1件は部分回収と欠測保持|
|修正後の実値整合性|headroom／log bound／ceil等は未実行|TYPE 4 数値検証保留|別の説明専用one-shot承認へ|

## PRの数学的段階

固定providerのfinite_rte_distributionは、偶数k=0,2,…,Kについてpaired-order weightを合計した1-stepの`b`を返す。raw pathは各stepでbにより規格化される。h6_matched_port_v1.compact_signalは`b=finite_rte_distribution(...).exact_finite_distribution`、`log_B=R*log(b)`を保存する。Rは保存cellのq/r scheduleに結合したstep数である。従ってnominalな全trajectoryのnormalizationはb^Rに対応し、b単独と累積Bは別段階の量である。

生成元：`c31ef424175c02379eafebe8fc6c7548c1d9aabe:src/trottertracks/resource_applicability/h6_matched_port_v1.py:36`および同commitの`src/trotterlib/rte.py:1451`。H4/H6/H8の実際の封印providerについて、このport・finite law・precision_rows・u_aware_shotsの4fileを各provider hashとGit blobへ照合した（計12組）。provider SOURCEはH4=217c971d…、H6=87b6b9f4…、H8=c31ef424…で、今回調べた4fileのbytesは各系で一致した。

旧parser `5c256067…:scripts/prepare_mechanism_analysis.py`はsignal.bを`B`へ転記し、累積log_Bと組にした。旧coreの`math.log(B)==log_B`は、同じ数学的対象についてのみ意味を持つguardであり、この組には適用できない。元u_aware_shotsの同guardは、明示された保存normalization_upperとlog_B_upperの同一性を検査するものである。元guardを削除・緩和する根拠はない。

修正では`step_b`、`step_count`、`log_B_nominal`を別fieldにし、有限RTEについて`R*log(step_b)`と保存nominal log_Bの段階対応を確認する。saved upper binary64を使う経路は別名にし、その場合のstrict log bindingを維持する。実科学値でこの検査を実行したとは主張しない。

## PR／PFに共通するshot入力

実providerのh6_matched_accounting_v1.precision_rowsは、`u_aware_shots(..., log_B_upper=record.log_B + factor*allowance.log_B_margin, ...)`を呼ぶ。D0 saved-value accountingも同じ式を用いる。normalization_upperはこの呼出しでは渡さず、元shot規則は`B_upper=exp(log_B_upper)`経路を使う。

`log_Nstar = log(2) + 2*log_B_upper - 2*log(h) + log(log(2/alpha_axis))`。

旧runnerは保存marginを結合せず、nominal log_Bで上式を評価した。PFのnominal log_B=0でも、shot規則のempirical log marginは別の入力である。PFのraw unitary normalizationと、条件付きshot会計で用いるeffective upperを区別する。上界というfield名からrigorous certificateを認定しない。

生成元：`c31ef424…:src/trottertracks/resource_applicability/h6_matched_accounting_v1.py:34`、同commitの`ax2b_numerical_accounting.py:21`、`8601b921…:scripts/d0_frozen_shot_rule_v1.py`と`d0_saved_value_reanalysis_v1.py:65`。指定された8601b921…／31ef62a1…の費用会計は保存shotを利用する側であり、今回の修正対象はその費用coreではない。

## 判定の限界

この診断はSOURCE・schema・fieldの数学的意味による。実値の差、必要な許容差、ceil境界、修正後の説明完了件数は再評価していない。最初に失敗した旧guardより後段の正常性も未認定。未知の不一致を丸め誤差として扱わない。coreを疑わせる静的な重大問題は今回見つからず、元結果の再計算・改訂は0。

[出典trace](mechanism_explanation_source_trace.csv)、[欠測調査](mechanism_explanation_missing_input_report.md)、[修正仕様](mechanism_explanation_fix_specification.md)へ続く。
