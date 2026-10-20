# 段階B出力schema・会計規約

すべて2026-10-21の1回の保存値解析が正本。後続文書作業では新しい数値解析を行わない。費用単位Gは保存analytic cosine/sine shot数で重み付けしたfull measured Hadamard wrapperのcompiled RZ。CXやpretranspile数を上界へ転用しない。

|ファイル|行/単位|主要fieldと意味|
|---|---|---|
|result.json|conditions 8|固定candidate/system/rank/ε/q/r/K、元SOURCE/result commit/cohort、shots、PF固定G、law、finite_sample、groups、conditional_expectation、tasks|
|probabilities.csv|8条件×4層=32行|condition/candidate/stratum/probability/support_status。確率は100桁評価の文字列。K2 S4の理論支持外とK4 positive未観測を区別|
|observed_strata.csv|32行|n/mean_G_exact/SE_G/sample_min_G/sample_max_G/unknown_population_mean_G。n0の推定値は空欄、n1はSE空欄。母平均は全行未知|
|sensitivity.csv|8行|finite_sample_mean_G、PF_G、observed_plugin_G、unknown_mass、PF_minus_observed_plugin、unknown_mixture_mean_threshold_G、threshold_over_observed_S0_mean、conditional_direction、population_guarantee=False|
|trajectory_costs.csv|1,526行|candidateとreplicaが一意。stratum/order2/order4/G_exact/RZ_cosine/RZ_sine/event_digest/trace_sha256。生event列・seed・絶対locatorは含めない|
|independent_numeric_audit.json|8照合|Fraction/isqrt/DP/moments別実装の保存照合。独立担当者による科学reviewではない|
|stratum_probabilities.svg|8条件/4層|保存結果の図。支持外と正の小確率を区別|
|contract.json / execution_approval_receipt.json|固定実行前|SOURCE/input/runtime/定義/精度/予算/STOPとユーザー承認の局所転記。独立署名ではない|
|grant_issue_receipt.json / grant_consumption_receipt.json|発行時/消費時|新grant digestのみ。発行receiptのUNUSEDは実行前時点を保全、消費receiptとprivate markerは1回消費済み|
|native_exit.json / driver_exit_receipt.json|実科学worker/driver|workerの同親wait4 exit0/reapと、実行toolで観測したdriver exit0を区別|
|completion_audit.json / STOP.json|科学終了時|SOURCE/input/runtime・旧証拠SHA・call予算・禁止操作0、mandatory STOP|
|documentation_audit.json / status_manifest.json / SHA256_manifest.json|保存・文書監査|CSV/JSON転記一致、構造/来歴/SHA/既存文書保全。新科学処理なし|

JSONのmean_exact/variance_exactはFraction表記、Decimalは文字列、未知値はnull。CSVではnullを空欄に写し、ゼロと区別する。sample_min/maxは保存標本だけの範囲。unknown_termsは未観測support内の平均費用・分散をnullで保持し、観測層母平均も未認定。mathematically_guaranteed_cost_upper_bound、unconditional_population_mean_G、confidence_intervalは全条件null。

主法則は保存binary64 τのcanonical有限law。保存categorical pの丸め差とsimplex診断を別fieldで保持する。元lawを書き換えず、実PRNG bit lawやinterval certificateは未保証。旧pilotは系・DF・p/q/r/K/cohortが異なり、新Pへpoolしない。

構造監査scriptは保存fieldの対応を調べるだけで、確率・平均・SE・閾値を再計算しない。SHA manifestは自身だけを自己参照から除く。GitHub公開は別承認まで行わない。段階Bでmandatory STOP。
