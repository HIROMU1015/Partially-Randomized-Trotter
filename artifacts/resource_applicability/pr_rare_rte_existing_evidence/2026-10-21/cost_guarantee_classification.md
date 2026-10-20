# 費用上界・解析的保証・経験的推定の分類

2026-10-21 JST。既存の主張・実装・保存結果を分類した。新しい数理的上界を採択していない。

|ID|種類|既存の内容|成立範囲|限界・不足|正本|
|---|---|---|---|---|---|
|MATH_NORMALIZATION|既存数学的関係|B_infinity≤exp(τ²)、finite B_K|normalization/attenuation|compiled費用ではない|[CONVENTIONS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_conventions.md)|
|MATH_RESIDUAL|既存数学的関係|step/occurrence/round Taylor residual|operator/signal誤差|RZ/CXの確率的tail上界でない|[CONVENTIONS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_conventions.md)|
|FINITE_LAW|法則の定義＋実装|a_n/B_K・component iid・multinomial|固定tail/time/r/Kのモデル|SOURCE実装/seed分離と数学的独立性/丸め保証を分ける|[RTE_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trotterlib/rte.py)|
|EXACT_EVENT|保存exact有限model期待値|全event列挙×実compiled metric|旧H4 K0単event|指定P full-wrapper全trajectoryの期待値ではない|[EVENT_EXACT_SAVED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_v1.json)|
|MC_WHOLE_WRAPPER|経験的推定|無条件paired標本平均・SE|固定8条件と固定shot|未観測層の母平均保証でない|[H6_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json)|
|CONDITIONAL_PILOT|経験的条件付き推定|s2/multi/s4小標本|旧p5またはp7 q4/8等|別Pへ移送しない。部分supportのn8で全support上限にしない|[B_DIAGNOSTIC](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/b_diagnostic_scope_summary.json)|
|SENSITIVITY|解析的条件付き感度|未知μを変数にした逆転境界|旧H6/H8保存S0/single点等に条件付く|逆転境界を未知費用の実測値/厳密上界にしない|[H8_RARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md)|
|BOOTSTRAP|経験的再抽選診断|保存経験分布/累積列|H4同P、H8旧union|未観測層を再現しない。分位範囲≠厳密CI/selection-adjusted coverage|[H4_MC](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/docs/research/track_a_h4_pr_cost_variability_20261019.md)|
|SAMPLE_MAX|保存標本の最大値|group min-max/観測maximum|保存trajectoryのみ|全support compiled boundでない|[H4_MC](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/docs/research/track_a_h4_pr_cost_variability_20261019.md)|
|NATIVE_GUARD|実装上の制限|native instruction/最大request/transpile/size guard|当該API・成功枝|guard通過・全support正常終了の証明でない|[H6_RARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)|
|THEORY_DP|固定分解の解析的モデル|A/B/native mixed有限law期待値・phase lower/upper|明示loweringとbasis再利用政策|Fractionは入力丸めを消さない。実Qiskit upper boundでない|[THEORY_REUSE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_v2_existing_theory_reconciliation_20261016.md)|
|L1|固定profile下の条件付きコード事実|最終RZ≤完全basis translation直後の総gate数|target/error/routing等なしの特定pipeline|native命令数/理論RZへ置換不可。今回Pへの仮定結合未完|[TIER_R](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md)|
|L2|一部静的確認・未充足|固定mapのtemplate加法性|同じreplacement map/前処理に条件付く|全native t(g)、CRZZ/CCX再帰、mutable library、Rust/runtime、alphabet未確認|[TIER_R](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md)|
|TIER_R|未採択・未評価|law×正当化したtranslation重みの期待費用案|必要仮定確定後の候補|数値上界なし。今回新上界採択・数値評価しない|[TIER_R](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md)|
|CLUSTER_Q_PROXY|解析的proxy＋局所経験検証|connected-cluster・q依存費用fit/holdout|指定snapshot/範囲/compiler|局所5%/2%診断をrigorous boundにしない|[COST_HISTORY](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_compiled_cost_validation_summary.md)|
|PF100|条件付き保存値診断|重複100scope/unique40のPF単価政策比較|旧確認union8/6 n32|PR rare expectation/P再確認/全候補母平均保証ではない|[Q100](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_pf_q_model_evidence_integration_20261020.md)|

## exact期待費用の保存例

旧H4、1 Å/STO-3G/DF12/L_D3/δ=.1/refinement4/K0、controlledなしの単eventについて、`single_event_exact`に`enumerated_event_count=218`、`rz_count=289.24012667527643`、`cx_count=84.98664101463322`、`event_probability_sum=1.0000000000000009`、`standard_error=null`が保存される。[EVENT_EXACT_SAVED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_v1.json)。これは固定有限model・固定compilerでの数値確率和であり、exact rational/interval arithmeticやfull Hadamard wrapperの全trajectory期待値ではない。表中の「exact」はそのmodel内の列挙を指す。

## 理論gate countをcompiled RZ上界にしない具体的理由

既存7候補比較では理論モデルとnative explicit loweringが一致しないmetric行があり、同一eventでもtheoryとcompiledは別指標だった。実compiled/theoryの観測比を全supportの係数上界にはしない。controlled-RZZのnative CCX–CRZ–CCX経路と理論CX–CRZ–CX経路の差、XXPlusYYのrule選択、identity/scalar phase、Gaussian basis inverse、event境界reuse、mutable libraryを固定する必要がある。[THEORY_COMPARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e454b91185ed7b0f92a3790f800ba5d65d5f156/docs/research/track_a_pr_theory_vs_compiled_completed_20261018.md)、[TIER_R](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md)。

L1はTの定義を完全basis translation直後に限定する。L2が未完のままpretranspileの高水準命令数をTへ代入しない。T自体の未観測support上限/期待値が保存・証明されていない以上、この調査で新たな数値Tier Rを出すことはしない。
