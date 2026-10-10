# 2026-10-10 Track A H6：保存DF入力完成の実装準備

[独立レビュー](../track_a_h6_df_hermitization_independent_review_2026-10-10.md)採用後、ユーザーの「次の作業に進んで」に基づき[準備source/検証/seal](../track_a_h6_saved_df_completion_preparation_v1.md)を固定。
source d2d45235724d1b956fb250c11a87985ed9200072、science194・validation3、synthetic122 passed（新41/旧38/旧43）。旧sourceや結果は変更しない。
判定余裕1%、summary/係数照合を結果前に固定。既存loaderの無重みgateを迂回せず新policy-bound loaderを用意。
DF受理・state生成の実データ実行はまだ認可されていない。CPU2/1worker/BLAS1、phase60/120/900・total1080秒、AS8GiB/output32MiB/matvec10000の新対象を提示。
旧診断grant消費済み。新grantなし、real H6 raw数値decode/受理/state/sampling/compile0。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
