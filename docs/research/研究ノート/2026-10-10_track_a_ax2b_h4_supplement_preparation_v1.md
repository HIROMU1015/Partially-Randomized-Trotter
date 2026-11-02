# 2026-10-10 Track A：H4補完準備 v1

ユーザー提示の[GPT時間上限STOP後レビュー](../track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md)
§§10–14,19に従い、[新source・補完準備](../track_a_ax2b_h4_supplement_preparation_v1.md)を固定する。
旧6 correctness/12 MP、原STOP、source/freezesは保全する。
cell/dps-local MP演算子cache、S4 2 cellとexplicit 4群の独立単位、原子的進捗保存を追加した。
122 local synthetic/metadata tests pass。分子H4の再評価・速度測定、sampling、回路build/compileは今回0。
科学target・stage列・MP80/120・occupation独立構成を維持し、旧sourceは編集しない。

新source `67aa6bb54dd5385eb3c56def1b6052da12643447`、180 science closure/2 validationをfreeze。
各unitの対象・予算・新outputを固定するが、CPU/resource割当と別grantは未確定、metadata sealは未実施。
`H4_SUPPLEMENT_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
N/G=null、u未認定・UNDETERMINED。H6の既存実装は再利用候補として維持し、入力生成もpilotも開始しない。
