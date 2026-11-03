# 2026-10-10 Track A H4 v5後レビュー案とH6契約案

H4 v5の完了・保存監査を受け、利用者の継続指示により、科学レビュー案、不足検証一覧、H6技術pilot契約案を作成した。
完了済みv5 execution worktreeの2,510 filesをbyte一致で別worktreeへコピーし、元証拠を保持した。
既存source・保存JSON・一次資料を静的に確認した。新しい科学計算やsource実装はない。

H4のscopeはlinear 1.00 Å/STO-3G/legacy DF rank12/generation-prefix/T=0.8、
prefix0/6/12、q1/q4（δ0.8/0.2）、保存state/36-dimensional sector。
8 correctness/28 wrapperの完了を受け、四次q4の費用欠落は閉じたが、総u・精度・shot/総費用は未確定とした。
primitiveの±0.2検査は全stage時間の証明でなく、expm/eigh差や残差もroundoff上界ではない。
既存shot helperと三状態判定は再利用可能だが、u-aware shot接続と数値証拠の区分が必要。

H6案はq1/q2へ縮小し、B0を加えた7 cell・36 wrapper・4 trajectory・8 occurrence。
共通tol1e-8は旧提案を維持した未採用案、actual rank/state/hash/資源は未固定。
400次元sector referenceと既存full-vector native作用・sector finite作用を使う案を示し、full denseの単純拡張を除いた。
技術GOと精度・本比較GOを区別する案は独立レビュー待ちで、旧契約へ例外を自動適用しない。

判断は `H4_POST_REVIEW_DRAFT_H6_NOT_AUTHORIZED`。旧RQ・結果・認定状態は変更しない。
science source157 hashes、旧execution inventory649 entriesと元worktreeを保持する。
H4再計算、Hamiltonian/state生成、signal、sampling、compile、GPU、Track B変更、commit/pushはない。
次は独立科学レビューでuの証拠強度、必要H4検証、H6案と本比較条件を判断し、今回はmandatory STOP。

[科学レビュー案](../track_a_ax2b_h4_post_scientific_review_v1.md)、
[不足検証一覧](../track_a_ax2b_h4_validation_gaps_v1.md)、
[H6契約案](../track_a_ax2b_h6_pilot_contract_draft_v1.md)を記録する。
