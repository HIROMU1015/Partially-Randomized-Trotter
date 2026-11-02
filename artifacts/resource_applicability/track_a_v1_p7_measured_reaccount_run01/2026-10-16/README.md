# V1 p7実測費用：ε=.05の限定再会計

linear H6/1Å/STO-3G/tol-only DF19/T=.8/δ=.8/B0+I p7 q1 S2/P=0/RZ。通常signed signal bias・経験的allowance、倍率1/10。入力結果commit a73ce484…、実行code commit e078a7e6…。

保存shot cosine10216/sine14356と実測各14492 RZによりG_det=356097424 RZ。保存確認union8/n32のPR G=403423200 RZ。PR/決定論=25213950/22256089、PR削減率=-295786100/22256089 %（−13.29012029%）。両倍率で同じ結果。1候補の保存点比較で逆転を維持し、母平均・厳密bias保証・全候補実測最良は認定しない。

- [報告・式・到達範囲](../../../../docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md)
- [JSON結果](comparison.json) / [CSV結果](comparison.csv)
- [保存値の選択](selected_saved_inputs.json)：plan/shot/旧比較をそのまま抜粋。原planの未承認flagは履歴として保持。
- [承認receipt](authority_receipt.json) / [gate receipt](preflight_receipt.json)
- [終了terminal](execution_terminal.json) / [会計予約](reservation.json) / [終了監査・入力/旧raw hash](completion_audit.json)
- [実行runner](../../../../scripts/track_a_v1_p7_measured_reaccount.py)：defaultはplan-only。実行には別の新grantが必要。
- [元実測](../../track_a_v1_p7_compile_run01/2026-10-16/validation_complete.json)
- [元PR保存比較](../../track_a_v1_reaccount_result/2026-10-16/comparisons.json)

private grant/lock/rawは変更せず保全。公開receiptではprivate絶対pathと他job一覧を省く。比較・予約・終了・監査はprivate原本のbyte copy。算術のDecimal/cross-product照合はprimary agent自身のローカル検証であり、独立レビュー/外部再現/CI全体PASSではない。

SOURCE22/provider230/runtime34273/旧科学入力15/新会計入力8、旧compile raw22、旧grant等663の不変を照合。新compile/signal/reference/scalar/primitive/SCF/DF/state/sampling/shot/fit/候補再選択/V2は0。wall0.010498秒、exit0、mandatory STOP。次Phaseへ自動移行しない。pushは利用者が行う。
