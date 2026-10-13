# 2026-10-13 C1固定pilot一回実行・結果保存

ユーザーは固定4入力/940eigの一回実行と独立監査・結果公開後STOPを明示承認した。
直前の中断ではread-only preflightまでだったことを確認し、新しい認可JSONを別commit `9c3b5c82942edfef3df27aeadf41a2299dc39270`へ保存した。
source `ddaa05d2b51d575c3cfea1194bbfdffa182f9636`、入力・契約・plan・runtimeは不変。
IDE Active file/Open tabsで対象をD1/D2へ変えず、会話で継続しているC1を実行した。

940calls一回/retry0完了、936unitary/8064比較行の保存独立監査PASS、追加eig0。
[結果・限界](../qc_c1_macrocycle_pilot_v1_results_20261013.md)、[次の科学判断](../qc_c1_macrocycle_pilot_v1_review_handoff_20261013.md)。
通常state16比較で全3selectorの5%改善0、bad-state16比較はeligibleなし。全4caseの真kernel dは正。
jointのexact coefficient-only PF precision hは6/32valid、26/32null。known-composition同値性を独立新規性と数えない。
非分子2qubit/39methods/6h/model target10^-5/10^-6/価格3,6という限定scopeで、総QPE/RPE資源・分子性能はUNKNOWN。

2workers/1thread、runner8.73s、BCH/情報/selector8.03s、case matrix約0.11–0.12s、saved checker1.42s。
前処理が主要負担として判明したが、凍結sourceへ変更や追加性能runを行わない。
20locked files/旧229files/planはbyte保持、既存global description FAILを無断修正しない。
一回認可消費済み、結果・監査・remote取得公開後STOP。N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了を維持。
新しい科学的継続判断は別のGPT開始承認へ戻し、C4や追加caseを自動起動しない。
