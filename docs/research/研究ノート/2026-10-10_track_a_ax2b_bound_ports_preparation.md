# 2026-10-10 Track A：H4限定検証・H6 backend接続準備

利用者の「作業を進めて」を受け、[独立レビュー §21](../track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)のsource/synthetic作業を継続した。
[v2準備報告](../track_a_ax2b_bound_ports_preparation_v2.md)にsource、記録schema、入力/実行gate、上限をまとめた。

H4の独立occupation/80・120桁MP、native/Horner stage endpoint、signed bias、explicit eventのphase/basis/register/測定axisを接続した。
H6は独立したsector reference・full-vector native作用・bounded7 cell/36 wrapperのbackendと、stdlib fail-closed launcherを追加した。
H6 `target.T`のbackend渡し忘れをmock通し検査で検出して同値接続した。科学条件は変えない。
test globがphase markerまで数えた失敗と、negative-header test自身をguardで遮断したfixture失敗も保存監査へ記録する。

source/testing receiptを別inventoryへ追加。旧v5科学source/results/freeze、172 preparation hashes、科学manifest、Track B、既存dirty/untrackedを保全する。
今回のtestsは合成行列・mock request/compiler・dummy processのみで、分子科学証拠ではない。
入力生成・actual coverage・資源割当・別grantは未固定。H4-E専用300秒を加えたtotal3000秒も未割当proposal。
新科学計算/sampling/circuit build/compile0、`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
