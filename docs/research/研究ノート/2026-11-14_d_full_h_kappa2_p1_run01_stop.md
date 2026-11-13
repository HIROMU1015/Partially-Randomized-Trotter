# 2026-11-14：Full-H κ2 P-1 run01のSTOP（科学計算前）

利用者の承認に従い、固定caller（`04bead115`）でP-1を1回実行した。結果は`STOP_WIRING_CHECK`である。

**経過**
- 親processの検査（byte、cell、seed）は通過した。
- 子processの`install()`、provider照合、元`setup()`も通過した。
- 元Full-Hの`context()`がimportする`h46_m5_tol_contract`が、import時に`references/h46_m5_tol_pilot_inputs_v1.json`を読む。私の実行treeのsparse checkoutにこのtracked fileが無く、`FileNotFoundError`で停止した。

**影響**
- 保存入力の復元、sampling、compose、compileはいずれも0件。retryしていない。

**STOP後の技術確認**
- `references/`をcheckoutしたうえで、子processのimport連鎖（12 module）をimportだけで確認した。科学計算はしていない。

**次の判断**
- run02には、次の2点とユーザーの承認が必要である：tracked fileを欠かない実行tree、出力labelの変更（run01は保全するため、caller修正が要る）。

[REPORT](../d_full_h_kappa2_p1_results_20261114/REPORT.md)。mandatory STOP。
