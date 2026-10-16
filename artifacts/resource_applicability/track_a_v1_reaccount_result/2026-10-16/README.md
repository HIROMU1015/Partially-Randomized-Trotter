# V1 5精度再会計：完了STOP

[結果・条件・限界](../../../../docs/research/track_a_v1_reaccount_results_20261016.md)。通常bias・固定費用モデルの条件下で、最良対照の強化によりε=.05でPRの点費用優位が反転。非相殺診断の全domain最良は未解決を維持する。

- [全60比較CSV](comparisons.csv)、[全比較・family最良・報告trigger JSON](comparisons.json)。
- [格子・S4追加の費用変化](grid_effects.csv)。
- [全4320会計行、可逆gzip](candidate_accounting.json.gz)、[元bytes/SHA・完全復元receipt](accounting_export_receipt.json)。
- [実測/model/確認・探索PRを区別した410軸行](axis_costs.json)。
- [探索72/n8の別scope参照](exploration_reference.json)。
- [結果集計](result_summary.json)、[driver終了](execution_terminal.json)、[owned launcher終了](launcher_terminal.json)。
- [SOURCE/runtime/input/653旧ファイル保全](completion_audit.json)。
- [stdlibによる保存結果再照合](verify_public_results.py)。`python3 -B verify_public_results.py`で実行できる。

SOURCE2261a3ad…、旧表2170照合一致、会計実評価4040、壁時計6.024秒、native exit0、全回収。新signal/primitive/native回路/compile/fit/sampling/V2は0。実測をmodelで上書きせず、モデルの全grid誤差保証・厳密bias保証・母集団最良は主張しない。mandatory STOP・次Phase未承認。
