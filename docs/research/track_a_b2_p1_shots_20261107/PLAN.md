# B2 P1：固定保存18roundのshot会計

2026-11-07 JST。ユーザーが全18roundの必要shot数・整数集計を明示承認済み。
起点はbranch/energy結果d424bb8bec1aaf1224a33513a06e64801749a69f。
信号結果d7a55ac181de71cd2b2c49644810373cafd0a450、endpoint原結果ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5を再利用する。

P1の入力、r4/K2/LD3/delta.02、全18round、6誤差cap、u=1e-8は不変更。
保存boundsを既存B2SignalRecord.boundsで照合し、account_radius_roundへそのまま渡す。
rho_raw=A*(rho_PF_lower−corrected_disk)、beta_stat=.24、各軸alpha=.05/36を維持。
既存Hoeffding式とceil→Python intを使用。cost=Noneで価格・費用を計算しない。
全点適格の場合だけ整数合計・寄与を算出し、CSV/JSON再読込のDecimal和でも照合する。
alphaの指定値1/720を36軸へ配分した厳密有理数和1/20も記録する。

新simulation、branch再解析、private入力、sampling、回路build/compile、価格・baseline・最適化・詳細レビューは0件。
SOURCE/人工testを実行前commit固定し、実際のshot算術を行って公開後mandatory STOP。
出力はengineering allowance、理想fresh IID・bounded outcomesに条件付く十分shot数でnot_certifiedを維持する。
