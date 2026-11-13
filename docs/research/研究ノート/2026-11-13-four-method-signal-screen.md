# 2026-11-13：4方式signal screenの技術準備

確認済み：B menu `51500b724a75f9535a2c831d50bcfe09eabb9326`を変更せず、新20 physical candidateと保存40セルのlocatorを固定した。既存A branch、基点`97146481f7b05e9418d8fc016f133c7b78d93d08`を継続し、B/C/Dの作業には変更しない。[REPORT](../h4_four_method_signal_screen_preparation_20261113/REPORT.md)と[有限計画](../h4_four_method_signal_screen_preparation_20261113/SCREEN_PLAN.md)が今回の正本。

Full PF/B0Iはsigned macro-stepのmatrix power、Full-H RTEは全非scalar（one-body含む）のgrouped spectral powerを再利用する。Full-Hの明示q/Rは保存κやDF B3で代替せず、raw/correctedを独立伝播する。PFはB=1によりraw=corrected。保存reference pairが3時点とも存在し、新referenceは0 actionの計画とした。

技術検証：`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B tests/check_h4_four_method_screen.py -v`で9件、`python3 -B tests/check_h4_pr_physical_inputs.py -v`で8件、計17件PASS。旧依存SOURCEのinvalid-escape DeprecationWarningを保持。人工行列・mockの照合であり実Hamiltonianの性能実証ではない。

未確認：実20セルのsignal・bias・attenuation・数値安定性、全RMSE資格、候補の優劣。今回、新科学計算・保存数値再解析・N/C/Gは0。次はユーザーが固定20 mean＋6 normalizer/local診断、reference0を別途承認し、Aが実行、Bが後続候補案へ接続する。公開照合後mandatory STOP。
