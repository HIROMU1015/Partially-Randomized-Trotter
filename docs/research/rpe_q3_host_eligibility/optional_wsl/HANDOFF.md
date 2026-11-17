# local0-wsl 任意参加worker

[REPORT](REPORT.md)に固定SOURCE・通常入口・全16セルの割当表・次回の配置方法をまとめた。
[機械可読の割当表](HOST_ALLOCATION.json)と[検査結果](CHECKS.json)を参照。

全16セル実入力dry-check PASS。WSL受付OFF・上限1・required_hosts追加なし。
既存12セルとWSL PR-DFを再利用し、不足3セルだけ実機検査した。実H4 Q3・本番queue投入0件。

次回Join前に、生成WORKER_PROFILEと固定共通transportを新しい専用directoryへ配置する。
旧PROFILE.current・spoolを保存したままなので、旧profileへのfallbackはしない。
