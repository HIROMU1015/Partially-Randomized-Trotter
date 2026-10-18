# 技術検査・保存値転記の検査記録

主担当Codexによる確認。AI subagent、別担当の独立review、immutable CIではない。

- 人工unittest：26/26 PASS、fail/skip0。文字列のexact保存、SE転記、ID/軸/cohort/実測model一致、
  既知A82欠測の維持、元H4 stage維持とD overlay、SHA不一致拒否、real-data描画拒否、既存output拒否を検査した。
- 保存実結果の技術転記：32比較、64軸、説明62軸。保存CSV間の同名field等値とID一致を照合。
  G・比率・積・和・SE・log・ceilの評価やランキングはしていない。
- Matplotlibの人工描画：2 panel群をSVG/PNGへ生成。両方のPNGを主担当が目視し、ラベル／modelの中抜き／
  人工データの明示を確認。実分子値をrendererへ渡していない。
- 初回人工previewは既存venvのinstalled trotterlib優先でimport失敗。初回logを保全し、図表用の純粋moduleだけを
  file指定で読むよう修正した。科学packageの初期化を避け、最終26テストと人工previewをPASSした。
- A82初回metadata準備はTのfloat／文字列のschema違いで停止。文字列"0.8"のexact照合へ修正し、失敗記録を保全。
  実科学数値解析・有効grant発行・科学retryは0。

CPU/RAM/workerへ新しい固定上限は設定していない。空き資源をread-only観測し、小さいscalar処理と人工描画を
独立したown processで実行した。共有OS/cgroup/venv/package/cache/driver・他user/jobの変更は0。
Matplotlib cacheは今回専用領域、pyc writeはprocess単位で無効化した。
