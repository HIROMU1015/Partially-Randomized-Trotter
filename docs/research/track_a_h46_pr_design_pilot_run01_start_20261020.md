# H4/H6 PR DESIGN_PILOT run01 起動記録

科学SOURCE `de5485d4b7c6de25dd9b6137c9a12bce697242c6`。両系のSOURCE/input/compiler・seed・保存・fresh RAM/CPU/scratch/inode gateを通過し、H4 9/H6 28 workerで各396 trajectory・792軸の一回取得を起動した。

[start snapshot](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_execution/2026-10-20/start_snapshot.json)は起動時点の観測であり、最終PASSではない。各群33本、専用cohort、level1 byte可逆圧縮、元SHAとarchive SHA、paired native検証、exact call台帳を保持する。独立supervisor/controllerが結果保存・終了監査・公開可能metadataのlocal commitまで行い、mandatory STOPする。自動retry・top-up・n9切替え・統計解析・n_E採択はない。

実行開始後のSOURCE・入力・grant・markerは変更しない。途中の一時的なPSI/CPU競合や無関係なworktree更新だけでworkerを停止しない。新条件を他研究へ適用しない。

次の担当：Codexが取得の進捗と技術終了監査を確認する。ユーザーは取得完了後、pilot限定数値評価を別途承認するか判断する。
