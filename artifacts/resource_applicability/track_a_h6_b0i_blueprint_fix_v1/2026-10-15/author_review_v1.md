# H6 B0 blueprint fix — author review

利用者のサブエージェント禁止に従い、同一担当Codexが差分を確認。独立レビューではない。

SOURCE `6b4d3026bf6c30bea3d74e7ccaeb06bda0e9baaf`。新しいidentity checker、adapter callback、runnerの詳細receipt/error handling、fixture/test/harness、固定metadata6ファイルの差分を確認した。

- 現在profile/input/provider/case/digestの固定を実科学loaderより前に確認する。未認可は従来auth gateで先に拒否する。
- 旧digestを固定参照として保持し、非preparation_hash fieldの差は拒否する。現profile期待値の自動更新/legacy fallbackなし。
- 新checker通過後も元bridgeの完全blueprint digest照合・入力再照合・元prepared mutation guard・一度の予約が働く。
- 数学・scalar/reference/u/shot規則・native lowering・compiler・6compile/12probes/全A/B barrier・並列cleanup sourceは変更していない。
- 不一致診断はbounded metadataだけ。first STOPを保持し、診断IO失敗は別欄へ保存する。
- 新12＋影響70の82人工tests PASS。現profileの3件を2回の限定準備診断で確認。実native build/probe/compile・本計算再起動は0。
- 旧preparation payloadが完全保存されていないため、旧と現basis数値の内部差や全bit同一性は未証明。将来Phase Bの物理correctness gateを必須のまま保持する。

修正と現在profile metadata結合はPASS。実H6の完了・費用取得は未検証、新grant/新launchは未認可。サブエージェント0。
