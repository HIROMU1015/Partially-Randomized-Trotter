# 技術検査と科学gate

専用Python `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B`。

| test | 結果 | 範囲 |
|---|---:|---|
| check_pf_shortlist_mean_completion.py | 6 PASS | 固定48/30/10・early actual幅、保存値/SOURCE保持、C.executeで48のみcall、scalar、foreign step拒否、FAILで後続停止 |
| check_saved_pf_mean_staging.py | 9 PASS | 既存staging30値、候補/価格対応・scalar・未取得未昇格・SOURCE保全 |
| B check_rpe_total_baseline_binding.py | 7 PASS | 既存intake/保存/人工baseline binding |
| B check_rpe_fixed_grid_total.py | 6 PASS | 既存total-view/人工fixed grid |
| check_pf_shortlist_complete_b.py | 6候補PASS | 実packetのT.create/T.bind_signal、13round完備・単一SOURCE/identity/law/provider binding、mean保持、science_calls0 |

人工・回帰test計28件PASS。新6件はnp.loadを禁止した人工/mock・metadataであり、実H4のmean再計算ではない。B testも元の人工fixture範囲のみ。

実科学はSOURCE fb9a1878b固定後のfresh run01一回。6worker/14.2922秒、48meanすべて局所gatePASS、保存30copy、FAIL/UNKNOWN0、retry0。step/power/norm gate1e−9、pair1e−10を変更していない。最大残差はREPORT/ROUND_SUMMARY/rawに保存。

B検査はA所有の一時detached checkout @9c0e12d87で無変更実行。MSE/DP/risk/shot solverを呼ばず、B active worktreeは変更しない。これは作者自身の技術確認で、独立査読・外部再現・全RPE保証ではない。
