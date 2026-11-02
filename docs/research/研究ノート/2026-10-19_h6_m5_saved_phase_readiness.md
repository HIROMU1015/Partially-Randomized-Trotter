# 2026-10-19 H6 full m5：保存phase診断の実行準備

起点a6032394の計画を、新branch `track-a-h6-m5-saved-phase-readiness-20261019`で実装した。SOURCEは`99e8723d402245a9ad923e7dbc0800f47bc575b6`。通常2/拡張2再生、同じ保存low、未知gate/実精度不足/通常非再現はSTOP。原raw1e-9未満の条件・旧STOPを維持する。技術再現atol1e-12は新しい実行契約案として固定したが、実行・正式採択は別承認待ち。

人工24件PASS（2.046秒）、metadata拒否12件、実DRAFT両modeの事前拒否を確認。人工13qubitのadapter/4spawn/reportと、外部wait4・異常停止/reapを含む。inverse人工参照のdtypeを訂正した経緯は報告に記録した。独立レビューではなく作者自身の確認、subagent0。

SOURCE/input/runtime1831fileをsealし、既存684file不変。実H6のQPY/NPZはbyte hashのみで復元0、新科学数値解析/compile/有効grant0。通常worktreeの更新を凍結せず、host PSIだけでkillしない。公開pushをせずlocal commit後mandatory STOP。

次はユーザーが固定SOURCE/planの一回解析を承認するか判断する。Codexによる実行はその後だけ。詳細は[準備報告](../track_a_h6_m5_saved_phase_readiness_20261019.md)。
