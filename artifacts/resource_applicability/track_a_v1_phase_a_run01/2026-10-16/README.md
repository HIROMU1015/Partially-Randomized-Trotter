# V1 Phase A run01：科学予約前STOP

[実行・停止・診断・保全報告](../../../../docs/research/track_a_v1_phase_a_run01_stop_20261016.md)。

- [承認・SOURCE/config/grant binding](execution_binding_v1.json)
- [fresh gate要約](fresh_gate_summary_v1.json)
- [launcher終了・回収](launcher_terminal_v1.json)
- [644既存ファイル不変・科学予約0](preservation_v1.json)
- [人工stdlib memory診断・原因未確定](memory_diagnostic_v1.json)
- [監視修正の限定人工検証](observer_fix_tests_v1.json)、[別領域の未承認config](observer_fix_preparation_v1.json)

`OBSERVER_RSS_CEILING`、driver exit -15、全回収。科学lock/output/予約なし。
本番RSSの数値は保存されておらず、root cause未確定。再起動・上限変更・V2/compile/次Phase移行0。
監視値の保存とexec前peakの区別を修正し、47限定人工testsとpinned Pythonの影響17件PASS。再実行用grant未発行。
NPZ・raw config・grant本文・絶対private path・cacheは公開対象へ含めない。
