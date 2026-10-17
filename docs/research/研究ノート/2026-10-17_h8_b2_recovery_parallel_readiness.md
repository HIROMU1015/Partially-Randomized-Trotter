# 2026-10-17 H8 B2修正・並列化

利用者の修正・並列化指示に基づき、新規科学計算を開始せず、残り2軸の新scopeを実装。原STOPとB0保存2軸を保全した。control/fingerprint/QPYを各2worker、compileは2%判定を挟み直列1。親大回路解放・PSI raw保存・独立native wait・marker集計修正を接続した。

34人工testsと2 qubit QPY/旧新compiled一致を確認。5core/112.3125GiB案、SOURCE77ce12d8c935702dc78f294b5863b4904f9c6a2c、31source/265provider/34,273runtime/16inputをseal。無効grantだけ、H8新compile/fit0、C未承認。

[詳細と証拠](../track_a_h8_phase_b2_recovery_parallel_readiness_20261017.md)。公開後mandatory STOP。
