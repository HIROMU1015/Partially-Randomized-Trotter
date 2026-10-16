# Track A V1 技術準備（2026-10-16 JST）

利用者の「作業を始めて」「続きを行って」により、[V1実装・人工検証・実行準備](../track_a_v1_h6_deterministic_readiness_20261016.md)を専用branchで進めた。
新科学計算は認可範囲外なので0、旧grant/ledger・H5を保持。数値結果や厳密保証の採択は行わない。公開・remote照合後mandatory STOP。
## Phase Aの準備継続とV2保留

利用者が並列化を重視し、実行承認前のPhase A準備を指示。その後V2は指定保存commitとの読取専用照合へ変更。V1のscience条件は保持し、最大16worker・共通reference/spectra・prefix単位共有を実装、44限定人工tests PASS。新科学計算・新grant0。V2のH6基本416行は既存832行に含まれ再計算不要。混合政策・較正等の不足は[別報告](../track_a_v2_existing_theory_reconciliation_20261016.md)へ。既存SOURCE/results/grant/ledger/H5を保全し、V1承認待ち・V2照合報告後STOP。

## Phase A run01の起動と停止

「V1 Phase A を開始して」で承認。SOURCE69e99143…・新grant01、source15/provider230/runtime34,273/input5と資源gate通過後、launcherのOBSERVER_RSS_CEILINGで約0.027秒STOP。driver exit -15/全回収、科学lock/output/予約なし。既存grant/ledger等644不変。pinned stdlib診断は停止を再現せず、本番実測値未記録のため原因未確定。再起動・上限変更・次Phase移行なし。[停止報告](../track_a_v1_phase_a_run01_stop_20261016.md)を保存してmandatory STOP。

本番の再実行をせず、RSS停止時の現在RSS/VmHWM/ru_maxrss/ASをterminalに残す監視修正を準備。exec前lifetime peakを現在executableのVmHWMと区別し、上限値は不変。47人工tests、pinned Pythonの影響17件PASS。別領域のconfig草案は未承認、新再実行grant/CPU/command0。run01原因は未確定のまま、mandatory STOP。
