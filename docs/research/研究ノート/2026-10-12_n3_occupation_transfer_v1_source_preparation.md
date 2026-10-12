# 2026-10-12 N3 OCC v1 source preparation

完了済み科学レビューのGOを受領し、実装・synthetic検証・source/plan固定・公開までを実施する。
固定8入力の科学実行は未認可。root/既存source/results/STOPを保持する専用worktreeを使用。
旧証拠のc_star継承、raw/same-method inheritance/fallback/operationalを分離。
κ0=NORM、ν1=旧族の仕様へ訂正。下界・各方法・separatorの予約capとcost DAGを固定。
外部固有値和の最大valid lowerを6候補内で選び、OCCは同6候補内でaffineまで検証。
NumPy候補とexact PSDを分離、独立checkerはstdlibで元行列LDLを再構成する。
24 synthetic testsと87765 saved checks PASS、2workers/各numeric thread1をsyntheticで確認。
未commitの初期synthetic packetをdevelopment_before_*へ保持し、現在sourceのpacketと区別。
CPU32/affinity32、memoryを確認。pyenv bare pythonにはNumPyがないため既存の
/tmp/prt-representation-venv（Python3.12.3/NumPy1.26.4）を使用。GPU driver接続不能、GPU0。
実装は原理・数式を変更せず、詳細な同内容科学レビューを再要求しない。
次は実行plan準備確認と明示認可。N1停止・N2保留・中心テーマnull。
[報告](../n3_occupation_transfer_v1_source_preparation_20261012.md)、[追補](../n3_occupation_transfer_amendment_v2_20261012.md)。
