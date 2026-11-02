# V1 Phase A 並列実行準備（未承認）

[実行範囲・資源・gate・STOP](../../../../docs/research/track_a_v1_phase_a_parallel_readiness_20261016.md)。

- [固定plan](phase_a_plan_v1.json)、[SOURCE/input/profile binding](preparation_binding_v1.json)、[静的bytes確認](static_preflight_v1.json)
- [限定人工検証](limited_tests_v1.json)
- 実装：[runner](../../../../scripts/track_a_v1_runner.py)、[独立observer](../../../../scripts/launch_track_a_v1_phase_a.py)、[owned workers](../../../../scripts/track_a_phase_a_parallel.py)、[signal task](../../../../scripts/track_a_v1_parallel_signal.py)
- [tests](../../../../tests/test_track_a_v1_phase_a_parallel.py)

新科学計算・新grant0、CPU[]、approved/runtime false、command null。旧config/grant/ledgerを変更しない。
one-shot、失敗予約の返却なし、自動retry/resumeなし、Phase終了時mandatory STOP。
