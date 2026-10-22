# H2共通native・共通allowance：技術基盤

H2/ε_sig=0.0005のFull S2/Y4・PR S2/Y4固定4候補について、隣接complex QR preparation、signed-piece nativeの固定SOURCE、共通診断schema、旧A由来の適応allowance規則を技術準備した。

状態：**人工36テストPASS、静的metadata照合完了、科学実行準備未完、mandatory STOP**。nativeはmock operation adapterと静的call recipeまで実装し、実科学backendは未接続。共通nativeとallowanceは正式採択前。科学的な順位・逆転原因・比較Cの成立は更新していない。

- [全体報告・不足・承認事項](../../artifacts/resource_applicability/track_a_h2_common_native_allowance_preparation/2026-10-23/REPORT.md)
- [native/Gaussian/compiler設計](../../artifacts/resource_applicability/track_a_h2_common_native_allowance_preparation/2026-10-23/NATIVE_DESIGN.md)
- [allowance入力・単位・欠測規則](../../artifacts/resource_applicability/track_a_h2_common_native_allowance_preparation/2026-10-23/ALLOWANCE_SCHEMA.md)
- [Gate 0〜4の非実行契約案](../../artifacts/resource_applicability/track_a_h2_common_native_allowance_preparation/2026-10-23/EXECUTION_PLAN.md)
- [実装](../../src/trotterlib/h2_common_native_policy.py)、[draft検査CLI](../../scripts/prepare_h2_common_native.py)、[固定Git/AST監査](../../scripts/audit_h2_common_native_preparation.py)、[人工テスト](../../tests/test_h2_common_native_policy.py)

前段で不在だった旧H2既知metadata3原本は今回存在を確認し、成功recoveryのcontrol/resultも受領projectionと一致した。調べた2候補worker直下のQPY/vector受入証拠は不足したまま。PR両pathのvector/work/full-stage norm/closure、共通reference結合、実runtime sealは今後の科学承認に必要。今回の実H2回路build/compile/simulation、sampling/signal/reference評価、新u/N/G評価は0。

次の担当：ユーザーが共通native・適応allowance・reference再利用と追加診断範囲を判断する。Codexの実科学runner接続・exact budget確定と科学実行は別campaign/別承認であり、自動進行しない。
