# H2共通native・共通allowance production基盤

固定H2・ε=.0005のFull/PR×S2/Y4に対し、旧mockを保持して別のproduction入口を実装した。固定科学kernel、instrumented両path診断、旧reference再結合、適応allowance、会計、one-shot/ledger/import sealを接続した。

状態は**人工72件PASS／科学実行0／binding・政策・PR C witnessはBLOCKED／mandatory STOP**。productionのH2数値動作は未確認。

- [全体報告・承認対象](../../artifacts/resource_applicability/track_a_h2_common_production/2026-10-23/REPORT.md)
- [実装とprovenance](../../artifacts/resource_applicability/track_a_h2_common_production/2026-10-23/IMPLEMENTATION.md)
- [科学callのexact契約](../../artifacts/resource_applicability/track_a_h2_common_production/2026-10-23/CALL_CONTRACT.md)
- [CLI](../../scripts/run_h2_common_production.py)、[control](../../scripts/h2_common_production_contract.py)、[backend](../../scripts/h2_common_production_backend.py)、[診断](../../scripts/h2_common_production_diagnostics.py)、[静的準備](../../scripts/prepare_h2_common_production.py)、[人工テスト](../../tests/test_h2_common_production.py)

次：ユーザーがnative/allowance/reference・診断定義・最小科学範囲を判断する。Codexによるruntime/input binding完成と科学実行は別指示/承認の範囲であり、自動開始しない。
