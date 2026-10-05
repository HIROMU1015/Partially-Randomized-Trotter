# Track B BF-1 assembly guard revision

2026-10-05 JST。

`296ec7e4c025f09e4bfda96e56e08d32388b81c5`に対して利用者から
`PROCEED_TO_FINAL_BF1_EXECUTION_REVIEW`を受領した。source hash/environmentと既存33 testsは通過したが、
local final reviewの追加1×1 synthetic witnessでRe bias errorがreported u_signalを超えた。
判定は`REVISE_NUMERICAL_ASSEMBLY_GUARD_BEFORE_AUTHORIZATION`。これは独立signoffではない。
旧reviewとそのwitnessを保存し、誤差伝播だけを最小修正する。

利用者の次段要求に従い、各generator/scalarのassembly budget、full-target合計、
signed unitary/finite factorのLipschitz・normと逐次error伝播を明示した。
identity抽出で使用したactual eigensystemのresidualも同じsourceで検査する。
[v3 amendment](../../tracks/algorithm_codesign/bf1_assembly_guard_revision_v3.md)に条件付き導出と実装対応を記録する。
新しいsynthetic regressionとv3 source-content packetを作成し、v1/v2・domain・旧witnessは保持する。

RQ、family、input、32 evaluations/arm、O/L/F、q/R/K、allocation、primary 5%、上限、
cross-objective診断、source→authorization-only child方式を変更しない。
Track Aのartifact/status/source・共有APIは変更しない。
science inputを開かず、BF-1、正式authorization、commit/pushは未実施。
改訂source固定・最終review・明示的実行指示の後だけ一回のBF-1を行い、全case停止して全面再評価する。
