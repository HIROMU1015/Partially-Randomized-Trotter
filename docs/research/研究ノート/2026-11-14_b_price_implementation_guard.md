# 2026-11-14：Bの直接cost/compare/select経路にCの価格実装guardを接続

利用者の技術実装指示に従い、Cの`price_implementation_guard`（`3a8b37f1a`）を、Bの直接経路に接続した。

- **迂回点**：変更前の`rpe_shot_allocation.compare`は、extensionの自己整合だけを見ていた。旧Full-H汎用RZZの価格に新loweringのラベルを付けると、`B.cost`まで進んだ。controllerの`B.cost`と`B.select`は、実loweringを検査しない。
- **修正**：Cのguardを固定blobからそのまま実行し、C API hookだけをBのlive controllerと固定D adapterに結び付けた。
  - `compare(root, comparison_policy)`は、すべての価格算術より前に検査する。
  - policyなしの科学N packetはpendingになる。
  - guard付きの`cost`・`select`入口を追加した。
  - 算術、N資格、canonical ID、元recordは変更していない。
- **検証**：すべて人工データと回帰testで、実データの科学計算は0件。
  - 新規test 9件。
  - 本branchの回帰31 file／349件PASS。
  - B head＋patchでは、既存のI2の5 file失敗だけが残り、失敗内容はpatchの有無で同一。
  - Cの51件PASS。
  - C経由とB直接経路の照合23 caseがすべて一致。
- **残課題**：
  - controllerを直接呼ぶ経路は、算術primitiveとして残した（固定DP runnerの保護対象であるため）。
  - 実sidecarとpolicyの採択は未了。

[REPORT](../b_price_implementation_guard_20261114/REPORT.md)／[HANDOFF](../b_price_implementation_guard_20261114/HANDOFF.md)。

B本線には適用していない。公開とremote照合の後はmandatory STOPとする。
