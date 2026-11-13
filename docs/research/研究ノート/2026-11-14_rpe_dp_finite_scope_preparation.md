# 2026-11-14：実H4 DPの部分追跡範囲を具体化

利用者が追跡範囲を提示してから実H4の科学実行を別承認する方針を指定したため、実rawへの適用を行わず、既存06db9aefeのcode/57件の検証記録を確認した。組数不足で後半round全体を残余へ落とすことを避けるため、保持数Kに対する全roundのcount直方体上限を定義した。

中心案はK8192、P141,559,036/構成、LD1/LD2各1件だけ。固定count位相格子ではm0..2を全包含し、全roundの保持状態を伝播する。人工13round/N8でK4096/8192、人工prefixN64/60でK8192をbenchmarkし、約18/31/203秒を記録。実H4の性能又は残余の予測には昇格せず、整数bit長増加から数十分〜数時間の可能性を見込む非保証の実行計画とした。

[REPORT](../rpe_dp_finite_scope_preparation_20261114/REPORT.md)／[有限JSON](../rpe_dp_finite_scope_preparation_20261114/FINITE_SCOPE.json)／[HANDOFF](../rpe_dp_finite_scope_preparation_20261114/HANDOFF.md)。既存科学条件は再採択させず、新追跡範囲と2件の科学実行だけをユーザー判断事項とする。新kernel/callerやshot optimizerは不要、元SRC/raw/結果/STOPとA/C/監査worktreeは保全。実H4新MSE/追加shot/MC/C/G0、公開照合後mandatory STOP。
