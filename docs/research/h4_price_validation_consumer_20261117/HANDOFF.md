# local0・C同期への引渡し

公開先は従来と同じ `origin / codex/h4-saved-rpe-cost-catalog-20261116`。local0の参照先変更や新しい同期処理は不要。

- HTML：`docs/research/h4_search_map_20261117/index.html`
- 機械可読：同`INDEX.json`の`price_validation_models`と各pointの`price_validations`
- 同期：`bash scripts/run_h4_catalog_sync.sh`（既存10分cron）

PR-Pauli → LD2 →「独立実測あり」の配分詳細から価格検証を開く。m5のq32/R33は独立2経路、X/Y各平均68,264 RZ、差は+0.044612%（予測−実測、分母＝実測）。軸別4 receiptへ固定リンクを持つ。各receipt内のselectorも表示する。

既存の条件付き価格・Q3判定・資格は変えない。m5の2経路を他roundの実測や全law期待値へ昇格しない。検証待ち/欠測のsidecarはQ3や価格会計を妨げない。

各producerは既存controller公開へ`PRICE_MODEL_VALIDATIONS.json`を含める。共通`publish_h4_search_run.py`はこのoptional fileも索引化する。sidecarのみ更新時も通常同期で再表示される。新しい取得workerやfitはC側に取り込んでいない。

[REPORT](REPORT.md)／[検収](CHECKS.json)／[検査log](TESTS.txt)。ロードマップ本体・稼働中run・科学SOURCE・旧STOPを保全。今回の完了点はconsumer公開であり、新科学計算は0件。
