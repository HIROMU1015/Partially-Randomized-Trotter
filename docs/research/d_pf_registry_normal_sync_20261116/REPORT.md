# Dの決定論PF registryをCの通常10分同期へ接続

`D_PF_EVIDENCE_REGISTRY_V1`の読込みを既存C同期へ取り込んだ。新しいFull PF・S2・要求幅2hは、**信号13 round・compiled価格26軸あり、Q3待ち**として扱う。配分・Q3判定・MSE・達成費用はnullのまま、既存のp7・Yoshida4・32h/5の評価済み点は増やさず保全する。

[HANDOFF](HANDOFF.md)／[比較表](../h4_saved_rpe_cost_catalog_20261116/run01/REPORT.md)／[機械可読catalog](../h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json)／[通常同期の検収](SYNC_VERIFICATION.json)。

## 固定入力と再利用

Dの指示された公開SOURCEは`352539e2749989b4ddc3ab74c31f486982bfe265`、入口は`docs/research/d_pf_on_demand_evidence_20261116/{HANDOFF.md,C_RESULT_JOIN.json}`。DのHANDOFFに記載されたconsumer差分だけを選択的に取り込んだ。他担当branch/worktree・原結果・旧STOPは変更していない。

- `pf_optimizer_connection.py`はDの同commitとbyte一致で再利用する。Cから呼ぶのは`hydrate/read_source`のmetadata/source joinだけであり、同moduleのcompletion・探索・route生成は起動しない。
- `Catalog.pf_evidence_registry`とschema dispatch、通常syncのschema許可、Q3新規0件かつbestなしのmetadata-only ROUND_RESULTをREFERENCE_ONLYとする条件を再利用した。C既存の`make/result`、Git reader、discovery、比較表、cronを維持する。
- C側にはtemplateのSOURCEと`TEMPLATE_ONLY_NOT_EVALUATED`を明示し、配分容器を評価済み配分へ昇格させない。表示は信号・価格のcoverageとQ3待ちを分ける。
- D readerが使うbaseline physical-IDの実装はC既存版と同一であることを確認した。新readerの変更も通常同期の実装cache keyへ含める。

着手時C HEADは`bd78ae252f92f7c84b6af6ae2a3bad202a5bd1bb`。D branchとregistryファイルは既に検出されていたが、未対応schemaとしてPENDINGだった。公開branchのglobや固定manifestへ利用者が手動でcommit/pathを追加する必要はない。

## 接続する2構成

|構成|configuration ID|Cでの扱い|
|---|---|---|
|Full PF S2・2h|`9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981`|13信号・直接10＋条件付き16の26価格。Q3_WAITING、allocation/N_X/N_Y/q3_status/MSE/conditional_cost=null|
|B0I p7 Yoshida4・32h/5|`dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`|保存準備済み根拠をsignal_only_historyへ保持。既存の配分別Q3・費用・価格SOURCEを変更せず、比較点を増やさない|

新Full PFのconfiguration・13信号・26価格はD registryが指す`9c02f47287a6622e01af00f151cb9776c3c9c83d`の保存成果、Q3_inputsは`f1d576c201e014be070859c7407523e92829268b`のtemplate。価格kind・unit_price_RZ・元receipt/modelの`price_evidence`・compiler profileを軸別にコピーする。価格modelの新fit・予測や総額の算出はしない。条件付き価格の未知長q誤差を0とせず、正式E[C]/G資格を与えない。

## 保存入力の検査

`tests/check_pf_registry_publication_sync.py`：**3 PASS（11.877秒）**。

1. D helperのbyte一致と、通常公開branch discoveryによるregistry検出。
2. 保存C入力57点へregistryを結合し58点へ。新Fullの13/26 coverage、直接10、Q3待ち・null、既存p7行の完全不変・保存費用372035410の保持、方式最安の不変。同じregistryを再読込してcatalog一致。取得用completion hookは呼出し禁止にして確認。
3. Dの保存metadata-only optimizer返却はREFERENCE_ONLYであり、Q3適合点を捏造しない。

人工Hamiltonianや新しい数値producerは使用せず、固定公開JSONと通常同期経路で検収する。検査は作者自身による技術確認であり、独立科学レビューではない。新signal・sampling・compile・fit・Q3は0件。

## 実際の定期同期

2026-11-16 **13:20:01〜13:20:24 UTC（22:20 JST）**の通常10分cronが、実装`9c70a1aeee7ee9179e6e6db79ac2e3a9d048821b`でregistryを検出・受入し、`8b22ca2341f7bdf421fba7a3c018a4ff6d9b281f`へ自動commit/pushした。所要22.085秒、終了0。57→58点の増分は新Full PFのQ3待ち1件のみ。固定manifestや比較表への手動追記は行っていない。

- 新Full：point `8b1c6f7dd27644bfd14f153b`、信号13 round・価格26軸・Q3_WAITING。配分・Q3・MSE・費用はすべてnull。
- p7：既存2行の同期前後のSHA-256は両方`0e60e1b192ad3bebec160bb0caba21f0f717a754dca3da640c63d4df02efb4e1`。B_PRICE不適合/nullとA_SHAPE適合/372035410の内容を完全に維持した。準備情報を別比較点として追加していない。
- 同期が固定したregistryのlocatorは、bytesを最後に変更した`578fc2e7383ce76e0f895a7e7766c6baa45d95ca:C_RESULT_JOIN.json`（上記D directory内）。指定352539e27の同ファイルと同じblob `542a764145b67bd58b952dab80ec9cb4f0b5085f`であり、科学SOURCEを352539へ付け替えたものではない。DのQ3新規0返却はREFERENCE_ONLYとなった。

13:21:10 UTCにcronと同じ通常入口を再実行し、**UNCHANGED**・終了0（1.696秒）。HEAD・58候補・catalog SHA-256 `bbcc630fc8c80fc276ef8ba03c4070e9e378301a192115c50be5fc311b46e0aa`が不変、追加commitと重複登録は0件。定期同期は継続する。
