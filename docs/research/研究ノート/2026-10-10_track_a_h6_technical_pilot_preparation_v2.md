# 2026-10-10 Track A H6 pilot v2接続修正・未実行

ユーザー継続指示を[coverage修正・synthetic・新seal準備](../track_a_h6_technical_pilot_preparation_v2.md)へ結合。
原v1 ACTUAL_PRIMITIVE_COVERAGE/STOPを保存し、登録validation timesを固定coverageへ明記。actual bounds/差分を拒否前に保存する別versionを追加。
science source `0b04886869efb9d08b07d6517300da2bc0123f4a`、206 science/3 validation、156 local synthetic（v2 70/v1 37/coverage49）通過。
7 cell/36 wrapper・245 keys/735 probes・入力/政策/計算上限不変。旧CPU [0,2,5,6]/4 threadsを固定。
実H6 numerical decode/prepare/signal/sampling/build/compile0。新grant未発行。旧v1 grant消費済み。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。次の一回runは別認可。その後原結果/欠測公開・GPT独立科学レビューへ戻す。
