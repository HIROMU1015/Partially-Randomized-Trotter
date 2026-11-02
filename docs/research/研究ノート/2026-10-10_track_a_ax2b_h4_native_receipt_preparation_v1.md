# 2026-10-10 Track A：H4-P実装・実行前固定

利用者の「作業を進めて」を、直前の提案の1–2（専用runner・合成テスト・source/input/env/resources固定）へ適用した。
旧metadata-only記録は改変せず、[新しい準備契約](../track_a_ax2b_h4_native_receipt_preparation_v1.md)を追加する。
source・合成検査・metadataだけを既存Track A branchへ公開する。実H4入力の数値load/native分子準備は別認可前のため行わない。

H4-PとH4限定検証は異なるgrant schemaを要求する。native receiptを取得するplanのsealとscience manifestのsealも分離した。
理由は、instruction上界取得自体にbasis/symbolic tailのnative準備が必要であり、metadata-onlyでは実値を埋められないため。
900秒・AS8GiB・output16MiB・CPU3・worker/BLAS1・load1/prepare8を指定する。これらは未来の実行条件で、今回の資源予約・実使用ではない。

専用48合成testsはlocal engineering evidence。分子correctness・精度適格性・資源的優位を結論しない。
旧source/科学結果/freezesと既存dirty差分を保存し、indexには今回の入口だけを追加する。
`H4_NATIVE_RECEIPT_NOT_AUTHORIZED` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
