# 検出事項と境界

最終独立consumerは公開108件の分類と一致し、今回の人工scopeで新しいProducer不整合は検出しなかった。旧SRC_TRACE_RESERVE_ACCOUNTINGは保全rawから独立再検出し、V2のS/P1 rawと新規writer観測で修正を確認した。これは科学実行のGOではない。

固定の旧10suiteを原文で実行したところ622 PASS/1 FAILとなり、過去の「623 PASS」をそのまま再現できなかった。失敗testは `CandidateAcceptance.test_cached_observer_call_count_label_exclusive_partition` の `endswith(':leaf')` で、固定observerはco_qualnameを使うため実labelが `...<locals>.leaf` となる。両observerの全計測行とpartitionは一致する。legacy_label_assertion_reproduction.jsonに実値を保存した。

旧test/consumer SOURCEを改変せず、新規versioned harness内でこの1個のsuffixだけを `.<locals>.leaf` へ明示補正した。その結果は623 PASS。原版のFAILと補正後PASSは別記録であり、原版623 PASSとは表示しない。正式な旧test修正を取り込むかは後続保守で判断できる。

開発中のconsumer誤りは今回のscopeで修正した。summary/final CPUの異なるsnapshot、exact Fraction表現、保存失敗後の予約notification、state closeの不確実性、異常STOP入力に対するIndexError、markerより前のdiagnostic ACK、canonical envelopeを扱う。初期matrix/testログは診断として保存し、最終結果へ加算していない。最初の旧回帰では限定checkoutの依存不足もあり、そのログを保持した。

Producer固定SOURCE、旧SOURCE/結果、旧V1 consumerは変更していない。新規consumerの作者自身の検査を、別reviewerによる独立レビューとは呼ばない。今回の「独立」はProducer修正セッションと異なるconsumer実装・検証経路を指す。
