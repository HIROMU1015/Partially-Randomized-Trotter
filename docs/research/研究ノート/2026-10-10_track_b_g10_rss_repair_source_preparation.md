# 2026-10-10 Track B G10 RSS限定修正source準備

利用者の限定修正指示を受け、原因監査commit `21f66137ab205fb0d71ee6ee001844f4c2d725e3`
から独立branchでS2を準備した。旧S/A/Rと17-row technical prefix、marker、STOPは維持する。

全resultの再帰コピーとJSON chunk list/joinを逐次encoderへ置換し、不要なold/pending/local参照を解放した。
m5 deepcopy、全event/IR/budget fields、科学的処理順序、confidence/errorと全capsは維持した。
正常I/Oをguardに含め、partialとcompletionを区別し、失敗時はbounded receiptのみを保存する設計とした。

保存JSONのI/O-only同値性と人工typed fixture、focused tests、AST差分、protected provenanceを検査した。
本番の512 MiB以内での完了は未確認。新A2、production marker、科学runは0。

報告: [GPT実行前レビュー](../../tracks/algorithm_codesign/g10_rss_repair_source_and_gpt_review_20261010.md)。
根拠: `artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/`。
準備完了でmandatory STOP。次はGPTのsource reviewと別途明示one-shot承認が必要。
研究方針・新規性・科学的判断はGPTへ戻す。
