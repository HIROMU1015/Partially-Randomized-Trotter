# PF次数による優劣逆転：既存証拠の技術監査

2026-10-23 JST。固定6 commitのSOURCEと保存結果を調査し、H2/H4/H6×5精度を対応付けた。

保存順位の逆転候補はH2のε=.005/.001/.0005とH4のε=.005の4条件。残り11条件は同方向だった。shot関数の静的同一性を確認した一方、allowance政策差とnative/runtime未結合が残るため、科学的比較Cと因果機構は未認定である。

[技術報告全文](../../artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/REPORT.md)、[75行の保存証拠対応表](../../artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/evidence_rows_75.csv)、[SOURCE索引](../../artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/SOURCE_INDEX.md)、[技術検証記録](../../artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/verification.json)。

完成済み監査packetのコピーを専用ブランチへ固定した。既存SOURCE・結果・grant・ledger・marker・STOPは変更せず、新規科学計算・shot再計算・費用再会計・候補再ランキング・compile・詳細科学レビューは0。監査段階の原記録と、後続のコミット用格納の来歴は[packet案内](../../artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/README.md)で区別する。

次の作業：ユーザーが提示コマンドで専用ブランチをpushする。科学的な追加判断・計算の承認は別途必要であり、Codexはmandatory STOP。
