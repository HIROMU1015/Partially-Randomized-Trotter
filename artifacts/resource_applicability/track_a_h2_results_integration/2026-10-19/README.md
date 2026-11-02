# H2正式受入・研究資料統合packet

2026-10-19。H2 P `f2d1a86c` の2条件はユーザーが有限標本点結果として正式採用。その他の文書更新案の採択とは区別する。

- `adoption_record.json` / `H2_adopted_points.csv`：正式受入した2条件。
- `github_read_receipt.json` / `stage_evidence.json,csv`：F/FP/D/A/S_STATS/Pの公開SOURCE、固定GitHub結果・契約・終了監査。
- `H468_main_table_UNCHANGED.csv` / `H468_methods_UNCHANGED.md`：公開済み固定資料のbyte同一snapshot。
- `H2468_comparison_separate.csv,md`：元主表6条件＋H2追加2条件の研究資料。論文主表の置換ではない。
- `protocol_v2_H2_amendment_DRAFT.json`：v1不変、既存草案・別採択版を固定参照するH2追記案。
- `paper_manifest_update_DRAFT.json`：固定baseのpath/SHAと下記追加案の適用規則。正本には未適用。
- `paper_data_manifest_H2_additions_DRAFT.csv`：PR/PFの4行。元1576行を変更しない。
- `claim_evidence_matrix_H2_additions_DRAFT.csv`：H2の2 claim。元14行を変更しない。
- `evidence_ledger_H2_additions_DRAFT.csv`：保存G_PR/SE/G_PF/n/statusの10行。元316行を変更しない。
- `preservation.json` / `structural_validation.json`：主agentによる文書構造・hash保全照合。独立科学レビューではない。
- `publication_scope.json` / `artifact_manifest.json` / `MANDATORY_STOP.json`：公開範囲、成果物hash、停止境界。

原図34ファイル・H468主表は不変。H2は旧v1対象外、本文採否未判断。H8の公開詳細PR監査不足、H4非再現、H6粗精度PF低費用を保持する。母平均、希少event費用/分散、完全入力手順同一性は保証しない。新しい科学計算・統計解析・sampling・compile・数値比計算は0。

再確認は `python -B -S scripts/integrate_h2_research_documents.py --verify`。現在のH2 private証拠のmetadata hash検査を含むため、このworktreeと初回保全snapshotが必要。初回snapshotはH2専用の `/tmp/h2-results-integration-20261019-preservation.json` に置き、private path/payloadをGitへ追加しない。公開部分の再照合には固定commitと公開receiptを使える。

入口は [統合報告](../../../../docs/research/track_a_h2_results_integration_20261019.md)。ユーザーが次に判断するのはH2掲載範囲、paper/CEM/台帳案の適用、H2 v2追記案の採択。今回完了後はmandatory STOP。
