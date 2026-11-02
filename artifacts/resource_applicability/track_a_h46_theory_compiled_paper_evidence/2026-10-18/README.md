# H4/H6理論・compiled比較：論文化向け証拠整理の入口

[独立して読める証拠整理資料](../../../../docs/research/track_a_h46_theory_compiled_paper_evidence_20261018.md)。
正本は修正済みRESULT `6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1`。
本bundleは文書整理であり、科学SOURCEの新規実行・grant・新しい数値結論を含まない。

|成果物|用途|
|---|---|
|[Claim–Evidence Matrix](claim_evidence_matrix.md) / [JSON](claim_evidence_matrix.json)|12候補、支持範囲・反証でなく証拠不足の区別、正式採択false|
|[表候補索引](table_index.md) / [全528行索引](row_index.json)|期待費用84、same-event84、shot込み360を別比較として参照|
|[条件表A](table_a_conditions.json)|固定7候補とcompiler・標本・input来歴|
|[表B1：期待費用](table_b_expectation_candidates.json)|mixed28行、期待値対保存n32/n8平均。全モデル84行は正本|
|[表B2：同一event](table_b_same_event_candidates.json)|mixed28行、各候補の1保存event。全モデル84行は正本|
|[表C：shot込み](table_c_candidates.json)|全6pair×5精度、mixed/RZ/倍率1の30条件。不適格も保持|
|[表D：S4](table_d_s4_candidates.json)|8受入行、新旧・native・compiled不変の参照|
|[図一覧](figure_inventory.md) / [JSON](figure_inventory.json)|旧SVG2件は修正前。将来4設計案、描画0|
|[未解決事項](limitations.md) / [JSON](limitations.json)|13事項、固定研究の制約とより強い主張の証拠要求|
|[provenance索引](provenance_index.json)|SOURCE/RESULT/公表mirror/input receipt/固定ファイルhash|
|[source closure照合](source_closure_checks.json)|科学codeを実行せずGit blobのSHAを照合|
|[判断事項](user_decisions.json)|ユーザーが次に判断する5点。採択は未実施|
|[技術検査](technical_checks.json) / [検査器](verify_dossier.py)|抽出値・固定行数・schema・link・保全の技術的確認|

元結果の表・labelは上書きしない。各rowの正確な有理数、欠測、SOURCEを維持する。
原稿・正式主図・新規性・中心claimの採択は未実施。次の担当はユーザー。公開確認後mandatory STOP。
