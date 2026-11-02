# PF選択比較の準備packet（実数値会計未承認）

基点7430a386。H6 DF19 PF136/確認PR8 n32、H8 DF28 PF184/確認PR6 n32。保存モデル使用（実測混在）側と、対応するモデル単価だけを直接実測へ置換する側の比較案。実shot再計算・実総費用再会計・実再選択/順位/比の計算0。

| File | 内容 |
| --- | --- |
| execution_contract_DRAFT.json / grant_INVALID.json | 未認可契約、allowed_calls=[]、launch commandなし |
| saved_condition_rows.csv/json | 保存6,680行。shot/bias/ε/factor、元総費用、cohort、SOURCE/SHA/locator |
| axis_price_lineage.csv/json | PF640軸slot。新450/既存8/保持旧102/単価null80を区別 |
| candidate_domains.json | H6 32/96/136、H8 32/184の固定集合 |
| extra8_existing_receipts.json | 450外の公開実測8軸。新規取得なし |
| baseline_saved_comparisons.json | 元100 scope比較を転記。新ランキングではない |
| condition_coverage.csv/json | 3案×100条件のpresence/metadata検査。新費用なし |
| output_schema.json / *_TEMPLATE.csv | 未来の総費用・選択/比・順位変化の空schema |
| source_inventory.json / input_bindings.json | 固定Git入力SHAと分子/DF/compiler/profile結合 |
| github_input_receipt.json / github_verification.json | read-only GitHub確認、62個の内容/blob照合 |
| preparation_summary.json | 準備件数。ALL_SAVED458は条件付き70/適格性未解決30 |
| structural_validation.json / technical_source_manifest.json | 著者の18検査と新技術SOURCE hash |
| packet_manifest.json / publication_scope.json | 成果物hashと新しい未承認公開範囲 |
| STOP.json | 準備完了、実会計/選択禁止のmandatory STOP |

[報告書](../../../../docs/research/track_a_pf_selection_comparison_preparation_20261020.md)、[準備runner](../../../../scripts/prepare_pf_selection_comparison.py)、[人工専用prototype](../../../../src/trotterlib/pf_selection_comparison_plan.py)、[18検査](../../../../tests/test_pf_selection_comparison_preparation.py)。

`python3 -B scripts/prepare_pf_selection_comparison.py --check`は固定Git資料からの転記一致を確認する。`--run`は入力を読む前に拒否する。prototypeも実データを拒否し、本番の会計実装は未接続。

ALL_SAVED458の70件はscope重複を含む（unique30）。H6 noncancellation30 scope条件は元適格性が未解決。限定450 subsetの72 ready/28該当なしは全登録集合の結論へ拡張しない。無条件比較可能0。H6 profile差、runtime未証明、S4仮説、初回572の最終gate欠測、旧profile待ち48のflagを保持し、旧48を再取得しない。

今回のnew packetは公開未承認。旧SOURCE/raw/grant/ledger/marker/STOP・論文結果を保全し、詳細科学レビュー/evidence freezeなし。ユーザーによる別の科学実行承認までmandatory STOP。
