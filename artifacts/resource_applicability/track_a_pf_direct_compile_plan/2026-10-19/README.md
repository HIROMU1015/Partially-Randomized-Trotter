# PF直接コンパイル拡充：計画のみ

[計画・判断材料](../../../../docs/research/track_a_pf_direct_compile_expansion_plan_20261019.md)。新規科学実行・compile・sampling・fit・費用再会計0。実行はユーザーの計画確認と明示承認後。

|資料|内容|
|---|---|
|[candidate_inventory.csv](candidate_inventory.csv)|684登録候補。DF入力系列別・両軸測定状況|
|[axis_inventory.csv](axis_inventory.csv)|1368軸の直接/モデル/単価欠測・再利用条件|
|[axis_evidence.csv](axis_evidence.csv)|保存測定・保存予測の文字列転記、SOURCE・locator|
|[inventory_counts.csv](inventory_counts.csv)|入力系列ごとの棚卸し件数|
|[proposed_compile_axes.csv](proposed_compile_axes.csv)|未実測788軸、全登録qを保持|
|[conditional_reacquisition_axes.csv](conditional_reacquisition_axes.csv)|旧profile等が未解決の既存48軸。自動再実行なし|
|[saved_mixed_comparisons.csv](saved_mixed_comparisons.csv)|保存比較のPF選択と費用status、再選択なし|
|[model_registry.csv](model_registry.csv)|既存q依存式・係数、評価/fitなし|
|[system_bindings.json](system_bindings.json)|既存入力・compiler/profileのbinding転記|
|[source_inventory.json](source_inventory.json)|固定commit/path/blob/SHAとGitHub URL|
|[github_read_receipt.json](github_read_receipt.json)|GitHub固定12ファイルの直接bytes照合|
|[saved_resource_receipts.json](saved_resource_receipts.json)|既存wall/RSS等。driver/worker/aggregateを区別|
|[feasibility_assessment.json](feasibility_assessment.json)|数値的な新予測を行わない実行可能性の見通し|
|[execution_contract_DRAFT.json](execution_contract_DRAFT.json)|科学call範囲案。SOURCE未seal、認可・grantなし|
|[future_data_schema.json](future_data_schema.json)|直接観測と複数モデルを分離する将来schema|
|[structural_validation.json](structural_validation.json)|作者自身のmetadata構造検証|
|[publication_scope.json](publication_scope.json)|計画・既公開metadataの公開範囲|
|[artifact_manifest.json](artifact_manifest.json)|新packetのSHAと文書builder|
|[MANDATORY_STOP.json](MANDATORY_STOP.json)|科学実行未認可・mandatory STOP|

候補の直接観測が存在することと、将来の異なる実行環境で流用可能であることは別。profile不明を現profileで埋めず、未知の単価を0としない。保存予測の単価と直接測定を別行で保持し、総費用は再計算していない。
