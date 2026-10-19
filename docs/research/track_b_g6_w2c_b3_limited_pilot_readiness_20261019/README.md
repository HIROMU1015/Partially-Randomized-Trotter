# 限定G7 pilot：実行前技術準備

状態は **UNADOPTED / 実科学入口閉鎖 / mandatory STOP**。実G7/F3 sampling、実科学import、新科学計算、実authorization/grant/claimは今回すべて0件。Full Returnの優位を採択していない。

固定producer公開基点は `7ae4bd0b951dae05380538cd5c22a3ead5328ab2`、独立consumer原本は `bf601f8a9723a61b7a05b534019f8e036a362735`。新しい人工受入SOURCEは `b9cabcfc6cf304ab65fc8a540c33ce2173f6cbaa`。SOURCE履歴はbundleでも保全する。consumer導入は選択copyでありmerge/rebaseではない。

- [consumer導入とsmoke](server_consumer_integration.md)：正常12件・障害35件、費用46件照合、parent crashの1件は欠測のまま。
- [回帰結果](artificial_regression_results.md)：consumer関連540件、別の既存v3統合125件、新候補31件を区別。過去481件と540件の合算ではない。
- [admission証拠](admission_one_shot_evidence.md)／[STOP履歴](cooperative_stop_evidence.md)／[pending](pending_evidence_resolution.md)。新しい人工履歴で旧47件の欠測を埋めていない。
- [科学scope候補](pilot_scientific_scope_candidate.md)／[計測設計](pilot_measurement_design.md)／[契約候補](pilot_contract_candidate.json)。推奨検討案は各profile cell 256 calls、逐次・P1の両条件で計16,896 calls。科学的十分性や正式採択を意味しない。
- [readiness](pilot_execution_readiness.md)／[全N判断表](pilot_vs_full_N_decision_matrix.md)／[次段階](future_execution_handoff.md)。今回のSOURCEはfake v3 projectionを検証する準備実装であり、実科学bridgeを含む正式releaseではない。
- [SOURCE・runtime](source_and_runtime_manifest.md)／[source_inventory.json](source_inventory.json)／[verification.json](verification.json)／[publication_receipt.json](publication_receipt.json)。公開後の内容照合はpublication receiptを参照。

試験は主担当Codexが実施した。offline照合はproducerをimportしない別実装のcheckerによるもの。別人・別AIによる独立reviewを受けたとは主張しない。

次の担当は**ユーザーとGPT**。pilot規模、比較条件、証拠粒度、正式SOURCE採択と別の科学実行承認を判断する。Codexは自動移行しない。

公開content commit `f3ba0cd09d3c09c5d04a6b48c3d6c9da63fff867` の684 files / 57,212,260 bytesを読み戻し、全SHA256とremote HEADを照合済み。詳細はpublication_receipt.json。receipt自身は自己hashせず、後続metadata commitを別途照合する。
