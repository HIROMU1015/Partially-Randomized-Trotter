# 共有対象一覧（ローカル候補）

全て未送信。共有推奨は候補内容への判定であり、外部送信・pushの承認を意味しない。

| file | 目的 | bytes | SHA256 | 機密性・判定 |
|---|---|---:|---|---|
| share_candidate/00_README_GPT.md | 結論・読み方・固定commit・検証限界 | 6626 | `85adcad6ff9b818970cfd0b7cc89a8f067cd9fe32cc9520b634b4fcccfe08f72` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/01_original_profile_receipt.json | 原契約内profileのSHA・digest規約・192軸への結合 | 7732 | `cc8e184345e8bd0d7f741d0abb0f89201d4a289f4a1e95a9cf2dc8ec787d907c` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/02_saved_PR_circuit_search.json | 回路未取得・探索範囲・未保存/削除/アクセスの区別 | 5686 | `612d91008efe95d0436e21406c7f5aadac859bf33cb3cac71214799590681bdc` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/03_axis_coverage.csv | 192軸のSOURCE・fingerprint・task/measurement SHA・再利用2軸 | 89095 | `d504a9232c7fb7fc796abe588475d7d9f24de03644abfed7133b752a50b7ce57` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/04_source_and_provenance.json | 固定SOURCE関数抜粋・元file SHA・来歴 | 16658 | `975e5f644b3abbc93993be6b40c0fa4176e29d38ace05d00f102615d9705e1d6` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/05_missing_status_changes.json | 旧欠測statusとの差分と変更しない項目 | 1365 | `6feb3b49d42fc24f633170ad3bbe137db53c9d3c48b3d7410aedeb8832ac8b45` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/06_preservation_and_scope_audit.json | 読取保全・許可範囲・科学実行0の記録 | 644 | `9d0f6b3bee6b120c0296dcc6848383d157eaa1a94a6d8ca6d20eb8bbb5c68cf2` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/supplement_manifest.json | 共有候補のfile SHA・未送信状態 | 1516 | `be67fd80e2671543db8d87cb81b3610a22e8e1df54c0770abdfb81528bbfb261` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| share_candidate/verify_supplement.py | 標準libraryによるoffline構造・SHA・改変検知 | 6673 | `46d50e9a17bed77ae1b54b6d3f769053160048949746fa468f8a697a5a383dea` | 機密情報を除いた共有候補／共有を推奨する候補。今回の外部共有には別承認が必要 |
| H8_ORIGINAL_ARTIFACTS_SUPPLEMENT_20261020.zip | 9fileを一括添付する補足ZIP。既公開旧ZIPと組で使用 | 39004 | `212ac51307ce491f4ce32a790c4896b211419bc4de674b2bb2c36cb8658ef0c8` | 機密情報を除いた共有候補／承認後の添付用として推奨 |

残る開示範囲：原本SHA・過去の実行構造・既公開SOURCEとの相関。内部pathを除いたため、第三者は原profile canonical digestをこの候補だけで再計算できない。

非共有：原契約全文、原profile literal bytes、内部path付き所在/探索/読取索引。これらはrepository公開を推奨しない。必要な受領者に限定したローカル閲覧、またはユーザーが承認した暗号化経路を別途指定する。

添付推奨（別承認後）：`H8_ORIGINAL_ARTIFACTS_SUPPLEMENT_20261020.zip` と既公開 `H8_GPT_SHARE_PACKET_20261020.zip`。
