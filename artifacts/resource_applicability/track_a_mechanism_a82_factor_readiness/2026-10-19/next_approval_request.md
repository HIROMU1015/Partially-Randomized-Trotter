# 次に必要な承認

担当：**ユーザー**。固定SOURCE `06c94745f50345b2d93714760608d9d11b83e784`、contract SHA `fe6300e3320bff41c4131873fbdc8b5cbe048013bf2b1060469eaa50b6c36ea0`、input binding SHA `5696bfcdfacf6be2bdd1b89f17e504c11629b649cf8766e61b03b26a7931c50a` に結合した、A82_OLD_PR 1比較・cosine/sine 2軸の比較説明因子生成one-shot **1回**の実行を承認するか判断してください。

今回の準備承認は実データの因子計算承認を含まないため、ここでmandatory STOPしています。正本はlocal A82_factor_one_shot_contract_DRAFT.json。レビュー用の[契約説明](factor_one_shot_contract_DRAFT.md)と[SOURCE/入力manifest](source_input_manifest.json)を用意しました。

担当：**Codex（別途実行承認後）**。新しい明示承認receiptとfactor専用grantを発行し、fresh SHA照合後に一回だけ計算する。PR/PF side、core32、他31比較を再計算せず、PASS後は保存deltaを論文来歴へ統合して再びmandatory STOPする。不一致/gate不通過はretryしない。

担当：**ユーザー**。新資料の公開範囲も指定してください。現時点で新資料の明示公開承認を特定できないためpushなし。担当：**Codex（公開承認後）**が承認対象のみpushしremote SHAを確認する。
