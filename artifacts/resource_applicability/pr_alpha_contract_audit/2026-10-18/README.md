# PR alpha contract audit

A-1/A-2の固定SOURCE/保存metadata監査と人工境界検査。

- [監査報告と系別契約表](../../../../docs/research/pr_alpha_contract_audit_20261018.md)
- `technical_validation.json`: 固定SOURCE/結果44 fileのcommit・Git blob・bytes/SHA、AST照合11件、人工検査16件、保存結果の状態分類。
- [stdlib専用script](../../../../scripts/audit_pr_alpha_contracts_20261018.py): 固定Git blobの読取と人工入力のみ。科学providerをimportせず、保存shot/費用/順位/失敗確率を再計算しない。

再検査はrepository rootから `python3 -B scripts/audit_pr_alpha_contracts_20261018.py`。
既存のNPZ/科学run/grant/ledger/markerは操作しない。主担当自身の技術検査であり独立reviewではない。
条件付き測定モデルの整合性を、厳密bias・費用母平均・RPE総成功率のcertificateへ昇格させない。
`MANDATORY_STOP`、次科学stage未認可。
