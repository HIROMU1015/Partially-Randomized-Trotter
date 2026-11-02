# PF直接compile：契約分割案（未実行）

[報告書](../../../../docs/research/track_a_pf_direct_compile_contract_split_20261020.md)。
固定788軸を[現行4系列722軸](current_4series_722_contract_DRAFT.json)と
[legacy H4 DF12 66軸](legacy_h4_df12_66_contract_DRAFT.json)へ分割した。
各CSVは既存提案の行を無変更で転記したもの。取得範囲・科学実行は未承認。

- [方針記録](partition_decision.json)
- [構造検証](structural_validation.json)
- [旧packet保全](preservation.json)
- [公開範囲](publication_scope.json)
- [STOP](MANDATORY_STOP.json)

`proposed_task_ids`と`proposed_call_caps`は将来契約の案で、実行許可ではない。
`selected_task_ids=null`、有効call数0、grantなし。SOURCE/runtime/loader/gateは未結合。
承認・grant・ledger・output・marker・STOPを契約ごとに独立させ、自動移行しない。

**mandatory STOP。**
