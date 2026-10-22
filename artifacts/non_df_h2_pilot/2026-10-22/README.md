# H2非DF pilotの証拠

[固定条件と技術検証](../../../docs/non_df_h2_pilot_validation.md)。
`execution_contract.json` と `authorization.md` を先に固定する。
`SOURCE.json` と新規 `grant.json` を結合し、`run01/marker.json` の排他的生成でone-shotを消費する。
消費済みrun・失敗runのretry/resumeは行わない。

`run01/inputs/` は新H2の同一MO積分・Pauli/DF・状態・独立FCI/sector照合。
各 `run01/{representation}_{method}/` はfixed signal、bias、normalization、compiled RZ、
5精度の十分NとG、numerical evidence、event/order診断と技術gate。
`run01/STOP.json` は全runの成功／失敗・mandatory STOP。新stageへの自動承認ではない。

pilotの有限MC・経験的数値allowanceをrigorous保証・一般的優位性・全RPE最終費用と扱わない。
既存DF/非DFの旧SOURCE・結果・grant・ledger・marker・STOP・図は不変。

最終状態：`FAILED_NO_RETRY_MANDATORY_STOP`。受理2case（Pauli PF/DF PR）、失敗2case（Pauli PR/DF PF）。`result_inventory.json` に未取得・未受理を明記し、`preservation.json` で旧証拠を照合した。途中保存値を成功値に読み替えない。科学SOURCE b5ed2c6e2399b55a0c24eb8437f842892a6703ef。再実行しない。
