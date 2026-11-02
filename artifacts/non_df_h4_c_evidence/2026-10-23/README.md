# H4 C候補証拠run01：実装欠陥・不受理・mandatory STOP

[報告書](../../../docs/non_df_h4_c_candidate_run01_20261023.md)。branch `codex/non-df-h4-c-evidence-run01-20261023`、実行SOURCE49c18ccc3、基点c4497d8b4。新grant一回消費、科学retry0。referenceの右側state作用欠落を確認し、Pauli20軸rawは不受理、DF20軸は未完了。旧N/G=null/UNDETERMINEDを維持。新規pushなし。

| 資料 | 内容 |
|---|---|
| [承認](launch01/approval.json) / [active契約](launch01/active_contract.json) / [preflight](launch01/preflight.json) | 次段階の承認、固定scope、SOURCE/runtime/input保全確認 |
| [grant](run01/grant.json) / [CONSUMED](run01/CONSUMED.json) / [実行STOP](run01/STOP.json) | 新grant8af94df1…を一回消費し、FAILED_NO_RETRY |
| [Pauli結果](run01/pauli_jw_result.json) / [call ledger](run01/call_ledger.json) | raw保存。reference/bias/判定は不受理。予約数を完成件数としない |
| [全40軸CSV](axis_outcomes.csv) / [JSON](axis_outcomes.json) | 不受理20軸と最終結果未取得20軸を区別。再会計なし |
| [source診断](diagnosis01/source_diagnosis.json) / [人工FAIL](diagnosis01/artificial_regression.json) | 人工1qubit恒等演算で期待1・実際0、1 FAIL。H4を新評価しない |
| [未適用patch](diagnosis01/proposed_reference_fix.patch) | reference state作用と列vector shape検査の修正案 |
| [停止理由](launch01/integrity_stop_request.json) / [最終観測](launch01/observation03.json) | 自分のDF witnessだけを同定し停止。native exitと全回収を記録 |
| [結果監査](completion_audit.json) / [監査source](saved_result_audit.py) / [今回STOP](STOP.json) | 保存証拠・hash/native/予算・旧結果保全。数値再実行なし |
| [before](launch01/preservation_before.json) / [after](launch01/preservation_after.json) | 旧SOURCE/raw/grant/claim/STOPとB結果保全 |
| [CONTINUATION.md](CONTINUATION.md) | 次回の担当・承認・新SOURCEと新grantの必要範囲 |

旧C技術パッケージの「人工110 PASS・backend13 PASS」は履歴であり、今回の欠陥を除外する十分な検証ではなかった。記録を上書きせず、新回帰FAILを追加した。新runのnativeを旧H4の欠落nativeへ転用しない。
