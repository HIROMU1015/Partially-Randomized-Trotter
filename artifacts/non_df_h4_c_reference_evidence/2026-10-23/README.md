# H4 C修正版：新grantによる固定40軸実行

[実行・結果・残る制約](../../../docs/non_df_h4_c_reference_evidence_20261023.md)。SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、branch `codex/non-df-h4-c-reference-evidence-20261023`。18:53 JSTに新grantを一回消費し起動した。

- `launch01/approval.json` / `active_contract.json` / `contract_delta.json`：利用者の承認・新契約・科学条件不変。
- `launch01/preflight.json` / `preflight_artificial_reference/`：source/runtime/inputのSHAと人工14件PASS。
- `launch01/dispatcher.json` / `DISPATCH_REQUEST.json`：一回だけの起動・outer native witness。
- `run01/`：今回の新grant・CONSUMED・claim・ledger・model・tier/result・native receipt・STOP。
- `launch01/observation*.json`：既知runと所有processだけの読取観測。予約数は完成call数ではない。
- `saved_result_audit.py`：終了後の保存構造・SHA・native監査のみ。科学値を再計算しない。

**19:15 JSTに終了し、4方式×5精度×2軸の40軸すべて64 bit TARGET_METを保存した。** 全worker/bridge/supervisorのnative exit0・回収と旧2,239 file SHA一致を確認。正式certificate受理0、旧N/G未確定・元native欠落を保持する。

`axis_outcomes.csv/json`、`normalization_saved.json`、`native_audit.json`、`completion_audit.json`、`result_inventory.json`が保存結果・終了監査の入口。`launch01/CHAT_HANDOFF.json` は計算中の19:13時点の履歴であり、現在の終了状態は `run01/STOP.json` と最終監査を読む。

local commit後mandatory STOP。新科学計算・grant再利用・N/G再会計・pushなし。
