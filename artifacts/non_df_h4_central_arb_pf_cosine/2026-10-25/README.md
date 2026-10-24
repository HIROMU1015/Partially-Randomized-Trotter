# H4 Pauli central PF cosine Arb包含評価

2026-10-25の利用者明示承認に基づくPF cosine1件を完了。
科学SOURCE92a5f104a2e9962353870e87f1d910d236adba41、64bit・状態作用成功1/総試行1で保存予算以内と確定。
rho上端180675711/18889465931478580854784（約9.565e-15）。96/128bit未実行。
native exit0/reap・controller exit0・mandatory STOP。正式certificate未受理。
[結果報告書](../../../docs/non_df_h4_central_arb_pf_cosine_20261025.md)。

- `approval.md`：対象・call・精度・STOP・回復の明示承認。
- `inputs.json`：実際に使う保存QPY/descriptor/raw model/C-targetなど9ファイルのSHA。
- `structural_preflight.json`：実QPYと保存descriptor一致。新compile/状態作用0。
- `arb_artificial_tests.json`、`qpy_artificial_tests.json`：新10＋再利用25＝人工35 PASS。
- `preservation_before.json`、`prior_manifest_rows.json`：旧証拠412ファイル・旧manifest186行。
- `runtime.json`、`seal.json`、`active_contract.json`：新SOURCE固定後に保存する。
- `run01/`：新one-shot grant、call ledger、各bit区間と予算比較、native exit/reap、STOP。
- `reported_intervals.json`：exact端点・外向き表示。rho>=0との共通部分を併記しraw ballは無変更。
- `verification.json`：SOURCE/runtime/入力・旧証拠と保存端点・native終了の65項目照合。
- `publication_status.json`：結果commit/push/remote SHAの時点記録。
- `pre_grant_*`：旧system zlib SHA差によるgrant前setup停止。科学0/grant0、変更主体UNKNOWN、新SOURCEで回復。

準備kernelのnative作用とL2半径を共通化し、production側だけが固定PF入力とdispatchを確認する。
既存Cのexact変換・ball演算・端点、CallBudget/save、native witnessを再利用する。
旧準備契約・科学SOURCE/結果/grant/ledger/marker/STOPは改変しない。
正式certificate受理・PR拡張・R2・N/G・新compile/sampling/state生成は承認外。
