# H4 Pauli central PF cosine Arb包含評価

2026-10-25の利用者明示承認に基づくPF cosine 1件。
科学SOURCE固定前の段階では実H4状態作用0、新科学grant0。
64→96→128bit、各精度1回・初回最大3状態作用。予算との関係が決まれば直ちに終了する。

- `approval.md`：対象・call・精度・STOP・回復の明示承認。
- `inputs.json`：実際に使う保存QPY/descriptor/raw model/C-targetなど9ファイルのSHA。
- `structural_preflight.json`：実QPYと保存descriptor一致。新compile/状態作用0。
- `arb_artificial_tests.json`、`qpy_artificial_tests.json`：新10＋再利用25＝人工35 PASS。
- `preservation_before.json`、`prior_manifest_rows.json`：旧証拠412ファイル・旧manifest186行。
- `runtime.json`、`seal.json`、`active_contract.json`：新SOURCE固定後に保存する。
- `run01/`：新one-shot grant、call ledger、各bit区間と予算比較、native exit/reap、STOP。

準備kernelのnative作用とL2半径を共通化し、production側だけが固定PF入力とdispatchを確認する。
既存Cのexact変換・ball演算・端点、CallBudget/save、native witnessを再利用する。
旧準備契約・科学SOURCE/結果/grant/ledger/marker/STOPは改変しない。
正式certificate受理・PR拡張・R2・N/G・新compile/sampling/state生成は承認外。
