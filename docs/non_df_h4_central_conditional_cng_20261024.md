# H4 Pauli central-control：条件付きC/N/G総資源比較

固定H4/1Å/STO-3G/n8、Pauli-JW184項/PR prefix27、S2/T=.8/q8/r4/K4、
ε=.05/.01/.005/.001/.0005、両軸、保存PR256本/PF1本。
DF-nativeはtol-only1e-8/rank10、prefix1＋one-bodyの旧条件付き結果を再利用する。

基点 `564cd71936fa27225624b00a7d5d1f12dc4cc6dc`。
[承認原文](../artifacts/non_df_h4_central_conditional_cng/2026-10-24/approval.txt)・
[GitHub照合](../artifacts/non_df_h4_central_conditional_cng/2026-10-24/github_verification.json)。
専用branch `codex/non-df-h4-central-conditional-cng-20261024`。

## SOURCE固定前の技術準備

既存h4_conditional_cngの有理数shot/ceil/guard、paired集計、固定統計を再利用する。
比較関数は内部bounds読み取りだけを保存端点参照へ差し替え、DF evaluatorの呼出を防ぐ。
旧precision_accounting/fixed_signal_shotsのguardは既存人工検査を再利用する。
新14＋旧16人工PASS。全て主agent自身の検査。
Cの保存bias_distance.upper、signed bias、B_saved/log_B_upperの定義を保持し、
新ρを独立未知変数とする。旧generic残差を転用しない。C totalの減算・C/L加算なし。

Pauli10条件20軸・guard端点・15比較・paired G統計1batchだけを、新SOURCE/runtime/input/seal/grant固定後に実行する。
compile/build/sampling/SCF/FCI/DF/state/signal/元C-L再計算0。
元H4 N/G null・native UNKNOWN・hash FAIL/原因UNKNOWN・C/L COMBINATION_BLOCKED・
旧compiler MISSING_EVIDENCE・certificate未受理を維持する。

最新版運用指示§8.4/10.1/10.2を適用。参照された§10.3は現ファイルに存在しないため規則を補作しない。
軽い保存scalar処理は単一workerが実用的で、CPU/RAM/affinity上限・共有設定変更なし。
