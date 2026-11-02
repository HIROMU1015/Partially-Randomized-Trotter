# H4 PR tail / implementation-residual investigation

[重点調査報告](../../../docs/non_df_h4_tail_rho_investigation_20261024.md)。基点 `5600f0e4b28c140a6a50b8aba47425860972a932`、納品branch `codex/non-df-h4-tail-rho-audit-delivery-20261024`。**静的調査のみ。新しい実H4数値・費用・確率・N/G評価、sampling/compile、科学grantは0。Mandatory STOP。**

Compiled費用upperは未確立。Literalの既存counterを使うsymbolic envelopeと、固定compilerの費用増加率だけを保証する十分条件を整理した。Rhoは12辺のscopeを分け、同じ保存角度のSOURCE対応で診断を減らせる候補を示した。母平均の無条件優位性・actual rho適格性は認定しない。

- [証拠scope map](evidence_map.json)：費用upperとpopulation upperはnull、全12辺の証拠・不足を表示。
- [2件の非実行契約案](next_contracts_DRAFT.json)：費用envelope/既存cohort母平均上界と、DF literal残差の限定辺。実行flag false、SOURCE/runtime/grant null。435件や追加256本を既定にしない。
- [明示承認範囲](approval.txt)、[GitHub取得](github_verification.json)、[静的検証](verification.json)、[最終checker](check_delivery.py)／[再利用したtoken・join checker](check_static.py)、[最終人工6件](artificial_tests_final.log)。既存counter/形式lemmaを人工入力にだけ適用。
- [保全611ファイル](preservation_before.json)、[終了](STOP.json)、[公開状態](publication_status.json)。旧SOURCE/結果/grant/ledger/marker/STOPと、旧null/COMBINATION_BLOCKED/MISSING_EVIDENCE/native UNKNOWNを保持。

参照正本：前回[条件付き結果](../../non_df_h4_conditional_cng_results/2026-10-24/README.md)、Lの[全324 support](../../non_df_h4_local_diagnostic/2026-10-23/audit02/component_support.csv)、[C/L接続報告](../../../docs/non_df_h4_cl_transfer_20261023.md)。管理用に旧表やrawを複製していない。

新科学実行には、選ばれた範囲のユーザー明示承認、専用実装・人工検証・SOURCE/input/runtime固定、未使用grantが別途必要。今回は実行可能化しない。

着手worktreeのchecker更新を上書きせず、別worktreeに保全した。[コピー時のSHA](delivery_isolation.json)。旧ログ・追加receiptも保存し、最終検証はこの納品worktreeで再実施した。
