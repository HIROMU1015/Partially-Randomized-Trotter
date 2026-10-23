# H4数値精度B＋C

[診断結果・来歴・完了報告](../../../docs/non_df_h4_precision_bc_20261023.md)。Bは許可済みの保存40軸をone-shotで完了、Cは設計・実装計画・人工検査まで完了。B技術SOURCE `2bd212288e8530e519c4a8135257df37f7608662`、基点 `ebe20403bcb5c4439162614f64d991a769d7ed68`、旧科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`。local commitのみ、pushなし。

| 成果物 | 内容 |
|---|---|
| [B_run01/diagnostics.csv](B_run01/diagnostics.csv) / [JSON](B_run01/diagnostics.json) / [Markdown](B_run01/diagnostics.md) | 固定4方式×5精度×2軸の必要誤差予算。R22/H18、N/Gはnullを維持 |
| [B_run01/summary.json](B_run01/summary.json) | R約1.53〜90.11倍、H約107.04〜2823.40倍のu改善が必要 |
| [B_run01/SOURCE.json](B_run01/SOURCE.json) / [contract](B_run01/contract.json) / [grant](B_run01/grant.json) | 実行前固定、B専用grant一回消費済み。再実行不可 |
| [B_run01/verification.json](B_run01/verification.json) / [native_process](B_run01/native_process.json) | 新B自身だけのexit0/direct wait、40軸・独立算術40件PASS |
| [C設計書](../../../docs/non_df_h4_numerical_enclosure_design_20261023.md) | 数値包含・bias・有限law normalization・保存入力対応の設計 |
| [C_precision_method_map.csv](C_precision_method_map.csv) / [JSON](C_precision_method_map.json) | 40軸の必要予算と手法・設計目標（certificateではない） |
| [C_execution_contract_DRAFT.json](C_execution_contract_DRAFT.json) / [実装計画](C_implementation_plan.json) | 承認待ち完成案。実H4 runnerは未実装、3実行flag false、科学grant未発行 |
| [technical_tests_final.json](technical_tests_final.json) / [C_artificial_checks_final.json](C_artificial_checks_final.json) | 最終33＋別backend人工13検査PASS。開発中ログは加算しない |
| [C_backend_install.json](C_backend_install.json) | private python-flint0.9.0/FLINT3.6.0、wheel SHA/runtime SHA記録。共有環境不変 |
| [github_retrieval.json](github_retrieval.json) | GitHub固定調査commitと4資料の実取得・照合 |
| [preservation_before](preservation_before.json) / [after](preservation_after.json) / [終了監査](completion_audit.json) | 旧SOURCE738・結果/claim1147・調査17 file等の保全 |
| [CONTINUATION.md](CONTINUATION.md) / [STOP.json](STOP.json) | 担当・次回承認・今回専用mandatory STOP |

`B_contract_DRAFT.json`は実行前の履歴資料。実行に使った契約はB_run01/contract.jsonで、既に消費済み。Cの実H4計算・certificate生成・N/G計算は0。通常float値の高精度再計算だけを厳密certificateへ格上げしていない。元H4 native exit/reap/STOPは未証明、監視消失原因UNKNOWNのまま。今回の新STOPは元runの終了証拠を代替しない。
