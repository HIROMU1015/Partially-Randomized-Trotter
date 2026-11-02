# H4 C production：技術実装と停止証拠

[技術報告・保証範囲・未解決gate](../../../docs/non_df_h4_c_production_20261023.md)。branch `codex/non-df-h4-c-production-20261023`、基点b0f94185。実H4の数値証拠生成0、科学grant未発行、N/Gは旧nullを維持。local commitのみ、pushなし。

| 資料 | 内容 |
|---|---|
| [SOURCE.json](SOURCE.json) | 新実装commit・code/runtime/input/参照先のSHA seal。科学grantではない |
| [C_execution_contract.json](C_execution_contract.json) | 固定40軸、64/96/128bits、元の科学call最大予算。3実行flag false |
| [input_manifest.json](input_manifest.json) / [schema](input_schema_static.json) | SHAとNPY headerだけの実H4静的検査。payload数値評価なし |
| [runtime_manifest.json](runtime_manifest.json) / [legacy](legacy_runtime_manifest.json) | private FLINTと封印済み旧準備runtime。共有環境変更なし |
| [tests_release/tests.json](tests_release/tests.json) / [backend13](backend13_regression.json) | 人工・回帰・実行制御の最終検査。旧開発ログは合算しない |
| [artificial_bridge_final](artificial_bridge_final) | 旧runtime人工2qubit→production経路、Pauli/DF各20軸。H4証拠ではない |
| [GitHub照合](github_verification.json) | 固定commitと主9資料の実取得・canonical blob SHA照合 |
| [before](preservation_before.json) / [after](preservation_after.json) | 旧SOURCE/raw/claim/B結果・grant/STOPを保持 |
| [completion_audit.json](completion_audit.json) / [STOP.json](STOP.json) | 科学実行0、技術SOURCE確認、保全・停止 |
| [CONTINUATION.md](CONTINUATION.md) | 次回の承認・grant発行・起動手順と残る制約 |

nominal normalization、実装確率対応、encoded operatorとliteral wrapperへの転送、数値設計目標、正式certificate受理は別gate。元H4監視消失はUNKNOWN/native未証明のまま。今回のSTOPは旧runの終了証拠を補完しない。
