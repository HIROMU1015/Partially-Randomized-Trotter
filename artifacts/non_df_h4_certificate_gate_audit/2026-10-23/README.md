# H4残存certificate gate：原因・証拠充足性監査

[主報告書](../../../docs/non_df_h4_certificate_gate_audit_20261023.md)。基点 `86b8b002704aced8ca239414080f53793de81658`、C科学SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`。専用branch `codex/non-df-h4-certificate-gate-audit-20261023`。

DF falseの直接原因はexporterの表現別proof policy。range逸脱の検出ではない。二値測定の支持集合とencoded DF行列のnorm/期待値転送を分離した。wrapper/compilerの厳密転送、正式受理層は未完成。今回の新しいcertificateは生成・受理していない。

- `static_flag_path.json`：ASTで追跡したflag設定・転記・scope。
- `source_references.json`：固定SOURCE・SHA・行位置・GitHubリンク。
- `certificate_gate_matrix.csv/json`：4方式×5精度×2軸、14 gate/軸。正式受理false、旧N/G null保持。
- `gate_catalog.json`：各gateの必要条件・既存証拠・不足・解決区分。
- `wrapper_transfer_map.csv/json`：数値包含からliteral wrapper/compilerへ至る9境界。
- `legacy_compiler_evidence.json`：514軌道のevent/価格/compiler/fingerprint照合。既知4caseにpostcompiled QPY/QASMなし。
- `saved_model_structure.json` / `DF_local_gate_registry.json`：11基底・148局所hashの構造情報。行列を再計算していない。
- `minimal_next_contract_DRAFT.json`：技術修正、必要な局所数値証拠、compiler証明、条件付き受理を分けた非実行案。全実行flag false。
- `github_*.json` / `github_verification.json`：取得した37固定fileとrun directory・commit/treeの照合。
- `artificial_tests.json/log`：人工14 PASS、実H4/sampling/compile/高精度・区間計算0。
- `structure_tests.json/log` / `structural_audit.json`：構造12 PASS、JSON/CSV/AST/SHA/joinの照合。
- `SOURCE.json` / `preservation_before.json` / `preservation_after.json` / `completion_audit.json` / `result_inventory.json` / `STOP.json`：今回の調査SOURCE・保全・保存監査。
- `publication_status.json`：監査結果commitのpush試行・公開状態。監査終了時のpush待機snapshotとは別に記録する。

主担当自身の調査・検査であり、独立科学レビューではない。元H4 native未証明/原因UNKNOWNとC run native PASS、保存40軸TARGET_METと正式受理0軸を区別して保持。今回の実H4科学計算・数値再解析・certificate生成/受理・N/G会計・新grantは0。更新共通指示により本調査branchのcommitとpushを行い、公開結果を別記する。科学作業はmandatory STOP。
