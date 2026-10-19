# PF直接compile：技術準備packet

[報告と残る本番結合](../../../../docs/research/track_a_pf_direct_compile_technical_preparation_20261020.md)。**提案788軸、選択0軸、科学実行未認可。**

- [scope_acceptance.json](scope_acceptance.json)：研究適合の判断と技術準備承認。取得範囲/科学実行の承認はfalse。
- [execution_contract_DRAFT.json](execution_contract_DRAFT.json)：選択null・有効grantなし・call上限0の未認可契約。
- [backend_binding_DRAFT.json](backend_binding_DRAFT.json)：固定SOURCEの静的API対応、旧driverの除外処理、本番loader未結合。
- [parallel_dispatch_DRAFT.json](parallel_dispatch_DRAFT.json)：動的admission方針とworker未結合の状態。
- [readiness_boundaries.json](readiness_boundaries.json)：実装済み部分と未開始の本番結合を区別。
- [preservation.json](preservation.json)：計画commitの全packet bytes不変。
- [technical_source_closure.json](technical_source_closure.json)：新技術code bytes。科学実行SOURCE sealではない。
- [synthetic_validation.json](synthetic_validation.json)：fake providerと人工契約の検査。分子検証ではない。
- [structural_validation.json](structural_validation.json)：metadata preflight。production dispatch無効。
- [artifact_manifest.json](artifact_manifest.json)：新成果物SHA。
- [MANDATORY_STOP.json](MANDATORY_STOP.json)：実入力/回路/control/compile/fit等0、次stage未認可。

元の[788提案軸](../../track_a_pf_direct_compile_plan/2026-10-19/proposed_compile_axes.csv)と[48軸の再照合候補](../../track_a_pf_direct_compile_plan/2026-10-19/conditional_reacquisition_axes.csv)は変更していない。
