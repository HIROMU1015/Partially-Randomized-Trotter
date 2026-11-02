# Native DF gate model — 技術証跡

[主要資料・数式・SOURCE対応](../../../../docs/research/track_a_native_df_theoretical_gate_model_20261016.md)。

- [source_provenance_v1.json](source_provenance_v1.json): 固定起点043と別native参照87のGit blob/SHA-256。異なるSOURCEを同一版として扱わない。
- [synthetic_validation_v1.json](synthetic_validation_v1.json): 27 local artificial testsと新sourceのSHA。実分子データ・Qiskit・sampling・compileは使わない。
- [technical_audit_v1.json](technical_audit_v1.json): 作業範囲、既存source・dirty保全、残る制限。主担当の技術チェック。

一般identity phaseの期待費用はboundとして保持。実compiled費用やpopulation winnerは未認定。
新しい科学結果・実行認可ではない。公開後MANDATORY_STOP、next_stage_authorized=false。
