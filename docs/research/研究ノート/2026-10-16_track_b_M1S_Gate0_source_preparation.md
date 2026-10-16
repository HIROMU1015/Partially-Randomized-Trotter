# 2026-10-16 Track B M1-S Gate 0 SOURCE準備

ユーザーの添付指示により、05513eeaを基点にSOURCE実装・静的検査・契約候補・公開のみを実施した。科学実行、実/模擬入力での数値correctness test、契約正式採択は明示的に対象外。

保存H4 1.00 Å/STO-3G/8 spin orbitals/DF rank12、保存順G3/6/9を技術実装対象とし、JWの符号付き非冗長Pauli lawとidentity→HD案を実装した。occupation basisの別経路再構成、state精度、W/v/rho、単位別許容値、分母・適用域label、保存監査をSOURCEに含めた。科学処理は未実行。

最終SOURCEは4bd39ca68426f754111e47750534a7a56627107b。初回27a97708も保持した。既定status/help/schema/static configは数値import/NPZ読込みなし。30件の非科学的チェックをPython3.10/3.11で確認した。数値的correctnessは未検証。科学的許容値17件・runtime/limits・実験契約採択・別実施承認は未決。

W/vのengineering誤差区間を厳密なexact-state保証としない。ρによる自動GO/STOP、有限step dressing、native価格/shot数、T1c/Q4/Q5の拡張は行わない。

[契約最終候補](../../tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_candidate_20261016.md)、[証拠索引](../../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/evidence_index.json)を追加。過去本文・旧STOP・SOURCE・レビュー・入力・結果・承認・markerを保持し、親worktreeを変更していない。

分類IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED、CANDIDATE / NOT ADOPTED / SCIENTIFIC EXECUTION NOT AUTHORIZED。新規科学計算/Pauli表/回路/compile/simulation/数値correctness test/新authority/markerは全て0。公開後mandatory STOP。次はユーザーの科学条件採択、続いてさらに別の明示科学実施承認。
