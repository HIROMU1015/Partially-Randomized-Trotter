# mainに収録した既存研究成果

2026-11-03 JST。枝ごとに保存されていた成果を、研究単位でmainに収録する。
以下は既存成果の配置と出典の案内であり、新しい科学的採否・保証・再実行の判断ではない。

## 第1回：Track B G6（2026-10-10）

| 入口 | main内の資料 |
|---|---|
| 結果・取得範囲・未解決点 | [G6報告](../tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md) |
| 保存結果・STOP・SOURCE binding | [原証拠ディレクトリ](../../artifacts/track_b_g6_return_generator_audit/2026-10-10/)と[evidence manifest](../../artifacts/track_b_g6_return_generator_audit/2026-10-10/evidence_manifest_v1.json) |
| 数学・実装・文献の元監査 | [独立数学監査](../tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)、[finite-bit/access監査](../tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)、[prior-art表](../tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md) |
| 再現用の原SOURCE | [形式prototype](../../src/trottertracks/algorithm_codesign/return_aggregation.py)、[元technical runner](../../scripts/tracks/algorithm_codesign/g6_return_generator_audit.py)、[元test SOURCE](../../tests/tracks/algorithm_codesign/test_g6_return_aggregation.py) |
| 入力レビュー | [G5レビュー原本](track_b_G5_research_direction_review_20261010.md)と[入力identity](../../artifacts/track_b_g6_return_generator_audit/2026-10-10/review_input_identity.json) |
| 当時の工程記録 | [2026-10-10の研究ノート](研究ノート/2026-10-10_track_b_g6_return_generator_audit.md) |

元の公開snapshotは `28cfabb1d47e0e1824bce1e154a9c289738fa2b9`。
上記22ファイルはそのsnapshotのGit blobと同じ内容・実行modeで収録している。
元の `evidence_manifest_v1.json`、SOURCE hash、status、marker、STOP、既存の数値や判定は変更していない。
`src/trottertracks` は研究別SOURCEの名前空間であり、共通 `trotterlib` APIへの組込みを意味しない。

原報告の技術状態は `G6_IDEAL_IDENTITIES_VERIFIED_NATIVE_AND_METHOD_VALUE_UNRESOLVED`、mandatory STOPである。
保存された16 testsの記録は当時のsource-bound検査であり、今回のmain収録で再実行したものではない。
今回の確認はファイル同一性・配置・リンク・manifest構造だけで、native実装・性能・独立新規性・外部科学再現を新たに認定していない。

元runnerは当時のSOURCE、runtime、保護対象、既存markerに結び付く。mainへコピーしたことは、その一回実行やSTOPの解除ではない。
再現を検討する場合は元snapshotと既存scopeを確認する。Git管理外のrawやserver内のcacheはmainへの収録対象に含まれていない。

## 以後の取り込み方

次は古いG10等について、当時の準備版と後続結果を対応させ、結果・文書・SOURCEを研究単位で収録する。
共通ファイルに複数版がある場合は、元SOURCE identityを保持して解決する。
枝の削除は、対象の成果がmain内で辿れ、既存保存tagの履歴保全と継続作業の確認を終えた後に扱う。
