# G6全生成手順と既知法の対応監査

2026-10-18 JST。ユーザー指定のread-only技術監査。**Green再帰、Euler pairing、ordinaryの非列挙proposal、importance補正には具体的な既知対応がある。一方、return収集後の係数からparent/child lawを作り、量子前zero、実dyadic weight、bias/range、Uを使う予算まで同じ情報条件で再現する全手順の先行対応は、指定資料では確定できない。完全同値・完全非同値・独立新規性のいずれも認定しない。**

これは「既知部品の結合で説明できない」という結論ではない。G6自身のSOURCEはそれらを接続している。未確立なのは、その接続までが既知文献に明示された手順の直接的な特殊化なのか、どこに構成的な貢献が残るかという比較である。部品の既知性も、文献未発見も、単独では結論にしない。

## 固定資料と独立性

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- Branch/worktree：`track-b-g6-generator-prior-art-alignment-20261018`の独立worktree。
- **新branchの基点**：`d4163acd7b577afaa7d6db8c29db64f76ec03ae7`。
- [構成論文Evidence Packet（C01–C09）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/README.md)：M1〜M12とD1〜D5を既存の基準として参照。作り直さない。
- [非Pauli応用Packet（N01–N06）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/README.md)：別branchの固定commitからread-only参照。mergeなし。
- G6 kernel K=`28cfabb1…`、数学監査MATH/finite-bit監査BIT=`bc649c59…`、G7 generator G/provider CP=`ab2549f4…`、後続Uの既存公開レビューU=`101bf342…`。**基点と資料取得元commitは別々に記録**する。
- 22件の固定GitHub資料、7件の一次本文の取得/hash/確認範囲は[source_inventory.json](source_inventory.json)、[literature_scope.json](literature_scope.json)。私的Claude/GPT原本は取得・転載しない。既存レビューの新しい推論を今回の証明としない。
- 共通運用指示は親checkoutの`research_project_instructions_unified_2026-10-17.md`を確認し適用。identityと作業許可の境界は[evidence_index.json](evidence_index.json)。

## 成果物

| ファイル | 役割 |
|---|---|
| **[generation_pipeline_alignment.md](generation_pipeline_alignment.md)** | 最重要。P1〜P12×7比較経路＝84対応セル。status、処理、条件、一次箇所、残る問題 |
| [known_method_composition_audit.md](known_method_composition_audit.md) | 強い「既知Green＋Euler＋rejection＋IS」反論の成立範囲と接続点 |
| [access_precision_complexity_comparison.md](access_precision_complexity_comparison.md) | 共通access、有限target、精度・予算・古典/量子費用 |
| [remaining_independent_claim_candidates.md](remaining_independent_claim_candidates.md) | 既存D1〜D5から残る条件付き主張候補と最小追加根拠の型 |
| [handoff.md](handoff.md) | 決定的な問い、追加調査と科学判断の境界、担当と承認 |
| [evidence_index.json](evidence_index.json) | 84セルの全比較項目、P/M/D対応、scope・権限・非実行記録 |
| [source_inventory.json](source_inventory.json) | 固定Git blob・公開bytes/SHA256、SOURCEの静的位置 |
| [literature_scope.json](literature_scope.json) | 固定版・実読箇所、既存Packetの本文hashとの照合 |
| [verification.json](verification.json) | 文書/JSON/リンク/hash/変更範囲/保全の技術検査 |
| [publication_receipt.json](publication_receipt.json) | 内容commitの公開照合。最終receipt commitの自己hash循環は避ける |
| [verify_metadata.py](verify_metadata.py) | stdlibのみの管理検査器。科学SOURCE/testは実行しない |

## 固定する対象とSTOP

正の明示有理p、Σp=1、0<x≤1、有限奇数m、σ=±1、Hermitian involution Qを仮定し、同じ`M=P_m(-iσxR)`を対象とする。有限bitでは一般にM̃であり、Mとの差をbiasとして記録する。exponential/channel/別cutoffへ無断で置換しない。

新科学計算、保存値再解析、科学SOURCEのimport/実行/変更、科学test、新しいsampler・同値性定理・非同値性定理、回路build/compile/合成、私的原本公開、論文執筆・投稿判断は行わない。普遍的量子総資源改善を構成論文の必要条件にも加えない。

変更は今回の専用文書・metadataだけ。既存2 Packet、D＋E、G6〜G10、DF/Gaussian、Parent allocation、他研究の契約・grant・ledger・marker・claim・STOPを保持する。公開・remote照合・保全確認後、**mandatory STOP**。科学レビュー・新実装・計算・他branch統合へ自動移行しない。
