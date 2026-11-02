# G6/G7：理論・アルゴリズム深化の未解決課題選別

2026-10-18 JST。ユーザー承認の静的技術調査と次作業の提案。**優先候補は2件：既存の有限bit保証を実効的な全費用契約まで閉じる問題（C1）と、一般次数で追加return集約が保証・費用の判断を変える成立条件／限界（C2）。** 自動H/K選択器の実装だけ、上界の微小な改善だけを新しい研究成果とは数えない。

ユーザーは、直近GPTレビューで局所正値分解と有限精度保証の構成的価値が評価されたと明示している。その研究状態を前提にする。指定原本 `track_b_g6_constructive_contribution_scientific_review_20261018.md` はrepository・添付領域で未取得。本文の条件・推論・具体的な推奨順位は推測せず、私的原本も公開しない。本調査はそのレビューの代行・再開催ではない。独立新規性、普遍的資源改善、投稿GOは未確定のまま。

## 作業基点と固定資料

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 専用branch/worktree：`track-b-g6-theory-deepening-triage-20261018`。
- **基点commit**：`5f17a7548b7eef1976f45f5d09bdc979e2e0a81b`。[全生成手順監査][A01]を再利用し、84セルの先行研究調査はやり直さない。
- [構成論文PacketのM1〜M12][C01]は `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`。[非Pauli Packet][N01]は別branchの `128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6` からread-only参照。merge・改変なし。
- 原数学監査[MATH]、finite-bit監査[BIT]、G6 kernel[K]、G7 generator[G]／provider[CP]、G7契約[G7C]、既存公開U導出[U]を静的照合。P5の弱い対照を避けるため、既存[G9 P5 closed-form SOURCE][P5]のproduction部分だけ追加確認した。
- 固定GitHub資料27件のcommit・blob・bytes/SHA256・実読範囲は [source_inventory.json](source_inventory.json)。基点と各資料の取得元を区別する。共通運用指示2026-10-17と親checkoutの案内資料はread-only参照し、identityだけ記録する。

## 成果物

| ファイル | 内容 |
|---|---|
| [theory_gap_matrix.md](theory_gap_matrix.md) | T1〜T6：既解決事項、真の不足、追加成果との関係 |
| [proof_and_contract_risks.md](proof_and_contract_risks.md) | 量化・境界・precision・bit費用・providerの未閉鎖点と確度 |
| **[candidate_priority_decision.md](candidate_priority_decision.md)** | 中心成果物。8候補の比較、優先2件の10項目、負結果の価値と停止条件 |
| [minimal_next_work_proposals.md](minimal_next_work_proposals.md) | 優先候補の最小作業、入力・成果物・担当・承認・GO/STOP |
| [handoff.md](handoff.md) | ユーザー・GPT・Codexへの引継ぎと今回のSTOP |
| [evidence_index.json](evidence_index.json) | gap・risk・candidate・固定箇所・実施権限の機械索引 |
| [verification.json](verification.json) | JSON、リンク、固定bytes、変更範囲、保全の管理検査 |
| [publication_receipt.json](publication_receipt.json) | 内容commitの公開照合。最終receipt commitはpush後に独立照合 |
| [verify_metadata.py](verify_metadata.py) | stdlibのみの文書管理検査。候補の科学的検証コードではない |

## 固定する研究対象と非実施事項

有限L≥1、明示された正の有理p、Σp=1、有理0<x≤1、有限奇数m≥1、σ=±1、Hermitian involution Q。対象は同じ有限operator `M=P_m(-iσxΣp_iQ_i)`。x=0は別のidentity経路。finite-bit mean M̃、元のfinite M、物理Hamiltonian、exponential、channelは区別する。一般形式のため分子・geometry・basis・DF rank・L_Dは非該当で、新しい数値を引用しない。

今回の変更は専用文書・metadataだけ。新定理・下界・反例の導出、新sampler、selectorやangle計算の実装、科学SOURCEのimport/実行/変更、科学test、保存値再解析、回路・compile・合成、論文執筆、科学レビューは0件。研究方針を正式採択せず、contract/grantを発行しない。既存STOP/CLOSED/HOLD、SOURCE、結果、grant、ledger、marker、claimを保持する。

**公開・remote・保全照合後mandatory STOP。** 本文の提案は後続作業の開始許可ではない。

<!-- FIXED_REFERENCES -->

[A01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/README.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[C01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[N01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/README.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
