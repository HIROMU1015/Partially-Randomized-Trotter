# G6/G7 W2a：一般次数のnormalizer比較問題

2026-10-18 JST。**今回の対象は理想係数normalizerと認証上界Uの定義・既存証拠・未解決命題だけ。** 新しい証明、反例、漸近展開、計算、sampler設計は行わない。

固定基点は `de709339e2539fa4d9806a84e738040d52c97be1`、専用branchは `track-b-g6-w2a-general-order-normalizer-20261018`。[正式なW2提案][T03]のW2aを具体化する。W1 `track-b-g6-precision-api-cost-w1-20261018` の未commit内容、index、lock、process、出力は参照・変更せず、公開されても自動merge/rebaseしない。H/K・API・全bit費用を再調査しない。

比較の対象は同じ有限 `P_m(-iσxΣp_iQ_i)`。有限L≥1、明示された正有理p、Σp=1、有理0<x≤1、奇数m≥5、σ=±1、同じHilbert空間上のHermitian involution Qを固定する。x=0はidentity別経路、m=3は既存結果の引用のみ。m=5とm≥7を分ける。formal reduced-word収集のnormalizerを扱い、全可能分解の最小値やchannel一致は主張しない。

| 成果物 | 内容 |
|---|---|
| [baseline_normalizer_definition_matrix.md](baseline_normalizer_definition_matrix.md) | 6方式のtarget・係数・event・normalizer・取得条件、P5とtailの境界 |
| **[general_order_comparison_question.md](general_order_comparison_question.md)** | W2a-1〜3と、次の証明・反証の提案を最大1命題へ限定 |
| [handoff.md](handoff.md) | 既知／未解決、正負の意味、W1との分担、担当・承認 |
| [evidence_index.json](evidence_index.json) / [source_inventory.json](source_inventory.json) | status・選択命題と、固定commit・blob・bytes/SHA256・実読範囲 |
| [verification.json](verification.json) / [publication_receipt.json](publication_receipt.json) | 文書管理・保全と公開照合。科学的certificateではない |

固定資料は指定済みcommitの18件のみ。[G6数学監査][M]、[finite-bit監査][F]、[G7契約][GC]、[G7 SOURCE][G]、[Uの既存導出][U]、[P5 SOURCE][P5]、同じ指定P5 commitにある[既存P5導出][PC]を使用する。原Kernel[K]、[全生成監査][A01]と[構成Packet][C01]は定義の所在確認に限定し、84セルの文献監査は繰り返さない。私的レビュー、保存数値、分子/native結果は公開しない。

共通運用指示2026-10-17は本会話で既読の同一bytesを確認し、§8.4を再確認した。18件の独立公開text/hash取得を並列化し、専用cacheを使用。科学SOURCE/testのimport・実行・変更、新しいN/moment/費用順位・break-even、論文・新規性認定は0件。W2b/W2cには判断への影響だけを引き継ぐ。

変更は今回の専用8ファイルだけ。既存SOURCE/results/contract/grant/ledger/marker/claim、D＋E、STOP/CLOSED/HOLDを保持する。保全検査では親tracked/直下untrackedのbytesと、W1を除く登録済み他worktreeのHEAD/index/lockを比較する。動いているW1の内部を監視した、あるいはW1自身の変更がないとは主張しない。

**commit・push、remote HEAD・公開bytes/SHA256・保全確認後mandatory STOP。** 選択命題の証明・反例探索・漸近解析、W2b/W2c、新実装・計算・科学レビューへ自動移行しない。

<!-- FIXED_REFERENCES -->

[A01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/README.md
[C01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/README.md
[F]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[PC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[T03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/de709339e2539fa4d9806a84e738040d52c97be1/docs/research/track_b_g6_theory_deepening_triage_20261018/minimal_next_work_proposals.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
