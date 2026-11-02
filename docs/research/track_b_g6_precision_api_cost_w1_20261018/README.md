# G6/G7 W1：有限bit保証・API充足条件・全処理費用の差分仕様監査

2026-10-18 JST。ユーザーが指定したC1/W1の静的技術監査。基点は `de709339e2539fa4d9806a84e738040d52c97be1`、branchは `track-b-g6-precision-api-cost-w1-20261018`。Codex Bの `4cf2a948ed348f842f3c338f6b46091124742430` は固定Git objectから参照し、mergeしていない。

**既存lemmaは、正の明示有理入力に対するsupport・有限bit精度の存在・補正mean・bias/range/moment・angle処理可能性までを与える。これを否定する新しい理論上の穴は今回特定されなかった。未完成なのは、全到達parentで既存検査を満たす事前精度指定の接続、identity/domain契約、angle実装、具体的な全処理bit費用の会計である。未実装を未証明と扱わない。**

全処理について具体的な数式上界や実秒の保証を完成したわけではない。既存の粗い多項式構成可能性と、凍結SOURCEの全処理上界は別である。追加の標準的複雑性解析は必要だが、非自明な新定理が不可欠であるという根拠は得ていない。したがって現時点でW1を独立の新理論研究として開始する必要はない。[残るgap](remaining_proof_vs_implementation_gaps.md)

## 対象と維持する条件

有限L≥1、明示された正の有理数p、Σp=1、有理0<x≤1、有限奇数m≥1、σ=±1、同じ空間上のHermitian involution Q_i。対象は `M=P_m(-iσxΣp_iQ_i)`。finite-bitのmeanは通常M̃であり、Mとは係数近似biasを介して比較する。x=0は別のidentity経路。fresh uniform independent bitsとexact controlled-Qの条件付きqueryモデルを保持する。一般形式なので分子・geometry・basis・DF rank・L_Dは非該当。

Codex Aの[W1直接仕様][A-WORK]を適用し、C2/W2の新しい比較命題・証明は扱わない。Bの[I01〜I18][B-MATRIX]と[V1〜V6][B-ACCEPT]は既存接続監査として利用し、DF入力、ordinary request、wrapper/backend/state/accumulator、native/PR費用を再監査しない。[Bの固定ハンドオフ][B-HANDOFF]を参照する。

## 成果物

| 資料 | 用途 |
|---|---|
| **[precision_api_obligation_map.md](precision_api_obligation_map.md)** | W1-1〜6の差分義務、出典・API行・条件・費用・分類・次作業 |
| [precision_and_failure_boundaries.md](precision_and_failure_boundaries.md) | support、root/丸め、3種類の失敗、事前一様条件、confidence |
| [classical_cost_obligation_map.md](classical_cost_obligation_map.md) | setup/online、bit、root、sort、selector、reject、angle、memory、全N |
| [remaining_proof_vs_implementation_gaps.md](remaining_proof_vs_implementation_gaps.md) | 既存保証・仕様・未実装・未解析・物理未認証の分離 |
| [minimal_followup_plan.md](minimal_followup_plan.md) | 最小の次作業、担当・完了条件・必要承認・STOP |
| [handoff.md](handoff.md) | ユーザー/GPTへの9項目の回答とW2への条件 |
| [obligation_index.json](obligation_index.json) | 狭い義務単位のstatusと未閉鎖理由 |
| [source_inventory.json](source_inventory.json) / [evidence_index.json](evidence_index.json) | 固定commit/path、実読範囲、blob・bytes/SHA256、対応 |
| [verification.json](verification.json) / [publication_receipt.json](publication_receipt.json) | 管理検査と公開照合。科学testではない |

## 読解・権限・保全の範囲

[数学監査][MATH]、[有限bit監査][BIT]、[Uの既存§7][U]、[M1〜M12][CLAIMS]、G6/G7 SOURCEを静的に照合した。Aの6資料は固定版を読解、Bの7資料は前回監査での読解を固定bytes照合して再利用した。Bの接続点を新しい監査結果として数えない。私的Claude/GPTレビュー、実入力、科学結果の数値は取得・転載・再解析していない。共通運用指示2026-10-17の原本は既読identityと一致し、§8.4を再確認した。詳細はinventory。

変更は本ディレクトリのMarkdown/JSONだけ。PROJECT_MAP、manifest、旧記録、SOURCE、test、raw、契約、grant、ledger、marker、claim、STOPは変更しない。親tracked状態・直下Markdown原本、以前の自分の公開Packet/B領域、既知の消費済みledgerのhashを保全確認する。Aのlive worktreeや他作業のindex/lock/processを調査・操作していないため、他者の全状態を監視したとは主張しない。

科学SOURCEの変更/import/実行、科学testの作成/実行、新科学計算、保存数値再解析、回路build/compile/合成、新証明、新実行契約/authorization/grant/marker/claimは**0件**。共有設定や他者のjobは操作せず、人工的なCPU/RAM/worker上限も新設しない。管理処理はGit、テキスト、AST、JSON、hash、公開bytes照合に限定。

**公開・保全確認後mandatory STOP。** 新しい研究採択、H/K selector・angle・sampler実装、科学レビュー、論文執筆、W2/C2へ自動移行しない。

<!-- FIXED_REFERENCES -->

[A-WORK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/de709339e2539fa4d9806a84e738040d52c97be1/docs/research/track_b_g6_theory_deepening_triage_20261018/minimal_next_work_proposals.md
[B-ACCEPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/verification_acceptance_plan.md
[B-HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/handoff.md
[B-MATRIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/interface_contract_matrix.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[CLAIMS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
