# G6/G7 F1＋F2：事前精度契約と全処理bit費用の技術的閉鎖

2026-10-18 JST。基点 `8870ed37855ca8da97f8370e0a295d21cf5414f6`、branch `track-b-g6-f1-f2-precision-bit-cost-20261018`。W1の[F1/F2提案][W-PLAN]をユーザーの今回の指示に従って具体化する。新研究採択・新証明・実装・科学実行ではない。

**F1は、提供された固定H/K_rootが全到達parentで既存guardを通るための、列挙不要な十分条件まで仕様化した。F2は、明示した保守的な標準bitモデルで、setup・3種類のtrial経路・weight・memory・全Nの条件付き費用を具体化した。新しい非自明定理が必要となる箇所は特定されなかった。** 自動精度選択、identity入口、angle enclosure、証拠記録のSOURCE接続は未実装。物理providerのstrict error/native費用は別scopeで未認証。

F1の条件は**既存下界・既存丸め処理に対する受入不等式**であり、H/Kを返す新selector・普遍的な選択公式・adaptive refinementではない。入力へ値を代入していない。F2の上界はSOURCEの処理列を標準算術モデルへ置いた通常導出であり、Pythonの実CPU秒・tight complexity・他方式への優位を認定しない。

## 成果物

| ファイル | 内容 |
|---|---|
| **[preflight_precision_contract.md](preflight_precision_contract.md)** | F1：型/domain、全parent精度のP0〜P5、canonical差、identity、failure、confidence |
| **[full_bit_complexity_accounting.md](full_bit_complexity_accounting.md)** | F2：演算モデル、bit長、root/rounding/selector、setup/local/3経路、angle、memory、全N |
| [api_and_cost_closure_matrix.md](api_and_cost_closure_matrix.md) | 今回閉じた条件・未実装・条件付き費用・受入条件 |
| [implementation_acceptance_requirements.md](implementation_acceptance_requirements.md) | 将来の実装受入要件。今回はtest/codeなし |
| [remaining_gaps_and_decisions.md](remaining_gaps_and_decisions.md) | 完成・通常実装・追加解析・物理未認証・別scopeを分離 |
| [handoff.md](handoff.md) | ユーザー/GPTへの回答、P_U/W2への入力条件、次の承認とSTOP |
| [closure_index.json](closure_index.json) | 統合義務・status・未実装の機械索引 |
| [source_inventory.json](source_inventory.json) / [evidence_index.json](evidence_index.json) | 固定SOURCE/資料・読解範囲・bytes/SHA256・標準算術一次資料 |
| [verification.json](verification.json) / [publication_receipt.json](publication_receipt.json) | 文書/metadata管理検査、公開bytes照合・保全 |

## 対象・独立性・権限

有限正有理p、Σp=1、有理0<x≤1、有限奇数m、σ=±1、Hermitian involution Q_i、fresh independent uniform bits、exact controlled-Qの条件付きqueryモデル。対象はfinite `M=P_m(-iσxΣp_iQ_i)`、digital meanはM̃。x=0は別identity経路。分子geometry/basis/DF rank/L_Dは一般有理入力の本仕様には非該当。Taylor remainder・元Hamiltonian誤差・長時間PF/RPEは含めない。

W1全7資料、[MATH]/[BIT]/[G7C]/[U]と固定[K]/[G]/[CP]を再利用。Bの[固定接続監査][B-README]は参照だけとし、I01〜I18/V1〜V6、DF入力、ordinary request、backend/state/weighted accumulator、native費用を再監査していない。

Codex Aの `track-b-g6-pu-analytic-proof-20261018` のlive worktree・未commit内容・index・lock・process・出力にはアクセスしない。P_Uの証明/反証、normalizer比較、W2b/W2cは行わず、Aの結果を待たない。他branchのmerge/rebaseなし。変更は本ディレクトリのMarkdown/JSONだけ。

共通運用指示2026-10-17は前回完全読解済み原本とhash一致を確認し、§8.4を再確認。親PROJECT_MAP・研究概要も既読bytesと一致する。公開一次資料の限定調査は標準算術とPythonの数値型仕様に限り、新規先行法探索ではない。私的レビュー原本や文献PDF/HTMLの全文は公開しない。

科学SOURCE変更/import/実行、科学testの作成/実行、科学計算、保存数値再解析、sampling、回路build/compile/合成/simulation、新非自明証明、科学実行contract/authorization/grant/marker/claimは**0件**。D＋E、Parent、G6〜G10、旧STOP/CLOSED/HOLD、親dirty/untrackedと既存科学証拠を保全する。共有設定や他者jobの操作、人工的なCPU/RAM/worker制限は行わない。

**公開・保全確認後mandatory STOP。** F1/F2の技術仕様完成を、実装完了・実行成功・独立新規性・論文化GOへ格上げしない。

<!-- FIXED_REFERENCES -->

[B-README]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/README.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
[W-PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8870ed37855ca8da97f8370e0a295d21cf5414f6/docs/research/track_b_g6_precision_api_cost_w1_20261018/minimal_followup_plan.md
