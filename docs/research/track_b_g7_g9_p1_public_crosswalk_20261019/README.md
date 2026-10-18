# G7/G9 P1先行照合：公開B3/C1準備と既存証拠の接続

2026-10-19 JST。技術状態：**`P1_PRELIMINARY_PUBLIC_PREPARATION_CROSSWALK_COMPLETE`**。

**公開済みB3/C1準備版との静的照合は完了した。将来の実行driverとの最終P1照合は未実施である。** 既存G9 v2のmatched-native比較はその固定scopeで引用可能。新C1への再利用はprovider・実装・価格粒度・policyのbindingを条件とする。新しいcompileが常に必要という結論ではない。

## 固定対象と今回の差分

- 直接基点：[G7/G9監査][AUD] `ff0bed24ba8fdb361a2890646ef84347ae37d457`。基本比較・既存順位・限界はこの原本を継承し、作り直していない。
- read-only照合対象：[公開B3/C1準備][PREP] `c8cbeb18f76aef2cb1329d8c810036372fa07cc0`。補助SOURCEは[BS][BS]、SHA256 `35a4a0d8b28fd1a50791cc2afd09ac86800a1f37b488dc4cee5906864ea7f0e1`。
- G7結果 `a689694080f4b7600fe67cf77d841d1cbbf04503`／科学SOURCE `ab2549f41b3546fb3940342a2162dd9ee93699c4`、G9 v2結果 `c95736fd2990f5ef6dd1cb5866421fd78bb28687`／科学SOURCE `0ef2b92738750a3c0d187743dd4b7c9b927db802`。必要なfield・関数だけ再参照した。
- 正式共通指示2026-10-17のローカル原本と[固定公開版][COMMON]が同一bytesであることを確認し、§8.4を再確認した。入口・研究概要の過去の全文読解は同一SHA256で再利用した。

対象はknown developmentの有限P5 first operator moment、p=(1/5,3/10,1/2)、x=5/7、m=5。G7は抽象Hermitian involution、G9は明示3-system-qubit synthetic/I1 context。分子、geometry、basis、DF rank、split、δ窓は非該当。exact exponential・QPE総費用の評価ではない。

今回確定した新しい対応は、(i) B3 Layer A/Bと強い対照の不足、(ii) C1の9-field primitive価格keyとG9 event価格の粒度差、(iii) strict-phase RZ保存recordの再利用条件、(iv) 将来driverで追加確認する差分である。

## 成果物

| 資料 | 内容 |
|---|---|
| [B3/C1–G9 identity対応](b3_c1_g9_identity_crosswalk.md) | 数学・law/policyを継承する範囲と、新しいlayer/provider/schema対応 |
| [native証拠再利用表](native_evidence_reuse_matrix.md) | 6区分、固定field、条件、compileが不要／必要となる境界 |
| [強い対照の古典費用不足](classical_strong_comparator_gaps.md) | P5Closed/CTSで既にあるSOURCEと、ないproduction・監査・保存接続 |
| [P1-final差分チェックリスト](p1_final_delta_checklist.md) | Codex A公開後に確認する事項のみ。将来作業の自動承認ではない |
| [Decision Handoff](handoff.md) | Q1〜Q5、追加B2・P2/P3・C2とユーザー判断 |
| [再利用分類metadata](reuse_inventory.json) | 判定対象、scope、固定根拠、保留条件 |
| [証拠索引](evidence_index.json) / [SOURCE索引](source_inventory.json) | commit/path/field/bytes/SHA256と読解範囲 |
| [管理検査](verification.json) / [公開照合](publication_receipt.json) | 静的・metadata・保全・公開同一性。数値再検証ではない |

公開準備の48人工テストPASSは**既存記録の引用**で、今回の再実行でも実G7 correctnessでもない。[BTEST][BTEST] 候補は`adopted=false`、`contract_grants_execution=false`で、実runnerはまだ提供しない。[BC][BC] `reject_real_execution`はmetadataを変更しても実実行を拒否する。[BS][BS]

## 実施境界・保全

独立branch `track-b-g7-g9-p1-public-crosswalk-20261019`、独立checkoutで新規監査文書・管理JSONだけを追加した。基点の全blob・共通索引は不変。他branchをmerge/rebaseせず、公開B3/C1の固定Git objectのみを取得した。Codex Aのlive worktree/index/lock/process/未commit成果は参照も変更もしていない。その後のdriverや契約が完成・採択済みとは扱わない。

**0件**：新科学計算、保存科学値の再解析・再ランキング、新B2縮約、科学SOURCEのimport/実行/変更、実装、sampling/profiling/全event再列挙、回路build/compile/synthesis、非自明証明、先行研究再調査、重要科学レビュー、provider/研究方針/科学条件の採択、authorization/grant/marker/claim発行・消費・再利用。

管理用処理はGit/text/AST/JSON構造・保存fieldの同一性・hash・リンク・公開read-backのみ。RZの数値誤差や価格を再算定していない。CPU32、available RAM約49 GiB、/tmp空き約375 GiBの観測時点で独立hash/read-backを効率化し、共有設定・cache・他processを変更していない。旧固定resource制約は維持した。

公開・保全確認後は**mandatory STOP**。将来のP1-final照合、P2/P3、B3/C1実行、新native取得には自動移行しない。

<!-- FIXED_REFERENCES -->

[AUD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ff0bed24ba8fdb361a2890646ef84347ae37d457/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/README.md
[BC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/execution_contract_candidate.json
[BS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py
[BTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/artificial_test_results.json
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md
[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/README.md
